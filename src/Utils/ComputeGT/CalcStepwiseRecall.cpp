#include <Utils/ComputeGT/CalcStepwiseRecall.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <hdf5.h>
#include <iostream>

std::map<uint64_t, std::vector<std::pair<size_t, std::vector<std::vector<float>>>>> 
    COMPUTE_GT::readStepwiseHDF5(const std::string& filename, const std::string& groupName) {
  std::map<uint64_t, std::vector<std::pair<size_t, std::vector<std::vector<float>>>>> result;

  hid_t fileId = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (fileId < 0) {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return result;
  }

  hid_t groupId = H5Gopen2(fileId, groupName.c_str(), H5P_DEFAULT);
  if (groupId < 0) {
    std::cerr << "Error opening group: " << groupName << std::endl;
    H5Fclose(fileId);
    return result;
  }

  size_t k;
  hid_t attrId = H5Aopen(groupId, "k", H5P_DEFAULT);
  if (attrId < 0) {
    std::cerr << "Error opening attribute 'k' in group: " << groupName << std::endl;
    H5Gclose(groupId);
    H5Fclose(fileId);
    return result;
  }
  herr_t status = H5Aread(attrId, H5T_NATIVE_HSIZE, &k);
  if (status < 0) {
    std::cerr << "Error reading attribute 'k' in group: " << groupName << std::endl;
    H5Aclose(attrId);
    H5Gclose(groupId);
    H5Fclose(fileId);
    return result;
  }
  H5Aclose(attrId);

  hsize_t numObjects;
  H5Gget_num_objs(groupId, &numObjects);
  for (hsize_t i = 0; i < numObjects; i++) {
    char batchName[32];
    H5Gget_objname_by_idx(groupId, i, batchName, sizeof(batchName));
    std::string batchGroupName = std::string(batchName);

  if (batchGroupName.substr(0, 6) != "batch_" || 
      !std::all_of(batchGroupName.begin() + 6, batchGroupName.end(), ::isdigit)) {
    std::cerr << "Invalid batch group name: " << batchGroupName << std::endl;
    continue;
  }
  uint64_t batchNum = std::stoul(batchGroupName.substr(6));

    hid_t batchGroupId = H5Gopen2(groupId, batchGroupName.c_str(), H5P_DEFAULT);
    if (batchGroupId < 0) {
      std::cerr << "Error opening batch group: " << batchGroupName << std::endl;
      continue;
    }

    uint64_t step = batchNum; 

    hsize_t numQueries;
    H5Gget_num_objs(batchGroupId, &numQueries);
    for (hsize_t q = 0; q < numQueries; q++) {
      char queryName[32];
      H5Gget_objname_by_idx(batchGroupId, q, queryName, sizeof(queryName));
      std::string queryGroupName = std::string(queryName);
      hid_t queryGroupId = H5Gopen2(batchGroupId, queryGroupName.c_str(), H5P_DEFAULT);
      if (queryGroupId < 0) {
        std::cerr << "Error opening query group: " << queryGroupName << std::endl;
        continue;
      }

      if (queryGroupName.substr(0, 6) != "query_" || 
          !std::all_of(queryGroupName.begin() + 6, queryGroupName.end(), ::isdigit)) {
        std::cerr << "Invalid query group name: " << queryGroupName << std::endl;
        H5Gclose(queryGroupId);
        continue;
      }
      size_t queryIdx = std::stoul(queryGroupName.substr(6));

      hsize_t numTensors;
      H5Gget_num_objs(queryGroupId, &numTensors);
      if (numTensors != k) {
        std::string fullPath = "/" + groupName + "/" + batchGroupName + "/" + queryGroupName;
        H5Gclose(queryGroupId);
        H5Gclose(batchGroupId);
        H5Gclose(groupId);
        H5Fclose(fileId);
        throw std::runtime_error("Error: query " + fullPath + " included " + std::to_string(numTensors) + 
                                " tensor, expected " + std::to_string(k));
      }

      std::vector<std::vector<float>> tensors;
      for (hsize_t t = 0; t < numTensors; t++) {
        char tensorName[32];
        H5Gget_objname_by_idx(queryGroupId, t, tensorName, sizeof(tensorName));
        std::string dsName = std::string(tensorName);
        hid_t datasetId = H5Dopen2(queryGroupId, dsName.c_str(), H5P_DEFAULT);
        if (datasetId < 0) {
          std::cerr << "Error opening dataset: " << dsName << std::endl;
          continue;
        }
        hid_t dataspaceId = H5Dget_space(datasetId);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspaceId, dims, NULL);

        if (dims[0] == 0) {
          std::string fullPath = "/" + groupName + "/" + batchGroupName + "/" + queryGroupName + "/" + dsName;
          H5Sclose(dataspaceId);
          H5Dclose(datasetId);
          H5Gclose(queryGroupId);
          H5Gclose(batchGroupId);
          H5Gclose(groupId);
          H5Fclose(fileId);
          throw std::runtime_error("Error: Empty dataset found at: " + fullPath);
        }
        
        std::vector<float> vec(dims[0]);
        herr_t status = H5Dread(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
        if (status < 0) {
          std::cerr << "Error reading dataset: " << dsName << std::endl;
        } else {
          tensors.push_back(std::move(vec));
        }
        H5Sclose(dataspaceId);
        H5Dclose(datasetId);
      }
      result[step].emplace_back(queryIdx, std::move(tensors));
      H5Gclose(queryGroupId);
    }
    H5Gclose(batchGroupId);
  }
  H5Gclose(groupId);
  H5Fclose(fileId);
  return result;
}

float COMPUTE_GT::computeL2Distance(const std::vector<float> &v1,
                                      const std::vector<float> &v2) {
  if (v1.size() != v2.size())
    return std::numeric_limits<float>::max();
  float sum = 0.0;
  for (size_t i = 0; i < v1.size(); i++) {
    float diff = v1[i] - v2[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

double COMPUTE_GT::calcRecall(const std::vector<std::vector<float>>& annVectors,
                              const std::vector<std::vector<float>>& gtVectors,
                              float threshold) {
  if (annVectors.empty() || gtVectors.empty()) {
    std::cerr << "Error: Input vectors cannot be empty." << std::endl;
    return 0.0;
  }

  size_t correctCount = 0;
  for (const auto& annVec : annVectors) {
    for (const auto& gtVec : gtVectors) {
      if (computeL2Distance(annVec, gtVec) < threshold) {
        correctCount++;
        break;  
      }
    }
  }
  return static_cast<double>(correctCount) / gtVectors.size();
}

std::vector<std::pair<size_t, double>>
COMPUTE_GT::calcStepwiseRecall(const std::string &annsFile, const std::string &gtFile) {
  auto gtData = readStepwiseHDF5(gtFile, "/groundtruth");
  auto annData = readStepwiseHDF5(annsFile, "/search_results");

  std::vector<std::pair<size_t, double>> stepwiseRecall;

  for (const auto &[step, annQueries] : annData) {
    double totalRecall = 0.0;
    size_t queryCount = 0;

    for (const auto &[annQueryIdx, annVectors] : annQueries) {
      auto gtIt = gtData.find(step);
      if (gtIt != gtData.end()) {
        for (const auto &[gtQueryIdx, gtVectors] : gtIt->second) {
          if (annQueryIdx == gtQueryIdx) {
            double recall = calcRecall(annVectors, gtVectors, 1e-6);
            totalRecall += recall;
            queryCount++;
            break;
          }
        }
      }
    }

    if (queryCount > 0) {
      double avgRecall = totalRecall / queryCount;
      stepwiseRecall.emplace_back(static_cast<size_t>(step), avgRecall);
    }
  }

  std::cout << "RECALL SIZE " << stepwiseRecall.size() << std::endl;
  return stepwiseRecall;
}
