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

  hsize_t numObjects;
  H5Gget_num_objs(groupId, &numObjects);
  for (hsize_t i = 0; i < numObjects; i++) {
    char batchName[32];
    H5Gget_objname_by_idx(groupId, i, batchName, sizeof(batchName));
    std::string batchGroupName = std::string(batchName);
    hid_t batchGroupId = H5Gopen2(groupId, batchGroupName.c_str(), H5P_DEFAULT);
    if (batchGroupId < 0) {
      std::cerr << "Error opening batch group: " << batchGroupName << std::endl;
      continue;
    }

    uint64_t step = std::stoul(batchGroupName.substr(6)); 

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

      std::cout << "queryGroupName " << queryGroupName << std::endl;
      size_t queryIdx = std::stoul(queryGroupName.substr(6)); 

      std::vector<std::vector<float>> tensors;
      for (size_t j = 0; ; j++) {
        std::string dsName = "tensor_" + std::to_string(j);
        hid_t datasetId = H5Dopen2(queryGroupId, dsName.c_str(), H5P_DEFAULT);
        if (datasetId < 0) break;
        hid_t dataspaceId = H5Dget_space(datasetId);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspaceId, dims, NULL);
        std::vector<float> vec(dims[0]);
        H5Dread(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, vec.data());
        tensors.push_back(std::move(vec));
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

double
COMPUTE_GT::calcRecallWithQueryVec(const std::vector<std::vector<float>> &queryVectors,
                                    const std::vector<std::vector<float>> &annsResult,
                                    const std::vector<std::vector<float>> &gtVectors,
                                    float threshold) {
  if (queryVectors.empty() || annsResult.empty() || gtVectors.empty()) {
    std::cerr << "Error: Input vectors cannot be empty." << std::endl;
    return 0.0;
  }

  size_t correct_count = 0;
  size_t total_count = annsResult.size();

  for (size_t i = 0; i < queryVectors.size(); i++) {
    bool queryFound = false;
    size_t gtIndex = 0;
    for (size_t j = 0; j < gtVectors.size(); j++) {
      if (computeL2Distance(queryVectors[i], gtVectors[j]) < threshold) {
        queryFound = true;
        gtIndex = j;
        break;
      }
    }
    if (!queryFound) {
      std::cerr << "Warning: Query vector not found in GT." << std::endl;
      continue;
    }

    for (const auto &annVec : annsResult) {
      if (computeL2Distance(annVec, gtVectors[gtIndex]) < threshold) {
        correct_count++;
      }
    }
  }
  return static_cast<double>(correct_count) / total_count;
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
            double recall = calcRecallWithQueryVec(gtVectors, annVectors, gtVectors);
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
  return stepwiseRecall;
}
