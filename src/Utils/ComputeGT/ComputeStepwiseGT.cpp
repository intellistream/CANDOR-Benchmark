/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/14
 *  Description:
 */

#include <hdf5.h>

#include <Utils/ComputeGT/ComputeGT.hpp>
#include <Utils/ComputeGT/ComputeStepwiseGT.hpp>

void COMPUTE_GT::computeStepwiseGT(const std::string& baseFile, const std::string& queryFile,
                                    const std::string& gtFile, size_t k, 
                                    const std::string& distFn, size_t batchSize, 
                                    size_t initialCount, size_t qSize) {
  COMPUTE_GT::Metric metric;
  if (distFn == "l2") {
    metric = COMPUTE_GT::Metric::L2;
  } else if (distFn == "mips") {
    metric = COMPUTE_GT::Metric::INNER_PRODUCT;
  } else if (distFn == "cosine") {
    metric = COMPUTE_GT::Metric::COSINE;
  } else {
    std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
    return;
  }

  float* baseData = nullptr;
  size_t npoints, dim;
  loadBinAsFloat<float>(baseFile.c_str(), baseData, npoints, dim, 0);

  if (initialCount > npoints) {
    std::cerr << "Initial count exceeds total number of points." << std::endl;
    delete[] baseData;
    return;
  }

  float* queryData = nullptr;
  size_t nqueries;
  loadBinAsFloat<float>(queryFile.c_str(), queryData, nqueries, dim, 0, qSize);

  size_t* closestPoints = new size_t[nqueries * k];
  float* distClosestPoints = new float[nqueries * k];

  size_t currentPoints = initialCount;
  size_t step = 0;

  hid_t fileId = H5Fcreate(gtFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (fileId < 0) {
    throw std::runtime_error("Failed to create HDF5 file: " + gtFile);
  }

  hid_t rootGroupId = H5Gcreate2(fileId, "/groundtruth", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (rootGroupId < 0) {
    H5Fclose(fileId);
    throw std::runtime_error("Failed to create HDF5 group: /groundtruth");
  }

  hid_t attrDatatype = H5Tcopy(H5T_NATIVE_HSIZE);
  hid_t attrDataspace = H5Screate(H5S_SCALAR);
  hid_t attrId = H5Acreate2(rootGroupId, "k", attrDatatype, attrDataspace, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attrId, H5T_NATIVE_HSIZE, &k);
  H5Aclose(attrId);
  H5Sclose(attrDataspace);
  H5Tclose(attrDatatype);

  while (currentPoints < npoints) {
    size_t nextStepPoints = currentPoints + batchSize > npoints ? npoints : currentPoints + batchSize;
    size_t insertCount = nextStepPoints - currentPoints;
    currentPoints = nextStepPoints;

    std::string batchGroupName = "batch_" + std::to_string(currentPoints);
    hid_t batchGroupId = H5Gcreate2(rootGroupId, batchGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (batchGroupId < 0) {
      H5Gclose(rootGroupId);
      H5Fclose(fileId);
      throw std::runtime_error("Failed to create batch group: " + batchGroupName);
    }

    exactKnn(dim, k, closestPoints, distClosestPoints, currentPoints, baseData, nqueries, queryData, metric);

    for (size_t i = 0; i < nqueries; i++) {
      std::string queryIdxName = "query_" + std::to_string(i);
      hid_t queryGroupId = H5Gcreate2(batchGroupId, queryIdxName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (queryGroupId < 0) {
        H5Gclose(batchGroupId);
        H5Gclose(rootGroupId);
        H5Fclose(fileId);
        throw std::runtime_error("Failed to create query group: " + queryIdxName);
      }

      hsize_t dims[1] = {dim};
      for (size_t j = 0; j < k; j++) {
        size_t gtIdx = closestPoints[i * k + j];
        std::string datasetName = "tensor_" + std::to_string(j);
        hid_t dataspaceId = H5Screate_simple(1, dims, nullptr);
        hid_t datasetId = H5Dcreate2(queryGroupId, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspaceId, 
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, baseData + gtIdx * dim);
        if (status < 0) throw std::runtime_error("Failed to write dataset " + datasetName);

        H5Dclose(datasetId);
        H5Sclose(dataspaceId);
      }
      H5Gclose(queryGroupId);
    }
    H5Gclose(batchGroupId);
    std::cout << "Batch " << currentPoints << " completed. Inserted " << insertCount << " vectors." << std::endl;
  }
  H5Gclose(rootGroupId);
  H5Fclose(fileId);

  delete[] baseData;
  delete[] queryData;
  delete[] closestPoints;
  delete[] distClosestPoints;
}