/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/14
 *  Description:
 */

#include <Utils/ComputeGT/ComputeStepwiseGT.hpp>
#include <Utils/ComputeGT/ComputeGT.hpp>

void COMPUTE_GT::computeStepwiseGT(const std::string& baseFile, const std::string& queryFile,
                                    const std::string& gtFile, size_t k, 
                                    const std::string& distFn, size_t batchSize, size_t initialCount) {
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
  loadBinAsFloat<float>(queryFile.c_str(), queryData, nqueries, dim, 0);

  size_t currentPoints = initialCount;
  size_t step = 0;

  std::ofstream writer(gtFile, std::ios::binary);
  if (!writer) {
    std::cerr << "Error opening file: " << gtFile << std::endl;
    delete[] baseData;
    delete[] queryData;
    return;
  }

  while (currentPoints < npoints) {
    size_t nextStepPoints = currentPoints + batchSize > npoints ? npoints : currentPoints + batchSize;
    size_t insertCount = nextStepPoints - currentPoints;
    currentPoints = nextStepPoints;
    step++;

    float* gtVectors = new float[nqueries * dim];
    size_t* closestPoints = new size_t[nqueries * k];
    float* distClosestPoints = new float[nqueries * k];

    exactKnn(dim, k, closestPoints, distClosestPoints, currentPoints, baseData, nqueries, queryData, metric);

    for (size_t i = 0; i < nqueries; i++) {
      size_t gtIdx = closestPoints[i * k];
      std::memcpy(gtVectors + i * dim, baseData + gtIdx * dim, dim * sizeof(float));
    }

    uint64_t step64 = static_cast<uint64_t>(step);
    writer.write(reinterpret_cast<const char*>(&step64), sizeof(step64));
    writer.write(reinterpret_cast<const char*>(&insertCount), sizeof(insertCount));
    writer.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    for (size_t i = 0; i < nqueries; i++) 
      writer.write(reinterpret_cast<const char*>(&i), sizeof(i));  
    writer.write(reinterpret_cast<const char*>(gtVectors), nqueries * dim * sizeof(float));

    delete[] gtVectors;
    delete[] closestPoints;
    delete[] distClosestPoints;

    std::cout << "Step " << step << " completed. Inserted " << insertCount 
              << " vectors. Total: " << currentPoints << std::endl;
  }

  writer.close();
  delete[] baseData;
  delete[] queryData;
}
