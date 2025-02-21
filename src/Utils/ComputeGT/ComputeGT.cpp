/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/21
 *  Description:
 */

#include <Utils/ComputeGT/ComputeGT.hpp>

#include <iostream>
#include <string>
#include <mkl.h>   

void COMPUTE_GT::computeL2sq(float* pointsL2sq, const float* matrix, const int64_t numPoints, const uint64_t dim) {
#pragma omp parallel for schedule(static, 65536)
  for (int64_t i = 0; i < numPoints; ++i) {
    pointsL2sq[i] = cblas_sdot((int64_t)dim, matrix + i * dim, 1, matrix + i * dim, 1);
  }
}

void COMPUTE_GT::distSqToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                                  const float* pointsL2sq, size_t nqueries, const float* queries,
                                  const float* queriesL2sq, float* onesVec) {
  bool onesVecAlloc = false;
  if (onesVec == nullptr) {
    onesVec = new float[std::max(npoints, nqueries)]();
    std::fill_n(onesVec, std::max(npoints, nqueries), 1.0f);
    onesVecAlloc = true;
  }
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float)-2.0, 
                points, dim, queries, dim, (float)0.0, distMatrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, 
                pointsL2sq, npoints, onesVec, nqueries, (float)1.0, distMatrix, npoints);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float)1.0, 
                onesVec, npoints, queriesL2sq, nqueries, (float)1.0, distMatrix, npoints);

  if (onesVecAlloc)
      delete[] onesVec;
}

void COMPUTE_GT::innerProdToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                                    size_t nqueries, const float* queries, float* onesVec) {
  cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, -1.0f,
              points, dim, queries, dim, 0.0f, distMatrix, npoints);
}

void COMPUTE_GT::exactKnn(const size_t dim, const size_t k, size_t* closestPoints, float* distClosestPoints,
                            size_t npoints, float* pointsIn, size_t nqueries, 
                            float* queriesIn, COMPUTE_GT::Metric metric) {
  float* pointsL2sq = new float[npoints];
  float* queriesL2sq = new float[nqueries];
  computeL2sq(pointsL2sq, pointsIn, npoints, dim);
  computeL2sq(queriesL2sq, queriesIn, nqueries, dim);

  float* distMatrix = new float[npoints * nqueries];

  if (metric == COMPUTE_GT::Metric::L2) {
    distSqToPoints(dim, distMatrix, npoints, pointsIn, pointsL2sq, nqueries, queriesIn, queriesL2sq);
  } else {
    innerProdToPoints(dim, distMatrix, npoints, pointsIn, nqueries, queriesIn);
  }

#pragma omp parallel for schedule(dynamic, 16)
  for (size_t q = 0; q < nqueries; ++q) {
    std::priority_queue<std::pair<float, size_t>> pq;
    for (size_t p = 0; p < npoints; ++p) {
      float dist = distMatrix[p + q * npoints];
      if (pq.size() < k) {
        pq.emplace(dist, p);
      } else if (dist < pq.top().first) {
        pq.pop();
        pq.emplace(dist, p);
      }
    }
    for (int i = k - 1; i >= 0; --i) {
      closestPoints[q * k + i] = pq.top().second;
      distClosestPoints[q * k + i] = pq.top().first;
      pq.pop();
    }
  }

  delete[] distMatrix;
  delete[] pointsL2sq;
  delete[] queriesL2sq;
}

void COMPUTE_GT::computeVecGT(const std::string& baseFile, const std::string& queryFile,
                                const std::string& gtFile, size_t k, const std::string& distFn) {
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

  float* queryData = nullptr;
  size_t nqueries;
  loadBinAsFloat<float>(queryFile.c_str(), queryData, nqueries, dim, 0);

  std::ofstream writer(gtFile, std::ios::binary);
  if (!writer) {
    std::cerr << "Error opening file: " << gtFile << std::endl;
    delete[] baseData;
    delete[] queryData;
    return;
  }

  float* gtVectors = new float[nqueries * k * dim];
  size_t* closestPoints = new size_t[nqueries * k];
  float* distClosestPoints = new float[nqueries * k];

  exactKnn(dim, k, closestPoints, distClosestPoints, npoints, baseData, nqueries, queryData, metric);

  for (size_t i = 0; i < nqueries; i++) {
    for (size_t j = 0; j < k; j++) {
      size_t gtIdx = closestPoints[i * k + j];
      std::memcpy(gtVectors + (i * k + j) * dim, baseData + gtIdx * dim, dim * sizeof(float));
    }
  }

  uint32_t nqueriesI32 = static_cast<uint32_t>(nqueries);
  uint32_t kI32 = static_cast<uint32_t>(k);
  uint32_t dimI32 = static_cast<uint32_t>(dim);

  writer.write(reinterpret_cast<char*>(&nqueriesI32), sizeof(uint32_t));
  writer.write(reinterpret_cast<char*>(&kI32), sizeof(uint32_t));
  writer.write(reinterpret_cast<char*>(&dimI32), sizeof(uint32_t));

  writer.write(reinterpret_cast<char*>(gtVectors), nqueries * k * dim * sizeof(float));

  writer.write(reinterpret_cast<char*>(distClosestPoints), nqueries * k * sizeof(float));

  writer.close();
  delete[] gtVectors;
  delete[] closestPoints;
  delete[] distClosestPoints;
  delete[] baseData;
  delete[] queryData;

  std::cout << "Compute GT done on file: " <<  gtFile << std::endl;
}
