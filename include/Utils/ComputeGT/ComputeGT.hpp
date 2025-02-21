/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/14
 *  Description:
 */

#ifndef UTILS_COMPUTE_GT_COMPUTE_GT_HPP
#define UTILS_COMPUTE_GT_COMPUTE_GT_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstring>
#include <omp.h>

#define ALIGNMENT 512
#define PARTSIZE 10000000

namespace COMPUTE_GT {

enum Metric {
  L2 = 0,
  INNER_PRODUCT = 1,
  COSINE = 2,
  FAST_L2 = 3
};

void computeL2sq(float* pointsL2sq, const float* matrix, const int64_t numPoints, const uint64_t dim);

void distSqToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                      const float* pointsL2sq, size_t nqueries, const float* queries,
                      const float* queriesL2sq, float* onesVec = nullptr);

void innerProdToPoints(const size_t dim, float* distMatrix, size_t npoints, const float* points,
                          size_t nqueries, const float* queries, float* onesVec = nullptr);

void exactKnn(const size_t dim, const size_t k, size_t* closestPoints, float* distClosestPoints,
                size_t npoints, float* pointsIn, size_t nqueries, float* queriesIn, Metric metric);

void computeVecGT(const std::string& baseFile, const std::string& queryFile,
                    const std::string& gtFile, size_t k, const std::string& distFn);

template <class T> T *aligned_malloc(const size_t n, const size_t alignment) {
  return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
}

template <typename T>
inline void loadBinAsFloat(const char *filename, float *&data, size_t &npts, size_t &ndims, int partNum) {
  std::ifstream reader;
  reader.exceptions(std::ios::failbit | std::ios::badbit);
  reader.open(filename, std::ios::binary);
  std::cout << "Reading bin file " << filename << " ...\n";
  int nptsI32, ndimsI32;
  reader.read((char *)&nptsI32, sizeof(int));
  reader.read((char *)&ndimsI32, sizeof(int));
  uint64_t startId = partNum * PARTSIZE;
  uint64_t endId = (std::min)(startId + PARTSIZE, (uint64_t)nptsI32);
  npts = endId - startId;
  ndims = (uint64_t)ndimsI32;
  std::cout << "#pts in part = " << npts << ", #dims = " << ndims << ", size = " << npts * ndims * sizeof(T) << "B"
            << std::endl;

  reader.seekg(startId * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
  T *dataT = new T[npts * ndims];
  reader.read((char *)dataT, sizeof(T) * npts * ndims);
  std::cout << "Finished reading part of the bin file." << std::endl;
  reader.close();
  data = aligned_malloc<float>(npts * ndims, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
  for (int64_t i = 0; i < (int64_t)npts; i++) {
    for (int64_t j = 0; j < (int64_t)ndims; j++) {
      float curValFloat = (float)dataT[i * ndims + j];
      std::memcpy((char *)(data + i * ndims + j), (char *)&curValFloat, sizeof(float));
    }
  }
  delete[] dataT;
  std::cout << "Finished converting part data to float." << std::endl;
}

}

#endif // UTILS_COMPUTE_GT_COMPUTE_GT_HPP
