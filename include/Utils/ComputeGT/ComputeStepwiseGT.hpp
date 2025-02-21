/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/14
 *  Description:
 */

#ifndef UTILS_COMPUTE_GT_COMPUTE_STEPWISE_GT_HPP
#define UTILS_COMPUTE_GT_COMPUTE_STEPWISE_GT_HPP

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

void computeStepwiseGT(const std::string& baseFile, const std::string& queryFile,
                          const std::string& gtFile, size_t k, const std::string& distFn,
                          size_t batchSize, size_t initSize);
}

#endif // UTILS_COMPUTE_GT_COMPUTE_STEPWISE_GT_HPP
