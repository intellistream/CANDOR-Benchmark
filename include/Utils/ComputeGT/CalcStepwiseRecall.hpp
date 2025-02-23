/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/14
 *  Description:
 */

#ifndef UTILS_COMPUTE_GT_CALC_STEPWISE_RECALL_HPP
#define UTILS_COMPUTE_GT_CALC_STEPWISE_RECALL_HPP

#include <string>
#include <map>
#include <vector>
#include <cstdint>

namespace COMPUTE_GT {

std::map<uint64_t, std::vector<std::pair<size_t, std::vector<std::vector<float>>>>> 
  readStepwiseHDF5(const std::string& filename, const std::string& groupName);

double calcRecallWithQueryVec(const std::vector<std::vector<float>>& queryVectors,
                                const std::vector<std::vector<float>>& annsResult,
                                const std::vector<std::vector<float>>& gtVectors,
                                float threshold = 1e-6);

std::vector<std::pair<size_t, double>> calcStepwiseRecall(const std::string& annsFile, 
                                                            const std::string& gtFile);

float computeL2Distance(const std::vector<float>& v1, const std::vector<float>& v2);

} 

#endif // UTILS_COMPUTE_GT_CALC_STEPWISE_RECALL_HPP
