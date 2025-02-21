#include <Utils/ComputeGT/CalcStepwiseRecall.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>

bool COMPUTE_GT::readStepwiseFile(const std::string& filename, uint64_t& npts, uint64_t& ndims, 
                                    std::vector<size_t>* indices, std::vector<std::vector<float>>& data, 
                                    bool readIndices) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return false;
  }

  if (!file.read(reinterpret_cast<char*>(&npts), sizeof(uint64_t))) return false;
  if (!file.read(reinterpret_cast<char*>(&ndims), sizeof(uint64_t))) return false;

  if (readIndices && indices) {
    indices->resize(npts);
    for (size_t i = 0; i < npts; i++) {
      if (!file.read(reinterpret_cast<char*>(&(*indices)[i]), sizeof(size_t))) return false;
    }
  }

  data.resize(npts, std::vector<float>(ndims));
  for (size_t i = 0; i < npts; i++) {
    if (!file.read(reinterpret_cast<char*>(data[i].data()), ndims * sizeof(float))) return false;
  }

  return true;
}

double COMPUTE_GT::calcRecallWithQueryVec(const std::vector<std::vector<float>>& queryVectors,
                                            const std::vector<std::vector<float>>& annsResult,
                                            const std::vector<std::vector<float>>& gtVectors) {
  if (queryVectors.empty() || annsResult.empty() || gtVectors.empty()) {
    std::cerr << "Error: Input vectors cannot be empty." << std::endl;
    return 0.0;
  }

  size_t nqueries = queryVectors.size();
  size_t ndims = annsResult[0].size();
  size_t correct_count = 0;

  std::map<std::vector<float>, size_t> gtIndexMap;
  for (size_t i = 0; i < gtVectors.size(); ++i) {
    gtIndexMap[gtVectors[i]] = i;
  }

  for (size_t i = 0; i < nqueries; i++) {
    auto it = gtIndexMap.find(queryVectors[i]);
    if (it == gtIndexMap.end()) {
      std::cerr << "Warning: Query vector not found in GT file." << std::endl;
      continue;
    }
    size_t gtIndex = it->second;

    std::set<int> gtSet;
    for (const auto& val : gtVectors[gtIndex]) {
      gtSet.insert(static_cast<int>(val));
    }

    for (float val : annsResult[i]) {
      int predictedId = static_cast<int>(val);
      if (gtSet.find(predictedId) != gtSet.end()) {
        correct_count++;
      }
    }
  }

  return static_cast<double>(correct_count) / (nqueries * ndims);
}

void COMPUTE_GT::calcStepwiseRecall(const std::string& annsFile, 
                                      const std::string& gtFile, 
                                      const std::string& outFile) {
  uint64_t npts, ndims;
  std::vector<std::vector<float>> gtVectors;
  if (!readStepwiseFile(gtFile, npts, ndims, nullptr, gtVectors, false)) {
    std::cerr << "Failed to read ground truth file." << std::endl;
    return;
  }

  uint64_t step, batchNpts, batchNdims;
  std::vector<size_t> queryIndices;
  std::vector<std::vector<float>> annsResult;

  while (readStepwiseFile(annsFile, batchNpts, batchNdims, &queryIndices, annsResult, true)) {
    double recall = calcRecallWithQueryVec(gtVectors, annsResult, gtVectors);
    std::ofstream file(outFile, std::ios::app);
    if (!file) {
      std::cerr << "Failed to open output file." << std::endl;
      return;
    }
    file << "Step " << step << ": Recall = " << recall << std::endl;
    std::cout << "Step " << step << ": Recall = " << recall << std::endl;
    file.close();
  }

  std::cout << "Stepwise recall written to " << outFile << std::endl;
}
