/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/11
 *  Description:
 */

#include <CANDY/ConcurrentIndex.h>
#include <Utils/UtilityFunctions.h>
#include <Utils/ThreadPool.hpp>
#include <time.h>
#include <chrono>
#include <assert.h>

bool CANDY::ConcurrentIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string concurrentlAlgoTag = cfg->tryString("concurrentAlgoTag", "flat", true);
  IndexTable it;
  myIndexAlgo = it.getIndex(concurrentlAlgoTag);
  if (myIndexAlgo == nullptr) {
    throw std::runtime_error("Index algorithm not found: " + concurrentlAlgoTag);
    return false;
  }

  vecDim = cfg->tryI64("vecDim", 128, true);;
  writeRatio = cfg->tryDouble("ccWriteRatio", 0.5, true);
  batchSize = cfg->tryI64("ccBatchSize", 100, true);
  numThreads = cfg->tryI64("ccNumThreads", 1, true);
  randomMode = cfg->tryI64("ccRandomMode", 1, false);

  myIndexAlgo->setConfig(cfg);
  return true;
}

bool CANDY::ConcurrentIndex::loadInitialTensor(torch::Tensor &t) {
  auto ru = myIndexAlgo->loadInitialTensor(t);
  initSize = t.size(0);
  return ru;
}

bool CANDY::ConcurrentIndex::ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &qt, int64_t k) {
  if (!myIndexAlgo) {
    throw std::runtime_error("Index algorithm not initialized.");
    return false;
  }

  size_t insertTotal = t.size(0);
  size_t searchTotal = insertTotal * ((1 - writeRatio) / writeRatio);

  size_t qtSize = qt.size(0);
  size_t searchBatchSize = batchSize * ((1 - writeRatio) / writeRatio);

  std::atomic<size_t> insertOps = 0, searchOps = 0;
  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex, resultMutex, bufferMutex;

  INTELLI::ThreadPool pool(numThreads);
  std::vector<SearchRecord> globalSearchBuffer;

  while (insertOps < insertTotal || searchOps < searchTotal) {
    for (size_t i = 0; i < batchSize; ++i) {
      size_t idx = insertOps.fetch_add(1);
      if (idx >= insertTotal) break;
      pool.enqueueTask([&, idx] {
        try {
          auto in = t[idx];
          myIndexAlgo->insertTensor(in);
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastExceptMutex);
          lastException = std::current_exception();
        }
      });
    }

    for (size_t i = 0; i < searchBatchSize; ++i) {
      size_t idx = searchOps.fetch_add(1);
      if (idx >= searchTotal) break;
      pool.enqueueTask([&, idx] {
        try {
          size_t queryIdx = randomMode ? (std::rand() % qtSize) : (idx % qtSize);
          auto q = qt[queryIdx];
          auto res = myIndexAlgo->searchTensor(q, k);
          std::unique_lock<std::mutex> lock(bufferMutex);
          globalSearchBuffer.emplace_back(queryIdx, queryIdx, res);
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastExceptMutex);
          lastException = std::current_exception();
        }
      });
    }
  }

  pool.waitForTasks();

  {
    std::unique_lock<std::mutex> lock(resultMutex);
    searchRes.insert(searchRes.end(), globalSearchBuffer.begin(), globalSearchBuffer.end());
  }

  if (lastException) {
    std::rethrow_exception(lastException);
  }

  return true;
}

bool CANDY::ConcurrentIndex::ccSaveResultAsFile(std::string &outFile) {
  std::ofstream file(outFile, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open result file.");
    return false;
  }
  
  for (const auto& rec : searchRes) {
    uint64_t step = std::get<0>(rec);
    uint64_t queryIdx = std::get<1>(rec);
    auto results = std::get<2>(rec);

    file.write(reinterpret_cast<const char*>(&step), sizeof(step));
    file.write(reinterpret_cast<const char*>(&queryIdx), sizeof(queryIdx));

    for (const auto& tensor : results) {
      auto data = tensor.data_ptr<float>();
      file.write(reinterpret_cast<const char*>(data), tensor.numel() * sizeof(float));
    }
  }
  file.close();

  return true;
}

std::vector<torch::Tensor> CANDY::ConcurrentIndex::searchTensor(torch::Tensor &q, int64_t k) {
  return myIndexAlgo->searchTensor(q, k);
}

void CANDY::ConcurrentIndex::reset() {

}