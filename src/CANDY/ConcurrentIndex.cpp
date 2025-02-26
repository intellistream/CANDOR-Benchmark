/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/11
 *  Description:
 */

#include <CANDY/ConcurrentIndex.h>
#include <Utils/ThreadPool.hpp>
#include <Utils/UtilityFunctions.h>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <hdf5.h>
#include <set>
#include <time.h>

bool CANDY::ConcurrentIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  assert(cfg);
  std::string concurrentlAlgoTag =
      cfg->tryString("concurrentAlgoTag", "flat", true);
  IndexTable it;
  myIndexAlgo = it.getIndex(concurrentlAlgoTag);
  if (myIndexAlgo == nullptr) {
    throw std::runtime_error("Index algorithm not found: " +
                             concurrentlAlgoTag);
    return false;
  }

  vecDim = cfg->tryI64("vecDim", 128, true);
  ;
  writeRatio = cfg->tryDouble("ccWriteRatio", 0.5, true);
  batchSize = cfg->tryI64("ccBatchSize", 100, true);
  numThreads = cfg->tryI64("ccNumThreads", 1, true);
  randomMode = cfg->tryI64("ccRandomMode", 1, false);
  ccQuerySize = cfg->tryI64("ccQuerySize", 100, true);

  myIndexAlgo->setConfig(cfg);

  return true;
}

bool CANDY::ConcurrentIndex::loadInitialTensor(torch::Tensor &t) {
  auto ru = myIndexAlgo->loadInitialTensor(t);
  initialSize = t.size(0);
  return ru;
}

bool CANDY::ConcurrentIndex::ccInsertAndSearchTensor(torch::Tensor &t,
                                                     torch::Tensor &qt,
                                                     int64_t k) {
  if (!myIndexAlgo) {
    throw std::runtime_error("Index algorithm not initialized.");
    return false;
  }

  ccQuery = qt;
  ANNK = k;

  size_t insertTotal = t.size(0);
  size_t searchTotal = insertTotal * ((1 - writeRatio) / writeRatio);
  size_t qtSize = qt.size(0);
  size_t searchBatchSize = batchSize * ((1 - writeRatio) / writeRatio);

  std::atomic<size_t> nextInsertIdx = 0, nextSearchIdx = 0;
  std::atomic<size_t> completedInserts = 0, completedSearches = 0;

  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex, resultMutex, insertLatencyMutex,
      searchLatencyMutex;

  std::vector<double> insertLatencies, searchLatencies;

  INTELLI::ThreadPool pool(numThreads);

  auto startTime = std::chrono::high_resolution_clock::now();
  while (nextInsertIdx < insertTotal || nextSearchIdx < searchTotal) {
    size_t startInsert = nextInsertIdx.fetch_add(batchSize);
    size_t endInsert = std::min(startInsert + batchSize, insertTotal);

    for (size_t idx = startInsert; idx < endInsert; ++idx) {
      pool.enqueueTask([&, idx] {
        auto taskStart = std::chrono::high_resolution_clock::now();
        try {
          auto in = t[idx];
          myIndexAlgo->insertTensor(in);
          completedInserts.fetch_add(1);

          auto taskEnd = std::chrono::high_resolution_clock::now();
          double latency =
              std::chrono::duration<double, std::milli>(taskEnd - taskStart)
                  .count();
          {
            std::unique_lock<std::mutex> lock(insertLatencyMutex);
            insertLatencies.push_back(latency);
          }
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastExceptMutex);
          lastException = std::current_exception();
        }
      });
    }

    size_t startSearch = nextSearchIdx.fetch_add(searchBatchSize);
    size_t endSearch = std::min(startSearch + searchBatchSize, searchTotal);

    size_t batchIdx = startSearch / searchBatchSize;
    size_t batchEnd = initialSize + (batchIdx + 1) * searchBatchSize;

    for (size_t idx = startSearch; idx < endSearch; ++idx) {
      pool.enqueueTask([&, idx, batchEnd] {
        auto taskStart = std::chrono::high_resolution_clock::now();
        try {
          size_t queryIdx =
              randomMode ? (std::rand() % qtSize) : (idx % qtSize);
          auto q = qt[queryIdx];
          auto res = myIndexAlgo->searchIndex(q, k);
          completedSearches.fetch_add(1);

          auto taskEnd = std::chrono::high_resolution_clock::now();
          double latency =
              std::chrono::duration<double, std::milli>(taskEnd - taskStart)
                  .count();
          {
            std::unique_lock<std::mutex> lock(searchLatencyMutex);
            searchLatencies.push_back(latency);
          }
          {
            std::unique_lock<std::mutex> lock(resultMutex);
            searchRes.emplace_back(batchEnd, queryIdx, res);
          }
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastExceptMutex);
          lastException = std::current_exception();
        }
      });
    }
  }

  pool.waitForTasks();

  auto endTime = std::chrono::high_resolution_clock::now();
  double elapsedSec =
      std::chrono::duration<double>(endTime - startTime).count();

  insertThroughput = completedInserts / elapsedSec;
  searchThroughput = completedSearches / elapsedSec;

  if (lastException) {
    std::rethrow_exception(lastException);
  }

  auto getPercentile = [](std::vector<double> &latencies,
                          double percentile) -> double {
    if (latencies.empty())
      return 0.0;
    std::sort(latencies.begin(), latencies.end());
    size_t index = static_cast<size_t>(percentile * latencies.size() / 100.0);
    return latencies[std::min(index, latencies.size() - 1)];
  };

  auto getAverageLatency = [](const std::vector<double> &latencies) -> double {
    if (latencies.empty())
      return 0.0;
    return std::accumulate(latencies.begin(), latencies.end(), 0.0) /
           latencies.size();
  };

  insertLatencyAvg = getAverageLatency(insertLatencies);
  insertLatency95 = getPercentile(insertLatencies, 95.0);

  searchLatencyAvg = getAverageLatency(searchLatencies);
  searchLatency95 = getPercentile(searchLatencies, 95.0);

  return true;
}

std::map<std::string, double>
CANDY::ConcurrentIndex::ccSaveAndGetResults(std::string &outFile) {
  std::map<std::string, double> metrics = {
      {"insertThroughput", std::round(insertThroughput * 100) / 100.0},
      {"searchThroughput", std::round(searchThroughput * 100) / 100.0},
      {"insertLatencyAvg", std::round(insertLatencyAvg * 10000) / 10000.0},
      {"searchLatencyAvg", std::round(searchLatencyAvg * 10000) / 10000.0},
      {"insertLatency95", std::round(insertLatency95 * 10000) / 10000.0},
      {"searchLatency95", std::round(searchLatency95 * 10000) / 10000.0}};
  calcStepwiseRecall(outFile);

  std::cout << "Concurrent search results saved: " << searchRes.size()
            << std::endl;
  return metrics;
}

void CANDY::ConcurrentIndex::calcStepwiseRecall(std::string &outFile) {
  std::unordered_map<int64_t, std::tuple<double, int, double>> batchData;
  std::mutex batchDataMutex;

  INTELLI::ThreadPool pool(std::thread::hardware_concurrency());

  std::unordered_map<int64_t,
                     std::vector<std::pair<int64_t, std::vector<int64_t>>>>
      batchQueries;
  for (const auto &record : searchRes) {
    int64_t batch = std::get<0>(record);
    int64_t queryIdx = std::get<1>(record);
    std::vector<int64_t> res = std::get<2>(record);
    batchQueries[batch].emplace_back(queryIdx, res);
  }

  for (const auto &[batch, queries] : batchQueries) {
    pool.enqueueTask([&batchData, &batchDataMutex, batch, queries, this]() {
      std::vector<torch::Tensor> rangeTVec =
          myIndexAlgo->getDataByTags(0, batch);
      torch::Tensor rangeT =
          torch::stack(rangeTVec); 

      for (const auto &[queryIdx, searchResults] : queries) {
        torch::Tensor query = ccQuery[queryIdx];
        if (query.dim() == 1) {
          query = query.unsqueeze(0); 
        }

        torch::Tensor dist = torch::norm(query - rangeT, 2, 1); 
        std::vector<int64_t> sortedIndices(batch);
        for (int64_t i = 0; i < batch; ++i) {
          sortedIndices[i] = i;
        }
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&dist](int64_t a, int64_t b) {
                    return dist[a].item<float>() < dist[b].item<float>();
                  });

        int k = searchResults.size();
        std::vector<int64_t> trueKnn(sortedIndices.begin(),
                                     sortedIndices.begin() + k);

        int intersectionCount = 0;
        for (int64_t trueIdx : trueKnn) {
          if (std::find(searchResults.begin(), searchResults.end(), trueIdx) !=
              searchResults.end()) {
            intersectionCount++;
          }
        }
        double recall = static_cast<double>(intersectionCount) / k;

        {
          std::lock_guard<std::mutex> lock(batchDataMutex);
          auto &[sum, count, minRecall] = batchData[batch];
          sum += recall;
          count += 1;
          if (count == 1 || recall < minRecall) {
            minRecall = recall;
          }
        }
      }
    });
  }

  // 等待所有並行任務完成
  pool.waitForTasks();

  // 將結果寫入文件
  std::ofstream ofs(outFile);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open output file: " << outFile << std::endl;
    return;
  }
  ofs << "Batch, AvgRecall, WorstRecall\n";
  for (const auto &[batch, data] : batchData) {
    auto [sum, count, minRecall] = data;
    double avgRecall = sum / count;
    ofs << batch << ", " << avgRecall << ", " << minRecall << "\n";
  }
  ofs.close();
}

std::vector<torch::Tensor>
CANDY::ConcurrentIndex::searchTensor(torch::Tensor &q, int64_t k) {
  return myIndexAlgo->searchTensor(q, k);
}

void CANDY::ConcurrentIndex::reset() {}