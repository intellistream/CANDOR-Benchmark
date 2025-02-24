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
#include <set>
#include <cmath>
#include <assert.h>
#include <hdf5.h>

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
  ccQuerySize = cfg->tryI64("ccQuerySize", 100, true);

  myIndexAlgo->setConfig(cfg);
  return true;
}

bool CANDY::ConcurrentIndex::loadInitialTensor(torch::Tensor &t) {
  auto ru = myIndexAlgo->loadInitialTensor(t);
  initialSize = t.size(0);
  return ru;
}

bool CANDY::ConcurrentIndex::ccInsertAndSearchTensor(torch::Tensor &t, torch::Tensor &qt, int64_t k) {
  if (!myIndexAlgo) {
    throw std::runtime_error("Index algorithm not initialized.");
    return false;
  }

  ANNK = k;
  
  size_t insertTotal = t.size(0);
  size_t searchTotal = insertTotal * ((1 - writeRatio) / writeRatio);
  size_t qtSize = qt.size(0);
  size_t searchBatchSize = batchSize * ((1 - writeRatio) / writeRatio);

  std::atomic<size_t> nextInsertIdx = 0, nextSearchIdx = 0;
  std::atomic<size_t> completedInserts = 0, completedSearches = 0;

  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex, resultMutex, insertLatencyMutex, searchLatencyMutex;

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
          double latency = std::chrono::duration<double, std::milli>(taskEnd - taskStart).count();
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
          size_t queryIdx = randomMode ? (std::rand() % qtSize) : (idx % qtSize);
          auto q = qt[queryIdx];
          auto res = myIndexAlgo->searchTensor(q, k)[0];
          completedSearches.fetch_add(1); 

          auto taskEnd = std::chrono::high_resolution_clock::now();
          double latency = std::chrono::duration<double, std::milli>(taskEnd - taskStart).count();
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
  double elapsedSec = std::chrono::duration<double>(endTime - startTime).count();

  insertThroughput = completedInserts / elapsedSec;
  searchThroughput = completedSearches / elapsedSec;

  if (lastException) {
    std::rethrow_exception(lastException);
  }

  auto getPercentile = [](std::vector<double> &latencies, double percentile) -> double {
    if (latencies.empty()) return 0.0;
    std::sort(latencies.begin(), latencies.end());
    size_t index = static_cast<size_t>(percentile * latencies.size() / 100.0);
    return latencies[std::min(index, latencies.size() - 1)];
  };

  auto getAverageLatency = [](const std::vector<double> &latencies) -> double {
    if (latencies.empty()) return 0.0;
    return std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  };

  insertLatencyAvg = getAverageLatency(insertLatencies);
  insertLatency95 = getPercentile(insertLatencies, 95.0);

  searchLatencyAvg = getAverageLatency(searchLatencies);
  searchLatency95 = getPercentile(searchLatencies, 95.0);

  return true;
}

std::map<std::string, double> CANDY::ConcurrentIndex::ccSaveAndGetResults(std::string& outFile) {
  std::map<std::string, double> metrics;
  metrics["insertThroughput"] = int(insertThroughput);
  metrics["searchThroughput"] = int(searchThroughput);
  metrics["insertLatencyAvg"] = std::round(insertLatencyAvg * 10000) / 10000.0;
  metrics["searchLatencyAvg"] = std::round(searchLatencyAvg * 10000) / 10000.0;
  metrics["insertLatency95"] = std::round(insertLatency95 * 10000) / 10000.0;
  metrics["searchLatency95"] = std::round(searchLatency95 * 10000) / 10000.0;

  hid_t fileId = -1, groupId = -1, batchGroupId = -1, queryGroupId = -1, dataspaceId, datasetId;
  herr_t status;

  if (searchRes.empty()) 
    return metrics;

  fileId = H5Fcreate(outFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (fileId < 0) {
    std::cerr << "Failed to create HDF5 file: " << outFile << std::endl;
    return metrics;
  }

  // std::cout << "Creating group: /search_results" << std::endl;
  groupId = H5Gcreate2(fileId, "/search_results", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (groupId < 0) {
    std::cerr << "Failed to create HDF5 group: /search_results" << std::endl;
    H5Fclose(fileId);
    return metrics;
  }

  hid_t attrDatatype = H5Tcopy(H5T_NATIVE_HSIZE);
  hid_t attrDataspace = H5Screate(H5S_SCALAR);
  hid_t attrId = H5Acreate2(groupId, "k", attrDatatype, attrDataspace, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attrId, H5T_NATIVE_HSIZE, &ANNK);
  H5Aclose(attrId);
  H5Sclose(attrDataspace);
  H5Tclose(attrDatatype);

  std::map<BatchIndex, std::vector<std::pair<QueryIndex, torch::Tensor>>> groupedResults;
  for (const auto& rec : searchRes) {
    BatchIndex step = std::get<0>(rec);
    QueryIndex queryIdx = std::get<1>(rec);
    auto t = std::get<2>(rec);
    groupedResults[step].emplace_back(queryIdx, t);
  }

  for (const auto& [step, queries] : groupedResults) {
    std::string batchGroupName = "batch_" + std::to_string(step);
    // std::cout << "Creating batch group: " << batchGroupName << std::endl;
    batchGroupId = H5Gcreate2(groupId, batchGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (batchGroupId < 0) {
      std::cerr << "Failed to create batch group: " << batchGroupName << std::endl;
      H5Gclose(groupId);
      H5Fclose(fileId);
      return metrics;
    }

    std::set<QueryIndex> seenQueryIdx;
    for (const auto& [queryIdx, t] : queries) {
      std::string queryGroupName = "query_" + std::to_string(queryIdx);
      if (seenQueryIdx.count(queryIdx) > 0) continue;
      seenQueryIdx.insert(queryIdx);

      queryGroupId = H5Gcreate2(batchGroupId, queryGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      if (queryGroupId < 0) {
        std::cerr << "Failed to create query group: " << queryGroupName << std::endl;
        H5Gclose(batchGroupId);
        H5Gclose(groupId);
        H5Fclose(fileId);
        return metrics;
      }

      int64_t len = t.size(0);
      for (size_t j = 0; j < t.size(0); ++j) {
        const auto& tensor = t[j];
        float* data = tensor.data_ptr<float>();
        if (!data) {
          std::cerr << "Invalid tensor data for " << queryGroupName << "/tensor_" << j << std::endl;
          continue;
        }
        hsize_t dims[1] = {static_cast<hsize_t>(tensor.numel())};
        std::string datasetName = "tensor_" + std::to_string(j);
        // std::cout << "Creating dataset: " << batchGroupName << "/" << queryGroupName << "/" << datasetName << std::endl;
        dataspaceId = H5Screate_simple(1, dims, nullptr);
        datasetId = H5Dcreate2(queryGroupId, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspaceId, 
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (datasetId < 0) {
          std::cerr << "Failed to create dataset: " << datasetName << std::endl;
          H5Sclose(dataspaceId);
          continue;
        }
        status = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        if (status < 0) {
          std::cerr << "Failed to write dataset: " << datasetName << std::endl;
        }
        H5Dclose(datasetId);
        H5Sclose(dataspaceId);
      }
      H5Gclose(queryGroupId);
    }
    H5Gclose(batchGroupId);
  }
  H5Gclose(groupId);
  H5Fclose(fileId);

  std::cout << "Concurrent search results saved: " << outFile << std::endl;
  return metrics;
}

std::vector<torch::Tensor> CANDY::ConcurrentIndex::searchTensor(torch::Tensor &q, int64_t k) {
  return myIndexAlgo->searchTensor(q, k);
}

void CANDY::ConcurrentIndex::reset() {

}