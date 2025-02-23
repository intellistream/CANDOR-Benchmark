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

  std::atomic<size_t> nextInsertIdx = 0, nextSearchIdx = 0;
  std::atomic<size_t> completedInserts = 0, completedSearches = 0;

  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex, resultMutex, bufferMutex;

  INTELLI::ThreadPool pool(numThreads);

  thread_local std::vector<SearchRecord> localSearchBuffer;

  thread_local std::vector<double> localInsertLatencies, localSearchLatencies;

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
          localInsertLatencies.push_back(latency);
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastExceptMutex);
          lastException = std::current_exception();
        }
      });
    }

    size_t startSearch = nextSearchIdx.fetch_add(searchBatchSize);
    size_t endSearch = std::min(startSearch + searchBatchSize, searchTotal);

    for (size_t idx = startSearch; idx < endSearch; ++idx) {
      pool.enqueueTask([&, idx] {
        auto taskStart = std::chrono::high_resolution_clock::now();
        try {
          size_t queryIdx = randomMode ? (std::rand() % qtSize) : (idx % qtSize);
          auto q = qt[queryIdx];
          auto res = myIndexAlgo->searchTensor(q, k);
          completedSearches.fetch_add(1);

          auto taskEnd = std::chrono::high_resolution_clock::now();
          double latency = std::chrono::duration<double, std::milli>(taskEnd - taskStart).count();
          localSearchLatencies.push_back(latency);

          localSearchBuffer.emplace_back(queryIdx, queryIdx, res);
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

  std::vector<double> insertLatencies, searchLatencies;
  insertLatencies.insert(insertLatencies.end(), localInsertLatencies.begin(), localInsertLatencies.end());
  searchLatencies.insert(searchLatencies.end(), localSearchLatencies.begin(), localSearchLatencies.end());

  {
    std::unique_lock<std::mutex> lock(resultMutex);
    searchRes.insert(searchRes.end(), localSearchBuffer.begin(), localSearchBuffer.end());
  }

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
  metrics["insertThroughput"] = insertThroughput;
  metrics["searchThroughput"] = searchThroughput;

  metrics["insertLatencyAvg"] = insertLatencyAvg;
  metrics["searchLatencyAvg"] = searchLatencyAvg;

  metrics["insertLatency95"] = insertLatency95;
  metrics["searchLatency95"] = searchLatency95;

  hid_t fileId, groupId, queryGroupId, dataspaceId, datasetId, attributeId;
  herr_t status;

  fileId = H5Fcreate(outFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (fileId < 0) {
    throw std::runtime_error("Failed to create HDF5 file: " + outFile);
  }

  groupId = H5Gcreate2(fileId, HDF5_GROUP_NAME, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (groupId < 0) {
    H5Fclose(fileId);
    throw std::runtime_error("Failed to create HDF5 group: " + std::string(HDF5_GROUP_NAME));
  }

  for (size_t i = 0; i < searchRes.size(); ++i) {
    const auto& rec = searchRes[i];
    uint64_t step = std::get<0>(rec);
    uint64_t queryIdx = std::get<1>(rec);
    const auto& results = std::get<2>(rec);

    std::string queryGroupName = std::string(HDF5_QUERY_GROUP_PREFIX) + std::to_string(i);

    queryGroupId = H5Gcreate2(groupId, queryGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (queryGroupId < 0) {
      H5Gclose(groupId);
      H5Fclose(fileId);
      throw std::runtime_error("Failed to create query group: " + queryGroupName);
    }

    dataspaceId = H5Screate(H5S_SCALAR);
    attributeId = H5Acreate2(queryGroupId, HDF5_STEP_NAME, H5T_NATIVE_UINT64, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attributeId, H5T_NATIVE_UINT64, &step);
    H5Aclose(attributeId);
    H5Sclose(dataspaceId);

    dataspaceId = H5Screate(H5S_SCALAR);
    attributeId = H5Acreate2(queryGroupId, HDF5_QUERY_IDX_NAME, H5T_NATIVE_UINT64, dataspaceId, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attributeId, H5T_NATIVE_UINT64, &queryIdx);
    H5Aclose(attributeId);
    H5Sclose(dataspaceId);

    for (size_t j = 0; j < results.size(); ++j) {
      const auto& tensor = results[j];
      float* data = tensor.data_ptr<float>();
      hsize_t dims[1] = {tensor.numel()};

      std::string datasetName = queryGroupName + std::string(HDF5_DATA_PREFIX) + std::to_string(j);

      dataspaceId = H5Screate_simple(1, dims, nullptr);
      datasetId = H5Dcreate2(fileId, datasetName.c_str(), H5T_NATIVE_FLOAT, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

      H5Dclose(datasetId);
      H5Sclose(dataspaceId);
    }
    H5Gclose(queryGroupId);
  }
  H5Gclose(groupId);
  H5Fclose(fileId);

  std::cout << "Concurrent search results saved: " << outFile <<std::endl;

  return metrics;
}

std::vector<torch::Tensor> CANDY::ConcurrentIndex::searchTensor(torch::Tensor &q, int64_t k) {
  return myIndexAlgo->searchTensor(q, k);
}

void CANDY::ConcurrentIndex::reset() {

}