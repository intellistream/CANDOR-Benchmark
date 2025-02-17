/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/11
 *  Description:
 */

#include <CANDY/ConcurrentIndex.h>
#include <Utils/UtilityFunctions.h>
#include <time.h>
#include <chrono>
#include <assert.h>

CANDY::ThreadPool::ThreadPool(size_t numThreads) {
  for (size_t i = 0; i < numThreads; i++) {
    workers.emplace_back([this] {
      while (true) {
        std::unique_lock<std::mutex> lock(this->queueMutex);
        this->condition.wait(lock, [this] { 
          return this->stop || !this->tasks.empty(); 
        });
        if (this->stop && this->tasks.empty()) return;
        task = std::move(this->tasks.front());
        this->tasks.pop();
      }
      task();
    });
  }
}

CANDY::ThreadlPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) worker.join();
}

void CANDY::ThreadlPool::enqueueTask(std::function<void> task) {
  {
    std::unqiue_lock<std::mutex> lock(queueMutex);
    stop = true;
  }
  condition.notify_one();
}

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

  std::atomic<size_t> commitedOps(0); 
  size_t writeTotal = t.size(0);
  size_t searchTotal = qt.size(0);
  std::exception_ptr lastException = nullptr;
  std::mutex lastExceptMutex, resultMutex;

  ThreadPool pool(numThreads);

  size_t step = 0;
  while (commitedOps < writeTotal) {
    size_t insertCnt = std::min(batchSize, static_cast<int64_t>(writeTotal - commitedOps.load()));
    size_t searchCnt = insertCnt / writeRatio;  
    
    for (size_t i = 0; i < insertCnt; i++) {
      pool.enqueueTask([&, i, step] {
        try {
          auto in = t[commitedOps.fetch_add(1)];
          myIndexAlgo->insertTensor(in);
        } catch (...) {
          std::unique_lock<std::mutex> lock(lastException);
          lastException = std::current_exception();
        }
      });
    }

    std::atomic<size_t> currentSearch(0);
    std::vector<std::thread> readThreads;

    for (size_t i = 0; i < searchCnt; i++) {
      pool.enqueueTask([&, i, step] {
        thread_local std::vector<SearchRecord> localSearchBuffer;
        while (true) {
          size_t idx = currentSearch.fetch_add(1);
          if (idx >= searchCnt) break;
          size_t queryIdx = randomMode ? (std::rand() % searchTotal) : (idx % searchTotal);
          try {
            auto q = qt[queryIdx];
            auto res = myIndexAlgo->searchTensor(q, k);
            localSearchBuffer.emplace_back(commitedOps.load(), queryIdx, res);
          } catch (...) {
            std::unique_lock<std::mutex> lock(lastExceptMutex);
            lastException = std::current_exception();
          }
        }
        std::unique_lock<std::mutex> lock(resultMutex);
        searchRes.insert(searchRes.end(), localSearchBuffer.begin(), localSearchBuffer.end());
      })
    }
    ++step;
  }

  pool.enqueueTask([&] {
    std::unique_lock<std::mutex> lock(resultMutex);
    searchRes.insert(searchRes.end(), localSearchBuffer.begin(), localSearchBuffer.end());
    localSearchBuffer.clear();
  });

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