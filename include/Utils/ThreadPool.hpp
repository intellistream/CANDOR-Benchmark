/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/17
 *  Description:
 */

#ifndef CANDY_INCLUDE_UTILS_THREADPOOL_HPP_
#define CANDY_INCLUDE_UTILS_THREADPOOL_HPP_

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <exception>

namespace INTELLI {

class ThreadPool {
 public:
  ThreadPool(size_t numThreads);

  ~ThreadPool();

  void enqueueTask(std::function<void()> task);

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queueMutex;
  std::condition_variable condition;
  bool stop = false;
};

}

#endif // CANDY_INCLUDE_UTILS_THREADPOOL_HPP_