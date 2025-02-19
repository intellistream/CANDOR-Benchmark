/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/17
 *  Description:
 */

#include <Utils/ThreadPool.hpp>

namespace INTELLI {

INTELLI::ThreadPool::ThreadPool(size_t numThreads) : 
    stop(false), activeTasks(0) {
  for (size_t i = 0; i < numThreads; ++i) {
    workers.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queueMutex);
          this->condition.wait(lock, [this] { 
            return this->stop || !this->tasks.empty(); 
          });
          if (this->stop && this->tasks.empty()) return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
          ++activeTasks;
        }
        task();
        --activeTasks;
        finishCondition.notify_one();
      }
    });
  }
}

INTELLI::ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) worker.join();
}

void INTELLI::ThreadPool::enqueueTask(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    if (stop) throw std::runtime_error("ThreadPool has been stopped.");
    tasks.push(std::move(task));
  }
  condition.notify_one();
}

void INTELLI::ThreadPool::waitForTasks() {
  std::unique_lock<std::mutex> lock(queueMutex);
  finishCondition.wait(lock, [this] { return tasks.empty() && activeTasks == 0; });
}

}