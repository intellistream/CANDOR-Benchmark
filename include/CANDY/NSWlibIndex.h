/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#ifndef CANDY_NSWLIBINDEX_H
#define CANDY_NSWLIBINDEX_H

#include <hnswlib/hnswlib.h>
#include <CANDY/AbstractIndex.h>

#include <any>

namespace CANDY {
class NSWlibHNSW {
 private:
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  std::atomic<int64_t> count{0};
  int64_t vecDim;

 public:
  template <typename T>
  void init(const std::string& spaceType, int64_t vecDim, int64_t maxElements, int64_t M, int64_t ef) {
    this->vecDim = vecDim;
    if (spaceType == "L2") {
      space = std::make_unique<hnswlib::L2Space>(vecDim);
    } else if (spaceType == "IP") {
      space = std::make_unique<hnswlib::InnerProductSpace>(vecDim);
    } else {
      throw std::invalid_argument("Unsupported space type");
    }
    index = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), maxElements, M, ef);
  }

  template <typename T>
  void insertTensor(torch::Tensor &t) {
    if (!index) throw std::runtime_error("NSWlib_HNSW not initialized");
    auto vec = t.to(torch::kCPU).contiguous();
    index->addPoint(vec.data_ptr<float>(), count++);
  }

  template <typename T>
  std::vector<torch::Tensor> searchTensor(torch::Tensor &qt, int64_t k) {
    if (!index) throw std::runtime_error("NSWlibHNSW not initialized");
    auto q = qt.to(torch::kCPU).contiguous();
    auto res = index->searchKnn(q.data_ptr<float>(), k);

    std::vector<torch::Tensor> resT;
    while (!res.empty()) {
      char* data_ptr = static_cast<char*>(index->getDataByInternalId(res.top().second));
      torch::Tensor t = torch::from_blob(data_ptr, {vecDim}, torch::kFloat32).clone();
      resT.push_back(t);
      res.pop();
    }
    return resT;
  }
};

class NSWlibIndex : public AbstractIndex {
 protected:
  int64_t vecDim;
  int64_t M;
  int64_t ef;
  int64_t maxElements;
  std::string metricType;

  NSWlibHNSW index;

 public:
  NSWlibIndex() = default;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual bool loadInitialTensor(torch::Tensor &t);

  virtual bool insertTensor(torch::Tensor &t);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
};

typedef std::shared_ptr<class CANDY::NSWlibIndex> NSWlibIndexPtr;

#define newNSWlibIndex std::make_shared<CANDY::NSWlibIndex>
}

#endif // CANDY_NSWLIBINDEX_H

