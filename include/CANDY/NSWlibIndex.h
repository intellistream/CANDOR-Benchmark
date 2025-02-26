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

class NSWlibIndex : public AbstractIndex {
 protected:
  int64_t vecDim;
  int64_t maxElements;
  std::string metricType;
  std::atomic<int64_t> count{0};

  int64_t M;
  int64_t ef;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;

 public:
  NSWlibIndex() = default;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual bool insertTensor(torch::Tensor &t);

  virtual std::vector<torch::Tensor> getDataByTags(int64_t start, int64_t end);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);

  virtual std::vector<faiss::idx_t> searchIndex(torch::Tensor q, int64_t k);
};

typedef std::shared_ptr<class CANDY::NSWlibIndex> NSWlibIndexPtr;

#define newNSWlibIndex std::make_shared<CANDY::NSWlibIndex>
}

#endif // CANDY_NSWLIBINDEX_H

