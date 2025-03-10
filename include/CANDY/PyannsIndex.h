/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#ifndef CANDY_PYANNSINDEX_H
#define CANDY_PYANNSINDEX_H

#include "../../thirdparty/DiskANN/include/index.h"
#include "../../thirdparty/DiskANN/include/parameters.h"
#include "../../thirdparty/DiskANN/include/index_factory.h"

#include <CANDY/AbstractIndex.h>

#include <any>

namespace CANDY {

class PyannsIndex : public AbstractIndex {
 protected:
  int64_t vecDim;
  int64_t maxElements;
  std::string metricType;
  std::atomic<int64_t> count{1};

  unsigned L;               
  unsigned R;                
  float alpha;

  std::unique_ptr<diskann::Index<float, uint32_t, uint32_t>> index;

 public:
  PyannsIndex() = default;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual bool insertTensor(torch::Tensor &t);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);

  std::vector<faiss::idx_t> searchIndex(torch::Tensor qt, int64_t k);

 private:
  std::vector<torch::Tensor> getDataByTags(int64_t start, int64_t end);
};

typedef std::shared_ptr<class CANDY::PyannsIndex> PyannsIndexPtr;

#define newPyannsIndex std::make_shared<CANDY::PyannsIndex>
}

#endif // CANDY_PYANNSINDEX_H

