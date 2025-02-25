/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#ifndef CANDY_DISKANNINDEX_H
#define CANDY_DISKANNINDEX_H

#include "../../thirdparty/DiskANN/include/index.h"
#include "../../thirdparty/DiskANN/include/parameters.h"
#include <CANDY/AbstractIndex.h>

#include <any>

namespace CANDY {

class DiskANNIndex : public AbstractIndex {
 protected:
  int64_t vecDim;
  int64_t maxElements;
  std::string metricType;
  std::atomic<int64_t> count{0};

  unsigned L;               
  unsigned R;                
  float alpha;

  std::unique_ptr<diskann::Index<float>> index;

 public:
  DiskANNIndex() = default;

  virtual bool setConfig(INTELLI::ConfigMapPtr cfg);

  virtual bool insertTensor(torch::Tensor &t);

  virtual std::vector<torch::Tensor> searchTensor(torch::Tensor &q, int64_t k);
};

typedef std::shared_ptr<class CANDY::DiskANNIndex> DiskANNIndexPtr;

#define newDiskANNIndex std::make_shared<CANDY::DiskANNIndex>
}

#endif // CANDY_DISKANNINDEX_H

