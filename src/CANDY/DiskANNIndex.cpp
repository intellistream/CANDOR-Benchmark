/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#include <CANDY/DiskANNIndex.h>

bool CANDY::DiskANNIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  std::cout << "setting ...  \n";
  AbstractIndex::setConfig(cfg);
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  maxElements = cfg->tryI64("maxElements", 10000, true);
  metricType = cfg->tryString("metricType", "IP", true);

  L = cfg->tryI64("L", 400, true);
  R = cfg->tryI64("R", 32, true);
  alpha = cfg->tryDouble("alpha", 1.2f, true);

  int64_t numThreads = cfg->tryI64("numThreads", 1, true);

  diskann::Metric metric;
  if (metricType == "L2") {
    metric = diskann::L2;
  } else if (metricType == "IP") {
    metric = diskann::INNER_PRODUCT;
  } else {
    throw std::invalid_argument("Unsupported metric type");
  }

  diskann::IndexWriteParameters paras =
      diskann::IndexWriteParametersBuilder(L, R)
          .with_filter_list_size(0)
          .with_alpha(alpha)
          .with_saturate_graph(false)
          .with_num_threads(numThreads)
          .build();
  index = std::make_unique<diskann::Index<float, uint32_t, uint32_t>>(
      metric, vecDim, maxElements, true, paras, 100, numThreads, true, true,
      false, 0, false);

  return true;
}

bool CANDY::DiskANNIndex::insertTensor(torch::Tensor &t) {
  if (!index)
    throw std::runtime_error("DiskANN not initialized");
  auto vec = t.to(torch::kCPU).contiguous();
  int64_t curCnt = count.fetch_add(1, std::memory_order_relaxed);
  if (index->insert_point(vec.data_ptr<float>(), static_cast<uint32_t>(curCnt)))
    std::cerr << "error in inserting point " << curCnt << std::endl;
  return true;
}

std::vector<torch::Tensor> CANDY::DiskANNIndex::searchTensor(torch::Tensor &qt,
                                                             int64_t k) {
  if (!index)
    throw std::runtime_error("DiskANN not initialized");

  auto q = qt.to(torch::kCPU).contiguous();

  bool isBatch = (q.dim() == 2);
  int64_t rows = isBatch ? q.size(0) : 1;
  std::vector<torch::Tensor> resT(rows);

  for (int64_t i = 0; i < rows; i++) {
    torch::Tensor query;
    if (isBatch)
      query = q.slice(0, i, i + 1).squeeze(0);
    else
      query = q;

    std::vector<uint32_t> resIds(k);
    std::vector<float> distances(k);
    index->search(query.data_ptr<float>(), k, L, resIds.data(),
                  distances.data());

    torch::Tensor resultTensor = torch::zeros({k, vecDim}, torch::kFloat32);
    for (uint32_t j = 0; j < k && j < resIds.size(); j++) {
      uint32_t tag = resIds[j];
      float *vec = new float[vecDim];
      int status = index->get_vector_by_tag(tag, vec);
      if (status == 0)
        resultTensor.slice(0, j, j + 1) =
            torch::from_blob(vec, {vecDim}, torch::kFloat32).clone();
      delete[] vec;
    }
    resT[i] = resultTensor;
  }

  return resT;
}

std::vector<faiss::idx_t> CANDY::DiskANNIndex::searchIndex(torch::Tensor qt,
                                                           int64_t k) {
  auto q = qt.to(torch::kCPU).contiguous();
  if (q.dim() != 1)
    throw std::runtime_error(
        "Expected a single query vector with shape (d,), got shape " +
        std::to_string(q.dim()));

  std::vector<uint32_t> resIds(k);
  std::vector<float> distances(k);
  index->search(q.data_ptr<float>(), k, L, resIds.data(), distances.data());
  std::vector<faiss::idx_t> result(resIds.begin(), resIds.end());
  return result;
}

std::vector<torch::Tensor> CANDY::DiskANNIndex::getDataByTags(int64_t start,
                                                              int64_t end) {
  int64_t numElements = end - start;
  std::vector<torch::Tensor> result(numElements);
  for (int64_t i = 0; i < numElements; ++i) {
    uint32_t tag = start + i;
    float *vec = new float[vecDim];
    int status = index->get_vector_by_tag(tag, vec);

    if (status) {
      torch::Tensor tensor =
          torch::from_blob(vec, {vecDim}, torch::kFloat32).clone();
      result[i] = tensor;
    } else {
      result[i] = torch::zeros({vecDim}, torch::kFloat32);
    }
    delete[] vec;
  }
  return result;
}
