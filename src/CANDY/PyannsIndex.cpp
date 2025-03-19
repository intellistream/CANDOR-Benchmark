/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#include <thread>
#include <CANDY/PyannsIndex.h>

bool CANDY::PyannsIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  maxElements = cfg->tryI64("maxElements", 10000, true);
  metricType = cfg->tryString("metricType", "IP", true);

  L = cfg->tryI64("L", 400, true);
  R = cfg->tryI64("R", 32, true);
  alpha = cfg->tryDouble("alpha", 1.2f, true);

  int64_t numThreads = cfg->tryI64("numThreads", std::thread::hardware_concurrency(), true);

  diskann::Metric metric;
  if (metricType == "L2") {
    metric = diskann::L2;
  } else if (metricType == "IP") {
    metric = diskann::INNER_PRODUCT;
  } else {
    throw std::invalid_argument("Unsupported metric type");
  }

  auto params = diskann::IndexWriteParametersBuilder(L, R)
                                      .with_alpha(alpha)
                                      .with_saturate_graph(false)
                                      .with_num_threads(numThreads)
                                      .build();
  auto searchParams = diskann::IndexSearchParams(params.search_list_size, params.num_threads);
  diskann::IndexConfig config = diskann::IndexConfigBuilder()
                                            .with_metric(diskann::L2)
                                            .with_dimension(vecDim)
                                            .with_max_points(maxElements)
                                            .is_dynamic_index(true)
                                            .with_index_write_params(params)
                                            .with_index_search_params(searchParams)
                                            .with_data_type(diskann_type_to_name<float>())
                                            .with_tag_type(diskann_type_to_name<uint32_t>())
                                            .with_label_type(diskann_type_to_name<uint32_t>())
                                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                                            .is_enable_tags(true)
                                            .is_filtered(false)
                                            .with_num_frozen_pts(1)
                                            .is_concurrent_consolidate(true)
                                            .build();

  diskann::IndexFactory factory = diskann::IndexFactory(config);
  index = factory.create_instance();
  return true;
}

bool CANDY::PyannsIndex::insertTensor(torch::Tensor &t) {
  if (!index)
    throw std::runtime_error("DiskANN not initialized");
  auto vec = t.to(torch::kCPU).contiguous();
  int64_t curCnt = count.fetch_add(1, std::memory_order_relaxed);
  index->insert_point(vec.data_ptr<float>(), static_cast<uint32_t>(curCnt));
  return true;
}

std::vector<torch::Tensor> CANDY::PyannsIndex::searchTensor(torch::Tensor &qt,
                                                             int64_t k) {
  if (!index)
    throw std::runtime_error("DiskANN not initialized");

  auto q = qt.to(torch::kCPU).contiguous();

  bool isBatch = (q.dim() == 2);
  int64_t rows = isBatch ? q.size(0) : 1;
  std::vector<torch::Tensor> resT(rows);
  std::vector<uint32_t> resTags(k * rows);
  std::vector<float *> res = std::vector<float *>();

  for (int64_t i = 0; i < rows; i++) {
    torch::Tensor query;
    if (isBatch) query = q.slice(0, i, i + 1).squeeze(0);
    else query = q;
    index->search_with_tags(q.data_ptr<float>(), k, L, resTags.data() + i * k, nullptr, res);
    resT[i] = torch::from_blob(res.data(), 
                                {(int64_t)res.size(), vecDim}, 
                                torch::TensorOptions().dtype(torch::kFloat32));
  }

  return resT;
}

std::vector<faiss::idx_t> CANDY::PyannsIndex::searchIndex(torch::Tensor qt,
                                                           int64_t k) {
  auto q = qt.to(torch::kCPU).contiguous();
  if (q.dim() != 1)
    throw std::runtime_error(
        "Expected a single query vector with shape (d,), got shape " +
        std::to_string(q.dim()));

  std::vector<uint32_t> resTags(k);
  std::vector<float *> res = std::vector<float *>();

  index->search_with_tags(q.data_ptr<float>(), k, L, resTags.data(), nullptr, res);

  std::vector<faiss::idx_t> result(resTags.begin(), resTags.end());
  return result;
}

std::vector<torch::Tensor> CANDY::PyannsIndex::getDataByTags(int64_t start,
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