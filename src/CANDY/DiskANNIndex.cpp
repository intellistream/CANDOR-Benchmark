/*
 *  Copyright (C) 2024 by the INTELLI team
 *  Created by: Junyao Dong
 *  Created on: 2025/02/18
 *  Description:
 */

#include <CANDY/DiskANNIndex.h>

bool CANDY::DiskANNIndex::setConfig(INTELLI::ConfigMapPtr cfg) {
  AbstractIndex::setConfig(cfg);
  assert(cfg);
  vecDim = cfg->tryI64("vecDim", 768, true);
  maxElements = cfg->tryI64("maxElements", 10000, true);
  metricType = cfg->tryString("metricType", "IP", true);

  L = cfg->tryI64("L", 400, true);                   
  R = cfg->tryI64("R", 32, true);        
  alpha = cfg->tryDouble("alpha", 1.2f, true);
  
  // diskann::Metric metric;
  // if (metricType == "L2") {
  //   metric = diskann::Metric::L2;
  // } else if (metricType == "IP") {
  //   metric = diskann::Metric::INNER_PRODUCT;
  // } else {
  //   throw std::invalid_argument("Unsupported metric type");
  // }
  
  // auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
  //                                       .with_filter_list_size(Lf)
  //                                       .with_alpha(alpha)
  //                                       .with_saturate_graph(false)
  //                                       .with_num_threads(num_threads)
  //                                       .build();

  // auto filter_params = diskann::IndexFilterParamsBuilder()
  //                           .with_universal_label(universal_label)
  //                           .with_label_file(label_file)
  //                           .with_save_path_prefix(index_path_prefix)
  //                           .build();

  // auto config = diskann::IndexConfigBuilder()
  //                   .with_metric(metric)
  //                   .with_dimension(data_dim)
  //                   .with_max_points(data_num)
  //                   .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
  //                   .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
  //                   .with_data_type(data_type)
  //                   .with_label_type(label_type)
  //                   .is_dynamic_index(false)
  //                   .with_index_write_params(index_build_params)
  //                   .is_enable_tags(false)
  //                   .is_use_opq(use_opq)
  //                   .is_pq_dist_build(use_pq_build)
  //                   .with_num_pq_chunks(build_PQ_bytes)
  //                   .build();

  // auto index_factory = diskann::IndexFactory(config);
  // auto index = index_factory.create_instance();
  // index->build(data_path, data_num, filter_params);

  return true;
}

bool CANDY::DiskANNIndex::insertTensor(torch::Tensor &t) {
  if (!index) throw std::runtime_error("DiskANN not initialized");
  auto vec = t.to(torch::kCPU).contiguous();
  int64_t curCnt = count.fetch_add(1, std::memory_order_relaxed);
  // index->insert_point(vec.data_ptr<float>(), static_cast<uint32_t>(curCnt));
  return true;
}

std::vector<torch::Tensor> CANDY::DiskANNIndex::searchTensor(torch::Tensor &qt, int64_t k) {
  if (!index) throw std::runtime_error("DiskANN not initialized");

  auto q = qt.to(torch::kCPU).contiguous();

  bool isBatch = (q.dim() == 2);
  int64_t rows = isBatch ? q.size(0) : 1;
  std::vector<torch::Tensor> resT(rows);

  // for (int64_t i = 0; i < rows; i++) {
  //   torch::Tensor query;
  //   if (isBatch) 
  //     query = q.slice(0, i, i + 1).squeeze(0); 
  //   else 
  //     query = q; 

  //   std::vector<std::pair<uint32_t, float>> res;
  //   index->search(query.data_ptr<float>(), k, L, res);

  //   torch::Tensor resultTensor = torch::zeros({k, vecDim}, torch::kFloat32);
  //   for (int64_t j = 0; j < k && j < results.size(); j++) {
  //     float* data_ptr = index->get_vector(results[j].first);
  //     resultTensor.slice(0, j, j + 1) = torch::from_blob(data_ptr, {vecDim}, torch::kFloat32).clone();
  //   }
  //   resT[i] = resultTensor;
  // }

  return resT;
}

