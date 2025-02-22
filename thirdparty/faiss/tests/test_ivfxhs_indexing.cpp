/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexXHS.h>
#include <faiss/index_io.h>

TEST(IndexXHS, accuracy) {
    // 维度
    int d = 64;
    // 训练集大小
    size_t nt = 10000;
    // 数据库大小
    size_t nb = 1000;
    // 查询数量
    int nq = 200;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // 生成训练数据
    std::vector<float> trainvecs(nt * d);
    for (size_t i = 0; i < nt * d; i++) {
        trainvecs[i] = distrib(rng);
    }

    // 创建索引
    faiss::IndexFlatL2 coarse_quantizer(d);
    int ncentroids = 25;
    faiss::IndexXHS index(&coarse_quantizer, d, ncentroids, 16, 8);
    faiss::IndexFlatL2 index_gt(d);

    // 训练索引
    index.train(nt, trainvecs.data());

    // 生成数据库数据
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }
    index.add(nb, database.data());
    index_gt.add(nb, database.data());

    // 生成查询数据
    std::vector<float> queries(nq * d);
    for (size_t i = 0; i < nq * d; i++) {
        queries[i] = distrib(rng);
    }

    // 获取 ground-truth 结果
    std::vector<faiss::idx_t> gt_nns(nq);
    std::vector<float> gt_dis(nq);
    index_gt.search(nq, queries.data(), 1, gt_dis.data(), gt_nns.data());

    // 在 `IndexXHS` 上进行 ANN 搜索
    index.nprobe = 5;
    int k = 5;
    std::vector<faiss::idx_t> nns(k * nq);
    std::vector<float> dis(k * nq);
    index.search(nq, queries.data(), k, dis.data(), nns.data());

    // 计算准确率
    int n_ok = 0;
    for (int q = 0; q < nq; q++) {
        for (int i = 0; i < k; i++) {
            if (nns[q * k + i] == gt_nns[q]) {
                n_ok++;
            }
        }
    }

    // 期望命中率 > 40%
    EXPECT_GT(n_ok, nq * 0.4);
}
