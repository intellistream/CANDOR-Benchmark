/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 25-2-19 下午3:32
 * Description: ${DESCRIPTION}
 */
#include <faiss/invlists/BlockInvertedListsXHS.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

namespace faiss {


BlockInvertedListsXHS::BlockInvertedListsXHS(
        size_t nlist,
        size_t vec_per_block,
        size_t block_size)
        : InvertedLists(nlist, InvertedLists::INVALID_CODE_SIZE),
          n_per_block(vec_per_block),
          block_size(block_size),
          rearrange_threshold(10000) { // 设置碎片整理的阈值
    memory_pool.resize(1UL << 30); // 预分配1GB GPU内存
    cur_pool_ptr.store(0);
    heads.resize(nlist, nullptr);
    tails.resize(nlist, nullptr);
}

BlockInvertedListsXHS::BlockInvertedListsXHS(
        size_t nlist,
        const CodePacker* packer)
        : InvertedLists(nlist, InvertedLists::INVALID_CODE_SIZE),
          n_per_block(packer->nvec),
          block_size(packer->block_size),
          packer(packer),
          rearrange_threshold(10000) { // 设置碎片整理的阈值
    memory_pool.resize(1UL << 30); // 预分配1GB GPU内存
    cur_pool_ptr.store(0);
    heads.resize(nlist, nullptr);
    tails.resize(nlist, nullptr);
}
// 分配新的 GPU 内存块
MemoryBlock* BlockInvertedListsXHS::allocate_block() {
    size_t block_mem = sizeof(MemoryBlock) + n_per_block * (sizeof(idx_t) + code_size);
    block_mem = (block_mem + 63) & ~63;
    size_t ptr = cur_pool_ptr.fetch_add(block_mem);
    FAISS_ASSERT(ptr + block_mem <= memory_pool.size());

    auto* block = reinterpret_cast<MemoryBlock*>(memory_pool.data() + ptr);
    block->prev = nullptr;
    block->next = nullptr;
    block->capacity = n_per_block;
    block->size = 0;
    block->is_merged = false;
    block->ids = reinterpret_cast<idx_t*>(block + 1);
    block->codes = reinterpret_cast<uint8_t*>(block->ids + n_per_block);
    return block;
}

// 插入向量数据
void BlockInvertedListsXHS::insert_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes) {
    FAISS_ASSERT(list_no < nlist);

    // 获取当前尾部块
    MemoryBlock* current = tails[list_no];

    // 计算单个向量的编码数据大小
    size_t single_code_size = packer ? packer->code_size : code_size;

    size_t remaining = n_entry; // 还需要插入的向量数量
    const idx_t* cur_ids = ids_in;
    const uint8_t* cur_codes = codes;

    while (remaining > 0) {
        if (!current || current->size >= current->capacity) {
            // 需要分配新块
            MemoryBlock* new_block = allocate_block();

            // 更新链表
            if (tails[list_no]) {
                tails[list_no]->next = new_block;
                new_block->prev = tails[list_no];
            } else {
                heads[list_no] = new_block;
            }
            tails[list_no] = new_block;
            current = new_block;
        }

        // 计算可以存入当前块的向量数量
        size_t space_left = current->capacity - current->size;
        size_t to_insert = std::min(space_left, remaining);

        // 逐个复制 ID 到 GPU
        cudaMemcpy(current->ids + current->size, cur_ids, to_insert * sizeof(idx_t), cudaMemcpyHostToDevice);

        // 逐个复制 Codes 到 GPU
        for (size_t i = 0; i < to_insert; ++i) {
            // 每次插入一个向量，调用 pack_1 来压缩代码
            if (packer) {
                FAISS_ASSERT(packer->code_size > 0);
                FAISS_ASSERT(current->codes != nullptr);
                // printf("insert:packer\n");
                // 获取目标位置
                uint8_t* dst_codes = current->codes + (current->size + i) * single_code_size;
                packer->pack_1(cur_codes + i * single_code_size, 0, dst_codes);
            } else {
                // 如果没有packer，直接复制数据
                cudaMemcpy(current->codes + (current->size + i) * single_code_size,
                           cur_codes + i * single_code_size,
                           single_code_size,
                           cudaMemcpyHostToDevice);
            }
        }
        // 更新计数
        current->size += to_insert;
        remaining -= to_insert;
        cur_ids += to_insert;
        cur_codes += to_insert * single_code_size;
    }

    // 检查是否触发碎片整理
    size_t total_size = 0;
    total_size = list_size(list_no);
    if (total_size >= rearrange_threshold) {
        rearrange_if_needed(list_no);
    }
}

// 碎片整理
void BlockInvertedListsXHS::rearrange_if_needed(size_t list_no) {
    MemoryBlock* current = heads[list_no];
    // printf("rearrange_if_needed\n");
    while (current) {
        if (current->is_merged) {
            current = current->next;
            continue;
        }
        MemoryBlock* next_block = current->next;
        if (next_block && current->size + next_block->size <= n_per_block) {
            cudaMemcpyAsync(current->ids + current->size, next_block->ids,
                            next_block->size * sizeof(idx_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(current->codes + current->size * code_size, next_block->codes,
                            next_block->size * code_size, cudaMemcpyDeviceToDevice);
            cudaStreamSynchronize(0);
            current->size += next_block->size;
            current->next = next_block->next;
            next_block->is_merged = true;  // 确保标记更新
        }
        current = current->next;
    }
}

size_t BlockInvertedListsXHS::list_size(size_t list_no) const {
    FAISS_ASSERT(list_no < nlist);
    size_t total = 0;
    MemoryBlock* p = heads[list_no];
    while (p != nullptr) {
        total += p->size;
        p = p->next;
    }
    return total;
}

const uint8_t* BlockInvertedListsXHS::get_codes(size_t list_no) const {
    FAISS_ASSERT(list_no < nlist);
    // tips:返回首个内存块的codes地址（需外部处理链表访问）
    return heads[list_no] ? heads[list_no]->codes : nullptr;
}

const idx_t* BlockInvertedListsXHS::get_ids(size_t list_no) const {
    FAISS_ASSERT(list_no < nlist);
    // tips:返回首个内存块的ids地址（同上）
    return heads[list_no] ? heads[list_no]->ids : nullptr;
}

size_t BlockInvertedListsXHS::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    FAISS_ASSERT(list_no < nlist);
    // 若当前列表为空，直接插入
    if (heads[list_no] == nullptr) {
        insert_entries(list_no, n_entry, ids, code);
        return 0;
    }
    // 否则追加到尾部（需计算偏移量）
    size_t offset = list_size(list_no);
    insert_entries(list_no, n_entry, ids, code);
    return offset;
}

void BlockInvertedListsXHS::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    FAISS_THROW_MSG("BlockInvertedListsXHS::update_entries not implemented");
}

void BlockInvertedListsXHS::resize(size_t list_no, size_t new_size) {
    FAISS_ASSERT(list_no < nlist);
    size_t current_size = list_size(list_no);
    if (new_size == current_size) return;

    if (new_size < current_size) {
        // 缩减：从尾部删除多余块
        MemoryBlock* p = tails[list_no];
        while (p != nullptr && current_size > new_size) {
            size_t remove_num = std::min(p->size, current_size - new_size);
            p->size -= remove_num;
            current_size -= remove_num;
            if (p->size == 0) {
                // 删除空块
                if (p->prev) p->prev->next = nullptr;
                tails[list_no] = p->prev;
                if (p == heads[list_no]) heads[list_no] = nullptr;
                MemoryBlock* to_delete = p;
                p = p->prev;
                // 注意：内存池无需释放，由预分配管理
            } else {
                p = p->prev;
            }
        }
    } else {
        // 扩展：分配新块填充0（需按block_size对齐）
        size_t add_num = new_size - current_size;
        size_t needed_blocks = (add_num + n_per_block - 1) / n_per_block;
        std::vector<idx_t> dummy_ids(add_num, -1);
        std::vector<uint8_t> dummy_codes(add_num * code_size, 0);
        // 插入填充数据
        insert_entries(list_no, add_num, dummy_ids.data(), dummy_codes.data());
    }
}

BlockInvertedListsXHS::~BlockInvertedListsXHS() {
    for (size_t i = 0; i < nlist; i++) {
        // MemoryBlock* block = heads[i];
        // while (block) {
        //     MemoryBlock* next = block->next;
        //     delete block;
        //     block = next;
        // }
        heads[i] = nullptr;
        tails[i] = nullptr;
    }
}

BlockInvertedListsXHSIOHook::BlockInvertedListsXHSIOHook()
        : InvertedListsIOHook("ilxs", typeid(BlockInvertedListsXHS).name()) {}

void BlockInvertedListsXHSIOHook::write(const InvertedLists* ils_in, IOWriter* f) const {
    uint32_t h = fourcc("ilxs");
    WRITE1(h);
    const BlockInvertedListsXHS* il = dynamic_cast<const BlockInvertedListsXHS*>(ils_in);
    WRITE1(il->nlist);
    WRITE1(il->code_size);
    WRITE1(il->n_per_block);
    WRITE1(il->block_size);

    // 逐个倒排列表存储
    for (size_t i = 0; i < il->nlist; i++) {
        size_t list_size = il->list_size(i);
        WRITE1(list_size);

        if (list_size > 0) {
            // **拷贝 GPU 端数据到 CPU**
            std::vector<idx_t> ids_cpu(list_size);
            std::vector<uint8_t> codes_cpu(list_size * il->code_size);

            cudaMemcpy(ids_cpu.data(), il->get_ids(i), list_size * sizeof(idx_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(codes_cpu.data(), il->get_codes(i), list_size * il->code_size, cudaMemcpyDeviceToHost);

            // **写入文件**
            WRITEVECTOR(ids_cpu);
            WRITEVECTOR(codes_cpu);
        }
    }
}

InvertedLists* BlockInvertedListsXHSIOHook::read(IOReader* f, int /* io_flags */) const {
    BlockInvertedListsXHS* il = new BlockInvertedListsXHS();
    READ1(il->nlist);
    READ1(il->code_size);
    READ1(il->n_per_block);
    READ1(il->block_size);

    il->heads.resize(il->nlist, nullptr);
    il->tails.resize(il->nlist, nullptr);

    for (size_t i = 0; i < il->nlist; i++) {
        size_t list_size;
        READ1(list_size);

        if (list_size > 0) {
            // **从文件读取数据**
            std::vector<idx_t> ids_cpu(list_size);
            std::vector<uint8_t> codes_cpu(list_size * il->code_size);
            READVECTOR(ids_cpu);
            READVECTOR(codes_cpu);

            // **分配 GPU 内存并拷贝数据**
            MemoryBlock* block = il->allocate_block();
            cudaMemcpy(block->ids, ids_cpu.data(), list_size * sizeof(idx_t), cudaMemcpyHostToDevice);
            cudaMemcpy(block->codes, codes_cpu.data(), list_size * il->code_size, cudaMemcpyHostToDevice);
            block->size = list_size;

            il->heads[i] = block;
            il->tails[i] = block;
        }
    }

    return il;
}
} // namespace faiss
