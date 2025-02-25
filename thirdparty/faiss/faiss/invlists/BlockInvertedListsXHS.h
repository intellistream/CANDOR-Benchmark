/*
 * Copyright (C) 2024 by the INTELLI team
 * Created on: 25-2-19 下午3:14
 * Description: ${DESCRIPTION}
 */


#ifndef BLOCKINVERTEDLISTSXHS_H
#define BLOCKINVERTEDLISTSXHS_H

#pragma once

#include <atomic>
#include <vector>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/InvertedListsIOHook.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/impl/CodePacker.h>

namespace faiss {

class CodePackerXHS : public CodePacker {
public:
    size_t M;               // PQ 子向量数
    size_t nbits_per_idx;   // 每个索引的 bit 数
    // size_t code_size;       // 每个向量的存储字节数

    explicit CodePackerXHS(size_t M, size_t nbits_per_idx)
        : M(M), nbits_per_idx(nbits_per_idx) {

        code_size = (M * nbits_per_idx + 7) / 8;
    }

    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block) const final override {
        assert(offset == 0);
        std::memcpy(block, flat_code, code_size);
    }

    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code) const final override {
        assert(offset == 0);
        std::memcpy(flat_code, block, code_size);
    }
};

// GPU 端内存块结构
struct MemoryBlock {
    MemoryBlock* prev;       // 前驱块指针
    MemoryBlock* next;       // 后继块指针
    size_t capacity;         // 块最大存储向量数
    size_t size;             // 当前存储向量数
    idx_t* ids;              // 向量ID数组（GPU内存）
    uint8_t* codes;          // 向量编码数据（GPU内存，按32维交错对齐）
    bool is_merged;          // 标记是否已合并（用于碎片整理）
};

struct BlockInvertedListsXHS : InvertedLists {
    size_t n_per_block;      // 每块存储的向量数
    size_t block_size;       // 每块存储的字节数
    const CodePacker* packer = nullptr;

    // GPU 端内存池管理
    AlignedTable<uint8_t> memory_pool;      // 预分配的连续GPU内存池
    std::atomic<size_t> cur_pool_ptr;       // 当前内存分配位置（原子操作）

    // 每个倒排链表的头部和尾部
    std::vector<MemoryBlock*> heads;
    std::vector<MemoryBlock*> tails;

    // 触发碎片整理的阈值
    size_t rearrange_threshold;

    BlockInvertedListsXHS() : InvertedLists(0, InvertedLists::INVALID_CODE_SIZE),
          n_per_block(0),
          block_size(0){}
    BlockInvertedListsXHS(size_t nlist, size_t vec_per_block, size_t block_size);
    BlockInvertedListsXHS(size_t nlist, const CodePacker* packer);
    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    // works only on empty BlockInvertedLists
    // the codes should be of size ceil(n_entry / n_per_block) * block_size
    // and padded with 0s
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    /// not implemented
    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    // also pads new data with 0s
    void resize(size_t list_no, size_t new_size) override;

    MemoryBlock* allocate_block(); // 分配新的 GPU 内存块
    const uint8_t* get_single_code(size_t list_no, size_t offset) const override;
    void insert_entries(size_t list_no, size_t n_entry, const idx_t* ids_in, const uint8_t* codes);
    void rearrange_if_needed(size_t list_no); // 碎片整理

    ~BlockInvertedListsXHS();

    struct ScopedCodes {
        const BlockInvertedListsXHS* il;  // 指向 BlockInvertedLists
        const uint8_t* codes;          // 指向 codes 数据
        size_t list_no;                // 倒排列表编号

        // 构造函数：获取整个 list_no 的 codes
        ScopedCodes(const BlockInvertedListsXHS* il, size_t list_no)
            : il(il), codes(il->get_codes(list_no)), list_no(list_no) {}

        // 构造函数：获取 list_no 的某个 offset 的 code
        ScopedCodes(const BlockInvertedListsXHS* il, size_t list_no, size_t offset)
            : il(il), codes(il->get_single_code(list_no, offset)), list_no(list_no) {}

        // 获取 codes 指针
        const uint8_t* get() {
            return codes;
        }

        ~ScopedCodes() {
            il->release_codes(list_no, codes);
        }
    };

    struct ScopedIds {
        const BlockInvertedListsXHS* il;  // 指向 BlockInvertedLists
        const idx_t* ids;              // 指向 ID 数据
        size_t list_no;                // 倒排列表编号

        // 构造函数：获取 list_no 的 ID 列表
        ScopedIds(const BlockInvertedListsXHS* il, size_t list_no)
            : il(il), ids(il->get_ids(list_no)), list_no(list_no) {}

        // 获取 ID 指针
        const idx_t* get() {
            return ids;
        }

        // 重载 `operator[]`，支持按索引访问 ID
        idx_t operator[](size_t i) const {
            MemoryBlock* block = il->heads[list_no];
            size_t offset = i;
            while (block && offset >= block->size) {
                offset -= block->size;
                block = block->next;
            }
            return block->ids[offset];
        }

        ~ScopedIds() {
            il->release_ids(list_no, ids);
        }
    };

};

struct BlockInvertedListsXHSIOHook : InvertedListsIOHook {
    BlockInvertedListsXHSIOHook();
    void write(const InvertedLists* ils, IOWriter* f) const override;
    InvertedLists* read(IOReader* f, int io_flags) const override;
};

} // namespace faiss


#endif //BLOCKINVERTEDLISTSXHS_H
