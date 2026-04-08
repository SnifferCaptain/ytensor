#pragma once
/***************
 * @file flash_attention.hpp
 * @brief AVX2 Flash Attention 内核声明
 *
 * - 使用 blockwise online softmax，避免显式物化完整注意力矩阵
 * - 分数块计算复用 GEMM 的 6x16 行主序微内核
 * - 支持布尔掩码和 predicate 掩码
 * - 支持 attention bias
 ***************/

#include "sgemm.hpp"
#include "sgemv.hpp"

#if defined(__AVX2__) && defined(__FMA__)

#include <type_traits>

namespace yt::kernel::avx2 {

void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int q_len,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t rsq,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t rso,
    int64_t cso,
    const bool* mask = nullptr,
    int64_t mask_stride = 0,
    const float* bias = nullptr,
    int64_t rsbias = 0,
    int64_t csbias = 0
);

template <typename Func>
void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int q_len,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t rsq,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t rso,
    int64_t cso,
    Func&& mask,
    const float* bias = nullptr,
    int64_t rsbias = 0,
    int64_t csbias = 0
);

}  // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/flash_attention.inl"

#endif  // __AVX2__ && __FMA__
