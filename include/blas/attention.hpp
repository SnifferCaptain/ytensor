#pragma once
/***************
 * @file attention.hpp
 * @brief YBLAS attention 扩展算子
 ***************/

#include "level3.hpp"

#if YT_USE_YBLAS

#include <type_traits>

namespace yt::blas {

/// @brief 计算单个attention head的scaled dot-product attention
/// @details q_len大于1时使用分块QK、在线softmax与V融合累加；q_len等于1时使用decode GEMV路径。
/// @note mask中true表示可见。全false score tile会跳过QK微内核、softmax与V累加。
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

/// @brief 使用坐标mask policy计算单个attention head
/// @details func至少支持bool(int query_index, int key_index)，可选tileAllTrue/tileAllFalse用于整块跳过。
/// @note 谓词必须可重复调用且无副作用；并行prefill要求并发调用安全；tile接口必须与逐元素结果一致。
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

}  // namespace yt::blas

#include "../../src/blas/attention.inl"

#endif  // YT_USE_YBLAS
