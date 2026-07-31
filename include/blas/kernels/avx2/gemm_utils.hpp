#pragma once
/***************
 * @file gemm_utils.hpp
 * @brief GEMM shape、context tuning与共享workspace
 ***************/

#if defined(__AVX2__) && defined(__FMA__)

#include <algorithm>
#include <cstddef>
#include <cstdlib>

#include "../../context.hpp"

namespace yt::blas {

/// @brief GEMM物理微内核尺寸
/// @details 该结构同时定义packing与row microkernel ABI。
struct GemmKernelSpec {
    int row_mr;
    int row_nr;
    int vector_lanes;
};

inline constexpr GemmKernelSpec default_gemm_kernel_spec = {
    .row_mr = 6,
    .row_nr = 16,
    .vector_lanes = 8,
};

constexpr int MR_ROW = default_gemm_kernel_spec.row_mr;
constexpr int NR_ROW = default_gemm_kernel_spec.row_nr;
constexpr int NR_BLOCKS = NR_ROW / default_gemm_kernel_spec.vector_lanes;

/// @brief 向下对齐整数并保证结果不小于align
int align_down_to(int value, int align);

/// @brief 设置GEMM分块尺寸
/// @note setter会执行最小值限制和微内核尺寸对齐。
void set_gemm_block_sizes(BlasContext& context, int mc, int kc, int nc);

void set_gemm_block_sizes(int mc, int kc, int nc);

void set_gemm_cache_sizes_bytes(BlasContext& context, size_t l1_bytes, size_t l2_bytes, size_t l3_bytes);

void set_gemm_cache_sizes_bytes(size_t l1_bytes, size_t l2_bytes, size_t l3_bytes);

void set_gemm_cache_sizes_kb(BlasContext& context, int l1_kb, int l2_kb, int l3_kb);

void set_gemm_cache_sizes_kb(int l1_kb, int l2_kb, int l3_kb);

int gemm_mc(const BlasContext& context);

int gemm_kc(const BlasContext& context);

int gemm_nc(const BlasContext& context);

int gemm_mc();

int gemm_kc();

int gemm_nc();

void* aligned_alloc_64(size_t size);

void aligned_free_64(void* ptr);

/// @brief 64字节对齐的FP32工作区
/// @details ensure仅增长容量；分配失败时保留原工作区且不抛异常。
struct AlignedBuffer {
    float* data = nullptr;
    size_t capacity = 0;

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    void ensure(size_t n);
    ~AlignedBuffer() {
        if (data) aligned_free_64(data);
    }
};

}  // namespace yt::blas

#include "../../../../src/blas/kernels/avx2/gemm_blocking.inl"
#include "../../../../src/blas/kernels/avx2/aligned_buffer.inl"
#endif
