#pragma once
/***************
 * @file level3.hpp
 * @brief AVX2 BLAS Level-3物理微内核
 * @author SnifferCaptain
 * @date 2026-03-10
 ***************/

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>  // AVX2 / FMA intrinsics

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>  // std::integer_sequence / std::make_integer_sequence

#include "gemm_utils.hpp"
#include "pack.hpp"

namespace yt::blas {

// ==================== FMA核心 ====================

/// @brief 在编译期展开row FMA的MR行
template <int MR, int NB, int... Is>
__attribute__((always_inline)) void
fma_row_step(const float* __restrict A, const __m256 b[], __m256 c[][NB], std::integer_sequence<int, Is...>);

/// @brief 计算完整row microkernel tile
/// @details A/B必须采用Spec对应的packed panel，ldc为C的元素行步幅。
/// @note Accumulate=false覆盖C，Accumulate=true读取并累加已有C。
template <typename ComputeType, GemmKernelSpec Spec, bool Accumulate>
__attribute__((always_inline, hot)) void row_microkernel(
    const ComputeType* __restrict A, const ComputeType* __restrict B, ComputeType* __restrict C, int kc,
    int ldc
);

/// @brief 将typed Level-3 frame分发到AVX2 FP32物理微内核
/// @details storage type转换由typed packer完成，packed A/B与microkernel均由Spec定义。
template <typename ComputeType, GemmKernelSpec Spec>
struct Avx2GemmMicrokernelDispatcher {
    static_assert(std::is_same_v<ComputeType, float>, "AVX2 GEMM dispatcher currently supports FP32 compute");
    static constexpr GemmKernelSpec spec = Spec;

    template <typename SourceType>
    static void packA(
        const SourceType* source, ComputeType* packed, int mc, int kc, int64_t row_stride,
        int64_t column_stride
    ) {
        pack_a_typed_panel<SourceType, ComputeType, Spec>(source, packed, mc, kc, row_stride, column_stride);
    }

    template <typename SourceType>
    static void packB(
        const SourceType* source, ComputeType* packed, int kc, int nc, int64_t row_stride,
        int64_t column_stride
    ) {
        pack_b_typed_panel<SourceType, ComputeType, Spec>(source, packed, kc, nc, row_stride, column_stride);
    }

    template <bool Accumulate>
    static void compute(
        const ComputeType* packed_a, const ComputeType* packed_b, ComputeType* output, int kc, int ldc
    ) {
        row_microkernel<ComputeType, Spec, Accumulate>(packed_a, packed_b, output, kc, ldc);
    }
};

/// @brief 消费A[kc][MR_ROW]和B[kc][NR_ROW] packed panel
template <int MR>
__attribute__((hot)) void fma_loop_row(
    const float* __restrict A, const float* __restrict B, __m256 c[MR_ROW][NR_BLOCKS], int kc
);

// ==================== 边界分发 ====================

template <int MAX_MR>
void dispatch_fma_row(const float* A, const float* B, __m256 c[MR_ROW][NR_BLOCKS], int mr, int kc);

}  // namespace yt::blas

#include "../../../../src/blas/kernels/avx2/gemm.inl"

#endif  // __AVX2__ && __FMA__
