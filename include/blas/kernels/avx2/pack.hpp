#pragma once
/***************
 * @file pack.hpp
 * @brief AVX2 BLAS Level-1m物理内核
 ***************/

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

#include <cstdint>

#include "../../../type/float_spec.hpp"
#include "gemm_utils.hpp"

namespace yt::blas {

/// @brief 按Spec将任意storage type的A转换并打包为ComputeType panel
template <typename SourceType, typename ComputeType, GemmKernelSpec Spec>
void pack_a_typed_panel(const SourceType* A, ComputeType* packed, int mc, int kc, int64_t rsa, int64_t csa);

/// @brief 按Spec将任意storage type的B转换并打包为ComputeType panel
template <typename SourceType, typename ComputeType, GemmKernelSpec Spec>
void pack_b_typed_panel(const SourceType* B, ComputeType* packed, int kc, int nc, int64_t rsb, int64_t csb);

/// @brief 将任意步幅A打包为[kc][MR_ROW] panel
/// @note caller负责提供足够大且不与输入重叠的目标存储。
void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa);

/// @brief 将任意步幅B打包为[kc][NR_ROW] panel
void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb);

}  // namespace yt::blas

#include "../../../../src/blas/kernels/avx2/pack_a.inl"
#include "../../../../src/blas/kernels/avx2/pack_b.inl"

#endif
