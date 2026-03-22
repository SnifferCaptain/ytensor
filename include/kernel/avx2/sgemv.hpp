#pragma once
/***************
 * @file sgemv.hpp
 * @brief 单精度矩阵-向量乘法（SGEMV）AVX2内核声明
 ***************/

#include "gemm_utils.hpp"
#include "sdot.hpp"
#include "sger.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

template<int NCOLS>
void gemv_dot_ncols(const float* __restrict A, const float* const* __restrict Bcols,
                    int k, float* sums);

float gemv_dot_1col(const float* __restrict A, const float* __restrict B, int k);

template<int UNROLL_N = 4>
void gemv_row_colmajor_kernel(const float* __restrict A, const float* __restrict B,
                              float* __restrict C, int n, int k, float alpha, float beta,
                              int64_t csa, int64_t csb);

void gemv_row_rowmajor_kernel(const float* __restrict A, const float* __restrict B,
                              float* __restrict C, int n, int k, float alpha, float beta);

template<int UNROLL_N = 4>
void gemv_row_simd(const float* __restrict A, const float* __restrict B, float* __restrict C,
                   int n, int k, float alpha, float beta,
                   int64_t csa, int64_t rsb, int64_t csb, int64_t csc);

void gemv_col_simd(const float* A, const float* B, float* C,
                   int m, int k, float alpha, float beta,
                   int64_t rsa, int64_t csa, int64_t rsb, int64_t rsc);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/sgemv.inl"

#endif // __AVX2__ && __FMA__
