#pragma once
/***************
 * @file sger.hpp
 * @brief 单精度向量外积更新（SGER）AVX2内核声明
 ***************/

#include "gemm_utils.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::blas {

void sger(int m, int n,
          float alpha, float beta,
          const float* __restrict x, int64_t incx,
          const float* __restrict y, int64_t incy,
          float* __restrict C, int64_t rsc, int64_t csc);

void sger(int m, int n, float alpha,
          const float* __restrict x, const float* __restrict y,
          float* __restrict A, int lda);

} // namespace yt::blas

#include "../../../src/blas/avx2/sger.inl"

#endif // __AVX2__ && __FMA__
