#pragma once
/***************
 * @file sdot.hpp
 * @brief 单精度向量点积（SDOT）AVX2内核声明
 ***************/

#include "gemm_utils.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

float sdot_stride(const float* x, const float* y, int n,
                  int64_t incx, int64_t incy);

float sdot_contiguous(const float* __restrict x, const float* __restrict y, int n);

float sdot(const float* x, const float* y, int n,
           int64_t incx = 1, int64_t incy = 1);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/sdot.inl"

#endif // __AVX2__ && __FMA__
