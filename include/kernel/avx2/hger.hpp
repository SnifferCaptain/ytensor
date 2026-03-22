#pragma once
/***************
 * @file hger.hpp
 * @brief 半精度向量外积更新（HGER）声明
 ***************/

#include "hgemm.hpp"
#include "sger.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

void hger(int m, int n,
          float alpha, float beta,
          const yt::float16* x, int64_t incx,
          const yt::float16* y, int64_t incy,
          yt::float16* C, int64_t rsc, int64_t csc);

void hger(int m, int n, float alpha,
          const yt::float16* x, const yt::float16* y,
          yt::float16* A, int lda);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/hger.inl"

#endif // __AVX2__ && __FMA__
