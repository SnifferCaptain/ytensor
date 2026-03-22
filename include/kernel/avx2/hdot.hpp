#pragma once
/***************
 * @file hdot.hpp
 * @brief 半精度向量点积（HDOT）声明
 ***************/

#include "hgemm.hpp"
#include "sdot.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

float hdot(const yt::float16* x, const yt::float16* y, int n,
           int64_t incx = 1, int64_t incy = 1);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/hdot.inl"

#endif // __AVX2__ && __FMA__
