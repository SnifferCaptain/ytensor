#pragma once
/***************
 * @file hgemv.hpp
 * @brief 半精度矩阵-向量乘法（HGEMV）声明
 ***************/

#include "hgemm.hpp"
#include "sgemv.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y,
    int m, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsx,
    int64_t rsy
);

void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* z,
    int n, int k,
    float alpha, float beta,
    int64_t csx,
    int64_t rsb, int64_t csb,
    int64_t csz
);

void hgemv(const yt::float16* A, const yt::float16* x, yt::float16* y,
           int m, int k, float alpha = 1.0f, float beta = 0.0f);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/hgemv.inl"

#endif // __AVX2__ && __FMA__
