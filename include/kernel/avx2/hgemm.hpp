#pragma once
/***************
 * @file hgemm.hpp
 * @brief 半精度矩阵乘法（HGEMM）声明
 ***************/

#include "gemm_utils.hpp"
#include "sgemm.hpp"
#include "../../types/float_spec.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

void f16_to_f32_block(const yt::float16* src, float* dst, int count);
void f32_to_f16_block(const float* src, yt::float16* dst, int count);

void pack_a_row_f16(const yt::float16* A, float* packed, int mc, int kc, int lda);
void pack_b_row_f16(const yt::float16* B, float* packed, int kc, int nc, int ldb);
void pack_a_generic_f16(const yt::float16* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa);
void pack_b_generic_f16(const yt::float16* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb);

void store_c_f16(const float* c_buf, yt::float16* C, int mr, int nr, int ldc);
void store_c_generic_f16(const float* c_buf, yt::float16* C, int mr, int nr, int64_t rsc, int64_t csc);

void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc
);

void hgemm(const yt::float16* A, const yt::float16* B, yt::float16* C,
           int m, int n, int k, float alpha = 1.0f, float beta = 0.0f);

void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C,
             int m, int n, int k,
             int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc);

void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/hgemm.inl"

#endif // __AVX2__ && __FMA__
