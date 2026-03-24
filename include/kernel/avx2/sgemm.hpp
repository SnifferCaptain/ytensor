#pragma once
/***************
 * @file sgemm.hpp
 * @brief 高性能SGEMM矩阵乘法声明（AVX2）
 ***************/

#include "gemm_utils.hpp"
#include "sdot.hpp"
#include "sger.hpp"
#include "sgemv.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

void pack_a_col(const float* A, float* packed, int mc, int kc, int lda);
void pack_b_col(const float* B, float* packed, int kc, int nc, int ldb);
void pack_a_row(const float* A, float* packed, int mc, int kc, int lda);
void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa);
void pack_b_row(const float* B, float* packed, int kc, int nc, int ldb);
void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb);

void pack_blockA_row_single_tile(const float* A, float* dest, int i, int mr, int kc, int lda);
void pack_blockB_row_par(const float* B, float* packed, int nc, int kc, int ldb);
void pack_blockA_row_par(const float* A, float* packed, int mc, int kc, int lda);

void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k);
void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k);

void sgemm_masked(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc,
    const bool* mask
);

template <typename Func>
void sgemm_masked(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc,
    Func&& func
);

void sgemm(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc
);

void sgemm(const float* A, const float* B, float* C, int m, int n, int k,
           float alpha = 1.0f, float beta = 0.0f);
void matmul(const float* A, const float* B, float* C, int m, int n, int k,
            int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc);
void matmul(const float* A, const float* B, float* C, int m, int n, int k);
void matmul(const float* A, const float* B, float* C, int m, int n, int k,
            float alpha, float beta);
void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, int nthreads = 0);
void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/sgemm.inl"

#endif // __AVX2__ && __FMA__
