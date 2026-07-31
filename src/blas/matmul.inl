#pragma once
/***************
 * file: blas/matmul.inl
 * purpose: convenience matrix multiplication operators.
 ***************/

namespace yt::blas {

inline void matmul(
    const float* A, const float* B, float* C, int m, int n, int k, int64_t rsa, int64_t csa, int64_t rsb,
    int64_t csb, int64_t rsc, int64_t csc
) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta) {
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void hmatmul(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
) {
    hgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k) {
    hgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

inline void matmul_parallel(
    const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads
) {
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = g_num_threads;
    const int old = g_num_threads;
    g_num_threads = nthreads;
    sgemm_rowmajor(A, B, C, m, n, k);
    g_num_threads = old;
#else
    sgemm_rowmajor(A, B, C, m, n, k);
#endif
}

inline void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(A, B, C, m, n, k);
}

}  // namespace yt::blas
