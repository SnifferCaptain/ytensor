#pragma once
/***************
 * file: blas/sgemm.inl
 * purpose: public SGEMM compatibility wrappers over typed GEMM.
 ***************/

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace yt::blas {

inline void sgemm_colmajor(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k
) {
    gemm(context, A, B, C, m, n, k, 1.0f, 0.0f, 1, m, 1, k, 1, m);
}

inline void sgemm_rowmajor(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k
) {
    gemm(context, A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

inline void sgemm_masked(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, const bool* mask
) {
    if (mask == nullptr) throw std::invalid_argument("sgemm_masked: mask must not be null");
    gemm_masked(
        context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc,
        GemmDenseBoolMaskPolicy{mask, n}
    );
}

template <typename Func>
requires(!std::is_pointer_v<std::decay_t<Func>>) inline void sgemm_masked(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, Func&& func
) {
    static_assert(
        std::is_invocable_r_v<bool, const std::decay_t<Func>&, int, int>,
        "sgemm_masked func must be callable as bool(int, int)"
    );
    gemm_masked(
        context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc, std::forward<Func>(func)
    );
}

inline void sgemm(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
) {
    gemm(context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc);
}

inline void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(defaultBlasContext(), A, B, C, m, n, k);
}

inline void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_rowmajor(defaultBlasContext(), A, B, C, m, n, k);
}

inline void sgemm_masked(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, const bool* mask
) {
    sgemm_masked(defaultBlasContext(), A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc, mask);
}

template <typename Func>
requires(!std::is_pointer_v<std::decay_t<Func>>) inline void sgemm_masked(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, Func&& func
) {
    sgemm_masked(
        defaultBlasContext(), A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc,
        std::forward<Func>(func)
    );
}

inline void sgemm(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
) {
    sgemm(defaultBlasContext(), A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc);
}

inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta) {
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void sgemm(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta
) {
    sgemm(context, A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

}  // namespace yt::blas
