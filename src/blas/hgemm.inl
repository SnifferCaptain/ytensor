#pragma once
/***************
 * file: blas/hgemm.inl
 * purpose: public HGEMM compatibility wrappers over typed GEMM.
 ***************/

namespace yt::blas {

inline void hgemm(
    const BlasContext& context, const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n,
    int k, float alpha, float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc
) {
    gemm(context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc);
}

inline void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, float alpha, float beta,
    int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
) {
    hgemm(defaultBlasContext(), A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc);
}

inline void hgemm(
    const BlasContext& context, const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n,
    int k, float alpha, float beta
) {
    hgemm(context, A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, float alpha, float beta
) {
    hgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

}  // namespace yt::blas
