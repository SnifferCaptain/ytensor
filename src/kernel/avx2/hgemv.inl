#pragma once

#include <memory>

namespace yt::kernel::avx2 {

inline void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y,
    int m, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsx,
    int64_t rsy
) {
    if (m <= 0 || k <= 0) return;

    std::unique_ptr<float[]> fA(new float[(size_t)m * k]);
    std::unique_ptr<float[]> fx(new float[k]);
    std::unique_ptr<float[]> fy(new float[m]);

    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            fA[i * k + p] = static_cast<float>(A[i * rsa + p * csa]);

    for (int p = 0; p < k; ++p)
        fx[p] = static_cast<float>(x[p * rsx]);

    gemv_col_simd(fA.get(), fx.get(), fy.get(), m, k, alpha, 0.0f, k, 1, 1, 1);

    for (int i = 0; i < m; ++i) {
        float r = fy[i] + (beta != 0.0f ? beta * static_cast<float>(y[i * rsy]) : 0.0f);
        y[i * rsy] = yt::float16(r);
    }
}

inline void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* z,
    int n, int k,
    float alpha, float beta,
    int64_t csx,
    int64_t rsb, int64_t csb,
    int64_t csz
) {
    if (n <= 0 || k <= 0) return;

    std::unique_ptr<float[]> fx(new float[k]);
    std::unique_ptr<float[]> fB(new float[(size_t)k * n]);
    std::unique_ptr<float[]> fz(new float[n]);

    for (int p = 0; p < k; ++p)
        fx[p] = static_cast<float>(x[p * csx]);

    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j)
            fB[p * n + j] = static_cast<float>(B[p * rsb + j * csb]);

    gemv_row_simd(fx.get(), fB.get(), fz.get(), n, k, alpha, 0.0f, 1, n, 1, 1);

    for (int j = 0; j < n; ++j) {
        float r = fz[j] + (beta != 0.0f ? beta * static_cast<float>(z[j * csz]) : 0.0f);
        z[j * csz] = yt::float16(r);
    }
}

inline void hgemv(const yt::float16* A, const yt::float16* x, yt::float16* y,
                  int m, int k, float alpha, float beta) {
    hgemv_col(A, x, y, m, k, alpha, beta, k, 1, 1, 1);
}

} // namespace yt::kernel::avx2
