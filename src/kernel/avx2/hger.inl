#pragma once

#include <memory>

namespace yt::kernel::avx2 {

inline void hger(int m, int n,
                 float alpha, float beta,
                 const yt::float16* x, int64_t incx,
                 const yt::float16* y, int64_t incy,
                 yt::float16* C, int64_t rsc, int64_t csc) {
    if (m <= 0 || n <= 0) return;

    std::unique_ptr<float[]> fx(new float[m]);
    std::unique_ptr<float[]> fy(new float[n]);
    for (int i = 0; i < m; ++i) fx[i] = static_cast<float>(x[i * incx]);
    for (int j = 0; j < n; ++j) fy[j] = static_cast<float>(y[j * incy]);

    if (csc == 1 && rsc >= n) {
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * fx[i];
            yt::float16* row = C + i * rsc;
            int j = 0;
#if __F16C__
            __m256 va = _mm256_set1_ps(a_val);
            for (; j + 8 <= n; j += 8) {
                __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
                __m256 vc = _mm256_cvtph_ps(h);
                __m256 vy = _mm256_loadu_ps(fy.get() + j);
                if (beta == 0.0f)
                    vc = _mm256_mul_ps(va, vy);
                else
                    vc = _mm256_fmadd_ps(va, vy, _mm256_mul_ps(_mm256_set1_ps(beta), vc));
                __m128i hr = _mm256_cvtps_ph(vc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row + j), hr);
            }
#endif
            for (; j < n; ++j)
                row[j] = yt::float16(a_val * fy[j] + beta * static_cast<float>(row[j]));
        }
        return;
    }

    for (int i = 0; i < m; ++i) {
        float a_val = alpha * fx[i];
        for (int j = 0; j < n; ++j) {
            yt::float16* cp = C + i * rsc + j * csc;
            *cp = yt::float16(a_val * fy[j] + beta * static_cast<float>(*cp));
        }
    }
}

inline void hger(int m, int n, float alpha,
                 const yt::float16* x, const yt::float16* y,
                 yt::float16* A, int lda) {
    hger(m, n, alpha, 1.0f, x, 1, y, 1, A, lda, 1);
}

} // namespace yt::kernel::avx2
