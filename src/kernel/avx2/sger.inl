#pragma once

namespace yt::kernel::avx2 {

__attribute__((hot))
inline void sger(int m, int n,
                 float alpha, float beta,
                 const float* __restrict x, int64_t incx,
                 const float* __restrict y, int64_t incy,
                 float* __restrict C, int64_t rsc, int64_t csc) {
    if (m <= 0 || n <= 0) return;

    if (csc == 1 && rsc >= n && incy == 1) {
        __m256 vbeta = _mm256_set1_ps(beta);
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            __m256 va0 = _mm256_set1_ps(alpha * x[i * incx]);
            __m256 va1 = _mm256_set1_ps(alpha * x[(i + 1) * incx]);
            __m256 va2 = _mm256_set1_ps(alpha * x[(i + 2) * incx]);
            __m256 va3 = _mm256_set1_ps(alpha * x[(i + 3) * incx]);
            float* c0 = C + i * rsc;
            float* c1 = C + (i + 1) * rsc;
            float* c2 = C + (i + 2) * rsc;
            float* c3 = C + (i + 3) * rsc;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vy = _mm256_loadu_ps(y + j);
                if (beta == 0.0f) {
                    _mm256_storeu_ps(c0 + j, _mm256_mul_ps(va0, vy));
                    _mm256_storeu_ps(c1 + j, _mm256_mul_ps(va1, vy));
                    _mm256_storeu_ps(c2 + j, _mm256_mul_ps(va2, vy));
                    _mm256_storeu_ps(c3 + j, _mm256_mul_ps(va3, vy));
                } else {
                    _mm256_storeu_ps(c0 + j, _mm256_fmadd_ps(va0, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c0 + j))));
                    _mm256_storeu_ps(c1 + j, _mm256_fmadd_ps(va1, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c1 + j))));
                    _mm256_storeu_ps(c2 + j, _mm256_fmadd_ps(va2, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c2 + j))));
                    _mm256_storeu_ps(c3 + j, _mm256_fmadd_ps(va3, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c3 + j))));
                }
            }
            float a0 = alpha * x[i * incx];
            float a1 = alpha * x[(i + 1) * incx];
            float a2 = alpha * x[(i + 2) * incx];
            float a3 = alpha * x[(i + 3) * incx];
            for (; j < n; ++j) {
                float yj = y[j];
                c0[j] = a0 * yj + beta * c0[j];
                c1[j] = a1 * yj + beta * c1[j];
                c2[j] = a2 * yj + beta * c2[j];
                c3[j] = a3 * yj + beta * c3[j];
            }
        }
        for (; i < m; ++i) {
            float a_val = alpha * x[i * incx];
            float* c_row = C + i * rsc;
            __m256 va = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vy = _mm256_loadu_ps(y + j);
                if (beta == 0.0f)
                    _mm256_storeu_ps(c_row + j, _mm256_mul_ps(va, vy));
                else
                    _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(va, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c_row + j))));
            }
            for (; j < n; ++j) c_row[j] = a_val * y[j] + beta * c_row[j];
        }
        return;
    }

    for (int i = 0; i < m; ++i) {
        float a_val = alpha * x[i * incx];
        for (int j = 0; j < n; ++j) {
            float* c_ptr = C + i * rsc + j * csc;
            *c_ptr = a_val * y[j * incy] + beta * (*c_ptr);
        }
    }
}

inline void sger(int m, int n, float alpha,
                 const float* __restrict x, const float* __restrict y,
                 float* __restrict A, int lda) {
    sger(m, n, alpha, 1.0f, x, 1, y, 1, A, lda, 1);
}

} // namespace yt::kernel::avx2
