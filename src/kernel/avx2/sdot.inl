#pragma once

namespace yt::kernel::avx2 {

inline float sdot_stride(const float* x, const float* y, int n,
                         int64_t incx, int64_t incy) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
    return sum;
}

__attribute__((hot))
inline float sdot_contiguous(const float* __restrict x, const float* __restrict y, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_loadu_ps(y + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i +  8), _mm256_loadu_ps(y + i +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);

    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = hsum_ps(acc0);

    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

inline float sdot(const float* x, const float* y, int n,
                  int64_t incx, int64_t incy) {
    if (n <= 0) return 0.0f;
    if (incx == 1 && incy == 1)
        return sdot_contiguous(x, y, n);
    return sdot_stride(x, y, n, incx, incy);
}

} // namespace yt::kernel::avx2
