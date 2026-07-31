#pragma once
/***************
 * file: blas/kernels/avx2/dotxv.inl
 * purpose: AVX2向量点积物理内核实现。
 ***************/

namespace yt::blas {

inline float avx2HorizontalSum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
inline float dotxv(
    int n, float alpha, const Storage* x, int64_t incx, const Storage* y, int64_t incy, float beta, float rho
) {
    float sum = 0.0f;
    if (n > 0 && alpha != 0.0f) {
        int i = 0;
        if constexpr (Avx2FloatStorage<Storage>::available) {
            if (incx == 1 && incy == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                for (; i + 32 <= n; i += 32) {
                    acc0 = _mm256_fmadd_ps(
                        Avx2FloatStorage<Storage>::load(x + i),
                        Avx2FloatStorage<Storage>::load(y + i),
                        acc0
                    );
                    acc1 = _mm256_fmadd_ps(
                        Avx2FloatStorage<Storage>::load(x + i + 8),
                        Avx2FloatStorage<Storage>::load(y + i + 8),
                        acc1
                    );
                    acc2 = _mm256_fmadd_ps(
                        Avx2FloatStorage<Storage>::load(x + i + 16),
                        Avx2FloatStorage<Storage>::load(y + i + 16),
                        acc2
                    );
                    acc3 = _mm256_fmadd_ps(
                        Avx2FloatStorage<Storage>::load(x + i + 24),
                        Avx2FloatStorage<Storage>::load(y + i + 24),
                        acc3
                    );
                }
                for (; i + 8 <= n; i += 8) {
                    acc0 = _mm256_fmadd_ps(
                        Avx2FloatStorage<Storage>::load(x + i),
                        Avx2FloatStorage<Storage>::load(y + i),
                        acc0
                    );
                }
                sum = avx2HorizontalSum(
                    _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3))
                );
            }
        }
        for (; i < n; ++i) {
            sum += static_cast<float>(x[i * incx]) * static_cast<float>(y[i * incy]);
        }
    }

    // beta为0时不读取rho，允许调用方传入未初始化的输出存储。
    return beta == 0.0f ? alpha * sum : beta * rho + alpha * sum;
}

}  // namespace yt::blas
