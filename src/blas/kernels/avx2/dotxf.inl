#pragma once
/***************
 * file: blas/kernels/avx2/dotxf.inl
 * purpose: AVX2融合多列点积物理内核实现。
 ***************/

namespace yt::blas {

template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
inline void dotxf(
    const BlasContext& context, int n, int f, float alpha, const Storage* x, int64_t incx, const Storage* a,
    int64_t rsa, int64_t csa, float beta, float* y, int64_t incy
) {
    if (f <= 0) return;
    if (n <= 0 || alpha == 0.0f) {
        scalv(f, beta, y, incy);
        return;
    }

    if constexpr (Avx2FloatStorage<Storage>::available) {
        if (context.df == 4 && f == 4 && incx == 1 && rsa == 1) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int i = 0;
            const Storage* a0 = a;
            const Storage* a1 = a + csa;
            const Storage* a2 = a + 2 * csa;
            const Storage* a3 = a + 3 * csa;
            for (; i + 8 <= n; i += 8) {
                __m256 vx = Avx2FloatStorage<Storage>::load(x + i);
                acc0 = _mm256_fmadd_ps(vx, Avx2FloatStorage<Storage>::load(a0 + i), acc0);
                acc1 = _mm256_fmadd_ps(vx, Avx2FloatStorage<Storage>::load(a1 + i), acc1);
                acc2 = _mm256_fmadd_ps(vx, Avx2FloatStorage<Storage>::load(a2 + i), acc2);
                acc3 = _mm256_fmadd_ps(vx, Avx2FloatStorage<Storage>::load(a3 + i), acc3);
            }
            float sums[4] = {
                avx2HorizontalSum(acc0),
                avx2HorizontalSum(acc1),
                avx2HorizontalSum(acc2),
                avx2HorizontalSum(acc3)
            };
            for (; i < n; ++i) {
                float xv = static_cast<float>(x[i]);
                sums[0] += xv * static_cast<float>(a0[i]);
                sums[1] += xv * static_cast<float>(a1[i]);
                sums[2] += xv * static_cast<float>(a2[i]);
                sums[3] += xv * static_cast<float>(a3[i]);
            }
            for (int j = 0; j < 4; ++j) {
                // beta为0时不读取旧y，允许GEMV输出存储尚未初始化。
                y[j * incy] = beta == 0.0f ? alpha * sums[j] : beta * y[j * incy] + alpha * sums[j];
            }
            return;
        }
    }

    for (int j = 0; j < f; ++j) {
        float previous = beta == 0.0f ? 0.0f : y[j * incy];
        y[j * incy] = dotxv(n, alpha, x, incx, a + j * csa, rsa, beta, previous);
    }
}

}  // namespace yt::blas
