#pragma once
/***************
 * file: blas/kernels/avx2/axpyf.inl
 * purpose: AVX2融合多向量AXPY物理内核实现。
 ***************/

namespace yt::blas {

template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
inline void axpyf(
    const BlasContext& context, int n, int f, float alpha, const Storage* a, int64_t rsa, int64_t csa,
    const Storage* x, int64_t incx, float* y, int64_t incy
) {
    if (n <= 0 || f <= 0 || alpha == 0.0f) return;

    if constexpr (Avx2FloatStorage<Storage>::available) {
        if (context.af == 8 && f == 8 && rsa == 1 && incy == 1) {
            __m256 scales[8];
            for (int j = 0; j < 8; ++j) {
                scales[j] = _mm256_set1_ps(alpha * static_cast<float>(x[j * incx]));
            }
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 vy = Avx2FloatStorage<float>::load(y + i);
                for (int j = 0; j < 8; ++j) {
                    vy = _mm256_fmadd_ps(scales[j], Avx2FloatStorage<Storage>::load(a + j * csa + i), vy);
                }
                Avx2FloatStorage<float>::store(y + i, vy);
            }
            for (; i < n; ++i) {
                float value = y[i];
                for (int j = 0; j < 8; ++j) {
                    value += alpha * static_cast<float>(x[j * incx]) * static_cast<float>(a[i + j * csa]);
                }
                y[i] = value;
            }
            return;
        }
    }

    for (int j = 0; j < f; ++j) {
        axpyv(n, alpha * static_cast<float>(x[j * incx]), a + j * csa, rsa, y, incy);
    }
}

}  // namespace yt::blas
