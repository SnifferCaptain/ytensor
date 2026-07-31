#pragma once
/***************
 * file: blas/kernels/avx2/axpyv.inl
 * purpose: AVX2向量AXPY物理内核实现。
 ***************/

namespace yt::blas {

template <typename XStorage, typename YStorage>
    requires(
        (std::is_same_v<XStorage, float> && std::is_same_v<YStorage, float>) ||
        (std::is_same_v<XStorage, yt::float16> && std::is_same_v<YStorage, yt::float16>) ||
        (std::is_same_v<XStorage, yt::float16> && std::is_same_v<YStorage, float>)
    )
inline void axpyv(int n, float alpha, const XStorage* x, int64_t incx, YStorage* y, int64_t incy) {
    if (n <= 0 || alpha == 0.0f) return;

    int i = 0;
    if constexpr (Avx2FloatStorage<XStorage>::available && Avx2FloatStorage<YStorage>::available) {
        if (incx == 1 && incy == 1) {
            __m256 va = _mm256_set1_ps(alpha);
            for (; i + 8 <= n; i += 8) {
                __m256 result = _mm256_fmadd_ps(
                    va, Avx2FloatStorage<XStorage>::load(x + i), Avx2FloatStorage<YStorage>::load(y + i)
                );
                Avx2FloatStorage<YStorage>::store(y + i, result);
            }
        }
    }
    for (; i < n; ++i) {
        y[i * incy] = YStorage(
            static_cast<float>(y[i * incy]) + alpha * static_cast<float>(x[i * incx])
        );
    }
}

}  // namespace yt::blas
