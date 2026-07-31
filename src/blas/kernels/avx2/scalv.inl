#pragma once
/***************
 * file: blas/kernels/avx2/scalv.inl
 * purpose: AVX2向量缩放物理内核实现。
 ***************/

namespace yt::blas {

template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
inline void scalv(int n, float beta, Storage* x, int64_t incx) {
    if (beta == 0.0f) {
        setv(n, 0.0f, x, incx);
        return;
    }
    if (beta == 1.0f || n <= 0) return;

    int i = 0;
    if constexpr (Avx2FloatStorage<Storage>::available) {
        if (incx == 1) {
            __m256 vb = _mm256_set1_ps(beta);
            for (; i + 8 <= n; i += 8) {
                __m256 result = _mm256_mul_ps(vb, Avx2FloatStorage<Storage>::load(x + i));
                Avx2FloatStorage<Storage>::store(x + i, result);
            }
        }
    }
    for (; i < n; ++i) x[i * incx] = Storage(beta * static_cast<float>(x[i * incx]));
}

}  // namespace yt::blas
