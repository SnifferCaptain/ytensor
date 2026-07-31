#pragma once
/***************
 * file: blas/kernels/avx2/setv.inl
 * purpose: AVX2向量赋值物理内核实现。
 ***************/

namespace yt::blas {

template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
inline void setv(int n, float value, Storage* x, int64_t incx) {
    if (n <= 0) return;

    int i = 0;
    if constexpr (std::is_same_v<Storage, float>) {
        if (incx == 1) {
            __m256 vector = _mm256_set1_ps(value);
            for (; i + 8 <= n; i += 8) Avx2FloatStorage<Storage>::store(x + i, vector);
        }
    }
    Storage storedValue(value);
    for (; i < n; ++i) x[i * incx] = storedValue;
}

}  // namespace yt::blas
