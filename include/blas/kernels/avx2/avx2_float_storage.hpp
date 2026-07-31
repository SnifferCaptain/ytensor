#pragma once
/***************
 * @file avx2_float_storage.hpp
 * @brief AVX2浮点存储的八元素加载与存储转换
 ***************/

#include <immintrin.h>

#include "../../../type/float_spec.hpp"

namespace yt::blas {

template <typename Storage>
struct Avx2FloatStorage;

template <>
struct Avx2FloatStorage<float> {
    static constexpr bool available = true;

    static __m256 load(const float* values) {
        return _mm256_loadu_ps(values);
    }

    static void store(float* values, __m256 vector) {
        _mm256_storeu_ps(values, vector);
    }
};

template <>
struct Avx2FloatStorage<yt::float16> {
#if defined(__F16C__)
    static constexpr bool available = true;

    static __m256 load(const yt::float16* values) {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values)));
    }

    static void store(yt::float16* values, __m256 vector) {
        __m128i converted = _mm256_cvtps_ph(vector, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(values), converted);
    }
#else
    static constexpr bool available = false;
#endif
};

}  // namespace yt::blas
