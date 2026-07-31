#pragma once
/***************
 * file: blas/dot.inl
 * purpose: BLAS dot兼容入口。
 ***************/

namespace yt::blas {

inline float sdot_stride(const float* x, const float* y, int n, int64_t incx, int64_t incy) {
    return dotxv(n, 1.0f, x, incx, y, incy, 0.0f, 0.0f);
}

inline float sdot_contiguous(const float* x, const float* y, int n) {
    return dotxv(n, 1.0f, x, 1, y, 1, 0.0f, 0.0f);
}

inline float sdot(const float* x, const float* y, int n, int64_t incx, int64_t incy) {
    return dotxv(n, 1.0f, x, incx, y, incy, 0.0f, 0.0f);
}

inline float hdot(const yt::float16* x, const yt::float16* y, int n, int64_t incx, int64_t incy) {
    return dotxv(n, 1.0f, x, incx, y, incy, 0.0f, 0.0f);
}

}  // namespace yt::blas
