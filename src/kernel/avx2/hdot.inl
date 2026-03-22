#pragma once

namespace yt::kernel::avx2 {

inline float hdot(const yt::float16* x, const yt::float16* y, int n,
                  int64_t incx, int64_t incy) {
    if (n <= 0) return 0.0f;

    constexpr int BLOCK = 256;
    float sum = 0.0f;

    if (incx == 1 && incy == 1) {
        alignas(32) float fx[BLOCK], fy[BLOCK];
        int i = 0;
        for (; i + BLOCK <= n; i += BLOCK) {
            f16_to_f32_block(x + i, fx, BLOCK);
            f16_to_f32_block(y + i, fy, BLOCK);
            sum += sdot_contiguous(fx, fy, BLOCK);
        }
        if (i < n) {
            int rem = n - i;
            f16_to_f32_block(x + i, fx, rem);
            f16_to_f32_block(y + i, fy, rem);
            sum += sdot_contiguous(fx, fy, rem);
        }
    } else {
        for (int i = 0; i < n; ++i)
            sum += static_cast<float>(x[i * incx]) * static_cast<float>(y[i * incy]);
    }
    return sum;
}

} // namespace yt::kernel::avx2
