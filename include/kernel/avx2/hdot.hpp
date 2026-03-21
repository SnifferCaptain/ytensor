#pragma once
/***************
 * @file hdot.hpp
 * @brief 半精度向量点积（HDOT）实现
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * 策略: float16 → float32 分块转换 → 调用 sdot → 返回 float
 * 若启用 F16C，转换采用硬件指令；否则使用软件回退
 ***************/

#include "hgemm.hpp"   // 包含 f16_to_f32_block 等转换工具
#include "sdot.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// 半精度点积: result = sum(x[i] * y[i])，内部以 float32 计算
// ============================================================================

/// @brief 半精度向量点积
/// @param x    float16 向量，步幅 incx
/// @param y    float16 向量，步幅 incy
/// @param n    元素个数
/// @return     float 精度的点积结果
inline float hdot(const yt::float16* x, const yt::float16* y, int n,
                   int64_t incx = 1, int64_t incy = 1) {
    if (n <= 0) return 0.0f;

    constexpr int BLOCK = 256;  // 每次转换的块大小
    float sum = 0.0f;

    if (incx == 1 && incy == 1) {
        // 快速路径：连续内存，分块转换后调用 sdot_contiguous
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
        // 通用步幅回退（较少见场景，标量处理）
        for (int i = 0; i < n; ++i)
            sum += static_cast<float>(x[i * incx]) * static_cast<float>(y[i * incy]);
    }
    return sum;
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
