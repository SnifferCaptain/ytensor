#pragma once
/***************
 * @file hger.hpp
 * @brief 半精度向量外积更新（HGER）实现
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * C[m×n] = alpha * x[m] * y[n]^T + beta * C[m×n]，内部以 float32 计算
 ***************/

#include "hgemm.hpp"   // f16_to_f32_block, f32_to_f16_block
#include "sger.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// 半精度外积更新
// ============================================================================

/// @brief HGER: C[m×n] = alpha * x * y^T + beta * C（行主序快速路径）
/// @param m     行数
/// @param n     列数
/// @param alpha x*y^T 缩放因子
/// @param beta  已有矩阵 C 的缩放因子
/// @param x     float16 列向量 x[m]，步幅 incx
/// @param y     float16 行向量 y[n]，步幅 incy
/// @param C     float16 输出矩阵，支持任意步幅
/// @param rsc   C 行步幅
/// @param csc   C 列步幅
inline void hger(int m, int n,
                  float alpha, float beta,
                  const yt::float16* x, int64_t incx,
                  const yt::float16* y, int64_t incy,
                  yt::float16* C, int64_t rsc, int64_t csc) {
    if (m <= 0 || n <= 0) return;

    // 转换 x 到 float32
    std::unique_ptr<float[]> fx(new float[m]);
    std::unique_ptr<float[]> fy(new float[n]);
    for (int i = 0; i < m; ++i) fx[i] = static_cast<float>(x[i * incx]);
    for (int j = 0; j < n; ++j) fy[j] = static_cast<float>(y[j * incy]);

    // 快速路径：行主序 C（连续列）
    if (csc == 1 && rsc >= n) {
        // 使用 f32 逐行更新，再写回 f16
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * fx[i];
            yt::float16* row = C + i * rsc;
            int j = 0;
#if __F16C__
            __m256 va = _mm256_set1_ps(a_val);
            for (; j + 8 <= n; j += 8) {
                // 加载 f16 → f32
                __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
                __m256 vc = _mm256_cvtph_ps(h);
                __m256 vy = _mm256_loadu_ps(fy.get() + j);
                if (beta == 0.0f)
                    vc = _mm256_mul_ps(va, vy);
                else
                    vc = _mm256_fmadd_ps(va, vy, _mm256_mul_ps(_mm256_set1_ps(beta), vc));
                // 写回 f16
                __m128i hr = _mm256_cvtps_ph(vc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row + j), hr);
            }
#endif
            for (; j < n; ++j)
                row[j] = yt::float16(a_val * fy[j] + beta * static_cast<float>(row[j]));
        }
        return;
    }

    // 通用步幅回退
    for (int i = 0; i < m; ++i) {
        float a_val = alpha * fx[i];
        for (int j = 0; j < n; ++j) {
            yt::float16* cp = C + i * rsc + j * csc;
            *cp = yt::float16(a_val * fy[j] + beta * static_cast<float>(*cp));
        }
    }
}

/// @brief HGER 简化接口: A += alpha * x * y^T（行主序A，无beta，无步幅）
inline void hger(int m, int n, float alpha,
                  const yt::float16* x, const yt::float16* y,
                  yt::float16* A, int lda) {
    hger(m, n, alpha, 1.0f, x, 1, y, 1, A, lda, 1);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
