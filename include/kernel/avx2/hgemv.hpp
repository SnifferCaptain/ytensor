#pragma once
/***************
 * @file hgemv.hpp
 * @brief 半精度矩阵-向量乘法（HGEMV）实现
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * C = alpha * A @ x + beta * C，内部以 float32 计算
 * 支持 行向量×矩阵 和 矩阵×列向量 两种模式
 ***************/

#include "hgemm.hpp"   // f16_to_f32_block, f32_to_f16_block
#include "sgemv.hpp"

#if defined(__AVX2__) && defined(__FMA__)

#include <memory>

namespace yt::kernel::avx2 {

// ============================================================================
// 半精度 GEMV 主入口
// mode: false = 矩阵×列向量 (C[m×1] = A[m×k] @ B[k×1])
//        true  = 行向量×矩阵 (C[1×n] = A[1×k] @ B[k×n])
// ============================================================================

/// @brief 矩阵×列向量: y[m×1] = alpha * A[m×k] @ x[k×1] + beta * y
inline void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y,
    int m, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsx,
    int64_t rsy
) {
    if (m <= 0 || k <= 0) return;

    // 将 A, x 转换为 float32（按行/块转换）
    std::unique_ptr<float[]> fA(new float[(size_t)m * k]);
    std::unique_ptr<float[]> fx(new float[k]);
    std::unique_ptr<float[]> fy(new float[m]);

    // 转换 A（逐元素，支持任意步幅）
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            fA[i * k + p] = static_cast<float>(A[i * rsa + p * csa]);

    // 转换 x
    for (int p = 0; p < k; ++p)
        fx[p] = static_cast<float>(x[p * rsx]);

    // 调用 f32 GEMV（行主序 fA，连续 fx）
    gemv_col_simd(fA.get(), fx.get(), fy.get(), m, k, alpha, 0.0f, k, 1, 1, 1);

    // 写回 float16，合并 beta * y
    for (int i = 0; i < m; ++i) {
        float r = fy[i] + (beta != 0.0f ? beta * static_cast<float>(y[i * rsy]) : 0.0f);
        y[i * rsy] = yt::float16(r);
    }
}

/// @brief 行向量×矩阵: z[1×n] = alpha * x[1×k] @ B[k×n] + beta * z
inline void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* z,
    int n, int k,
    float alpha, float beta,
    int64_t csx,
    int64_t rsb, int64_t csb,
    int64_t csz
) {
    if (n <= 0 || k <= 0) return;

    std::unique_ptr<float[]> fx(new float[k]);
    std::unique_ptr<float[]> fB(new float[(size_t)k * n]);
    std::unique_ptr<float[]> fz(new float[n]);

    for (int p = 0; p < k; ++p)
        fx[p] = static_cast<float>(x[p * csx]);

    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j)
            fB[p * n + j] = static_cast<float>(B[p * rsb + j * csb]);

    gemv_row_simd(fx.get(), fB.get(), fz.get(), n, k, alpha, 0.0f, 1, n, 1, 1);

    for (int j = 0; j < n; ++j) {
        float r = fz[j] + (beta != 0.0f ? beta * static_cast<float>(z[j * csz]) : 0.0f);
        z[j * csz] = yt::float16(r);
    }
}

/// @brief hgemv 通用接口（行主序简化版）
inline void hgemv(const yt::float16* A, const yt::float16* x, yt::float16* y,
                   int m, int k, float alpha = 1.0f, float beta = 0.0f) {
    hgemv_col(A, x, y, m, k, alpha, beta, k, 1, 1, 1);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
