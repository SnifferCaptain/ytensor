#pragma once
/***************
 * @file sger.hpp
 * @brief 单精度向量外积更新（SGER）AVX2内核
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * BLAS风格接口: A[m×n] += alpha * x[m] * y[n]^T
 * 4行并行展开，支持任意步幅
 ***************/

#include "gemm_utils.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// SGER：C[m×n] = alpha*x*y^T + beta*C
// ============================================================================

/// @brief 高性能 SGER（外积更新），行主序快速路径 + 通用步幅回退
/// @param m     行数（x 向量长度）
/// @param n     列数（y 向量长度）
/// @param alpha 缩放因子
/// @param beta  已有矩阵 C 的缩放因子
/// @param x     列向量 x[m]，步幅 incx
/// @param y     行向量 y[n]，步幅 incy
/// @param C     输出矩阵，支持任意步幅 (rsc行步幅, csc列步幅)
__attribute__((hot))
inline void sger(int m, int n,
                  float alpha, float beta,
                  const float* __restrict x, int64_t incx,
                  const float* __restrict y, int64_t incy,
                  float* __restrict C, int64_t rsc, int64_t csc) {
    if (m <= 0 || n <= 0) return;

    // 快速路径：行主序 C（连续行，y 也连续）
    if (csc == 1 && rsc >= n && incy == 1) {
        __m256 vbeta = _mm256_set1_ps(beta);
        int i = 0;
        // 主循环：4行并行
        for (; i + 4 <= m; i += 4) {
            __m256 va0 = _mm256_set1_ps(alpha * x[i * incx]);
            __m256 va1 = _mm256_set1_ps(alpha * x[(i + 1) * incx]);
            __m256 va2 = _mm256_set1_ps(alpha * x[(i + 2) * incx]);
            __m256 va3 = _mm256_set1_ps(alpha * x[(i + 3) * incx]);
            float* c0 = C + i * rsc;
            float* c1 = C + (i + 1) * rsc;
            float* c2 = C + (i + 2) * rsc;
            float* c3 = C + (i + 3) * rsc;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vy = _mm256_loadu_ps(y + j);
                if (beta == 0.0f) {
                    _mm256_storeu_ps(c0 + j, _mm256_mul_ps(va0, vy));
                    _mm256_storeu_ps(c1 + j, _mm256_mul_ps(va1, vy));
                    _mm256_storeu_ps(c2 + j, _mm256_mul_ps(va2, vy));
                    _mm256_storeu_ps(c3 + j, _mm256_mul_ps(va3, vy));
                } else {
                    _mm256_storeu_ps(c0 + j, _mm256_fmadd_ps(va0, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c0 + j))));
                    _mm256_storeu_ps(c1 + j, _mm256_fmadd_ps(va1, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c1 + j))));
                    _mm256_storeu_ps(c2 + j, _mm256_fmadd_ps(va2, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c2 + j))));
                    _mm256_storeu_ps(c3 + j, _mm256_fmadd_ps(va3, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c3 + j))));
                }
            }
            // 尾部
            float a0 = alpha * x[i * incx];
            float a1 = alpha * x[(i + 1) * incx];
            float a2 = alpha * x[(i + 2) * incx];
            float a3 = alpha * x[(i + 3) * incx];
            for (; j < n; ++j) {
                float yj = y[j];
                c0[j] = a0 * yj + beta * c0[j];
                c1[j] = a1 * yj + beta * c1[j];
                c2[j] = a2 * yj + beta * c2[j];
                c3[j] = a3 * yj + beta * c3[j];
            }
        }
        // 剩余行
        for (; i < m; ++i) {
            float a_val = alpha * x[i * incx];
            float* c_row = C + i * rsc;
            __m256 va = _mm256_set1_ps(a_val);
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vy = _mm256_loadu_ps(y + j);
                if (beta == 0.0f)
                    _mm256_storeu_ps(c_row + j, _mm256_mul_ps(va, vy));
                else
                    _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(va, vy, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c_row + j))));
            }
            for (; j < n; ++j) c_row[j] = a_val * y[j] + beta * c_row[j];
        }
        return;
    }

    // 通用回退：任意步幅
    for (int i = 0; i < m; ++i) {
        float a_val = alpha * x[i * incx];
        for (int j = 0; j < n; ++j) {
            float* c_ptr = C + i * rsc + j * csc;
            *c_ptr = a_val * y[j * incy] + beta * (*c_ptr);
        }
    }
}

/// @brief SGER 简化接口：A += alpha * x * y^T（行主序A，无beta）
/// @param m     行数
/// @param n     列数
/// @param alpha 缩放因子
/// @param x     列向量 x[m]
/// @param y     行向量 y[n]
/// @param A     行主序矩阵 A[m × lda]
/// @param lda   A 的列步幅（leading dimension）
inline void sger(int m, int n, float alpha,
                  const float* __restrict x, const float* __restrict y,
                  float* __restrict A, int lda) {
    sger(m, n, alpha, 1.0f, x, 1, y, 1, A, lda, 1);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
