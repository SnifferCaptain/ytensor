#pragma once
/***************
 * @file sdot.hpp
 * @brief 单精度向量点积（SDOT）AVX2内核
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * BLAS风格接口: result = sum(x[i] * y[i])
 * 四路累加器展开，充分利用 FMA 流水线，支持任意步幅
 ***************/

#include "gemm_utils.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// 核心点积：连续内存，四路累加器展开（最大化 ILP）
// ============================================================================

/// @brief 标量 dot product（通用步幅回退）
inline float sdot_stride(const float* x, const float* y, int n,
                          int64_t incx, int64_t incy) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i * incx] * y[i * incy];
    return sum;
}

/// @brief AVX2 SIMD dot product（要求连续内存 incx==incy==1）
/// @param n 元素个数
/// @return x · y
__attribute__((hot))
inline float sdot_contiguous(const float* __restrict x, const float* __restrict y, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    // 主循环：每轮处理 32 个元素（4 × AVX 向量，充分流水）
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_loadu_ps(y + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i +  8), _mm256_loadu_ps(y + i +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), acc3);
    }
    // 尾部：每轮 8 个元素
    for (; i + 8 <= n; i += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), acc0);

    // 合并四路累加器
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float sum = hsum_ps(acc0);

    // 标量尾部
    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

// ============================================================================
// 通用接口
// ============================================================================

/// @brief 计算单精度向量点积: result = sum_{i=0}^{n-1} x[i*incx] * y[i*incy]
/// @param x   向量x的起始指针
/// @param y   向量y的起始指针
/// @param n   元素个数
/// @param incx x的步幅（通常=1）
/// @param incy y的步幅（通常=1）
inline float sdot(const float* x, const float* y, int n,
                   int64_t incx = 1, int64_t incy = 1) {
    if (n <= 0) return 0.0f;
    if (incx == 1 && incy == 1)
        return sdot_contiguous(x, y, n);
    return sdot_stride(x, y, n, incx, incy);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
