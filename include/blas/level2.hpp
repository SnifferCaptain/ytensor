#pragma once
/***************
 * @file level2.hpp
 * @brief BLAS Level-2矩阵向量操作
 ***************/

#include <cstdint>
#include <type_traits>
#include <vector>

#include "config.hpp"
#include "level1f.hpp"

#if YT_USE_YBLAS

namespace yt::blas {

/// @brief 使用typed storage计算行式GEMV，FP16输出使用FP32累加。
template <typename T>
void gemv_row(
    const T* x, const T* B, T* y, int n, int k, float alpha, float beta, int64_t incx, int64_t rsb,
    int64_t csb, int64_t incy, const BlasContext& context = defaultBlasContext()
);

/// @brief 使用typed storage计算列式GEMV，FP16输出使用FP32累加。
template <typename T>
void gemv_col(
    const T* A, const T* x, T* y, int m, int k, float alpha, float beta, int64_t rsa, int64_t csa,
    int64_t incx, int64_t incy, const BlasContext& context = defaultBlasContext()
);

/// @brief 使用typed storage计算GER，FP16输出使用FP32累加。
template <typename T>
void ger(
    int m, int n, float alpha, float beta, const T* x, int64_t incx, const T* y, int64_t incy, T* C,
    int64_t rsc, int64_t csc
);

/// @brief 计算y[n] = beta*y + alpha*x[k]*B[k,n]
/// @details 矩阵地址为base[row*rsb+col*csb]，所有步幅均以元素计。
void gemv_row_simd(
    const float* x, const float* B, float* y, int n, int k, float alpha, float beta, int64_t incx,
    int64_t rsb, int64_t csb, int64_t incy, const BlasContext& context = defaultBlasContext()
);

/// @brief 计算y[m] = beta*y + alpha*A[m,k]*x[k]
/// @details 矩阵地址为base[row*rsa+col*csa]，所有步幅均以元素计。
void gemv_col_simd(
    const float* A, const float* x, float* y, int m, int k, float alpha, float beta, int64_t rsa, int64_t csa,
    int64_t incx, int64_t incy, const BlasContext& context = defaultBlasContext()
);

/// @brief 计算C(i,j) = beta*C(i,j) + alpha*x(i)*y(j)
/// @note 该接口显式包含标准BLAS GER没有的beta参数。
void sger(
    int m, int n, float alpha, float beta, const float* x, int64_t incx, const float* y, int64_t incy,
    float* C, int64_t rsc, int64_t csc
);

void sger(int m, int n, float alpha, const float* x, const float* y, float* C, int lda);

/// @brief 计算FP16列式GEMV
/// @details 每个输出元素使用FP32累加，完成后转换回FP16。
void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y, int m, int k, float alpha, float beta,
    int64_t rsa, int64_t csa, int64_t incx, int64_t incy, const BlasContext& context = defaultBlasContext()
);

/// @brief 计算FP16行式GEMV
/// @details 每个输出元素使用FP32累加，完成后转换回FP16。
void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* y, int n, int k, float alpha, float beta,
    int64_t incx, int64_t rsb, int64_t csb, int64_t incy, const BlasContext& context = defaultBlasContext()
);

void hgemv(
    const yt::float16* A, const yt::float16* x, yt::float16* y, int m, int k, float alpha = 1.0f,
    float beta = 0.0f
);

/// @brief 计算FP16 GER并使用FP32临时累加
void hger(
    int m, int n, float alpha, float beta, const yt::float16* x, int64_t incx, const yt::float16* y,
    int64_t incy, yt::float16* C, int64_t rsc, int64_t csc
);

void hger(int m, int n, float alpha, const yt::float16* x, const yt::float16* y, yt::float16* C, int lda);

}  // namespace yt::blas

#include "../../src/blas/gemv_row.inl"
#include "../../src/blas/gemv_col.inl"
#include "../../src/blas/ger.inl"
#endif  // YT_USE_YBLAS
