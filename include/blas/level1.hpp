#pragma once
/***************
 * @file level1.hpp
 * @brief BLAS Level-1v向量操作
 ***************/

#include "../type/float_spec.hpp"
#include "config.hpp"

#if YT_USE_YBLAS
#include "kernels/avx2/level1.hpp"

namespace yt::blas {

/// @brief 计算带元素步幅的FP32向量点积
/// @note 负步幅要求指针已经指向逻辑首元素；n小于等于0时返回0。
float sdot_stride(const float* x, const float* y, int n, int64_t incx, int64_t incy);

/// @brief 计算连续FP32向量点积
float sdot_contiguous(const float* x, const float* y, int n);

/// @brief 计算FP32向量点积
float sdot(const float* x, const float* y, int n, int64_t incx = 1, int64_t incy = 1);

/// @brief 计算FP16向量点积并使用FP32累加
float hdot(const yt::float16* x, const yt::float16* y, int n, int64_t incx = 1, int64_t incy = 1);

}  // namespace yt::blas

#include "../../src/blas/dot.inl"
#endif  // YT_USE_YBLAS
