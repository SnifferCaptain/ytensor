#pragma once
/***************
 * @file level1.hpp
 * @brief AVX2 BLAS Level-1v物理内核
 ***************/

#include <immintrin.h>

#include <cstdint>
#include <type_traits>

#include "../../../type/float_spec.hpp"
#include "avx2_float_storage.hpp"

namespace yt::blas {

/// @brief 计算rho = beta*rho + alpha*dot(x,y)
/// @details 步幅以元素计；FP16重载使用FP32累加。
/// @note 物理内核不分配内存、不创建线程，也不检查输入存储范围。
template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
float dotxv(
    int n, float alpha, const Storage* x, int64_t incx, const Storage* y, int64_t incy, float beta, float rho
);

/// @brief 计算y = y + alpha*x
template <typename XStorage, typename YStorage>
    requires(
        (std::is_same_v<XStorage, float> && std::is_same_v<YStorage, float>) ||
        (std::is_same_v<XStorage, yt::float16> && std::is_same_v<YStorage, yt::float16>) ||
        (std::is_same_v<XStorage, yt::float16> && std::is_same_v<YStorage, float>)
    )
void axpyv(int n, float alpha, const XStorage* x, int64_t incx, YStorage* y, int64_t incy);

/// @brief 计算x = beta*x
template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
void scalv(int n, float beta, Storage* x, int64_t incx);

/// @brief 将向量元素设置为指定值
template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
void setv(int n, float value, Storage* x, int64_t incx);

}  // namespace yt::blas

#include "../../../../src/blas/kernels/avx2/dotxv.inl"
#include "../../../../src/blas/kernels/avx2/axpyv.inl"
#include "../../../../src/blas/kernels/avx2/setv.inl"
#include "../../../../src/blas/kernels/avx2/scalv.inl"
