#pragma once
/***************
 * @file level1f.hpp
 * @brief AVX2 BLAS Level-1f物理内核
 ***************/

#include "../../context.hpp"
#include "../../level1.hpp"

#include <type_traits>

namespace yt::blas {

/// @brief 融合计算多列点积
/// @details a(i,j)=a[i*rsa+j*csa]，计算y[j]=beta*y[j]+alpha*sum_i(x[i]*a(i,j))。
/// @note FP16重载将输入转换后累加到FP32输出；调用方负责线程划分。
template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
void dotxf(
    const BlasContext& context, int n, int f, float alpha, const Storage* x, int64_t incx, const Storage* a,
    int64_t rsa, int64_t csa, float beta, float* y, int64_t incy
);

/// @brief 融合计算多向量AXPY
/// @details a(i,j)=a[i*rsa+j*csa]，计算y[i]+=alpha*sum_j(a(i,j)*x[j])。
template <typename Storage>
    requires(std::is_same_v<Storage, float> || std::is_same_v<Storage, yt::float16>)
void axpyf(
    const BlasContext& context, int n, int f, float alpha, const Storage* a, int64_t rsa, int64_t csa,
    const Storage* x, int64_t incx, float* y, int64_t incy
);

}  // namespace yt::blas

#include "../../../../src/blas/kernels/avx2/dotxf.inl"
#include "../../../../src/blas/kernels/avx2/axpyf.inl"
