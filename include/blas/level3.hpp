#pragma once
/***************
 * @file level3.hpp
 * @brief BLAS Level-3矩阵乘法frame
 ***************/

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "config.hpp"
#include "level1.hpp"
#include "level2.hpp"
#include "pack.hpp"

#if YT_USE_YBLAS
#include "kernels/avx2/level3.hpp"

namespace yt::blas {

template <typename T>
inline constexpr bool is_supported_gemm_storage_v =
    std::is_same_v<T, float> || std::is_same_v<T, yt::float16>;

template <typename AType, typename BType, typename CType, typename ComputeType, GemmKernelSpec Spec>
struct DefaultGemmDispatcher {
    static_assert(is_supported_gemm_storage_v<AType>);
    static_assert(is_supported_gemm_storage_v<BType>);
    static_assert(is_supported_gemm_storage_v<CType>);
    static_assert(std::is_same_v<ComputeType, float>);
    using type = Avx2GemmMicrokernelDispatcher<ComputeType, Spec>;
};

/// @brief 使用指定storage、compute类型和物理微内核Spec计算GEMM
/// @details A/B在packing时转换为ComputeType，C在完整K归约后按CType输出合同写回。
template <
    typename AType, typename BType, typename CType, typename ComputeType = float,
    GemmKernelSpec Spec = default_gemm_kernel_spec,
    typename Dispatcher = typename DefaultGemmDispatcher<AType, BType, CType, ComputeType, Spec>::type>
void gemm(
    const BlasContext& context, const AType* A, const BType* B, CType* C, int m, int n, int k,
    ComputeType alpha, ComputeType beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc
);

/// @brief 使用坐标mask policy计算typed GEMM
/// @details mask返回true的位置执行GEMM更新，false位置不读写C；无mask GEMM使用all-true policy进入同一frame。
template <
    typename AType, typename BType, typename CType, typename ComputeType = float,
    GemmKernelSpec Spec = default_gemm_kernel_spec,
    typename Dispatcher = typename DefaultGemmDispatcher<AType, BType, CType, ComputeType, Spec>::type,
    typename MaskType>
void gemm_masked(
    const BlasContext& context, const AType* A, const BType* B, CType* C, int m, int n, int k,
    ComputeType alpha, ComputeType beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc, MaskType&& mask
);

/// @brief 使用紧密列主序布局计算C = A*B
/// @note A、B、C必须全部为列主序，且该接口固定alpha=1、beta=0。
void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k);

/// @brief 使用紧密行主序布局计算C = A*B
/// @note A、B、C必须全部为行主序，且该接口固定alpha=1、beta=0。
void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k);

void sgemm_colmajor(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k
);

void sgemm_rowmajor(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k
);

/// @brief 使用布尔掩码计算SGEMM
/// @details 仅在mask(i,j)为true时更新C(i,j)，false位置保持原值。
/// @note mask使用紧密行主序[m,n]布局。
void sgemm_masked(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, const bool* mask
);

void sgemm_masked(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, const bool* mask
);

/// @brief 使用坐标谓词计算SGEMM
/// @details func至少支持bool(int row, int col)，可选tileAllTrue/tileAllFalse由实现检测。
/// @note 谓词必须可重复调用且无副作用，tile接口必须与逐元素结果一致。
template <typename Func>
requires(!std::is_pointer_v<std::decay_t<Func>>) void sgemm_masked(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, Func&& func
);

template <typename Func>
requires(!std::is_pointer_v<std::decay_t<Func>>) void sgemm_masked(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc, Func&& func
);

/// @brief SGEMM兼容入口，使用任意元素步幅计算C = beta*C + alpha*A*B
/// @details 矩阵地址为base[row*row_stride+col*column_stride]，步幅以元素计。
/// @note 该接口转发到typed gemm；维度必须非负，A/B不得与可写C发生会覆盖后续输入的重叠。
void sgemm(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta, int64_t rsa,
    int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
);

void sgemm(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k, float alpha,
    float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
);

void sgemm(
    const float* A, const float* B, float* C, int m, int n, int k, float alpha = 1.0f, float beta = 0.0f
);

void sgemm(
    const BlasContext& context, const float* A, const float* B, float* C, int m, int n, int k,
    float alpha = 1.0f, float beta = 0.0f
);

void matmul(
    const float* A, const float* B, float* C, int m, int n, int k, int64_t rsa, int64_t csa, int64_t rsb,
    int64_t csb, int64_t rsc, int64_t csc
);

void matmul(const float* A, const float* B, float* C, int m, int n, int k);

void matmul(const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta);

/// @brief 使用指定线程数计算紧密行主序矩阵乘法
/// @note 该兼容接口临时覆盖进程级线程设置，不支持并发调用。
void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, int nthreads = 0);

void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k);

/// @brief BLAS兼容FP16 storage GEMM入口
/// @details 该兼容接口仅适配类型与默认Spec，实际计算转发到typed gemm。
void hgemm(
    const BlasContext& context, const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n,
    int k, float alpha, float beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc
);

void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, float alpha, float beta,
    int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
);

void hgemm(
    const BlasContext& context, const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n,
    int k, float alpha = 1.0f, float beta = 0.0f
);

void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, float alpha = 1.0f,
    float beta = 0.0f
);

void hmatmul(
    const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k, int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb, int64_t rsc, int64_t csc
);

void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k);

}  // namespace yt::blas

#include "../../src/blas/gemm.inl"
#include "../../src/blas/gemm_masked.inl"
#include "../../src/blas/sgemm.inl"
#include "../../src/blas/hgemm.inl"
#include "../../src/blas/matmul.inl"

#endif  // YT_USE_YBLAS
