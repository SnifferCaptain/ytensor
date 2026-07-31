#pragma once
/***************
 * @file: strided/matmul.hpp
 * @brief: strided layout 的 matmul 职责入口。
 ***************/

#include "../ytensor_concepts.hpp"
#include "../ytensor_infos.hpp"

namespace yt {
class YTensorBase;
template <typename T, int dim>
class YTensor;
}  // namespace yt

namespace yt::strided {

/// @brief 按 backend 偏好计算 runtime tensor 矩阵乘法。
/// @note backend 不满足 dtype/layout 条件时会依次回退到可用的 Eigen 或 Naive 实现。
YTensorBase matmul(
    const YTensorBase& left, const YTensorBase& right,
    yt::info::MatmulBackend backend = yt::info::defaultMatmulBackend
);

/// @brief 按 backend 偏好使用二维 bool mask 计算 runtime tensor 掩码矩阵乘法。
/// @note backend 不满足 dtype/layout 条件时会回退到可用实现。
YTensorBase masked_matmul(
    const YTensorBase& left, const YTensorBase& right, const YTensorBase& mask, double maskedValue = 0.0,
    yt::info::MatmulBackend backend = yt::info::defaultMatmulBackend
);

/// @brief typed 矩阵乘法，输出 rank 至少为 2，并保留广播 batch 维度。
/// @note backend 是偏好而非强制要求；不支持当前 dtype/layout 时会安全回退。
template <typename T, int leftDim, int rightDim>
YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> matmul(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right,
    yt::info::MatmulBackend backend = yt::info::defaultMatmulBackend
);

/// @brief 使用二维 bool tensor mask 计算 typed 掩码矩阵乘法。
/// @note backend 是偏好而非强制要求；不支持当前 dtype/layout 时会安全回退。
template <typename T, int leftDim, int rightDim>
YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> masked_matmul(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, const YTensor<bool, 2>& mask,
    const T& maskedValue = static_cast<T>(0), yt::info::MatmulBackend backend = yt::info::defaultMatmulBackend
);

/// @brief 使用坐标谓词 func 计算 typed 掩码矩阵乘法。
/// @details func 至少支持 `bool(int row, int col)`；可选 tile 谓词接口由 YBLAS backend 按 concept 检测。
/// @note backend 是偏好而非强制要求；不支持当前 dtype/layout 时会安全回退。
template <typename T, int leftDim, int rightDim, typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> masked_matmul(
        const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, Func&& func,
        const T& maskedValue = static_cast<T>(0),
        yt::info::MatmulBackend backend = yt::info::defaultMatmulBackend
    );

}  // namespace yt::strided
