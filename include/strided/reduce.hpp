#pragma once
/***************
 * @file: strided/reduce.hpp
 * @brief: strided layout 的 reduce 职责入口。
 ***************/

#include <utility>
#include <vector>

namespace yt {
class YTensorBase;
template <typename T, int dim>
class YTensor;
}  // namespace yt

namespace yt::strided {

/// @brief 沿单个 axis 对 runtime tensor 求和并保留 reduced 维度。
YTensorBase sum(const YTensorBase& tensor, int axis);
/// @brief 沿多个 axes 对 runtime tensor 求和并保留 reduced 维度。
YTensorBase sum(const YTensorBase& tensor, const std::vector<int>& axes);
/// @brief 沿单个 axis 计算 runtime tensor 均值并保留 reduced 维度。
YTensorBase mean(const YTensorBase& tensor, int axis);
/// @brief 沿多个 axes 计算 runtime tensor 均值并保留 reduced 维度。
YTensorBase mean(const YTensorBase& tensor, const std::vector<int>& axes);
/// @brief 返回单轴 runtime 最大值及轴内索引。
std::pair<YTensorBase, YTensorBase> max(const YTensorBase& tensor, int axis);
/// @brief 返回多轴 runtime 最大值及 row-major flattened 索引。
std::pair<YTensorBase, YTensorBase> max(const YTensorBase& tensor, const std::vector<int>& axes);

/// @brief 沿单个 axis 对 rank 大于 1 的 typed tensor 求和。
template <typename T, int dim>
YTensor<T, dim> sum(const YTensor<T, dim>& tensor, int axis) requires(dim > 1);

/// @brief 沿多个 axes 对 rank 大于 1 的 typed tensor 求和。
template <typename T, int dim>
YTensor<T, dim> sum(const YTensor<T, dim>& tensor, std::vector<int> axes) requires(dim > 1);

/// @brief 对 rank-1 typed tensor 求和并返回 scalar。
template <typename T, int dim>
T sum(const YTensor<T, dim>& tensor, int axis = 0) requires(dim == 1);

/// @brief 沿单个 axis 计算 rank 大于 1 的 typed tensor 均值。
template <typename T, int dim>
YTensor<T, dim> mean(const YTensor<T, dim>& tensor, int axis) requires(dim > 1);

/// @brief 沿多个 axes 计算 rank 大于 1 的 typed tensor 均值。
template <typename T, int dim>
YTensor<T, dim> mean(const YTensor<T, dim>& tensor, std::vector<int> axes) requires(dim > 1);

/// @brief 计算 rank-1 typed tensor 均值并返回 scalar。
template <typename T, int dim>
T mean(const YTensor<T, dim>& tensor, int axis = 0) requires(dim == 1);

/// @brief 返回单轴 typed 最大值及轴内索引。
template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> max(const YTensor<T, dim>& tensor, int axis) requires(dim > 1);

/// @brief 返回多轴 typed 最大值及 row-major flattened 索引。
template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> max(
    const YTensor<T, dim>& tensor, std::vector<int> axes
) requires(dim > 1);

/// @brief 返回 rank-1 typed tensor 的最大值及索引。
template <typename T, int dim>
std::pair<T, int> max(const YTensor<T, dim>& tensor, int axis = 0) requires(dim == 1);

}  // namespace yt::strided
