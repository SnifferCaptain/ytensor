#pragma once
/***************
 * @file: strided/copy.hpp
 * @brief: strided layout 的 copy/clone/contiguous 职责入口。
 ***************/

#include <vector>

namespace yt {
class YTensorBase;
template <typename T, int dim>
class YTensor;
}  // namespace yt

namespace yt::strided {

/// @brief 创建逻辑内容相同、连续存储且不共享原 storage 的 runtime tensor。
YTensorBase clone(const YTensorBase& tensor);

/// @brief 将 src 的逻辑元素复制到形状相同的 dst，必要时执行 dtype cast。
/// @return dst 的引用。
YTensorBase& copy_(YTensorBase& dst, const YTensorBase& src);

/// @brief 返回 tensor 的连续版本；已经连续时保留共享 storage 的浅拷贝语义。
YTensorBase contiguous(const YTensorBase& tensor);

/// @brief 必要时用连续副本替换 tensor。
YTensorBase& contiguous_(YTensorBase& tensor);

/// @brief 沿 axis 拼接 dtype 和非拼接维度一致的 runtime tensors。
YTensorBase concat(const std::vector<YTensorBase>& tensors, int axis = 0);

/// @brief typed copy facade，复用 runtime Strided owner。
template <typename T, int dim>
YTensor<T, dim>& copy_(YTensor<T, dim>& dst, const YTensorBase& src);

/// @brief 创建 typed tensor 的连续独立副本。
template <typename T, int dim>
YTensor<T, dim> clone(const YTensor<T, dim>& tensor);

/// @brief 返回 typed tensor 的连续版本。
template <typename T, int dim>
YTensor<T, dim> contiguous(const YTensor<T, dim>& tensor);

/// @brief 必要时原地连续化 typed tensor。
template <typename T, int dim>
YTensor<T, dim>& contiguous_(YTensor<T, dim>& tensor);
}  // namespace yt::strided
