#pragma once
/***************
 * @file: strided/view.hpp
 * @brief: strided layout 的 view 类职责入口。
 ***************/

#include <algorithm>
#include <cstddef>
#include <vector>

namespace yt {
class YTensorBase;
template <typename T, int dim>
class YTensor;
}  // namespace yt

namespace yt::strided {

/// @brief 返回strided张量的逻辑shape。
std::vector<int> shape(const YTensorBase& tensor);
/// @brief 返回指定逻辑维度的长度，维度支持循环索引。
int shape(const YTensorBase& tensor, int dim);
/// @brief 返回由shape计算出的连续逻辑stride。
std::vector<int> logicalStride(const YTensorBase& tensor);
/// @brief 返回当前view实际访问storage时使用的物理stride。
std::vector<int> physicalStride(const YTensorBase& tensor);
/// @brief 返回指定维度的逻辑stride。
int logicalStride(const YTensorBase& tensor, int dim);
/// @brief 返回指定维度的物理stride。
int physicalStride(const YTensorBase& tensor, int dim);
/// @brief 返回逻辑元素数量，标量shape的size为1。
size_t size(const YTensorBase& tensor);
/// @brief 返回逻辑维度数量。
int ndim(const YTensorBase& tensor);
/// @brief 判断指定半开维度区间是否连续。
bool isContiguous(const YTensorBase& tensor, int fromDim, int toDim);
/// @brief 返回指定区间中连续尾部开始的维度。
int isContiguousFrom(const YTensorBase& tensor, int fromDim, int toDim);
/// @brief 判断不同逻辑坐标是否映射到不同storage位置。
bool isDisjoint(const YTensorBase& tensor);
/// @brief 判断两个view实际触及的storage字节跨度是否可能重叠。
bool physicalSpansOverlap(const YTensorBase& left, const YTensorBase& right);
/// @brief 按YTensor多负维度规则补全shape，并保证元素数量不变。
std::vector<int> autoShape(const YTensorBase& tensor, const std::vector<int>& shape);

/// @brief 创建共享storage的slice view。
/// @note start/end使用YTensor循环索引和autoFix语义；负step从区间末端反向访问。
YTensorBase slice(
    const YTensorBase& tensor, int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true
);
/// @brief 原地替换为slice view，不复制storage。
YTensorBase& slice_(
    YTensorBase& tensor, int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true
);
/// @brief 按维度顺序创建共享storage的permute view。
YTensorBase permute(const YTensorBase& tensor, const std::vector<int>& newOrder);
/// @brief 原地更新为permute view。
YTensorBase& permute_(YTensorBase& tensor, const std::vector<int>& newOrder);
/// @brief 交换两个维度并返回共享storage的view。
YTensorBase transpose(const YTensorBase& tensor, int dim0 = -2, int dim1 = -1);
/// @brief 为连续张量创建元素数量相同的新shape view。
YTensorBase view(const YTensorBase& tensor, const std::vector<int>& newShape);
/// @brief 必要时先连续化，再创建指定shape的张量。
YTensorBase reshape(const YTensorBase& tensor, const std::vector<int>& newShape);
/// @brief 插入长度为1的维度并返回共享storage的view。
YTensorBase unsqueeze(const YTensorBase& tensor, int dim);
/// @brief 原地插入长度为1的维度。
YTensorBase& unsqueeze_(YTensorBase& tensor, int dim);
/// @brief 删除指定singleton维度；默认删除全部singleton维度。
YTensorBase squeeze(const YTensorBase& tensor, int dim = -1);
/// @brief 原地删除singleton维度。
YTensorBase& squeeze_(YTensorBase& tensor, int dim = -1);
/// @brief 使用zero stride扩展singleton维度并返回共享storage的view。
YTensorBase repeat(const YTensorBase& tensor, const std::vector<int>& times);
/// @brief 原地使用zero stride扩展singleton维度。
YTensorBase& repeat_(YTensorBase& tensor, const std::vector<int>& times);
/// @brief 插入滑动窗口维度并返回共享storage的view。
YTensorBase unfold(const YTensorBase& tensor, int atDim, int kernel, int stride = 1, int dilation = 1);
/// @brief 原地插入滑动窗口维度。
YTensorBase& unfold_(YTensorBase& tensor, int atDim, int kernel, int stride = 1, int dilation = 1);
/// @brief 按物理stride重排维度，并将负stride归一化为正stride view。
YTensorBase mostContinuousView(const YTensorBase& tensor);
/// @brief 将逻辑坐标转换为连续逻辑索引。
size_t logicalIndex(const YTensorBase& tensor, const std::vector<int>& position);
/// @brief 将逻辑坐标转换为相对当前view起点的有符号物理偏移。
int relativeOffset(const YTensorBase& tensor, const std::vector<int>& position);
/// @brief 将连续逻辑索引转换为相对data指针的有符号物理偏移。
std::ptrdiff_t relativeOffset(const YTensorBase& tensor, size_t logicalIndex);
/// @brief 将逻辑坐标转换为相对storage起点的绝对物理偏移。
int storageOffset(const YTensorBase& tensor, const std::vector<int>& position);
/// @brief 将逻辑坐标转换为相对data指针的物理索引。
size_t physicalIndex(const YTensorBase& tensor, const std::vector<int>& position);
/// @brief 将连续逻辑索引转换为相对data指针的物理索引。
size_t physicalIndex(const YTensorBase& tensor, size_t logicalIndex);
/// @brief 将连续逻辑索引转换为storage起点的绝对元素索引，并验证storage边界。
size_t storageIndex(const YTensorBase& tensor, size_t logicalIndex);
/// @brief 将连续逻辑索引转换为逻辑坐标。
std::vector<int> coordinate(const YTensorBase& tensor, size_t index);
/// @brief 按给定长度拆分axis，返回共享原storage的slice views。
std::vector<YTensorBase> split(const YTensorBase& tensor, const std::vector<int>& splitSizes, int axis);
/// @brief 将axis等分为指定份数，返回共享原storage的slice views。
std::vector<YTensorBase> split(const YTensorBase& tensor, int parts, int axis);

/// @brief 将最后两个维度包装为runtime二维矩阵view元素。
/// @note 外层 tensor 拥有 YTensorBase wrapper 对象；每个 wrapper 与输入 tensor 共享 values storage。
YTensorBase matView(const YTensorBase& tensor);

/// @brief typed版本的矩阵view，保留元素类型和编译期batch维度。
/// @note 外层 tensor 管理 wrapper 生命周期，内层矩阵 view 与输入 tensor 共享 values storage。
template <typename T, int dim>
YTensor<YTensor<T, 2>, std::max(1, dim - 2)> matView(const YTensor<T, dim>& tensor);

#if YT_USE_EIGEN
/// @brief 为batch tensor创建Eigen矩阵view张量。
/// @warning Eigen::Map 只借用输入 storage 的裸指针；使用返回值期间输入 storage 必须保持存活且不得被替换。
template <typename T, int dim>
auto matViewEigen(const YTensor<T, dim>& tensor) requires(dim > 2);

/// @brief 为一维或二维tensor创建单个Eigen矩阵view。
/// @warning Eigen::Map 只借用输入 storage 的裸指针；使用返回值期间输入 storage 必须保持存活且不得被替换。
template <typename T, int dim>
auto matViewEigen(const YTensor<T, dim>& tensor) requires(dim <= 2);
#endif

}  // namespace yt::strided
