#pragma once
/***************
 * @file broadcast.hpp
 * @brief 广播操作函数声明
 ***************/

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../utils/parallel_for.hpp"
#include "../ytensor_concepts.hpp"
#include "../ytensor_types.hpp"

namespace yt::strided {

/// @brief 计算多个shape统一广播后的结果shape。
/// @throws std::runtime_error 输入shape无法广播时抛出。
std::vector<int> computeBroadcastShape(const std::vector<std::vector<int>>& shapes);

/// @brief 编译期计算参数包中张量类型的数量
/// @tparam Args 参数类型包
/// @return 张量参数的数量
template <typename... Args>
constexpr size_t countTensors();

/// @brief 使用模板递归在编译期展开N个索引的累加
/// @tparam N 张量数量
/// @tparam I 当前处理的张量索引
template <size_t N, size_t I = 0>
struct IndexAccumulator {
    template <typename StridesArray, typename IndicesArray>
    static inline void accumulate(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx);
};

/// @brief 编译期展开的N元索引计算器（使用编译期dim）
/// @tparam N 张量数量
/// @tparam Dim 维度（编译期常量）
template <size_t N, int Dim>
struct NaryIndexComputer {
    /// @brief 计算N个张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param strides N个张量的stride数组，每个是Dim维的数组
    /// @return 包含N个数据索引的数组
    template <typename LogicStrideArray, typename ShapeArray, typename StridesArray>
    static inline std::array<int, N> compute(
        int index, const LogicStrideArray& logicStride, const ShapeArray& shape, const StridesArray& strides
    );

    /// @brief 同时计算this和N个其他张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param thisStride this的stride数组
    /// @param strides N个张量的stride数组
    /// @param thisDataIdx 输出：this的数据索引
    /// @return 包含N个数据索引的数组
    template <typename LogicStrideArray, typename ShapeArray, typename ThisStrideArray, typename StridesArray>
    static inline std::array<int, N> computeWithThis(
        int index, const LogicStrideArray& logicStride, const ShapeArray& shape,
        const ThisStrideArray& thisStride, const StridesArray& strides, int& thisDataIdx
    );

   private:
    template <size_t I, typename IndicesArray, typename StridesArray>
    static inline void accumulateAll(
        IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx
    );
};

/// @brief 计算广播索引（运行时ndim版本）
/// @param index 线性索引
/// @param logicStride 逻辑stride
/// @param shape 广播shape
/// @param thisStride this张量的stride
/// @param otherStrides 其他张量的stride数组
/// @param thisIdx 输出：this的数据索引
/// @param otherIndices 输出：其他张量的数据索引
/// @param ndim 维度数
void computeBroadcastIndicesRuntime(
    int index, const std::vector<int>& logicStride, const std::vector<int>& shape,
    const std::vector<int>& thisStride, const std::vector<std::vector<int>>& otherStrides, int& thisIdx,
    std::vector<int>& otherIndices, int ndim
);

/// @brief 计算N个张量的广播索引（编译期展开）
/// @tparam N 张量数量
/// @param index 线性索引
/// @param logicStride 逻辑stride
/// @param broadcastShape 广播shape
/// @param strides 每个张量的广播stride数组
/// @param opdim 操作维度
/// @return 包含N个数据索引的数组
template <size_t N>
std::array<int, N> computeBroadcastIndices(
    int index, const std::vector<int>& logicStride, const std::vector<int>& broadcastShape,
    const std::array<const int*, N>& strides, int opdim
);

/// @brief 获取张量在广播shape下的stride
/// @param shape 原始张量的shape
/// @param stride 原始张量的stride
/// @param broadcastShape 广播后的shape
/// @return 广播stride（对于被广播的维度，stride为0）
std::vector<int> getBroadcastStride(
    const std::vector<int>& shape, const std::vector<int>& stride, const std::vector<int>& broadcastShape
);

/// @brief 计算张量在给定索引处的实际数据索引
/// @param linearIndex 线性索引
/// @param logicStride 逻辑stride（连续存储）
/// @param broadcastStride 广播stride
/// @param broadcastShape 广播shape
/// @return 实际数据索引
int computeDataIndex(
    int linearIndex, const std::vector<int>& logicStride, const std::vector<int>& broadcastStride,
    const std::vector<int>& broadcastShape
);

/// @brief 统一的广播操作函数（非原地），支持N元张量/标量操作
/// @tparam resultDim 结果张量维度。当所有张量参数不包含YTensorBase时可省略（自动推导）；
/// @tparam Func 函数类型，签名为 ReturnType func(const T&, const T&, ...) 或 ReturnType func(T, T, ...)
/// @tparam Args 参数类型，可以是YTensor、YTensorBase或标量（可转换为func参数类型）
/// @param func 操作函数，返回类型用于推断结果张量的标量类型
/// @param tensors 输入的张量或标量
/// @return 返回结果张量 YTensor<ReturnType, resultDim>，形状为所有输入张量广播后的形状
template <int _resultDim = -1, typename Func, typename... Args>
auto broadcast(Func&& func, Args&&... tensors);

/// @brief 统一的广播原地操作函数，支持N元张量/标量操作
/// @tparam TensorType 目标张量类型（YTensor<T, dim>）
/// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
/// @tparam Args 参数类型，可以是YTensor或标量（可转换为func参数类型）
/// @param target 目标张量（将被原地修改）
/// @param func 操作函数，第一个参数为target的元素引用
/// @param tensors 输入的张量或标量
/// @return 返回target的引用
template <typename TensorType, typename Func, typename... Args>
TensorType& broadcastInplace(TensorType& target, Func&& func, Args&&... tensors);

/// @brief 运行时broadcast：返回YTensorBase的广播函数
/// @details 当输入参数中包含YTensorBase（非模板YTensor）时使用此版本，
///          因为输出维度在编译期无法确定，必须返回YTensorBase
/// @tparam ScalarT 标量类型（从func推断）
/// @tparam Func 函数类型
/// @tparam Args 参数类型（可以是YTensor/YTensorBase或可转换为ScalarT的值）
/// @param func 操作函数
/// @param tensors 输入的张量或标量
/// @return YTensorBase结果
/// @note 参数Args中每个类型必须是YTensor/YTensorBase或可转换为标量类型的值
template <typename Func, typename... Args>
yt::YTensorBase broadcastBase(Func&& func, Args&&... tensors);

/// @brief 对 runtime target 执行 callable 驱动的原地广播。
/// @details target 必须具有广播结果 shape；与 target 物理跨度重叠的 tensor 参数会先快照，
///          防止原地写入污染后续读取。
/// @tparam Func 首个参数为目标元素引用的 callable 类型
/// @tparam Args YTensor/YTensorBase 或可转换为 callable 标量类型的参数
/// @return target 的引用
template <typename Func, typename... Args>
yt::YTensorBase& broadcastInplaceBase(yt::YTensorBase& target, Func&& func, Args&&... tensors);

/// @brief 通过统一runtime broadcast kernel计算结果张量。
/// @param op broadcast运算类型。
/// @param inputs 参与广播的输入张量，dtype必须一致。
/// @param outputDtype 输出张量dtype；比较操作使用bool。
YTensorBase broadcast(
    yt::type::YBroadcastOp op, const std::vector<const YTensorBase*>& inputs, const std::string& outputDtype
);

/// @brief 将统一runtime broadcast结果写入已有输出张量（原地变体）。
/// @note output的shape必须与inputs广播结果一致。
/// @details 通过 YDTypeKernels::broadcastInplace 派发到具体dtype实现。
YTensorBase& broadcast_(
    yt::type::YBroadcastOp op, YTensorBase& output, const std::vector<const YTensorBase*>& inputs
);

/// @brief 将标量转换成与目标张量dtype一致的rank-0张量。
/// @tparam Scalar 标量类型
/// @param tensor 用于参考dtype的目标张量
/// @param scalar 输入标量
/// @return rank-0 YTensorBase，dtype与tensor一致
template <typename Scalar>
YTensorBase scalarTensor(const YTensorBase& tensor, const Scalar& scalar);

/// @brief 对 typed tensor 的每个逻辑元素调用 func，并原地保存结果。
/// @details func 可接收 `T&`，或接收 `T&` 与逻辑坐标；重叠 view 使用确定性的串行写入。
/// @param flop 非零时允许对 disjoint tensor 启用并行执行
template <typename T, int dim, typename Func>
YTensor<T, dim>& forEach(YTensor<T, dim>& tensor, Func&& func, double flop);

/// @brief 将 typed tensor 的每个逻辑元素原地设置为 value。
/// @note 对重叠 view 只遍历其逻辑映射，不假设底层 storage disjoint。
template <typename T, int dim>
YTensor<T, dim>& fill(YTensor<T, dim>& tensor, const T& value);

}  // namespace yt::strided
