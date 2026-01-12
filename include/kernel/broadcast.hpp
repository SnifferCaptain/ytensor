#pragma once
#include <vector>
#include <stdexcept>
#include <array>
#include <utility>
#include <tuple>
#include <type_traits>
#include "../ytensor_concepts.hpp"
#include "../ytensor_types.hpp"
#include "parallel_for.hpp"

namespace yt::kernel {
/// @brief 编译期计算参数包中张量类型的数量
/// @tparam Args 参数类型包
/// @return 张量参数的数量
template<typename... Args>
constexpr size_t countTensors() {
    if constexpr (sizeof...(Args) == 0) {
        return 0;
    } else {
        return (static_cast<size_t>(::yt::traits::is_ytensor_v<Args>) + ...);
    }
}

/// @brief 使用模板递归在编译期展开N个索引的累加
/// @tparam N 张量数量
/// @tparam I 当前处理的张量索引
template<size_t N, size_t I = 0>
struct IndexAccumulator {
    template<typename StridesArray, typename IndicesArray>
    static inline void accumulate(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
        indices[I] += posi * strides[I][dimIdx];
        if constexpr (I + 1 < N) {
            IndexAccumulator<N, I + 1>::accumulate(indices, posi, strides, dimIdx);
        }
    }
};

/// @brief 编译期展开的N元索引计算器（使用编译期dim）
/// @tparam N 张量数量
/// @tparam Dim 维度（编译期常量）
template<size_t N, int Dim>
struct NaryIndexComputer {
    /// @brief 计算N个张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param strides N个张量的stride数组，每个是Dim维的数组
    /// @return 包含N个数据索引的数组
    template<typename LogicStrideArray, typename ShapeArray, typename StridesArray>
    static inline std::array<int, N> compute(
        int index,
        const LogicStrideArray& logicStride,
        const ShapeArray& shape,
        const StridesArray& strides)
    {
        std::array<int, N> indices = {};
        // 外层循环使用编译期常量Dim
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            // 内层循环使用编译期常量N，通过递归展开
            accumulateAll<0>(indices, posi, strides, d);
        }
        return indices;
    }
    
    /// @brief 同时计算this和N个其他张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param thisStride this的stride数组
    /// @param strides N个张量的stride数组
    /// @param thisDataIdx 输出：this的数据索引
    /// @return 包含N个数据索引的数组
    template<typename LogicStrideArray, typename ShapeArray, typename ThisStrideArray, typename StridesArray>
    static inline std::array<int, N> computeWithThis(
        int index,
        const LogicStrideArray& logicStride,
        const ShapeArray& shape,
        const ThisStrideArray& thisStride,
        const StridesArray& strides,
        int& thisDataIdx)
    {
        if constexpr (N == 0) {
            // 原地transform，返回空即可。一般来说不会到这里来
            return {};
        }
        std::array<int, N> indices = {};
        thisDataIdx = 0;
        // 外层循环使用编译期常量Dim，计算一次posi同时更新所有索引
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            thisDataIdx += posi * thisStride[d];
            // 内层循环使用编译期常量N，通过递归展开
            accumulateAll<0>(indices, posi, strides, d);
        }
        return indices;
    }
    
private:
    template<size_t I, typename IndicesArray, typename StridesArray>
    static inline void accumulateAll(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
        indices[I] += posi * strides[I][dimIdx];
        if constexpr (I + 1 < N) {
            accumulateAll<I + 1>(indices, posi, strides, dimIdx);
        }
    }
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
inline void computeBroadcastIndicesRuntime(
    int index,
    const std::vector<int>& logicStride,
    const std::vector<int>& shape,
    const std::vector<int>& thisStride,
    const std::vector<std::vector<int>>& otherStrides,
    int& thisIdx,
    std::vector<int>& otherIndices,
    int ndim)
{
    thisIdx = 0;
    std::fill(otherIndices.begin(), otherIndices.end(), 0);
    for (int i = 0; i < ndim; ++i) {
        int posi = (index / logicStride[i]) % shape[i];
        thisIdx += posi * thisStride[i];
        for (size_t t = 0; t < otherStrides.size(); ++t) {
            otherIndices[t] += posi * otherStrides[t][i];
        }
    }
}

/// @brief 计算N个张量的广播索引（编译期展开）
/// @tparam N 张量数量
/// @param index 线性索引
/// @param logicStride 逻辑stride
/// @param broadcastShape 广播shape
/// @param strides 每个张量的广播stride数组
/// @param opdim 操作维度
/// @return 包含N个数据索引的数组
template<size_t N>
inline std::array<int, N> computeBroadcastIndices(
    int index,
    const std::vector<int>& logicStride,
    const std::vector<int>& broadcastShape,
    const std::array<const int*, N>& strides,
    int opdim)
{
    std::array<int, N> indices = {};
    for (int i = 0; i < opdim; ++i) {
        int posi = (index / logicStride[i]) % broadcastShape[i];
        IndexAccumulator<N>::accumulate(indices, posi, strides, i);
    }
    return indices;
}

// ==================== 原有函数 ====================

/// @brief 计算多个张量的广播shape
/// @param shapes 所有参与广播的张量的shape列表
/// @return 广播后的shape
/// @throw std::runtime_error 如果shapes无法广播
inline std::vector<int> computeBroadcastShape(const std::vector<std::vector<int>>& shapes) {
    if (shapes.empty()) return {};
    
    // 找最大维度
    size_t maxDim = 0;
    for (const auto& s : shapes) {
        maxDim = std::max(maxDim, s.size());
    }
    
    std::vector<int> result(maxDim, 1);
    for (const auto& shape : shapes) {
        size_t offset = maxDim - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t ri = i + offset;
            if (result[ri] == 1) {
                result[ri] = shape[i];
            } else if (shape[i] != 1 && shape[i] != result[ri]) {
                throw std::runtime_error("Broadcast: shapes cannot be broadcast together");
            }
        }
    }
    return result;
}

/// @brief 获取张量在广播shape下的stride
/// @param shape 原始张量的shape
/// @param stride 原始张量的stride
/// @param broadcastShape 广播后的shape
/// @return 广播stride（对于被广播的维度，stride为0）
inline std::vector<int> getBroadcastStride(const std::vector<int>& shape, 
                                           const std::vector<int>& stride,
                                           const std::vector<int>& broadcastShape) {
    size_t offset = broadcastShape.size() - shape.size();
    std::vector<int> result(broadcastShape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == broadcastShape[i + offset]) {
            result[i + offset] = stride[i];
        }
        // else: shape[i] == 1, stride stays 0 (broadcast)
    }
    return result;
}

/// @brief 计算张量在给定索引处的实际数据索引
/// @param linearIndex 线性索引
/// @param logicStride 逻辑stride（连续存储）
/// @param broadcastStride 广播stride
/// @param broadcastShape 广播shape
/// @return 实际数据索引
inline int computeDataIndex(int linearIndex, 
                            const std::vector<int>& logicStride,
                            const std::vector<int>& broadcastStride,
                            const std::vector<int>& broadcastShape) {
    int dataIndex = 0;
    int opdim = static_cast<int>(broadcastShape.size());
    #pragma omp simd reduction(+:dataIndex)
    for (int i = 0; i < opdim; ++i) {
        int posi = (linearIndex / logicStride[i]) % broadcastShape[i];
        dataIndex += posi * broadcastStride[i];
    }
    return dataIndex;
}

/// @brief 统一的广播操作函数（非原地），支持N元张量/标量操作
/// @tparam Func 函数类型，签名为 ReturnType func(const T&, const T&, ...) 或 ReturnType func(T, T, ...)
/// @tparam Args 参数类型，可以是YTensor或标量（可转换为func参数类型）
/// @param func 操作函数，返回类型用于推断结果张量的标量类型
/// @param tensors 输入的张量或标量
/// @return 返回结果张量，形状为所有输入张量广播后的形状，返回类型由func的返回值推断
template <typename Func, typename... Args>
auto broadcast(Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    
    using ScalarType = ::yt::traits::first_arg_of_t<Func>;
    
    // 编译时类型检查：确保所有参数要么是张量类型，要么可转换为标量类型
    static_assert(all_valid_broadcast_args<ScalarType, Args...>(), "broadcast: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
    // 收集所有张量的shape
    std::vector<std::vector<int>> shapes;
    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);
    
    if (shapes.empty()) {
        throw std::runtime_error("broadcast: at least one tensor argument required");
    }

    // 计算广播shape
    auto broadcastShape = computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());
    
    // 计算逻辑stride（连续存储）
    std::vector<int> logicStride(opdim);
    int stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = stride;
        stride *= broadcastShape[i];
    }
    int totalSize = stride;
    
    // 收集每个张量参数的广播stride
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarType*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            // 区分YTensor<U, d>和YTensorBase
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                dataPtrs.push_back(arg.data());
                allContiguous = allContiguous && arg.isContiguous();
            } else {
                dataPtrs.push_back(arg.template data<ScalarType>());
                allContiguous = allContiguous && arg.isContiguous();
            }
            // 检查shape是否与broadcastShape相同
            auto argShape = arg.shape();
            if (argShape.size() != broadcastShape.size()) {
                allShapeEqual = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != broadcastShape[i]) {
                        allShapeEqual = false;
                        break;
                    }
                }
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);
    
    // 推断返回类型
    using ReturnType = std::invoke_result_t<Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarType, Args>>())...>;
    
    // 创建结果张量（使用最大维度）
    constexpr int resultDim = max_dim<Args...>() > 0 ? max_dim<Args...>() : 1;
    yt::YTensor<ReturnType, resultDim> result;
    
    // 根据broadcastShape的实际维度调整
    if (opdim == resultDim) {
        result.reserve(broadcastShape);
    } else if (opdim < resultDim) {
        std::vector<int> paddedShape(resultDim - opdim, 1);
        paddedShape.insert(paddedShape.end(), broadcastShape.begin(), broadcastShape.end());
        result.reserve(paddedShape);
    } else {
        std::vector<int> trimmedShape(broadcastShape.end() - resultDim, broadcastShape.end());
        result.reserve(trimmedShape);
    }
    
    // Fastpath: 所有张量都是连续的且shape相同
    if (allContiguous && allShapeEqual) {
        ReturnType* resultPtr = result.data_();
        
        parallelFor(0, totalSize, [&](int index) {
            // 使用初始化列表强制从左到右求值顺序
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueDirect = [&](auto&& arg) -> ScalarType {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<ScalarType>(arg);
                }
            };
            
            // 使用初始化列表的大括号初始化来强制求值顺序
            // C++ 标准保证大括号初始化列表中的元素从左到右求值
            ScalarType values[] = {getValueDirect(tensors)...};
            
            // 使用 std::index_sequence 来调用 func
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                resultPtr[index] = func(values[Is]...);
            }(std::make_index_sequence<sizeof...(Args)>{});
        });
        return result;
    }
    
    // Slowpath: 需要广播的情况
    ReturnType* resultPtr = result.data_();
    constexpr size_t numTensors = countTensors<Args...>();
    
    // 使用编译期dim优化 - 提取所有必要的std::array
    std::array<std::array<int, resultDim>, numTensors> tensorStrides;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int i = 0; i < resultDim; ++i) {
            tensorStrides[t][i] = (i < static_cast<int>(broadcastStrides[t].size())) ? broadcastStrides[t][i] : 0;
        }
    }
    
    // 提取结果shape和stride为std::array用于编译期循环
    std::array<int, resultDim> resultShape;
    std::array<int, resultDim> resultLogicStride;
    auto resShape = result.shape();
    for (int i = 0; i < resultDim; ++i) {
        resultShape[i] = resShape[i];
    }
    // 计算连续stride
    stride = 1;
    for (int i = resultDim - 1; i >= 0; --i) {
        resultLogicStride[i] = stride;
        stride *= resultShape[i];
    }
    
    parallelFor(0, totalSize, [&](int index) {
        // 使用NaryIndexComputer进行N元索引计算
        auto indices = NaryIndexComputer<numTensors, resultDim>::compute(
            index, resultLogicStride, resultShape, tensorStrides);
        
        // 使用初始化列表强制从左到右求值顺序
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto getValue = [&](auto&& arg) -> ScalarType {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][indices[idx]];
            } else {
                return static_cast<ScalarType>(arg);
            }
        };
        
        // 使用初始化列表的大括号初始化来强制求值顺序
        ScalarType values[] = {getValue(tensors)...};
        
        // 使用 std::index_sequence 来调用 func
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            resultPtr[index] = func(values[Is]...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    });
    
    return result;
}

/// @brief 统一的广播原地操作函数，支持N元张量/标量操作
/// @tparam TensorType 目标张量类型（YTensor<T, dim>）
/// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
/// @tparam Args 参数类型，可以是YTensor或标量（可转换为func参数类型）
/// @param target 目标张量（将被原地修改）
/// @param func 操作函数，第一个参数为target的元素引用
/// @param tensors 输入的张量或标量
/// @return 返回target的引用
template <typename TensorType, typename Func, typename... Args>
TensorType& broadcastInplace(TensorType& target, Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    using T = typename TensorType::scalarType;
    constexpr int dim = TensorType::ndim;
    
    // 编译时类型检查：确保所有参数要么是张量类型，要么可转换为标量类型
    static_assert(all_valid_broadcast_args<T, Args...>(), "broadcastInplace: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
    // 收集所有张量的shape（包括target）
    std::vector<std::vector<int>> shapes;
    shapes.push_back(target.shape());
    
    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);
    
    // 计算广播shape
    auto broadcastShape = computeBroadcastShape(shapes);
    
    // 验证target的shape与广播shape兼容（inplace要求target的shape必须等于广播shape）
    auto targetShapeVec = target.shape();
    if (static_cast<int>(broadcastShape.size()) != dim) {
        throw std::runtime_error("broadcastInplace: result dimension mismatch");
    }
    for (int i = 0; i < dim; ++i) {
        if (targetShapeVec[i] != broadcastShape[i]) {
            throw std::runtime_error("broadcastInplace: target tensor shape must match broadcast shape");
        }
    }
    
    int totalSize = target.size();
    bool allContiguous = target.isContiguous();
    bool allEqualShape = true;
    
    [[maybe_unused]] auto checkContiguousAndShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (!arg.isContiguous()) {
                allContiguous = false;
            }
            auto argShape = arg.shape();
            if (argShape.size() != shapes[0].size()) {
                allEqualShape = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != shapes[0][i]) {
                        allEqualShape = false;
                        break;
                    }
                }
            }
        }
    };
    (checkContiguousAndShape(tensors), ...);
    
    if (allContiguous && allEqualShape) {
        // 所有张量是否连续且shape相同
        T* targetDataPtr = target.data();
        
        // 收集所有张量的数据指针
        std::vector<const T*> dataPtrs;
        [[maybe_unused]] auto collectPtrs = [&](auto&& arg) {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                if constexpr (is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                    dataPtrs.push_back(arg.data());
                } else {
                    dataPtrs.push_back(arg.template data<T>());
                }
            }
        };
        (collectPtrs(tensors), ...);
        
        parallelFor(0, totalSize, [&](int index) {
            // 使用初始化列表强制从左到右求值顺序
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueFast = [&](auto&& arg) -> T {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<T>(arg);
                }
            };
            
            // 使用初始化列表的大括号初始化来强制求值顺序
            T values[] = {getValueFast(tensors)...};
            
            // 使用 std::index_sequence 来调用 func
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                func(targetDataPtr[index], values[Is]...);
            }(std::make_index_sequence<sizeof...(Args)>{});
        });
        
        return target;
    }
    
    // 计算逻辑stride
    auto logicStride = target.stride();
    
    // 收集每个张量参数的广播stride
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const T*> dataPtrs;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            // 区分YTensor<U, d>和YTensorBase
            if constexpr (is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                // YTensor
                dataPtrs.push_back(arg.data());
            } else {
                // YTensorBase
                dataPtrs.push_back(arg.template data<T>());
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);
    
    // 使用编译期展开优化多参数内核
    constexpr size_t numTensors = countTensors<Args...>();
    
    // N元通用优化：使用编译期常量dim和numTensors
    // 将所有张量的stride存入编译期大小的数组
    std::array<std::array<int, dim>, numTensors> strideArrays;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int d = 0; d < dim; ++d) {
            strideArrays[t][d] = broadcastStrides[t][d];
        }
    }
    
    // 将target的shape和stride也存入编译期数组
    std::array<int, dim> targetShape, targetStride;
    auto targetShapeV = target.shape();
    auto targetStrideV = target.stride_();
    for (int d = 0; d < dim; ++d) {
        targetShape[d] = targetShapeV[d];
        targetStride[d] = targetStrideV[d];
    }
    
    parallelFor(0, totalSize, [&](int index) {
        // 使用NaryIndexComputer同时计算target和所有其他张量的索引
        int targetDataIdx = 0;
        auto tensorIndices = NaryIndexComputer<numTensors, dim>::computeWithThis(
            index, logicStride, targetShape, targetStride, strideArrays, targetDataIdx
        );
        
        // 使用初始化列表强制从左到右求值顺序
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto getValue = [&](auto&& arg) -> T {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][tensorIndices[idx]];
            } else {
                return static_cast<T>(arg);
            }
        };
        
        // 使用初始化列表的大括号初始化来强制求值顺序
        T values[] = {getValue(tensors)...};
        
        // 使用 std::index_sequence 来调用 func
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            func(target.atData_(targetDataIdx), values[Is]...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    });
    
    return target;
}

/// @brief 编译期版本：判断底层存储的某个位置是否属于当前view
/// @tparam Dim 维度数
/// @param delta 相对于view起始offset的偏移量
/// @param shape view的shape数组
/// @param stride view的stride数组
/// @return 如果该位置属于view返回true，否则返回false
template<int Dim>
inline bool isPositionInView(int delta, const std::array<int, Dim>& shape, const std::array<int, Dim>& stride) {
    for (int b = 0; b < Dim; ++b) {
        if (shape[b] == 1) {
            continue;
        } else if (stride[b] != 0) {
            int step = delta / stride[b];
            if (step < 0 || step >= shape[b]) {
                return false;
            }
            delta -= step * stride[b];
        }
    }
    return (delta == 0);
}

/// @brief 运行时版本：判断底层存储的某个位置是否属于当前view
/// @param delta 相对于view起始offset的偏移量
/// @param shape view的shape
/// @param stride view的stride
/// @return 如果该位置属于view返回true，否则返回false
inline bool isPositionInViewRuntime(int delta, const std::vector<int>& shape, const std::vector<int>& stride) {
    int ndim = static_cast<int>(shape.size());
    for (int b = 0; b < ndim; ++b) {
        if (shape[b] == 1) {
            continue;
        } else if (stride[b] != 0) {
            int step = delta / stride[b];
            if (step < 0 || step >= shape[b]) {
                return false;
            }
            delta -= step * stride[b];
        }
    }
    return (delta == 0);
}

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
yt::YTensorBase broadcastBase(Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    
    // 编译时类型检查：确保所有参数要么是张量类型，要么可转换为标量类型
    using ScalarT = ::yt::traits::first_arg_of_t<Func>;
    
    static_assert(all_valid_broadcast_args<ScalarT, Args...>(), "broadcastBase: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
    // 收集所有张量的shape
    std::vector<std::vector<int>> shapes;
    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);
    
    if (shapes.empty()) {
        throw std::runtime_error("broadcastBase: at least one tensor argument required");
    }

    // 计算广播shape
    auto broadcastShape = computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());
    
    // 计算逻辑stride（连续存储）
    std::vector<int> logicStride(opdim);
    int stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = stride;
        stride *= broadcastShape[i];
    }
    int totalSize = stride;
    
    // 收集每个张量参数的广播stride
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarT*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            // 区分YTensor<U, d>和YTensorBase
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                dataPtrs.push_back(arg.data());
                allContiguous = allContiguous && arg.isContiguous();
            } else {
                dataPtrs.push_back(arg.template data<ScalarT>());
                allContiguous = allContiguous && arg.isContiguous();
            }
            // 检查shape是否与broadcastShape相同
            auto argShape = arg.shape();
            if (argShape.size() != broadcastShape.size()) {
                allShapeEqual = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != broadcastShape[i]) {
                        allShapeEqual = false;
                        break;
                    }
                }
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);
    
    // 推断返回类型
    using ReturnType = std::invoke_result_t<Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarT, Args>>())...>;
    
    // 获取dtype字符串
    std::string resultDtype = yt::types::getTypeName<ReturnType>();
    
    // 创建YTensorBase结果
    yt::YTensorBase result(broadcastShape, resultDtype);
    ReturnType* resultPtr = result.template data<ReturnType>();
    
    // Fastpath: 所有张量都是连续的且shape相同
    if (allContiguous && allShapeEqual) {
        parallelFor(0, totalSize, [&](int index) {
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueDirect = [&](auto&& arg) -> ScalarT {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<ScalarT>(arg);
                }
            };
            
            ScalarT values[] = {getValueDirect(tensors)...};
            
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                resultPtr[index] = func(values[Is]...);
            }(std::make_index_sequence<sizeof...(Args)>{});
        });
        return result;
    }
    
    // Slowpath: 运行时广播索引计算
    size_t numTensors = dataPtrs.size();
    
    parallelFor(0, totalSize, [&](int index) {
        // 运行时计算每个张量的数据索引
        std::vector<int> tensorIndices(numTensors, 0);
        for (int d = 0; d < opdim; ++d) {
            int posi = (index / logicStride[d]) % broadcastShape[d];
            for (size_t t = 0; t < numTensors; ++t) {
                tensorIndices[t] += posi * broadcastStrides[t][d];
            }
        }
        
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto getValue = [&](auto&& arg) -> ScalarT {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][tensorIndices[idx]];
            } else {
                return static_cast<ScalarT>(arg);
            }
        };
        
        ScalarT values[] = {getValue(tensors)...};
        
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            resultPtr[index] = func(values[Is]...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    });
    
    return result;
}

} // namespace yt::kernel
