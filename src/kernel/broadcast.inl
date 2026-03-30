#pragma once
/***************
 * @file broadcast.inl
 * @brief 广播操作函数实现
 ***************/

namespace yt::kernel {

template<typename... Args>
constexpr size_t countTensors() {
    if constexpr (sizeof...(Args) == 0) {
        return 0;
    } else {
        return (static_cast<size_t>(::yt::traits::is_ytensor_v<Args>) + ...);
    }
}

template<size_t N, size_t I>
template<typename StridesArray, typename IndicesArray>
inline void IndexAccumulator<N, I>::accumulate(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
    indices[I] += posi * strides[I][dimIdx];
    if constexpr (I + 1 < N) {
        IndexAccumulator<N, I + 1>::accumulate(indices, posi, strides, dimIdx);
    }
}

template<size_t N, int Dim>
template<typename LogicStrideArray, typename ShapeArray, typename StridesArray>
inline std::array<int, N> NaryIndexComputer<N, Dim>::compute(
    int index,
    const LogicStrideArray& logicStride,
    const ShapeArray& shape,
    const StridesArray& strides)
{
    std::array<int, N> indices = {};
    for (int d = 0; d < Dim; ++d) {
        int posi = (index / logicStride[d]) % shape[d];
        accumulateAll<0>(indices, posi, strides, d);
    }
    return indices;
}

template<size_t N, int Dim>
template<typename LogicStrideArray, typename ShapeArray, typename ThisStrideArray, typename StridesArray>
inline std::array<int, N> NaryIndexComputer<N, Dim>::computeWithThis(
    int index,
    const LogicStrideArray& logicStride,
    const ShapeArray& shape,
    const ThisStrideArray& thisStride,
    const StridesArray& strides,
    int& thisDataIdx)
{
    if constexpr (N == 0) {
        thisDataIdx = 0;
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            thisDataIdx += posi * thisStride[d];
        }
        return {};
    }
    std::array<int, N> indices = {};
    thisDataIdx = 0;
    for (int d = 0; d < Dim; ++d) {
        int posi = (index / logicStride[d]) % shape[d];
        thisDataIdx += posi * thisStride[d];
        accumulateAll<0>(indices, posi, strides, d);
    }
    return indices;
}

template<size_t N, int Dim>
template<size_t I, typename IndicesArray, typename StridesArray>
inline void NaryIndexComputer<N, Dim>::accumulateAll(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
    indices[I] += posi * strides[I][dimIdx];
    if constexpr (I + 1 < N) {
        accumulateAll<I + 1>(indices, posi, strides, dimIdx);
    }
}

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

inline std::vector<int> computeBroadcastShape(const std::vector<std::vector<int>>& shapes) {
    if (shapes.empty()) return {};
    
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

inline std::vector<int> getBroadcastStride(const std::vector<int>& shape, 
                                           const std::vector<int>& stride,
                                           const std::vector<int>& broadcastShape) {
    size_t offset = broadcastShape.size() - shape.size();
    std::vector<int> result(broadcastShape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == broadcastShape[i + offset]) {
            result[i + offset] = stride[i];
        }
    }
    return result;
}

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

template <int _resultDim, typename Func, typename... Args>
auto broadcast(Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    
    constexpr int computedDim = max_dim<Args...>();
    constexpr int resultDim = (_resultDim > 0) ? _resultDim : (computedDim > 0 ? computedDim : 1);
    static_assert(_resultDim > 0 || all_ytensor_template<Args...>(), "broadcast: when using YTensorBase, you must explicitly specify resultDim, ");
    
    using ScalarType = ::yt::traits::first_arg_of_t<Func>;
    
    static_assert(all_valid_broadcast_args<ScalarType, Args...>(), "broadcast: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
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

    auto broadcastShape = computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());
    
    if constexpr (_resultDim > 0) {
        if (opdim != _resultDim) {
            throw std::runtime_error("broadcast: specified resultDim (" + std::to_string(_resultDim) + 
                ") does not match actual broadcast dimension (" + std::to_string(opdim) + ")");
        }
    }
    
    std::vector<int> logicStride(opdim);
    int stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = stride;
        stride *= broadcastShape[i];
    }
    int totalSize = stride;
    
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarType*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                dataPtrs.push_back(arg.data());
                allContiguous = allContiguous && arg.isContiguous();
            } else {
                dataPtrs.push_back(arg.template data<ScalarType>());
                allContiguous = allContiguous && arg.isContiguous();
            }
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
    
    using ReturnType = std::invoke_result_t<Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarType, Args>>())...>;
    
    yt::YTensor<ReturnType, resultDim> result;
    
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
    
    if (allContiguous && allShapeEqual) {
        ReturnType* resultPtr = result.data_();
        
        parallelFor(0, totalSize, [&](int index) {
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueDirect = [&](auto&& arg) -> ScalarType {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<ScalarType>(arg);
                }
            };
            
            ScalarType values[] = {getValueDirect(tensors)...};
            
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                resultPtr[index] = func(values[Is]...);
            }(std::make_index_sequence<sizeof...(Args)>{});
        });
        return result;
    }
    
    ReturnType* resultPtr = result.data_();
    constexpr size_t numTensors = countTensors<Args...>();
    
    std::array<std::array<int, resultDim>, numTensors> tensorStrides;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int i = 0; i < resultDim; ++i) {
            tensorStrides[t][i] = (i < static_cast<int>(broadcastStrides[t].size())) ? broadcastStrides[t][i] : 0;
        }
    }
    
    std::array<int, resultDim> resultShape;
    std::array<int, resultDim> resultLogicStride;
    auto resShape = result.shape();
    for (int i = 0; i < resultDim; ++i) {
        resultShape[i] = resShape[i];
    }
    stride = 1;
    for (int i = resultDim - 1; i >= 0; --i) {
        resultLogicStride[i] = stride;
        stride *= resultShape[i];
    }
    
    parallelFor(0, totalSize, [&](int index) {
        auto indices = NaryIndexComputer<numTensors, resultDim>::compute(
            index, resultLogicStride, resultShape, tensorStrides);
        
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto getValue = [&](auto&& arg) -> ScalarType {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][indices[idx]];
            } else {
                return static_cast<ScalarType>(arg);
            }
        };
        
        ScalarType values[] = {getValue(tensors)...};
        
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            resultPtr[index] = func(values[Is]...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    });
    
    return result;
}

template <typename TensorType, typename Func, typename... Args>
TensorType& broadcastInplace(TensorType& target, Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    using T = typename TensorType::scalarType;
    constexpr int dim = TensorType::ndim;
    
    static_assert(all_valid_broadcast_args<T, Args...>(), "broadcastInplace: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
    std::vector<std::vector<int>> shapes;
    shapes.push_back(target.shape());
    
    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);
    
    auto broadcastShape = computeBroadcastShape(shapes);
    
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
        T* targetDataPtr = target.data();
        
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
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueFast = [&](auto&& arg) -> T {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<T>(arg);
                }
            };
            
            T values[] = {getValueFast(tensors)...};
            
            [&]<size_t... Is>(std::index_sequence<Is...>) {
                func(targetDataPtr[index], values[Is]...);
            }(std::make_index_sequence<sizeof...(Args)>{});
        });
        
        return target;
    }
    
    auto logicStride = target.stride();
    
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const T*> dataPtrs;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            if constexpr (is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                dataPtrs.push_back(arg.data());
            } else {
                dataPtrs.push_back(arg.template data<T>());
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);
    
    constexpr size_t numTensors = countTensors<Args...>();
    
    std::array<std::array<int, dim>, numTensors> strideArrays;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int d = 0; d < dim; ++d) {
            strideArrays[t][d] = broadcastStrides[t][d];
        }
    }
    
    std::array<int, dim> targetShape, targetStride;
    auto targetShapeV = target.shape();
    auto targetStrideV = target.stride_();
    for (int d = 0; d < dim; ++d) {
        targetShape[d] = targetShapeV[d];
        targetStride[d] = targetStrideV[d];
    }
    
    parallelFor(0, totalSize, [&](int index) {
        int targetDataIdx = 0;
        auto tensorIndices = NaryIndexComputer<numTensors, dim>::computeWithThis(
            index, logicStride, targetShape, targetStride, strideArrays, targetDataIdx
        );
        
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto getValue = [&](auto&& arg) -> T {
            if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][tensorIndices[idx]];
            } else {
                return static_cast<T>(arg);
            }
        };
        
        T values[] = {getValue(tensors)...};
        
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            func(target.atData_(targetDataIdx), values[Is]...);
        }(std::make_index_sequence<sizeof...(Args)>{});
    });
    
    return target;
}

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

template <typename Func, typename... Args>
yt::YTensorBase broadcastBase(Func&& func, Args&&... tensors) {
    using namespace ::yt::traits;
    
    using ScalarT = ::yt::traits::first_arg_of_t<Func>;
    
    static_assert(all_valid_broadcast_args<ScalarT, Args...>(), "broadcastBase: all arguments must be either YTensor/YTensorBase or convertible to scalar type");
    
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

    auto broadcastShape = computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());
    
    std::vector<int> logicStride(opdim);
    int stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = stride;
        stride *= broadcastShape[i];
    }
    int totalSize = stride;
    
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarT*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                dataPtrs.push_back(arg.data());
                allContiguous = allContiguous && arg.isContiguous();
            } else {
                dataPtrs.push_back(arg.template data<ScalarT>());
                allContiguous = allContiguous && arg.isContiguous();
            }
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
    
    using ReturnType = std::invoke_result_t<Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarT, Args>>())...>;
    
    std::string resultDtype = yt::types::getTypeName<ReturnType>();
    
    yt::YTensorBase result(broadcastShape, resultDtype);
    ReturnType* resultPtr = result.template data<ReturnType>();
    
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
    
    size_t numTensors = dataPtrs.size();
    
    parallelFor(0, totalSize, [&](int index) {
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
