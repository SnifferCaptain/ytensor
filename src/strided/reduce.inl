#pragma once
/***************
 * file: strided/reduce.inl
 * purpose: strided layout 的 reduce 职责实现。
 ***************/

#include <algorithm>
#include <limits>

namespace yt::strided {

// ==================== runtime dtype kernels ====================

// 执行单轴keep-dim sum/mean；op选择位于元素循环之外。
// 注意：整数sum使用同宽模加法；empty axis的mean保持零初始化结果，不执行除零。
template <typename DType>
inline void stridedReduceKernel(
    yt::type::YReduceOp op, YTensorBase& out, const YTensorBase& input, int axis
) {
    const DType* inputData = input.data<DType>();
    DType* outputData = out.data<DType>();
    int axisSize = input.shape(axis);
    size_t outSize = out.size();

    switch (op) {
        case yt::type::YReduceOp::Sum:
        case yt::type::YReduceOp::Mean:
            yt::utils::parallelFor(
                0, yt::utils::checkedIntSize(outSize, "strided::sum"),
                [&](int i) {
                    auto coord = out.toCoord(i);
                    DType accum{};
                    for (int j = 0; j < axisSize; ++j) {
                        auto subCoord = coord;
                        subCoord[axis] = j;
                        std::ptrdiff_t physIdx = yt::strided::relativeOffset(input, subCoord);
                        accum = yt::strided::modularAdd(accum, inputData[physIdx]);
                    }
                    if (op == yt::type::YReduceOp::Mean && axisSize != 0) {
                        accum = accum / static_cast<DType>(axisSize);
                    }
                    outputData[i] = accum;
                },
                static_cast<double>(axisSize)
            );
            return;
        default:
            throw std::runtime_error("strided reduce: op not implemented");
    }
}

// 执行单轴keep-dim max并写出axis-local int32索引。
// 注意：`maxOneAxis`已拒绝empty axis；严格`>`比较使相同最大值保留首次出现位置。
template <typename DType>
inline void stridedReduceIndexedKernel(
    yt::type::YReduceOp op, YTensorBase& values, YTensorBase& indices, const YTensorBase& input, int axis
) {
    const DType* inputData = input.data<DType>();
    DType* valueData = values.data<DType>();
    int32_t* indexData = indices.data<int32_t>();
    int axisSize = input.shape(axis);
    size_t outSize = values.size();

    switch (op) {
        case yt::type::YReduceOp::Max:
            yt::utils::parallelFor(
                0, yt::utils::checkedIntSize(outSize, "strided::max"),
                [&](int i) {
                    auto coord = values.toCoord(i);
                    auto firstCoord = coord;
                    firstCoord[axis] = 0;
                    std::ptrdiff_t firstPhysIdx = yt::strided::relativeOffset(input, firstCoord);
                    DType maxVal = inputData[firstPhysIdx];
                    int32_t maxIdx = 0;
                    for (int j = 1; j < axisSize; ++j) {
                        auto subCoord = coord;
                        subCoord[axis] = j;
                        std::ptrdiff_t physIdx = yt::strided::relativeOffset(input, subCoord);
                        DType value = inputData[physIdx];
                        if (value > maxVal) {
                            maxVal = value;
                            maxIdx = j;
                        }
                    }
                    valueData[i] = maxVal;
                    indexData[i] = maxIdx;
                },
                static_cast<double>(axisSize)
            );
            return;
        default:
            throw std::runtime_error("strided indexed reduce: op not implemented");
    }
}

// 将一个builtin dtype的reduce能力合并进kernel table，不覆盖已有非空槽。
inline void registerBuiltinReduceKernel(
    const std::string& dtype, void (*reduce)(yt::type::YReduceOp, YTensorBase&, const YTensorBase&, int),
    void (*reduceIndexed)(yt::type::YReduceOp, YTensorBase&, YTensorBase&, const YTensorBase&, int)
) {
    yt::type::YDTypeKernels kernels;
    kernels.reduce = reduce;
    kernels.reduceIndexed = reduceIndexed;
    yt::type::mergeDTypeKernels(dtype, kernels);
}

// 惰性安装所有builtin reduction kernel；进程内首次调用后不再重复注册。
inline void ensureBuiltinReduceKernels() {
    // local-static保证进程内只初始化一次；merge只填空槽，不覆盖用户预注册kernel。
    static const bool initialized = []() {
        registerBuiltinReduceKernel("float32", &stridedReduceKernel<float>, &stridedReduceIndexedKernel<float>);
        registerBuiltinReduceKernel("float64", &stridedReduceKernel<double>, &stridedReduceIndexedKernel<double>);
        registerBuiltinReduceKernel("int8", &stridedReduceKernel<int8_t>, &stridedReduceIndexedKernel<int8_t>);
        registerBuiltinReduceKernel("int16", &stridedReduceKernel<int16_t>, &stridedReduceIndexedKernel<int16_t>);
        registerBuiltinReduceKernel("int32", &stridedReduceKernel<int32_t>, &stridedReduceIndexedKernel<int32_t>);
        registerBuiltinReduceKernel("int64", &stridedReduceKernel<int64_t>, &stridedReduceIndexedKernel<int64_t>);
        registerBuiltinReduceKernel("uint8", &stridedReduceKernel<uint8_t>, &stridedReduceIndexedKernel<uint8_t>);
        registerBuiltinReduceKernel("uint16", &stridedReduceKernel<uint16_t>, &stridedReduceIndexedKernel<uint16_t>);
        registerBuiltinReduceKernel("uint32", &stridedReduceKernel<uint32_t>, &stridedReduceIndexedKernel<uint32_t>);
        registerBuiltinReduceKernel("uint64", &stridedReduceKernel<uint64_t>, &stridedReduceIndexedKernel<uint64_t>);
        registerBuiltinReduceKernel("bfloat16", &stridedReduceKernel<yt::bfloat16>, &stridedReduceIndexedKernel<yt::bfloat16>);
        registerBuiltinReduceKernel("float16", &stridedReduceKernel<yt::float16>, &stridedReduceIndexedKernel<yt::float16>);
        registerBuiltinReduceKernel("float8_e5m2", &stridedReduceKernel<yt::float8_e5m2>, &stridedReduceIndexedKernel<yt::float8_e5m2>);
        registerBuiltinReduceKernel("float8_e4m3", &stridedReduceKernel<yt::float8_e4m3>, &stridedReduceIndexedKernel<yt::float8_e4m3>);
        registerBuiltinReduceKernel("float8_e8m0", &stridedReduceKernel<yt::float8_e8m0>, &stridedReduceIndexedKernel<yt::float8_e8m0>);
        registerBuiltinReduceKernel("float8_ue8m0", &stridedReduceKernel<yt::float8_ue8m0>, &stridedReduceIndexedKernel<yt::float8_ue8m0>);
        return true;
    }();
    (void)initialized;
}

// runtime单轴reduce编排：规范化axis、保留rank并调用已选dtype kernel。
inline YTensorBase reduceOneAxis(
    const YTensorBase& tensor, int axis, yt::type::YReduceOp op,
    const yt::type::YDTypeKernels& kernels, const std::string& context
) {
    int dim = tensor.ndim();
    if (dim == 0) throw std::runtime_error("[yt::strided::" + context + "] Cannot reduce a 0-dim tensor");
    axis = (axis % dim + dim) % dim;
    auto newShape = tensor.shape();
    newShape[axis] = 1;
    YTensorBase out(newShape, tensor.dtype());
    if (!kernels.reduce) throw std::runtime_error("strided::" + context + ": dtype kernel not implemented");
    kernels.reduce(op, out, tensor, axis);
    return out;
}

// runtime单轴max编排；empty axis无identity，因此在分配输出前拒绝。
inline std::pair<YTensorBase, YTensorBase> maxOneAxis(
    const YTensorBase& tensor, int axis, const yt::type::YDTypeKernels& kernels
) {
    int dim = tensor.ndim();
    if (dim == 0) throw std::runtime_error("[yt::strided::max] Cannot max a 0-dim tensor");
    axis = (axis % dim + dim) % dim;
    if (tensor.shape(axis) == 0) throw std::domain_error("strided::max: cannot reduce an empty axis");
    auto newShape = tensor.shape();
    newShape[axis] = 1;
    YTensorBase values(newShape, tensor.dtype());
    YTensorBase indices(newShape, "int32");
    if (!kernels.reduceIndexed) throw std::runtime_error("strided::max: dtype kernel not implemented");
    kernels.reduceIndexed(yt::type::YReduceOp::Max, values, indices, tensor, axis);
    return {values, indices};
}

YT_IMPL_INLINE YTensorBase sum(const YTensorBase& tensor, int axis) {
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    return reduceOneAxis(tensor, axis, yt::type::YReduceOp::Sum, kernels, "sum");
}

YT_IMPL_INLINE YTensorBase sum(const YTensorBase& tensor, const std::vector<int>& axes) {
    YTensorBase result = tensor;
    std::vector<int> sortedAxes = axes;
    const int dim = tensor.ndim();
    if (dim == 0 && !sortedAxes.empty()) {
        throw std::runtime_error("[yt::strided::sum] Cannot sum a 0-dim tensor");
    }
    // 规范化后降序去重，使逐轴keep-dim reduction顺序稳定且不重复计算。
    for (int& axis : sortedAxes) axis = (axis % dim + dim) % dim;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    sortedAxes.erase(std::unique(sortedAxes.begin(), sortedAxes.end()), sortedAxes.end());
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    for (int axis : sortedAxes) {
        result = reduceOneAxis(result, axis, yt::type::YReduceOp::Sum, kernels, "sum");
    }
    return result;
}

YT_IMPL_INLINE YTensorBase mean(const YTensorBase& tensor, int axis) {
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    return reduceOneAxis(tensor, axis, yt::type::YReduceOp::Mean, kernels, "mean");
}

YT_IMPL_INLINE YTensorBase mean(const YTensorBase& tensor, const std::vector<int>& axes) {
    YTensorBase result = tensor;
    std::vector<int> sortedAxes = axes;
    const int dim = tensor.ndim();
    if (dim == 0 && !sortedAxes.empty()) {
        throw std::runtime_error("[yt::strided::mean] Cannot mean a 0-dim tensor");
    }
    // 固定顺序对整数逐轴截断和浮点舍入都很重要，不能依赖调用者传入顺序。
    for (int& axis : sortedAxes) axis = (axis % dim + dim) % dim;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    sortedAxes.erase(std::unique(sortedAxes.begin(), sortedAxes.end()), sortedAxes.end());
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    for (int axis : sortedAxes) {
        result = reduceOneAxis(result, axis, yt::type::YReduceOp::Mean, kernels, "mean");
    }
    return result;
}

YT_IMPL_INLINE std::pair<YTensorBase, YTensorBase> max(const YTensorBase& tensor, int axis) {
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    return maxOneAxis(tensor, axis, kernels);
}

YT_IMPL_INLINE std::pair<YTensorBase, YTensorBase> max(
    const YTensorBase& tensor, const std::vector<int>& axes
) {
    if (axes.empty()) throw std::invalid_argument("strided::max: axes must not be empty");
    YTensorBase values = tensor;
    YTensorBase indices;
    std::vector<int> sortedAxes = axes;
    const int dim = tensor.ndim();
    if (dim == 0 && !sortedAxes.empty()) {
        throw std::runtime_error("[yt::strided::max] Cannot max a 0-dim tensor");
    }
    for (int& axis : sortedAxes) axis = (axis % dim + dim) % dim;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    sortedAxes.erase(std::unique(sortedAxes.begin(), sortedAxes.end()), sortedAxes.end());
    ensureBuiltinReduceKernels();
    const auto kernels = yt::type::getDTypeKernels(tensor.dtype());
    for (int axis : sortedAxes) {
        auto result = maxOneAxis(values, axis, kernels);
        values = result.first;
        indices = result.second;
    }
    return {values, indices};
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> sum(const YTensor<T, dim>& tensor, int axis) requires(dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = tensor.shape();
    newShape[axis] = 1;
    YTensor<T, dim> out(newShape);
    size_t outSize = out.size();
    int axisSize = tensor.shape(axis);

    // 输出坐标的reduced axis固定为0，再沿输入该axis遍历；每个输出位置独立可并行。
    yt::utils::parallelFor(
        0, yt::utils::checkedIntSize(outSize, "strided::sum"),
        [&](int i) {
            auto coord = out.toCoord(i);
            T accum = T(0);
            for (int j = 0; j < axisSize; ++j) {
                auto subCoord = coord;
                subCoord[axis] = j;
                accum = yt::strided::modularAdd(accum, tensor.at(subCoord));
            }
            out.atData_(i) = accum;
        },
        static_cast<double>(axisSize)
    );
    return out;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> sum(const YTensor<T, dim>& tensor, std::vector<int> axes) requires(dim > 1) {
    for (auto& axis : axes) {
        axis = (axis % dim + dim) % dim;
    }
    YTensor<T, dim> result = tensor;
    std::sort(axes.begin(), axes.end(), std::greater<int>());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    for (int axis : axes) {
        result = yt::strided::sum(result, axis);
    }
    return result;
}

template <typename T, int dim>
YT_IMPL_INLINE T sum(const YTensor<T, dim>& tensor, int) requires(dim == 1) {
    // rank-1只有一个逻辑reduction axis，参数仅为保持统一API签名。
    T accum = T(0);
    int size = yt::utils::checkedIntSize(tensor.size(), "strided::sum");
    for (int i = 0; i < size; ++i) {
        accum = yt::strided::modularAdd(accum, tensor.at(i));
    }
    return accum;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> mean(const YTensor<T, dim>& tensor, int axis) requires(dim > 1) {
    axis = (axis % dim + dim) % dim;
    YTensor<T, dim> out = yt::strided::sum(tensor, axis);
    // 与rank-1 mean保持一致：empty axis的sum结果已经是T(0)，无需执行除零。
    if (tensor.shape(axis) == 0) return out;
    T denom = static_cast<T>(tensor.shape(axis));
    out.broadcastInplace([denom](T& value) { value = value / denom; });
    return out;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> mean(const YTensor<T, dim>& tensor, std::vector<int> axes) requires(dim > 1) {
    for (auto& axis : axes) {
        axis = (axis % dim + dim) % dim;
    }
    // 固定axis顺序，使整数逐轴截断不再取决于调用者传入顺序。
    std::sort(axes.begin(), axes.end(), std::greater<int>());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    YTensor<T, dim> result = tensor;
    for (int axis : axes) {
        result = yt::strided::mean(result, axis);
    }
    return result;
}

template <typename T, int dim>
YT_IMPL_INLINE T mean(const YTensor<T, dim>& tensor, int axis) requires(dim == 1) {
    // rank-1只有axis 0；empty tensor沿用sum identity并返回T(0)。
    (void)axis;
    int size = yt::utils::checkedIntSize(tensor.size(), "strided::mean");
    if (size == 0) {
        return T(0);
    }
    return yt::strided::sum(tensor, 0) / static_cast<T>(size);
}

template <typename T, int dim>
YT_IMPL_INLINE std::pair<YTensor<T, dim>, YTensor<int, dim>> max(
    const YTensor<T, dim>& tensor, int axis
) requires(dim > 1) {
    axis = (axis % dim + dim) % dim;
    if (tensor.shape(axis) == 0) {
        throw std::domain_error("strided::max: cannot reduce an empty axis");
    }
    auto newShape = tensor.shape();
    newShape[axis] = 1;
    YTensor<T, dim> values(newShape);
    YTensor<int, dim> indices(newShape);
    size_t outSize = values.size();
    int axisSize = tensor.shape(axis);

    yt::utils::parallelFor(
        0, yt::utils::checkedIntSize(outSize, "strided::max"),
        [&](int i) {
            auto coord = values.toCoord(i);
            T maxVal = tensor.at(coord);
            int maxIdx = 0;
            for (int j = 0; j < axisSize; ++j) {
                auto subCoord = coord;
                subCoord[axis] = j;
                const T& value = tensor.at(subCoord);
                    // 严格比较保证tie时保留axis内首次出现位置。
                    if (value > maxVal) {
                    maxVal = value;
                    maxIdx = j;
                }
            }
            values.atData_(i) = maxVal;
            indices.atData_(i) = maxIdx;
        },
        static_cast<double>(axisSize)
    );
    return {values, indices};
}

template <typename T, int dim>
YT_IMPL_INLINE std::pair<YTensor<T, dim>, YTensor<int, dim>> max(
    const YTensor<T, dim>& tensor, std::vector<int> axes
) requires(dim > 1) {
    if (axes.empty()) throw std::invalid_argument("strided::max: axes must not be empty");
    for (auto& axis : axes) {
        axis = (axis % dim + dim) % dim;
    }
    // 升序axis定义reduced子空间的row-major顺序，最后一个axis变化最快。
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    if (axes.size() == 1) return yt::strided::max(tensor, axes.front());

    std::vector<int> outputShape = tensor.shape();
    size_t reducedSize = 1;
    for (int axis : axes) {
        const int extent = tensor.shape(axis);
        if (extent == 0) throw std::domain_error("strided::max: cannot reduce an empty axis");
        if (reducedSize > static_cast<size_t>(std::numeric_limits<int>::max()) /
                              static_cast<size_t>(extent)) {
            throw std::overflow_error("strided::max: reduced space exceeds int index range");
        }
        reducedSize *= static_cast<size_t>(extent);
        outputShape[axis] = 1;
    }

    YTensor<T, dim> values(outputShape);
    YTensor<int, dim> indices(outputShape);
    const int reducedCount = yt::utils::checkedIntSize(reducedSize, "strided::max reduced space");
    const int outputCount = yt::utils::checkedIntSize(values.size(), "strided::max output");
    yt::utils::parallelFor(
        0, outputCount,
        [&](int outputIndex) {
            auto coord = values.toCoord(outputIndex);
            T maxValue = tensor.at(coord);
            int maxIndex = 0;
            for (int flatIndex = 1; flatIndex < reducedCount; ++flatIndex) {
                int remaining = flatIndex;
                // 逆序解码flatIndex，使最后一个选中axis成为fastest-varying维度。
                for (auto axisIt = axes.rbegin(); axisIt != axes.rend(); ++axisIt) {
                    const int extent = tensor.shape(*axisIt);
                    coord[*axisIt] = remaining % extent;
                    remaining /= extent;
                }
                const T& candidate = tensor.at(coord);
                if (candidate > maxValue) {
                    maxValue = candidate;
                    maxIndex = flatIndex;
                }
            }
            values.atData_(outputIndex) = maxValue;
            indices.atData_(outputIndex) = maxIndex;
        },
        static_cast<double>(reducedCount)
    );
    return {values, indices};
}

template <typename T, int dim>
YT_IMPL_INLINE std::pair<T, int> max(const YTensor<T, dim>& tensor, int) requires(dim == 1) {
    // rank-1只有axis 0；严格比较让tie保持第一个最大值索引。
    if (tensor.size() == 0) {
        throw std::domain_error("strided::max: cannot reduce an empty tensor");
    }
    T maxVal = tensor.at(0);
    int maxIdx = 0;
    int size = yt::utils::checkedIntSize(tensor.size(), "strided::max");
    for (int i = 0; i < size; ++i) {
        const T& value = tensor.at(i);
        if (value > maxVal) {
            maxVal = value;
            maxIdx = i;
        }
    }
    return {maxVal, maxIdx};
}

}  // namespace yt::strided
