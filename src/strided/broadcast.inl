#pragma once
/***************
 * file: broadcast.inl
 * purpose: 广播操作函数实现
 ***************/

namespace yt::strided {

// ==================== typed arithmetic and mutation safety ====================

// 编译期检查typed tensor参数的scalar类型是否与callable合同一致。
// 注意：普通标量由后续convertible检查处理；这里只拒绝不同scalar类型的YTensor混用。
template <typename Expected, typename Arg>
struct HasMatchingTypedScalar : std::true_type {};

template <typename Expected, typename Actual, int Dim>
struct HasMatchingTypedScalar<Expected, YTensor<Actual, Dim>> : std::is_same<Expected, Actual> {};

template <typename Expected, typename Arg>
inline constexpr bool hasMatchingTypedScalar =
    HasMatchingTypedScalar<Expected, std::remove_cvref_t<Arg>>::value;

// 在宽类型中累加广播物理偏移，避免stride乘加产生有符号溢出。
YT_IMPL_INLINE void checkedBroadcastOffsetAdd(int& offset, int position, int physicalStride) {
    const int64_t next = static_cast<int64_t>(offset) +
                         static_cast<int64_t>(position) * static_cast<int64_t>(physicalStride);
    if (next < std::numeric_limits<int>::min() || next > std::numeric_limits<int>::max()) {
        throw std::overflow_error("strided broadcast: physical index exceeds int range");
    }
    offset = static_cast<int>(next);
}

// 遍历可写逻辑位置；只有logical-to-physical映射不重叠时才允许并行写入。
// 注意：zero-stride/repeat/unfold等重叠view必须串行，保证同一storage位置的写入顺序确定。
template <typename Tensor, typename Func>
YT_IMPL_INLINE void forEachMutableBroadcastIndex(
    Tensor& target, int total, Func&& func, bool enableParallel = true
) {
    if (target.isDisjoint() && enableParallel) {
        yt::utils::parallelFor(0, total, std::forward<Func>(func));
    } else {
        for (int index = 0; index < total; ++index) func(index);
    }
}

// 将unsigned位模式恢复为同宽整数，避免实现定义的越界signed转换。
template <typename T>
YT_IMPL_INLINE T fromUnsignedBits(std::make_unsigned_t<T> value) {
    if constexpr (std::is_signed_v<T>) return std::bit_cast<T>(value);
    return value;
}

// 为整数定义同宽模加法；浮点和自定义算术类型沿用其operator语义。
template <typename T>
YT_IMPL_INLINE T modularAdd(const T& left, const T& right) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        using U = std::make_unsigned_t<T>;
        return fromUnsignedBits<T>(static_cast<U>(static_cast<U>(left) + static_cast<U>(right)));
    } else {
        return left + right;
    }
}

// 为整数定义同宽模减法，规避C++ signed overflow未定义行为。
template <typename T>
YT_IMPL_INLINE T modularSub(const T& left, const T& right) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        using U = std::make_unsigned_t<T>;
        return fromUnsignedBits<T>(static_cast<U>(static_cast<U>(left) - static_cast<U>(right)));
    } else {
        return left - right;
    }
}

// 为整数定义同宽模乘法，保证typed/runtime kernel的溢出语义一致。
template <typename T>
YT_IMPL_INLINE T modularMul(const T& left, const T& right) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        using U = std::make_unsigned_t<T>;
        return fromUnsignedBits<T>(static_cast<U>(static_cast<U>(left) * static_cast<U>(right)));
    } else {
        return left * right;
    }
}

// 通过unsigned位模式执行左移，并在移位前验证count范围。
template <typename T>
YT_IMPL_INLINE T bitPreservingLeftShift(const T& value, const T& count) {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
    if (count < 0 || static_cast<uint64_t>(count) >= sizeof(T) * 8) {
        throw std::domain_error("strided broadcast: invalid left shift count");
    }
    using U = std::make_unsigned_t<T>;
    return fromUnsignedBits<T>(static_cast<U>(static_cast<U>(value) << static_cast<unsigned>(count)));
}

// 校验整数除零和min/-1溢出后执行除法。
template <typename T>
YT_IMPL_INLINE T checkedDivide(const T& left, const T& right) {
    if constexpr (std::is_integral_v<T>) {
        bool overflow = false;
        if constexpr (std::is_signed_v<T>) {
            overflow = left == std::numeric_limits<T>::min() && right == static_cast<T>(-1);
        }
        if (right == 0 || overflow) {
            throw std::domain_error("strided broadcast: invalid integer division");
        }
    }
    return left / right;
}

// 执行整数安全取模或浮点fmod，并拒绝未定义的整数输入。
template <typename T>
YT_IMPL_INLINE T checkedModulo(const T& left, const T& right) {
    if constexpr (std::is_integral_v<T>) {
        bool overflow = false;
        if constexpr (std::is_signed_v<T>) {
            overflow = left == std::numeric_limits<T>::min() && right == static_cast<T>(-1);
        }
        if (right == 0 || overflow) {
            throw std::domain_error("strided broadcast: invalid integer modulo");
        }
        return left % right;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::fmod(left, right);
    } else {
        return left % right;
    }
}

// 校验右移count；signed负值的右移结果保持当前编译器/C++实现语义。
template <typename T>
YT_IMPL_INLINE T checkedRightShift(const T& value, const T& count) {
    static_assert(std::is_integral_v<T> && !std::is_same_v<T, bool>);
    if (count < 0 || static_cast<uint64_t>(count) >= sizeof(T) * 8) {
        throw std::domain_error("strided broadcast: invalid right shift count");
    }
    return value >> count;
}

// 将编译期broadcast op映射到统一的安全算术语义，供typed operator热路径复用。
template <yt::type::YBroadcastOp Op, typename T>
YT_IMPL_INLINE T typedBinaryArithmetic(const T& left, const T& right) {
    if constexpr (Op == yt::type::YBroadcastOp::Add) return modularAdd(left, right);
    if constexpr (Op == yt::type::YBroadcastOp::Sub) return modularSub(left, right);
    if constexpr (Op == yt::type::YBroadcastOp::Mul) return modularMul(left, right);
    if constexpr (Op == yt::type::YBroadcastOp::Div) return checkedDivide(left, right);
    if constexpr (Op == yt::type::YBroadcastOp::BitAnd) return left & right;
    if constexpr (Op == yt::type::YBroadcastOp::BitOr) return left | right;
    if constexpr (Op == yt::type::YBroadcastOp::BitXor) return left ^ right;
    if constexpr (Op == yt::type::YBroadcastOp::LShift) {
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return bitPreservingLeftShift(left, right);
        } else {
            throw std::runtime_error("strided broadcast: dtype does not support <<");
        }
    }
    if constexpr (Op == yt::type::YBroadcastOp::RShift) {
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return checkedRightShift(left, right);
        } else {
            throw std::runtime_error("strided broadcast: dtype does not support >>");
        }
    }
}

// YTensorBase scalar构造的特权入口，负责注册dtype对象的真实C++生命周期。
struct BroadcastAccess {
    static YTensorBase scalarFromRegisteredValue(
        const std::string& dtype, const void* value, const yt::type::TypeRegItem& typeInfo
    ) {
        if (typeInfo.size <= 0) {
            throw std::invalid_argument("strided::scalarTensor: registered scalar has invalid size");
        }
        YTensorBase result;
        result._dtype = dtype;
        result._element_size = static_cast<size_t>(typeInfo.size);
        result.stridedShape().clear();
        result.stridedStride().clear();
        result.stridedOffset() = 0;

        char* raw = new char[result._element_size];
        if (typeInfo.isPOD) {
            std::memcpy(raw, value, result._element_size);
            result._memory = YMemory(std::shared_ptr<char[]>(raw), result._element_size);
            return result;
        }
        if (!typeInfo.copyConstruct || !typeInfo.destructor) {
            delete[] raw;
            throw std::runtime_error(
                "strided::scalarTensor: non-POD scalar requires copyConstruct and destructor"
            );
        }
        try {
            typeInfo.copyConstruct(raw, value);
        } catch (...) {
            delete[] raw;
            throw;
        }
        // shared deleter恰好析构一个已成功placement-constructed的scalar对象。
        auto destructor = typeInfo.destructor;
        result._memory = YMemory(
            std::shared_ptr<char[]>(raw, [destructor](char* ptr) {
                destructor(ptr);
                delete[] ptr;
            }),
            result._element_size, "cpu", false
        );
        return result;
    }
};

// 计算多个shape统一广播后的结果shape（右对齐对齐，维度为1时可广播）。
// 参数 shapes：参与广播的shape列表
// 返回：广播后的 shape
// 异常 std::runtime_error：形状不兼容时抛出
// 注意：算法：取最大维数，各shape右对齐；对应维若一方为1则取另一方，否则必须相等。
inline std::vector<int> computeBroadcastShape(const std::vector<std::vector<int>>& shapes) {
    if (shapes.empty()) return {};
    size_t maxDim = 0;
    for (const auto& shape : shapes) maxDim = std::max(maxDim, shape.size());

    std::vector<int> result(maxDim, 1);
    for (const auto& shape : shapes) {
        size_t offset = maxDim - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t resultIndex = offset + i;
            if (result[resultIndex] == 1) {
                result[resultIndex] = shape[i];
            } else if (shape[i] != 1 && shape[i] != result[resultIndex]) {
                throw std::runtime_error("Broadcast: shapes cannot be broadcast together");
            }
        }
    }
    return result;
}

template <typename T, int dim, typename Func>
YTensor<T, dim>& forEach(YTensor<T, dim>& tensor, Func&& func, double flop) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::forEach: layout not implemented");
    }

    constexpr bool oneArgFunc =
        std::is_invocable_v<Func, T&> && !std::is_invocable_v<Func, T&, const std::vector<int>&>;
    if constexpr (oneArgFunc) {
        // 无坐标callable复用统一inplace广播，从而继承alias快照和重叠view串行写入规则。
        return yt::strided::broadcastInplace(tensor, [&func](T& value) {
            using ResultType = std::invoke_result_t<Func, T&>;
            if constexpr (std::is_void_v<ResultType>) {
                func(value);
            } else {
                value = func(value);
            }
        });
    } else {
        auto apply = [&func](T& value, const std::vector<int>& coord) {
            using ResultType = std::invoke_result_t<Func, T&, const std::vector<int>&>;
            if constexpr (std::is_void_v<ResultType>) {
                func(value, coord);
            } else {
                value = func(value, coord);
            }
        };
        int total = yt::utils::checkedIntSize(tensor.size(), "strided::forEach");
        // 坐标callable仍需构造逻辑坐标；连续view可直接索引data，其他view通过at映射物理位置。
        if (tensor.isContiguous()) {
            T* data = tensor.data();
            forEachMutableBroadcastIndex(
                tensor, total,
                [&](int index) {
                    auto coord = tensor.toCoord(index);
                    apply(data[index], coord);
                },
                flop != 0.0
            );
        } else {
            forEachMutableBroadcastIndex(
                tensor, total,
                [&](int index) {
                    auto coord = tensor.toCoord(index);
                    apply(tensor.at(coord), coord);
                },
                flop != 0.0
            );
        }
        return tensor;
    }
}

template <typename T, int dim>
YTensor<T, dim>& fill(YTensor<T, dim>& tensor, const T& value) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::fill: layout not implemented");
    }

    if (tensor.size() == 0) return tensor;
    // 先按绝对stride重排并归一化反向轴，以便尽可能按连续块填充。
    auto view = tensor.mostContinuousView();
    int contiguousFrom = view.isContiguousFrom();
    if (contiguousFrom == 0) {
        std::fill(view.data(), view.data() + view.size(), value);
        return tensor;
    }
    if (contiguousFrom >= dim) {
        yt::strided::broadcastInplace(tensor, [value](T& item) { item = value; });
        return tensor;
    }

    // 前缀坐标决定一个连续尾块；不同outer index写入互不重叠时才会并行。
    size_t contiguousSize = 1;
    size_t outerSize = 1;
    for (int i = contiguousFrom; i < dim; ++i) {
        contiguousSize *= view.shape(i);
    }
    for (int i = 0; i < contiguousFrom; ++i) {
        outerSize *= view.shape(i);
    }

    T* data = view.data();
    forEachMutableBroadcastIndex(
        view, yt::utils::checkedIntSize(outerSize, "strided::fill"), [&](int outerIndex) {
            int remaining = outerIndex;
            int64_t offset = 0;
            for (int i = contiguousFrom - 1; i >= 0; --i) {
                int coord = remaining % view.shape(i);
                remaining /= view.shape(i);
                offset += static_cast<int64_t>(coord) * view.stride_(i);
            }
            if (offset < std::numeric_limits<std::ptrdiff_t>::min() ||
                offset > std::numeric_limits<std::ptrdiff_t>::max()) {
                throw std::overflow_error("strided::fill: physical offset exceeds ptrdiff_t range");
            }
            std::fill(
                data + static_cast<std::ptrdiff_t>(offset),
                data + static_cast<std::ptrdiff_t>(offset) + contiguousSize, value
            );
        }
    );
    return tensor;
}

template <typename... Args>
constexpr size_t countTensors() {
    if constexpr (sizeof...(Args) == 0) {
        return 0;
    } else {
        return (static_cast<size_t>(::yt::utils::is_ytensor_v<Args>) + ...);
    }
}

template <size_t N, size_t I>
template <typename StridesArray, typename IndicesArray>
inline void IndexAccumulator<N, I>::accumulate(
    IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx
) {
    checkedBroadcastOffsetAdd(indices[I], posi, strides[I][dimIdx]);
    if constexpr (I + 1 < N) {
        IndexAccumulator<N, I + 1>::accumulate(indices, posi, strides, dimIdx);
    }
}

template <size_t N, int Dim>
template <typename LogicStrideArray, typename ShapeArray, typename StridesArray>
inline std::array<int, N> NaryIndexComputer<N, Dim>::compute(
    int index, const LogicStrideArray& logicStride, const ShapeArray& shape, const StridesArray& strides
) {
    std::array<int, N> indices = {};
    for (int d = 0; d < Dim; ++d) {
        int posi = (index / logicStride[d]) % shape[d];
        accumulateAll<0>(indices, posi, strides, d);
    }
    return indices;
}

template <size_t N, int Dim>
template <typename LogicStrideArray, typename ShapeArray, typename ThisStrideArray, typename StridesArray>
inline std::array<int, N> NaryIndexComputer<N, Dim>::computeWithThis(
    int index, const LogicStrideArray& logicStride, const ShapeArray& shape,
    const ThisStrideArray& thisStride, const StridesArray& strides, int& thisDataIdx
) {
    // 零输入参数仍需计算target位置，同时避免实例化indices[0]。
    if constexpr (N == 0) {
        thisDataIdx = 0;
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            checkedBroadcastOffsetAdd(thisDataIdx, posi, thisStride[d]);
        }
        return {};
    }
    std::array<int, N> indices = {};
    thisDataIdx = 0;
    for (int d = 0; d < Dim; ++d) {
        int posi = (index / logicStride[d]) % shape[d];
        checkedBroadcastOffsetAdd(thisDataIdx, posi, thisStride[d]);
        accumulateAll<0>(indices, posi, strides, d);
    }
    return indices;
}

template <size_t N, int Dim>
template <size_t I, typename IndicesArray, typename StridesArray>
inline void NaryIndexComputer<N, Dim>::accumulateAll(
    IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx
) {
    checkedBroadcastOffsetAdd(indices[I], posi, strides[I][dimIdx]);
    if constexpr (I + 1 < N) {
        accumulateAll<I + 1>(indices, posi, strides, dimIdx);
    }
}

inline void computeBroadcastIndicesRuntime(
    int index, const std::vector<int>& logicStride, const std::vector<int>& shape,
    const std::vector<int>& thisStride, const std::vector<std::vector<int>>& otherStrides, int& thisIdx,
    std::vector<int>& otherIndices, int ndim
) {
    thisIdx = 0;
    std::fill(otherIndices.begin(), otherIndices.end(), 0);
    for (int i = 0; i < ndim; ++i) {
        int posi = (index / logicStride[i]) % shape[i];
        checkedBroadcastOffsetAdd(thisIdx, posi, thisStride[i]);
        for (size_t t = 0; t < otherStrides.size(); ++t) {
            checkedBroadcastOffsetAdd(otherIndices[t], posi, otherStrides[t][i]);
        }
    }
}

template <size_t N>
inline std::array<int, N> computeBroadcastIndices(
    int index, const std::vector<int>& logicStride, const std::vector<int>& broadcastShape,
    const std::array<const int*, N>& strides, int opdim
) {
    std::array<int, N> indices = {};
    for (int i = 0; i < opdim; ++i) {
        int posi = (index / logicStride[i]) % broadcastShape[i];
        IndexAccumulator<N>::accumulate(indices, posi, strides, i);
    }
    return indices;
}

inline std::vector<int> getBroadcastStride(
    const std::vector<int>& shape, const std::vector<int>& stride, const std::vector<int>& broadcastShape
) {
    size_t offset = broadcastShape.size() - shape.size();
    // 缺失的leading axis和被扩展的singleton axis都保留zero stride。
    std::vector<int> result(broadcastShape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == broadcastShape[i + offset]) {
            result[i + offset] = stride[i];
        }
    }
    return result;
}

inline int computeDataIndex(
    int linearIndex, const std::vector<int>& logicStride, const std::vector<int>& broadcastStride,
    const std::vector<int>& broadcastShape
) {
    int dataIndex = 0;
    int opdim = static_cast<int>(broadcastShape.size());
    for (int i = 0; i < opdim; ++i) {
        int posi = (linearIndex / logicStride[i]) % broadcastShape[i];
        checkedBroadcastOffsetAdd(dataIndex, posi, broadcastStride[i]);
    }
    return dataIndex;
}

template <yt::type::YBroadcastOp Op, typename T>
YT_IMPL_INLINE void validateTypedIntegerPair(const T& left, const T& right) {
    if constexpr (!std::is_integral_v<T> || std::is_same_v<T, bool>) {
        return;
    } else if constexpr (Op == yt::type::YBroadcastOp::Div) {
        (void)checkedDivide(left, right);
    } else if constexpr (Op == yt::type::YBroadcastOp::Mod) {
        (void)checkedModulo(left, right);
    } else if constexpr (Op == yt::type::YBroadcastOp::LShift) {
        (void)bitPreservingLeftShift(left, right);
    } else if constexpr (Op == yt::type::YBroadcastOp::RShift) {
        (void)checkedRightShift(left, right);
    }
}

// 在任何并行写入前遍历完整广播结果，验证可能抛错的整数操作。
// 注意：预校验保证除零、min/-1和非法shift不会留下部分修改的输出。
template <yt::type::YBroadcastOp Op, typename T, int LeftDim, int RightDim>
YT_IMPL_INLINE void validateTypedBinaryOperation(
    const YTensor<T, LeftDim>& left, const YTensor<T, RightDim>& right
) {
    if constexpr (
        std::is_integral_v<T> && !std::is_same_v<T, bool> &&
        (Op == yt::type::YBroadcastOp::Div || Op == yt::type::YBroadcastOp::Mod ||
         Op == yt::type::YBroadcastOp::LShift || Op == yt::type::YBroadcastOp::RShift)
    ) {
        auto outShape = computeBroadcastShape({left.shape(), right.shape()});
        std::vector<int> logicStride(outShape.size());
        size_t stride = 1;
        for (int i = static_cast<int>(outShape.size()) - 1; i >= 0; --i) {
            logicStride[i] = yt::utils::checkedIntSize(stride, "typed broadcast validation stride");
            if (outShape[i] != 0 &&
                stride > static_cast<size_t>(std::numeric_limits<int>::max()) / outShape[i]) {
                throw std::overflow_error("typed broadcast validation size exceeds int range");
            }
            stride *= static_cast<size_t>(outShape[i]);
        }
        auto leftStride = getBroadcastStride(left.shape(), left.stride_(), outShape);
        auto rightStride = getBroadcastStride(right.shape(), right.stride_(), outShape);
        const T* leftData = left.data();
        const T* rightData = right.data();
        int total = yt::utils::checkedIntSize(stride, "typed broadcast validation");
        for (int i = 0; i < total; ++i) {
            int leftIndex = computeDataIndex(i, logicStride, leftStride, outShape);
            int rightIndex = computeDataIndex(i, logicStride, rightStride, outShape);
            validateTypedIntegerPair<Op>(leftData[leftIndex], rightData[rightIndex]);
        }
    }
}

template <yt::type::YBroadcastOp Op, typename T, int Dim>
YT_IMPL_INLINE void validateTypedBinaryOperation(const YTensor<T, Dim>& left, const T& right) {
    if constexpr (
        std::is_integral_v<T> && !std::is_same_v<T, bool> &&
        (Op == yt::type::YBroadcastOp::Div || Op == yt::type::YBroadcastOp::Mod ||
         Op == yt::type::YBroadcastOp::LShift || Op == yt::type::YBroadcastOp::RShift)
    ) {
        int total = yt::utils::checkedIntSize(left.size(), "typed scalar broadcast validation");
        for (int i = 0; i < total; ++i) {
            validateTypedIntegerPair<Op>(left.atData(i), right);
        }
    }
}

template <int requestedResultDim, typename Func, typename... Args>
auto broadcast(Func&& func, Args&&... tensors) {
    using namespace ::yt::utils;

    constexpr int computedDim = max_dim<Args...>();
    constexpr int resultDim =
        (requestedResultDim > 0) ? requestedResultDim : (computedDim > 0 ? computedDim : 1);
    static_assert(
        requestedResultDim > 0 || all_ytensor_template<Args...>(),
        "broadcast: when using YTensorBase, you must explicitly specify resultDim, "
    );

    using ScalarType = ::yt::utils::first_arg_of_t<Func>;

    static_assert(
        all_valid_broadcast_args<ScalarType, Args...>(),
        "broadcast: all arguments must be either YTensor/YTensorBase or convertible to scalar type"
    );
    static_assert(
        (hasMatchingTypedScalar<ScalarType, Args> && ...),
        "broadcast: typed tensor scalar types must match the callable scalar type"
    );

    std::vector<std::vector<int>> shapes;
    [[maybe_unused]] auto collectShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    if (shapes.empty()) {
        throw std::runtime_error("broadcast: at least one tensor argument required");
    }

    auto broadcastShape = yt::strided::computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());

    if constexpr (requestedResultDim > 0) {
        if (opdim != requestedResultDim) {
            throw std::runtime_error(
                "broadcast: specified resultDim (" + std::to_string(requestedResultDim) +
                ") does not match actual broadcast dimension (" + std::to_string(opdim) + ")"
            );
        }
    }

    std::vector<int> logicStride(opdim);
    size_t stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = yt::utils::checkedIntSize(stride, "strided::broadcast logical stride");
        if (broadcastShape[i] != 0 &&
            stride > static_cast<size_t>(std::numeric_limits<int>::max()) / broadcastShape[i]) {
            throw std::overflow_error("strided::broadcast: element count exceeds int indexing range");
        }
        stride *= static_cast<size_t>(broadcastShape[i]);
    }
    int totalSize = yt::utils::checkedIntSize(stride, "strided::broadcast");

    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarType*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    std::string tensorDtype;

    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (tensorDtype.empty()) tensorDtype = arg.dtype();
            if (arg.dtype() != tensorDtype ||
                (!yt::utils::is_ytensor_v<ScalarType> &&
                 arg.dtype() != yt::type::getTypeName<ScalarType>())) {
                throw std::invalid_argument("broadcast: tensor dtype does not match callable scalar type");
            }
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

    using ReturnType = std::invoke_result_t<
        Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarType, Args>>())...>;

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

    // 只有shape完全相等时，连续输入的同一linear index才表示同一逻辑坐标。
    if (allContiguous && allShapeEqual) {
        ReturnType* resultPtr = result.data_();

        yt::utils::parallelFor(0, totalSize, [&](int index) {
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueDirect = [&](auto&& arg) -> ScalarType {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<ScalarType>(arg);
                }
            };

            ScalarType values[] = {getValueDirect(tensors)...};

            [&]<size_t... Is>(std::index_sequence<Is...>) { resultPtr[index] = func(values[Is]...); }
            (std::make_index_sequence<sizeof...(Args)>{});
        });
        return result;
    }

    ReturnType* resultPtr = result.data_();
    constexpr size_t numTensors = countTensors<Args...>();

    // 通用路径把每个输入右对齐到结果rank，再由编译期index computer展开物理偏移。
    std::array<std::array<int, resultDim>, numTensors> tensorStrides;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int i = 0; i < resultDim; ++i) {
            tensorStrides[t][i] =
                (i < static_cast<int>(broadcastStrides[t].size())) ? broadcastStrides[t][i] : 0;
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
        resultLogicStride[i] = yt::utils::checkedIntSize(stride, "strided::broadcast logical stride");
        stride *= resultShape[i];
    }

    yt::utils::parallelFor(0, totalSize, [&](int index) {
        auto indices = NaryIndexComputer<numTensors, resultDim>::compute(
            index, resultLogicStride, resultShape, tensorStrides
        );

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

        [&]<size_t... Is>(std::index_sequence<Is...>) { resultPtr[index] = func(values[Is]...); }
        (std::make_index_sequence<sizeof...(Args)>{});
    });

    return result;
}

// typed原地广播核心；调用方必须先处理与target重叠的输入快照。
template <typename TensorType, typename Func, typename... Args>
TensorType& applyTypedBroadcastInplace(TensorType& target, Func&& func, Args&&... tensors) {
    using namespace ::yt::utils;
    using T = typename TensorType::scalarType;
    constexpr int dim = TensorType::ndim;

    static_assert(
        all_valid_broadcast_args<T, Args...>(),
        "broadcastInplace: all arguments must be either YTensor/YTensorBase or convertible to scalar type"
    );
    static_assert(
        (hasMatchingTypedScalar<T, Args> && ...),
        "broadcastInplace: typed tensor scalar types must match the target scalar type"
    );

    std::vector<std::vector<int>> shapes;
    shapes.push_back(target.shape());

    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    auto broadcastShape = yt::strided::computeBroadcastShape(shapes);

    // 原地操作不能扩张target metadata，完整广播shape必须与target逐维一致。
    auto targetShapeVec = target.shape();
    if (static_cast<int>(broadcastShape.size()) != dim) {
        throw std::runtime_error("broadcastInplace: result dimension mismatch");
    }
    for (int i = 0; i < dim; ++i) {
        if (targetShapeVec[i] != broadcastShape[i]) {
            throw std::runtime_error("broadcastInplace: target tensor shape must match broadcast shape");
        }
    }

    int totalSize = yt::utils::checkedIntSize(target.size(), "strided::broadcastInplace");
    bool allContiguous = target.isContiguous();
    bool allEqualShape = true;

    [[maybe_unused]] auto checkContiguousAndShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (arg.dtype() != target.dtype()) {
                throw std::invalid_argument("broadcastInplace: tensor dtype must match target dtype");
            }
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

    // 连续且shape一致时可绕过坐标分解；广播维或任一strided输入走下方通用路径。
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

        forEachMutableBroadcastIndex(target, totalSize, [&](int index) {
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueFast = [&](auto&& arg) -> T {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<T>(arg);
                }
            };

            std::array<T, sizeof...(Args)> values{getValueFast(tensors)...};

            [&]<size_t... Is>(std::index_sequence<Is...>) { func(targetDataPtr[index], values[Is]...); }
            (std::make_index_sequence<sizeof...(Args)>{});
        });

        return target;
    }

    // 通用路径同时计算target和每个输入的物理位置，支持负stride和zero-stride广播。
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

    forEachMutableBroadcastIndex(target, totalSize, [&](int index) {
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

        std::array<T, sizeof...(Args)> values{getValue(tensors)...};

        [&]<size_t... Is>(std::index_sequence<Is...>) { func(target.atData_(targetDataIdx), values[Is]...); }
        (std::make_index_sequence<sizeof...(Args)>{});
    });

    return target;
}

template <typename TensorType, typename Func, typename... Args>
TensorType& broadcastInplace(TensorType& target, Func&& func, Args&&... tensors) {
    if constexpr (sizeof...(Args) == 0) {
        return applyTypedBroadcastInplace(target, std::forward<Func>(func));
    } else {
        // 原地写入可能污染重叠slice/transpose/unfold输入；dispatch前先快照这些参数。
        auto prepare = [&target](auto&& arg) {
            using Arg = std::decay_t<decltype(arg)>;
            if constexpr (yt::utils::is_ytensor_v<Arg>) {
                if (yt::strided::physicalSpansOverlap(target, arg)) {
                    return Arg(arg.clone());
                }
            }
            return Arg(std::forward<decltype(arg)>(arg));
        };
        auto prepared = std::tuple{prepare(std::forward<Args>(tensors))...};
        return std::apply(
            [&](auto&... args) -> TensorType& {
                return applyTypedBroadcastInplace(target, std::forward<Func>(func), args...);
            },
            prepared
        );
    }
}

template <typename DType, typename Func, size_t N, size_t... I>
YT_IMPL_INLINE void invokeBroadcastArguments(
    Func& func, DType& target, const std::array<DType, N>& values, std::index_sequence<I...>
) {
    func(target, values[I]...);
}

// runtime callable原地广播核心，执行dtype/layout/shape校验和一次性索引分发。
template <typename Func, typename... Args>
yt::YTensorBase& applyTypeErasedBroadcastInplace(
    yt::YTensorBase& target, Func&& func, Args&&... tensors
) {
    using namespace ::yt::utils;

    if (!target.isStrided()) {
        throw std::runtime_error("strided::broadcastInplaceBase: layout not implemented");
    }

    using DType = std::remove_cvref_t<first_arg_of_t<Func>>;
    static_assert(
        (hasMatchingTypedScalar<DType, Args> && ...),
        "broadcastInplaceBase: typed tensor scalar types must match the callable scalar type"
    );
    const std::string expectedDtype = target.dtype();
    if constexpr (!yt::utils::is_ytensor_v<DType>) {
        if (expectedDtype != yt::type::getTypeName<DType>()) {
            throw std::invalid_argument("strided::broadcastInplaceBase: target dtype does not match callable");
        }
    }

    std::vector<std::vector<int>> shapes;
    shapes.push_back(target.shape());

    [[maybe_unused]] auto collectShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (arg.dtype() != expectedDtype) {
                throw std::invalid_argument(
                    "strided::broadcastInplaceBase: tensor dtype does not match callable"
                );
            }
            if constexpr (!is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                if (!arg.isStrided()) {
                    throw std::runtime_error("strided::broadcastInplaceBase: layout not implemented");
                }
            }
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    auto broadcastShape = yt::strided::computeBroadcastShape(shapes);
    int targetDim = target.ndim();
    if (static_cast<int>(broadcastShape.size()) != targetDim) {
        throw std::runtime_error("strided::broadcastInplaceBase: result dimension mismatch");
    }

    auto targetShape = target.shape();
    for (int i = 0; i < targetDim; ++i) {
        if (targetShape[i] != broadcastShape[i]) {
            throw std::runtime_error(
                "strided::broadcastInplaceBase: target tensor shape must match broadcast shape"
            );
        }
    }

    int totalSize = yt::utils::checkedIntSize(target.size(), "strided::broadcastInplaceBase");
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

    DType* targetDataPtr = target.data<DType>();

    // 与typed路径相同：只有完全同shape的连续tensor可按linear index直接读取。
    if (allContiguous && allEqualShape) {
        forEachMutableBroadcastIndex(target, totalSize, [&](int index) {
            [[maybe_unused]] auto getValueFast = [&](auto&& arg) -> decltype(auto) {
                if constexpr (is_ytensor_v<decltype(arg)>) {
                    if constexpr (is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                        return arg.data()[index];
                    } else {
                        return arg.template data<DType>()[index];
                    }
                } else {
                    return static_cast<DType>(std::forward<decltype(arg)>(arg));
                }
            };
            std::array<DType, sizeof...(Args)> values{getValueFast(tensors)...};
            invokeBroadcastArguments(
                func, targetDataPtr[index], values, std::make_index_sequence<sizeof...(Args)>{}
            );
        });
        return target;
    }

    auto logicStride = target.stride();
    auto targetStride = target.stride_();
    constexpr size_t numArgs = sizeof...(Args);
    std::array<std::vector<int>, numArgs> broadcastStrides;
    std::array<const DType*, numArgs> dataPtrs{};

    [[maybe_unused]] size_t argIdx = 0;
    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides[argIdx] = getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape);
            if constexpr (is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                dataPtrs[argIdx] = arg.data();
            } else {
                dataPtrs[argIdx] = arg.template data<DType>();
            }
        } else {
            // 标量不占用tensor pointer槽，zero stride使它在所有逻辑位置保持同一值。
            broadcastStrides[argIdx].assign(static_cast<size_t>(targetDim), 0);
        }
        ++argIdx;
    };
    (collectBroadcastInfo(tensors), ...);

    std::array<const int*, numArgs + 1> stridesArray;
    stridesArray[0] = targetStride.data();
    for (size_t i = 0; i < numArgs; ++i) {
        stridesArray[i + 1] = broadcastStrides[i].data();
    }
    forEachMutableBroadcastIndex(target, totalSize, [&](int index) {
        auto indices = computeBroadcastIndices<numArgs + 1>(
            index, logicStride, broadcastShape, stridesArray, targetDim
        );
        int targetIndex = indices[0];

        [[maybe_unused]] size_t valueIndex = 0;
        auto getValue = [&](auto&& arg) -> DType {
            const size_t indexInPack = valueIndex++;
            if constexpr (is_ytensor_v<decltype(arg)>) {
                return dataPtrs[indexInPack][indices[indexInPack + 1]];
            } else {
                return static_cast<DType>(arg);
            }
        };
        std::array<DType, numArgs> values{getValue(tensors)...};
        invokeBroadcastArguments(
            func, targetDataPtr[targetIndex], values, std::make_index_sequence<numArgs>{}
        );
    });

    return target;
}

template <typename Func, typename... Args>
yt::YTensorBase& broadcastInplaceBase(yt::YTensorBase& target, Func&& func, Args&&... tensors) {
    // callable runtime路径与operator路径使用相同alias合同：重叠输入读取操作前快照。
    auto prepare = [&target](auto&& arg) {
        using Arg = std::decay_t<decltype(arg)>;
        if constexpr (yt::utils::is_ytensor_v<Arg>) {
            if (yt::strided::physicalSpansOverlap(target, arg)) {
                return Arg(arg.clone());
            }
        }
        return Arg(std::forward<decltype(arg)>(arg));
    };
    auto prepared = std::tuple{prepare(std::forward<Args>(tensors))...};
    return std::apply(
        [&](auto&... args) -> YTensorBase& {
            return applyTypeErasedBroadcastInplace(target, std::forward<Func>(func), args...);
        },
        prepared
    );
}

template <typename Func, typename... Args>
yt::YTensorBase broadcastBase(Func&& func, Args&&... tensors) {
    using namespace ::yt::utils;

    using ScalarT = ::yt::utils::first_arg_of_t<Func>;

    static_assert(
        all_valid_broadcast_args<ScalarT, Args...>(),
        "broadcastBase: all arguments must be either YTensor/YTensorBase or convertible to scalar type"
    );
    static_assert(
        (hasMatchingTypedScalar<ScalarT, Args> && ...),
        "broadcastBase: typed tensor scalar types must match the callable scalar type"
    );

    std::vector<std::vector<int>> shapes;
    [[maybe_unused]] auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if constexpr (!is_ytensor_template_v<std::decay_t<decltype(arg)>>) {
                if (!arg.isStrided()) {
                    throw std::runtime_error("strided::broadcastBase: layout not implemented");
                }
            }
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    if (shapes.empty()) {
        throw std::runtime_error("broadcastBase: at least one tensor argument required");
    }

    auto broadcastShape = yt::strided::computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());

    std::vector<int> logicStride(opdim);
    size_t stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = yt::utils::checkedIntSize(stride, "strided::broadcast logical stride");
        if (broadcastShape[i] != 0 &&
            stride > static_cast<size_t>(std::numeric_limits<int>::max()) / broadcastShape[i]) {
            throw std::overflow_error("strided::broadcast: element count exceeds int indexing range");
        }
        stride *= static_cast<size_t>(broadcastShape[i]);
    }
    int totalSize = yt::utils::checkedIntSize(stride, "strided::broadcast");

    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarT*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;
    std::string tensorDtype;

    [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (tensorDtype.empty()) tensorDtype = arg.dtype();
            if (arg.dtype() != tensorDtype ||
                (!yt::utils::is_ytensor_v<ScalarT> &&
                 arg.dtype() != yt::type::getTypeName<ScalarT>())) {
                throw std::invalid_argument("broadcastBase: tensor dtype does not match callable scalar type");
            }
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

    using ReturnType = std::invoke_result_t<
        Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarT, Args>>())...>;

    std::string resultDtype = yt::type::getTypeName<ReturnType>();

    yt::YTensorBase result(broadcastShape, resultDtype);
    ReturnType* resultPtr = result.template data<ReturnType>();

    if (allContiguous && allShapeEqual) {
        yt::utils::parallelFor(0, totalSize, [&](int index) {
            [[maybe_unused]] size_t tensorIdx = 0;
            [[maybe_unused]] auto getValueDirect = [&](auto&& arg) -> ScalarT {
                if constexpr (is_ytensor_v<std::decay_t<decltype(arg)>>) {
                    return dataPtrs[tensorIdx++][index];
                } else {
                    return static_cast<ScalarT>(arg);
                }
            };

            ScalarT values[] = {getValueDirect(tensors)...};

            [&]<size_t... Is>(std::index_sequence<Is...>) { resultPtr[index] = func(values[Is]...); }
            (std::make_index_sequence<sizeof...(Args)>{});
        });
        return result;
    }

    // runtime rank无法使用固定大小NaryIndexComputer，按实际tensor数量构造索引工作区。
    size_t numTensors = dataPtrs.size();

    yt::utils::parallelFor(0, totalSize, [&](int index) {
        std::vector<int> tensorIndices(numTensors, 0);
        for (int d = 0; d < opdim; ++d) {
            int posi = (index / logicStride[d]) % broadcastShape[d];
            for (size_t t = 0; t < numTensors; ++t) {
                checkedBroadcastOffsetAdd(tensorIndices[t], posi, broadcastStrides[t][d]);
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

        [&]<size_t... Is>(std::index_sequence<Is...>) { resultPtr[index] = func(values[Is]...); }
        (std::make_index_sequence<sizeof...(Args)>{});
    });

    return result;
}

// 校验二元广播输入：恰好两个非空输入，dtype一致，且均为Strided布局。
// 参数 inputs：输入张量指针列表
// 参数 opName：操作名称，用于错误消息
// 异常 std::invalid_argument：输入数量或空指针不合法时抛出
// 异常 std::runtime_error：layout或dtype不匹配时抛出
inline void validateBroadcastInputs(
    const std::vector<const YTensorBase*>& inputs, const std::string& opName
) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(opName + ": expected exactly two inputs");
    }
    if (inputs[0] == nullptr) throw std::invalid_argument(opName + ": input must not be null");
    auto dtype = inputs[0]->dtype();
    for (auto* t : inputs) {
        if (t == nullptr) throw std::invalid_argument(opName + ": input must not be null");
        if (!t->isStrided()) {
            throw std::runtime_error(opName + ": layout not implemented");
        }
        if (t->dtype() != dtype) {
            throw std::runtime_error(opName + ": dtype mismatch");
        }
    }
}

// 二元广播运算的泛型实现，支持连续快速路径与广播索引路径。
// 模板参数 Input：输入元素类型
// 模板参数 Output：输出元素类型
// 模板参数 Func：运算函数
template <typename Input, typename Output, typename Func>
inline void applyBroadcastOperation(
    YTensorBase& out, const YTensorBase& a, const YTensorBase& b, Func&& func
) {
    auto outShape = out.shape();
    auto logicStride = out.stride();
    auto aStride = getBroadcastStride(a.shape(), a.stride_(), outShape);
    auto bStride = getBroadcastStride(b.shape(), b.stride_(), outShape);
    auto outStride = getBroadcastStride(out.shape(), out.stride_(), outShape);
    Output* outData = out.data<Output>();
    const Input* aData = a.data<Input>();
    const Input* bData = b.data<Input>();
    int total = yt::utils::checkedIntSize(out.size(), "strided broadcast kernel");

    // 连续性本身不足以直接线性索引：shape不同仍可能包含zero-stride广播轴。
    if (out.isContiguous() && a.isContiguous() && b.isContiguous() && out.shapeMatch(a.shape()) &&
        out.shapeMatch(b.shape())) {
        forEachMutableBroadcastIndex(out, total, [&](int i) { outData[i] = func(aData[i], bData[i]); });
        return;
    }

    forEachMutableBroadcastIndex(out, total, [&](int i) {
        int aIndex = computeDataIndex(i, logicStride, aStride, outShape);
        int bIndex = computeDataIndex(i, logicStride, bStride, outShape);
        int outIndex = computeDataIndex(i, logicStride, outStride, outShape);
        outData[outIndex] = func(aData[aIndex], bData[bIndex]);
    });
}

// 判断broadcast op是否产生bool结果。
inline bool isComparisonBroadcast(yt::type::YBroadcastOp op) {
    return op == yt::type::YBroadcastOp::Less || op == yt::type::YBroadcastOp::LessEqual ||
           op == yt::type::YBroadcastOp::Greater || op == yt::type::YBroadcastOp::GreaterEqual ||
           op == yt::type::YBroadcastOp::Equal || op == yt::type::YBroadcastOp::NotEqual;
}

// 算术运算的预校验（如整数除零、移位越界检查）。
// 模板参数 T：元素类型
// 模板参数 Validator：校验回调
template <typename T, typename Validator>
inline void validateBroadcastOperation(
    const YTensorBase& out, const YTensorBase& a, const YTensorBase& b, Validator&& validator
) {
    auto outShape = out.shape();
    auto logicStride = out.stride();
    auto aStride = getBroadcastStride(a.shape(), a.stride_(), outShape);
    auto bStride = getBroadcastStride(b.shape(), b.stride_(), outShape);
    const T* aData = a.data<T>();
    const T* bData = b.data<T>();
    int total = yt::utils::checkedIntSize(out.size(), "strided broadcast validation");
    for (int index = 0; index < total; ++index) {
        int aIndex = computeDataIndex(index, logicStride, aStride, outShape);
        int bIndex = computeDataIndex(index, logicStride, bStride, outShape);
        validator(aData[aIndex], bData[bIndex]);
    }
}

// 模板化广播运算 kernel，op分支在元素循环**之外**，避免运行时反复分发。
// 模板参数 T：元素类型
// 参数 op：broadcast运算类型
// 参数 out：输出张量
// 参数 inputs：输入张量（当前仅支持2个）
// 针对 Div/Mod/LShift/RShift 做整数安全预校验；
// 比较运算输出 bool，其余输出与输入同类型。
template <typename T>
void stridedBroadcastKernel(
    yt::type::YBroadcastOp op, YTensorBase& out, const std::vector<const YTensorBase*>& inputs
) {
    if (inputs.size() != 2) {
        throw std::runtime_error("strided broadcast: expected two inputs");
    }
    const YTensorBase& a = *inputs[0];
    const YTensorBase& b = *inputs[1];
    // 会抛domain error的整数操作先完成全量验证，再进入可能并行的写入pass。
    switch (op) {
        case yt::type::YBroadcastOp::Add:
            applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) {
                return modularAdd(x, y);
            });
            break;
        case yt::type::YBroadcastOp::Sub:
            applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) {
                return modularSub(x, y);
            });
            break;
        case yt::type::YBroadcastOp::Mul:
            applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) {
                return modularMul(x, y);
            });
            break;
        case yt::type::YBroadcastOp::Div:
            if constexpr (std::is_integral_v<T>) {
                validateBroadcastOperation<T>(out, a, b, [](const T& x, const T& y) {
                    bool overflow = false;
                    if constexpr (std::is_signed_v<T>) {
                        overflow = x == std::numeric_limits<T>::min() && y == static_cast<T>(-1);
                    }
                    if (y == 0 || overflow) {
                        throw std::domain_error("strided broadcast: invalid integer division");
                    }
                });
            }
            applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x / y; });
            break;
        case yt::type::YBroadcastOp::Mod:
            if constexpr (std::is_integral_v<T>) {
                validateBroadcastOperation<T>(out, a, b, [](const T& x, const T& y) {
                    bool overflow = false;
                    if constexpr (std::is_signed_v<T>) {
                        overflow = x == std::numeric_limits<T>::min() && y == static_cast<T>(-1);
                    }
                    if (y == 0 || overflow) {
                        throw std::domain_error("strided broadcast: invalid integer modulo");
                    }
                });
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x % y; });
            } else if constexpr (std::is_floating_point_v<T>) {
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) {
                    return std::fmod(x, y);
                });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support %");
            }
            break;
        case yt::type::YBroadcastOp::BitAnd:
            if constexpr (std::is_integral_v<T>) {
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x & y; });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support &");
            }
            break;
        case yt::type::YBroadcastOp::BitOr:
            if constexpr (std::is_integral_v<T>) {
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x | y; });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support |");
            }
            break;
        case yt::type::YBroadcastOp::BitXor:
            if constexpr (std::is_integral_v<T>) {
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x ^ y; });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support ^");
            }
            break;
        case yt::type::YBroadcastOp::LShift:
            if constexpr (std::is_integral_v<T>) {
                validateBroadcastOperation<T>(out, a, b, [](const T&, const T& y) {
                    if (y < 0 || static_cast<uint64_t>(y) >= sizeof(T) * 8) {
                        throw std::domain_error("strided broadcast: invalid left shift count");
                    }
                });
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) {
                    return bitPreservingLeftShift(x, y);
                });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support <<");
            }
            break;
        case yt::type::YBroadcastOp::RShift:
            if constexpr (std::is_integral_v<T>) {
                validateBroadcastOperation<T>(out, a, b, [](const T&, const T& y) {
                    if (y < 0 || static_cast<uint64_t>(y) >= sizeof(T) * 8) {
                        throw std::domain_error("strided broadcast: invalid right shift count");
                    }
                });
                applyBroadcastOperation<T, T>(out, a, b, [](const T& x, const T& y) { return x >> y; });
            } else {
                throw std::runtime_error("strided broadcast: dtype does not support >>");
            }
            break;
        case yt::type::YBroadcastOp::Less:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x < y; });
            break;
        case yt::type::YBroadcastOp::LessEqual:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x <= y; });
            break;
        case yt::type::YBroadcastOp::Greater:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x > y; });
            break;
        case yt::type::YBroadcastOp::GreaterEqual:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x >= y; });
            break;
        case yt::type::YBroadcastOp::Equal:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x == y; });
            break;
        case yt::type::YBroadcastOp::NotEqual:
            applyBroadcastOperation<T, bool>(out, a, b, [](const T& x, const T& y) { return x != y; });
            break;
        default:
            throw std::runtime_error("strided broadcast: op not implemented");
    }
}

// bool dtype仅开放比较kernel；算术/位运算不在YTensor bool合同内。
inline void stridedBoolComparisonKernel(
    yt::type::YBroadcastOp op, YTensorBase& out, const std::vector<const YTensorBase*>& inputs
) {
    if (inputs.size() != 2) {
        throw std::runtime_error("strided bool broadcast: expected two inputs");
    }
    const YTensorBase& a = *inputs[0];
    const YTensorBase& b = *inputs[1];
    switch (op) {
        case yt::type::YBroadcastOp::Less:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x < y; });
            return;
        case yt::type::YBroadcastOp::LessEqual:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x <= y; });
            return;
        case yt::type::YBroadcastOp::Greater:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x > y; });
            return;
        case yt::type::YBroadcastOp::GreaterEqual:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x >= y; });
            return;
        case yt::type::YBroadcastOp::Equal:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x == y; });
            return;
        case yt::type::YBroadcastOp::NotEqual:
            applyBroadcastOperation<bool, bool>(out, a, b, [](bool x, bool y) { return x != y; });
            return;
        default:
            throw std::runtime_error("strided bool broadcast: only comparisons are supported");
    }
}

// 注册 T 类型的内置广播 kernel（使用 mergeDTypeKernels，不覆盖已有注册）。
template <typename T>
inline void registerBuiltinKernel(const std::string& dtype) {
    yt::type::YDTypeKernels kernels;
    kernels.broadcast = &stridedBroadcastKernel<T>;
    kernels.broadcastInplace = &stridedBroadcastKernel<T>;
    yt::type::mergeDTypeKernels(dtype, kernels);
}

// 确保内置数值类型的广播 kernel 均已注册（惰性初始化，第一次调用时执行一次）。
// 注意：mergeDTypeKernels只填充空槽，不会覆盖用户在首次调用前注册的实现。
inline void ensureBuiltinKernels() {
    static const bool initialized = []() {
        registerBuiltinKernel<float>("float32");
        registerBuiltinKernel<double>("float64");
        registerBuiltinKernel<int8_t>("int8");
        registerBuiltinKernel<int16_t>("int16");
        registerBuiltinKernel<int32_t>("int32");
        registerBuiltinKernel<int64_t>("int64");
        registerBuiltinKernel<uint8_t>("uint8");
        registerBuiltinKernel<uint16_t>("uint16");
        registerBuiltinKernel<uint32_t>("uint32");
        registerBuiltinKernel<uint64_t>("uint64");
        yt::type::YDTypeKernels boolKernels;
        boolKernels.broadcast = &stridedBoolComparisonKernel;
        boolKernels.broadcastInplace = &stridedBoolComparisonKernel;
        yt::type::mergeDTypeKernels("bool", boolKernels);
        registerBuiltinKernel<yt::bfloat16>("bfloat16");
        registerBuiltinKernel<yt::float16>("float16");
        registerBuiltinKernel<yt::float8_e5m2>("float8_e5m2");
        registerBuiltinKernel<yt::float8_e4m3>("float8_e4m3");
        registerBuiltinKernel<yt::float8_e8m0>("float8_e8m0");
        registerBuiltinKernel<yt::float8_ue8m0>("float8_ue8m0");
        return true;
    }();
    (void)initialized;
}

template <typename T>
YT_IMPL_INLINE bool isBuiltinBroadcastKernel(const yt::type::YDTypeKernels& kernels) {
    return kernels.broadcastInplace == &stridedBroadcastKernel<T>;
}

YT_IMPL_INLINE bool isBuiltinBroadcastKernel(
    const std::string& dtype, const yt::type::YDTypeKernels& kernels
) {
    if (dtype == "float32") return isBuiltinBroadcastKernel<float>(kernels);
    if (dtype == "float64") return isBuiltinBroadcastKernel<double>(kernels);
    if (dtype == "int8") return isBuiltinBroadcastKernel<int8_t>(kernels);
    if (dtype == "int16") return isBuiltinBroadcastKernel<int16_t>(kernels);
    if (dtype == "int32") return isBuiltinBroadcastKernel<int32_t>(kernels);
    if (dtype == "int64") return isBuiltinBroadcastKernel<int64_t>(kernels);
    if (dtype == "uint8") return isBuiltinBroadcastKernel<uint8_t>(kernels);
    if (dtype == "uint16") return isBuiltinBroadcastKernel<uint16_t>(kernels);
    if (dtype == "uint32") return isBuiltinBroadcastKernel<uint32_t>(kernels);
    if (dtype == "uint64") return isBuiltinBroadcastKernel<uint64_t>(kernels);
    if (dtype == "bfloat16") return isBuiltinBroadcastKernel<yt::bfloat16>(kernels);
    if (dtype == "float16") return isBuiltinBroadcastKernel<yt::float16>(kernels);
    if (dtype == "float8_e5m2") return isBuiltinBroadcastKernel<yt::float8_e5m2>(kernels);
    if (dtype == "float8_e4m3") return isBuiltinBroadcastKernel<yt::float8_e4m3>(kernels);
    if (dtype == "float8_e8m0") return isBuiltinBroadcastKernel<yt::float8_e8m0>(kernels);
    if (dtype == "float8_ue8m0") return isBuiltinBroadcastKernel<yt::float8_ue8m0>(kernels);
    if (dtype == "bool") return kernels.broadcastInplace == &stridedBoolComparisonKernel;
    return false;
}

// 统一 runtime broadcast 入口：校验、计算广播shape、查找dtype kernel、派发执行。
// 参数 op：broadcast运算类型
// 参数 inputs：参与广播的输入张量，dtype必须一致
// 参数 outputDtype：输出dtype（比较操作用"bool"，其余同输入）
// 返回：广播后的 YTensorBase
YT_IMPL_INLINE YTensorBase
broadcast(yt::type::YBroadcastOp op, const std::vector<const YTensorBase*>& inputs, const std::string& outputDtype) {
    validateBroadcastInputs(inputs, "strided::broadcast");
    const std::string expectedDtype = isComparisonBroadcast(op) ? "bool" : inputs[0]->dtype();
    if (outputDtype != expectedDtype) {
        throw std::invalid_argument("strided::broadcast: output dtype does not match operation");
    }
    std::vector<std::vector<int>> shapes;
    shapes.reserve(inputs.size());
    for (auto* t : inputs) {
        shapes.push_back(t->shape());
    }
    auto outShape = computeBroadcastShape(shapes);
    YTensorBase out(outShape, outputDtype);
    ensureBuiltinKernels();
    const auto& kernels = yt::type::getDTypeKernels(inputs[0]->dtype());
    if (!kernels.broadcast) {
        throw std::runtime_error("strided::broadcast: dtype kernel not implemented");
    }
    kernels.broadcast(op, out, inputs);
    return out;
}

// 统一 runtime broadcast 原地变体：通过 YDTypeKernels::broadcastInplace 派发。
// 参数 op：broadcast运算类型
// 参数 output：已有输出张量，shape须与广播结果一致
// 参数 inputs：输入张量
// 返回：output 的引用
YT_IMPL_INLINE YTensorBase&
broadcast_(yt::type::YBroadcastOp op, YTensorBase& output, const std::vector<const YTensorBase*>& inputs) {
    validateBroadcastInputs(inputs, "strided::broadcast_");
    if (!output.isStrided()) throw std::runtime_error("strided::broadcast_: layout not implemented");
    const std::string expectedDtype = isComparisonBroadcast(op) ? "bool" : inputs[0]->dtype();
    if (output.dtype() != expectedDtype) {
        throw std::invalid_argument("strided::broadcast_: output dtype does not match operation");
    }
    std::vector<std::vector<int>> shapes;
    shapes.reserve(inputs.size());
    for (auto* t : inputs) {
        shapes.push_back(t->shape());
    }
    auto outShape = computeBroadcastShape(shapes);
    if (static_cast<int>(outShape.size()) != output.ndim() || !output.shapeMatch(outShape)) {
        throw std::runtime_error("strided::broadcast_: output shape must match broadcast shape");
    }
    ensureBuiltinKernels();
    const auto& kernels = yt::type::getDTypeKernels(inputs[0]->dtype());
    if (!kernels.broadcastInplace) {
        throw std::runtime_error("strided::broadcast_: dtype kernel not implemented");
    }
    // 内置kernel遵守本文件的串行重叠写入合同；custom kernel没有该证明，要求output disjoint。
    if (!output.isDisjoint() && !isBuiltinBroadcastKernel(inputs[0]->dtype(), kernels)) {
        throw std::runtime_error(
            "strided::broadcast_: custom dtype kernel requires a disjoint mutable output"
        );
    }
    // Inplace output可能与shift/transpose/unfold input共享storage；先快照可避免并行写入污染后续读取。
    std::vector<YTensorBase> snapshots;
    snapshots.reserve(inputs.size());
    std::vector<const YTensorBase*> safeInputs;
    safeInputs.reserve(inputs.size());
    for (const YTensorBase* input : inputs) {
        if (yt::strided::physicalSpansOverlap(output, *input)) {
            snapshots.push_back(yt::strided::clone(*input));
            safeInputs.push_back(&snapshots.back());
        } else {
            safeInputs.push_back(input);
        }
    }
    kernels.broadcastInplace(op, output, safeInputs);
    return output;
}

// 将标量转换为 rank-0 YTensorBase，dtype 与目标张量一致。
// 模板参数 Scalar：标量类型
// 参数 tensor：目标张量（用于参考 dtype）
// 参数 scalar：标量值
// 返回：rank-0 张量；标量按目标dtype的既有转换规则写入
template <typename Scalar>
YT_IMPL_INLINE YTensorBase
scalarTensor(const YTensorBase& tensor, const Scalar& scalar) {
    using ScalarType = std::remove_cv_t<Scalar>;
    auto materialize = []<typename Source>(const Source& value, const std::string& dtype) {
        auto typeInfo = yt::type::getTypeInfo(dtype);
        if (!typeInfo) {
            throw std::invalid_argument("strided::scalarTensor: scalar dtype is not registered");
        }
        return BroadcastAccess::scalarFromRegisteredValue(dtype, &value, typeInfo->get());
    };

    // 优先保留显式注册的C++ scalar类型；否则归一化到库内置canonical dtype再走统一cast。
    YTensorBase source;
    std::string sourceDtype;
    {
        std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
        const auto& registry = yt::type::internal::getMutableTypeRegistry();
        auto registered = registry.find(typeid(ScalarType).name());
        if (registered != registry.end()) sourceDtype = registered->second.name;
    }

    if (!sourceDtype.empty()) {
        source = materialize(scalar, sourceDtype);
    } else if constexpr (yt::type::TypeListContains_v<ScalarType, yt::type::ExtendedFloatTypes>) {
        sourceDtype = yt::type::getBaseTypeName<ScalarType>();
        source = materialize(scalar, sourceDtype);
    } else if constexpr (std::is_enum_v<ScalarType>) {
        using SourceType = std::underlying_type_t<ScalarType>;
        SourceType canonical = static_cast<SourceType>(scalar);
        sourceDtype = yt::type::getBaseTypeName<SourceType>();
        source = materialize(canonical, sourceDtype);
    } else if constexpr (std::is_same_v<ScalarType, bool>) {
        sourceDtype = "bool";
        source = materialize(scalar, sourceDtype);
    } else if constexpr (std::is_floating_point_v<ScalarType>) {
        using SourceType = std::conditional_t<std::is_same_v<ScalarType, float>, float, double>;
        SourceType canonical = static_cast<SourceType>(scalar);
        sourceDtype = yt::type::getBaseTypeName<SourceType>();
        source = materialize(canonical, sourceDtype);
    } else if constexpr (std::is_integral_v<ScalarType>) {
        using SourceType = std::conditional_t<
            std::is_signed_v<ScalarType>,
            std::conditional_t<(sizeof(ScalarType) <= 1), int8_t,
                std::conditional_t<(sizeof(ScalarType) <= 2), int16_t,
                    std::conditional_t<(sizeof(ScalarType) <= 4), int32_t, int64_t>>>,
            std::conditional_t<(sizeof(ScalarType) <= 1), uint8_t,
                std::conditional_t<(sizeof(ScalarType) <= 2), uint16_t,
                    std::conditional_t<(sizeof(ScalarType) <= 4), uint32_t, uint64_t>>>>;
        SourceType canonical = static_cast<SourceType>(scalar);
        sourceDtype = yt::type::getBaseTypeName<SourceType>();
        source = materialize(canonical, sourceDtype);
    } else {
        throw std::invalid_argument("strided::scalarTensor: scalar dtype is not registered");
    }
    if (sourceDtype == tensor.dtype()) return source;

    // 复用copy_可集中执行pairwise cast、范围校验和custom dtype转换合同。
    YTensorBase result({}, tensor.dtype());
    yt::strided::copy_(result, source);
    return result;
}

}  // namespace yt::strided
