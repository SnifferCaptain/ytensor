/***************
 * file: ytensor_math.inl
 * purpose: YTensor<T, dim>算术operator和数学facade实现。
 ***************/

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <typeinfo>

#include "../include/strided/broadcast.hpp"
#include "../include/ytensor_infos.hpp"

// typed callable原地广播facade；layout选择后由Strided owner处理alias和索引。
template <typename T, int dim>
template <typename Func, typename... Args>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::broadcastInplace(Func&& func, Args&&... tensors) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::broadcastInplace(
                *this, std::forward<Func>(func), std::forward<Args>(tensors)...
            );
        default:
            throw std::runtime_error("YTensor::broadcastInplace: layout not implemented");
    }
}

// ==================== arithmetic and comparison operators ====================

// 该宏生成tensor/scalar的创建型和原地operator：
// 1. compile-time concept决定T是否支持操作；
// 2. 可能失败的整数输入在写入前全量预校验；
// 3. layout facade只选择owner，广播/alias/重叠写入规则由yt::strided统一维护。
// ENABLE_IF_T_INPLACE与ENABLE_IF_T分支当前行为一致，但保留二者以表达未来可单独限制赋值能力的concept合同。
#define YT_YTENSOR_OPERATOR(OP, ENABLE_IF_T, OP_ID)                                            \
    template <typename T, int dim>                                                             \
    template <int dim1>                                                                        \
    auto yt::YTensor<T, dim>::operator OP(const yt::YTensor<T, dim1>& other) const {           \
        if constexpr (ENABLE_IF_T<T>) {                                                        \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            switch (layoutType()) {                                                            \
                case YLayoutType::Strided:                                                     \
                    return yt::strided::broadcast(                                             \
                        [](const T& a, const T& b) {                                           \
                            return yt::strided::typedBinaryArithmetic<OP_ID>(a, b);            \
                        }, *this, other                                                        \
                    );                                                                         \
                default:                                                                       \
                    throw std::runtime_error(std::string(#OP) + ": layout not implemented");   \
            }                                                                                  \
        } else {                                                                               \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                    \
        }                                                                                      \
    }                                                                                          \
                                                                                               \
    template <typename T, int dim>                                                             \
    template <int dim1>                                                                        \
    yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator OP##=(                                  \
        const yt::YTensor<T, dim1>& other                                                      \
    ) {                                                                                        \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                              \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            return this->broadcastInplace([](T& a, const T& b) {                               \
                a = yt::strided::typedBinaryArithmetic<OP_ID>(a, b);                           \
            }, other);                                                                         \
        } else if constexpr (ENABLE_IF_T<T>) {                                                 \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            return this->broadcastInplace([](T& a, const T& b) {                               \
                a = yt::strided::typedBinaryArithmetic<OP_ID>(a, b);                           \
            }, other);                                                                         \
        } else {                                                                               \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                 \
        }                                                                                      \
    }                                                                                          \
                                                                                               \
    template <typename T, int dim>                                                             \
    auto yt::YTensor<T, dim>::operator OP(const T& other) const {                              \
        if constexpr (ENABLE_IF_T<T>) {                                                        \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            switch (layoutType()) {                                                            \
                case YLayoutType::Strided:                                                     \
                    return yt::strided::broadcast(                                             \
                        [](const T& a, const T& b) {                                           \
                            return yt::strided::typedBinaryArithmetic<OP_ID>(a, b);            \
                        }, *this, other                                                        \
                    );                                                                         \
                default:                                                                       \
                    throw std::runtime_error(std::string(#OP) + ": layout not implemented");   \
            }                                                                                  \
        } else {                                                                               \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                    \
        }                                                                                      \
    }                                                                                          \
                                                                                               \
    template <typename T, int dim>                                                             \
    yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator OP##=(const T& other) {                 \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                              \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            return broadcastInplace([](T& a, const T& b) {                                    \
                a = yt::strided::typedBinaryArithmetic<OP_ID>(a, b);                           \
            }, other);                                                                         \
        } else if constexpr (ENABLE_IF_T<T>) {                                                 \
            yt::strided::validateTypedBinaryOperation<OP_ID>(*this, other);                    \
            return broadcastInplace([](T& a, const T& b) {                                    \
                a = yt::strided::typedBinaryArithmetic<OP_ID>(a, b);                           \
            }, other);                                                                         \
        } else {                                                                               \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                 \
            return *this;                                                                      \
        }                                                                                      \
    }

YT_YTENSOR_OPERATOR(+, yt::utils::HAVE_ADD, yt::type::YBroadcastOp::Add)
YT_YTENSOR_OPERATOR(-, yt::utils::HAVE_SUB, yt::type::YBroadcastOp::Sub)
YT_YTENSOR_OPERATOR(*, yt::utils::HAVE_MUL, yt::type::YBroadcastOp::Mul)
YT_YTENSOR_OPERATOR(/, yt::utils::HAVE_DIV, yt::type::YBroadcastOp::Div)
// %有特殊处理
YT_YTENSOR_OPERATOR(&, yt::utils::HAVE_AND, yt::type::YBroadcastOp::BitAnd)
YT_YTENSOR_OPERATOR(|, yt::utils::HAVE_OR, yt::type::YBroadcastOp::BitOr)
YT_YTENSOR_OPERATOR(^, yt::utils::HAVE_XOR, yt::type::YBroadcastOp::BitXor)
YT_YTENSOR_OPERATOR(<<, yt::utils::HAVE_LSHIFT, yt::type::YBroadcastOp::LShift)
YT_YTENSOR_OPERATOR(>>, yt::utils::HAVE_RSHIFT, yt::type::YBroadcastOp::RShift)

#undef YT_YTENSOR_OPERATOR

// 比较运算复用相同broadcast shape规则，但结果scalar固定为bool。
#define YT_YTENSOR_CMP_OPERATOR(OP, ENABLE_IF_T)                                             \
    template <typename T, int dim>                                                           \
    template <int dim1>                                                                      \
    auto yt::YTensor<T, dim>::operator OP(const yt::YTensor<T, dim1>& other) const {         \
        if constexpr (ENABLE_IF_T<T>) {                                                      \
            switch (layoutType()) {                                                          \
                case YLayoutType::Strided:                                                   \
                    return yt::strided::broadcast(                                           \
                        [](const T& a, const T& b) { return a OP b; }, *this, other          \
                    );                                                                       \
                default:                                                                     \
                    throw std::runtime_error(std::string(#OP) + ": layout not implemented"); \
            }                                                                                \
        } else {                                                                             \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                  \
        }                                                                                    \
    }                                                                                        \
                                                                                             \
    template <typename T, int dim>                                                           \
    auto yt::YTensor<T, dim>::operator OP(const T& other) const {                            \
        if constexpr (ENABLE_IF_T<T>) {                                                      \
            switch (layoutType()) {                                                          \
                case YLayoutType::Strided:                                                   \
                    return yt::strided::broadcast(                                           \
                        [](const T& a, const T& b) { return a OP b; }, *this, other          \
                    );                                                                       \
                default:                                                                     \
                    throw std::runtime_error(std::string(#OP) + ": layout not implemented"); \
            }                                                                                \
        } else {                                                                             \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                  \
        }                                                                                    \
    }

YT_YTENSOR_CMP_OPERATOR(<, yt::utils::HAVE_LT)
YT_YTENSOR_CMP_OPERATOR(<=, yt::utils::HAVE_LE)
YT_YTENSOR_CMP_OPERATOR(>, yt::utils::HAVE_GT)
YT_YTENSOR_CMP_OPERATOR(>=, yt::utils::HAVE_GE)
YT_YTENSOR_CMP_OPERATOR(==, yt::utils::HAVE_EQ)
YT_YTENSOR_CMP_OPERATOR(!=, yt::utils::HAVE_NEQ)

#undef YT_YTENSOR_CMP_OPERATOR

// ==================== modulo specializations ====================

// modulo单独处理：整数/custom operator走checkedModulo，内置浮点走std::fmod。
// 注意：整数除零和min/-1在并行写入前验证，失败不会产生部分结果。
template <typename T, int dim>
template <int dim1>
auto yt::YTensor<T, dim>::operator%(const yt::YTensor<T, dim1>& other) const {
    if constexpr (yt::utils::HAVE_MOD<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        switch (layoutType()) {
            case YLayoutType::Strided:
                return yt::strided::broadcast(
                    [](const T& a, const T& b) { return yt::strided::checkedModulo(a, b); }, *this, other
                );
            default:
                throw std::runtime_error("%: layout not implemented");
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        switch (layoutType()) {
            case YLayoutType::Strided:
                return yt::strided::broadcast(
                    [](const T& a, const T& b) { return std::fmod(a, b); }, *this, other
                );
            default:
                throw std::runtime_error("%: layout not implemented");
        }
    } else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
        return yt::strided::broadcast([](const T& a, const T&) { return a; }, *this, other);
    }
}

template <typename T, int dim>
template <int dim1>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator%=(const yt::YTensor<T, dim1>& other) {
    if constexpr (yt::utils::HAVE_MOD_INPLACE<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        return broadcastInplace([](T& a, const T& b) { a = yt::strided::checkedModulo(a, b); }, other);
    } else if constexpr (yt::utils::HAVE_MOD<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        return broadcastInplace([](T& a, const T& b) { a = yt::strided::checkedModulo(a, b); }, other);
    } else if constexpr (std::is_floating_point_v<T>) {
        return broadcastInplace([](T& a, const T& b) { a = fmod(a, b); }, other);
    } else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
        return *this; /* unreachable */
    }
}

template <typename T, int dim>
auto yt::YTensor<T, dim>::operator%(const T& other) const {
    if constexpr (yt::utils::HAVE_MOD<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        switch (layoutType()) {
            case YLayoutType::Strided:
                return yt::strided::broadcast(
                    [](const T& a, const T& b) { return yt::strided::checkedModulo(a, b); }, *this, other
                );
            default:
                throw std::runtime_error("%: layout not implemented");
        }
    } else if constexpr (std::is_floating_point_v<T>) {
        switch (layoutType()) {
            case YLayoutType::Strided:
                return yt::strided::broadcast(
                    [](const T& a, const T& b) { return std::fmod(a, b); }, *this, other
                );
            default:
                throw std::runtime_error("%: layout not implemented");
        }
    } else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
        return yt::strided::broadcast([](const T& a, const T&) { return a; }, *this, other); /* unreachable */
    }
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator%=(const T& other) {
    if constexpr (yt::utils::HAVE_MOD_INPLACE<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        return broadcastInplace([](T& a, const T& b) { a = yt::strided::checkedModulo(a, b); }, other);
    } else if constexpr (yt::utils::HAVE_MOD<T>) {
        yt::strided::validateTypedBinaryOperation<yt::type::YBroadcastOp::Mod>(*this, other);
        return broadcastInplace([](T& a, const T& b) { a = yt::strided::checkedModulo(a, b); }, other);
    } else if constexpr (std::is_floating_point_v<T>) {
        return broadcastInplace([](T& a, const T& b) { a = fmod(a, b); }, other);
    } else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
        return *this; /* unreachable */
    }
}

// ==================== matrix and reduction facades ====================

// 将最后两个轴包装为matrix元素；wrapper共享原scalar storage。
template <typename T, int dim>
yt::YTensor<yt::YTensor<T, 2>, std::max(1, dim - 2)> yt::YTensor<T, dim>::matView() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matView(*this);
        default:
            throw std::runtime_error("YTensor::matView: layout not implemented");
    }
}

// typed matmul facade；rank-1按1xN解释，输出rank至少为2且batch轴执行广播。
template <typename T, int dim>
template <int dim1>
yt::YTensor<T, yt::utils::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::matmul(
    const yt::YTensor<T, dim1>& other, yt::info::MatmulBackend backend
) const {
    static_assert(yt::utils::HAVE_ADD<T> && yt::utils::HAVE_MUL<T>, "Type must have add and mul in matmul");
    static_assert(dim >= 1 && dim1 >= 1, "matmul only support dim >= 1");
    const int rightRows = (dim1 == 1) ? 1 : other.shape(dim1 - 2);
    if (this->shape(-1) != rightRows) {
        throwShapeNotMatch("matmul", other.shape());
    }

    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matmul(*this, other, backend);
        default:
            throw std::runtime_error("YTensor::matmul: layout not implemented");
    }
}

// 二维bool mask按输出(row,col)寻址，并复用于所有广播batch。
template <typename T, int dim>
template <int dim1>
yt::YTensor<T, yt::utils::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::masked_matmul(
    const yt::YTensor<T, dim1>& other, const yt::YTensor<bool, 2>& mask, const T& maskedValue,
    yt::info::MatmulBackend backend
) const {
    static_assert(
        yt::utils::HAVE_ADD<T> && yt::utils::HAVE_MUL<T>, "Type must have add and mul in masked_matmul"
    );
    static_assert(dim >= 1 && dim1 >= 1, "masked_matmul only support dim >= 1");
    const int rightRows = (dim1 == 1) ? 1 : other.shape(dim1 - 2);
    if (this->shape(-1) != rightRows) {
        throwShapeNotMatch("masked_matmul", other.shape());
    }

    const int outRows = (dim == 1) ? 1 : this->shape(dim - 2);
    const int outCols = other.shape(-1);
    if (mask.shape(0) != outRows || mask.shape(1) != outCols) {
        throw std::invalid_argument(
            "Function \"masked_matmul\" mask shape not match: expected YTensor[" + std::to_string(outRows) +
            ", " + std::to_string(outCols) + "] but got YTensor[" + std::to_string(mask.shape(0)) + ", " +
            std::to_string(mask.shape(1)) + "]"
        );
    }

    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::masked_matmul(*this, other, mask, maskedValue, backend);
        default:
            throw std::runtime_error("YTensor::masked_matmul: layout not implemented");
    }
}

// callable mask版本；backend可并行调用predicate，调用方需保证其线程安全。
template <typename T, int dim>
template <int dim1, typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    yt::YTensor<T, yt::utils::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::masked_matmul(
        const yt::YTensor<T, dim1>& other, Func&& func, const T& maskedValue, yt::info::MatmulBackend backend
    )
const {
    static_assert(
        yt::utils::HAVE_ADD<T> && yt::utils::HAVE_MUL<T>, "Type must have add and mul in masked_matmul"
    );
    static_assert(dim >= 1 && dim1 >= 1, "masked_matmul only support dim >= 1");
    static_assert(
        std::is_invocable_r_v<bool, std::decay_t<Func>, int, int>,
        "masked_matmul func must be callable as bool(int, int)"
    );
    const int rightRows = (dim1 == 1) ? 1 : other.shape(dim1 - 2);
    if (this->shape(-1) != rightRows) {
        throwShapeNotMatch("masked_matmul", other.shape());
    }

    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::masked_matmul(*this, other, std::forward<Func>(func), maskedValue, backend);
        default:
            throw std::runtime_error("YTensor::masked_matmul: layout not implemented");
    }
}

// reduction在rank>1时保留被归约轴为extent 1，rank-1重载返回scalar。
template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::sum(int axis) const requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::sum(*this, axis);
        default:
            throw std::runtime_error("YTensor::sum: layout not implemented");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::sum(std::vector<int> axis) const requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::sum(*this, std::move(axis));
        default:
            throw std::runtime_error("YTensor::sum: layout not implemented");
    }
}

template <typename T, int dim>
T yt::YTensor<T, dim>::sum(int) const requires(dim == 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::sum(*this, 0);
        default:
            throw std::runtime_error("YTensor::sum: layout not implemented");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::mean(int axis) const requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mean(*this, axis);
        default:
            throw std::runtime_error("YTensor::mean: layout not implemented");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::mean(std::vector<int> axes) const requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mean(*this, std::move(axes));
        default:
            throw std::runtime_error("YTensor::mean: layout not implemented");
    }
}

template <typename T, int dim>
T yt::YTensor<T, dim>::mean(int) const requires(dim == 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mean(*this, 0);
        default:
            throw std::runtime_error("YTensor::mean: layout not implemented");
    }
}

template <typename T, int dim>
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> yt::YTensor<T, dim>::max(int axis) const
    requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::max(*this, axis);
        default:
            throw std::runtime_error("YTensor::max: layout not implemented");
    }
}

template <typename T, int dim>
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> yt::YTensor<T, dim>::max(std::vector<int> axis) const
    requires(dim > 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::max(*this, std::move(axis));
        default:
            throw std::runtime_error("YTensor::max: layout not implemented");
    }
}

template <typename T, int dim>
std::pair<T, int> yt::YTensor<T, dim>::max(int) const requires(dim == 1) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::max(*this, 0);
        default:
            throw std::runtime_error("YTensor::max: layout not implemented");
    }
}

// ==================== borrowed Eigen maps ====================

#if YT_USE_EIGEN
// 返回的Map只借用tensor scalar storage；替换/释放tensor storage后Map失效。
template <typename T, int dim>
yt::YTensor<typename yt::YTensor<T, dim>::EigenMatrixMap, std::max(1, dim - 2)>
yt::YTensor<T, dim>::matViewEigen() const requires(dim > 2) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matViewEigen(*this);
        default:
            throw std::runtime_error("YTensor::matViewEigen: layout not implemented");
    }
}

template <typename T, int dim>
typename yt::YTensor<T, dim>::EigenMatrixMap yt::YTensor<T, dim>::matViewEigen() const requires(dim <= 2) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matViewEigen(*this);
        default:
            throw std::runtime_error("YTensor::matViewEigen: layout not implemented");
    }
}

#endif  // YT_USE_EIGEN
