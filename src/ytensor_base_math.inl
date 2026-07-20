/***************
 * file: ytensor_base_math.inl
 * purpose: YTensorBase 数学运算的实现
 * author: SnifferCaptain
 ***************/

#include "../include/strided/broadcast.hpp"
#include "../include/strided/matmul.hpp"
#include "../include/strided/reduce.hpp"
#include "../include/strided/view.hpp"

namespace yt {

// YTensorBase::broadcastInplace 的泛型版本，按布局类型派发到具体后端。
// 注意：facade 函数：根据 layoutType() 路由到 yt::strided::broadcastInplaceBase 等。
template <typename Func, typename... Args>
YTensorBase& YTensorBase::broadcastInplace(Func&& func, Args&&... tensors) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            // alias快照属于mutable owner核心，保证直接调用和facade调用具有同一安全语义。
            return yt::strided::broadcastInplaceBase(
                *this, std::forward<Func>(func), std::forward<Args>(tensors)...
            );
        default:
            throw std::runtime_error("YTensorBase::broadcastInplace: layout not implemented");
    }
}

// YT_IMPL_BINARY_OP宏：
// 算术运算符 facade 宏：生成 operator OP 与 operator OP=，
// 将标量通过 scalarTensor 转为 rank-0 张量后统一走 strided::broadcast / broadcast_ 路由。
#define YT_IMPL_BINARY_OP(OP, OP_NAME, OP_ID)                                                 \
    YT_IMPL_INLINE YTensorBase YTensorBase::operator OP(const YTensorBase& other) const {     \
        switch (layoutType()) {                                                               \
            case YLayoutType::Strided:                                                        \
                return yt::strided::broadcast(OP_ID, {this, &other}, dtype());                 \
            default:                                                                          \
                throw std::runtime_error(std::string(OP_NAME) + ": layout not implemented");  \
        }                                                                                     \
    }                                                                                         \
    YT_IMPL_INLINE YTensorBase& YTensorBase::operator OP##=(const YTensorBase& other) {       \
        switch (layoutType()) {                                                               \
            case YLayoutType::Strided:                                                        \
                return yt::strided::broadcast_(OP_ID, *this, {this, &other});                 \
            default:                                                                          \
                throw std::runtime_error(std::string(OP_NAME) + "=: layout not implemented"); \
        }                                                                                     \
    }                                                                                         \
    template <typename T>                                                                     \
    YTensorBase YTensorBase::operator OP(const T& scalar) const {                             \
        switch (layoutType()) {                                                               \
            case YLayoutType::Strided:                                                        \
                {                                                                             \
                    auto scalarInput = yt::strided::scalarTensor(*this, scalar);               \
                    return yt::strided::broadcast(OP_ID, {this, &scalarInput}, dtype());       \
                }                                                                             \
            default:                                                                          \
                throw std::runtime_error(std::string(OP_NAME) + ": layout not implemented");  \
        }                                                                                     \
    }                                                                                         \
    template <typename T>                                                                     \
    YTensorBase& YTensorBase::operator OP##=(const T& scalar) {                               \
        switch (layoutType()) {                                                               \
            case YLayoutType::Strided:                                                        \
                {                                                                             \
                    auto scalarInput = yt::strided::scalarTensor(*this, scalar);               \
                    return yt::strided::broadcast_(OP_ID, *this, {this, &scalarInput});        \
                }                                                                             \
            default:                                                                          \
                throw std::runtime_error(std::string(OP_NAME) + "=: layout not implemented"); \
        }                                                                                     \
    }

YT_IMPL_BINARY_OP(+, "+", yt::type::YBroadcastOp::Add)
YT_IMPL_BINARY_OP(-, "-", yt::type::YBroadcastOp::Sub)
YT_IMPL_BINARY_OP(*, "*", yt::type::YBroadcastOp::Mul)
YT_IMPL_BINARY_OP(/, "/", yt::type::YBroadcastOp::Div)
YT_IMPL_BINARY_OP(%, "%", yt::type::YBroadcastOp::Mod)
YT_IMPL_BINARY_OP(&, "&", yt::type::YBroadcastOp::BitAnd)
YT_IMPL_BINARY_OP(|, "|", yt::type::YBroadcastOp::BitOr)
YT_IMPL_BINARY_OP(^, "^", yt::type::YBroadcastOp::BitXor)
YT_IMPL_BINARY_OP(<<, "<<", yt::type::YBroadcastOp::LShift)
YT_IMPL_BINARY_OP(>>, ">>", yt::type::YBroadcastOp::RShift)

#undef YT_IMPL_BINARY_OP

// YT_IMPL_CMP_OP宏：
// 比较运算符 facade 宏：输出 dtype 固定为 "bool"。
#define YT_IMPL_CMP_OP(OP, OP_NAME, OP_ID)                                                   \
    YT_IMPL_INLINE YTensorBase YTensorBase::operator OP(const YTensorBase& other) const {    \
        switch (layoutType()) {                                                              \
            case YLayoutType::Strided:                                                       \
                return yt::strided::broadcast(OP_ID, {this, &other}, "bool");                \
            default:                                                                         \
                throw std::runtime_error(std::string(OP_NAME) + ": layout not implemented"); \
        }                                                                                    \
    }                                                                                        \
    template <typename T>                                                                    \
    YTensorBase YTensorBase::operator OP(const T& scalar) const {                            \
        switch (layoutType()) {                                                              \
            case YLayoutType::Strided:                                                       \
                {                                                                            \
                    auto scalarInput = yt::strided::scalarTensor(*this, scalar);              \
                    return yt::strided::broadcast(OP_ID, {this, &scalarInput}, "bool");      \
                }                                                                            \
            default:                                                                         \
                throw std::runtime_error(std::string(OP_NAME) + ": layout not implemented"); \
        }                                                                                    \
    }

YT_IMPL_CMP_OP(<, "<", yt::type::YBroadcastOp::Less)
YT_IMPL_CMP_OP(<=, "<=", yt::type::YBroadcastOp::LessEqual)
YT_IMPL_CMP_OP(>, ">", yt::type::YBroadcastOp::Greater)
YT_IMPL_CMP_OP(>=, ">=", yt::type::YBroadcastOp::GreaterEqual)
YT_IMPL_CMP_OP(==, "==", yt::type::YBroadcastOp::Equal)
YT_IMPL_CMP_OP(!=, "!=", yt::type::YBroadcastOp::NotEqual)

#undef YT_IMPL_CMP_OP

// facade: 按 layout 路由到 strided::sum（单轴）。
YT_IMPL_INLINE YTensorBase YTensorBase::sum(int axis) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::sum(*this, axis);
        default:
            throw std::runtime_error("YTensorBase::sum: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::sum（多轴）。
YT_IMPL_INLINE YTensorBase YTensorBase::sum(const std::vector<int>& axes) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::sum(*this, axes);
        default:
            throw std::runtime_error("YTensorBase::sum: layout not implemented");
    }
}

// facade: 按layout路由到strided::mean（单轴keep-dim）。
YT_IMPL_INLINE YTensorBase YTensorBase::mean(int axis) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mean(*this, axis);
        default:
            throw std::runtime_error("YTensorBase::mean: layout not implemented");
    }
}

// facade: 按layout路由到strided::mean（多轴按owner定义顺序归约）。
YT_IMPL_INLINE YTensorBase YTensorBase::mean(const std::vector<int>& axes) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mean(*this, axes);
        default:
            throw std::runtime_error("YTensorBase::mean: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::max（单轴，返回值+索引）。
YT_IMPL_INLINE std::pair<YTensorBase, YTensorBase> YTensorBase::max(int axis) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::max(*this, axis);
        default:
            throw std::runtime_error("YTensorBase::max: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::max（多轴，返回值+索引）。
YT_IMPL_INLINE std::pair<YTensorBase, YTensorBase> YTensorBase::max(const std::vector<int>& axes) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::max(*this, axes);
        default:
            throw std::runtime_error("YTensorBase::max: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::matView。
YT_IMPL_INLINE YTensorBase YTensorBase::matView() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matView(*this);
        default:
            throw std::runtime_error("YTensorBase::matView: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::matmul。
YT_IMPL_INLINE YTensorBase
YTensorBase::matmul(const YTensorBase& other, yt::info::MatmulBackend backend) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::matmul(*this, other, backend);
        default:
            throw std::runtime_error("YTensorBase::matmul: layout not implemented");
    }
}

// facade: 按 layout 路由到 strided::masked_matmul。
YT_IMPL_INLINE YTensorBase YTensorBase::masked_matmul(
    const YTensorBase& other, const YTensorBase& mask, double maskedValue, yt::info::MatmulBackend backend
) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::masked_matmul(*this, other, mask, maskedValue, backend);
        default:
            throw std::runtime_error("YTensorBase::masked_matmul: layout not implemented");
    }
}

YT_IMPL_INLINE void YTensorBase::throwOperatorNotSupport(
    const std::string& typeName, const std::string& opName
) {
    throw std::runtime_error("[YTensorBase] Operator " + opName + " not support for type " + typeName);
}

YT_IMPL_INLINE void YTensorBase::throwShapeNotMatch(
    const std::string& opName, const std::vector<int>& otherShape
) const {
    std::string thisShapeStr = "[";
    auto thisShape = shape();
    for (size_t i = 0; i < thisShape.size(); ++i) {
        thisShapeStr += std::to_string(thisShape[i]);
        if (i + 1 < thisShape.size()) thisShapeStr += ", ";
    }
    thisShapeStr += "]";

    std::string otherShapeStr = "[";
    for (size_t i = 0; i < otherShape.size(); ++i) {
        otherShapeStr += std::to_string(otherShape[i]);
        if (i + 1 < otherShape.size()) otherShapeStr += ", ";
    }
    otherShapeStr += "]";

    throw std::runtime_error(
        "[YTensorBase::" + opName + "] Shape mismatch: " + thisShapeStr + " vs " + otherShapeStr
    );
}

}  // namespace yt
