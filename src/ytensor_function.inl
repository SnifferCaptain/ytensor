/***************
 * file: ytensor_function.inl
 * purpose: 高层function共享数值helper、axis规则和子模块聚合入口。
 ***************/

#include "../include/ytensor_concepts.hpp"
#if YT_USE_AVX2
#include "../include/function/flash_attention.hpp"
#endif

namespace yt::function {

// ==================== stable scalar helpers ====================

template <typename T>
inline T _zero() {
    return static_cast<T>(0);
}

template <typename T>
inline T _one() {
    return static_cast<T>(1);
}

template <typename T>
inline T _expValue(const T& value) {
    using std::exp;
    return exp(value);
}

template <typename T>
inline T _log1pValue(const T& value) {
    using std::log1p;
    return log1p(value);
}

template <typename T>
inline T _tanhValue(const T& value) {
    using std::tanh;
    return tanh(value);
}

template <typename T>
inline T _absValue(const T& value) {
    using std::abs;
    return abs(value);
}

template <typename T>
inline T _stableSigmoid(const T& value) {
    // 按符号选择指数形式，避免大负数计算exp(-value)上溢。
    const T zero_v = _zero<T>();
    const T one_v = _one<T>();
    if (value >= zero_v) {
        T z = _expValue(-value);
        return one_v / (one_v + z);
    }
    T z = _expValue(value);
    return z / (one_v + z);
}

template <typename T>
inline T _stableSoftplus(const T& value) {
    // max(x,0)+log1p(exp(-abs(x)))在正负极值下都保持稳定。
    const T zero_v = _zero<T>();
    const T positive = value > zero_v ? value : zero_v;
    return positive + _log1pValue(_expValue(-_absValue(value)));
}

// ==================== axis and broadcast-shape helpers ====================

// 按项目循环索引规则把任意正负axis规范化到[0, dim)。
template <int dim>
inline int _normalizeAxis(int axis) {
    return (axis % dim + dim) % dim;
}

// 规范化axis集合并拒绝重复项，保留调用方给定顺序。
template <int dim>
inline std::vector<int> _normalizeAxes(const std::vector<int>& axes) {
    if (axes.empty()) {
        throw std::invalid_argument("yt::function: axes must not be empty");
    }
    std::vector<int> normalized;
    normalized.reserve(axes.size());
    std::vector<bool> used(dim, false);
    for (int axis : axes) {
        int normalized_axis = _normalizeAxis<dim>(axis);
        if (used[normalized_axis]) {
            throw std::invalid_argument("yt::function: duplicate axes are not allowed");
        }
        used[normalized_axis] = true;
        normalized.push_back(normalized_axis);
    }
    return normalized;
}

// 验证permutation并返回其inverse mapping。
template <int dim>
inline std::vector<int> _inversePermutation(const std::vector<int>& order) {
    if (static_cast<int>(order.size()) != dim) {
        throw std::invalid_argument("yt::function: permutation size must match tensor dim");
    }
    std::vector<int> inverse(dim, -1);
    for (int i = 0; i < dim; ++i) {
        int axis = _normalizeAxis<dim>(order[i]);
        if (inverse[axis] != -1) {
            throw std::invalid_argument("yt::function: invalid permutation with duplicate axis");
        }
        inverse[axis] = i;
    }
    return inverse;
}

template <int dim>
inline std::vector<int> _axesExcept(const std::vector<int>& excludedAxes) {
    std::vector<bool> used(dim, false);
    for (int axis : _normalizeAxes<dim>(excludedAxes)) {
        used[axis] = true;
    }

    std::vector<int> axes;
    axes.reserve(dim);
    for (int axis = 0; axis < dim; ++axis) {
        if (!used[axis]) {
            axes.push_back(axis);
        }
    }
    return axes;
}

template <int dim>
inline std::vector<int> _makeBroadcastShape(int axis, int size) {
    std::vector<int> shape(dim, 1);
    shape[axis] = size;
    return shape;
}

template <int dim>
inline std::vector<int> _makeBroadcastShape(const std::vector<int>& axes, const std::vector<int>& sizes) {
    if (axes.size() != sizes.size()) {
        throw std::invalid_argument("yt::function: axes size must match broadcast sizes");
    }
    std::vector<int> shape(dim, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
        shape[axes[i]] = sizes[i];
    }
    return shape;
}

// 将affine参数reshape为可沿指定axes广播的view，并验证reference对应extent。
template <typename T, int dim, int affineDim>
inline yt::YTensor<T, dim> _makeAffineView(
    const yt::YTensor<T, affineDim>& affine,
    const std::vector<int>& axes,
    const yt::YTensor<T, dim>& reference,
    const std::string& funcName,
    const std::string& affineName
) {
    auto normalized_axes = _normalizeAxes<dim>(axes);
    if (static_cast<int>(normalized_axes.size()) != affineDim) {
        throw std::invalid_argument(funcName + ": " + affineName + " dim must match axes size");
    }

    auto affine_shape = affine.shape();
    for (int i = 0; i < affineDim; ++i) {
        if (affine_shape[i] != reference.shape(normalized_axes[i])) {
            throw std::invalid_argument(funcName + ": " + affineName + " shape must match affine axes");
        }
    }

    return affine.template view<dim>(_makeBroadcastShape<dim>(normalized_axes, affine_shape));
}

}  // namespace yt::function

// 子模块共享以上internal helper；聚合顺序保证定义在模板实例化前可见。
#include "function/ops.inl"
#include "function/activation.inl"
#include "function/normalization.inl"
#include "function/loss.inl"

void yt::function::throwNotSupport(const std::string& funcName, const std::string& caseDiscription) {
    throw std::invalid_argument("Function " + funcName + " is not supported for case: " + caseDiscription);
}
