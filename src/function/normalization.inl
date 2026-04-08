namespace yt::function {

template<typename T, int dim>
yt::YTensor<T, dim>& _standardizeAxes_(yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps) {
    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);

    if constexpr (dim == 1) {
        T mean_val = x.mean();
        T var_val = static_cast<T>(0);
        for (int i = 0; i < x.shape(0); ++i) {
            T diff = x.at(i) - mean_val;
            var_val += diff * diff;
        }
        var_val /= static_cast<T>(x.shape(0));
        T inv_std = static_cast<T>(1) / std::sqrt(var_val + eps);
        x.broadcastInplace([mean_val, inv_std](T& a) {
            a = (a - mean_val) * inv_std;
        });
    } else {
        auto mean_vals = x.mean(normalized_axes);
        auto centered = x - mean_vals;
        auto var_vals = (centered * centered).mean(normalized_axes);
        x = centered;
        x.broadcastInplace([eps](T& a, const T& b) {
            a /= std::sqrt(b + eps);
        }, var_vals);
    }

    return x;
}

template<typename T, int dim>
yt::YTensor<T, dim>& _rmsAxes_(yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps) {
    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);

    if constexpr (dim == 1) {
        T mean_sq = static_cast<T>(0);
        for (int i = 0; i < x.shape(0); ++i) {
            mean_sq += x.at(i) * x.at(i);
        }
        mean_sq /= static_cast<T>(x.shape(0));
        T inv_rms = static_cast<T>(1) / std::sqrt(mean_sq + eps);
        x.broadcastInplace([inv_rms](T& a) {
            a *= inv_rms;
        });
    } else {
        auto mean_sq = (x * x).mean(normalized_axes);
        x.broadcastInplace([eps](T& a, const T& b) {
            a /= std::sqrt(b + eps);
        }, mean_sq);
    }

    return x;
}

template<typename T, int dim>
yt::YTensor<T, dim>& _lpAxes_(yt::YTensor<T, dim>& x, const std::vector<int>& axes, T p, T eps) {
    if (p <= static_cast<T>(0)) {
        throw std::invalid_argument("yt::function::normalize: p must be positive");
    }

    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);
    T inv_p = static_cast<T>(1) / p;

    if constexpr (dim == 1) {
        T sum_abs_p = static_cast<T>(0);
        for (int i = 0; i < x.shape(0); ++i) {
            sum_abs_p += std::pow(yt::function::_absValue(x.at(i)), p);
        }
        T inv_norm = static_cast<T>(1) / std::pow(sum_abs_p + eps, inv_p);
        x.broadcastInplace([inv_norm](T& a) {
            a *= inv_norm;
        });
    } else {
        auto norm_base = x.clone();
        norm_base.broadcastInplace([p](T& a) {
            a = std::pow(yt::function::_absValue(a), p);
        });
        auto norm_vals = norm_base.sum(normalized_axes);
        x.broadcastInplace([eps, inv_p](T& a, const T& b) {
            a /= std::pow(b + eps, inv_p);
        }, norm_vals);
    }

    return x;
}

template<typename T, int dim, int gammaDim>
void _applyScale_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const std::vector<int>& axes,
    const char* funcName
) {
    auto gamma_view = yt::function::_makeAffineView<T, dim>(
        gamma, axes, x, funcName, "gamma"
    );
    x.broadcastInplace([](T& a, const T& b) {
        a *= b;
    }, gamma_view);
}

template<typename T, int dim, int biasDim>
void _applyBias_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, biasDim>& bias,
    const std::vector<int>& axes,
    const char* funcName
) {
    auto bias_view = yt::function::_makeAffineView<T, dim>(
        bias, axes, x, funcName, "bias"
    );
    x.broadcastInplace([](T& a, const T& b) {
        a += b;
    }, bias_view);
}

template<int dim>
int _normalizeChannelAxis(int channelAxis, const char* funcName) {
    int normalized_axis = yt::function::_normalizeAxis<dim>(channelAxis);
    if (normalized_axis == 0) {
        throw std::invalid_argument(std::string(funcName) + ": channelAxis must not be batch axis 0");
    }
    return normalized_axis;
}

template<int dim>
std::vector<int> _channelFirstOrder(int channelAxis) {
    std::vector<int> order;
    order.reserve(dim);
    order.push_back(0);
    order.push_back(channelAxis);
    for (int axis = 1; axis < dim; ++axis) {
        if (axis != channelAxis) {
            order.push_back(axis);
        }
    }
    return order;
}

template<int dim>
std::vector<int> _instanceAxes(int channelAxis) {
    auto axes = yt::function::_axesExcept<dim>({0, channelAxis});
    if (axes.empty()) {
        throw std::invalid_argument("yt::function::instanceNorm: requires at least one spatial axis");
    }
    return axes;
}

template<int dim>
std::vector<int> _batchAxes(int channelAxis) {
    return yt::function::_axesExcept<dim>({channelAxis});
}

}  // namespace yt::function

// ========== layerNorm ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::layerNorm(const yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps) {
    auto output = x.clone();
    yt::function::layerNorm_(output, axes, eps);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::layerNorm(const yt::YTensor<T, dim>& x, int axis, T eps) {
    return yt::function::layerNorm(x, std::vector<int>{axis}, eps);
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::layerNorm_(yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::layerNorm_()");
    return yt::function::_standardizeAxes_(x, axes, eps);
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::layerNorm_(yt::YTensor<T, dim>& x, int axis, T eps) {
    return yt::function::layerNorm_(x, std::vector<int>{axis}, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::layerNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    const std::vector<int>& axes,
    T eps
) {
    auto output = x.clone();
    yt::function::layerNorm_(output, gamma, bias, axes, eps);
    return output;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::layerNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int axis,
    T eps
) {
    return yt::function::layerNorm(x, gamma, bias, std::vector<int>{axis}, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::layerNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    const std::vector<int>& axes,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::layerNorm_()");
    yt::function::layerNorm_(x, axes, eps);
    yt::function::_applyScale_(x, gamma, axes, "yt::function::layerNorm_");
    yt::function::_applyBias_(x, bias, axes, "yt::function::layerNorm_");
    return x;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::layerNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int axis,
    T eps
) {
    return yt::function::layerNorm_(x, gamma, bias, std::vector<int>{axis}, eps);
}

// ========== rmsNorm ==========

template<typename T, int dim, int weightDim>
yt::YTensor<T, dim> yt::function::rmsNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, weightDim>& weight,
    const std::vector<int>& axes,
    T eps
) {
    auto output = x.clone();
    yt::function::rmsNorm_(output, weight, axes, eps);
    return output;
}

template<typename T, int dim, int weightDim>
yt::YTensor<T, dim> yt::function::rmsNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, weightDim>& weight,
    int axis,
    T eps
) {
    return yt::function::rmsNorm(x, weight, std::vector<int>{axis}, eps);
}

template<typename T, int dim, int weightDim>
yt::YTensor<T, dim>& yt::function::rmsNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, weightDim>& weight,
    const std::vector<int>& axes,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::rmsNorm_()");
    yt::function::_rmsAxes_(x, axes, eps);
    yt::function::_applyScale_(x, weight, axes, "yt::function::rmsNorm_");
    return x;
}

template<typename T, int dim, int weightDim>
yt::YTensor<T, dim>& yt::function::rmsNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, weightDim>& weight,
    int axis,
    T eps
) {
    return yt::function::rmsNorm_(x, weight, std::vector<int>{axis}, eps);
}

// ========== normalize (Lp) ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::normalize(
    const yt::YTensor<T, dim>& x,
    const std::vector<int>& axes,
    T p,
    T eps
) {
    auto output = x.clone();
    yt::function::normalize_(output, axes, p, eps);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::normalize(
    const yt::YTensor<T, dim>& x,
    int axis,
    T p,
    T eps
) {
    return yt::function::normalize(x, std::vector<int>{axis}, p, eps);
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::normalize_(
    yt::YTensor<T, dim>& x,
    const std::vector<int>& axes,
    T p,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::normalize_()");
    return yt::function::_lpAxes_(x, axes, p, eps);
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::normalize_(
    yt::YTensor<T, dim>& x,
    int axis,
    T p,
    T eps
) {
    return yt::function::normalize_(x, std::vector<int>{axis}, p, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::normalize(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    const std::vector<int>& axes,
    T p,
    T eps
) {
    auto output = x.clone();
    yt::function::normalize_(output, gamma, bias, axes, p, eps);
    return output;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::normalize(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int axis,
    T p,
    T eps
) {
    return yt::function::normalize(x, gamma, bias, std::vector<int>{axis}, p, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::normalize_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    const std::vector<int>& axes,
    T p,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::normalize_()");
    yt::function::normalize_(x, axes, p, eps);
    yt::function::_applyScale_(x, gamma, axes, "yt::function::normalize_");
    yt::function::_applyBias_(x, bias, axes, "yt::function::normalize_");
    return x;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::normalize_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int axis,
    T p,
    T eps
) {
    return yt::function::normalize_(x, gamma, bias, std::vector<int>{axis}, p, eps);
}

// ========== instanceNorm ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::instanceNorm(
    const yt::YTensor<T, dim>& x,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::instanceNorm_(output, channelAxis, eps);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::instanceNorm_(
    yt::YTensor<T, dim>& x,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::instanceNorm_()");
    static_assert(dim >= 2, "instanceNorm requires dim >= 2");
    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::instanceNorm_");
    auto axes = yt::function::_instanceAxes<dim>(normalized_channel);
    return yt::function::_standardizeAxes_(x, axes, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::instanceNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::instanceNorm_(output, gamma, bias, channelAxis, eps);
    return output;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::instanceNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::instanceNorm_()");
    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::instanceNorm_");
    yt::function::instanceNorm_(x, normalized_channel, eps);
    yt::function::_applyScale_(x, gamma, std::vector<int>{normalized_channel}, "yt::function::instanceNorm_");
    yt::function::_applyBias_(x, bias, std::vector<int>{normalized_channel}, "yt::function::instanceNorm_");
    return x;
}

// ========== batchNorm ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::batchNorm(
    const yt::YTensor<T, dim>& x,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::batchNorm_(output, channelAxis, eps);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::batchNorm_(
    yt::YTensor<T, dim>& x,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::batchNorm_()");
    static_assert(dim >= 2, "batchNorm requires dim >= 2");
    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::batchNorm_");
    auto axes = yt::function::_batchAxes<dim>(normalized_channel);
    return yt::function::_standardizeAxes_(x, axes, eps);
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::batchNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::batchNorm_(output, gamma, bias, channelAxis, eps);
    return output;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::batchNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::batchNorm_()");
    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::batchNorm_");
    yt::function::batchNorm_(x, normalized_channel, eps);
    yt::function::_applyScale_(x, gamma, std::vector<int>{normalized_channel}, "yt::function::batchNorm_");
    yt::function::_applyBias_(x, bias, std::vector<int>{normalized_channel}, "yt::function::batchNorm_");
    return x;
}

// ========== groupNorm ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::groupNorm(
    const yt::YTensor<T, dim>& x,
    int groups,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::groupNorm_(output, groups, channelAxis, eps);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::groupNorm_(
    yt::YTensor<T, dim>& x,
    int groups,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::groupNorm_()");
    static_assert(dim >= 2, "groupNorm requires dim >= 2");

    if (groups <= 0) {
        throw std::invalid_argument("yt::function::groupNorm: groups must be positive");
    }

    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::groupNorm_");
    int channels = x.shape(normalized_channel);
    if (channels % groups != 0) {
        throw std::invalid_argument("yt::function::groupNorm: channel count must be divisible by groups");
    }

    auto order = yt::function::_channelFirstOrder<dim>(normalized_channel);
    auto inverse_order = yt::function::_inversePermutation<dim>(order);
    auto permuted = normalized_channel == 1 ? x.contiguous() : x.permute(order).contiguous();
    auto permuted_shape = permuted.shape();

    std::vector<int> grouped_shape;
    grouped_shape.reserve(dim + 1);
    grouped_shape.push_back(permuted_shape[0]);
    grouped_shape.push_back(groups);
    grouped_shape.push_back(channels / groups);
    for (int axis = 2; axis < dim; ++axis) {
        grouped_shape.push_back(permuted_shape[axis]);
    }

    auto grouped = permuted.template view<dim + 1>(grouped_shape);
    std::vector<int> reduce_axes;
    for (int axis = 2; axis < dim + 1; ++axis) {
        reduce_axes.push_back(axis);
    }
    yt::function::_standardizeAxes_(grouped, reduce_axes, eps);

    auto restored = grouped.template view<dim>(permuted_shape);
    x = normalized_channel == 1 ? restored : restored.permute(inverse_order);
    return x;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim> yt::function::groupNorm(
    const yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int groups,
    int channelAxis,
    T eps
) {
    auto output = x.clone();
    yt::function::groupNorm_(output, gamma, bias, groups, channelAxis, eps);
    return output;
}

template<typename T, int dim, int gammaDim, int biasDim>
yt::YTensor<T, dim>& yt::function::groupNorm_(
    yt::YTensor<T, dim>& x,
    const yt::YTensor<T, gammaDim>& gamma,
    const yt::YTensor<T, biasDim>& bias,
    int groups,
    int channelAxis,
    T eps
) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::groupNorm_()");
    int normalized_channel = yt::function::_normalizeChannelAxis<dim>(channelAxis, "yt::function::groupNorm_");
    yt::function::groupNorm_(x, groups, normalized_channel, eps);
    yt::function::_applyScale_(x, gamma, std::vector<int>{normalized_channel}, "yt::function::groupNorm_");
    yt::function::_applyBias_(x, bias, std::vector<int>{normalized_channel}, "yt::function::groupNorm_");
    return x;
}
