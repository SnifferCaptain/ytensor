// ========== mseLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::mseLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::mseLoss()");
    return yt::kernel::broadcast([](const T& a, const T& b) {
        T diff = a - b;
        return diff * diff;
    }, input, target);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::mseLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::mseLoss_()");
    input.broadcastInplace([](T& a, const T& b) {
        T diff = a - b;
        a = diff * diff;
    }, target);
    return input;
}

// ========== crossEntropyLoss ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::crossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int axis) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::crossEntropyLoss()");
    axis = yt::function::_normalizeAxis<dim>(axis);

    auto shape = input.shape();
    yt::YTensor<T, dim> output(shape);

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) total_iterations *= s;

    #pragma omp parallel for if(total_iterations > 1024)
    for (int64_t idx = 0; idx < total_iterations; ++idx) {
        std::vector<int> iter_indices(iter_shape.size());
        int64_t temp_idx = idx;
        for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
            iter_indices[i] = temp_idx % iter_shape[i];
            temp_idx /= iter_shape[i];
        }

        std::vector<int> full_indices;
        int iter_pos = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == axis) {
                full_indices.push_back(0);
            } else {
                full_indices.push_back(iter_indices[iter_pos++]);
            }
        }

        // 计算logSoftmax
        T max_val = std::numeric_limits<T>::lowest();
        for (int d = 0; d < shape[axis]; ++d) {
            full_indices[axis] = d;
            max_val = std::max(max_val, input.at(full_indices));
        }

        T sum_exp = static_cast<T>(0);
        for (int d = 0; d < shape[axis]; ++d) {
            full_indices[axis] = d;
            sum_exp += std::exp(input.at(full_indices) - max_val);
        }
        T lse = std::log(sum_exp) + max_val;

        for (int d = 0; d < shape[axis]; ++d) {
            full_indices[axis] = d;
            T log_sm = input.at(full_indices) - lse;
            output.at(full_indices) = -target.at(full_indices) * log_sm;
        }
    }

    return output;
}

// ========== binaryCrossEntropyLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::binaryCrossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::binaryCrossEntropyLoss()");
    constexpr T eps = static_cast<T>(1e-7);
    return yt::kernel::broadcast([eps](const T& a, const T& b) {
        T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
        return -(b * std::log(clamped) + (static_cast<T>(1) - b) * std::log(static_cast<T>(1) - clamped));
    }, input, target);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::binaryCrossEntropyLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::binaryCrossEntropyLoss_()");
    constexpr T eps = static_cast<T>(1e-7);
    input.broadcastInplace([eps](T& a, const T& b) {
        T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
        a = -(b * std::log(clamped) + (static_cast<T>(1) - b) * std::log(static_cast<T>(1) - clamped));
    }, target);
    return input;
}

// ========== huberLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::huberLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::huberLoss()");
    return yt::kernel::broadcast([delta](const T& a, const T& b) {
        T diff = a - b;
        T abs_diff = std::abs(diff);
        if (abs_diff <= delta) {
            return static_cast<T>(0.5) * diff * diff;
        }
        return delta * (abs_diff - static_cast<T>(0.5) * delta);
    }, input, target);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::huberLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::huberLoss_()");
    input.broadcastInplace([delta](T& a, const T& b) {
        T diff = a - b;
        T abs_diff = std::abs(diff);
        if (abs_diff <= delta) {
            a = static_cast<T>(0.5) * diff * diff;
        } else {
            a = delta * (abs_diff - static_cast<T>(0.5) * delta);
        }
    }, target);
    return input;
}
