// ========== relu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::relu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    yt::YTensor<T, dim> op;
    if (order == 0) {
        op = yt::kernel::broadcast([](const T& a) {
            return std::max(a, static_cast<T>(0));
        }, x);
    } else if (order == 1) {
        op = yt::kernel::broadcast([](const T& a) {
            return static_cast<T>(a > 0);
        }, x);
    } else if (order > 1) {
        op = yt::YTensor<T, dim>::zeros(x.shape());
    } else {
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++) {
            fact *= i;
        }
        op = yt::kernel::broadcast([&pow, &fact](const T& a) {
            if (a > 0) {
                return std::pow(a, pow) / static_cast<T>(fact);
            }
            return static_cast<T>(0);
        }, x);
    }
    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::relu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            a = std::max(a, static_cast<T>(0));
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            a = static_cast<T>(a > 0);
        });
    } else if (order > 1) {
        x.broadcastInplace([](T& a) {
            a = static_cast<T>(0);
        });
    } else {
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++) {
            fact *= i;
        }
        x.broadcastInplace([&pow, &fact](T& a) {
            if (a > 0) {
                a = std::pow(a, pow) / static_cast<T>(fact);
            } else {
                a = static_cast<T>(0);
            }
        });
    }
    return x;
}

// ========== exp / exp_ ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::exp(const yt::YTensor<T, dim>& x, int) {
    return x.unaryOpTransform(0, [](const T& a, const T&) {
        return std::exp(a);
    });
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::exp_(yt::YTensor<T, dim>& x, int order) {
    (void)order;
    // exp 的任意阶导数/不计常数的原函数仍然是 exp
    x.broadcastInplace([](T& a) {
        using std::exp;
        a = exp(a);
    });
    return x;
}

// ========== sigmoid / sigmoid_ ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::sigmoid(const yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            return yt::function::_stableSigmoid(a);
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::_stableSigmoid(a);
            return sig * (yt::function::_one<T>() - sig);
        }, x);
    }
    if (order == -1) {
        return yt::function::softplus(x, 0);
    }
    throwNotSupport("yt::function::sigmoid", "order != 0, 1, -1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::sigmoid_(yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            a = yt::function::_stableSigmoid(a);
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            T sig = yt::function::_stableSigmoid(a);
            a = sig * (yt::function::_one<T>() - sig);
        });
    } else if (order == -1) {
        return yt::function::softplus_(x, 0);
    } else {
        throwNotSupport("yt::function::sigmoid_", "order != 0, 1, -1");
    }
    return x;
}

// ========== softmax ==========

namespace yt::function {

template<typename T, int dim>
yt::YTensor<T, dim> _softmaxSingleAxis(const yt::YTensor<T, dim>& x, int axis) {
    axis = yt::function::_normalizeAxis<dim>(axis);

    auto shape = x.shape();
    yt::YTensor<T, dim> output(shape);

    // 单轴softmax保留原来的快路径，避免多轴泛化后拖慢常见推理场景。
    if (x.isContiguous() && output.isContiguous()) {
        int64_t dim_size = shape[axis];

        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }

        int64_t inner_size = 1;
        for (int i = axis + 1; i < dim; ++i) {
            inner_size *= shape[i];
        }

        int64_t dim_stride = inner_size;
        int64_t outer_stride = dim_size * dim_stride;

        const T* input_data_base = x.data();
        T* output_data_base = output.data();

        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                const T* input_data = input_data_base + outer_idx * outer_stride + inner_idx;
                T* output_data = output_data_base + outer_idx * outer_stride + inner_idx;

                T max_val = input_data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, input_data[d * dim_stride]);
                }

                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(input_data[d * dim_stride] - max_val);
                    output_data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }

                for (int64_t d = 0; d < dim_size; ++d) {
                    output_data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) {
            total_iterations *= s;
        }

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

            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                output.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) /= sum_exp;
            }
        }
    }

    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& _softmaxSingleAxis_(yt::YTensor<T, dim>& x, int axis) {
    axis = yt::function::_normalizeAxis<dim>(axis);
    auto shape = x.shape();

    if (x.isContiguous()) {
        int64_t dim_size = shape[axis];

        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }

        int64_t inner_size = 1;
        for (int i = axis + 1; i < dim; ++i) {
            inner_size *= shape[i];
        }

        int64_t dim_stride = inner_size;
        int64_t outer_stride = dim_size * dim_stride;

        T* data_base = x.data();

        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                T* data = data_base + outer_idx * outer_stride + inner_idx;

                T max_val = data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, data[d * dim_stride]);
                }

                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(data[d * dim_stride] - max_val);
                    data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }

                for (int64_t d = 0; d < dim_size; ++d) {
                    data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) {
            total_iterations *= s;
        }

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

            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                x.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) /= sum_exp;
            }
        }
    }

    return x;
}

}  // namespace yt::function

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::softmax(const yt::YTensor<T, dim>& x, const std::vector<int>& axes) {
    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);
    if (normalized_axes.size() == 1) {
        return yt::function::_softmaxSingleAxis(x, normalized_axes[0]);
    }

    if constexpr (dim == 1) {
        T max_val = x.max(0).first;
        auto output = x.clone();
        output.broadcastInplace([max_val](T& a) {
            a = std::exp(a - max_val);
        });
        T sum_exp = output.sum();
        output.broadcastInplace([sum_exp](T& a) {
            a /= sum_exp;
        });
        return output;
    } else {
        auto max_vals = x.max(normalized_axes).first;
        auto output = x.clone();
        output.broadcastInplace([](T& a, const T& b) {
            a = std::exp(a - b);
        }, max_vals);
        auto sum_exp = output.sum(normalized_axes);
        output.broadcastInplace([](T& a, const T& b) {
            a /= b;
        }, sum_exp);
        return output;
    }
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::softmax(const yt::YTensor<T, dim>& x, int axis) {
    return yt::function::softmax(x, std::vector<int>{axis});
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::softmax_(yt::YTensor<T, dim>& x, const std::vector<int>& axes) {
    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);
    if (normalized_axes.size() == 1) {
        return yt::function::_softmaxSingleAxis_(x, normalized_axes[0]);
    }

    if constexpr (dim == 1) {
        T max_val = x.max(0).first;
        x.broadcastInplace([max_val](T& a) {
            a = std::exp(a - max_val);
        });
        T sum_exp = x.sum();
        x.broadcastInplace([sum_exp](T& a) {
            a /= sum_exp;
        });
        return x;
    } else {
        auto max_vals = x.max(normalized_axes).first;
        x.broadcastInplace([](T& a, const T& b) {
            a = std::exp(a - b);
        }, max_vals);
        auto sum_exp = x.sum(normalized_axes);
        x.broadcastInplace([](T& a, const T& b) {
            a /= b;
        }, sum_exp);
        return x;
    }
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::softmax_(yt::YTensor<T, dim>& x, int axis) {
    return yt::function::softmax_(x, std::vector<int>{axis});
}

// ========== leakyRelu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::leakyRelu(const yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::leakyRelu()");
    if (order == 0) {
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? a : alpha * a;
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? static_cast<T>(1) : alpha;
        }, x);
    }
    throwNotSupport("yt::function::leakyRelu", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::leakyRelu_(yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::leakyRelu_()");
    if (order == 0) {
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? a : alpha * a;
        });
    } else if (order == 1) {
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? static_cast<T>(1) : alpha;
        });
    } else {
        throwNotSupport("yt::function::leakyRelu_", "order != 0, 1");
    }
    return x;
}

// ========== elu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::elu(const yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::elu()");
    if (order == 0) {
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? a : alpha * (std::exp(a) - static_cast<T>(1));
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? static_cast<T>(1) : alpha * std::exp(a);
        }, x);
    }
    throwNotSupport("yt::function::elu", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::elu_(yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::elu_()");
    if (order == 0) {
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? a : alpha * (std::exp(a) - static_cast<T>(1));
        });
    } else if (order == 1) {
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? static_cast<T>(1) : alpha * std::exp(a);
        });
    } else {
        throwNotSupport("yt::function::elu_", "order != 0, 1");
    }
    return x;
}

// ========== selu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::selu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::selu()");
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            return a > static_cast<T>(0) ? l * a : l * al * (std::exp(a) - static_cast<T>(1));
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            return a > static_cast<T>(0) ? l : l * al * std::exp(a);
        }, x);
    }
    throwNotSupport("yt::function::selu", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::selu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::selu_()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            a = a > static_cast<T>(0) ? l * a : l * al * (std::exp(a) - static_cast<T>(1));
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            a = a > static_cast<T>(0) ? l : l * al * std::exp(a);
        });
    } else {
        throwNotSupport("yt::function::selu_", "order != 0, 1");
    }
    return x;
}

// ========== gelu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::gelu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::gelu()");
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            return static_cast<T>(0.5) * a * (static_cast<T>(1) + std::erf(a / sqrt2));
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            constexpr T inv_sqrt2pi = static_cast<T>(0.3989422804014326779399460599343);
            T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(a / sqrt2));
            T pdf = inv_sqrt2pi * std::exp(static_cast<T>(-0.5) * a * a);
            return cdf + a * pdf;
        }, x);
    }
    throwNotSupport("yt::function::gelu", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::gelu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::gelu_()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            a = static_cast<T>(0.5) * a * (static_cast<T>(1) + std::erf(a / sqrt2));
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            constexpr T inv_sqrt2pi = static_cast<T>(0.3989422804014326779399460599343);
            T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(a / sqrt2));
            T pdf = inv_sqrt2pi * std::exp(static_cast<T>(-0.5) * a * a);
            a = cdf + a * pdf;
        });
    } else {
        throwNotSupport("yt::function::gelu_", "order != 0, 1");
    }
    return x;
}

// ========== tanh ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::tanh(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::tanh()");
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            return std::tanh(a);
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            T t = std::tanh(a);
            return static_cast<T>(1) - t * t;
        }, x);
    }
    throwNotSupport("yt::function::tanh", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::tanh_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::tanh_()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            a = std::tanh(a);
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            T t = std::tanh(a);
            a = static_cast<T>(1) - t * t;
        });
    } else {
        throwNotSupport("yt::function::tanh_", "order != 0, 1");
    }
    return x;
}

// ========== swish (SiLU) ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::swish(const yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::_stableSigmoid(a);
            return a * sig;
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::_stableSigmoid(a);
            return sig * (yt::function::_one<T>() + a * (yt::function::_one<T>() - sig));
        }, x);
    }
    throwNotSupport("yt::function::swish", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::swish_(yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            T sig = yt::function::_stableSigmoid(a);
            a = a * sig;
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            T sig = yt::function::_stableSigmoid(a);
            a = sig * (yt::function::_one<T>() + a * (yt::function::_one<T>() - sig));
        });
    } else {
        throwNotSupport("yt::function::swish_", "order != 0, 1");
    }
    return x;
}

// ========== softplus ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::softplus(const yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            return yt::function::_stableSoftplus(a);
        }, x);
    }
    if (order == 1) {
        return yt::function::sigmoid(x, 0);
    }
    throwNotSupport("yt::function::softplus", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::softplus_(yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            a = yt::function::_stableSoftplus(a);
        });
    } else if (order == 1) {
        return yt::function::sigmoid_(x, 0);
    } else {
        throwNotSupport("yt::function::softplus_", "order != 0, 1");
    }
    return x;
}

// ========== mish ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::mish(const yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            T sp = yt::function::_stableSoftplus(a);
            return a * yt::function::_tanhValue(sp);
        }, x);
    }
    if (order == 1) {
        // mish'(x) = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
        return yt::kernel::broadcast([](const T& a) {
            T sp = yt::function::_stableSoftplus(a);
            T tsp = yt::function::_tanhValue(sp);
            T sig = yt::function::_stableSigmoid(a);
            return tsp + a * (yt::function::_one<T>() - tsp * tsp) * sig;
        }, x);
    }
    throwNotSupport("yt::function::mish", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::mish_(yt::YTensor<T, dim>& x, int order) {
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            T sp = yt::function::_stableSoftplus(a);
            a = a * yt::function::_tanhValue(sp);
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            T sp = yt::function::_stableSoftplus(a);
            T tsp = yt::function::_tanhValue(sp);
            T sig = yt::function::_stableSigmoid(a);
            a = tsp + a * (yt::function::_one<T>() - tsp * tsp) * sig;
        });
    } else {
        throwNotSupport("yt::function::mish_", "order != 0, 1");
    }
    return x;
}

// ========== hardSigmoid ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::hardSigmoid(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSigmoid()");
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            T v = a / static_cast<T>(6) + static_cast<T>(0.5);
            return std::max(static_cast<T>(0), std::min(static_cast<T>(1), v));
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            return (a > static_cast<T>(-3) && a < static_cast<T>(3))
                ? static_cast<T>(1) / static_cast<T>(6)
                : static_cast<T>(0);
        }, x);
    }
    throwNotSupport("yt::function::hardSigmoid", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::hardSigmoid_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSigmoid_()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            T v = a / static_cast<T>(6) + static_cast<T>(0.5);
            a = std::max(static_cast<T>(0), std::min(static_cast<T>(1), v));
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            a = (a > static_cast<T>(-3) && a < static_cast<T>(3))
                ? static_cast<T>(1) / static_cast<T>(6)
                : static_cast<T>(0);
        });
    } else {
        throwNotSupport("yt::function::hardSigmoid_", "order != 0, 1");
    }
    return x;
}

// ========== hardSwish ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::hardSwish(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSwish()");
    if (order == 0) {
        return yt::kernel::broadcast([](const T& a) {
            if (a <= static_cast<T>(-3)) return static_cast<T>(0);
            if (a >= static_cast<T>(3)) return a;
            return a * (a + static_cast<T>(3)) / static_cast<T>(6);
        }, x);
    }
    if (order == 1) {
        return yt::kernel::broadcast([](const T& a) {
            if (a <= static_cast<T>(-3)) return static_cast<T>(0);
            if (a >= static_cast<T>(3)) return static_cast<T>(1);
            return (static_cast<T>(2) * a + static_cast<T>(3)) / static_cast<T>(6);
        }, x);
    }
    throwNotSupport("yt::function::hardSwish", "order != 0, 1");
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::hardSwish_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSwish_()");
    if (order == 0) {
        x.broadcastInplace([](T& a) {
            if (a <= static_cast<T>(-3)) {
                a = static_cast<T>(0);
                return;
            }
            if (a >= static_cast<T>(3)) return;
            a = a * (a + static_cast<T>(3)) / static_cast<T>(6);
        });
    } else if (order == 1) {
        x.broadcastInplace([](T& a) {
            if (a <= static_cast<T>(-3)) {
                a = static_cast<T>(0);
                return;
            }
            if (a >= static_cast<T>(3)) {
                a = static_cast<T>(1);
                return;
            }
            a = (static_cast<T>(2) * a + static_cast<T>(3)) / static_cast<T>(6);
        });
    } else {
        throwNotSupport("yt::function::hardSwish_", "order != 0, 1");
    }
    return x;
}
