template <typename T, int dim0, int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> yt::function::matmul(const yt::YTensor<T, dim0>& a, const yt::YTensor<T, dim1>& b) {
    return a.matmul(b);
}

namespace yt::function {

template<typename T, int dim>
yt::YTensor<T, dim> _scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    const yt::YTensor<bool, 2>* mask,
    yt::YTensor<T, 2>* bias
);

template<typename T, int dim>
void _zeroFullyMaskedSdpaRows(
    yt::YTensor<T, dim>& output,
    const yt::YTensor<bool, 2>& mask
) {
    auto outputMatView = output.matView();
    outputMatView.broadcastInplace([&mask](yt::YTensor<T, 2>& mat) {
        for (int row = 0; row < mask.shape(0); ++row) {
            bool any_visible = false;
            for (int col = 0; col < mask.shape(1); ++col) {
                if (mask.at(row, col)) {
                    any_visible = true;
                    break;
                }
            }
            if (!any_visible) {
                for (int d = 0; d < mat.shape(1); ++d) {
                    mat.at(row, d) = static_cast<T>(0);
                }
            }
        }
    });
}

template<typename T, int dim, typename MaskFunc>
requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
void _zeroFullyMaskedSdpaRows(
    yt::YTensor<T, dim>& output,
    MaskFunc&& mask,
    int key_len
) {
    auto outputMatView = output.matView();
    auto maskFunc = std::forward<MaskFunc>(mask);
    outputMatView.broadcastInplace([&maskFunc, key_len](yt::YTensor<T, 2>& mat) {
        for (int row = 0; row < mat.shape(0); ++row) {
            bool any_visible = false;
            for (int col = 0; col < key_len; ++col) {
                if (maskFunc(row, col)) {
                    any_visible = true;
                    break;
                }
            }
            if (!any_visible) {
                for (int d = 0; d < mat.shape(1); ++d) {
                    mat.at(row, d) = static_cast<T>(0);
                }
            }
        }
    });
}

template<typename T, int dim, typename MaskFunc>
requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
yt::YTensor<T, dim> _scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    MaskFunc&& mask,
    yt::YTensor<T, 2>* bias
);

}  // namespace yt::function

// ========== linear ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::linear(const yt::YTensor<T, dim>& x, const yt::YTensor<T, 2>& weight) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::linear()");
    if (x.shape(-1) != weight.shape(1)) {
        throw std::invalid_argument("yt::function::linear: input feature size must match weight.shape(1)");
    }
    return x.matmul(weight.transpose());
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::linear(const yt::YTensor<T, dim>& x, const yt::YTensor<T, 2>& weight, const yt::YTensor<T, 1>& bias) {
    if (bias.shape(0) != weight.shape(0)) {
        throw std::invalid_argument("yt::function::linear: bias size must match weight.shape(0)");
    }
    auto output = yt::function::linear(x, weight);
    // bias 沿最后一维广播到输出张量
    auto bias_view = bias.template view<dim>(yt::function::_makeBroadcastShape<dim>(dim - 1, bias.shape(0)));
    output.broadcastInplace([](T& a, const T& b) {
        a += b;
    }, bias_view);
    return output;
}

// ========== scaledDotProductAttention ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::_scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    const yt::YTensor<bool, 2>* mask,
    yt::YTensor<T, 2>* bias
) {
#if YT_USE_AVX2
    if constexpr (!std::is_same_v<T, float>) {
        throwNotSupport("yt::function::scaledDotProductAttention", "FLASH_AVX2 backend only supports float");
        return yt::YTensor<T, dim>();
    } else {
        if (mask != nullptr && (mask->shape(0) != query.shape(-2) || mask->shape(1) != key.shape(-2))) {
            throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
        }
        if (bias != nullptr && (bias->shape(0) != query.shape(-2) || bias->shape(1) != key.shape(-2))) {
            throw std::invalid_argument("Bias shape must match the last two dimensions of the score tensor.");
        }

        auto queryMatView = query.matView();
        auto keyMatView = key.matView();
        auto valueMatView = value.matView();

        std::vector<int> outputShape;
        if constexpr (dim == 2) {
            outputShape = {query.shape(-2), value.shape(-1)};
        } else {
            outputShape = yt::kernel::computeBroadcastShape({queryMatView.shape(), keyMatView.shape(), valueMatView.shape()});
            outputShape.push_back(query.shape(-2));
            outputShape.push_back(value.shape(-1));
        }

        yt::YTensor<T, dim> output(outputShape);
        auto outputMatView = output.matView();

        yt::YTensor<bool, 2> maskContiguous;
        const bool* maskPtr = nullptr;
        int64_t maskStride = 0;
        if (mask != nullptr) {
            maskContiguous = mask->isContiguous() ? *mask : mask->contiguous();
            maskPtr = maskContiguous.data();
            maskStride = maskContiguous.stride_(0);
        }

        const float* biasPtr = nullptr;
        int64_t rsbias = 0;
        int64_t csbias = 0;
        if (bias != nullptr) {
            biasPtr = bias->data();
            rsbias = bias->stride_(0);
            csbias = bias->stride_(1);
        }

        outputMatView.broadcastInplace(
            [scale, maskPtr, maskStride, biasPtr, rsbias, csbias](yt::YTensor<T, 2>& o, const yt::YTensor<T, 2>& q, const yt::YTensor<T, 2>& k, const yt::YTensor<T, 2>& v) {
                const yt::YTensor<T, 2>* qKernel = &q;
                const yt::YTensor<T, 2>* kKernel = &k;
                const yt::YTensor<T, 2>* vKernel = &v;
                yt::YTensor<T, 2> qContiguous;
                yt::YTensor<T, 2> kContiguous;
                yt::YTensor<T, 2> vContiguous;

                // Decode(q_len==1) already has a dedicated fast path. For longer sequences,
                // materialize pathological strided views (e.g. transposed cache + zero-stride repeat)
                // so Flash Attention can hit its contiguous K/V accumulation path.
                if (q.shape(0) > 1) {
                    if (q.stride_(1) != 1) {
                        qContiguous = q.contiguous();
                        qKernel = &qContiguous;
                    }
                    if (k.stride_(1) != 1) {
                        kContiguous = k.contiguous();
                        kKernel = &kContiguous;
                    }
                    if (v.stride_(1) != 1) {
                        vContiguous = v.contiguous();
                        vKernel = &vContiguous;
                    }
                }

                auto qStride = qKernel->stride_();
                auto kStride = kKernel->stride_();
                auto vStride = vKernel->stride_();
                auto oStride = o.stride_();
                yt::kernel::avx2::flash_attention(
                    qKernel->data(),
                    kKernel->data(),
                    vKernel->data(),
                    o.data(),
                    qKernel->shape(0),
                    kKernel->shape(0),
                    qKernel->shape(1),
                    vKernel->shape(1),
                    scale,
                    static_cast<int64_t>(qStride[0]),
                    static_cast<int64_t>(qStride[1]),
                    static_cast<int64_t>(kStride[0]),
                    static_cast<int64_t>(kStride[1]),
                    static_cast<int64_t>(vStride[0]),
                    static_cast<int64_t>(vStride[1]),
                    static_cast<int64_t>(oStride[0]),
                    static_cast<int64_t>(oStride[1]),
                    maskPtr,
                    maskStride,
                    biasPtr,
                    rsbias,
                    csbias
                );
            },
            queryMatView, keyMatView, valueMatView
        );
        return output;
    }
#else
    (void)query;
    (void)key;
    (void)value;
    (void)scale;
    (void)mask;
    (void)bias;
    throwNotSupport("yt::function::scaledDotProductAttention", "FLASH_AVX2 backend requires AVX2/FMA");
    return yt::YTensor<T, dim>();
#endif
}

template<typename T, int dim, typename MaskFunc>
requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
yt::YTensor<T, dim> yt::function::_scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    MaskFunc&& mask,
    yt::YTensor<T, 2>* bias
) {
#if YT_USE_AVX2
    if constexpr (!std::is_same_v<T, float>) {
        throwNotSupport("yt::function::scaledDotProductAttention", "FLASH_AVX2 backend only supports float");
        return yt::YTensor<T, dim>();
    } else {
        auto queryMatView = query.matView();
        auto keyMatView = key.matView();
        auto valueMatView = value.matView();

        std::vector<int> outputShape;
        if constexpr (dim == 2) {
            outputShape = {query.shape(-2), value.shape(-1)};
        } else {
            outputShape = yt::kernel::computeBroadcastShape({queryMatView.shape(), keyMatView.shape(), valueMatView.shape()});
            outputShape.push_back(query.shape(-2));
            outputShape.push_back(value.shape(-1));
        }

        const float* biasPtr = nullptr;
        int64_t rsbias = 0;
        int64_t csbias = 0;
        if (bias != nullptr) {
            if (bias->shape(0) != query.shape(-2) || bias->shape(1) != key.shape(-2)) {
                throw std::invalid_argument("Bias shape must match the last two dimensions of the score tensor.");
            }
            biasPtr = bias->data();
            rsbias = bias->stride_(0);
            csbias = bias->stride_(1);
        }

        yt::YTensor<T, dim> output(outputShape);
        auto outputMatView = output.matView();
        auto maskFunc = std::forward<MaskFunc>(mask);
        outputMatView.broadcastInplace(
            [&maskFunc, scale, biasPtr, rsbias, csbias](yt::YTensor<T, 2>& o, const yt::YTensor<T, 2>& q, const yt::YTensor<T, 2>& k, const yt::YTensor<T, 2>& v) {
                const yt::YTensor<T, 2>* qKernel = &q;
                const yt::YTensor<T, 2>* kKernel = &k;
                const yt::YTensor<T, 2>* vKernel = &v;
                yt::YTensor<T, 2> qContiguous;
                yt::YTensor<T, 2> kContiguous;
                yt::YTensor<T, 2> vContiguous;

                if (q.shape(0) > 1) {
                    if (q.stride_(1) != 1) {
                        qContiguous = q.contiguous();
                        qKernel = &qContiguous;
                    }
                    if (k.stride_(1) != 1) {
                        kContiguous = k.contiguous();
                        kKernel = &kContiguous;
                    }
                    if (v.stride_(1) != 1) {
                        vContiguous = v.contiguous();
                        vKernel = &vContiguous;
                    }
                }

                auto qStride = qKernel->stride_();
                auto kStride = kKernel->stride_();
                auto vStride = vKernel->stride_();
                auto oStride = o.stride_();
                yt::kernel::avx2::flash_attention(
                    qKernel->data(),
                    kKernel->data(),
                    vKernel->data(),
                    o.data(),
                    qKernel->shape(0),
                    kKernel->shape(0),
                    qKernel->shape(1),
                    vKernel->shape(1),
                    scale,
                    static_cast<int64_t>(qStride[0]),
                    static_cast<int64_t>(qStride[1]),
                    static_cast<int64_t>(kStride[0]),
                    static_cast<int64_t>(kStride[1]),
                    static_cast<int64_t>(vStride[0]),
                    static_cast<int64_t>(vStride[1]),
                    static_cast<int64_t>(oStride[0]),
                    static_cast<int64_t>(oStride[1]),
                    maskFunc,
                    biasPtr,
                    rsbias,
                    csbias
                );
            },
            queryMatView, keyMatView, valueMatView
        );
        return output;
    }
#else
    (void)query;
    (void)key;
    (void)value;
    (void)scale;
    (void)mask;
    (void)bias;
    throwNotSupport("yt::function::scaledDotProductAttention", "FLASH_AVX2 backend requires AVX2/FMA");
    return yt::YTensor<T, dim>();
#endif
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::scaledDotProductAttention(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    yt::YTensor<bool, 2>* mask,
    std::nullptr_t,
    sdpaBackend backend
) {
    return yt::function::scaledDotProductAttention(
        query,
        key,
        value,
        scale,
        mask,
        static_cast<yt::YTensor<T, 2>*>(nullptr),
        backend
    );
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::scaledDotProductAttention(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    yt::YTensor<bool, 2>* mask,
    yt::YTensor<T, 2>* bias,
    sdpaBackend backend
) {
    if (static_cast<T>(0.0) == scale) {
        // auto
        scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(query.shape(-1)));
    }
    if (query.shape(-1) != key.shape(-1)) {
        throw std::invalid_argument("Query and key head dim must match.");
    }
    if (key.shape(-2) != value.shape(-2)) {
        throw std::invalid_argument("Key/value sequence length must match.");
    }
    if (backend == sdpaBackend::MATH) {
        yt::YTensor<T, dim> score;
        if (mask != nullptr) {
            if (mask->shape(0) != query.shape(-2) || mask->shape(1) != key.shape(-2)) {
                throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
            }
            score = query.masked_matmul(
                key.transpose(),
                *mask,
                static_cast<T>(-1e9)
            );
        } else {
            score = yt::function::matmul(query, key.transpose());
        }

        score.broadcastInplace([](T& a, const T& b) {
            a *= b;
        }, scale);
        if (bias != nullptr) {
            if (bias->shape(0) != score.shape(-2) || bias->shape(1) != score.shape(-1)) {
                throw std::invalid_argument("Bias shape must match the last two dimensions of the score tensor.");
            }
            score += *bias;
        }
        yt::function::softmax_(score, -1);
        auto output = yt::function::matmul(score, value);
        if (mask != nullptr) {
            yt::function::_zeroFullyMaskedSdpaRows(output, *mask);
        }
        return output;
    }
    if (backend == sdpaBackend::FLASH_AVX2) {
        return yt::function::_scaledDotProductAttentionFlash(query, key, value, scale, mask, bias);
    }

    throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
    return yt::YTensor<T, dim>();
}

template<typename T, int dim, typename MaskFunc>
requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
yt::YTensor<T, dim> yt::function::scaledDotProductAttention(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    MaskFunc&& mask,
    std::nullptr_t,
    sdpaBackend backend
) {
    return yt::function::scaledDotProductAttention(
        query,
        key,
        value,
        scale,
        std::forward<MaskFunc>(mask),
        static_cast<yt::YTensor<T, 2>*>(nullptr),
        backend
    );
}

template<typename T, int dim, typename MaskFunc>
requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
yt::YTensor<T, dim> yt::function::scaledDotProductAttention(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    MaskFunc&& mask,
    yt::YTensor<T, 2>* bias,
    sdpaBackend backend
) {
    if (static_cast<T>(0.0) == scale) {
        scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(query.shape(-1)));
    }
    if (query.shape(-1) != key.shape(-1)) {
        throw std::invalid_argument("Query and key head dim must match.");
    }
    if (key.shape(-2) != value.shape(-2)) {
        throw std::invalid_argument("Key/value sequence length must match.");
    }

    if (backend == sdpaBackend::MATH) {
        auto score = query.masked_matmul(
            key.transpose(),
            std::forward<MaskFunc>(mask),
            static_cast<T>(-1e9)
        );
        score.broadcastInplace([](T& a, const T& b) {
            a *= b;
        }, scale);
        if (bias != nullptr) {
            if (bias->shape(0) != score.shape(-2) || bias->shape(1) != score.shape(-1)) {
                throw std::invalid_argument("Bias shape must match the last two dimensions of the score tensor.");
            }
            score += *bias;
        }
        yt::function::softmax_(score, -1);
        auto output = yt::function::matmul(score, value);
        yt::function::_zeroFullyMaskedSdpaRows(output, std::forward<MaskFunc>(mask), key.shape(-2));
        return output;
    }
    if (backend == sdpaBackend::FLASH_AVX2) {
        return yt::function::_scaledDotProductAttentionFlash(
            query,
            key,
            value,
            scale,
            std::forward<MaskFunc>(mask),
            bias
        );
    }

    throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
    return yt::YTensor<T, dim>();
}

// ========== logsumexp ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logsumexp(const yt::YTensor<T, dim>& x, const std::vector<int>& axes) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logsumexp()");
    auto normalized_axes = yt::function::_normalizeAxes<dim>(axes);

    if constexpr (dim == 1) {
        T max_val = x.max(0).first;
        T sum_exp = static_cast<T>(0);
        for (int i = 0; i < x.shape(0); ++i) {
            sum_exp += std::exp(x.at(i) - max_val);
        }
        yt::YTensor<T, dim> output(1);
        output.at(0) = std::log(sum_exp) + max_val;
        return output;
    } else {
        auto max_vals = x.max(normalized_axes).first;
        auto exp_shifted = x.clone();
        exp_shifted.broadcastInplace([](T& a, const T& b) {
            a = std::exp(a - b);
        }, max_vals);
        auto reduced = exp_shifted.sum(normalized_axes);
        reduced.broadcastInplace([](T& a, const T& b) {
            a = std::log(a) + b;
        }, max_vals);
        return reduced;
    }
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logsumexp(const yt::YTensor<T, dim>& x, int axis) {
    return yt::function::logsumexp(x, std::vector<int>{axis});
}

// ========== logSoftmax ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logSoftmax(const yt::YTensor<T, dim>& x, const std::vector<int>& axes) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logSoftmax()");
    auto lse = yt::function::logsumexp(x, axes);
    auto output = x.clone();
    output.broadcastInplace([](T& a, const T& b) {
        a -= b;
    }, lse);
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logSoftmax(const yt::YTensor<T, dim>& x, int axis) {
    return yt::function::logSoftmax(x, std::vector<int>{axis});
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::logSoftmax_(yt::YTensor<T, dim>& x, const std::vector<int>& axes) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logSoftmax_()");
    auto lse = yt::function::logsumexp(x, axes);
    x.broadcastInplace([](T& a, const T& b) {
        a -= b;
    }, lse);
    return x;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::logSoftmax_(yt::YTensor<T, dim>& x, int axis) {
    return yt::function::logSoftmax_(x, std::vector<int>{axis});
}

// ========== maxPool1d ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::maxPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::maxPool1d()");
    axis = yt::function::_normalizeAxis<dim>(axis);
    if (stride < 0) stride = kernelSize;

    auto shape = x.shape();
    int input_size = shape[axis];
    int output_size = (input_size - kernelSize) / stride + 1;
    if (output_size <= 0) {
        throw std::invalid_argument("yt::function::maxPool1d: kernelSize too large for input dimension");
    }

    // 输出shape: axis维度变为output_size
    auto out_shape = shape;
    out_shape[axis] = output_size;
    yt::YTensor<T, dim> output(out_shape);

    // 构建迭代shape
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

        std::vector<int> in_indices, out_indices;
        int iter_pos = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == axis) {
                in_indices.push_back(0);
                out_indices.push_back(0);
            } else {
                in_indices.push_back(iter_indices[iter_pos]);
                out_indices.push_back(iter_indices[iter_pos]);
                iter_pos++;
            }
        }

        for (int o = 0; o < output_size; ++o) {
            int start = o * stride;
            in_indices[axis] = start;
            T max_val = x.at(in_indices);
            for (int k = 1; k < kernelSize; ++k) {
                in_indices[axis] = start + k;
                max_val = std::max(max_val, x.at(in_indices));
            }
            out_indices[axis] = o;
            output.at(out_indices) = max_val;
        }
    }

    return output;
}

// ========== avgPool1d ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::avgPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::avgPool1d()");
    axis = yt::function::_normalizeAxis<dim>(axis);
    if (stride < 0) stride = kernelSize;

    auto shape = x.shape();
    int input_size = shape[axis];
    int output_size = (input_size - kernelSize) / stride + 1;
    if (output_size <= 0) {
        throw std::invalid_argument("yt::function::avgPool1d: kernelSize too large for input dimension");
    }

    auto out_shape = shape;
    out_shape[axis] = output_size;
    yt::YTensor<T, dim> output(out_shape);

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) total_iterations *= s;

    T inv_k = static_cast<T>(1) / static_cast<T>(kernelSize);

    #pragma omp parallel for if(total_iterations > 1024)
    for (int64_t idx = 0; idx < total_iterations; ++idx) {
        std::vector<int> iter_indices(iter_shape.size());
        int64_t temp_idx = idx;
        for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
            iter_indices[i] = temp_idx % iter_shape[i];
            temp_idx /= iter_shape[i];
        }

        std::vector<int> in_indices, out_indices;
        int iter_pos = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == axis) {
                in_indices.push_back(0);
                out_indices.push_back(0);
            } else {
                in_indices.push_back(iter_indices[iter_pos]);
                out_indices.push_back(iter_indices[iter_pos]);
                iter_pos++;
            }
        }

        for (int o = 0; o < output_size; ++o) {
            int start = o * stride;
            T sum_val = static_cast<T>(0);
            for (int k = 0; k < kernelSize; ++k) {
                in_indices[axis] = start + k;
                sum_val += x.at(in_indices);
            }
            out_indices[axis] = o;
            output.at(out_indices) = sum_val * inv_k;
        }
    }

    return output;
}
