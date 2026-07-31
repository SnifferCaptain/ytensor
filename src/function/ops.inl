/***************
 * file: function/ops.inl
 * purpose: 高层线性层、attention、稳定归约和pooling实现。
 ***************/

template <typename T, int dim0, int dim1>
yt::YTensor<T, yt::utils::CONSTEXPR_MAX({dim0, dim1, 2})> yt::function::matmul(const yt::YTensor<T, dim0>& a, const yt::YTensor<T, dim1>& b) {
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

// 将完全不可见的attention query行强制归零。
// masked score使用有限sentinel时softmax仍可能产生非零分布，因此matmul后必须修正全mask行。
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

// callable mask版本的全mask行修正；mask返回true表示key对当前query可见。
// 注意：callback会被重复调用，必须在调用期间有效且不依赖一次性消费状态。
template<typename T, int dim, typename MaskFunc>
requires (!yt::utils::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
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
requires (!yt::utils::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
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

// YBLAS Flash Attention编排；仅float可用，二维mask/bias在所有广播batch间共享。
// 注意：output batch shape由Q/K/V matrix wrapper共同广播，kernel调用在函数返回前同步完成。
template<typename T, int dim>
yt::YTensor<T, dim> yt::function::_scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    const yt::YTensor<bool, 2>* mask,
    yt::YTensor<T, 2>* bias
) {
#if YT_USE_YBLAS
    if constexpr (!std::is_same_v<T, float>) {
        throwNotSupport("yt::function::scaledDotProductAttention", "FLASH backend only supports float");
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
            outputShape = yt::strided::computeBroadcastShape({queryMatView.shape(), keyMatView.shape(), valueMatView.shape()});
            outputShape.push_back(query.shape(-2));
            outputShape.push_back(value.shape(-1));
        }

        yt::YTensor<T, dim> output(outputShape);
        auto outputMatView = output.matView();

        // Flash mask接口按线性row-major读取，因此仅mask需要物化；bias backend显式接收二维stride。
        // 两类pointer都只在下方同步broadcast/kernel调用期间借用。
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

                // 单行decode已有专用strided fast path；多行query则物化病态view
                // （如转置cache叠加zero-stride repeat），使Flash kernel进入连续K/V累加路径。
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
                yt::blas::flash_attention(
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
    throwNotSupport(
        "yt::function::scaledDotProductAttention", "FLASH backend requires YBLAS"
    );
    return yt::YTensor<T, dim>();
#endif
}

// callable mask的YBLAS Flash Attention编排；callback在同步kernel期间按坐标调用。
// 注意：callback可能被backend并发调用，状态型实现必须自行保证线程安全。
template<typename T, int dim, typename MaskFunc>
requires (!yt::utils::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
yt::YTensor<T, dim> yt::function::_scaledDotProductAttentionFlash(
    yt::YTensor<T, dim>& query,
    yt::YTensor<T, dim>& key,
    yt::YTensor<T, dim>& value,
    T scale,
    MaskFunc&& mask,
    yt::YTensor<T, 2>* bias
) {
#if YT_USE_YBLAS
    if constexpr (!std::is_same_v<T, float>) {
        throwNotSupport("yt::function::scaledDotProductAttention", "FLASH backend only supports float");
        return yt::YTensor<T, dim>();
    } else {
        auto queryMatView = query.matView();
        auto keyMatView = key.matView();
        auto valueMatView = value.matView();

        std::vector<int> outputShape;
        if constexpr (dim == 2) {
            outputShape = {query.shape(-2), value.shape(-1)};
        } else {
            outputShape = yt::strided::computeBroadcastShape({queryMatView.shape(), keyMatView.shape(), valueMatView.shape()});
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

                // 多行query的Flash kernel要求连续inner axis；单行decode保留合法的strided零拷贝路径。
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
                yt::blas::flash_attention(
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
    throwNotSupport(
        "yt::function::scaledDotProductAttention", "FLASH backend requires YBLAS"
    );
    return yt::YTensor<T, dim>();
#endif
}

// scaled dot-product attention主入口。
// scale==0表示自动使用1/sqrt(headDim)；mask中true表示可见，二维mask/bias复用于所有batch。
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
            // 有限-1e9避免直接写入无穷导致部分低精度路径异常；全mask行在最终输出上单独归零。
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
    if (backend == sdpaBackend::FLASH) {
        return yt::function::_scaledDotProductAttentionFlash(query, key, value, scale, mask, bias);
    }

    throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
    return yt::YTensor<T, dim>();
}

// callable mask的SDPA入口；mask需可重复调用，因为score计算和全mask行修正都会使用它。
template<typename T, int dim, typename MaskFunc>
requires (!yt::utils::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
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
requires (!yt::utils::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
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
    if (backend == sdpaBackend::FLASH) {
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

// 使用max-shift实现数值稳定的logsumexp，并保留reduced轴用于后续广播。
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
        // x-max避免exp上溢；keep-dim max/sum可直接广播回原shape。
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

// 通过`x - logsumexp(x)`计算log-softmax，不改变输入layout。
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

// 原地log-softmax；仅修改values，不改变shape/stride metadata。
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

// 无padding的一维最大池化；只保留完整窗口，stride<0表示使用kernelSize。
// 注意：axis支持负索引，输出长度为floor((input-kernel)/stride)+1。
template<typename T, int dim>
yt::YTensor<T, dim> yt::function::maxPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::maxPool1d()");
    axis = yt::function::_normalizeAxis<dim>(axis);
    if (kernelSize <= 0) throw std::invalid_argument("yt::function::maxPool1d: kernelSize must be positive");
    if (stride < 0) stride = kernelSize;
    if (stride == 0) throw std::invalid_argument("yt::function::maxPool1d: stride must be positive");

    auto shape = x.shape();
    int input_size = shape[axis];
    int64_t output_size_wide =
        (static_cast<int64_t>(input_size) - kernelSize) / stride + 1;
    if (output_size_wide > std::numeric_limits<int>::max()) {
        throw std::overflow_error("yt::function::maxPool1d: output size overflow");
    }
    int output_size = static_cast<int>(output_size_wide);
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

    // 将非pool轴坐标展平；每个iteration写入独立输出slice，超过阈值后并行无数据竞争。
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

// 无padding的一维平均池化；窗口、stride和输出长度规则与maxPool1d一致。
// 注意：T为整数时inv_k按整数除法计算，这是当前typed算术合同。
template<typename T, int dim>
yt::YTensor<T, dim> yt::function::avgPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::avgPool1d()");
    axis = yt::function::_normalizeAxis<dim>(axis);
    if (kernelSize <= 0) throw std::invalid_argument("yt::function::avgPool1d: kernelSize must be positive");
    if (stride < 0) stride = kernelSize;
    if (stride == 0) throw std::invalid_argument("yt::function::avgPool1d: stride must be positive");

    auto shape = x.shape();
    int input_size = shape[axis];
    int64_t output_size_wide =
        (static_cast<int64_t>(input_size) - kernelSize) / stride + 1;
    if (output_size_wide > std::numeric_limits<int>::max()) {
        throw std::overflow_error("yt::function::avgPool1d: output size overflow");
    }
    int output_size = static_cast<int>(output_size_wide);
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

    // 非pool轴展平后各iteration写入独立slice；1024仅是并行粒度阈值。
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
