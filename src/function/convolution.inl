#pragma once
/***************
 * file: function/convolution.inl
 * purpose: 基于融合implicit-im2col packing的一维、二维卷积实现。
 ***************/

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace yt::function {

inline int computeConvolutionOutputExtent(
    int input_size, int kernel_size, int stride, int padding_before, int padding_after, int dilation,
    const char* operation
) {
    if (kernel_size <= 0)
        throw std::invalid_argument(std::string(operation) + ": kernel size must be positive");
    if (stride <= 0) throw std::invalid_argument(std::string(operation) + ": stride must be positive");
    if (padding_before < 0 || padding_after < 0)
        throw std::invalid_argument(std::string(operation) + ": padding must be non-negative");
    if (dilation <= 0) throw std::invalid_argument(std::string(operation) + ": dilation must be positive");

    const int64_t effective_kernel = static_cast<int64_t>(kernel_size - 1) * dilation + 1;
    const int64_t padded_input = static_cast<int64_t>(input_size) + padding_before + padding_after;
    if (effective_kernel > padded_input) {
        throw std::invalid_argument(std::string(operation) + ": effective kernel exceeds padded input");
    }
    const int64_t output_size = (padded_input - effective_kernel) / stride + 1;
    if (output_size > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string(operation) + ": output size overflow");
    }
    return static_cast<int>(output_size);
}

inline int checkedConvolutionExtentProduct(int left, int right, const char* operation, const char* quantity) {
    if (left < 0 || right < 0) {
        throw std::invalid_argument(std::string(operation) + ": " + quantity + " must be non-negative");
    }
    const int64_t product = static_cast<int64_t>(left) * right;
    if (product > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string(operation) + ": " + quantity + " overflow");
    }
    return static_cast<int>(product);
}

inline std::size_t checkedConvolutionBufferProduct(
    std::size_t left, std::size_t right, const char* operation, const char* quantity
) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(std::string(operation) + ": " + quantity + " overflow");
    }
    return left * right;
}

template <std::size_t Size>
inline std::array<int, Size> validateConvolutionPadding(
    const std::vector<int>& padding, const char* operation
) {
    if (padding.size() != Size) {
        throw std::invalid_argument(
            std::string(operation) + ": padding must contain " + std::to_string(Size) + " values"
        );
    }
    std::array<int, Size> result{};
    std::copy_n(padding.begin(), Size, result.begin());
    return result;
}

template <typename T>
inline void validateConvolutionChannels(
    int input_channels, int output_channels, int weight_input_channels, int groups,
    const yt::YTensor<T, 1>* bias, const char* operation
) {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, yt::float16>,
        "convolution supports float or float16 storage"
    );
    if (groups <= 0) throw std::invalid_argument(std::string(operation) + ": groups must be positive");
    if (input_channels <= 0 || output_channels <= 0) {
        throw std::invalid_argument(std::string(operation) + ": channels must be positive");
    }
    if (input_channels % groups != 0) {
        throw std::invalid_argument(std::string(operation) + ": input channels must be divisible by groups");
    }
    if (output_channels % groups != 0) {
        throw std::invalid_argument(std::string(operation) + ": output channels must be divisible by groups");
    }
    if (weight_input_channels != input_channels / groups) {
        throw std::invalid_argument(std::string(operation) + ": weight input channels do not match groups");
    }
    if (bias != nullptr && bias->shape(0) != output_channels) {
        throw std::invalid_argument(std::string(operation) + ": bias size must match output channels");
    }
}

#if YT_USE_YBLAS
template <typename T, yt::blas::GemmKernelSpec Spec, typename Dispatcher, typename PackB>
inline void computePackedConvolutionPanels(
    T* output, int output_channels, int output_spatial, int reduction_size, int64_t output_row_stride,
    const T* bias, int64_t bias_stride, const float* packed_weights, PackB&& pack_b, const char* operation
) {
    const int channel_tiles = output_channels / Spec.row_mr + (output_channels % Spec.row_mr != 0);
    const int spatial_tiles = output_spatial / Spec.row_nr + (output_spatial % Spec.row_nr != 0);
    if (output_row_stride < 0 || output_row_stride > std::numeric_limits<int>::max()) {
        throw std::overflow_error(std::string(operation) + ": output row stride overflow");
    }
    const std::size_t packed_input_size = checkedConvolutionBufferProduct(
        static_cast<std::size_t>(reduction_size), static_cast<std::size_t>(Spec.row_nr), operation,
        "packed input size"
    );

    auto compute_spatial_tile = [&](int tile, float* packed_input) {
        const int column = tile * Spec.row_nr;
        const int nr = std::min(Spec.row_nr, output_spatial - column);
        pack_b(packed_input, column, 0, nr, reduction_size);
        for (int channel_tile = 0; channel_tile < channel_tiles; ++channel_tile) {
            const int channel = channel_tile * Spec.row_mr;
            const int mr = std::min(Spec.row_mr, output_channels - channel);
            const float* packed_weight =
                packed_weights + static_cast<std::size_t>(channel_tile) * reduction_size * Spec.row_mr;
            if constexpr (std::is_same_v<T, float>) {
                if (mr == Spec.row_mr && nr == Spec.row_nr) {
                    float* destination = output + channel * output_row_stride + column;
                    Dispatcher::template compute<false>(
                        packed_weight, packed_input, destination, reduction_size,
                        static_cast<int>(output_row_stride)
                    );
                    if (bias != nullptr) {
                        for (int row = 0; row < Spec.row_mr; ++row) {
                            const float value = static_cast<float>(bias[(channel + row) * bias_stride]);
                            for (int j = 0; j < Spec.row_nr; ++j) {
                                destination[row * output_row_stride + j] += value;
                            }
                        }
                    }
                    continue;
                }
            }

            alignas(64) float result[Spec.row_mr * Spec.row_nr];
            Dispatcher::template compute<false>(
                packed_weight, packed_input, result, reduction_size, Spec.row_nr
            );
            for (int row = 0; row < mr; ++row) {
                const float bias_value =
                    bias == nullptr ? 0.0f : static_cast<float>(bias[(channel + row) * bias_stride]);
                for (int j = 0; j < nr; ++j) {
                    output[(channel + row) * output_row_stride + column + j] =
                        static_cast<T>(result[row * Spec.row_nr + j] + bias_value);
                }
            }
        }
    };

#ifdef _OPENMP
    const int thread_count = yt::blas::get_num_threads();
    if (thread_count > 1 && spatial_tiles > 1) {
        std::vector<std::vector<float>> packed_inputs;
        packed_inputs.reserve(thread_count);
        for (int thread = 0; thread < thread_count; ++thread) {
            packed_inputs.emplace_back(packed_input_size);
        }
#pragma omp parallel num_threads(thread_count) proc_bind(spread)
        {
            float* packed_input = packed_inputs[omp_get_thread_num()].data();
#pragma omp for schedule(static)
            for (int tile = 0; tile < spatial_tiles; ++tile) {
                compute_spatial_tile(tile, packed_input);
            }
        }
        return;
    }
#endif

    std::vector<float> packed_input(packed_input_size);
    for (int tile = 0; tile < spatial_tiles; ++tile) compute_spatial_tile(tile, packed_input.data());
}
#endif

#if !YT_USE_YBLAS
template <typename T>
inline void referenceConv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, const yt::YTensor<T, 1>* bias,
    yt::YTensor<T, 3>& output, int stride, std::array<int, 2> padding, int dilation, int groups
) {
    const int output_channels_per_group = output.shape(1) / groups;
    const int input_channels_per_group = input.shape(1) / groups;
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    for (int batch = 0; batch < output.shape(0); ++batch) {
        for (int output_channel = 0; output_channel < output.shape(1); ++output_channel) {
            const int group = output_channel / output_channels_per_group;
            for (int output_x = 0; output_x < output.shape(2); ++output_x) {
                float sum = bias == nullptr
                                ? 0.0f
                                : static_cast<float>(bias->data()[output_channel * bias->stride_(0)]);
                for (int input_channel_local = 0; input_channel_local < input_channels_per_group;
                     ++input_channel_local) {
                    const int input_channel = group * input_channels_per_group + input_channel_local;
                    for (int kernel_x = 0; kernel_x < weight.shape(2); ++kernel_x) {
                        const int64_t input_x = static_cast<int64_t>(output_x) * stride - padding[0] +
                                                static_cast<int64_t>(kernel_x) * dilation;
                        if (input_x < 0 || input_x >= input.shape(2)) continue;
                        sum += static_cast<float>(input.data(
                               )[batch * input_stride[0] + input_channel * input_stride[1] +
                                 input_x * input_stride[2]]) *
                               static_cast<float>(weight.data(
                               )[output_channel * weight_stride[0] + input_channel_local * weight_stride[1] +
                                 kernel_x * weight_stride[2]]);
                    }
                }
                output.data(
                )[batch * output_stride[0] + output_channel * output_stride[1] +
                  output_x * output_stride[2]] = static_cast<T>(sum);
            }
        }
    }
}

template <typename T>
inline void referenceConv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, const yt::YTensor<T, 1>* bias,
    yt::YTensor<T, 4>& output, std::array<int, 2> stride, std::array<int, 4> padding,
    std::array<int, 2> dilation, int groups
) {
    const int output_channels_per_group = output.shape(1) / groups;
    const int input_channels_per_group = input.shape(1) / groups;
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    for (int batch = 0; batch < output.shape(0); ++batch) {
        for (int output_channel = 0; output_channel < output.shape(1); ++output_channel) {
            const int group = output_channel / output_channels_per_group;
            for (int output_y = 0; output_y < output.shape(2); ++output_y) {
                for (int output_x = 0; output_x < output.shape(3); ++output_x) {
                    float sum = bias == nullptr
                                    ? 0.0f
                                    : static_cast<float>(bias->data()[output_channel * bias->stride_(0)]);
                    for (int input_channel_local = 0; input_channel_local < input_channels_per_group;
                         ++input_channel_local) {
                        const int input_channel = group * input_channels_per_group + input_channel_local;
                        for (int kernel_y = 0; kernel_y < weight.shape(2); ++kernel_y) {
                            const int64_t input_y = static_cast<int64_t>(output_y) * stride[0] - padding[2] +
                                                    static_cast<int64_t>(kernel_y) * dilation[0];
                            if (input_y < 0 || input_y >= input.shape(2)) continue;
                            for (int kernel_x = 0; kernel_x < weight.shape(3); ++kernel_x) {
                                const int64_t input_x = static_cast<int64_t>(output_x) * stride[1] -
                                                        padding[0] +
                                                        static_cast<int64_t>(kernel_x) * dilation[1];
                                if (input_x < 0 || input_x >= input.shape(3)) continue;
                                sum += static_cast<float>(input.data(
                                       )[batch * input_stride[0] + input_channel * input_stride[1] +
                                         input_y * input_stride[2] + input_x * input_stride[3]]) *
                                       static_cast<float>(weight.data(
                                       )[output_channel * weight_stride[0] +
                                         input_channel_local * weight_stride[1] +
                                         kernel_y * weight_stride[2] + kernel_x * weight_stride[3]]);
                            }
                        }
                    }
                    output.data(
                    )[batch * output_stride[0] + output_channel * output_stride[1] +
                      output_y * output_stride[2] + output_x * output_stride[3]] = static_cast<T>(sum);
                }
            }
        }
    }
}
#endif

template <typename T>
inline void depthwiseConv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, const yt::YTensor<T, 1>* bias,
    yt::YTensor<T, 3>& output, int stride, std::array<int, 2> padding, int dilation
) {
    const int batches = output.shape(0);
    const int output_channels = output.shape(1);
    const int output_length = output.shape(2);
    const int input_channels = input.shape(1);
    const int input_length = input.shape(2);
    const int kernel_size = weight.shape(2);
    const int outputs_per_input = output_channels / input_channels;
    const T* input_data = input.data();
    const T* weight_data = weight.data();
    const T* bias_data = bias == nullptr ? nullptr : bias->data();
    T* output_data = output.data();
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    const int64_t rows = static_cast<int64_t>(batches) * output_channels;
#ifdef _OPENMP
    const int thread_count = yt::blas::get_num_threads();
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(std::max(1, thread_count)) if (thread_count > 1 && rows >= 32)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const int output_channel = static_cast<int>(row % output_channels);
        const int batch = static_cast<int>(row / output_channels);
        const int input_channel = output_channel / outputs_per_input;
        const float bias_value =
            bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[output_channel * bias->stride_(0)]);
        const T* input_row = input_data + batch * input_stride[0] + input_channel * input_stride[1];
        const T* kernel = weight_data + output_channel * weight_stride[0];
        T* destination = output_data + batch * output_stride[0] + output_channel * output_stride[1];
        int x = 0;
#if YT_USE_AVX2
        if constexpr (std::is_same_v<T, float>) {
            if (stride == 1 && dilation == 1 && input_stride[2] == 1 && output_stride[2] == 1) {
                while (x < output_length && x - padding[0] < 0) {
                    float sum = bias_value;
                    for (int k = 0; k < kernel_size; ++k) {
                        const int64_t input_x = static_cast<int64_t>(x) - padding[0] + k;
                        if (input_x >= 0 && input_x < input_length)
                            sum += input_row[input_x] * kernel[k * weight_stride[2]];
                    }
                    destination[x++] = sum;
                }
                for (; x + 7 < output_length && x - padding[0] + 7 + kernel_size <= input_length; x += 8) {
                    __m256 sum = _mm256_set1_ps(bias_value);
                    for (int k = 0; k < kernel_size; ++k) {
                        const __m256 values = _mm256_loadu_ps(input_row + x - padding[0] + k);
                        sum = _mm256_fmadd_ps(values, _mm256_set1_ps(kernel[k * weight_stride[2]]), sum);
                    }
                    _mm256_storeu_ps(destination + x, sum);
                }
            }
        }
#endif
        for (; x < output_length; ++x) {
            float sum = bias_value;
            for (int k = 0; k < kernel_size; ++k) {
                const int64_t input_x =
                    static_cast<int64_t>(x) * stride - padding[0] + static_cast<int64_t>(k) * dilation;
                if (input_x >= 0 && input_x < input_length)
                    sum += static_cast<float>(input_row[input_x * input_stride[2]]) *
                           static_cast<float>(kernel[k * weight_stride[2]]);
            }
            destination[x * output_stride[2]] = static_cast<T>(sum);
        }
    }
}

template <typename T>
inline void depthwiseConv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, const yt::YTensor<T, 1>* bias,
    yt::YTensor<T, 4>& output, std::array<int, 2> stride, std::array<int, 4> padding,
    std::array<int, 2> dilation
) {
    const int batches = output.shape(0);
    const int output_channels = output.shape(1);
    const int output_height = output.shape(2);
    const int output_width = output.shape(3);
    const int input_channels = input.shape(1);
    const int input_height = input.shape(2);
    const int input_width = input.shape(3);
    const int kernel_height = weight.shape(2);
    const int kernel_width = weight.shape(3);
    const int outputs_per_input = output_channels / input_channels;
    const T* input_data = input.data();
    const T* weight_data = weight.data();
    const T* bias_data = bias == nullptr ? nullptr : bias->data();
    T* output_data = output.data();
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    const int64_t rows = static_cast<int64_t>(batches) * output_channels * output_height;
#ifdef _OPENMP
    const int thread_count = yt::blas::get_num_threads();
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(std::max(1, thread_count)) if (thread_count > 1 && rows >= 32)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        int64_t coordinate = row;
        const int output_y = static_cast<int>(coordinate % output_height);
        coordinate /= output_height;
        const int output_channel = static_cast<int>(coordinate % output_channels);
        const int batch = static_cast<int>(coordinate / output_channels);
        const int input_channel = output_channel / outputs_per_input;
        const float bias_value =
            bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[output_channel * bias->stride_(0)]);
        const T* input_plane = input_data + batch * input_stride[0] + input_channel * input_stride[1];
        const T* kernel = weight_data + output_channel * weight_stride[0];
        T* destination = output_data + batch * output_stride[0] + output_channel * output_stride[1] +
                         output_y * output_stride[2];
        int x = 0;
#if YT_USE_AVX2
        if constexpr (std::is_same_v<T, float>) {
            if (stride[1] == 1 && dilation[1] == 1 && input_stride[3] == 1 && output_stride[3] == 1) {
                while (x < output_width && x - padding[0] < 0) {
                    float sum = bias_value;
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int64_t input_y = static_cast<int64_t>(output_y) * stride[0] - padding[2] +
                                                static_cast<int64_t>(ky) * dilation[0];
                        if (input_y < 0 || input_y >= input_height) continue;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            const int64_t input_x = static_cast<int64_t>(x) - padding[0] + kx;
                            if (input_x >= 0 && input_x < input_width)
                                sum += input_plane[input_y * input_stride[2] + input_x] *
                                       kernel[ky * weight_stride[2] + kx * weight_stride[3]];
                        }
                    }
                    destination[x++] = sum;
                }
                for (; x + 7 < output_width && x - padding[0] + 7 + kernel_width <= input_width; x += 8) {
                    __m256 sum = _mm256_set1_ps(bias_value);
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int64_t input_y = static_cast<int64_t>(output_y) * stride[0] - padding[2] +
                                                static_cast<int64_t>(ky) * dilation[0];
                        if (input_y < 0 || input_y >= input_height) continue;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            const __m256 values = _mm256_loadu_ps(
                                input_plane + input_y * input_stride[2] + x - padding[0] + kx
                            );
                            sum = _mm256_fmadd_ps(
                                values, _mm256_set1_ps(kernel[ky * weight_stride[2] + kx * weight_stride[3]]),
                                sum
                            );
                        }
                    }
                    _mm256_storeu_ps(destination + x, sum);
                }
            }
        }
#endif
        for (; x < output_width; ++x) {
            float sum = bias_value;
            for (int ky = 0; ky < kernel_height; ++ky) {
                const int64_t input_y = static_cast<int64_t>(output_y) * stride[0] - padding[2] +
                                        static_cast<int64_t>(ky) * dilation[0];
                if (input_y < 0 || input_y >= input_height) continue;
                for (int kx = 0; kx < kernel_width; ++kx) {
                    const int64_t input_x = static_cast<int64_t>(x) * stride[1] - padding[0] +
                                            static_cast<int64_t>(kx) * dilation[1];
                    if (input_x >= 0 && input_x < input_width)
                        sum += static_cast<float>(
                                   input_plane[input_y * input_stride[2] + input_x * input_stride[3]]
                               ) *
                               static_cast<float>(kernel[ky * weight_stride[2] + kx * weight_stride[3]]);
                }
            }
            destination[x * output_stride[3]] = static_cast<T>(sum);
        }
    }
}

template <typename T>
yt::YTensor<T, 3> dispatchConv1dWithOptionalBias(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, const yt::YTensor<T, 1>* bias,
    int stride, std::array<int, 2> padding, int dilation, int groups
) {
#if YT_USE_YBLAS
    constexpr yt::blas::GemmKernelSpec spec = yt::blas::default_gemm_kernel_spec;
    using Dispatcher = typename yt::blas::DefaultGemmDispatcher<T, T, T, float, spec>::type;
#endif
    constexpr const char* operation = "yt::function::conv1d";
    const int batches = input.shape(0);
    const int input_channels = input.shape(1);
    const int input_length = input.shape(2);
    const int output_channels = weight.shape(0);
    const int kernel_size = weight.shape(2);
    validateConvolutionChannels(input_channels, output_channels, weight.shape(1), groups, bias, operation);
    const int output_length =
        computeConvolutionOutputExtent(
            input_length, kernel_size, stride, padding[0], padding[1], dilation, operation
        );

    yt::YTensor<T, 3> output(batches, output_channels, output_length);
    if (groups == input_channels) {
        depthwiseConv1d(input, weight, bias, output, stride, padding, dilation);
        return output;
    }

#if !YT_USE_YBLAS
    referenceConv1d(input, weight, bias, output, stride, padding, dilation, groups);
    return output;
#else
    const int input_channels_per_group = input_channels / groups;
    const int output_channels_per_group = output_channels / groups;
    const int reduction_size =
        checkedConvolutionExtentProduct(
            input_channels_per_group, kernel_size, operation, "reduction size"
        );
    const T* input_data = input.data();
    const T* weight_data = weight.data();
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    const int channel_tiles =
        output_channels_per_group / spec.row_mr + (output_channels_per_group % spec.row_mr != 0);
    const std::size_t packed_weight_panel_size = checkedConvolutionBufferProduct(
        static_cast<std::size_t>(reduction_size), static_cast<std::size_t>(spec.row_mr), operation,
        "packed weight panel size"
    );
    const std::size_t packed_weight_size = checkedConvolutionBufferProduct(
        static_cast<std::size_t>(channel_tiles), packed_weight_panel_size, operation, "packed weight size"
    );

    for (int group = 0; group < groups; ++group) {
        const int input_channel_start = group * input_channels_per_group;
        const int output_channel_start = group * output_channels_per_group;
        std::vector<float> packed_weights(packed_weight_size);
        float* packed_weight = packed_weights.data();
        for (int output_tile = 0; output_tile < output_channels_per_group; output_tile += spec.row_mr) {
            const int mr = std::min(spec.row_mr, output_channels_per_group - output_tile);
            int input_channel = 0;
            int kernel = 0;
            for (int p = 0; p < reduction_size; ++p) {
                for (int row = 0; row < mr; ++row) {
                    const int output_channel = output_channel_start + output_tile + row;
                    packed_weight[row] =
                        static_cast<float>(weight_data
                                               [output_channel * weight_stride[0] +
                                                input_channel * weight_stride[1] + kernel * weight_stride[2]]
                        );
                }
                for (int row = mr; row < spec.row_mr; ++row) packed_weight[row] = 0.0f;
                packed_weight += spec.row_mr;
                if (++kernel == kernel_size) {
                    kernel = 0;
                    ++input_channel;
                }
            }
        }

        for (int batch = 0; batch < batches; ++batch) {
            auto pack_b = [&](float* packed, int jj, int pp, int nc, int kc) {
                for (int jr = 0; jr < nc; jr += spec.row_nr) {
                    const int nr = std::min(spec.row_nr, nc - jr);
                    std::array<int64_t, spec.row_nr> input_x_base{};
                    for (int column = 0; column < nr; ++column) {
                        input_x_base[column] = static_cast<int64_t>(jj + jr + column) * stride - padding[0];
                    }
                    int input_channel = input_channel_start + pp / kernel_size;
                    int kernel = pp % kernel_size;
                    for (int p = 0; p < kc; ++p) {
                        const int64_t first_input_x =
                            input_x_base[0] + static_cast<int64_t>(kernel) * dilation;
                        if constexpr (std::is_same_v<T, float>) {
                            if (stride == 1 && input_stride[2] == 1 && first_input_x >= 0 &&
                                first_input_x + nr <= input_length) {
                                const float* source = input_data + batch * input_stride[0] +
                                                      input_channel * input_stride[1] + first_input_x;
                                std::copy_n(source, nr, packed);
                            } else {
                                for (int column = 0; column < nr; ++column) {
                                    const int64_t input_x =
                                        input_x_base[column] + static_cast<int64_t>(kernel) * dilation;
                                    packed[column] =
                                        input_x >= 0 && input_x < input_length
                                            ? input_data
                                                  [batch * input_stride[0] + input_channel * input_stride[1] +
                                                   input_x * input_stride[2]]
                                            : 0.0f;
                                }
                            }
                        } else {
                            for (int column = 0; column < nr; ++column) {
                                const int64_t input_x =
                                    input_x_base[column] + static_cast<int64_t>(kernel) * dilation;
                                packed[column] =
                                    input_x >= 0 && input_x < input_length
                                        ? static_cast<float>(
                                              input_data
                                                  [batch * input_stride[0] + input_channel * input_stride[1] +
                                                   input_x * input_stride[2]]
                                          )
                                        : 0.0f;
                            }
                        }
                        for (int column = nr; column < spec.row_nr; ++column) packed[column] = 0.0f;
                        packed += spec.row_nr;
                        if (++kernel == kernel_size) {
                            kernel = 0;
                            ++input_channel;
                        }
                    }
                }
            };
            T* output_group =
                output.data() + batch * output_stride[0] + output_channel_start * output_stride[1];
            const T* bias_group =
                bias == nullptr ? nullptr : bias->data() + output_channel_start * bias->stride_(0);
            computePackedConvolutionPanels<T, spec, Dispatcher>(
                output_group, output_channels_per_group, output_length, reduction_size, output_stride[1],
                bias_group, bias == nullptr ? 0 : bias->stride_(0), packed_weights.data(), pack_b, operation
            );
        }
    }
    return output;
#endif
}

template <typename T>
yt::YTensor<T, 4> dispatchConv2dWithOptionalBias(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, const yt::YTensor<T, 1>* bias,
    std::array<int, 2> stride, std::array<int, 4> padding, std::array<int, 2> dilation, int groups
) {
#if YT_USE_YBLAS
    constexpr yt::blas::GemmKernelSpec spec = yt::blas::default_gemm_kernel_spec;
    using Dispatcher = typename yt::blas::DefaultGemmDispatcher<T, T, T, float, spec>::type;
#endif
    constexpr const char* operation = "yt::function::conv2d";
    const int batches = input.shape(0);
    const int input_channels = input.shape(1);
    const int input_height = input.shape(2);
    const int input_width = input.shape(3);
    const int output_channels = weight.shape(0);
    const int kernel_height = weight.shape(2);
    const int kernel_width = weight.shape(3);
    validateConvolutionChannels(input_channels, output_channels, weight.shape(1), groups, bias, operation);
    const int output_height =
        computeConvolutionOutputExtent(
            input_height, kernel_height, stride[0], padding[2], padding[3], dilation[0], operation
        );
    const int output_width =
        computeConvolutionOutputExtent(
            input_width, kernel_width, stride[1], padding[0], padding[1], dilation[1], operation
        );

    yt::YTensor<T, 4> output(batches, output_channels, output_height, output_width);
    if (groups == input_channels) {
        depthwiseConv2d(input, weight, bias, output, stride, padding, dilation);
        return output;
    }

#if !YT_USE_YBLAS
    referenceConv2d(input, weight, bias, output, stride, padding, dilation, groups);
    return output;
#else
    const int input_channels_per_group = input_channels / groups;
    const int output_channels_per_group = output_channels / groups;
    const int kernel_area =
        checkedConvolutionExtentProduct(kernel_height, kernel_width, operation, "kernel area");
    const int reduction_size =
        checkedConvolutionExtentProduct(input_channels_per_group, kernel_area, operation, "reduction size");
    const int output_spatial =
        checkedConvolutionExtentProduct(output_height, output_width, operation, "output spatial size");
    const T* input_data = input.data();
    const T* weight_data = weight.data();
    const auto input_stride = input.stride_();
    const auto weight_stride = weight.stride_();
    const auto output_stride = output.stride_();
    const int channel_tiles =
        output_channels_per_group / spec.row_mr + (output_channels_per_group % spec.row_mr != 0);
    const std::size_t packed_weight_panel_size = checkedConvolutionBufferProduct(
        static_cast<std::size_t>(reduction_size), static_cast<std::size_t>(spec.row_mr), operation,
        "packed weight panel size"
    );
    const std::size_t packed_weight_size = checkedConvolutionBufferProduct(
        static_cast<std::size_t>(channel_tiles), packed_weight_panel_size, operation, "packed weight size"
    );

    for (int group = 0; group < groups; ++group) {
        const int input_channel_start = group * input_channels_per_group;
        const int output_channel_start = group * output_channels_per_group;
        std::vector<float> packed_weights(packed_weight_size);
        float* packed_weight = packed_weights.data();
        for (int output_tile = 0; output_tile < output_channels_per_group; output_tile += spec.row_mr) {
            const int mr = std::min(spec.row_mr, output_channels_per_group - output_tile);
            int input_channel = 0;
            int kernel_y = 0;
            int kernel_x = 0;
            for (int p = 0; p < reduction_size; ++p) {
                for (int row = 0; row < mr; ++row) {
                    const int output_channel = output_channel_start + output_tile + row;
                    packed_weight[row] = static_cast<float>(
                        weight_data
                            [output_channel * weight_stride[0] + input_channel * weight_stride[1] +
                             kernel_y * weight_stride[2] + kernel_x * weight_stride[3]]
                    );
                }
                for (int row = mr; row < spec.row_mr; ++row) packed_weight[row] = 0.0f;
                packed_weight += spec.row_mr;
                if (++kernel_x == kernel_width) {
                    kernel_x = 0;
                    if (++kernel_y == kernel_height) {
                        kernel_y = 0;
                        ++input_channel;
                    }
                }
            }
        }

        for (int batch = 0; batch < batches; ++batch) {
            auto pack_b = [&](float* packed, int jj, int pp, int nc, int kc) {
                for (int jr = 0; jr < nc; jr += spec.row_nr) {
                    const int nr = std::min(spec.row_nr, nc - jr);
                    std::array<int64_t, spec.row_nr> input_y_base{};
                    std::array<int64_t, spec.row_nr> input_x_base{};
                    for (int column = 0; column < nr; ++column) {
                        const int output_position = jj + jr + column;
                        input_y_base[column] =
                            static_cast<int64_t>(output_position / output_width) * stride[0] - padding[2];
                        input_x_base[column] =
                            static_cast<int64_t>(output_position % output_width) * stride[1] - padding[0];
                    }
                    int input_channel = input_channel_start + pp / kernel_area;
                    int kernel_offset = pp % kernel_area;
                    int kernel_y = kernel_offset / kernel_width;
                    int kernel_x = kernel_offset % kernel_width;
                    for (int p = 0; p < kc; ++p) {
                        const int64_t first_input_y =
                            input_y_base[0] + static_cast<int64_t>(kernel_y) * dilation[0];
                        const int64_t first_input_x =
                            input_x_base[0] + static_cast<int64_t>(kernel_x) * dilation[1];
                        const int first_output_x = (jj + jr) % output_width;
                        if constexpr (std::is_same_v<T, float>) {
                            if (stride[1] == 1 && input_stride[3] == 1 &&
                                first_output_x + nr <= output_width && first_input_y >= 0 &&
                                first_input_y < input_height && first_input_x >= 0 &&
                                first_input_x + nr <= input_width) {
                                const float* source = input_data + batch * input_stride[0] +
                                                      input_channel * input_stride[1] +
                                                      first_input_y * input_stride[2] + first_input_x;
                                std::copy_n(source, nr, packed);
                            } else {
                                for (int column = 0; column < nr; ++column) {
                                    const int64_t input_y =
                                        input_y_base[column] + static_cast<int64_t>(kernel_y) * dilation[0];
                                    const int64_t input_x =
                                        input_x_base[column] + static_cast<int64_t>(kernel_x) * dilation[1];
                                    packed[column] =
                                        input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                                input_x < input_width
                                            ? input_data
                                                  [batch * input_stride[0] + input_channel * input_stride[1] +
                                                   input_y * input_stride[2] + input_x * input_stride[3]]
                                            : 0.0f;
                                }
                            }
                        } else {
                            for (int column = 0; column < nr; ++column) {
                                const int64_t input_y =
                                    input_y_base[column] + static_cast<int64_t>(kernel_y) * dilation[0];
                                const int64_t input_x =
                                    input_x_base[column] + static_cast<int64_t>(kernel_x) * dilation[1];
                                packed[column] =
                                    input_y >= 0 && input_y < input_height && input_x >= 0 &&
                                            input_x < input_width
                                        ? static_cast<float>(
                                              input_data
                                                  [batch * input_stride[0] + input_channel * input_stride[1] +
                                                   input_y * input_stride[2] + input_x * input_stride[3]]
                                          )
                                        : 0.0f;
                            }
                        }
                        for (int column = nr; column < spec.row_nr; ++column) packed[column] = 0.0f;
                        packed += spec.row_nr;
                        if (++kernel_x == kernel_width) {
                            kernel_x = 0;
                            if (++kernel_y == kernel_height) {
                                kernel_y = 0;
                                ++input_channel;
                            }
                        }
                    }
                }
            };
            T* output_group =
                output.data() + batch * output_stride[0] + output_channel_start * output_stride[1];
            const T* bias_group =
                bias == nullptr ? nullptr : bias->data() + output_channel_start * bias->stride_(0);
            computePackedConvolutionPanels<T, spec, Dispatcher>(
                output_group, output_channels_per_group, output_spatial, reduction_size, output_stride[1],
                bias_group, bias == nullptr ? 0 : bias->stride_(0), packed_weights.data(), pack_b, operation
            );
        }
    }
    return output;
#endif
}

template <typename T>
yt::YTensor<T, 3> conv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, int stride,
    const std::vector<int>& padding, int dilation, int groups
) {
    return dispatchConv1dWithOptionalBias(
        input, weight, static_cast<const yt::YTensor<T, 1>*>(nullptr), stride,
        validateConvolutionPadding<2>(padding, "yt::function::conv1d"), dilation, groups
    );
}

template <typename T>
yt::YTensor<T, 3> conv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, const yt::YTensor<T, 1>& bias,
    int stride, const std::vector<int>& padding, int dilation, int groups
) {
    return dispatchConv1dWithOptionalBias(
        input, weight, &bias, stride, validateConvolutionPadding<2>(padding, "yt::function::conv1d"),
        dilation, groups
    );
}

template <typename T>
yt::YTensor<T, 4> conv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, std::array<int, 2> stride,
    const std::vector<int>& padding, std::array<int, 2> dilation, int groups
) {
    return dispatchConv2dWithOptionalBias(
        input, weight, static_cast<const yt::YTensor<T, 1>*>(nullptr), stride,
        validateConvolutionPadding<4>(padding, "yt::function::conv2d"), dilation, groups
    );
}

template <typename T>
yt::YTensor<T, 4> conv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, const yt::YTensor<T, 1>& bias,
    std::array<int, 2> stride, const std::vector<int>& padding, std::array<int, 2> dilation, int groups
) {
    return dispatchConv2dWithOptionalBias(
        input, weight, &bias, stride, validateConvolutionPadding<4>(padding, "yt::function::conv2d"),
        dilation, groups
    );
}

}  // namespace yt::function
