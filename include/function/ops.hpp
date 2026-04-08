#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

namespace yt::function {
    // ========== 通用算子 / 层 ==========

    template<typename T, int dim0, int dim1>
    yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> matmul(const yt::YTensor<T, dim0>& a, const yt::YTensor<T, dim1>& b);

    /// @brief 线性层，权重形状遵循PyTorch格式 [out_features, in_features]
    template<typename T, int dim>
    yt::YTensor<T, dim> linear(const yt::YTensor<T, dim>& x, const yt::YTensor<T, 2>& weight);

    /// @brief 带偏置的线性层，bias形状为 [out_features]
    template<typename T, int dim>
    yt::YTensor<T, dim> linear(const yt::YTensor<T, dim>& x, const yt::YTensor<T, 2>& weight, const yt::YTensor<T, 1>& bias);

    enum struct sdpaBackend {
        MATH,
        FLASH_AVX2
    };

    template<typename T, int dim>
    yt::YTensor<T, dim> scaledDotProductAttention(
        yt::YTensor<T, dim>& query,
        yt::YTensor<T, dim>& key,
        yt::YTensor<T, dim>& value,
        T scale = static_cast<T>(0.0),
        yt::YTensor<bool, 2>* mask = nullptr,
        yt::YTensor<T, 2>* bias = nullptr,
        sdpaBackend backend = sdpaBackend::MATH
    );

    template<typename T, int dim>
    yt::YTensor<T, dim> scaledDotProductAttention(
        yt::YTensor<T, dim>& query,
        yt::YTensor<T, dim>& key,
        yt::YTensor<T, dim>& value,
        T scale,
        yt::YTensor<bool, 2>* mask,
        std::nullptr_t,
        sdpaBackend backend = sdpaBackend::MATH
    );

    template<typename T, int dim, typename MaskFunc>
    requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
    yt::YTensor<T, dim> scaledDotProductAttention(
        yt::YTensor<T, dim>& query,
        yt::YTensor<T, dim>& key,
        yt::YTensor<T, dim>& value,
        T scale,
        MaskFunc&& mask,
        yt::YTensor<T, 2>* bias = nullptr,
        sdpaBackend backend = sdpaBackend::MATH
    );

    template<typename T, int dim, typename MaskFunc>
    requires (!yt::traits::is_ytensor_v<std::decay_t<MaskFunc>> && !std::is_pointer_v<std::decay_t<MaskFunc>>)
    yt::YTensor<T, dim> scaledDotProductAttention(
        yt::YTensor<T, dim>& query,
        yt::YTensor<T, dim>& key,
        yt::YTensor<T, dim>& value,
        T scale,
        MaskFunc&& mask,
        std::nullptr_t,
        sdpaBackend backend = sdpaBackend::MATH
    );

    // ========== 融合算子 ==========

    /// @brief LogSumExp融合算子，沿指定轴计算log(sum(exp(x)))
    template<typename T, int dim>
    yt::YTensor<T, dim> logsumexp(const yt::YTensor<T, dim>& x, const std::vector<int>& axes);

    template<typename T, int dim>
    yt::YTensor<T, dim> logsumexp(const yt::YTensor<T, dim>& x, int axis = -1);

    /// @brief LogSoftmax: x - logsumexp(x, axis/axes)
    template<typename T, int dim>
    yt::YTensor<T, dim> logSoftmax(const yt::YTensor<T, dim>& x, const std::vector<int>& axes);

    template<typename T, int dim>
    yt::YTensor<T, dim> logSoftmax(const yt::YTensor<T, dim>& x, int axis = -1);

    template<typename T, int dim>
    yt::YTensor<T, dim>& logSoftmax_(yt::YTensor<T, dim>& x, const std::vector<int>& axes);

    template<typename T, int dim>
    yt::YTensor<T, dim>& logSoftmax_(yt::YTensor<T, dim>& x, int axis = -1);

    // ========== 池化 ==========

    /// @brief 1D最大池化，沿指定轴进行
    template<typename T, int dim>
    yt::YTensor<T, dim> maxPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride = -1, int axis = -1);

    /// @brief 1D平均池化，沿指定轴进行
    template<typename T, int dim>
    yt::YTensor<T, dim> avgPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride = -1, int axis = -1);
}  // namespace yt::function
