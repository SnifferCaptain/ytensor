#pragma once

#include <vector>

namespace yt::function {
    // ========== 激活函数 ==========

    template<typename T, int dim>
    yt::YTensor<T, dim> relu(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& relu_(yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim> exp(const yt::YTensor<T, dim>& x, int order = 0);

    /// @brief exp原地版本
    template<typename T, int dim>
    yt::YTensor<T, dim>& exp_(yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim> sigmoid(const yt::YTensor<T, dim>& x, int order = 0);

    /// @brief sigmoid原地版本
    template<typename T, int dim>
    yt::YTensor<T, dim>& sigmoid_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Softmax激活函数，可对多个轴联合归一化
    template<typename T, int dim>
    yt::YTensor<T, dim> softmax(const yt::YTensor<T, dim>& x, const std::vector<int>& axes);

    template<typename T, int dim>
    yt::YTensor<T, dim> softmax(const yt::YTensor<T, dim>& x, int axis = -1);

    template<typename T, int dim>
    yt::YTensor<T, dim>& softmax_(yt::YTensor<T, dim>& x, const std::vector<int>& axes);

    template<typename T, int dim>
    yt::YTensor<T, dim>& softmax_(yt::YTensor<T, dim>& x, int axis = -1);

    /// @brief Leaky ReLU激活函数
    template<typename T, int dim>
    yt::YTensor<T, dim> leakyRelu(const yt::YTensor<T, dim>& x, T alpha = static_cast<T>(0.01), int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& leakyRelu_(yt::YTensor<T, dim>& x, T alpha = static_cast<T>(0.01), int order = 0);

    /// @brief ELU激活函数
    template<typename T, int dim>
    yt::YTensor<T, dim> elu(const yt::YTensor<T, dim>& x, T alpha = static_cast<T>(1.0), int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& elu_(yt::YTensor<T, dim>& x, T alpha = static_cast<T>(1.0), int order = 0);

    /// @brief SELU激活函数
    template<typename T, int dim>
    yt::YTensor<T, dim> selu(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& selu_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief GELU激活函数
    template<typename T, int dim>
    yt::YTensor<T, dim> gelu(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& gelu_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief tanh激活函数
    template<typename T, int dim>
    yt::YTensor<T, dim> tanh(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& tanh_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Swish/SiLU激活函数: x * sigmoid(x)
    template<typename T, int dim>
    yt::YTensor<T, dim> swish(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& swish_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Softplus激活函数: log(1 + exp(x))
    template<typename T, int dim>
    yt::YTensor<T, dim> softplus(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& softplus_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Mish激活函数: x * tanh(softplus(x))
    template<typename T, int dim>
    yt::YTensor<T, dim> mish(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& mish_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Hard Sigmoid激活函数: max(0, min(1, x/6 + 0.5))
    template<typename T, int dim>
    yt::YTensor<T, dim> hardSigmoid(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& hardSigmoid_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief Hard Swish激活函数: x * hardSigmoid(x)
    template<typename T, int dim>
    yt::YTensor<T, dim> hardSwish(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& hardSwish_(yt::YTensor<T, dim>& x, int order = 0);
}  // namespace yt::function
