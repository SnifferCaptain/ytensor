//////////////// hpp file "ytensor_math.hpp" //////////////////
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include "../ytensor.hpp"

/**
 * @brief 基于YTensor的常用函数库，对YTensor的常用操作进行封装。
 */
namespace yt::function{
    template<typename T, int dim0, int dim1>
    yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> matmul(const yt::YTensor<T, dim0>& a, const yt::YTensor<T, dim1>& b);

    template<typename T, int dim>
    yt::YTensor<T, dim> relu(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& relu_(yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim> exp(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim> sigmoid(const yt::YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim> softmax(const yt::YTensor<T, dim>& x, int axis = -1);

    template<typename T, int dim>
    yt::YTensor<T, dim>& softmax_(yt::YTensor<T, dim>& x, int axis = -1);

    enum struct sdpaBackend{
        MATH
    };

    template<typename T, int dim>
    yt::YTensor<T, dim> scaledDotProductAttention(
        yt::YTensor<T, dim>& query,
        yt::YTensor<T, dim>& key,
        yt::YTensor<T, dim>& value,
        T scale = static_cast<T>(0.0),
        yt::YTensor<T, 2>* mask = nullptr,
        sdpaBackend backend = sdpaBackend::MATH
    );

    // ========== 激活函数 ==========

    /// @brief exp原地版本
    template<typename T, int dim>
    yt::YTensor<T, dim>& exp_(yt::YTensor<T, dim>& x, int order = 0);

    /// @brief sigmoid原地版本
    template<typename T, int dim>
    yt::YTensor<T, dim>& sigmoid_(yt::YTensor<T, dim>& x, int order = 0);

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

    // ========== 融合算子 ==========

    /// @brief LogSumExp融合算子，沿指定轴计算log(sum(exp(x)))
    template<typename T, int dim>
    yt::YTensor<T, dim> logsumexp(const yt::YTensor<T, dim>& x, int axis = -1);

    /// @brief LogSoftmax: x - logsumexp(x, axis)
    template<typename T, int dim>
    yt::YTensor<T, dim> logSoftmax(const yt::YTensor<T, dim>& x, int axis = -1, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& logSoftmax_(yt::YTensor<T, dim>& x, int axis = -1, int order = 0);

    // ========== 归一化 ==========

    /// @brief Layer Normalization，沿指定轴进行归一化
    template<typename T, int dim>
    yt::YTensor<T, dim> layerNorm(const yt::YTensor<T, dim>& x, int axis = -1, T eps = static_cast<T>(1e-5), int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& layerNorm_(yt::YTensor<T, dim>& x, int axis = -1, T eps = static_cast<T>(1e-5), int order = 0);

    // ========== 池化 ==========

    /// @brief 1D最大池化，沿指定轴进行
    template<typename T, int dim>
    yt::YTensor<T, dim> maxPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride = -1, int axis = -1, int order = 0);

    /// @brief 1D平均池化，沿指定轴进行
    template<typename T, int dim>
    yt::YTensor<T, dim> avgPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride = -1, int axis = -1, int order = 0);

    // ========== 损失函数 ==========

    /// @brief 均方误差损失: (input - target)^2
    template<typename T, int dim>
    yt::YTensor<T, dim> mseLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& mseLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order = 0);

    /// @brief 交叉熵损失（含softmax）: -log(softmax(input)[target_class])，沿指定轴
    template<typename T, int dim>
    yt::YTensor<T, dim> crossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int axis = -1, int order = 0);

    /// @brief 二元交叉熵损失: -(target*log(input) + (1-target)*log(1-input))
    template<typename T, int dim>
    yt::YTensor<T, dim> binaryCrossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& binaryCrossEntropyLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order = 0);

    /// @brief Huber损失（Smooth L1）
    template<typename T, int dim>
    yt::YTensor<T, dim> huberLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta = static_cast<T>(1.0), int order = 0);

    template<typename T, int dim>
    yt::YTensor<T, dim>& huberLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta = static_cast<T>(1.0), int order = 0);

    inline void throwNotSupport(const std::string& funcName, const std::string& caseDiscription);
}// namespace yt::function
