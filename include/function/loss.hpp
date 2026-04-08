#pragma once

namespace yt::function {
    // ========== 损失函数 ==========

    /// @brief 均方误差损失: (input - target)^2
    template<typename T, int dim>
    yt::YTensor<T, dim> mseLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target);

    template<typename T, int dim>
    yt::YTensor<T, dim>& mseLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target);

    /// @brief 交叉熵损失（含softmax）: -log(softmax(input)[target_class])，沿指定轴
    template<typename T, int dim>
    yt::YTensor<T, dim> crossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int axis = -1);

    /// @brief 二元交叉熵损失: -(target*log(input) + (1-target)*log(1-input))
    template<typename T, int dim>
    yt::YTensor<T, dim> binaryCrossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target);

    template<typename T, int dim>
    yt::YTensor<T, dim>& binaryCrossEntropyLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target);

    /// @brief Huber损失（Smooth L1）
    template<typename T, int dim>
    yt::YTensor<T, dim> huberLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta = static_cast<T>(1.0));

    template<typename T, int dim>
    yt::YTensor<T, dim>& huberLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta = static_cast<T>(1.0));
}  // namespace yt::function
