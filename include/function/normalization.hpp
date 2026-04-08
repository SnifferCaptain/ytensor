#pragma once

#include <vector>

namespace yt::function {
    // ========== 归一化 ==========

    /// @brief Layer Normalization，沿指定轴做零均值单位方差归一化
    template<typename T, int dim>
    yt::YTensor<T, dim> layerNorm(const yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps = static_cast<T>(1e-5));

    template<typename T, int dim>
    yt::YTensor<T, dim> layerNorm(const yt::YTensor<T, dim>& x, int axis = -1, T eps = static_cast<T>(1e-5));

    template<typename T, int dim>
    yt::YTensor<T, dim>& layerNorm_(yt::YTensor<T, dim>& x, const std::vector<int>& axes, T eps = static_cast<T>(1e-5));

    template<typename T, int dim>
    yt::YTensor<T, dim>& layerNorm_(yt::YTensor<T, dim>& x, int axis = -1, T eps = static_cast<T>(1e-5));

    /// @brief 带gamma和bias的LayerNorm，gamma/bias的shape需与axes对应的维度一致
    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> layerNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        const std::vector<int>& axes,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> layerNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int axis = -1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& layerNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        const std::vector<int>& axes,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& layerNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int axis = -1,
        T eps = static_cast<T>(1e-5)
    );

    /// @brief RMS Normalization，沿指定轴进行RMS归一化并乘以weight
    template<typename T, int dim, int weightDim>
    yt::YTensor<T, dim> rmsNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, weightDim>& weight,
        const std::vector<int>& axes,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int weightDim>
    yt::YTensor<T, dim> rmsNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, weightDim>& weight,
        int axis = -1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int weightDim>
    yt::YTensor<T, dim>& rmsNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, weightDim>& weight,
        const std::vector<int>& axes,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int weightDim>
    yt::YTensor<T, dim>& rmsNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, weightDim>& weight,
        int axis = -1,
        T eps = static_cast<T>(1e-5)
    );

    /// @brief Lp归一化，沿指定轴将向量范数归一到1
    template<typename T, int dim>
    yt::YTensor<T, dim> normalize(
        const yt::YTensor<T, dim>& x,
        const std::vector<int>& axes,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim> normalize(
        const yt::YTensor<T, dim>& x,
        int axis = -1,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim>& normalize_(
        yt::YTensor<T, dim>& x,
        const std::vector<int>& axes,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim>& normalize_(
        yt::YTensor<T, dim>& x,
        int axis = -1,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    /// @brief 带gamma和bias的Lp归一化，gamma/bias的shape需与axes对应的维度一致
    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> normalize(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        const std::vector<int>& axes,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> normalize(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int axis = -1,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& normalize_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        const std::vector<int>& axes,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& normalize_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int axis = -1,
        T p = static_cast<T>(2),
        T eps = static_cast<T>(1e-12)
    );

    /// @brief InstanceNorm，按样本和通道分别对空间轴做归一化
    template<typename T, int dim>
    yt::YTensor<T, dim> instanceNorm(
        const yt::YTensor<T, dim>& x,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim>& instanceNorm_(
        yt::YTensor<T, dim>& x,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> instanceNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& instanceNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    /// @brief 朴素BatchNorm，按通道统计 batch + 空间维度，不包含EMA和推理融合
    template<typename T, int dim>
    yt::YTensor<T, dim> batchNorm(
        const yt::YTensor<T, dim>& x,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim>& batchNorm_(
        yt::YTensor<T, dim>& x,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> batchNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& batchNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    /// @brief GroupNorm，按组对 [group_channels, spatial...] 做归一化，gamma/bias按原始通道广播
    template<typename T, int dim>
    yt::YTensor<T, dim> groupNorm(
        const yt::YTensor<T, dim>& x,
        int groups,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim>
    yt::YTensor<T, dim>& groupNorm_(
        yt::YTensor<T, dim>& x,
        int groups,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim> groupNorm(
        const yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int groups,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );

    template<typename T, int dim, int gammaDim, int biasDim>
    yt::YTensor<T, dim>& groupNorm_(
        yt::YTensor<T, dim>& x,
        const yt::YTensor<T, gammaDim>& gamma,
        const yt::YTensor<T, biasDim>& bias,
        int groups,
        int channelAxis = 1,
        T eps = static_cast<T>(1e-5)
    );
}  // namespace yt::function
