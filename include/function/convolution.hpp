#pragma once
/***************
 * @file convolution.hpp
 * @brief 一维与二维卷积算子。
 ***************/

#include <array>
#include <vector>

namespace yt::function {

/// @brief 计算NCL布局的一维卷积
/// @details weight使用[C_out,C_in/groups,K]布局，padding按{left,right}补零。
/// @note 计算采用深度学习框架常用的互相关语义，不翻转卷积核。
template <typename T>
yt::YTensor<T, 3> conv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, int stride = 1,
    const std::vector<int>& padding = {0, 0}, int dilation = 1, int groups = 1
);

/// @brief 计算带通道偏置的NCL布局一维卷积
template <typename T>
yt::YTensor<T, 3> conv1d(
    const yt::YTensor<T, 3>& input, const yt::YTensor<T, 3>& weight, const yt::YTensor<T, 1>& bias,
    int stride = 1, const std::vector<int>& padding = {0, 0}, int dilation = 1, int groups = 1
);

/// @brief 计算NCHW布局的二维卷积
/// @details weight使用[C_out,C_in/groups,K_h,K_w]布局，padding按{left,right,top,bottom}补零。
/// @note 计算采用深度学习框架常用的互相关语义，不翻转卷积核。
template <typename T>
yt::YTensor<T, 4> conv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, std::array<int, 2> stride = {1, 1},
    const std::vector<int>& padding = {0, 0, 0, 0}, std::array<int, 2> dilation = {1, 1}, int groups = 1
);

/// @brief 计算带通道偏置的NCHW布局二维卷积
template <typename T>
yt::YTensor<T, 4> conv2d(
    const yt::YTensor<T, 4>& input, const yt::YTensor<T, 4>& weight, const yt::YTensor<T, 1>& bias,
    std::array<int, 2> stride = {1, 1}, const std::vector<int>& padding = {0, 0, 0, 0},
    std::array<int, 2> dilation = {1, 1}, int groups = 1
);

}  // namespace yt::function
