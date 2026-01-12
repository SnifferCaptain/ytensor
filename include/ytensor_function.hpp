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

    inline void throwNotSupport(const std::string& funcName, const std::string& caseDiscription);
}// namespace yt::function
