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
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> matmul(const YTensor<T, dim0>& a, const YTensor<T, dim1>& b);

    template<typename T, int dim>
    YTensor<T, dim> relu(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim>& relu_(YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> exp(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> sigmoid(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> softmax(const YTensor<T, dim>& x, int axis = -1);

    template<typename T, int dim>
    YTensor<T, dim>& softmax_(YTensor<T, dim>& x, int axis = -1);

    enum struct sdpaBackend{
        MATH
    };

    template<typename T, int dim>
    YTensor<T, dim> scaledDotProductAttention(
        YTensor<T, dim>& query,
        YTensor<T, dim>& key,
        YTensor<T, dim>& value,
        T scale = static_cast<T>(0.0),
        YTensor<T, 2>* mask = nullptr,
        sdpaBackend backend = sdpaBackend::MATH
    );

    inline void throwNotSupport(const std::string& funcName, const std::string& caseDiscription);
}// namespace yt::function

#include "../src/ytensor_function.inl"