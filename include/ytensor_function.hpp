//////////////// hpp file "ytensor_function.hpp" //////////////////
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../ytensor.hpp"

#include "function/ops.hpp"
#include "function/activation.hpp"
#include "function/normalization.hpp"
#include "function/loss.hpp"

/**
 * @brief 基于YTensor的常用函数库，对YTensor的常用操作进行封装。
 */
namespace yt::function {
    inline void throwNotSupport(const std::string& funcName, const std::string& caseDiscription);
}  // namespace yt::function
