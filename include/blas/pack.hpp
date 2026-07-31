#pragma once
/***************
 * @file pack.hpp
 * @brief BLAS Level-1m packing与数据转换
 ***************/

#include "config.hpp"

#if YT_USE_YBLAS
#include "kernels/avx2/pack.hpp"
#endif
