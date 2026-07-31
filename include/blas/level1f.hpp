#pragma once
/***************
 * @file level1f.hpp
 * @brief BLAS Level-1f融合向量操作
 ***************/

#include "context.hpp"
#include "config.hpp"
#include "level1.hpp"

#if YT_USE_YBLAS
#include "kernels/avx2/level1f.hpp"
#endif
