#pragma once
/***************
 * @file context.hpp
 * @brief YBLAS运行时调优参数
 ***************/

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yt::blas {

/// @brief YBLAS运行时调优参数
/// @details mc/kc/nc控制GEMM分块，af/df控制Level-1f融合宽度。
/// @note 应通过setter调整分块，保证参数为正且mc/nc与微内核尺寸对齐。
struct BlasContext {
    int mc = 642;
    int kc = 500;
    int nc = 4800;
    int af = 8;
    int df = 4;
};

/// @brief 获取进程级默认YBLAS调优参数
/// @note 返回对象可修改，并发写入不受同步保护。
BlasContext& defaultBlasContext();

/// @brief YBLAS内部OpenMP并行区使用的进程级线程数
/// @note 0表示保持串行；并发读写不受同步保护。
inline int g_num_threads = 0;

/// @brief 查询OpenMP或硬件提供的建议线程数
/// @note 该函数不会修改g_num_threads。
int default_num_threads();

/// @brief 设置进程级YBLAS线程数
/// @note n小于1时按1处理；OpenMP构建还会同步后续并行区的默认线程数。
void set_num_threads(int n);

/// @brief 获取当前进程级BLAS线程数
int get_num_threads();

}  // namespace yt::blas

#include "../../src/blas/context.inl"
