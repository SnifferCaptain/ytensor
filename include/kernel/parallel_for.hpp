#pragma once

#include "../ytensor_infos.hpp"

namespace yt::kernel{
/// @brief 并行for循环，根据任务量自动选择并行或串行执行
/// @param from 起始索引（包含）
/// @param to 结束索引（不包含）
/// @param func 可调用对象，接受一个int参数，表示当前索引
/// @param flop 每次迭代的浮点运算量估计。当问题规模大于minParOps时，开启多核并行执行。以单次浮点运算为单位1。
template<typename Func>
void parallelFor(int from, int to, Func&& func, double flop = 1.){
    if((to - from) * flop >= yt::infos::minParOps) {
        #pragma omp parallel for simd proc_bind(close)
        for (int i = from; i < to; i++) {
            func(i);
        }
    } else {
        #pragma omp simd
        for (int i = from; i < to; i++) {
            func(i);
        }
    }
}

}// namespace yt::kernel