#pragma once
/***************
 * @file parallel_for.inl
 * @brief 并行for循环实现
 ***************/

namespace yt::kernel {

template<typename Func>
void parallelFor(int from, int to, Func&& func, bool enableParallel) {
    // 忽略"变换失败"的警告
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif
    if(enableParallel) {
        #pragma omp parallel for proc_bind(close)
        for (int i = from; i < to; i++) {
            func(i);
        }
    } else {
        #pragma omp simd
        for (int i = from; i < to; i++) {
            func(i);
        }
    }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
}

} // namespace yt::kernel
