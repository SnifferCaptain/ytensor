#pragma once
/***************
 * file: blas/context.inl
 * purpose: BLAS运行时调优参数与线程设置实现。
 ***************/

namespace yt::blas {

inline BlasContext& defaultBlasContext() {
    static BlasContext context;
    return context;
}

inline int default_num_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
#endif
}

inline void set_num_threads(int n) {
    g_num_threads = std::max(1, n);
#ifdef _OPENMP
    omp_set_num_threads(g_num_threads);
#endif
}

inline int get_num_threads() { return g_num_threads; }

}  // namespace yt::blas
