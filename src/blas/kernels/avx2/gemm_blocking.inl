#pragma once
/***************
 * file: blas/kernels/avx2/gemm_blocking.inl
 * purpose: GEMM分块和cache参数调优。
 ***************/

#if defined(__AVX2__) && defined(__FMA__)
namespace yt::blas {

inline int align_down_to(int value, int align) {
    if (align <= 0) return value;
    return std::max(align, (value / align) * align);
}

inline void set_gemm_block_sizes(BlasContext& context, int mc, int kc, int nc) {
    context.mc = align_down_to(std::max(mc, MR_ROW), MR_ROW);
    context.kc = std::max(kc, 32);
    context.nc = align_down_to(std::max(nc, NR_ROW), NR_ROW);
}

inline void set_gemm_block_sizes(int mc, int kc, int nc) {
    set_gemm_block_sizes(defaultBlasContext(), mc, kc, nc);
}

inline void set_gemm_cache_sizes_bytes(
    BlasContext& context, size_t l1_bytes, size_t l2_bytes, size_t l3_bytes
) {
    if (l1_bytes == 0 || l2_bytes == 0 || l3_bytes == 0) return;
    constexpr double l1Use = 0.75;
    constexpr double l2Use = 0.80;
    constexpr double l3Use = 0.75;
    int kcFromL1 = static_cast<int>((l1_bytes * l1Use) / (sizeof(float) * (MR_ROW + NR_ROW)));
    int kcFromL2 = static_cast<int>((l2_bytes * l2Use) / (sizeof(float) * (MR_ROW + NR_ROW + 40)));
    int kc = align_down_to(std::max(std::min(kcFromL1, kcFromL2), 128), 16);
    kc = std::min(kc, 1536);
    int mc = static_cast<int>((l2_bytes * 0.85) / (sizeof(float) * std::max(kc, 1)));
    mc = std::min(align_down_to(std::max(mc - NR_ROW, MR_ROW), MR_ROW), 1024);
    int nc = static_cast<int>((l3_bytes * l3Use) / (sizeof(float) * std::max(kc, 1)));
    nc = std::min(align_down_to(std::max(nc, NR_ROW), NR_ROW), 12288);
    set_gemm_block_sizes(context, mc, kc, nc);
}

inline void set_gemm_cache_sizes_bytes(size_t l1, size_t l2, size_t l3) {
    set_gemm_cache_sizes_bytes(defaultBlasContext(), l1, l2, l3);
}

inline void set_gemm_cache_sizes_kb(BlasContext& context, int l1, int l2, int l3) {
    if (l1 <= 0 || l2 <= 0 || l3 <= 0) return;
    set_gemm_cache_sizes_bytes(
        context, static_cast<size_t>(l1) * 1024, static_cast<size_t>(l2) * 1024,
        static_cast<size_t>(l3) * 1024
    );
}

inline void set_gemm_cache_sizes_kb(int l1, int l2, int l3) {
    set_gemm_cache_sizes_kb(defaultBlasContext(), l1, l2, l3);
}

inline int gemm_mc(const BlasContext& context) { return context.mc; }

inline int gemm_kc(const BlasContext& context) { return context.kc; }

inline int gemm_nc(const BlasContext& context) { return context.nc; }

inline int gemm_mc() { return gemm_mc(defaultBlasContext()); }

inline int gemm_kc() { return gemm_kc(defaultBlasContext()); }

inline int gemm_nc() { return gemm_nc(defaultBlasContext()); }

}  // namespace yt::blas
#endif
