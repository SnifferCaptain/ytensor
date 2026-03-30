#pragma once
/***************
 * @file gemm_utils.inl
 * @brief GEMM通用工具实现：配置参数、内存工具、AVX2微内核
 ***************/

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// 配置参数实现
// ============================================================================

inline int align_down_to(int value, int align) {
    if (align <= 0) return value;
    return std::max(align, (value / align) * align);
}

inline void set_gemm_block_sizes(int mc, int kc, int nc) {
    g_gemm_mc = align_down_to(std::max(mc, MR_ROW), MR_ROW);
    g_gemm_kc = std::max(kc, 32);
    g_gemm_nc = align_down_to(std::max(nc, NR_ROW), NR_ROW);
}

inline void set_gemm_cache_sizes_bytes(size_t l1_bytes, size_t l2_bytes, size_t l3_bytes) {
    if (l1_bytes == 0 || l2_bytes == 0 || l3_bytes == 0) return;

    constexpr double l1_use = 0.75;
    constexpr double l2_use = 0.80;
    constexpr double l3_use = 0.75;
    constexpr size_t bytes_per_f32 = sizeof(float);

    int kc_from_l1 = static_cast<int>((l1_bytes * l1_use) / (bytes_per_f32 * (MR_ROW + NR_ROW)));
    int kc_from_l2 = static_cast<int>((l2_bytes * l2_use) / (bytes_per_f32 * (MR_ROW + NR_ROW + 40)));

    int kc = std::min(kc_from_l1, kc_from_l2);
    kc = align_down_to(std::max(kc, 128), 16);
    kc = std::min(kc, 1536);

    int mc_from_l2 = static_cast<int>((l2_bytes * 0.85) / (bytes_per_f32 * std::max(kc, 1)));
    mc_from_l2 = align_down_to(std::max(mc_from_l2 - NR_ROW, MR_ROW), MR_ROW);
    mc_from_l2 = std::min(mc_from_l2, 1024);

    int nc_from_l3 = static_cast<int>((l3_bytes * l3_use) / (bytes_per_f32 * std::max(kc, 1)));
    nc_from_l3 = align_down_to(std::max(nc_from_l3, NR_ROW), NR_ROW);
    nc_from_l3 = std::min(nc_from_l3, 12288);

    set_gemm_block_sizes(mc_from_l2, kc, nc_from_l3);
}

inline void set_gemm_cache_sizes_kb(int l1_kb, int l2_kb, int l3_kb) {
    if (l1_kb <= 0 || l2_kb <= 0 || l3_kb <= 0) return;
    set_gemm_cache_sizes_bytes(
        static_cast<size_t>(l1_kb) * 1024ULL,
        static_cast<size_t>(l2_kb) * 1024ULL,
        static_cast<size_t>(l3_kb) * 1024ULL
    );
}

inline int gemm_mc() { return g_gemm_mc; }
inline int gemm_kc() { return g_gemm_kc; }
inline int gemm_nc() { return g_gemm_nc; }

inline int default_num_threads() {
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
#endif
}

inline void set_num_threads(int n) { g_num_threads = std::max(1, n); }
inline int  get_num_threads() { return g_num_threads; }

// ============================================================================
// 内存：64字节对齐缓冲区实现
// ============================================================================

inline void* aligned_alloc_64(size_t size) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, 64);
#else
    if (posix_memalign(&ptr, 64, size) != 0) ptr = nullptr;
#endif
    return ptr;
}

inline void aligned_free_64(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

inline void AlignedBuffer::ensure(size_t n) {
    if (n > capacity) {
        auto* nd = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
        if (nd) {
            if (data) aligned_free_64(data);
            data = nd; capacity = n;
        }
    }
}

// ============================================================================
// 边界mask实现
// ============================================================================

inline void build_masks(__m256i* mask0, __m256i* mask1, int nr) {
    *mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - std::min(nr, 8)]));
    *mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - std::max(0, nr - 8)]));
}

inline float hsum_ps(__m256 v) {
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    lo         = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ============================================================================
// 通用 C 矩阵 load / store 实现
// ============================================================================

inline void load_c_row(__m256 c[][NR_BLOCKS], float* C, int mr, int nr, int ldc) {
    __m256i m0, m1;
    if (nr == NR_ROW) {
        for (int i = 0; i < mr; ++i) {
            for (int jj = 0; jj < NR_BLOCKS; ++jj)
                c[i][jj] = _mm256_loadu_ps(C + i * ldc + jj * 8);
        }
    } else {
        build_masks(&m0, &m1, nr);
        for (int i = 0; i < mr; ++i) {
            c[i][0] = _mm256_maskload_ps(C + i * ldc,     m0);
            if constexpr (NR_BLOCKS > 1)
                c[i][1] = _mm256_maskload_ps(C + i * ldc + 8, m1);
        }
    }
}

inline void store_c_row(const __m256 c[][NR_BLOCKS], float* C, int mr, int nr, int ldc) {
    __m256i m0, m1;
    if (nr == NR_ROW) {
        for (int i = 0; i < mr; ++i)
            for (int jj = 0; jj < NR_BLOCKS; ++jj)
                _mm256_storeu_ps(C + i * ldc + jj * 8, c[i][jj]);
    } else {
        build_masks(&m0, &m1, nr);
        for (int i = 0; i < mr; ++i) {
            _mm256_maskstore_ps(C + i * ldc,     m0, c[i][0]);
            if constexpr (NR_BLOCKS > 1)
                _mm256_maskstore_ps(C + i * ldc + 8, m1, c[i][1]);
        }
    }
}

inline void load_c_col(__m256 c[][2], float* C, int mr, int nr, int ldc) {
    __m256i m0, m1;
    if (mr == MR_COL) {
        for (int j = 0; j < nr; ++j) {
            c[j][0] = _mm256_loadu_ps(C + j * ldc);
            c[j][1] = _mm256_loadu_ps(C + j * ldc + 8);
        }
    } else {
        build_masks(&m0, &m1, mr);
        for (int j = 0; j < nr; ++j) {
            c[j][0] = _mm256_maskload_ps(C + j * ldc,     m0);
            c[j][1] = _mm256_maskload_ps(C + j * ldc + 8, m1);
        }
    }
}

inline void store_c_col(const __m256 c[][2], float* C, int mr, int nr, int ldc) {
    __m256i m0, m1;
    if (mr == MR_COL) {
        for (int j = 0; j < nr; ++j) {
            _mm256_storeu_ps(C + j * ldc,     c[j][0]);
            _mm256_storeu_ps(C + j * ldc + 8, c[j][1]);
        }
    } else {
        build_masks(&m0, &m1, mr);
        for (int j = 0; j < nr; ++j) {
            _mm256_maskstore_ps(C + j * ldc,     m0, c[j][0]);
            _mm256_maskstore_ps(C + j * ldc + 8, m1, c[j][1]);
        }
    }
}

// ============================================================================
// FMA核心实现
// ============================================================================

template<int MR, int NB, int... Is>
__attribute__((always_inline))
inline void fma_row_step(const float* __restrict A, const __m256 b[], __m256 c[][NB],
                         std::integer_sequence<int, Is...>) {
    (([&](){
        __m256 a = _mm256_broadcast_ss(&A[Is]);
        for (int jj = 0; jj < NB; ++jj)
            c[Is][jj] = _mm256_fmadd_ps(a, b[jj], c[Is][jj]);
    }()), ...);
}

template<int NR, int... Js>
__attribute__((always_inline))
inline void fma_col_step(const __m256 a0, const __m256 a1, const float* __restrict B,
                         __m256 c[][2], std::integer_sequence<int, Js...>) {
    (([&](){
        __m256 b = _mm256_broadcast_ss(&B[Js]);
        c[Js][0] = _mm256_fmadd_ps(a0, b, c[Js][0]);
        c[Js][1] = _mm256_fmadd_ps(a1, b, c[Js][1]);
    }()), ...);
}

template<int MR>
__attribute__((hot))
inline void fma_loop_row(const float* __restrict A, const float* __restrict B,
                          __m256 c[MR_ROW][NR_BLOCKS], int kc) {
    static_assert(MR >= 1 && MR <= MR_ROW, "MR must be in [1, MR_ROW]");
    auto seq = std::make_integer_sequence<int, MR>{};
    for (int p = 0; p < kc; ++p) {
        __m256 b[NR_BLOCKS];
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            b[jj] = _mm256_loadu_ps(B + p * NR_ROW + jj * 8);
        fma_row_step<MR, NR_BLOCKS>(A + p * MR_ROW, b, c, seq);
    }
}

template<int NR>
__attribute__((hot))
inline void fma_loop_col(const float* __restrict A, const float* __restrict B,
                          __m256 c[NR_COL][2], int kc) {
    static_assert(NR >= 1 && NR <= NR_COL, "NR must be in [1, NR_COL]");
    auto seq = std::make_integer_sequence<int, NR>{};
    for (int p = 0; p < kc; ++p) {
        __m256 a0 = _mm256_loadu_ps(A + p * MR_COL);
        __m256 a1 = _mm256_loadu_ps(A + p * MR_COL + 8);
        fma_col_step<NR>(a0, a1, B + p * NR_COL, c, seq);
    }
}

// ============================================================================
// 编译期递归 dispatch 实现
// ============================================================================

template<int MAX_MR>
inline void dispatch_fma_row(const float* A, const float* B,
                              __m256 c[MR_ROW][NR_BLOCKS], int mr, int kc) {
    if (mr == MAX_MR) fma_loop_row<MAX_MR>(A, B, c, kc);
    else if constexpr (MAX_MR > 1)
        dispatch_fma_row<MAX_MR - 1>(A, B, c, mr, kc);
}

template<int MAX_NR>
inline void dispatch_fma_col(const float* A, const float* B,
                              __m256 c[NR_COL][2], int nr, int kc) {
    if (nr == MAX_NR) fma_loop_col<MAX_NR>(A, B, c, kc);
    else if constexpr (MAX_NR > 1)
        dispatch_fma_col<MAX_NR - 1>(A, B, c, nr, kc);
}

// ============================================================================
// 微内核公共接口实现
// ============================================================================

__attribute__((always_inline, hot))
inline void kernel_row_full(const float* __restrict A, const float* __restrict B,
                             float* __restrict C, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    fma_loop_row<MR_ROW>(A, B, c, kc);
    for (int i = 0; i < MR_ROW; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            _mm256_storeu_ps(C + i * ldc + jj * 8, c[i][jj]);
}

__attribute__((always_inline, hot))
inline void kernel_row_full_accum(const float* __restrict A, const float* __restrict B,
                                   float* __restrict C, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS];
    for (int i = 0; i < MR_ROW; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            c[i][jj] = _mm256_loadu_ps(C + i * ldc + jj * 8);
    fma_loop_row<MR_ROW>(A, B, c, kc);
    for (int i = 0; i < MR_ROW; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            _mm256_storeu_ps(C + i * ldc + jj * 8, c[i][jj]);
}

inline void kernel_row_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);
    store_c_row(c, C, mr, nr, ldc);
}

inline void kernel_row_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    load_c_row(c, C, mr, nr, ldc);
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);
    store_c_row(c, C, mr, nr, ldc);
}

inline void kernel_row_alphabeta(float* A, float* B, float* C, int mr, int nr, int kc, int ldc,
                                  float alpha, float beta, bool first) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);

    __m256 valpha = _mm256_set1_ps(alpha);
    for (int i = 0; i < mr; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            c[i][jj] = _mm256_mul_ps(c[i][jj], valpha);

    if (first && beta != 0.0f) {
        __m256 vbeta = _mm256_set1_ps(beta);
        __m256 old[MR_ROW][NR_BLOCKS] = {};
        load_c_row(old, C, mr, nr, ldc);
        for (int i = 0; i < mr; ++i)
            for (int jj = 0; jj < NR_BLOCKS; ++jj)
                c[i][jj] = _mm256_fmadd_ps(vbeta, old[i][jj], c[i][jj]);
    } else if (!first) {
        __m256 old[MR_ROW][NR_BLOCKS] = {};
        load_c_row(old, C, mr, nr, ldc);
        for (int i = 0; i < mr; ++i)
            for (int jj = 0; jj < NR_BLOCKS; ++jj)
                c[i][jj] = _mm256_add_ps(c[i][jj], old[i][jj]);
    }
    store_c_row(c, C, mr, nr, ldc);
}

inline void kernel_row_generic_store(float* A, float* B, float* C, int mr, int nr, int kc,
                                      int64_t rsc, int64_t csc, float alpha, float beta, bool first) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);

    alignas(32) float tmp[MR_ROW][NR_ROW];
    for (int i = 0; i < mr; ++i) {
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            _mm256_storeu_ps(&tmp[i][jj * 8], c[i][jj]);
    }
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float* cp = C + i * rsc + j * csc;
            float val = alpha * tmp[i][j];
            if (first) val += beta * (*cp);
            else val += *cp;
            *cp = val;
        }
    }
}

inline void kernel_col_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[NR_COL][2] = {};
    dispatch_fma_col<NR_COL>(A, B, c, nr, kc);
    store_c_col(c, C, mr, nr, ldc);
}

inline void kernel_col_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[NR_COL][2] = {};
    load_c_col(c, C, mr, nr, ldc);
    dispatch_fma_col<NR_COL>(A, B, c, nr, kc);
    store_c_col(c, C, mr, nr, ldc);
}

inline void scatter_c(const __m256 c[][NR_BLOCKS], float* C, int mr, int nr,
                       int64_t rsc, int64_t csc, float alpha, float beta, bool accum) {
    alignas(32) float tmp[MR_ROW][NR_ROW];
    for (int i = 0; i < mr; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            _mm256_storeu_ps(&tmp[i][jj * 8], c[i][jj]);
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float* cp = C + i * rsc + j * csc;
            float val = alpha * tmp[i][j];
            if (accum) val += *cp;
            else if (beta != 0.0f) val += beta * (*cp);
            *cp = val;
        }
    }
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
