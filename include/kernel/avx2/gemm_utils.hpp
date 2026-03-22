#pragma once
/***************
 * @file gemm_utils.hpp
 * @brief GEMM通用工具：配置参数、内存工具、AVX2微内核（合并自gemm_common + gemm_microkernel）
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * - 命名空间: yt::kernel::avx2
 * - 微内核模板支持任意MR（通过 std::integer_sequence 展开，不再受限于6）
 * - MR_ROW / MR_COL / NR_ROW / NR_COL 可独立调整，dispatch 自动跟进
 * - 行主序微内核：MR_ROW行 × NR_ROW列（NR_ROW 须为8的倍数）
 * - 列主序微内核：MR_COL行 × NR_COL列
 ***************/


#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>   // AVX2 / FMA / F16C intrinsics
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <utility>       // std::integer_sequence / std::make_integer_sequence

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yt::kernel::avx2 {

// ============================================================================
// 配置参数
// ============================================================================

template<int MRRow_, int NRRow_, int MRCol_, int NRCol_>
struct GemmKernelShape {
    static constexpr int MR_ROW = MRRow_;
    static constexpr int NR_ROW = NRRow_;
    static constexpr int MR_COL = MRCol_;
    static constexpr int NR_COL = NRCol_;
    static_assert(NRRow_ % 8 == 0, "NR_ROW must be a multiple of 8 for AVX2 vector lanes");
};

using colMM6_16 = GemmKernelShape<6, 16, 16, 6>;
using DefaultGemmKernelShape = colMM6_16;

// 行主序微内核: MR_ROW行 × NR_ROW列
// AVX2 YMM寄存器数量=16：MR_ROW=6时使用12个累加寄存器+2个B寄存器+1个A广播=15, 安全
constexpr int MR_ROW = DefaultGemmKernelShape::MR_ROW;
constexpr int NR_ROW = DefaultGemmKernelShape::NR_ROW;
constexpr int NR_BLOCKS = NR_ROW / 8;

// 列主序微内核: MR_COL行 × NR_COL列
constexpr int MR_COL = DefaultGemmKernelShape::MR_COL;
constexpr int NR_COL = DefaultGemmKernelShape::NR_COL;

// L1/L2 cache 友好的分块大小（运行时可调）
inline int g_gemm_mc = 642;   // 应为 MR_ROW 的倍数
inline int g_gemm_kc = 500;
inline int g_gemm_nc = 4800;  // 应为 NR_ROW 的倍数

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

// 线程数控制（仅影响本模块GEMM相关调度）
inline int g_num_threads = 1;
inline void set_num_threads(int n) { g_num_threads = std::max(1, n); }
inline int  get_num_threads() { return g_num_threads; }

// ============================================================================
// 内存：64字节对齐缓冲区
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

/// @brief 线程本地可增长的64字节对齐 float 缓冲区
struct AlignedBuffer {
    float* data     = nullptr;
    size_t capacity = 0;

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void ensure(size_t n) {
        if (n > capacity) {
            auto* nd = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
            if (nd) {
                if (data) aligned_free_64(data);
                data = nd; capacity = n;
            }
        }
    }
    ~AlignedBuffer() { if (data) aligned_free_64(data); }
};

// ============================================================================
// 边界mask：用于非满AVX寄存器的 maskstore / maskload
// ============================================================================

alignas(64) inline const int8_t mask_table[32] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

/// @brief 为最后 nr(< 16) 个元素构建两个 AVX 整数掩码
inline void build_masks(__m256i* mask0, __m256i* mask1, int nr) {
    *mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - std::min(nr, 8)]));
    *mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - std::max(0, nr - 8)]));
}

/// @brief 水平求和 __m256
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
// 通用 C 矩阵 load / store（支持任意 MR, NR_BLOCKS 通过 if-else）
// ============================================================================

/// @brief 从行主序 C[mr × nr] 加载到 c[mr][NR_BLOCKS]
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

/// @brief 写回行主序 C[mr × nr]
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

/// @brief 从列主序 C[mr × nr] 加载到 c[nr][NR_BLOCKS_COL=2]
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

/// @brief 写回列主序 C[mr × nr]
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
// 行主序FMA核心：展开A的MR行，B每步加载NR_BLOCKS个AVX向量
// 使用 std::integer_sequence 在编译期展开，支持任意MR（不限于6）
// ============================================================================

/// 单步FMA迭代器：展开 MR 行，使用 index sequence
template<int MR, int NB, int... Is>
__attribute__((always_inline))
inline void fma_row_step(const float* __restrict A, const __m256 b[], __m256 c[][NB],
                         std::integer_sequence<int, Is...>) {
    // 对每行 i：c[i] += broadcast(A[i]) * b[0..NB-1]
    (([&](){
        __m256 a = _mm256_broadcast_ss(&A[Is]);
        for (int jj = 0; jj < NB; ++jj)
            c[Is][jj] = _mm256_fmadd_ps(a, b[jj], c[Is][jj]);
    }()), ...);
}

/// 单步FMA迭代器：展开 NR 列，列主序（B 用 broadcast，A 加载 MR_COL 个）
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

/// @brief 行主序FMA循环，MR行 × (NR_BLOCKS*8列)，A已打包为[kc,MR_ROW]，B已打包为[kc/NR_ROW块,NR_ROW]
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
        // A packed with stride MR_ROW (not MR), to allow masking extra rows with zeros
        fma_row_step<MR, NR_BLOCKS>(A + p * MR_ROW, b, c, seq);
    }
}

/// @brief 列主序FMA循环，NR列 × MR_COL行，A已打包为[kc,MR_COL]，B已打包为[kc,NR_COL]
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
// 编译期递归 dispatch：将运行时的 mr/nr 映射到对应的模板特化
// 支持任意 MAX，只需修改 MR_ROW / NR_COL 常量
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
// 微内核公共接口（与调用方约定的函数签名，替换旧版 kernel_6x16* / kernel_16x6*）
// ============================================================================

/// 行主序：完整 MR_ROW × NR_ROW 分块，零初始化，无需masking
__attribute__((always_inline, hot))
inline void kernel_row_full(const float* __restrict A, const float* __restrict B,
                             float* __restrict C, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    fma_loop_row<MR_ROW>(A, B, c, kc);
    for (int i = 0; i < MR_ROW; ++i)
        for (int jj = 0; jj < NR_BLOCKS; ++jj)
            _mm256_storeu_ps(C + i * ldc + jj * 8, c[i][jj]);
}

/// 行主序：完整分块，加载C累加（非零初始化，kc迭代累加）
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

/// 行主序：边界分块（mr ≤ MR_ROW, nr ≤ NR_ROW），零初始化
inline void kernel_row_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);
    store_c_row(c, C, mr, nr, ldc);
}

/// 行主序：边界分块，加载C累加
inline void kernel_row_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[MR_ROW][NR_BLOCKS] = {};
    load_c_row(c, C, mr, nr, ldc);
    dispatch_fma_row<MR_ROW>(A, B, c, mr, kc);
    store_c_row(c, C, mr, nr, ldc);
}

/// 行主序：带 alpha/beta 缩放的边界分块
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

/// 行主序：通用步幅存储（C有任意stride）
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

/// 列主序：零初始化
inline void kernel_col_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[NR_COL][2] = {};
    dispatch_fma_col<NR_COL>(A, B, c, nr, kc);
    store_c_col(c, C, mr, nr, ldc);
}

/// 列主序：加载累加
inline void kernel_col_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c[NR_COL][2] = {};
    load_c_col(c, C, mr, nr, ldc);
    dispatch_fma_col<NR_COL>(A, B, c, nr, kc);
    store_c_col(c, C, mr, nr, ldc);
}

// ============================================================================
// 向量化 C 写回（通用步幅版）：用于 generic sgemm
// ============================================================================
/// @brief 写回 c[mr][NR_BLOCKS] 到任意步幅的 C，支持 alpha/beta
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
