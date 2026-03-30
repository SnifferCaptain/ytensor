#pragma once
/***************
 * @file gemm_utils.hpp
 * @brief GEMM通用工具声明：配置参数、内存工具、AVX2微内核
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
#include <thread>
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

// 线程数控制（仅影响本模块GEMM相关调度）
inline int g_num_threads = 0;  // 延迟初始化

int align_down_to(int value, int align);
void set_gemm_block_sizes(int mc, int kc, int nc);
void set_gemm_cache_sizes_bytes(size_t l1_bytes, size_t l2_bytes, size_t l3_bytes);
void set_gemm_cache_sizes_kb(int l1_kb, int l2_kb, int l3_kb);

int gemm_mc();
int gemm_kc();
int gemm_nc();

int default_num_threads();
void set_num_threads(int n);
int  get_num_threads();

// ============================================================================
// 内存：64字节对齐缓冲区
// ============================================================================

void* aligned_alloc_64(size_t size);
void aligned_free_64(void* ptr);

/// @brief 线程本地可增长的64字节对齐 float 缓冲区
struct AlignedBuffer {
    float* data     = nullptr;
    size_t capacity = 0;

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void ensure(size_t n);
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
void build_masks(__m256i* mask0, __m256i* mask1, int nr);

/// @brief 水平求和 __m256
float hsum_ps(__m256 v);

// ============================================================================
// 通用 C 矩阵 load / store（支持任意 MR, NR_BLOCKS 通过 if-else）
// ============================================================================

/// @brief 从行主序 C[mr × nr] 加载到 c[mr][NR_BLOCKS]
void load_c_row(__m256 c[][NR_BLOCKS], float* C, int mr, int nr, int ldc);

/// @brief 写回行主序 C[mr × nr]
void store_c_row(const __m256 c[][NR_BLOCKS], float* C, int mr, int nr, int ldc);

/// @brief 从列主序 C[mr × nr] 加载到 c[nr][NR_BLOCKS_COL=2]
void load_c_col(__m256 c[][2], float* C, int mr, int nr, int ldc);

/// @brief 写回列主序 C[mr × nr]
void store_c_col(const __m256 c[][2], float* C, int mr, int nr, int ldc);

// ============================================================================
// 行主序FMA核心：展开A的MR行，B每步加载NR_BLOCKS个AVX向量
// 使用 std::integer_sequence 在编译期展开，支持任意MR（不限于6）
// ============================================================================

/// 单步FMA迭代器：展开 MR 行，使用 index sequence
template<int MR, int NB, int... Is>
__attribute__((always_inline))
void fma_row_step(const float* __restrict A, const __m256 b[], __m256 c[][NB],
                  std::integer_sequence<int, Is...>);

/// 单步FMA迭代器：展开 NR 列，列主序（B 用 broadcast，A 加载 MR_COL 个）
template<int NR, int... Js>
__attribute__((always_inline))
void fma_col_step(const __m256 a0, const __m256 a1, const float* __restrict B,
                  __m256 c[][2], std::integer_sequence<int, Js...>);

/// @brief 行主序FMA循环，MR行 × (NR_BLOCKS*8列)，A已打包为[kc,MR_ROW]，B已打包为[kc/NR_ROW块,NR_ROW]
template<int MR>
__attribute__((hot))
void fma_loop_row(const float* __restrict A, const float* __restrict B,
                  __m256 c[MR_ROW][NR_BLOCKS], int kc);

/// @brief 列主序FMA循环，NR列 × MR_COL行，A已打包为[kc,MR_COL]，B已打包为[kc,NR_COL]
template<int NR>
__attribute__((hot))
void fma_loop_col(const float* __restrict A, const float* __restrict B,
                  __m256 c[NR_COL][2], int kc);

// ============================================================================
// 编译期递归 dispatch：将运行时的 mr/nr 映射到对应的模板特化
// 支持任意 MAX，只需修改 MR_ROW / NR_COL 常量
// ============================================================================

template<int MAX_MR>
void dispatch_fma_row(const float* A, const float* B,
                      __m256 c[MR_ROW][NR_BLOCKS], int mr, int kc);

template<int MAX_NR>
void dispatch_fma_col(const float* A, const float* B,
                      __m256 c[NR_COL][2], int nr, int kc);

// ============================================================================
// 微内核公共接口（与调用方约定的函数签名，替换旧版 kernel_6x16* / kernel_16x6*）
// ============================================================================

/// 行主序：完整 MR_ROW × NR_ROW 分块，零初始化，无需masking
__attribute__((always_inline, hot))
void kernel_row_full(const float* __restrict A, const float* __restrict B,
                     float* __restrict C, int kc, int ldc);

/// 行主序：完整分块，加载C累加（非零初始化，kc迭代累加）
__attribute__((always_inline, hot))
void kernel_row_full_accum(const float* __restrict A, const float* __restrict B,
                           float* __restrict C, int kc, int ldc);

/// 行主序：边界分块（mr ≤ MR_ROW, nr ≤ NR_ROW），零初始化
void kernel_row_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc);

/// 行主序：边界分块，加载C累加
void kernel_row_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc);

/// 行主序：带 alpha/beta 缩放的边界分块
void kernel_row_alphabeta(float* A, float* B, float* C, int mr, int nr, int kc, int ldc,
                          float alpha, float beta, bool first);

/// 行主序：通用步幅存储（C有任意stride）
void kernel_row_generic_store(float* A, float* B, float* C, int mr, int nr, int kc,
                              int64_t rsc, int64_t csc, float alpha, float beta, bool first);

/// 列主序：零初始化
void kernel_col_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc);

/// 列主序：加载累加
void kernel_col_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc);

// ============================================================================
// 向量化 C 写回（通用步幅版）：用于 generic sgemm
// ============================================================================
/// @brief 写回 c[mr][NR_BLOCKS] 到任意步幅的 C，支持 alpha/beta
void scatter_c(const __m256 c[][NR_BLOCKS], float* C, int mr, int nr,
               int64_t rsc, int64_t csc, float alpha, float beta, bool accum);

} // namespace yt::kernel::avx2

#include "../../../src/kernel/avx2/gemm_utils.inl"

#endif // __AVX2__ && __FMA__
