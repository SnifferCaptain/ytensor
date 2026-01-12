#pragma once
/***************
 * @file gemm.hpp
 * @brief 高性能GEMM矩阵乘法实现，支持自动布局检测
 * @author SnifferCaptain
 * @date 2025-12-31
 * 
 * BLAS风格GEMM: C = alpha * A @ B + beta * C
 * 自动检测行主序/列主序并选择最优内核
 * 基于: https://salykova.github.io/matmul-cpu
 * 
 * 功能特性:
 * - 双微内核: 16x6列主序 和 6x16行主序
 * - 自动布局检测和内核选择
 * - OpenMP并行支持，通过set_num_threads配置线程数
 * - 针对点积和外积的特化内核
 * - 支持任意步幅
 ***************/

#include "../ytensor_infos.hpp"

#if YT_USE_AVX2

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yt::kernel::gemm {

// ============================================================================
// 配置参数
// ============================================================================

// 行主序微内核: 6行 x 16列
constexpr int MR_ROW = 6;
constexpr int NR_ROW = 16;

// 列主序微内核: 16行 x 6列
constexpr int MR_COL = 16;
constexpr int NR_COL = 6;

// 分块大小，必须是微内核尺寸的整数倍
// GEMM_MC 必须能被 MR_ROW=6 整除: 642 = 107 * 6
// GEMM_NC 必须能被 NR_ROW=16 整除: 4800 = 300 * 16
// 缓存考虑: GEMM_MC*GEMM_KC*4 应该放入L2缓存约512KB
// 当前配置 642*500*4 = 1.28MB > 512KB，可能造成缓存压力
// 使用 GEMM_ 前缀避免与 matmul_single.hpp 宏冲突
constexpr int GEMM_MC = 642;
constexpr int GEMM_KC = 500;
constexpr int GEMM_NC = 4800;

// GEMM操作的线程配置
// 注意: 此值仅配置GEMM操作的线程数
// 实际线程数通过OpenMP的num_threads子句指定
inline int g_num_threads = 1;

inline void set_num_threads(int n) {
    g_num_threads = std::max(1, n);
}

inline int get_num_threads() {
    return g_num_threads;
}

// ============================================================================
// 内存对齐
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

struct AlignedBuffer {
    float* data = nullptr;
    size_t capacity = 0;
    
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    void ensure(size_t n) {
        if (n > capacity) {
            float* new_data = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
            if (new_data) {
                if (data) aligned_free_64(data);
                data = new_data;
                capacity = n;
            }
        }
    }
    
    ~AlignedBuffer() { 
        if (data) aligned_free_64(data); 
    }
};

// 边界处理的掩码表
alignas(64) static const int8_t mask_table[32] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

inline void build_masks(__m256i* mask0, __m256i* mask1, int nr) {
    *mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - nr]));
    *mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - nr + 8]));
}

// ============================================================================
// 列主序内核 16x6 - 源自原版 sgemm.c
// 针对列主序布局高度优化的内核
// ============================================================================

// 列主序的FMA循环，处理不同的nr值
inline void fma_loop_col_1(float* A, float* B, __m256* c00, __m256* c01, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);
        *c00 = _mm256_fmadd_ps(a0, b, *c00);
        *c01 = _mm256_fmadd_ps(a1, b, *c01);
        A += 16; B += 6;
    }
}

inline void fma_loop_col_2(float* A, float* B, __m256* c00, __m256* c01, __m256* c10, __m256* c11, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);
        *c00 = _mm256_fmadd_ps(a0, b, *c00);
        *c01 = _mm256_fmadd_ps(a1, b, *c01);
        b = _mm256_broadcast_ss(B + 1);
        *c10 = _mm256_fmadd_ps(a0, b, *c10);
        *c11 = _mm256_fmadd_ps(a1, b, *c11);
        A += 16; B += 6;
    }
}

inline void fma_loop_col_3(float* A, float* B, __m256* c00, __m256* c01, __m256* c10, __m256* c11,
                           __m256* c20, __m256* c21, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);     *c00 = _mm256_fmadd_ps(a0, b, *c00); *c01 = _mm256_fmadd_ps(a1, b, *c01);
        b = _mm256_broadcast_ss(B + 1); *c10 = _mm256_fmadd_ps(a0, b, *c10); *c11 = _mm256_fmadd_ps(a1, b, *c11);
        b = _mm256_broadcast_ss(B + 2); *c20 = _mm256_fmadd_ps(a0, b, *c20); *c21 = _mm256_fmadd_ps(a1, b, *c21);
        A += 16; B += 6;
    }
}

inline void fma_loop_col_4(float* A, float* B, __m256* c00, __m256* c01, __m256* c10, __m256* c11,
                           __m256* c20, __m256* c21, __m256* c30, __m256* c31, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);     *c00 = _mm256_fmadd_ps(a0, b, *c00); *c01 = _mm256_fmadd_ps(a1, b, *c01);
        b = _mm256_broadcast_ss(B + 1); *c10 = _mm256_fmadd_ps(a0, b, *c10); *c11 = _mm256_fmadd_ps(a1, b, *c11);
        b = _mm256_broadcast_ss(B + 2); *c20 = _mm256_fmadd_ps(a0, b, *c20); *c21 = _mm256_fmadd_ps(a1, b, *c21);
        b = _mm256_broadcast_ss(B + 3); *c30 = _mm256_fmadd_ps(a0, b, *c30); *c31 = _mm256_fmadd_ps(a1, b, *c31);
        A += 16; B += 6;
    }
}

inline void fma_loop_col_5(float* A, float* B, __m256* c00, __m256* c01, __m256* c10, __m256* c11,
                           __m256* c20, __m256* c21, __m256* c30, __m256* c31,
                           __m256* c40, __m256* c41, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);     *c00 = _mm256_fmadd_ps(a0, b, *c00); *c01 = _mm256_fmadd_ps(a1, b, *c01);
        b = _mm256_broadcast_ss(B + 1); *c10 = _mm256_fmadd_ps(a0, b, *c10); *c11 = _mm256_fmadd_ps(a1, b, *c11);
        b = _mm256_broadcast_ss(B + 2); *c20 = _mm256_fmadd_ps(a0, b, *c20); *c21 = _mm256_fmadd_ps(a1, b, *c21);
        b = _mm256_broadcast_ss(B + 3); *c30 = _mm256_fmadd_ps(a0, b, *c30); *c31 = _mm256_fmadd_ps(a1, b, *c31);
        b = _mm256_broadcast_ss(B + 4); *c40 = _mm256_fmadd_ps(a0, b, *c40); *c41 = _mm256_fmadd_ps(a1, b, *c41);
        A += 16; B += 6;
    }
}

inline void fma_loop_col_6(float* A, float* B, __m256* c00, __m256* c01, __m256* c10, __m256* c11,
                           __m256* c20, __m256* c21, __m256* c30, __m256* c31,
                           __m256* c40, __m256* c41, __m256* c50, __m256* c51, int kc) {
    __m256 a0, a1, b;
    for (int p = 0; p < kc; ++p) {
        a0 = _mm256_loadu_ps(A);
        a1 = _mm256_loadu_ps(A + 8);
        b = _mm256_broadcast_ss(B);     *c00 = _mm256_fmadd_ps(a0, b, *c00); *c01 = _mm256_fmadd_ps(a1, b, *c01);
        b = _mm256_broadcast_ss(B + 1); *c10 = _mm256_fmadd_ps(a0, b, *c10); *c11 = _mm256_fmadd_ps(a1, b, *c11);
        b = _mm256_broadcast_ss(B + 2); *c20 = _mm256_fmadd_ps(a0, b, *c20); *c21 = _mm256_fmadd_ps(a1, b, *c21);
        b = _mm256_broadcast_ss(B + 3); *c30 = _mm256_fmadd_ps(a0, b, *c30); *c31 = _mm256_fmadd_ps(a1, b, *c31);
        b = _mm256_broadcast_ss(B + 4); *c40 = _mm256_fmadd_ps(a0, b, *c40); *c41 = _mm256_fmadd_ps(a1, b, *c41);
        b = _mm256_broadcast_ss(B + 5); *c50 = _mm256_fmadd_ps(a0, b, *c50); *c51 = _mm256_fmadd_ps(a1, b, *c51);
        A += 16; B += 6;
    }
}

// 列主序 16x6 内核，零初始化
inline void kernel_16x6_col_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    switch (nr) {
        case 1: fma_loop_col_1(A, B, &c00, &c01, kc); break;
        case 2: fma_loop_col_2(A, B, &c00, &c01, &c10, &c11, kc); break;
        case 3: fma_loop_col_3(A, B, &c00, &c01, &c10, &c11, &c20, &c21, kc); break;
        case 4: fma_loop_col_4(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, kc); break;
        case 5: fma_loop_col_5(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, &c40, &c41, kc); break;
        case 6: fma_loop_col_6(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, &c40, &c41, &c50, &c51, kc); break;
    }
    
    // 存储结果，列主序C每列有ldc行
    if (mr == 16) {
        if (nr >= 1) { _mm256_storeu_ps(C, c00); _mm256_storeu_ps(C+8, c01); }
        if (nr >= 2) { _mm256_storeu_ps(C+ldc, c10); _mm256_storeu_ps(C+ldc+8, c11); }
        if (nr >= 3) { _mm256_storeu_ps(C+2*ldc, c20); _mm256_storeu_ps(C+2*ldc+8, c21); }
        if (nr >= 4) { _mm256_storeu_ps(C+3*ldc, c30); _mm256_storeu_ps(C+3*ldc+8, c31); }
        if (nr >= 5) { _mm256_storeu_ps(C+4*ldc, c40); _mm256_storeu_ps(C+4*ldc+8, c41); }
        if (nr >= 6) { _mm256_storeu_ps(C+5*ldc, c50); _mm256_storeu_ps(C+5*ldc+8, c51); }
    } else {
        build_masks(&mask0, &mask1, mr);
        if (nr >= 1) { _mm256_maskstore_ps(C, mask0, c00); _mm256_maskstore_ps(C+8, mask1, c01); }
        if (nr >= 2) { _mm256_maskstore_ps(C+ldc, mask0, c10); _mm256_maskstore_ps(C+ldc+8, mask1, c11); }
        if (nr >= 3) { _mm256_maskstore_ps(C+2*ldc, mask0, c20); _mm256_maskstore_ps(C+2*ldc+8, mask1, c21); }
        if (nr >= 4) { _mm256_maskstore_ps(C+3*ldc, mask0, c30); _mm256_maskstore_ps(C+3*ldc+8, mask1, c31); }
        if (nr >= 5) { _mm256_maskstore_ps(C+4*ldc, mask0, c40); _mm256_maskstore_ps(C+4*ldc+8, mask1, c41); }
        if (nr >= 6) { _mm256_maskstore_ps(C+5*ldc, mask0, c50); _mm256_maskstore_ps(C+5*ldc+8, mask1, c51); }
    }
}

// 列主序 16x6 内核，加载累加
inline void kernel_16x6_col_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    // 加载已有的C值
    if (mr == 16) {
        if (nr >= 1) { c00 = _mm256_loadu_ps(C); c01 = _mm256_loadu_ps(C+8); }
        if (nr >= 2) { c10 = _mm256_loadu_ps(C+ldc); c11 = _mm256_loadu_ps(C+ldc+8); }
        if (nr >= 3) { c20 = _mm256_loadu_ps(C+2*ldc); c21 = _mm256_loadu_ps(C+2*ldc+8); }
        if (nr >= 4) { c30 = _mm256_loadu_ps(C+3*ldc); c31 = _mm256_loadu_ps(C+3*ldc+8); }
        if (nr >= 5) { c40 = _mm256_loadu_ps(C+4*ldc); c41 = _mm256_loadu_ps(C+4*ldc+8); }
        if (nr >= 6) { c50 = _mm256_loadu_ps(C+5*ldc); c51 = _mm256_loadu_ps(C+5*ldc+8); }
    } else {
        build_masks(&mask0, &mask1, mr);
        if (nr >= 1) { c00 = _mm256_maskload_ps(C, mask0); c01 = _mm256_maskload_ps(C+8, mask1); }
        if (nr >= 2) { c10 = _mm256_maskload_ps(C+ldc, mask0); c11 = _mm256_maskload_ps(C+ldc+8, mask1); }
        if (nr >= 3) { c20 = _mm256_maskload_ps(C+2*ldc, mask0); c21 = _mm256_maskload_ps(C+2*ldc+8, mask1); }
        if (nr >= 4) { c30 = _mm256_maskload_ps(C+3*ldc, mask0); c31 = _mm256_maskload_ps(C+3*ldc+8, mask1); }
        if (nr >= 5) { c40 = _mm256_maskload_ps(C+4*ldc, mask0); c41 = _mm256_maskload_ps(C+4*ldc+8, mask1); }
        if (nr >= 6) { c50 = _mm256_maskload_ps(C+5*ldc, mask0); c51 = _mm256_maskload_ps(C+5*ldc+8, mask1); }
    }
    
    switch (nr) {
        case 1: fma_loop_col_1(A, B, &c00, &c01, kc); break;
        case 2: fma_loop_col_2(A, B, &c00, &c01, &c10, &c11, kc); break;
        case 3: fma_loop_col_3(A, B, &c00, &c01, &c10, &c11, &c20, &c21, kc); break;
        case 4: fma_loop_col_4(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, kc); break;
        case 5: fma_loop_col_5(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, &c40, &c41, kc); break;
        case 6: fma_loop_col_6(A, B, &c00, &c01, &c10, &c11, &c20, &c21, &c30, &c31, &c40, &c41, &c50, &c51, kc); break;
    }
    
    // 存储结果
    if (mr == 16) {
        if (nr >= 1) { _mm256_storeu_ps(C, c00); _mm256_storeu_ps(C+8, c01); }
        if (nr >= 2) { _mm256_storeu_ps(C+ldc, c10); _mm256_storeu_ps(C+ldc+8, c11); }
        if (nr >= 3) { _mm256_storeu_ps(C+2*ldc, c20); _mm256_storeu_ps(C+2*ldc+8, c21); }
        if (nr >= 4) { _mm256_storeu_ps(C+3*ldc, c30); _mm256_storeu_ps(C+3*ldc+8, c31); }
        if (nr >= 5) { _mm256_storeu_ps(C+4*ldc, c40); _mm256_storeu_ps(C+4*ldc+8, c41); }
        if (nr >= 6) { _mm256_storeu_ps(C+5*ldc, c50); _mm256_storeu_ps(C+5*ldc+8, c51); }
    } else {
        if (nr >= 1) { _mm256_maskstore_ps(C, mask0, c00); _mm256_maskstore_ps(C+8, mask1, c01); }
        if (nr >= 2) { _mm256_maskstore_ps(C+ldc, mask0, c10); _mm256_maskstore_ps(C+ldc+8, mask1, c11); }
        if (nr >= 3) { _mm256_maskstore_ps(C+2*ldc, mask0, c20); _mm256_maskstore_ps(C+2*ldc+8, mask1, c21); }
        if (nr >= 4) { _mm256_maskstore_ps(C+3*ldc, mask0, c30); _mm256_maskstore_ps(C+3*ldc+8, mask1, c31); }
        if (nr >= 5) { _mm256_maskstore_ps(C+4*ldc, mask0, c40); _mm256_maskstore_ps(C+4*ldc+8, mask1, c41); }
        if (nr >= 6) { _mm256_maskstore_ps(C+5*ldc, mask0, c50); _mm256_maskstore_ps(C+5*ldc+8, mask1, c51); }
    }
}

// ============================================================================
// 行主序内核 6x16 - 列主序的转置版本
// ============================================================================

// 完整 6x16 分块的快速路径内核，无边界检查
// 此函数对性能至关重要，确保编译器充分优化
__attribute__((always_inline, hot))
inline void kernel_6x16_row_full(const float* __restrict A, const float* __restrict B, 
                                  float* __restrict C, int kc, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(&A[3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(&A[4]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(&A[5]); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        A += 6; B += 16;
    }
    
    // 存储结果
    _mm256_storeu_ps(C,        c00); _mm256_storeu_ps(C + 8,        c01);
    _mm256_storeu_ps(C + ldc,  c10); _mm256_storeu_ps(C + ldc + 8,  c11);
    _mm256_storeu_ps(C + 2*ldc, c20); _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc, c30); _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc, c40); _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc, c50); _mm256_storeu_ps(C + 5*ldc + 8, c51);
}

// 完整 6x16 分块的加载累加快速路径内核
__attribute__((always_inline, hot))
inline void kernel_6x16_row_full_accum(const float* __restrict A, const float* __restrict B,
                                        float* __restrict C, int kc, int ldc) {
    __m256 c00 = _mm256_loadu_ps(C);
    __m256 c01 = _mm256_loadu_ps(C + 8);
    __m256 c10 = _mm256_loadu_ps(C + ldc);
    __m256 c11 = _mm256_loadu_ps(C + ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c21 = _mm256_loadu_ps(C + 2*ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3*ldc);
    __m256 c31 = _mm256_loadu_ps(C + 3*ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4*ldc);
    __m256 c41 = _mm256_loadu_ps(C + 4*ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5*ldc);
    __m256 c51 = _mm256_loadu_ps(C + 5*ldc + 8);
    
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(&A[3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(&A[4]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(&A[5]); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        A += 6; B += 16;
    }
    
    _mm256_storeu_ps(C,        c00); _mm256_storeu_ps(C + 8,        c01);
    _mm256_storeu_ps(C + ldc,  c10); _mm256_storeu_ps(C + ldc + 8,  c11);
    _mm256_storeu_ps(C + 2*ldc, c20); _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc, c30); _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc, c40); _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc, c50); _mm256_storeu_ps(C + 5*ldc + 8, c51);
}

// 针对不同行数优化的FMA循环
inline void fma_loop_row_1x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a = _mm256_broadcast_ss(&A[0]);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        A += 6; B += 16;
    }
}

inline void fma_loop_row_2x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, __m256& c10, __m256& c11, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        A += 6; B += 16;
    }
}

inline void fma_loop_row_3x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, __m256& c10, __m256& c11,
                              __m256& c20, __m256& c21, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        A += 6; B += 16;
    }
}

inline void fma_loop_row_4x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, __m256& c10, __m256& c11,
                              __m256& c20, __m256& c21, __m256& c30, __m256& c31, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(&A[3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        A += 6; B += 16;
    }
}

inline void fma_loop_row_5x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, __m256& c10, __m256& c11,
                              __m256& c20, __m256& c21, __m256& c30, __m256& c31,
                              __m256& c40, __m256& c41, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(&A[3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(&A[4]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        A += 6; B += 16;
    }
}

inline void fma_loop_row_6x16(const float* A, const float* B,
                              __m256& c00, __m256& c01, __m256& c10, __m256& c11,
                              __m256& c20, __m256& c21, __m256& c30, __m256& c31,
                              __m256& c40, __m256& c41, __m256& c50, __m256& c51, int kc) {
    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B);
        __m256 b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(&A[0]); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(&A[1]); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(&A[2]); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(&A[3]); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(&A[4]); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(&A[5]); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        A += 6; B += 16;
    }
}

inline void kernel_6x16_row_zero(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    // 根据mr使用特化的FMA循环以减少不必要的计算
    switch (mr) {
        case 1: fma_loop_row_1x16(A, B, c00, c01, kc); break;
        case 2: fma_loop_row_2x16(A, B, c00, c01, c10, c11, kc); break;
        case 3: fma_loop_row_3x16(A, B, c00, c01, c10, c11, c20, c21, kc); break;
        case 4: fma_loop_row_4x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, kc); break;
        case 5: fma_loop_row_5x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, kc); break;
        default: fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc); break;
    }
    
    // 存储结果，行主序下C每行有ldc列
    if (nr == 16) {
        if (mr >= 1) { _mm256_storeu_ps(C, c00); _mm256_storeu_ps(C+8, c01); }
        if (mr >= 2) { _mm256_storeu_ps(C+ldc, c10); _mm256_storeu_ps(C+ldc+8, c11); }
        if (mr >= 3) { _mm256_storeu_ps(C+2*ldc, c20); _mm256_storeu_ps(C+2*ldc+8, c21); }
        if (mr >= 4) { _mm256_storeu_ps(C+3*ldc, c30); _mm256_storeu_ps(C+3*ldc+8, c31); }
        if (mr >= 5) { _mm256_storeu_ps(C+4*ldc, c40); _mm256_storeu_ps(C+4*ldc+8, c41); }
        if (mr >= 6) { _mm256_storeu_ps(C+5*ldc, c50); _mm256_storeu_ps(C+5*ldc+8, c51); }
    } else {
        build_masks(&mask0, &mask1, nr);
        if (mr >= 1) { _mm256_maskstore_ps(C, mask0, c00); _mm256_maskstore_ps(C+8, mask1, c01); }
        if (mr >= 2) { _mm256_maskstore_ps(C+ldc, mask0, c10); _mm256_maskstore_ps(C+ldc+8, mask1, c11); }
        if (mr >= 3) { _mm256_maskstore_ps(C+2*ldc, mask0, c20); _mm256_maskstore_ps(C+2*ldc+8, mask1, c21); }
        if (mr >= 4) { _mm256_maskstore_ps(C+3*ldc, mask0, c30); _mm256_maskstore_ps(C+3*ldc+8, mask1, c31); }
        if (mr >= 5) { _mm256_maskstore_ps(C+4*ldc, mask0, c40); _mm256_maskstore_ps(C+4*ldc+8, mask1, c41); }
        if (mr >= 6) { _mm256_maskstore_ps(C+5*ldc, mask0, c50); _mm256_maskstore_ps(C+5*ldc+8, mask1, c51); }
    }
}

inline void kernel_6x16_row_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    // 加载已有的C值
    if (nr == 16) {
        if (mr >= 1) { c00 = _mm256_loadu_ps(C); c01 = _mm256_loadu_ps(C+8); }
        if (mr >= 2) { c10 = _mm256_loadu_ps(C+ldc); c11 = _mm256_loadu_ps(C+ldc+8); }
        if (mr >= 3) { c20 = _mm256_loadu_ps(C+2*ldc); c21 = _mm256_loadu_ps(C+2*ldc+8); }
        if (mr >= 4) { c30 = _mm256_loadu_ps(C+3*ldc); c31 = _mm256_loadu_ps(C+3*ldc+8); }
        if (mr >= 5) { c40 = _mm256_loadu_ps(C+4*ldc); c41 = _mm256_loadu_ps(C+4*ldc+8); }
        if (mr >= 6) { c50 = _mm256_loadu_ps(C+5*ldc); c51 = _mm256_loadu_ps(C+5*ldc+8); }
    } else {
        build_masks(&mask0, &mask1, nr);
        if (mr >= 1) { c00 = _mm256_maskload_ps(C, mask0); c01 = _mm256_maskload_ps(C+8, mask1); }
        if (mr >= 2) { c10 = _mm256_maskload_ps(C+ldc, mask0); c11 = _mm256_maskload_ps(C+ldc+8, mask1); }
        if (mr >= 3) { c20 = _mm256_maskload_ps(C+2*ldc, mask0); c21 = _mm256_maskload_ps(C+2*ldc+8, mask1); }
        if (mr >= 4) { c30 = _mm256_maskload_ps(C+3*ldc, mask0); c31 = _mm256_maskload_ps(C+3*ldc+8, mask1); }
        if (mr >= 5) { c40 = _mm256_maskload_ps(C+4*ldc, mask0); c41 = _mm256_maskload_ps(C+4*ldc+8, mask1); }
        if (mr >= 6) { c50 = _mm256_maskload_ps(C+5*ldc, mask0); c51 = _mm256_maskload_ps(C+5*ldc+8, mask1); }
    }
    
    // 根据mr使用特化的FMA循环以减少不必要的计算
    switch (mr) {
        case 1: fma_loop_row_1x16(A, B, c00, c01, kc); break;
        case 2: fma_loop_row_2x16(A, B, c00, c01, c10, c11, kc); break;
        case 3: fma_loop_row_3x16(A, B, c00, c01, c10, c11, c20, c21, kc); break;
        case 4: fma_loop_row_4x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, kc); break;
        case 5: fma_loop_row_5x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, kc); break;
        default: fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc); break;
    }
    
    // 存储结果
    if (nr == 16) {
        if (mr >= 1) { _mm256_storeu_ps(C, c00); _mm256_storeu_ps(C+8, c01); }
        if (mr >= 2) { _mm256_storeu_ps(C+ldc, c10); _mm256_storeu_ps(C+ldc+8, c11); }
        if (mr >= 3) { _mm256_storeu_ps(C+2*ldc, c20); _mm256_storeu_ps(C+2*ldc+8, c21); }
        if (mr >= 4) { _mm256_storeu_ps(C+3*ldc, c30); _mm256_storeu_ps(C+3*ldc+8, c31); }
        if (mr >= 5) { _mm256_storeu_ps(C+4*ldc, c40); _mm256_storeu_ps(C+4*ldc+8, c41); }
        if (mr >= 6) { _mm256_storeu_ps(C+5*ldc, c50); _mm256_storeu_ps(C+5*ldc+8, c51); }
    } else {
        if (mr >= 1) { _mm256_maskstore_ps(C, mask0, c00); _mm256_maskstore_ps(C+8, mask1, c01); }
        if (mr >= 2) { _mm256_maskstore_ps(C+ldc, mask0, c10); _mm256_maskstore_ps(C+ldc+8, mask1, c11); }
        if (mr >= 3) { _mm256_maskstore_ps(C+2*ldc, mask0, c20); _mm256_maskstore_ps(C+2*ldc+8, mask1, c21); }
        if (mr >= 4) { _mm256_maskstore_ps(C+3*ldc, mask0, c30); _mm256_maskstore_ps(C+3*ldc+8, mask1, c31); }
        if (mr >= 5) { _mm256_maskstore_ps(C+4*ldc, mask0, c40); _mm256_maskstore_ps(C+4*ldc+8, mask1, c41); }
        if (mr >= 6) { _mm256_maskstore_ps(C+5*ldc, mask0, c50); _mm256_maskstore_ps(C+5*ldc+8, mask1, c51); }
    }
}

// 支持alpha/beta的行主序内核
// C = alpha * A @ B + beta * C
inline void kernel_6x16_row_alphabeta(float* A, float* B, float* C, int mr, int nr, int kc, int ldc,
                                       float alpha, float beta, bool first) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    // FMA循环：在寄存器中计算alpha * A @ B
    switch (mr) {
        case 1: fma_loop_row_1x16(A, B, c00, c01, kc); break;
        case 2: fma_loop_row_2x16(A, B, c00, c01, c10, c11, kc); break;
        case 3: fma_loop_row_3x16(A, B, c00, c01, c10, c11, c20, c21, kc); break;
        case 4: fma_loop_row_4x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, kc); break;
        case 5: fma_loop_row_5x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, kc); break;
        default: fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc); break;
    }
    
    // 将alpha应用到累加器
    __m256 valpha = _mm256_set1_ps(alpha);
    c00 = _mm256_mul_ps(c00, valpha); c01 = _mm256_mul_ps(c01, valpha);
    c10 = _mm256_mul_ps(c10, valpha); c11 = _mm256_mul_ps(c11, valpha);
    c20 = _mm256_mul_ps(c20, valpha); c21 = _mm256_mul_ps(c21, valpha);
    c30 = _mm256_mul_ps(c30, valpha); c31 = _mm256_mul_ps(c31, valpha);
    c40 = _mm256_mul_ps(c40, valpha); c41 = _mm256_mul_ps(c41, valpha);
    c50 = _mm256_mul_ps(c50, valpha); c51 = _mm256_mul_ps(c51, valpha);
    
    // 应用beta * C并存储
    if (first && beta != 0.0f) {
        __m256 vbeta = _mm256_set1_ps(beta);
        if (nr == 16) {
            if (mr >= 1) { c00 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C), c00); c01 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+8), c01); }
            if (mr >= 2) { c10 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+ldc), c10); c11 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+ldc+8), c11); }
            if (mr >= 3) { c20 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+2*ldc), c20); c21 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+2*ldc+8), c21); }
            if (mr >= 4) { c30 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+3*ldc), c30); c31 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+3*ldc+8), c31); }
            if (mr >= 5) { c40 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+4*ldc), c40); c41 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+4*ldc+8), c41); }
            if (mr >= 6) { c50 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+5*ldc), c50); c51 = _mm256_fmadd_ps(vbeta, _mm256_loadu_ps(C+5*ldc+8), c51); }
        } else {
            build_masks(&mask0, &mask1, nr);
            if (mr >= 1) { c00 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C, mask0), c00); c01 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+8, mask1), c01); }
            if (mr >= 2) { c10 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+ldc, mask0), c10); c11 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+ldc+8, mask1), c11); }
            if (mr >= 3) { c20 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+2*ldc, mask0), c20); c21 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+2*ldc+8, mask1), c21); }
            if (mr >= 4) { c30 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+3*ldc, mask0), c30); c31 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+3*ldc+8, mask1), c31); }
            if (mr >= 5) { c40 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+4*ldc, mask0), c40); c41 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+4*ldc+8, mask1), c41); }
            if (mr >= 6) { c50 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+5*ldc, mask0), c50); c51 = _mm256_fmadd_ps(vbeta, _mm256_maskload_ps(C+5*ldc+8, mask1), c51); }
        }
    } else if (!first) {
        // 从后续k块累加
        if (nr == 16) {
            if (mr >= 1) { c00 = _mm256_add_ps(c00, _mm256_loadu_ps(C)); c01 = _mm256_add_ps(c01, _mm256_loadu_ps(C+8)); }
            if (mr >= 2) { c10 = _mm256_add_ps(c10, _mm256_loadu_ps(C+ldc)); c11 = _mm256_add_ps(c11, _mm256_loadu_ps(C+ldc+8)); }
            if (mr >= 3) { c20 = _mm256_add_ps(c20, _mm256_loadu_ps(C+2*ldc)); c21 = _mm256_add_ps(c21, _mm256_loadu_ps(C+2*ldc+8)); }
            if (mr >= 4) { c30 = _mm256_add_ps(c30, _mm256_loadu_ps(C+3*ldc)); c31 = _mm256_add_ps(c31, _mm256_loadu_ps(C+3*ldc+8)); }
            if (mr >= 5) { c40 = _mm256_add_ps(c40, _mm256_loadu_ps(C+4*ldc)); c41 = _mm256_add_ps(c41, _mm256_loadu_ps(C+4*ldc+8)); }
            if (mr >= 6) { c50 = _mm256_add_ps(c50, _mm256_loadu_ps(C+5*ldc)); c51 = _mm256_add_ps(c51, _mm256_loadu_ps(C+5*ldc+8)); }
        } else {
            build_masks(&mask0, &mask1, nr);
            if (mr >= 1) { c00 = _mm256_add_ps(c00, _mm256_maskload_ps(C, mask0)); c01 = _mm256_add_ps(c01, _mm256_maskload_ps(C+8, mask1)); }
            if (mr >= 2) { c10 = _mm256_add_ps(c10, _mm256_maskload_ps(C+ldc, mask0)); c11 = _mm256_add_ps(c11, _mm256_maskload_ps(C+ldc+8, mask1)); }
            if (mr >= 3) { c20 = _mm256_add_ps(c20, _mm256_maskload_ps(C+2*ldc, mask0)); c21 = _mm256_add_ps(c21, _mm256_maskload_ps(C+2*ldc+8, mask1)); }
            if (mr >= 4) { c30 = _mm256_add_ps(c30, _mm256_maskload_ps(C+3*ldc, mask0)); c31 = _mm256_add_ps(c31, _mm256_maskload_ps(C+3*ldc+8, mask1)); }
            if (mr >= 5) { c40 = _mm256_add_ps(c40, _mm256_maskload_ps(C+4*ldc, mask0)); c41 = _mm256_add_ps(c41, _mm256_maskload_ps(C+4*ldc+8, mask1)); }
            if (mr >= 6) { c50 = _mm256_add_ps(c50, _mm256_maskload_ps(C+5*ldc, mask0)); c51 = _mm256_add_ps(c51, _mm256_maskload_ps(C+5*ldc+8, mask1)); }
        }
    }
    // first且beta==0时：直接存储alpha * result，该结果已计算完成
    
    // 存储结果
    if (nr == 16) {
        if (mr >= 1) { _mm256_storeu_ps(C, c00); _mm256_storeu_ps(C+8, c01); }
        if (mr >= 2) { _mm256_storeu_ps(C+ldc, c10); _mm256_storeu_ps(C+ldc+8, c11); }
        if (mr >= 3) { _mm256_storeu_ps(C+2*ldc, c20); _mm256_storeu_ps(C+2*ldc+8, c21); }
        if (mr >= 4) { _mm256_storeu_ps(C+3*ldc, c30); _mm256_storeu_ps(C+3*ldc+8, c31); }
        if (mr >= 5) { _mm256_storeu_ps(C+4*ldc, c40); _mm256_storeu_ps(C+4*ldc+8, c41); }
        if (mr >= 6) { _mm256_storeu_ps(C+5*ldc, c50); _mm256_storeu_ps(C+5*ldc+8, c51); }
    } else {
        if (nr < 16) build_masks(&mask0, &mask1, nr);
        if (mr >= 1) { _mm256_maskstore_ps(C, mask0, c00); _mm256_maskstore_ps(C+8, mask1, c01); }
        if (mr >= 2) { _mm256_maskstore_ps(C+ldc, mask0, c10); _mm256_maskstore_ps(C+ldc+8, mask1, c11); }
        if (mr >= 3) { _mm256_maskstore_ps(C+2*ldc, mask0, c20); _mm256_maskstore_ps(C+2*ldc+8, mask1, c21); }
        if (mr >= 4) { _mm256_maskstore_ps(C+3*ldc, mask0, c30); _mm256_maskstore_ps(C+3*ldc+8, mask1, c31); }
        if (mr >= 5) { _mm256_maskstore_ps(C+4*ldc, mask0, c40); _mm256_maskstore_ps(C+4*ldc+8, mask1, c41); }
        if (mr >= 6) { _mm256_maskstore_ps(C+5*ldc, mask0, c50); _mm256_maskstore_ps(C+5*ldc+8, mask1, c51); }
    }
}

// ============================================================================
// 打包函数
// ============================================================================

// 为列主序内核打包A矩阵，生成16xkc面板
inline void pack_a_col(const float* A, float* packed, int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR_COL) {
        int mr = std::min(MR_COL, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = A[p * lda + i + ii];
            }
            for (int ii = mr; ii < MR_COL; ++ii) {
                packed[ii] = 0.0f;
            }
            packed += MR_COL;
        }
    }
}

// 为列主序内核打包B矩阵，生成kcx6面板
inline void pack_b_col(const float* B, float* packed, int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR_COL) {
        int nr = std::min(NR_COL, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = B[(j + jj) * ldb + p];
            }
            for (int jj = nr; jj < NR_COL; ++jj) {
                packed[jj] = 0.0f;
            }
            packed += NR_COL;
        }
    }
}

// 为行主序内核打包A矩阵，生成6xkc面板
inline void pack_a_row(const float* A, float* packed, int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = A[(i + ii) * lda + p];
            }
            for (int ii = mr; ii < MR_ROW; ++ii) {
                packed[ii] = 0.0f;
            }
            packed += MR_ROW;
        }
    }
}

// 使用通用步幅打包A矩阵，生成6xkc面板用于行主序内核
inline void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = A[(i + ii) * rsa + p * csa];
            }
            for (int ii = mr; ii < MR_ROW; ++ii) {
                packed[ii] = 0.0f;
            }
            packed += MR_ROW;
        }
    }
}

// 为行主序内核打包B矩阵，生成kcx16面板，使用SIMD优化
inline void pack_b_row(const float* B, float* packed, int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        float* dest = packed + (j / NR_ROW) * NR_ROW * kc;
        
        if (nr == NR_ROW) {
            // 完整面板：使用SIMD复制
            for (int p = 0; p < kc; ++p) {
                const float* src = B + p * ldb + j;
                _mm256_storeu_ps(dest, _mm256_loadu_ps(src));
                _mm256_storeu_ps(dest + 8, _mm256_loadu_ps(src + 8));
                dest += NR_ROW;
            }
        } else {
            // 部分面板：标量复制
            for (int p = 0; p < kc; ++p) {
                for (int jj = 0; jj < nr; ++jj) {
                    dest[jj] = B[p * ldb + j + jj];
                }
                for (int jj = nr; jj < NR_ROW; ++jj) {
                    dest[jj] = 0.0f;
                }
                dest += NR_ROW;
            }
        }
    }
}

// 使用通用步幅打包B矩阵，生成kcx16面板用于行主序内核
inline void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = B[p * rsb + (j + jj) * csb];
            }
            for (int jj = nr; jj < NR_ROW; ++jj) {
                packed[jj] = 0.0f;
            }
            packed += NR_ROW;
        }
    }
}

// 存储到通用步幅C的6x16内核，较慢但灵活
inline void kernel_6x16_generic_store(float* A, float* B, float* C, 
                                      int mr, int nr, int kc, 
                                      int64_t rsc, int64_t csc,
                                      float alpha, float beta, bool first) {    
    // 使用SIMD计算，与FMA循环相同但存储到局部数组
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    
    fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc);
    
    // 提取到局部数组
    alignas(32) float tmp[MR_ROW][NR_ROW];
    _mm256_storeu_ps(&tmp[0][0], c00); _mm256_storeu_ps(&tmp[0][8], c01);
    _mm256_storeu_ps(&tmp[1][0], c10); _mm256_storeu_ps(&tmp[1][8], c11);
    _mm256_storeu_ps(&tmp[2][0], c20); _mm256_storeu_ps(&tmp[2][8], c21);
    _mm256_storeu_ps(&tmp[3][0], c30); _mm256_storeu_ps(&tmp[3][8], c31);
    _mm256_storeu_ps(&tmp[4][0], c40); _mm256_storeu_ps(&tmp[4][8], c41);
    _mm256_storeu_ps(&tmp[5][0], c50); _mm256_storeu_ps(&tmp[5][8], c51);
    
    // 使用通用步幅存储
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float* c_ptr = C + i * rsc + j * csc;
            if (first) {
                *c_ptr = alpha * tmp[i][j] + beta * (*c_ptr);
            } else {
                *c_ptr += alpha * tmp[i][j];
            }
        }
    }
}

// ============================================================================
// 特殊形状的专用内核
// 针对边界情况的高性能实现: 点积、外积、行向量乘矩阵、矩阵乘列向量
// ============================================================================

// 辅助函数: __m256的水平求和
inline float hsum_ps_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// 点积: C 1x1 = A 1xk @ B kx1 - 使用SIMD 4倍展开优化
inline float dot_product_simd(const float* A, const float* B, int k, 
                              int64_t csa, int64_t rsb) {
    float sum = 0.0f;
    
    // 快速路径：两者均连续
    if (csa == 1 && rsb == 1) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        
        int p = 0;
        // 4倍展开循环
        for (; p + 32 <= k; p += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(B + p), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 8), _mm256_loadu_ps(B + p + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 16), _mm256_loadu_ps(B + p + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 24), _mm256_loadu_ps(B + p + 24), acc3);
        }
        // 合并累加器
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        
        // 处理剩余的8元素块
        for (; p + 8 <= k; p += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(B + p), acc0);
        }
        
        sum = hsum_ps_avx(acc0);
        // 尾部处理
        for (; p < k; ++p) sum += A[p] * B[p];
    } else {
        // 步幅情况：仍尽可能向量化
        for (int p = 0; p < k; ++p) {
            sum += A[p * csa] * B[p * rsb];
        }
    }
    return sum;
}

// 外积：C[mxn] = A[mx1] @ B[1xn]，为SIMD存储优化
// 一次处理多行以更好地利用缓存
inline void outer_product_simd(const float* A, const float* B, float* C,
                               int m, int n, float alpha, float beta,
                               int64_t rsa, int64_t csb, int64_t rsc, int64_t csc) {
    // 快速路径：C是行主序连续且B是连续的
    if (csc == 1 && rsc >= n && csb == 1) {
        // 一次处理4行
        int i = 0;
        for (; i + 4 <= m; i += 4) {
            __m256 va0 = _mm256_set1_ps(alpha * A[i * rsa]);
            __m256 va1 = _mm256_set1_ps(alpha * A[(i + 1) * rsa]);
            __m256 va2 = _mm256_set1_ps(alpha * A[(i + 2) * rsa]);
            __m256 va3 = _mm256_set1_ps(alpha * A[(i + 3) * rsa]);
            __m256 vbeta = _mm256_set1_ps(beta);
            
            float* c0 = C + i * rsc;
            float* c1 = C + (i + 1) * rsc;
            float* c2 = C + (i + 2) * rsc;
            float* c3 = C + (i + 3) * rsc;
            
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vb = _mm256_loadu_ps(B + j);
                
                __m256 vc0 = _mm256_loadu_ps(c0 + j);
                __m256 vc1 = _mm256_loadu_ps(c1 + j);
                __m256 vc2 = _mm256_loadu_ps(c2 + j);
                __m256 vc3 = _mm256_loadu_ps(c3 + j);
                
                vc0 = _mm256_fmadd_ps(va0, vb, _mm256_mul_ps(vbeta, vc0));
                vc1 = _mm256_fmadd_ps(va1, vb, _mm256_mul_ps(vbeta, vc1));
                vc2 = _mm256_fmadd_ps(va2, vb, _mm256_mul_ps(vbeta, vc2));
                vc3 = _mm256_fmadd_ps(va3, vb, _mm256_mul_ps(vbeta, vc3));
                
                _mm256_storeu_ps(c0 + j, vc0);
                _mm256_storeu_ps(c1 + j, vc1);
                _mm256_storeu_ps(c2 + j, vc2);
                _mm256_storeu_ps(c3 + j, vc3);
            }
            // 尾部处理
            for (; j < n; ++j) {
                c0[j] = alpha * A[i * rsa] * B[j] + beta * c0[j];
                c1[j] = alpha * A[(i + 1) * rsa] * B[j] + beta * c1[j];
                c2[j] = alpha * A[(i + 2) * rsa] * B[j] + beta * c2[j];
                c3[j] = alpha * A[(i + 3) * rsa] * B[j] + beta * c3[j];
            }
        }
        // 剩余行
        for (; i < m; ++i) {
            float a_val = alpha * A[i * rsa];
            float* c_row = C + i * rsc;
            __m256 va = _mm256_set1_ps(a_val);
            __m256 vbeta = _mm256_set1_ps(beta);
            
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vb = _mm256_loadu_ps(B + j);
                __m256 vc = _mm256_loadu_ps(c_row + j);
                vc = _mm256_fmadd_ps(va, vb, _mm256_mul_ps(vbeta, vc));
                _mm256_storeu_ps(c_row + j, vc);
            }
            for (; j < n; ++j) {
                c_row[j] = a_val * B[j] + beta * c_row[j];
            }
        }
    } else if (csc == 1 && rsc >= n) {
        // C是行主序但B有步幅
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * A[i * rsa];
            float* c_row = C + i * rsc;
            for (int j = 0; j < n; ++j) {
                c_row[j] = a_val * B[j * csb] + beta * c_row[j];
            }
        }
    } else {
        // 通用步幅情况
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * A[i * rsa];
            for (int j = 0; j < n; ++j) {
                float* c_ptr = C + i * rsc + j * csc;
                *c_ptr = a_val * B[j * csb] + beta * (*c_ptr);
            }
        }
    }
}

// C[1xn] = A[1xk] @ B[kxn]：优化的行向量乘矩阵
// 对于行主序B：流式遍历k，每次迭代处理整个n
// 确保B的每行只被读取一次
inline void gemv_row_simd(const float* __restrict A, const float* __restrict B, float* __restrict C,
                          int n, int k, float alpha, float beta,
                          int64_t csa, int64_t rsb, int64_t csb, int64_t csc) {
    // 快速路径：A在k方向连续，B是行主序，C连续
    if (csa == 1 && csb == 1 && rsb == n && csc == 1) {
        // 初始化C
        if (beta == 0.0f) {
            memset(C, 0, n * sizeof(float));
        } else if (beta != 1.0f) {
            __m256 vbeta = _mm256_set1_ps(beta);
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                _mm256_storeu_ps(C + j, _mm256_mul_ps(vbeta, _mm256_loadu_ps(C + j)));
            }
            for (; j < n; ++j) C[j] *= beta;
        }
        
        // 流式遍历k：C += A[p] * B[p, :]
        // 将k展开4次以获得更好的指令级并行
        int p = 0;
        for (; p + 4 <= k; p += 4) {
            float a0 = alpha * A[p];
            float a1 = alpha * A[p + 1];
            float a2 = alpha * A[p + 2];
            float a3 = alpha * A[p + 3];
            __m256 va0 = _mm256_broadcast_ss(&a0);
            __m256 va1 = _mm256_broadcast_ss(&a1);
            __m256 va2 = _mm256_broadcast_ss(&a2);
            __m256 va3 = _mm256_broadcast_ss(&a3);
            
            const float* b0 = B + p * n;
            const float* b1 = B + (p + 1) * n;
            const float* b2 = B + (p + 2) * n;
            const float* b3 = B + (p + 3) * n;
            
            int j = 0;
            for (; j + 32 <= n; j += 32) {
                __m256 c0 = _mm256_loadu_ps(C + j);
                __m256 c1 = _mm256_loadu_ps(C + j + 8);
                __m256 c2 = _mm256_loadu_ps(C + j + 16);
                __m256 c3 = _mm256_loadu_ps(C + j + 24);
                
                c0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j), c0);
                c0 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j), c0);
                c0 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j), c0);
                c0 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j), c0);
                
                c1 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 8), c1);
                c1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 8), c1);
                c1 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 8), c1);
                c1 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 8), c1);
                
                c2 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 16), c2);
                c2 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 16), c2);
                c2 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 16), c2);
                c2 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 16), c2);
                
                c3 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 24), c3);
                c3 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 24), c3);
                c3 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 24), c3);
                c3 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 24), c3);
                
                _mm256_storeu_ps(C + j, c0);
                _mm256_storeu_ps(C + j + 8, c1);
                _mm256_storeu_ps(C + j + 16, c2);
                _mm256_storeu_ps(C + j + 24, c3);
            }
            for (; j + 8 <= n; j += 8) {
                __m256 c0 = _mm256_loadu_ps(C + j);
                c0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j), c0);
                c0 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j), c0);
                c0 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j), c0);
                c0 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j), c0);
                _mm256_storeu_ps(C + j, c0);
            }
            for (; j < n; ++j) {
                C[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
            }
        }
        
        // 剩余k
        for (; p < k; ++p) {
            float a_val = alpha * A[p];
            __m256 va = _mm256_broadcast_ss(&a_val);
            const float* b_row = B + p * n;
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                _mm256_storeu_ps(C + j, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), _mm256_loadu_ps(C + j)));
            }
            for (; j < n; ++j) C[j] += a_val * b_row[j];
        }
    } else {
        // 通用情况：用点积处理每列
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float* b_col = B + j * csb;
            
            if (csa == 1 && rsb == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int p = 0;
                for (; p + 16 <= k; p += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(b_col + p), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 8), _mm256_loadu_ps(b_col + p + 8), acc1);
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                for (; p + 8 <= k; p += 8) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(b_col + p), acc0);
                }
                sum = hsum_ps_avx(acc0);
                for (; p < k; ++p) sum += A[p] * b_col[p];
            } else {
                for (int p = 0; p < k; ++p) {
                    sum += A[p * csa] * b_col[p * rsb];
                }
            }
            C[j * csc] = alpha * sum + beta * C[j * csc];
        }
    }
}

// C[mx1] = A[mxk] @ B[kx1]：优化的矩阵乘列向量
// 一次处理8行以获得更好的内存访问模式
inline void gemv_col_simd(const float* A, const float* B, float* C,
                          int m, int k, float alpha, float beta,
                          int64_t rsa, int64_t csa, int64_t rsb, int64_t rsc) {
    // 快速路径：A是行主序，B连续，C连续
    if (csa == 1 && rsb == 1 && rsc == 1) {
        // 一次处理8行
        int i = 0;
        for (; i + 8 <= m; i += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();
            
            // 将k维度按8个一组处理
            int p = 0;
            for (; p + 8 <= k; p += 8) {
                __m256 vb = _mm256_loadu_ps(B + p);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + i * rsa + p), vb, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 1) * rsa + p), vb, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 2) * rsa + p), vb, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 3) * rsa + p), vb, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 4) * rsa + p), vb, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 5) * rsa + p), vb, acc5);
                acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 6) * rsa + p), vb, acc6);
                acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + 7) * rsa + p), vb, acc7);
            }
            
            // 水平求和
            float sum0 = hsum_ps_avx(acc0);
            float sum1 = hsum_ps_avx(acc1);
            float sum2 = hsum_ps_avx(acc2);
            float sum3 = hsum_ps_avx(acc3);
            float sum4 = hsum_ps_avx(acc4);
            float sum5 = hsum_ps_avx(acc5);
            float sum6 = hsum_ps_avx(acc6);
            float sum7 = hsum_ps_avx(acc7);
            
            // 处理尾部
            for (; p < k; ++p) {
                float b_val = B[p];
                sum0 += A[i * rsa + p] * b_val;
                sum1 += A[(i + 1) * rsa + p] * b_val;
                sum2 += A[(i + 2) * rsa + p] * b_val;
                sum3 += A[(i + 3) * rsa + p] * b_val;
                sum4 += A[(i + 4) * rsa + p] * b_val;
                sum5 += A[(i + 5) * rsa + p] * b_val;
                sum6 += A[(i + 6) * rsa + p] * b_val;
                sum7 += A[(i + 7) * rsa + p] * b_val;
            }
            
            // 存储结果
            C[i] = alpha * sum0 + beta * C[i];
            C[i + 1] = alpha * sum1 + beta * C[i + 1];
            C[i + 2] = alpha * sum2 + beta * C[i + 2];
            C[i + 3] = alpha * sum3 + beta * C[i + 3];
            C[i + 4] = alpha * sum4 + beta * C[i + 4];
            C[i + 5] = alpha * sum5 + beta * C[i + 5];
            C[i + 6] = alpha * sum6 + beta * C[i + 6];
            C[i + 7] = alpha * sum7 + beta * C[i + 7];
        }
        
        // 剩余行
        for (; i < m; ++i) {
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= k; p += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i * rsa + p), _mm256_loadu_ps(B + p), acc);
            }
            float sum = hsum_ps_avx(acc);
            for (; p < k; ++p) sum += A[i * rsa + p] * B[p];
            C[i] = alpha * sum + beta * C[i];
        }
    } else {
        // 通用情况
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            const float* a_row = A + i * rsa;
            
            if (csa == 1 && rsb == 1) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                int p = 0;
                for (; p + 16 <= k; p += 16) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p), _mm256_loadu_ps(B + p), acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 8), _mm256_loadu_ps(B + p + 8), acc1);
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                for (; p + 8 <= k; p += 8) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p), _mm256_loadu_ps(B + p), acc0);
                }
                sum = hsum_ps_avx(acc0);
                for (; p < k; ++p) sum += a_row[p] * B[p];
            } else {
                for (int p = 0; p < k; ++p) {
                    sum += a_row[p * csa] * B[p * rsb];
                }
            }
            C[i * rsc] = alpha * sum + beta * C[i * rsc];
        }
    }
}

// ============================================================================
// 列主序GEMM，使用16x6内核
// C m x n = A m x k @ B k x n，全部为列主序
// ============================================================================

inline void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    static thread_local AlignedBuffer buf_a, buf_b;
    // 分配带对齐填充的缓冲区
    // 列主序: A面板为 MR_COL x kc，B面板为 kc x NR_COL
    size_t a_buf_size = (static_cast<size_t>(GEMM_MC + MR_COL - 1) / MR_COL) * MR_COL * GEMM_KC;
    size_t b_buf_size = (static_cast<size_t>(GEMM_NC + NR_COL - 1) / NR_COL) * NR_COL * GEMM_KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    for (int j = 0; j < n; j += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - j);
        
        for (int p = 0; p < k; p += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - p);
            bool first = (p == 0);
            
            pack_b_col(B + j * k + p, buf_b.data, kc, nc, k);
            
            for (int i = 0; i < m; i += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - i);
                
                pack_a_col(A + p * m + i, buf_a.data, mc, kc, m);
                
                for (int jr = 0; jr < nc; jr += NR_COL) {
                    int nr = std::min(NR_COL, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR_COL) {
                        int mr = std::min(MR_COL, mc - ir);
                        
                        float* C_ij = C + (j + jr) * m + (i + ir);
                        // 打包后的A: 每个面板为 MR_COL x kc
                        // 打包后的B：每个面板是kc x NR_COL
                        float* packed_a = buf_a.data + (ir / MR_COL) * MR_COL * kc;
                        float* packed_b = buf_b.data + (jr / NR_COL) * NR_COL * kc;
                        
                        if (first) {
                            kernel_16x6_col_zero(packed_a, packed_b, C_ij, mr, nr, kc, m);
                        } else {
                            kernel_16x6_col_load(packed_a, packed_b, C_ij, mr, nr, kc, m);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 行主序GEMM，使用6x16内核
// C m x n = A m x k @ B k x n，全部为行主序
// 基于sgemm.c的循环结构以获得更好的缓存利用率
// ============================================================================

// 行主序GEMM的全局静态缓冲区，与sgemm.c模式相同
alignas(64) static float g_blockA_row[GEMM_MC * GEMM_KC];
alignas(64) static float g_blockB_row[GEMM_NC * GEMM_KC];

// 行主序B的并行打包 - SIMD优化
inline void pack_blockB_row_par(const float* B, float* packed, int nc, int kc, int ldb) {
#ifdef _OPENMP
    if (g_num_threads > 1) {
        #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
        for (int j = 0; j < nc; j += NR_ROW) {
            int nr = std::min(NR_ROW, nc - j);
            float* dest = packed + (j / NR_ROW) * NR_ROW * kc;
            
            if (nr == NR_ROW) {
                for (int p = 0; p < kc; ++p) {
                    const float* src = B + p * ldb + j;
                    _mm256_storeu_ps(dest, _mm256_loadu_ps(src));
                    _mm256_storeu_ps(dest + 8, _mm256_loadu_ps(src + 8));
                    dest += NR_ROW;
                }
            } else {
                for (int p = 0; p < kc; ++p) {
                    for (int jj = 0; jj < nr; ++jj) {
                        dest[jj] = B[p * ldb + j + jj];
                    }
                    for (int jj = nr; jj < NR_ROW; ++jj) {
                        dest[jj] = 0.0f;
                    }
                    dest += NR_ROW;
                }
            }
        }
    } else
#endif
    {
        for (int j = 0; j < nc; j += NR_ROW) {
            int nr = std::min(NR_ROW, nc - j);
            float* dest = packed + (j / NR_ROW) * NR_ROW * kc;
            
            if (nr == NR_ROW) {
                for (int p = 0; p < kc; ++p) {
                    const float* src = B + p * ldb + j;
                    _mm256_storeu_ps(dest, _mm256_loadu_ps(src));
                    _mm256_storeu_ps(dest + 8, _mm256_loadu_ps(src + 8));
                    dest += NR_ROW;
                }
            } else {
                for (int p = 0; p < kc; ++p) {
                    for (int jj = 0; jj < nr; ++jj) {
                        dest[jj] = B[p * ldb + j + jj];
                    }
                    for (int jj = nr; jj < NR_ROW; ++jj) {
                        dest[jj] = 0.0f;
                    }
                    dest += NR_ROW;
                }
            }
        }
    }
}

// 用于打包A的辅助函数，从并行版本提取
inline void pack_blockA_row_single_tile(const float* A, float* dest, int i, int mr, int kc, int lda) {
    int p = 0;
    if (mr == MR_ROW) {
        for (; p + 8 <= kc; p += 8) {
            if (p + 16 <= kc) {
                _mm_prefetch((const char*)(A + i * lda + p + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(A + (i+1) * lda + p + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(A + (i+2) * lda + p + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(A + (i+3) * lda + p + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(A + (i+4) * lda + p + 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(A + (i+5) * lda + p + 16), _MM_HINT_T0);
            }
            
            __m256 r0 = _mm256_loadu_ps(A + i * lda + p);
            __m256 r1 = _mm256_loadu_ps(A + (i+1) * lda + p);
            __m256 r2 = _mm256_loadu_ps(A + (i+2) * lda + p);
            __m256 r3 = _mm256_loadu_ps(A + (i+3) * lda + p);
            __m256 r4 = _mm256_loadu_ps(A + (i+4) * lda + p);
            __m256 r5 = _mm256_loadu_ps(A + (i+5) * lda + p);
            
            float* d = dest + p * MR_ROW;
            d[0] = r0[0]; d[1] = r1[0]; d[2] = r2[0]; d[3] = r3[0]; d[4] = r4[0]; d[5] = r5[0]; d += MR_ROW;
            d[0] = r0[1]; d[1] = r1[1]; d[2] = r2[1]; d[3] = r3[1]; d[4] = r4[1]; d[5] = r5[1]; d += MR_ROW;
            d[0] = r0[2]; d[1] = r1[2]; d[2] = r2[2]; d[3] = r3[2]; d[4] = r4[2]; d[5] = r5[2]; d += MR_ROW;
            d[0] = r0[3]; d[1] = r1[3]; d[2] = r2[3]; d[3] = r3[3]; d[4] = r4[3]; d[5] = r5[3]; d += MR_ROW;
            d[0] = r0[4]; d[1] = r1[4]; d[2] = r2[4]; d[3] = r3[4]; d[4] = r4[4]; d[5] = r5[4]; d += MR_ROW;
            d[0] = r0[5]; d[1] = r1[5]; d[2] = r2[5]; d[3] = r3[5]; d[4] = r4[5]; d[5] = r5[5]; d += MR_ROW;
            d[0] = r0[6]; d[1] = r1[6]; d[2] = r2[6]; d[3] = r3[6]; d[4] = r4[6]; d[5] = r5[6]; d += MR_ROW;
            d[0] = r0[7]; d[1] = r1[7]; d[2] = r2[7]; d[3] = r3[7]; d[4] = r4[7]; d[5] = r5[7];
        }
    }
    
    for (; p < kc; ++p) {
        for (int ii = 0; ii < mr; ++ii) {
            dest[p * MR_ROW + ii] = A[(i + ii) * lda + p];
        }
        for (int ii = mr; ii < MR_ROW; ++ii) {
            dest[p * MR_ROW + ii] = 0.0f;
        }
    }
}

// 并行打包A用于行主序  
inline void pack_blockA_row_par(const float* A, float* packed, int mc, int kc, int lda) {
#ifdef _OPENMP
    if (g_num_threads > 1) {
        #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
        for (int i = 0; i < mc; i += MR_ROW) {
            int mr = std::min(MR_ROW, mc - i);
            float* dest = packed + (i / MR_ROW) * MR_ROW * kc;
            pack_blockA_row_single_tile(A, dest, i, mr, kc, lda);
        }
    } else
#endif
    {
        for (int i = 0; i < mc; i += MR_ROW) {
            int mr = std::min(MR_ROW, mc - i);
            float* dest = packed + (i / MR_ROW) * MR_ROW * kc;
            pack_blockA_row_single_tile(A, dest, i, mr, kc, lda);
        }
    }
}

inline void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    // 使用sgemm.c风格的循环结构：单独处理第一个GEMM_KC块
    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        
        // 第一个GEMM_KC块：零初始化
        int kc = std::min(GEMM_KC, k);
        pack_blockB_row_par(B + jj, g_blockB_row, nc, kc, n);
        
        for (int ii = 0; ii < m; ii += GEMM_MC) {
            int mc = std::min(GEMM_MC, m - ii);
            pack_blockA_row_par(A + ii * k, g_blockA_row, mc, kc, k);
            
            // 对于行主序C，外层循环迭代ir以获得更好的缓存局部性
#ifdef _OPENMP
            // 仅在有多线程时使用OpenMP
            if (g_num_threads > 1) {
                #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        if (mr == MR_ROW && nr == NR_ROW) {
                            kernel_6x16_row_full(
                                g_blockA_row + (ir / MR_ROW) * MR_ROW * kc,
                                g_blockB_row + (jr / NR_ROW) * NR_ROW * kc,
                                C + (ii + ir) * n + (jj + jr),
                                kc, n);
                        } else {
                            kernel_6x16_row_zero(
                                g_blockA_row + (ir / MR_ROW) * MR_ROW * kc,
                                g_blockB_row + (jr / NR_ROW) * NR_ROW * kc,
                                C + (ii + ir) * n + (jj + jr),
                                mr, nr, kc, n);
                        }
                    }
                }
            } else
#endif
            {
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        if (mr == MR_ROW && nr == NR_ROW) {
                            kernel_6x16_row_full(
                                g_blockA_row + (ir / MR_ROW) * MR_ROW * kc,
                                g_blockB_row + (jr / NR_ROW) * NR_ROW * kc,
                                C + (ii + ir) * n + (jj + jr),
                                kc, n);
                        } else {
                            kernel_6x16_row_zero(
                                g_blockA_row + (ir / MR_ROW) * MR_ROW * kc,
                                g_blockB_row + (jr / NR_ROW) * NR_ROW * kc,
                                C + (ii + ir) * n + (jj + jr),
                                mr, nr, kc, n);
                        }
                    }
                }
            }
        }
        
        // 剩余GEMM_KC块：加载累加
        for (int pp = kc; pp < k; pp += GEMM_KC) {
            int kc2 = std::min(GEMM_KC, k - pp);
            pack_blockB_row_par(B + pp * n + jj, g_blockB_row, nc, kc2, n);
            
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                pack_blockA_row_par(A + ii * k + pp, g_blockA_row, mc, kc2, k);
                
#ifdef _OPENMP
                if (g_num_threads > 1) {
                    #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
                    for (int ir = 0; ir < mc; ir += MR_ROW) {
                        int mr = std::min(MR_ROW, mc - ir);
                        for (int jr = 0; jr < nc; jr += NR_ROW) {
                            int nr = std::min(NR_ROW, nc - jr);
                            if (mr == MR_ROW && nr == NR_ROW) {
                                kernel_6x16_row_full_accum(
                                    g_blockA_row + (ir / MR_ROW) * MR_ROW * kc2,
                                    g_blockB_row + (jr / NR_ROW) * NR_ROW * kc2,
                                    C + (ii + ir) * n + (jj + jr),
                                    kc2, n);
                            } else {
                                kernel_6x16_row_load(
                                    g_blockA_row + (ir / MR_ROW) * MR_ROW * kc2,
                                    g_blockB_row + (jr / NR_ROW) * NR_ROW * kc2,
                                    C + (ii + ir) * n + (jj + jr),
                                    mr, nr, kc2, n);
                            }
                        }
                    }
                } else
#endif
                {
                    for (int ir = 0; ir < mc; ir += MR_ROW) {
                        int mr = std::min(MR_ROW, mc - ir);
                        for (int jr = 0; jr < nc; jr += NR_ROW) {
                            int nr = std::min(NR_ROW, nc - jr);
                            if (mr == MR_ROW && nr == NR_ROW) {
                                kernel_6x16_row_full_accum(
                                    g_blockA_row + (ir / MR_ROW) * MR_ROW * kc2,
                                    g_blockB_row + (jr / NR_ROW) * NR_ROW * kc2,
                                    C + (ii + ir) * n + (jj + jr),
                                    kc2, n);
                            } else {
                                kernel_6x16_row_load(
                                    g_blockA_row + (ir / MR_ROW) * MR_ROW * kc2,
                                    g_blockB_row + (jr / NR_ROW) * NR_ROW * kc2,
                                    C + (ii + ir) * n + (jj + jr),
                                    mr, nr, kc2, n);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 带OpenMP的并行行主序GEMM
// ============================================================================

#ifdef _OPENMP

// 简化的并行版本 - 直接使用已带OpenMP的sgemm_rowmajor
inline void sgemm_rowmajor_parallel(const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads) {
    // sgemm_rowmajor现在使用全局g_num_threads进行并行
    sgemm_rowmajor(A, B, C, m, n, k);
}
#endif

// ============================================================================
// 支持步幅的通用GEMM
// 自动检测布局并选择最优内核
// ============================================================================

/**
 * @brief BLAS风格的单精度矩阵乘法
 * C = alpha * A @ B + beta * C
 * 
 * @param A 输入矩阵A指针
 * @param B 输入矩阵B指针
 * @param C 输出矩阵C指针，原地修改
 * @param m A和C的行数
 * @param n B和C的列数
 * @param k A的列数和B的行数
 * @param alpha A*B的标量乘数
 * @param beta 现有C值的标量乘数
 * @param rsa A的行步幅，连续行之间的距离
 * @param csa A的列步幅，连续列之间的距离
 * @param rsb B的行步幅
 * @param csb B的列步幅
 * @param rsc C的行步幅
 * @param csc C的列步幅
 */
inline void sgemm(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc
) {
    if (m == 0 || n == 0) return;
    
    // 处理特殊情况
    if (k == 0 || alpha == 0.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * rsc + j * csc] = 0.0f;
        } else if (beta != 1.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * rsc + j * csc] *= beta;
        }
        return;
    }
    
    // =========================================================================
    // 针对极端形状的特化内核
    // =========================================================================
    
    // 点积：m=1, n=1 产生标量结果
    if (m == 1 && n == 1) {
        float result = dot_product_simd(A, B, k, csa, rsb);
        *C = alpha * result + beta * (*C);
        return;
    }
    
    // 外积：k=1 产生秩1更新
    if (k == 1) {
        outer_product_simd(A, B, C, m, n, alpha, beta, rsa, csb, rsc, csc);
        return;
    }
    
    // 行向量乘矩阵：m=1
    if (m == 1) {
        gemv_row_simd(A, B, C, n, k, alpha, beta, csa, rsb, csb, csc);
        return;
    }
    
    // 矩阵乘列向量：n=1
    if (n == 1) {
        gemv_col_simd(A, B, C, m, k, alpha, beta, rsa, csa, rsb, rsc);
        return;
    }
    
    // 小m优化：当m较小或m%6!=0时使用行向k流式处理
    // 这避免了打包开销，对于小m比带填充的6x16内核更高效
    // 扩展范围：2<=m<=24，但仅当以下条件满足时
    //   - m<=6 时总是比带填充的6x16更好
    //   - 或者7<=m<=24 且 m%6!=0 时避免填充开销
    // 用于行主序A和B和C的情况
    bool use_small_m_kernel = false;
    if (m >= 2 && m <= 24 && n >= 64 && k >= 64 && csa == 1 && rsa == k && csb == 1 && rsb == n && csc == 1 && rsc == n) {
        if (m <= 6) {
            use_small_m_kernel = true;  // Always better for very small m
        } else if (m % MR_ROW != 0) {
            use_small_m_kernel = true;  // Avoid padding overhead
        }
    }
    
    if (use_small_m_kernel) {
        // 初始化C
        if (beta == 0.0f) {
            memset(C, 0, (size_t)m * n * sizeof(float));
        } else if (beta != 1.0f) {
            __m256 vbeta = _mm256_set1_ps(beta);
            for (int i = 0; i < m; ++i) {
                float* c_row = C + i * n;
                int j = 0;
                for (; j + 8 <= n; j += 8) {
                    _mm256_storeu_ps(c_row + j, _mm256_mul_ps(vbeta, _mm256_loadu_ps(c_row + j)));
                }
                for (; j < n; ++j) c_row[j] *= beta;
            }
        }
        
        // 对每个k同时处理所有m行
        // 这样只遍历B一次，C的行保持在缓存中
        int p = 0;
        for (; p + 4 <= k; p += 4) {
            // 处理A的每一行
            for (int i = 0; i < m; ++i) {
                float a0 = alpha * A[i * k + p];
                float a1 = alpha * A[i * k + p + 1];
                float a2 = alpha * A[i * k + p + 2];
                float a3 = alpha * A[i * k + p + 3];
                __m256 va0 = _mm256_broadcast_ss(&a0);
                __m256 va1 = _mm256_broadcast_ss(&a1);
                __m256 va2 = _mm256_broadcast_ss(&a2);
                __m256 va3 = _mm256_broadcast_ss(&a3);
                
                const float* b0 = B + p * n;
                const float* b1 = B + (p + 1) * n;
                const float* b2 = B + (p + 2) * n;
                const float* b3 = B + (p + 3) * n;
                float* c_row = C + i * n;
                
                int j = 0;
                for (; j + 32 <= n; j += 32) {
                    __m256 c0 = _mm256_loadu_ps(c_row + j);
                    __m256 c1 = _mm256_loadu_ps(c_row + j + 8);
                    __m256 c2 = _mm256_loadu_ps(c_row + j + 16);
                    __m256 c3 = _mm256_loadu_ps(c_row + j + 24);
                    
                    c0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j), c0);
                    c0 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j), c0);
                    c0 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j), c0);
                    c0 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j), c0);
                    
                    c1 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 8), c1);
                    c1 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 8), c1);
                    c1 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 8), c1);
                    c1 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 8), c1);
                    
                    c2 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 16), c2);
                    c2 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 16), c2);
                    c2 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 16), c2);
                    c2 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 16), c2);
                    
                    c3 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j + 24), c3);
                    c3 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j + 24), c3);
                    c3 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j + 24), c3);
                    c3 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j + 24), c3);
                    
                    _mm256_storeu_ps(c_row + j, c0);
                    _mm256_storeu_ps(c_row + j + 8, c1);
                    _mm256_storeu_ps(c_row + j + 16, c2);
                    _mm256_storeu_ps(c_row + j + 24, c3);
                }
                for (; j + 8 <= n; j += 8) {
                    __m256 c0 = _mm256_loadu_ps(c_row + j);
                    c0 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j), c0);
                    c0 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j), c0);
                    c0 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j), c0);
                    c0 = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j), c0);
                    _mm256_storeu_ps(c_row + j, c0);
                }
                for (; j < n; ++j) {
                    c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                }
            }
        }
        
        // 剩余k
        for (; p < k; ++p) {
            const float* b_row = B + p * n;
            for (int i = 0; i < m; ++i) {
                float a_val = alpha * A[i * k + p];
                __m256 va = _mm256_broadcast_ss(&a_val);
                float* c_row = C + i * n;
                int j = 0;
                for (; j + 8 <= n; j += 8) {
                    _mm256_storeu_ps(c_row + j, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), _mm256_loadu_ps(c_row + j)));
                }
                for (; j < n; ++j) c_row[j] += a_val * b_row[j];
            }
        }
        return;
    }
    
    // =========================================================================
    // 标准GEMM路径
    // =========================================================================
    
    // 检测布局：行主序有csx=1，列主序有rsx=1
    // 优化路径需要严格的连续布局，即步幅等于维度
    bool a_rowmajor_exact = (csa == 1 && rsa == k);  // A是m×k矩阵，行主序，无填充
    bool a_colmajor_exact = (rsa == 1 && csa == m);  // A是m×k矩阵，列主序，无填充
    bool b_rowmajor_exact = (csb == 1 && rsb == n);  // B是k×n矩阵，行主序，无填充
    bool b_colmajor_exact = (rsb == 1 && csb == k);  // B是k×n矩阵，列主序，无填充
    bool c_rowmajor_exact = (csc == 1 && rsc == n);  // C是m×n矩阵，行主序，无填充
    bool c_colmajor_exact = (rsc == 1 && csc == m);  // C是m×n矩阵，列主序，无填充
    
    // 对于通用路径，检查C是否至少是行主序，允许有填充
    bool c_rowmajor = (csc == 1 && rsc >= n);
    
    // 最佳情况：所有矩阵都有精确布局，使用优化路径
    if (a_colmajor_exact && b_colmajor_exact && c_colmajor_exact && alpha == 1.0f && beta == 0.0f) {
        sgemm_colmajor(const_cast<float*>(A), const_cast<float*>(B), C, m, n, k);
        return;
    }
    
    if (a_rowmajor_exact && b_rowmajor_exact && c_rowmajor_exact && alpha == 1.0f && beta == 0.0f) {
#ifdef _OPENMP
        if (g_num_threads > 1) {
            sgemm_rowmajor_parallel(A, B, C, m, n, k, g_num_threads);
        } else {
            sgemm_rowmajor(A, B, C, m, n, k);
        }
#else
        sgemm_rowmajor(A, B, C, m, n, k);
#endif
        return;
    }
    
    // 通用回退路径，支持任意步幅，使用SIMD微内核
    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_buf_size = (static_cast<size_t>(GEMM_MC + MR_ROW - 1) / MR_ROW) * MR_ROW * GEMM_KC;
    size_t b_buf_size = (static_cast<size_t>(GEMM_NC + NR_ROW - 1) / NR_ROW) * NR_ROW * GEMM_KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    // 对于通用情况使用行主序6x16微内核
    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        
        for (int pp = 0; pp < k; pp += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - pp);
            bool first = (pp == 0);
            
            // 使用通用步幅打包B矩阵
            pack_b_generic(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                
                // 使用通用步幅打包A矩阵
                pack_a_generic(A + ii * rsa + pp * csa, buf_a.data, mc, kc, rsa, csa);
                
                // 使用SIMD微内核计算
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        
                        float* packed_a = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* packed_b = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;
                        
                        // 当C是行主序连续时使用优化的SIMD内核
                        if (c_rowmajor) {
                            if (alpha == 1.0f) {
                                // 快速路径：alpha=1
                                if (first && beta == 0.0f) {
                                    kernel_6x16_row_zero(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                                } else if (first) {
                                    // 需要先将现有C乘以beta进行缩放
                                    for (int i = 0; i < mr; ++i)
                                        for (int j = 0; j < nr; ++j)
                                            C_ij[i * rsc + j] *= beta;
                                    kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                                } else {
                                    kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                                }
                            } else {
                                // 通用alpha/beta路径，使用SIMD
                                kernel_6x16_row_alphabeta(packed_a, packed_b, C_ij, mr, nr, kc, rsc, alpha, beta, first);
                            }
                        } else {
                            // 非连续C的通用存储路径
                            kernel_6x16_generic_store(packed_a, packed_b, C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                        }
                    }
                }
            }
        }
    }
}

// 便捷重载函数
inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k, 
                  float alpha = 1.0f, float beta = 0.0f) {
    // 默认：行主序布局
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k,
                   int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    // 通过sgemm路由以使用针对极端形状的特化内核
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

// 显式指定线程数的并行matmul
inline void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads = 0) {
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = g_num_threads;
    if (nthreads > 1) {
        sgemm_rowmajor_parallel(A, B, C, m, n, k, nthreads);
    } else {
        sgemm_rowmajor(A, B, C, m, n, k);
    }
#else
    // OpenMP不可用，使用串行版本
    sgemm_rowmajor(A, B, C, m, n, k);
#endif
}

// 列主序便捷函数，匹配原始sgemm.c接口
inline void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(A, B, C, m, n, k);
}

} // namespace yt::kernel::gemm

#endif // YT_USE_AVX2
