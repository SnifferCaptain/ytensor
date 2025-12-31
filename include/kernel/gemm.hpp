#pragma once
/***************
 * @file gemm.hpp
 * @brief High-performance GEMM with automatic layout detection
 * @author SnifferCaptain
 * @date 2025-12-31
 * 
 * BLAS-style GEMM: C = alpha * A @ B + beta * C (inplace)
 * Automatically detects row/column major and selects optimal kernel
 * Based on: https://salykova.github.io/matmul-cpu
 * 
 * Features:
 * - Dual micro-kernel: 16x6 (column-major) and 6x16 (row-major)
 * - Automatic layout detection and kernel selection
 * - OpenMP parallel support (define GEMM_NTHREADS or use set_num_threads())
 * - Specialized kernels for dot product and outer product
 * - Arbitrary stride support
 ***************/

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yt::kernel::gemm {

// ============================================================================
// Configuration
// ============================================================================

// Row-major micro-kernel: 6 rows x 16 columns
constexpr int MR_ROW = 6;
constexpr int NR_ROW = 16;

// Column-major micro-kernel: 16 rows x 6 columns (original sgemm.c)
constexpr int MR_COL = 16;
constexpr int NR_COL = 6;

// Block sizes (must be multiples of micro-kernel sizes for proper alignment)
// MC should be divisible by both MR_ROW(6) and MR_COL(16) ideally, or at least reasonable
// 642 = 107 * 6 = 40.125 * 16, chosen to balance cache usage and alignment
constexpr int MC = 642;
constexpr int KC = 500;
// NC should be divisible by both NR_ROW(16) and NR_COL(6): 4800 = 300*16 = 800*6
constexpr int NC = 4800;

// Thread configuration
#ifndef GEMM_NTHREADS
#define GEMM_NTHREADS 1
#endif

inline int g_num_threads = GEMM_NTHREADS;

inline void set_num_threads(int n) {
    g_num_threads = std::max(1, n);
#ifdef _OPENMP
    omp_set_num_threads(g_num_threads);
#endif
}

inline int get_num_threads() {
    return g_num_threads;
}

// ============================================================================
// Memory Alignment
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

// Mask table for boundary handling
alignas(64) static const int8_t mask_table[32] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

inline void build_masks(__m256i* mask0, __m256i* mask1, int nr) {
    *mask0 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - nr]));
    *mask1 = _mm256_cvtepi8_epi32(_mm_loadu_si64(&mask_table[16 - nr + 8]));
}

// ============================================================================
// Column-major kernel (16x6) - from original sgemm.c
// This is the highly optimized kernel for column-major layout
// ============================================================================

// FMA loops for different nr values (column-major)
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

// Column-major 16x6 kernel with zero init
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
    
    // Store results (column-major: C has ldc rows per column)
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

// Column-major 16x6 kernel with load accum
inline void kernel_16x6_col_load(float* A, float* B, float* C, int mr, int nr, int kc, int ldc) {
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    __m256i mask0, mask1;
    
    // Load existing C values
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
    
    // Store results
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
// Row-major kernel (6x16) - transpose of column-major
// ============================================================================

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
    
    fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc);
    
    // Store results (row-major: C has ldc columns per row)
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
    
    // Load existing C values
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
    
    fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc);
    
    // Store results
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

// ============================================================================
// Packing Functions
// ============================================================================

// Pack A for column-major kernel (16xkc panels)
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

// Pack B for column-major kernel (kcx6 panels)
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

// Pack A for row-major kernel (6xkc panels)
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

// Pack A with generic strides (6xkc panels for row-major kernel)
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

// Pack B for row-major kernel (kcx16 panels)
inline void pack_b_row(const float* B, float* packed, int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = B[p * ldb + j + jj];
            }
            for (int jj = nr; jj < NR_ROW; ++jj) {
                packed[jj] = 0.0f;
            }
            packed += NR_ROW;
        }
    }
}

// Pack B with generic strides (kcx16 panels for row-major kernel)
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

// 6x16 kernel that stores to generic stride C (slower but flexible)
inline void kernel_6x16_generic_store(float* A, float* B, float* C, 
                                      int mr, int nr, int kc, 
                                      int64_t rsc, int64_t csc,
                                      float alpha, float beta, bool first) {
    // Compute in registers
    float acc[MR_ROW][NR_ROW] = {};
    
    // Use SIMD for compute (same as FMA loop but store to local array)
    __m256 c00={}, c01={}, c10={}, c11={}, c20={}, c21={};
    __m256 c30={}, c31={}, c40={}, c41={}, c50={}, c51={};
    
    fma_loop_row_6x16(A, B, c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, kc);
    
    // Extract to local array
    alignas(32) float tmp[MR_ROW][NR_ROW];
    _mm256_storeu_ps(&tmp[0][0], c00); _mm256_storeu_ps(&tmp[0][8], c01);
    _mm256_storeu_ps(&tmp[1][0], c10); _mm256_storeu_ps(&tmp[1][8], c11);
    _mm256_storeu_ps(&tmp[2][0], c20); _mm256_storeu_ps(&tmp[2][8], c21);
    _mm256_storeu_ps(&tmp[3][0], c30); _mm256_storeu_ps(&tmp[3][8], c31);
    _mm256_storeu_ps(&tmp[4][0], c40); _mm256_storeu_ps(&tmp[4][8], c41);
    _mm256_storeu_ps(&tmp[5][0], c50); _mm256_storeu_ps(&tmp[5][8], c51);
    
    // Store with generic strides
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
// Specialized kernels for extreme shapes
// High-performance implementations for edge cases: dot product, outer product,
// row vector * matrix (gemv_row), matrix * column vector (gemv_col)
// ============================================================================

// Helper: horizontal sum of __m256
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

// Dot product: C[1x1] = A[1xk] @ B[kx1] - SIMD optimized with 4x unrolling
inline float dot_product_simd(const float* A, const float* B, int k, 
                              int64_t csa, int64_t rsb) {
    float sum = 0.0f;
    
    // Fast path: both contiguous
    if (csa == 1 && rsb == 1) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        
        int p = 0;
        // 4x unrolled loop
        for (; p + 32 <= k; p += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(B + p), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 8), _mm256_loadu_ps(B + p + 8), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 16), _mm256_loadu_ps(B + p + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p + 24), _mm256_loadu_ps(B + p + 24), acc3);
        }
        // Combine accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        
        // Handle remaining 8-element chunks
        for (; p + 8 <= k; p += 8) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(A + p), _mm256_loadu_ps(B + p), acc0);
        }
        
        sum = hsum_ps_avx(acc0);
        // Tail
        for (; p < k; ++p) sum += A[p] * B[p];
    } else {
        // Strided case - still vectorize where possible
        for (int p = 0; p < k; ++p) {
            sum += A[p * csa] * B[p * rsb];
        }
    }
    return sum;
}

// Outer product: C[mxn] = A[mx1] @ B[1xn] - optimized for SIMD stores
// Process multiple rows at a time for better cache utilization
inline void outer_product_simd(const float* A, const float* B, float* C,
                               int m, int n, float alpha, float beta,
                               int64_t rsa, int64_t csb, int64_t rsc, int64_t csc) {
    // Fast path: C is row-major contiguous and B is contiguous
    if (csc == 1 && rsc >= n && csb == 1) {
        // Process 4 rows at a time
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
            // Tail
            for (; j < n; ++j) {
                c0[j] = alpha * A[i * rsa] * B[j] + beta * c0[j];
                c1[j] = alpha * A[(i + 1) * rsa] * B[j] + beta * c1[j];
                c2[j] = alpha * A[(i + 2) * rsa] * B[j] + beta * c2[j];
                c3[j] = alpha * A[(i + 3) * rsa] * B[j] + beta * c3[j];
            }
        }
        // Remaining rows
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
        // C row-major but B strided
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * A[i * rsa];
            float* c_row = C + i * rsc;
            for (int j = 0; j < n; ++j) {
                c_row[j] = a_val * B[j * csb] + beta * c_row[j];
            }
        }
    } else {
        // Generic strided case
        for (int i = 0; i < m; ++i) {
            float a_val = alpha * A[i * rsa];
            for (int j = 0; j < n; ++j) {
                float* c_ptr = C + i * rsc + j * csc;
                *c_ptr = a_val * B[j * csb] + beta * (*c_ptr);
            }
        }
    }
}

// C[1xn] = A[1xk] @ B[kxn] - optimized row vector * matrix
// Tiled implementation for better cache utilization
inline void gemv_row_simd(const float* A, const float* B, float* C,
                          int n, int k, float alpha, float beta,
                          int64_t csa, int64_t rsb, int64_t csb, int64_t csc) {
    // Fast path: A contiguous in k, B row-major (rsb = n, csb = 1), C contiguous
    if (csa == 1 && csb == 1 && rsb == n && csc == 1) {
        // Tile sizes for better cache utilization
        constexpr int KB = 256;  // k tile size (fits in L1/L2)
        constexpr int NB = 512;  // n tile size
        
        // Initialize C if beta != 0
        if (beta != 0.0f && beta != 1.0f) {
            __m256 vbeta = _mm256_set1_ps(beta);
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 vc = _mm256_loadu_ps(C + j);
                _mm256_storeu_ps(C + j, _mm256_mul_ps(vbeta, vc));
            }
            for (; j < n; ++j) C[j] *= beta;
        } else if (beta == 0.0f) {
            memset(C, 0, n * sizeof(float));
        }
        
        // Tiled computation
        for (int pb = 0; pb < k; pb += KB) {
            int kc = std::min(KB, k - pb);
            
            for (int jb = 0; jb < n; jb += NB) {
                int nc = std::min(NB, n - jb);
                
                // Process 32 columns at a time within tile
                int j = 0;
                for (; j + 32 <= nc; j += 32) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    
                    for (int p = 0; p < kc; ++p) {
                        __m256 va = _mm256_broadcast_ss(A + pb + p);
                        const float* b_row = B + (pb + p) * n + jb + j;
                        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row), acc0);
                        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + 8), acc1);
                        acc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + 16), acc2);
                        acc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + 24), acc3);
                    }
                    
                    // Accumulate to C (alpha scaling)
                    __m256 valpha = _mm256_set1_ps(alpha);
                    float* c_ptr = C + jb + j;
                    _mm256_storeu_ps(c_ptr, _mm256_fmadd_ps(valpha, acc0, _mm256_loadu_ps(c_ptr)));
                    _mm256_storeu_ps(c_ptr + 8, _mm256_fmadd_ps(valpha, acc1, _mm256_loadu_ps(c_ptr + 8)));
                    _mm256_storeu_ps(c_ptr + 16, _mm256_fmadd_ps(valpha, acc2, _mm256_loadu_ps(c_ptr + 16)));
                    _mm256_storeu_ps(c_ptr + 24, _mm256_fmadd_ps(valpha, acc3, _mm256_loadu_ps(c_ptr + 24)));
                }
                
                // Process remaining 16 columns
                for (; j + 16 <= nc; j += 16) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    
                    for (int p = 0; p < kc; ++p) {
                        __m256 va = _mm256_broadcast_ss(A + pb + p);
                        const float* b_row = B + (pb + p) * n + jb + j;
                        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row), acc0);
                        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + 8), acc1);
                    }
                    
                    __m256 valpha = _mm256_set1_ps(alpha);
                    float* c_ptr = C + jb + j;
                    _mm256_storeu_ps(c_ptr, _mm256_fmadd_ps(valpha, acc0, _mm256_loadu_ps(c_ptr)));
                    _mm256_storeu_ps(c_ptr + 8, _mm256_fmadd_ps(valpha, acc1, _mm256_loadu_ps(c_ptr + 8)));
                }
                
                // Remaining columns
                for (; j < nc; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < kc; ++p) {
                        sum += A[pb + p] * B[(pb + p) * n + jb + j];
                    }
                    C[jb + j] += alpha * sum;
                }
            }
        }
    } else {
        // Generic case: process each column with dot product
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

// C[mx1] = A[mxk] @ B[kx1] - optimized matrix * column vector
// Process 8 rows at a time for better memory access patterns
inline void gemv_col_simd(const float* A, const float* B, float* C,
                          int m, int k, float alpha, float beta,
                          int64_t rsa, int64_t csa, int64_t rsb, int64_t rsc) {
    // Fast path: A row-major (csa=1), B contiguous, C contiguous
    if (csa == 1 && rsb == 1 && rsc == 1) {
        // Process 8 rows at a time
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
            
            // Process k dimension in chunks of 8
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
            
            // Horizontal sums
            float sum0 = hsum_ps_avx(acc0);
            float sum1 = hsum_ps_avx(acc1);
            float sum2 = hsum_ps_avx(acc2);
            float sum3 = hsum_ps_avx(acc3);
            float sum4 = hsum_ps_avx(acc4);
            float sum5 = hsum_ps_avx(acc5);
            float sum6 = hsum_ps_avx(acc6);
            float sum7 = hsum_ps_avx(acc7);
            
            // Handle tail
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
            
            // Store results
            C[i] = alpha * sum0 + beta * C[i];
            C[i + 1] = alpha * sum1 + beta * C[i + 1];
            C[i + 2] = alpha * sum2 + beta * C[i + 2];
            C[i + 3] = alpha * sum3 + beta * C[i + 3];
            C[i + 4] = alpha * sum4 + beta * C[i + 4];
            C[i + 5] = alpha * sum5 + beta * C[i + 5];
            C[i + 6] = alpha * sum6 + beta * C[i + 6];
            C[i + 7] = alpha * sum7 + beta * C[i + 7];
        }
        
        // Remaining rows
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
        // Generic case
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
// Column-major GEMM (uses 16x6 kernel)
// C[m x n] = A[m x k] @ B[k x n], all column-major
// ============================================================================

inline void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    static thread_local AlignedBuffer buf_a, buf_b;
    // Allocate buffers with proper padding for alignment
    // For col-major: A panels are MR_COL x kc, B panels are kc x NR_COL
    size_t a_buf_size = (static_cast<size_t>(MC + MR_COL - 1) / MR_COL) * MR_COL * KC;
    size_t b_buf_size = (static_cast<size_t>(NC + NR_COL - 1) / NR_COL) * NR_COL * KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    for (int j = 0; j < n; j += NC) {
        int nc = std::min(NC, n - j);
        
        for (int p = 0; p < k; p += KC) {
            int kc = std::min(KC, k - p);
            bool first = (p == 0);
            
            pack_b_col(B + j * k + p, buf_b.data, kc, nc, k);
            
            for (int i = 0; i < m; i += MC) {
                int mc = std::min(MC, m - i);
                
                pack_a_col(A + p * m + i, buf_a.data, mc, kc, m);
                
                for (int jr = 0; jr < nc; jr += NR_COL) {
                    int nr = std::min(NR_COL, nc - jr);
                    for (int ir = 0; ir < mc; ir += MR_COL) {
                        int mr = std::min(MR_COL, mc - ir);
                        
                        float* C_ij = C + (j + jr) * m + (i + ir);
                        // Packed A: each panel is MR_COL x kc
                        // Packed B: each panel is kc x NR_COL
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
// Row-major GEMM (uses 6x16 kernel)
// C[m x n] = A[m x k] @ B[k x n], all row-major
// ============================================================================

inline void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    static thread_local AlignedBuffer buf_a, buf_b;
    // Allocate buffers with proper padding for alignment
    // For row-major: A panels are MR_ROW x kc, B panels are kc x NR_ROW
    size_t a_buf_size = (static_cast<size_t>(MC + MR_ROW - 1) / MR_ROW) * MR_ROW * KC;
    size_t b_buf_size = (static_cast<size_t>(NC + NR_ROW - 1) / NR_ROW) * NR_ROW * KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    for (int j = 0; j < n; j += NC) {
        int nc = std::min(NC, n - j);
        
        for (int p = 0; p < k; p += KC) {
            int kc = std::min(KC, k - p);
            bool first = (p == 0);
            
            pack_b_row(B + p * n + j, buf_b.data, kc, nc, n);
            
            for (int i = 0; i < m; i += MC) {
                int mc = std::min(MC, m - i);
                
                pack_a_row(A + i * k + p, buf_a.data, mc, kc, k);
                
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        
                        float* C_ij = C + (i + ir) * n + (j + jr);
                        // Packed A: each panel is MR_ROW x kc, stored as kc x MR_ROW
                        // Packed B: each panel is kc x NR_ROW
                        float* packed_a = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* packed_b = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        
                        if (first) {
                            kernel_6x16_row_zero(packed_a, packed_b, C_ij, mr, nr, kc, n);
                        } else {
                            kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, n);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Parallel row-major GEMM with OpenMP
// ============================================================================

#ifdef _OPENMP
inline void sgemm_rowmajor_parallel(const float* A, const float* B, float* C, int m, int n, int k, int nthreads) {
    // For parallel execution, we need shared B buffer and per-thread A buffer
    size_t b_buf_size = (static_cast<size_t>(NC + NR_ROW - 1) / NR_ROW) * NR_ROW * KC;
    float* buf_b_shared = static_cast<float*>(aligned_alloc_64(b_buf_size * sizeof(float)));
    if (!buf_b_shared) {
        // Fallback to sequential
        sgemm_rowmajor(A, B, C, m, n, k);
        return;
    }
    
    for (int j = 0; j < n; j += NC) {
        int nc = std::min(NC, n - j);
        
        for (int p = 0; p < k; p += KC) {
            int kc = std::min(KC, k - p);
            bool first = (p == 0);
            
            // Pack B once (shared)
            pack_b_row(B + p * n + j, buf_b_shared, kc, nc, n);
            
            // Parallelize over rows
            #pragma omp parallel num_threads(nthreads)
            {
                // Each thread has its own A buffer
                static thread_local AlignedBuffer buf_a_local;
                size_t a_buf_size = (static_cast<size_t>(MC + MR_ROW - 1) / MR_ROW) * MR_ROW * KC;
                buf_a_local.ensure(a_buf_size);
                
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < m; i += MC) {
                    int mc = std::min(MC, m - i);
                    
                    pack_a_row(A + i * k + p, buf_a_local.data, mc, kc, k);
                    
                    for (int ir = 0; ir < mc; ir += MR_ROW) {
                        int mr = std::min(MR_ROW, mc - ir);
                        for (int jr = 0; jr < nc; jr += NR_ROW) {
                            int nr = std::min(NR_ROW, nc - jr);
                            
                            float* C_ij = C + (i + ir) * n + (j + jr);
                            float* packed_a = buf_a_local.data + (ir / MR_ROW) * MR_ROW * kc;
                            float* packed_b = buf_b_shared + (jr / NR_ROW) * NR_ROW * kc;
                            
                            if (first) {
                                kernel_6x16_row_zero(packed_a, packed_b, C_ij, mr, nr, kc, n);
                            } else {
                                kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, n);
                            }
                        }
                    }
                }
            }
        }
    }
    
    aligned_free_64(buf_b_shared);
}
#endif

// ============================================================================
// Generic GEMM with stride support
// Automatically detects layout and chooses optimal kernel
// ============================================================================

/**
 * @brief BLAS-style single precision matrix multiplication
 * C = alpha * A @ B + beta * C
 * 
 * @param A Input matrix A pointer
 * @param B Input matrix B pointer  
 * @param C Output matrix C pointer (modified in place)
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for existing C values
 * @param rsa Row stride of A (distance between consecutive rows)
 * @param csa Column stride of A (distance between consecutive columns)
 * @param rsb Row stride of B
 * @param csb Column stride of B
 * @param rsc Row stride of C
 * @param csc Column stride of C
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
    
    // Handle special cases
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
    // Specialized kernels for extreme shapes
    // =========================================================================
    
    // Dot product: m=1, n=1 -> scalar result
    if (m == 1 && n == 1) {
        float result = dot_product_simd(A, B, k, csa, rsb);
        *C = alpha * result + beta * (*C);
        return;
    }
    
    // Outer product: k=1 -> rank-1 update
    if (k == 1) {
        outer_product_simd(A, B, C, m, n, alpha, beta, rsa, csb, rsc, csc);
        return;
    }
    
    // Row vector times matrix: m=1
    if (m == 1) {
        gemv_row_simd(A, B, C, n, k, alpha, beta, csa, rsb, csb, csc);
        return;
    }
    
    // Matrix times column vector: n=1
    if (n == 1) {
        gemv_col_simd(A, B, C, m, k, alpha, beta, rsa, csa, rsb, rsc);
        return;
    }
    
    // =========================================================================
    // Standard GEMM paths
    // =========================================================================
    
    // Detect layout: row-major has csx=1, column-major has rsx=1
    // For optimized paths, we need EXACTLY contiguous layout (stride == dimension)
    bool a_rowmajor_exact = (csa == 1 && rsa == k);  // A is m x k, row-major, no padding
    bool a_colmajor_exact = (rsa == 1 && csa == m);  // A is m x k, column-major, no padding
    bool b_rowmajor_exact = (csb == 1 && rsb == n);  // B is k x n, row-major, no padding
    bool b_colmajor_exact = (rsb == 1 && csb == k);  // B is k x n, column-major, no padding
    bool c_rowmajor_exact = (csc == 1 && rsc == n);  // C is m x n, row-major, no padding
    bool c_colmajor_exact = (rsc == 1 && csc == m);  // C is m x n, column-major, no padding
    
    // For generic path, check if C is at least row-major (even with padding)
    bool c_rowmajor = (csc == 1 && rsc >= n);
    
    // Best case: all matrices have EXACT layout, use optimized path
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
    
    // Generic fallback with stride support - uses SIMD micro-kernel
    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_buf_size = (static_cast<size_t>(MC + MR_ROW - 1) / MR_ROW) * MR_ROW * KC;
    size_t b_buf_size = (static_cast<size_t>(NC + NR_ROW - 1) / NR_ROW) * NR_ROW * KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    // Use row-major 6x16 kernel for generic case
    for (int jj = 0; jj < n; jj += NC) {
        int nc = std::min(NC, n - jj);
        
        for (int pp = 0; pp < k; pp += KC) {
            int kc = std::min(KC, k - pp);
            bool first = (pp == 0);
            
            // Pack B with generic strides
            pack_b_generic(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            
            for (int ii = 0; ii < m; ii += MC) {
                int mc = std::min(MC, m - ii);
                
                // Pack A with generic strides
                pack_a_generic(A + ii * rsa + pp * csa, buf_a.data, mc, kc, rsa, csa);
                
                // Compute using SIMD micro-kernel
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        
                        float* packed_a = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* packed_b = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;
                        
                        // Use optimized SIMD kernel when C is row-major contiguous
                        if (c_rowmajor && alpha == 1.0f) {
                            if (first && beta == 0.0f) {
                                kernel_6x16_row_zero(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                            } else if (first) {
                                // Need to scale existing C by beta first
                                for (int i = 0; i < mr; ++i)
                                    for (int j = 0; j < nr; ++j)
                                        C_ij[i * rsc + j] *= beta;
                                kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                            } else {
                                kernel_6x16_row_load(packed_a, packed_b, C_ij, mr, nr, kc, rsc);
                            }
                        } else {
                            // Generic store for non-contiguous C or when alpha != 1
                            kernel_6x16_generic_store(packed_a, packed_b, C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                        }
                    }
                }
            }
        }
    }
}

// Convenience overloads
inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k, 
                  float alpha = 1.0f, float beta = 0.0f) {
    // Default: row-major layout
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k,
                   int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    // Route through sgemm to use specialized kernels for extreme shapes
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

// Parallel matmul with explicit thread count
inline void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads = 0) {
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = g_num_threads;
    if (nthreads > 1) {
        sgemm_rowmajor_parallel(A, B, C, m, n, k, nthreads);
    } else {
        sgemm_rowmajor(A, B, C, m, n, k);
    }
#else
    // OpenMP not available, use sequential version
    sgemm_rowmajor(A, B, C, m, n, k);
#endif
}

// Column-major convenience function (matches original sgemm.c interface)
inline void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(A, B, C, m, n, k);
}

} // namespace yt::kernel::gemm
