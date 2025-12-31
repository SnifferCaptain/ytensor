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
 ***************/

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>

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
constexpr int MC = 642;  // Multiple of MR_ROW=6 and MR_COL=16: lcm(6,16)=48, use 642=107*6
constexpr int KC = 500;
constexpr int NC = 4800; // Multiple of NR_ROW=16 and NR_COL=6: lcm(16,6)=48, use 4800=300*16=800*6

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

// ============================================================================
// Column-major GEMM (uses 16x6 kernel)
// C[m x n] = A[m x k] @ B[k x n], all column-major
// ============================================================================

inline void sgemm_colmajor(float* A, float* B, float* C, int m, int n, int k) {
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

inline void sgemm_rowmajor(float* A, float* B, float* C, int m, int n, int k) {
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
    
    // Detect layout: row-major has csx=1, column-major has rsx=1
    bool a_rowmajor = (csa == 1 && rsa >= k);
    bool a_colmajor = (rsa == 1 && csa >= m);
    bool b_rowmajor = (csb == 1 && rsb >= n);
    bool b_colmajor = (rsb == 1 && csb >= k);
    bool c_rowmajor = (csc == 1 && rsc >= n);
    bool c_colmajor = (rsc == 1 && csc >= m);
    
    // Best case: all matrices have same layout, use optimized path
    if (a_colmajor && b_colmajor && c_colmajor && alpha == 1.0f && beta == 0.0f) {
        sgemm_colmajor(const_cast<float*>(A), const_cast<float*>(B), C, m, n, k);
        return;
    }
    
    if (a_rowmajor && b_rowmajor && c_rowmajor && alpha == 1.0f && beta == 0.0f) {
        sgemm_rowmajor(const_cast<float*>(A), const_cast<float*>(B), C, m, n, k);
        return;
    }
    
    // Generic fallback with stride support
    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_buf_size = (static_cast<size_t>(MC + MR_ROW - 1) / MR_ROW) * MR_ROW * KC;
    size_t b_buf_size = (static_cast<size_t>(NC + NR_ROW - 1) / NR_ROW) * NR_ROW * KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);
    
    // Use row-major kernel for generic case
    for (int jj = 0; jj < n; jj += NC) {
        int nc = std::min(NC, n - jj);
        
        for (int pp = 0; pp < k; pp += KC) {
            int kc = std::min(KC, k - pp);
            bool first = (pp == 0);
            
            // Pack B with generic strides
            float* bp = buf_b.data;
            for (int j = 0; j < nc; j += NR_ROW) {
                int nr = std::min(NR_ROW, nc - j);
                for (int p = 0; p < kc; ++p) {
                    for (int jjj = 0; jjj < nr; ++jjj) {
                        bp[jjj] = B[(pp + p) * rsb + (jj + j + jjj) * csb];
                    }
                    for (int jjj = nr; jjj < NR_ROW; ++jjj) {
                        bp[jjj] = 0.0f;
                    }
                    bp += NR_ROW;
                }
            }
            
            for (int ii = 0; ii < m; ii += MC) {
                int mc = std::min(MC, m - ii);
                
                // Pack A with generic strides
                float* ap = buf_a.data;
                for (int i = 0; i < mc; i += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - i);
                    for (int p = 0; p < kc; ++p) {
                        for (int iii = 0; iii < mr; ++iii) {
                            ap[iii] = A[(ii + i + iii) * rsa + (pp + p) * csa];
                        }
                        for (int iii = mr; iii < MR_ROW; ++iii) {
                            ap[iii] = 0.0f;
                        }
                        ap += MR_ROW;
                    }
                }
                
                // Compute
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        
                        // Compute micro-kernel result
                        float acc[MR_ROW][NR_ROW] = {};
                        const float* ap2 = buf_a.data + ir * kc;
                        const float* bp2 = buf_b.data + jr * kc;
                        
                        for (int p = 0; p < kc; ++p) {
                            for (int i = 0; i < mr; ++i) {
                                float a_val = ap2[i];
                                for (int j = 0; j < nr; ++j) {
                                    acc[i][j] += a_val * bp2[j];
                                }
                            }
                            ap2 += MR_ROW;
                            bp2 += NR_ROW;
                        }
                        
                        // Store with alpha/beta scaling
                        for (int i = 0; i < mr; ++i) {
                            for (int j = 0; j < nr; ++j) {
                                float* c_ptr = C + (ii + ir + i) * rsc + (jj + jr + j) * csc;
                                if (first) {
                                    *c_ptr = alpha * acc[i][j] + beta * (*c_ptr);
                                } else {
                                    *c_ptr += alpha * acc[i][j];
                                }
                            }
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
    // Row-major: rsa=k, csa=1, rsb=n, csb=1, rsc=n, csc=1
    sgemm_rowmajor(const_cast<float*>(A), const_cast<float*>(B), C, m, n, k);
}

// Column-major convenience function (matches original sgemm.c interface)
inline void matmul_colmajor(float* A, float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(A, B, C, m, n, k);
}

} // namespace yt::kernel::gemm
