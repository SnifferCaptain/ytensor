#pragma once
/***************
 * @file gemm.hpp
 * @brief Row-major GEMM implementation (based on https://salykova.github.io/matmul-cpu)
 * @author SnifferCaptain
 * @date 2025-12-31
 * 
 * BLAS-style GEMM: C = alpha * A @ B + beta * C (inplace)
 * Supports arbitrary stride inputs with automatic kernel selection
 ***************/

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>

namespace yt::kernel {

namespace detail {

// ============================================================================
// Configuration
// ============================================================================

constexpr int MR = 6;    // Micro-kernel row count
constexpr int NR = 16;   // Micro-kernel column count
constexpr int MC = 72;   // A block row count
constexpr int KC = 256;  // K dimension block size
constexpr int NC = 4080; // B block column count

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
    
    void ensure(size_t n) {
        if (n > capacity) {
            float* new_data = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
            if (new_data) {
                if (data) aligned_free_64(data);
                data = new_data;
                capacity = n;
            }
            // If allocation fails, keep existing buffer (or nullptr)
        }
    }
    
    ~AlignedBuffer() { 
        if (data) aligned_free_64(data); 
    }
};

// ============================================================================
// Packing Functions
// ============================================================================

// Pack A: row-major optimized
inline void pack_a_row_major(const float* A, float* packed, int mc, int kc, int64_t lda) {
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = A[(i + ii) * lda + p];
            }
            for (int ii = mr; ii < MR; ++ii) {
                packed[ii] = 0.0f;
            }
            packed += MR;
        }
    }
}

// Pack A: generic stride
inline void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rs, int64_t cs) {
    for (int i = 0; i < mc; i += MR) {
        int mr = std::min(MR, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = A[(i + ii) * rs + p * cs];
            }
            for (int ii = mr; ii < MR; ++ii) {
                packed[ii] = 0.0f;
            }
            packed += MR;
        }
    }
}

// Pack B: row-major optimized
inline void pack_b_row_major(const float* B, float* packed, int kc, int nc, int64_t ldb) {
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = B[p * ldb + j + jj];
            }
            for (int jj = nr; jj < NR; ++jj) {
                packed[jj] = 0.0f;
            }
            packed += NR;
        }
    }
}

// Pack B: generic stride
inline void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rs, int64_t cs) {
    for (int j = 0; j < nc; j += NR) {
        int nr = std::min(NR, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = B[p * rs + (j + jj) * cs];
            }
            for (int jj = nr; jj < NR; ++jj) {
                packed[jj] = 0.0f;
            }
            packed += NR;
        }
    }
}

// ============================================================================
// Micro-kernels
// ============================================================================

// Full 6x16 micro-kernel for row-contiguous C
inline void kernel_6x16(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int kc, int64_t ldc,
    float alpha, float beta, bool first
) {
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
        
        A += MR; B += NR;
    }
    
    __m256 av = _mm256_broadcast_ss(&alpha);
    __m256 bv = _mm256_broadcast_ss(&beta);
    
    // Store results to C with alpha/beta scaling
    // Using a lambda for cleaner code while maintaining performance
    auto store_row = [&](float* row, __m256 acc0, __m256 acc1) {
        if (first && beta == 0.0f) {
            _mm256_storeu_ps(row, _mm256_mul_ps(acc0, av));
            _mm256_storeu_ps(row + 8, _mm256_mul_ps(acc1, av));
        } else if (first) {
            _mm256_storeu_ps(row, _mm256_fmadd_ps(acc0, av, _mm256_mul_ps(_mm256_loadu_ps(row), bv)));
            _mm256_storeu_ps(row + 8, _mm256_fmadd_ps(acc1, av, _mm256_mul_ps(_mm256_loadu_ps(row + 8), bv)));
        } else {
            _mm256_storeu_ps(row, _mm256_fmadd_ps(acc0, av, _mm256_loadu_ps(row)));
            _mm256_storeu_ps(row + 8, _mm256_fmadd_ps(acc1, av, _mm256_loadu_ps(row + 8)));
        }
    };
    
    store_row(C + 0 * ldc, c00, c01);
    store_row(C + 1 * ldc, c10, c11);
    store_row(C + 2 * ldc, c20, c21);
    store_row(C + 3 * ldc, c30, c31);
    store_row(C + 4 * ldc, c40, c41);
    store_row(C + 5 * ldc, c50, c51);
}

// Edge case micro-kernel for boundary handling
inline void kernel_edge(
    const float* A, const float* B, float* C,
    int mr, int nr, int kc,
    int64_t rsc, int64_t csc,
    float alpha, float beta, bool first
) {
    float acc[MR][NR] = {};
    
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) {
            float a = A[i];
            for (int j = 0; j < nr; ++j) {
                acc[i][j] += a * B[j];
            }
        }
        A += MR; B += NR;
    }
    
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float* c = C + i * rsc + j * csc;
            *c = first ? (alpha * acc[i][j] + beta * *c) : (*c + alpha * acc[i][j]);
        }
    }
}

} // namespace detail

// ============================================================================
// Main GEMM Function
// ============================================================================

/**
 * @brief BLAS-style single precision matrix multiplication
 * @param A Input matrix A pointer
 * @param B Input matrix B pointer
 * @param C Output matrix C pointer (modified in place)
 * @param m Number of rows of A and C
 * @param n Number of columns of B and C
 * @param k Number of columns of A and rows of B
 * @param alpha Scalar multiplier for A*B
 * @param beta Scalar multiplier for existing C values
 * @param rsa Row stride of A
 * @param csa Column stride of A
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
    using namespace detail;
    
    if (m == 0 || n == 0) return;
    
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
    
    // Detect row-major layout
    bool a_row = (csa == 1);
    bool b_row = (csb == 1);
    bool c_row = (csc == 1);
    
    // Thread-local buffers for packing (avoid repeated allocation)
    static thread_local AlignedBuffer buf_a, buf_b;
    buf_a.ensure(MC * KC);
    buf_b.ensure(KC * NC);
    
    for (int j = 0; j < n; j += NC) {
        int nc = std::min(NC, n - j);
        
        for (int p = 0; p < k; p += KC) {
            int kc = std::min(KC, k - p);
            bool first = (p == 0);
            
            // Pack B
            if (b_row)
                pack_b_row_major(B + p * rsb + j, buf_b.data, kc, nc, rsb);
            else
                pack_b_generic(B + p * rsb + j * csb, buf_b.data, kc, nc, rsb, csb);
            
            for (int i = 0; i < m; i += MC) {
                int mc = std::min(MC, m - i);
                
                // Pack A
                if (a_row)
                    pack_a_row_major(A + i * rsa + p, buf_a.data, mc, kc, rsa);
                else
                    pack_a_generic(A + i * rsa + p * csa, buf_a.data, mc, kc, rsa, csa);
                
                // Compute micro-kernel tiles
                for (int ir = 0; ir < mc; ir += MR) {
                    int mr = std::min(MR, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR) {
                        int nr = std::min(NR, nc - jr);
                        
                        float* C_ij = C + (i + ir) * rsc + (j + jr) * csc;
                        
                        if (mr == MR && nr == NR && c_row) {
                            kernel_6x16(
                                buf_a.data + ir * kc,
                                buf_b.data + jr * kc,
                                C_ij, kc, rsc, alpha, beta, first);
                        } else {
                            kernel_edge(
                                buf_a.data + ir * kc,
                                buf_b.data + jr * kc,
                                C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                        }
                    }
                }
            }
        }
    }
}

inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) {
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k,
                   int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f);
}

} // namespace yt::kernel
