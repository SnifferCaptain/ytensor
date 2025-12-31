#pragma once
/***************
 * @file gemm.hpp
 * @brief 现代C++行主序GEMM实现 (基于 https://salykova.github.io/matmul-cpu)
 * @author SnifferCaptain
 * @date 2025-12-31
 * 
 * 实现 BLAS 风格的 GEMM: C = alpha * A @ B + beta * C (inplace)
 * 支持任意stride输入，自动判断主序选择合适内核
 ***************/

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>

namespace yt::kernel {

namespace detail {

// ============================================================================
// 配置参数
// ============================================================================

constexpr int MR = 6;    // 微内核行数
constexpr int NR = 16;   // 微内核列数
constexpr int MC = 72;   // A块行数
constexpr int KC = 256;  // K方向块大小
constexpr int NC = 4080; // B块列数

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
    void ensure(size_t n) {
        if (n > capacity) {
            if (data) aligned_free_64(data);
            data = static_cast<float*>(aligned_alloc_64(n * sizeof(float)));
            capacity = n;
        }
    }
    ~AlignedBuffer() { if (data) aligned_free_64(data); }
};

// ============================================================================
// 打包函数 - 优化版
// ============================================================================

// A打包: 连续行优化
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

// A打包: 通用stride
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

// B打包: 连续行优化
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

// B打包: 通用stride
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
// 微内核
// ============================================================================

// 完整6x16微内核 - C行连续
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
    
    #define WRITE(i) { \
        float* row = C + i * ldc; \
        if (first && beta == 0.0f) { \
            _mm256_storeu_ps(row, _mm256_mul_ps(c##i##0, av)); \
            _mm256_storeu_ps(row+8, _mm256_mul_ps(c##i##1, av)); \
        } else if (first) { \
            _mm256_storeu_ps(row, _mm256_fmadd_ps(c##i##0, av, _mm256_mul_ps(_mm256_loadu_ps(row), bv))); \
            _mm256_storeu_ps(row+8, _mm256_fmadd_ps(c##i##1, av, _mm256_mul_ps(_mm256_loadu_ps(row+8), bv))); \
        } else { \
            _mm256_storeu_ps(row, _mm256_fmadd_ps(c##i##0, av, _mm256_loadu_ps(row))); \
            _mm256_storeu_ps(row+8, _mm256_fmadd_ps(c##i##1, av, _mm256_loadu_ps(row+8))); \
        } \
    }
    WRITE(0); WRITE(1); WRITE(2); WRITE(3); WRITE(4); WRITE(5);
    #undef WRITE
}

// 边界微内核
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
// 主GEMM函数
// ============================================================================

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
    
    // 检测是否是行主序连续
    bool a_row = (csa == 1);
    bool b_row = (csb == 1);
    bool c_row = (csc == 1);
    
    static thread_local AlignedBuffer buf_a, buf_b;
    buf_a.ensure(MC * KC);
    buf_b.ensure(KC * NC);
    
    for (int j = 0; j < n; j += NC) {
        int nc = std::min(NC, n - j);
        
        for (int p = 0; p < k; p += KC) {
            int kc = std::min(KC, k - p);
            bool first = (p == 0);
            
            // 打包B
            if (b_row)
                pack_b_row_major(B + p * rsb + j, buf_b.data, kc, nc, rsb);
            else
                pack_b_generic(B + p * rsb + j * csb, buf_b.data, kc, nc, rsb, csb);
            
            for (int i = 0; i < m; i += MC) {
                int mc = std::min(MC, m - i);
                
                // 打包A
                if (a_row)
                    pack_a_row_major(A + i * rsa + p, buf_a.data, mc, kc, rsa);
                else
                    pack_a_generic(A + i * rsa + p * csa, buf_a.data, mc, kc, rsa, csa);
                
                // 计算
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
