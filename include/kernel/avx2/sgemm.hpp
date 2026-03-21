#pragma once
/***************
 * @file sgemm.hpp
 * @brief 高性能SGEMM矩阵乘法实现（AVX2）
 * @author SnifferCaptain
 * @date 2026-03-10
 *
 * BLAS风格GEMM: C = alpha * A @ B + beta * C
 * 自动检测行主序/列主序并选择最优内核
 * 基于: https://salykova.github.io/matmul-cpu
 *
 * 功能特性:
 * - 模板化双微内核: 任意MR/NR（gemm_utils.hpp）
 * - 自动布局检测和内核选择
 * - OpenMP并行支持
 * - 针对点积（sdot）、外积（sger）、GEMV的特化内核
 * - 支持任意步幅
 * - 支持mask GEMM
 ***************/

#include "gemm_utils.hpp"
#include "sdot.hpp"
#include "sger.hpp"
#include "sgemv.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// 打包函数
// ============================================================================

inline void pack_a_col(const float* A, float* packed, int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR_COL) {
        int mr = std::min(MR_COL, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) packed[ii] = A[p * lda + i + ii];
            for (int ii = mr; ii < MR_COL; ++ii) packed[ii] = 0.0f;
            packed += MR_COL;
        }
    }
}

inline void pack_b_col(const float* B, float* packed, int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR_COL) {
        int nr = std::min(NR_COL, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) packed[jj] = B[(j + jj) * ldb + p];
            for (int jj = nr; jj < NR_COL; ++jj) packed[jj] = 0.0f;
            packed += NR_COL;
        }
    }
}

inline void pack_a_row(const float* A, float* packed, int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) packed[ii] = A[(i + ii) * lda + p];
            for (int ii = mr; ii < MR_ROW; ++ii) packed[ii] = 0.0f;
            packed += MR_ROW;
        }
    }
}

inline void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) packed[ii] = A[(i + ii) * rsa + p * csa];
            for (int ii = mr; ii < MR_ROW; ++ii) packed[ii] = 0.0f;
            packed += MR_ROW;
        }
    }
}

inline void pack_b_row(const float* B, float* packed, int kc, int nc, int ldb) {
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
                for (int jj = 0; jj < nr; ++jj) dest[jj] = B[p * ldb + j + jj];
                for (int jj = nr; jj < NR_ROW; ++jj) dest[jj] = 0.0f;
                dest += NR_ROW;
            }
        }
    }
}

inline void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) packed[jj] = B[p * rsb + (j + jj) * csb];
            for (int jj = nr; jj < NR_ROW; ++jj) packed[jj] = 0.0f;
            packed += NR_ROW;
        }
    }
}

// ============================================================================
// 行主序打包辅助函数（带SIMD优化和并行支持）
// ============================================================================

alignas(64) static float g_blockA_row[GEMM_MC * GEMM_KC];
alignas(64) static float g_blockB_row[GEMM_NC * GEMM_KC];

inline void pack_blockA_row_single_tile(const float* A, float* dest, int i, int mr, int kc, int lda) {
    int p = 0;
    if (mr == MR_ROW) {
        for (; p + 8 <= kc; p += 8) {
            if (p + 16 <= kc) {
                for (int ii = 0; ii < 6; ++ii)
                    _mm_prefetch((const char*)(A + (i+ii) * lda + p + 16), _MM_HINT_T0);
            }
            __m256 r[6];
            for (int ii = 0; ii < 6; ++ii) r[ii] = _mm256_loadu_ps(A + (i+ii) * lda + p);
            float* d = dest + p * MR_ROW;
            for (int pp = 0; pp < 8; ++pp) {
                for (int ii = 0; ii < 6; ++ii) d[ii] = r[ii][pp];
                d += MR_ROW;
            }
        }
    }
    for (; p < kc; ++p) {
        for (int ii = 0; ii < mr; ++ii) dest[p * MR_ROW + ii] = A[(i + ii) * lda + p];
        for (int ii = mr; ii < MR_ROW; ++ii) dest[p * MR_ROW + ii] = 0.0f;
    }
}

inline void pack_blockB_row_par(const float* B, float* packed, int nc, int kc, int ldb) {
    auto pack_tile = [&](int j) {
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
                for (int jj = 0; jj < nr; ++jj) dest[jj] = B[p * ldb + j + jj];
                for (int jj = nr; jj < NR_ROW; ++jj) dest[jj] = 0.0f;
                dest += NR_ROW;
            }
        }
    };
#ifdef _OPENMP
    if (g_num_threads > 1) {
        #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
        for (int j = 0; j < nc; j += NR_ROW) pack_tile(j);
    } else
#endif
    { for (int j = 0; j < nc; j += NR_ROW) pack_tile(j); }
}

inline void pack_blockA_row_par(const float* A, float* packed, int mc, int kc, int lda) {
    auto pack_tile = [&](int i) {
        int mr = std::min(MR_ROW, mc - i);
        pack_blockA_row_single_tile(A, packed + (i / MR_ROW) * MR_ROW * kc, i, mr, kc, lda);
    };
#ifdef _OPENMP
    if (g_num_threads > 1) {
        #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
        for (int i = 0; i < mc; i += MR_ROW) pack_tile(i);
    } else
#endif
    { for (int i = 0; i < mc; i += MR_ROW) pack_tile(i); }
}

// ============================================================================
// 列主序GEMM
// ============================================================================

inline void sgemm_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_buf_size = ((size_t)(GEMM_MC + MR_COL - 1) / MR_COL) * MR_COL * GEMM_KC;
    size_t b_buf_size = ((size_t)(GEMM_NC + NR_COL - 1) / NR_COL) * NR_COL * GEMM_KC;
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
                        float* pa = buf_a.data + (ir / MR_COL) * MR_COL * kc;
                        float* pb = buf_b.data + (jr / NR_COL) * NR_COL * kc;
                        if (first) kernel_col_zero(pa, pb, C_ij, mr, nr, kc, m);
                        else kernel_col_load(pa, pb, C_ij, mr, nr, kc, m);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 行主序GEMM
// ============================================================================

inline void sgemm_rowmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    auto compute_block = [&](int ii, int mc, int kc, int nc, int jj, bool zero_init) {
#ifdef _OPENMP
        if (g_num_threads > 1) {
            #pragma omp parallel for schedule(static) num_threads(g_num_threads) proc_bind(close)
            for (int ir = 0; ir < mc; ir += MR_ROW) {
                int mr = std::min(MR_ROW, mc - ir);
                for (int jr = 0; jr < nc; jr += NR_ROW) {
                    int nr = std::min(NR_ROW, nc - jr);
                    float* pa = g_blockA_row + (ir / MR_ROW) * MR_ROW * kc;
                    float* pb = g_blockB_row + (jr / NR_ROW) * NR_ROW * kc;
                    float* cij = C + (ii + ir) * n + (jj + jr);
                    if (zero_init) {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full(pa, pb, cij, kc, n);
                        else kernel_row_zero(pa, pb, cij, mr, nr, kc, n);
                    } else {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_accum(pa, pb, cij, kc, n);
                        else kernel_row_load(pa, pb, cij, mr, nr, kc, n);
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
                    float* pa = g_blockA_row + (ir / MR_ROW) * MR_ROW * kc;
                    float* pb = g_blockB_row + (jr / NR_ROW) * NR_ROW * kc;
                    float* cij = C + (ii + ir) * n + (jj + jr);
                    if (zero_init) {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full(pa, pb, cij, kc, n);
                        else kernel_row_zero(pa, pb, cij, mr, nr, kc, n);
                    } else {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_accum(pa, pb, cij, kc, n);
                        else kernel_row_load(pa, pb, cij, mr, nr, kc, n);
                    }
                }
            }
        }
    };

    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        int kc = std::min(GEMM_KC, k);
        pack_blockB_row_par(B + jj, g_blockB_row, nc, kc, n);
        for (int ii = 0; ii < m; ii += GEMM_MC) {
            int mc = std::min(GEMM_MC, m - ii);
            pack_blockA_row_par(A + ii * k, g_blockA_row, mc, kc, k);
            compute_block(ii, mc, kc, nc, jj, true);
        }
        for (int pp = kc; pp < k; pp += GEMM_KC) {
            int kc2 = std::min(GEMM_KC, k - pp);
            pack_blockB_row_par(B + pp * n + jj, g_blockB_row, nc, kc2, n);
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                pack_blockA_row_par(A + ii * k + pp, g_blockA_row, mc, kc2, k);
                compute_block(ii, mc, kc2, nc, jj, false);
            }
        }
    }
}

// ============================================================================
// 带mask的GEMM接口
// C = (alpha * A @ B + beta * C) & mask
// mask[m][n], true=更新, false=保留C原值
// 在微内核外层按分块检查mask，全mask则跳过，全非mask直接计算
// ============================================================================

inline void sgemm_masked(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc,
    const bool* mask  // [m x n], row-major, stride n
) {
    if (m == 0 || n == 0) return;
    if (k == 0 || alpha == 0.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    if (mask[i * n + j]) C[i * rsc + j * csc] = 0.0f;
        } else if (beta != 1.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    if (mask[i * n + j]) C[i * rsc + j * csc] *= beta;
        }
        return;
    }

    bool c_rowmajor = (csc == 1 && rsc >= n);

    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_buf_size = ((size_t)(GEMM_MC + MR_ROW - 1) / MR_ROW) * MR_ROW * GEMM_KC;
    size_t b_buf_size = ((size_t)(GEMM_NC + NR_ROW - 1) / NR_ROW) * NR_ROW * GEMM_KC;
    buf_a.ensure(a_buf_size);
    buf_b.ensure(b_buf_size);

    // 临时缓冲区用于部分mask的块
    alignas(32) float tmp_c[MR_ROW * NR_ROW];

    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        for (int pp = 0; pp < k; pp += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - pp);
            bool first = (pp == 0);
            pack_b_generic(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                pack_a_generic(A + ii * rsa + pp * csa, buf_a.data, mc, kc, rsa, csa);
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        
                        // 检查此块的mask状态
                        int mask_true = 0, mask_total = mr * nr;
                        for (int i = 0; i < mr; ++i)
                            for (int j = 0; j < nr; ++j)
                                if (mask[(ii + ir + i) * n + (jj + jr + j)]) ++mask_true;
                        
                        if (mask_true == 0) continue;  // 全部被mask，跳过
                        
                        float* pa = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* pb = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;
                        
                        if (mask_true == mask_total) {
                            // 全部未被mask，直接计算
                            if (c_rowmajor) {
                                if (alpha == 1.0f) {
                                    if (first && beta == 0.0f) kernel_row_zero(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                                    else if (first) {
                                        for (int i = 0; i < mr; ++i)
                                            for (int j = 0; j < nr; ++j) C_ij[i * rsc + j] *= beta;
                                        kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                                    } else kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                                } else {
                                    kernel_row_alphabeta(pa, pb, C_ij, mr, nr, kc, (int)rsc, alpha, beta, first);
                                }
                            } else {
                                kernel_row_generic_store(pa, pb, C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                            }
                        } else {
                            // 部分mask：先计算到临时缓冲区再按mask选择性写入
                            memset(tmp_c, 0, sizeof(tmp_c));
                            // 加载原始C值
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    tmp_c[i * NR_ROW + j] = C_ij[i * rsc + j * csc];

                            // 计算GEMM块到临时缓冲区
                            __m256 c_reg[MR_ROW][NR_BLOCKS] = {};
                            dispatch_fma_row<MR_ROW>(pa, pb, c_reg, mr, kc);
                            alignas(32) float gemm_result[MR_ROW][NR_ROW];
                            for (int i = 0; i < MR_ROW; ++i)
                                for (int jj2 = 0; jj2 < NR_BLOCKS; ++jj2)
                                    _mm256_storeu_ps(&gemm_result[i][jj2 * 8], c_reg[i][jj2]);

                            // 按mask写回
                            for (int i = 0; i < mr; ++i) {
                                for (int j = 0; j < nr; ++j) {
                                    if (mask[(ii + ir + i) * n + (jj + jr + j)]) {
                                        float val = alpha * gemm_result[i][j];
                                        if (first) val += beta * tmp_c[i * NR_ROW + j];
                                        else val += tmp_c[i * NR_ROW + j];
                                        C_ij[i * rsc + j * csc] = val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 通用SGEMM接口
// ============================================================================

inline void sgemm(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc
) {
    if (m == 0 || n == 0) return;
    if (k == 0 || alpha == 0.0f) {
        if (beta == 0.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) C[i * rsc + j * csc] = 0.0f;
        } else if (beta != 1.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j) C[i * rsc + j * csc] *= beta;
        }
        return;
    }
    
    // 特化内核
    if (m == 1 && n == 1) { *C = alpha * sdot(A, B, k, csa, rsb) + beta * (*C); return; }
    if (k == 1) { sger(m, n, alpha, beta, A, rsa, B, csb, C, rsc, csc); return; }
    if (m == 1) { gemv_row_simd(A, B, C, n, k, alpha, beta, csa, rsb, csb, csc); return; }
    if (n == 1) { gemv_col_simd(A, B, C, m, k, alpha, beta, rsa, csa, rsb, rsc); return; }
    
    // 布局检测
    bool a_rowmajor_exact = (csa == 1 && rsa == k);
    bool a_colmajor_exact = (rsa == 1 && csa == m);
    bool b_rowmajor_exact = (csb == 1 && rsb == n);
    bool b_colmajor_exact = (rsb == 1 && csb == k);
    bool c_rowmajor_exact = (csc == 1 && rsc == n);
    bool c_colmajor_exact = (rsc == 1 && csc == m);
    bool c_rowmajor = (csc == 1 && rsc >= n);
    
    if (a_colmajor_exact && b_colmajor_exact && c_colmajor_exact && alpha == 1.0f && beta == 0.0f) {
        sgemm_colmajor(A, B, C, m, n, k); return;
    }
    if (a_rowmajor_exact && b_rowmajor_exact && c_rowmajor_exact && alpha == 1.0f && beta == 0.0f) {
        sgemm_rowmajor(A, B, C, m, n, k); return;
    }
    
    // 通用回退
    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_sz = ((size_t)(GEMM_MC + MR_ROW - 1) / MR_ROW) * MR_ROW * GEMM_KC;
    size_t b_sz = ((size_t)(GEMM_NC + NR_ROW - 1) / NR_ROW) * NR_ROW * GEMM_KC;
    buf_a.ensure(a_sz); buf_b.ensure(b_sz);
    
    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        for (int pp = 0; pp < k; pp += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - pp);
            bool first = (pp == 0);
            pack_b_generic(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                pack_a_generic(A + ii * rsa + pp * csa, buf_a.data, mc, kc, rsa, csa);
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        float* pa = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* pb = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;
                        if (c_rowmajor) {
                            if (alpha == 1.0f) {
                                if (first && beta == 0.0f) kernel_row_zero(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                                else if (first) {
                                    for (int i = 0; i < mr; ++i)
                                        for (int j = 0; j < nr; ++j) C_ij[i * rsc + j] *= beta;
                                    kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                                } else kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                            } else {
                                kernel_row_alphabeta(pa, pb, C_ij, mr, nr, kc, (int)rsc, alpha, beta, first);
                            }
                        } else {
                            kernel_row_generic_store(pa, pb, C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                        }
                    }
                }
            }
        }
    }
}

// 便捷重载
inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k,
                  float alpha = 1.0f, float beta = 0.0f) {
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k,
                   int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

inline void matmul(const float* A, const float* B, float* C, int m, int n, int k,
                   float alpha, float beta) {
    sgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads = 0) {
#ifdef _OPENMP
    if (nthreads <= 0) nthreads = g_num_threads;
    int old = g_num_threads;
    g_num_threads = nthreads;
    sgemm_rowmajor(A, B, C, m, n, k);
    g_num_threads = old;
#else
    sgemm_rowmajor(A, B, C, m, n, k);
#endif
}

inline void matmul_colmajor(const float* A, const float* B, float* C, int m, int n, int k) {
    sgemm_colmajor(A, B, C, m, n, k);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
