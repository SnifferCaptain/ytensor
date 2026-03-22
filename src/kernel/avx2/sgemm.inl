#pragma once

#include <algorithm>
#include <cstring>

namespace yt::kernel::avx2 {

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
            __m256 r0 = _mm256_loadu_ps(A + i * lda + p);
            __m256 r1 = _mm256_loadu_ps(A + (i + 1) * lda + p);
            __m256 r2 = _mm256_loadu_ps(A + (i + 2) * lda + p);
            __m256 r3 = _mm256_loadu_ps(A + (i + 3) * lda + p);
            __m256 r4 = _mm256_loadu_ps(A + (i + 4) * lda + p);
            __m256 r5 = _mm256_loadu_ps(A + (i + 5) * lda + p);
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
        for (int ii = 0; ii < mr; ++ii) dest[p * MR_ROW + ii] = A[(i + ii) * lda + p];
        for (int ii = mr; ii < MR_ROW; ++ii) dest[p * MR_ROW + ii] = 0.0f;
    }
}

__attribute__((always_inline, hot))
inline void kernel_row_full_fast(const float* __restrict A, const float* __restrict B,
                                 float* __restrict C, int kc, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B + p * NR_ROW);
        __m256 b1 = _mm256_loadu_ps(B + p * NR_ROW + 8);
        const float* ap = A + p * MR_ROW;

        __m256 a = _mm256_broadcast_ss(ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C + 0 * ldc, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}

__attribute__((always_inline, hot))
inline void kernel_row_full_accum_fast(const float* __restrict A, const float* __restrict B,
                                       float* __restrict C, int kc, int ldc) {
    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc), c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc), c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc), c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc), c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc), c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc), c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    for (int p = 0; p < kc; ++p) {
        __m256 b0 = _mm256_loadu_ps(B + p * NR_ROW);
        __m256 b1 = _mm256_loadu_ps(B + p * NR_ROW + 8);
        const float* ap = A + p * MR_ROW;

        __m256 a = _mm256_broadcast_ss(ap + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(ap + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(ap + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(ap + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(ap + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(ap + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    _mm256_storeu_ps(C + 0 * ldc, c00); _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10); _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20); _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30); _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40); _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50); _mm256_storeu_ps(C + 5 * ldc + 8, c51);
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
    for (int j = 0; j < nc; j += NR_ROW) pack_tile(j);
}

inline void pack_blockA_row_par(const float* A, float* packed, int mc, int kc, int lda) {
    auto pack_tile = [&](int i) {
        int mr = std::min(MR_ROW, mc - i);
        pack_blockA_row_single_tile(A, packed + (i / MR_ROW) * MR_ROW * kc, i, mr, kc, lda);
    };
    for (int i = 0; i < mc; i += MR_ROW) pack_tile(i);
}

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
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_fast(pa, pb, cij, kc, n);
                        else kernel_row_zero(pa, pb, cij, mr, nr, kc, n);
                    } else {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_accum_fast(pa, pb, cij, kc, n);
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
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_fast(pa, pb, cij, kc, n);
                        else kernel_row_zero(pa, pb, cij, mr, nr, kc, n);
                    } else {
                        if (mr == MR_ROW && nr == NR_ROW) kernel_row_full_accum_fast(pa, pb, cij, kc, n);
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

inline void sgemm_masked(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsb, int64_t csb,
    int64_t rsc, int64_t csc,
    const bool* mask
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

                        int mask_true = 0, mask_total = mr * nr;
                        for (int i = 0; i < mr; ++i)
                            for (int j = 0; j < nr; ++j)
                                if (mask[(ii + ir + i) * n + (jj + jr + j)]) ++mask_true;

                        if (mask_true == 0) continue;

                        float* pa = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* pb = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;

                        if (mask_true == mask_total) {
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
                            memset(tmp_c, 0, sizeof(tmp_c));
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    tmp_c[i * NR_ROW + j] = C_ij[i * rsc + j * csc];

                            __m256 c_reg[MR_ROW][NR_BLOCKS] = {};
                            dispatch_fma_row<MR_ROW>(pa, pb, c_reg, mr, kc);
                            alignas(32) float gemm_result[MR_ROW][NR_ROW];
                            for (int i = 0; i < MR_ROW; ++i)
                                for (int jj2 = 0; jj2 < NR_BLOCKS; ++jj2)
                                    _mm256_storeu_ps(&gemm_result[i][jj2 * 8], c_reg[i][jj2]);

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

    if (m == 1 && n == 1) { *C = alpha * sdot(A, B, k, csa, rsb) + beta * (*C); return; }
    if (k == 1) { sger(m, n, alpha, beta, A, rsa, B, csb, C, rsc, csc); return; }
    if (m == 1) { gemv_row_simd(A, B, C, n, k, alpha, beta, csa, rsb, csb, csc); return; }
    if (n == 1) { gemv_col_simd(A, B, C, m, k, alpha, beta, rsa, csa, rsb, rsc); return; }

    bool a_rowmajor_exact = (csa == 1 && rsa == k);
    bool a_colmajor_exact = (rsa == 1 && csa == m);
    bool b_rowmajor_exact = (csb == 1 && rsb == n);
    bool b_colmajor_exact = (rsb == 1 && csb == k);
    bool c_rowmajor_exact = (csc == 1 && rsc == n);
    bool c_colmajor_exact = (rsc == 1 && csc == m);
    bool c_rowmajor = (csc == 1 && rsc >= n);

    bool a_exact = (a_rowmajor_exact || a_colmajor_exact);
    bool b_exact = (b_rowmajor_exact || b_colmajor_exact);
    bool c_exact = (c_rowmajor_exact || c_colmajor_exact);

#ifdef _OPENMP
    if (g_num_threads > 1 && alpha == 1.0f && beta == 0.0f && a_exact && b_exact && c_exact &&
        m >= 512 && n >= 512 && k >= 512) {
        static thread_local AlignedBuffer a_row_buf, b_row_buf, c_row_buf;

        const float* A_row = A;
        const float* B_row = B;
        float* C_row = C;
        bool need_a_convert = !a_rowmajor_exact;
        bool need_b_convert = !b_rowmajor_exact;
        bool need_c_scatter = !c_rowmajor_exact;

        if (need_a_convert) {
            a_row_buf.ensure((size_t)m * (size_t)k);
            float* A_rm = a_row_buf.data;
            #pragma omp parallel for collapse(2) schedule(static) num_threads(g_num_threads) proc_bind(close)
            for (int i = 0; i < m; ++i)
                for (int p = 0; p < k; ++p)
                    A_rm[i * k + p] = A[i * rsa + p * csa];
            A_row = A_rm;
        }

        if (need_b_convert) {
            b_row_buf.ensure((size_t)k * (size_t)n);
            float* B_rm = b_row_buf.data;
            #pragma omp parallel for collapse(2) schedule(static) num_threads(g_num_threads) proc_bind(close)
            for (int p = 0; p < k; ++p)
                for (int j = 0; j < n; ++j)
                    B_rm[p * n + j] = B[p * rsb + j * csb];
            B_row = B_rm;
        }

        if (need_c_scatter) {
            c_row_buf.ensure((size_t)m * (size_t)n);
            C_row = c_row_buf.data;
        }

        sgemm_rowmajor(A_row, B_row, C_row, m, n, k);

        if (need_c_scatter) {
            #pragma omp parallel for collapse(2) schedule(static) num_threads(g_num_threads) proc_bind(close)
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * rsc + j * csc] = C_row[i * n + j];
        }
        return;
    }
#endif

    if (a_colmajor_exact && b_colmajor_exact && c_colmajor_exact && alpha == 1.0f && beta == 0.0f) {
#ifdef _OPENMP
        if (g_num_threads > 1) {
            // C = A*B (all col-major)
            // Equivalent: C^T = B^T * A^T (all row-major views, no copy)
            sgemm_rowmajor(B, A, C, n, m, k);
            return;
        }
#endif
        sgemm_colmajor(A, B, C, m, n, k); return;
    }
    if (a_rowmajor_exact && b_rowmajor_exact && c_rowmajor_exact && alpha == 1.0f && beta == 0.0f) {
        sgemm_rowmajor(A, B, C, m, n, k); return;
    }

    if (a_rowmajor_exact && b_colmajor_exact && c_rowmajor_exact && alpha == 1.0f && beta == 0.0f) {
        static thread_local AlignedBuffer b_row_buf;
        b_row_buf.ensure((size_t)k * (size_t)n);
        float* B_row = b_row_buf.data;
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                B_row[p * n + j] = B[j * k + p];
        sgemm_rowmajor(A, B_row, C, m, n, k);
        return;
    }

    if (a_colmajor_exact && b_rowmajor_exact && c_rowmajor_exact && alpha == 1.0f && beta == 0.0f) {
        static thread_local AlignedBuffer a_row_buf;
        a_row_buf.ensure((size_t)m * (size_t)k);
        float* A_row = a_row_buf.data;
        for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p)
                A_row[i * k + p] = A[p * m + i];
        sgemm_rowmajor(A_row, B, C, m, n, k);
        return;
    }

    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_sz = ((size_t)(GEMM_MC + MR_ROW - 1) / MR_ROW) * MR_ROW * GEMM_KC;
    size_t b_sz = ((size_t)(GEMM_NC + NR_ROW - 1) / NR_ROW) * NR_ROW * GEMM_KC;
    buf_a.ensure(a_sz);
    buf_b.ensure(b_sz);

    auto compute_ii_block = [&](int ii, int jj, int pp, int mc, int nc, int kc, bool first, float* a_buf, float* b_buf) {
        pack_a_generic(A + ii * rsa + pp * csa, a_buf, mc, kc, rsa, csa);
        for (int ir = 0; ir < mc; ir += MR_ROW) {
            int mr = std::min(MR_ROW, mc - ir);
            for (int jr = 0; jr < nc; jr += NR_ROW) {
                int nr = std::min(NR_ROW, nc - jr);
                float* pa = a_buf + (ir / MR_ROW) * MR_ROW * kc;
                float* pb = b_buf + (jr / NR_ROW) * NR_ROW * kc;
                float* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;
                if (c_rowmajor) {
                    if (alpha == 1.0f) {
                        if (first && beta == 0.0f) kernel_row_zero(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                        else if (first) {
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j) C_ij[i * rsc + j] *= beta;
                            kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                        } else {
                            kernel_row_load(pa, pb, C_ij, mr, nr, kc, (int)rsc);
                        }
                    } else {
                        kernel_row_alphabeta(pa, pb, C_ij, mr, nr, kc, (int)rsc, alpha, beta, first);
                    }
                } else {
                    kernel_row_generic_store(pa, pb, C_ij, mr, nr, kc, rsc, csc, alpha, beta, first);
                }
            }
        }
    };

    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        for (int pp = 0; pp < k; pp += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - pp);
            bool first = (pp == 0);
            pack_b_generic(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            float* packed_b = buf_b.data;

#ifdef _OPENMP
            if (g_num_threads > 1) {
                int m_blocks = (m + GEMM_MC - 1) / GEMM_MC;
                #pragma omp parallel num_threads(g_num_threads) proc_bind(close)
                {
                    AlignedBuffer buf_a_thread;
                    buf_a_thread.ensure(a_sz);
                    #pragma omp for schedule(static)
                    for (int bi = 0; bi < m_blocks; ++bi) {
                        int ii = bi * GEMM_MC;
                        int mc = std::min(GEMM_MC, m - ii);
                        compute_ii_block(ii, jj, pp, mc, nc, kc, first, buf_a_thread.data, packed_b);
                    }
                }
            } else
#endif
            {
                for (int ii = 0; ii < m; ii += GEMM_MC) {
                    int mc = std::min(GEMM_MC, m - ii);
                    compute_ii_block(ii, jj, pp, mc, nc, kc, first, buf_a.data, packed_b);
                }
            }
        }
    }
}

inline void sgemm(const float* A, const float* B, float* C, int m, int n, int k,
                  float alpha, float beta) {
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

inline void matmul_parallel(const float* A, const float* B, float* C, int m, int n, int k, [[maybe_unused]] int nthreads) {
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
