#pragma once

namespace yt::kernel::avx2 {

namespace detail {

template<int NCOLS, int... Is>
inline void gemv_acc_zero(__m256 acc[][2], std::integer_sequence<int, Is...>) {
    (((acc[Is][0] = _mm256_setzero_ps()), (acc[Is][1] = _mm256_setzero_ps())), ...);
}

template<int NCOLS, int... Is>
inline void gemv_fma_step(const float* __restrict A, const float* const* __restrict Bcols,
                          __m256 acc[][2], int p, std::integer_sequence<int, Is...>) {
    __m256 a0 = _mm256_loadu_ps(A + p);
    __m256 a1 = _mm256_loadu_ps(A + p + 8);
    ((acc[Is][0] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(Bcols[Is] + p),     acc[Is][0]),
      acc[Is][1] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(Bcols[Is] + p + 8), acc[Is][1])), ...);
}

template<int NCOLS, int... Is>
inline void gemv_fma_step8(const float* __restrict A, const float* const* __restrict Bcols,
                           __m256 acc[][2], int p, std::integer_sequence<int, Is...>) {
    __m256 a0 = _mm256_loadu_ps(A + p);
    ((acc[Is][0] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(Bcols[Is] + p), acc[Is][0])), ...);
}

template<int NCOLS, int... Is>
inline void gemv_hsum_write(const __m256 acc[][2], float* sums,
                            std::integer_sequence<int, Is...>) {
    ((sums[Is] = hsum_ps(_mm256_add_ps(acc[Is][0], acc[Is][1]))), ...);
}

} // namespace detail

template<int NCOLS>
inline void gemv_dot_ncols(const float* __restrict A, const float* const* __restrict Bcols,
                           int k, float* sums) {
    static_assert(NCOLS >= 1, "NCOLS must be >= 1");
    __m256 acc[NCOLS][2];
    auto seq = std::make_integer_sequence<int, NCOLS>{};
    detail::gemv_acc_zero<NCOLS>(acc, seq);

    int p = 0;
    for (; p + 16 <= k; p += 16)
        detail::gemv_fma_step<NCOLS>(A, Bcols, acc, p, seq);
    for (; p + 8 <= k; p += 8)
        detail::gemv_fma_step8<NCOLS>(A, Bcols, acc, p, seq);

    detail::gemv_hsum_write<NCOLS>(acc, sums, seq);

    for (; p < k; ++p) {
        float av = A[p];
        for (int c = 0; c < NCOLS; ++c) sums[c] += av * Bcols[c][p];
    }
}

inline float gemv_dot_1col(const float* __restrict A, const float* __restrict B, int k) {
    return sdot_contiguous(A, B, k);
}

template<int UNROLL_N>
inline void gemv_row_colmajor_kernel(const float* __restrict A, const float* __restrict B,
                                     float* __restrict C, int n, int k, float alpha, float beta,
                                     int64_t csa, int64_t csb) {
    if (csa != 1) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            const float* b_col = B + j * csb;
            for (int p = 0; p < k; ++p) sum += A[p * csa] * b_col[p];
            C[j] = alpha * sum + beta * C[j];
        }
        return;
    }
    const int full = (n / UNROLL_N) * UNROLL_N;
#ifdef _OPENMP
    #pragma omp parallel num_threads(g_num_threads) if(g_num_threads > 1 && n >= 64)
    {
        #pragma omp for schedule(static) nowait
        for (int j = 0; j < full; j += UNROLL_N) {
            const float* cols[UNROLL_N];
            for (int c = 0; c < UNROLL_N; ++c) cols[c] = B + (j + c) * csb;
            float sums[UNROLL_N] = {};
            gemv_dot_ncols<UNROLL_N>(A, cols, k, sums);
            for (int c = 0; c < UNROLL_N; ++c)
                C[j + c] = (beta == 0.0f) ? alpha * sums[c] : alpha * sums[c] + beta * C[j + c];
        }
        #pragma omp single
        for (int j = full; j < n; ++j) {
            float sum = gemv_dot_1col(A, B + j * csb, k);
            C[j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[j];
        }
    }
#else
    for (int j = 0; j < full; j += UNROLL_N) {
        const float* cols[UNROLL_N];
        for (int c = 0; c < UNROLL_N; ++c) cols[c] = B + (j + c) * csb;
        float sums[UNROLL_N] = {};
        gemv_dot_ncols<UNROLL_N>(A, cols, k, sums);
        for (int c = 0; c < UNROLL_N; ++c)
            C[j + c] = (beta == 0.0f) ? alpha * sums[c] : alpha * sums[c] + beta * C[j + c];
    }
    for (int j = full; j < n; ++j) {
        float sum = gemv_dot_1col(A, B + j * csb, k);
        C[j] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * C[j];
    }
#endif
}

inline void gemv_row_rowmajor_kernel(const float* __restrict A, const float* __restrict B,
                                     float* __restrict C, int n, int k, float alpha, float beta) {
    if (beta == 0.0f) memset(C, 0, n * sizeof(float));
    else if (beta != 1.0f) {
        __m256 vb = _mm256_set1_ps(beta);
        int j = 0;
        for (; j + 8 <= n; j += 8) _mm256_storeu_ps(C+j, _mm256_mul_ps(vb, _mm256_loadu_ps(C+j)));
        for (; j < n; ++j) C[j] *= beta;
    }
    int p = 0;
    for (; p + 4 <= k; p += 4) {
        float a0 = alpha * A[p], a1 = alpha * A[p+1], a2 = alpha * A[p+2], a3 = alpha * A[p+3];
        __m256 va0 = _mm256_broadcast_ss(&a0), va1 = _mm256_broadcast_ss(&a1);
        __m256 va2 = _mm256_broadcast_ss(&a2), va3 = _mm256_broadcast_ss(&a3);
        const float* b0 = B + p * n; const float* b1 = B + (p+1) * n;
        const float* b2 = B + (p+2) * n; const float* b3 = B + (p+3) * n;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 vc = _mm256_loadu_ps(C + j);
            vc = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + j), vc);
            vc = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + j), vc);
            vc = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + j), vc);
            vc = _mm256_fmadd_ps(va3, _mm256_loadu_ps(b3 + j), vc);
            _mm256_storeu_ps(C + j, vc);
        }
        for (; j < n; ++j) C[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
    }
    for (; p < k; ++p) {
        float a_val = alpha * A[p];
        __m256 va = _mm256_broadcast_ss(&a_val);
        const float* b_row = B + p * n;
        int j = 0;
        for (; j + 8 <= n; j += 8)
            _mm256_storeu_ps(C + j, _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), _mm256_loadu_ps(C + j)));
        for (; j < n; ++j) C[j] += a_val * b_row[j];
    }
}

template<int UNROLL_N>
inline void gemv_row_simd(const float* __restrict A, const float* __restrict B, float* __restrict C,
                          int n, int k, float alpha, float beta,
                          int64_t csa, int64_t rsb, int64_t csb, int64_t csc) {
    if (csa == 1 && csb == 1 && rsb == n && csc == 1) {
        gemv_row_rowmajor_kernel(A, B, C, n, k, alpha, beta); return;
    }

    if (csa == 1 && rsb == 1 && csb == k && csc == 1) {
        gemv_row_colmajor_kernel<UNROLL_N>(A, B, C, n, k, alpha, beta, csa, csb);
        return;
    }

    if (csa == 1 && rsb == 1 && csc == 1) {
        gemv_row_colmajor_kernel<UNROLL_N>(A, B, C, n, k, alpha, beta, csa, csb); return;
    }

    if (csa == 1 && csc == 1 && csb == 1 && rsb != n) {
        if (beta == 0.0f) {
            std::memset(C, 0, n * sizeof(float));
        } else if (beta != 1.0f) {
            __m256 vbeta = _mm256_set1_ps(beta);
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                _mm256_storeu_ps(C + j, _mm256_mul_ps(_mm256_loadu_ps(C + j), vbeta));
            }
            for (; j < n; ++j) C[j] *= beta;
        }

        for (int p = 0; p < k; ++p) {
            float a_val = alpha * A[p];
            const float* b_row = B + p * rsb;
            __m256 va = _mm256_set1_ps(a_val);

            int j = 0;
            for (; j + 32 <= n; j += 32) {
                __m256 vc0 = _mm256_loadu_ps(C + j);
                __m256 vc1 = _mm256_loadu_ps(C + j + 8);
                __m256 vc2 = _mm256_loadu_ps(C + j + 16);
                __m256 vc3 = _mm256_loadu_ps(C + j + 24);
                vc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), vc0);
                vc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j + 8), vc1);
                vc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j + 16), vc2);
                vc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j + 24), vc3);
                _mm256_storeu_ps(C + j, vc0);
                _mm256_storeu_ps(C + j + 8, vc1);
                _mm256_storeu_ps(C + j + 16, vc2);
                _mm256_storeu_ps(C + j + 24, vc3);
            }
            for (; j + 8 <= n; j += 8) {
                __m256 vc = _mm256_loadu_ps(C + j);
                vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + j), vc);
                _mm256_storeu_ps(C + j, vc);
            }
            for (; j < n; ++j) C[j] += a_val * b_row[j];
        }
        return;
    }

    if (csa == 1 && csc == 1 && csb != 1 && rsb == 1) {
        if (beta == 0.0f) {
            std::memset(C, 0, n * sizeof(float));
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; ++j) C[j] *= beta;
        }

        for (int j = 0; j < n; ++j) {
            const float* b_col = B + j * csb;
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
            float sum = hsum_ps(acc0);
            for (; p < k; ++p) sum += A[p] * b_col[p];

            C[j] += alpha * sum;
        }
        return;
    }

    if (csa == 1 && csc == 1 && csb == 1) {
        const __m256i vindex = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256i vstride = _mm256_set1_epi32(static_cast<int>(rsb));
        __m256i vindices = _mm256_mullo_epi32(vindex, vstride);

#pragma omp parallel for schedule(static) if(n >= 256)
        for (int j = 0; j < n; ++j) {
            const float* b_col = B + j;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int p = 0;
            for (; p + 16 <= k; p += 16) {
                __m256 va0 = _mm256_loadu_ps(A + p);
                __m256 va1 = _mm256_loadu_ps(A + p + 8);
                __m256 vb0 = _mm256_i32gather_ps(b_col + p * rsb, vindices, 4);
                __m256 vb1 = _mm256_i32gather_ps(b_col + (p + 8) * rsb, vindices, 4);
                acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
                acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            for (; p + 8 <= k; p += 8) {
                __m256 va = _mm256_loadu_ps(A + p);
                __m256 vb = _mm256_i32gather_ps(b_col + p * rsb, vindices, 4);
                acc0 = _mm256_fmadd_ps(va, vb, acc0);
            }
            float sum = hsum_ps(acc0);
            for (; p < k; ++p) sum += A[p] * b_col[p * rsb];
            C[j] = alpha * sum + beta * C[j];
        }
        return;
    }

    if (csa == 1 && csc == 1) {
        if (rsb == 1) {
            if (beta == 0.0f) {
                std::memset(C, 0, n * sizeof(float));
            } else if (beta != 1.0f) {
                for (int j = 0; j < n; ++j) C[j] *= beta;
            }

            for (int j = 0; j < n; ++j) {
                const float* b_col = B + j * csb;
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
                float sum = hsum_ps(acc0);
                for (; p < k; ++p) sum += A[p] * b_col[p];
                C[j] += alpha * sum;
            }
            return;
        }

        if (beta == 0.0f) {
            std::memset(C, 0, n * sizeof(float));
        } else if (beta != 1.0f) {
            for (int j = 0; j < n; ++j) C[j] *= beta;
        }

        if (csb == 2) {
            constexpr int BLOCK_K = 8;
            constexpr int BLOCK_N = 3072;
            alignas(32) float b_packed[BLOCK_K * BLOCK_N];

            int p = 0;
            for (; p + BLOCK_K <= k; p += BLOCK_K) {
                for (int pp = 0; pp < BLOCK_K; ++pp) {
                    const float* b_row = B + (p + pp) * rsb;
                    float* dst = b_packed + pp * n;
                    int jj = 0;
                    for (; jj + 8 <= n; jj += 8) {
                        __m256 v0 = _mm256_loadu_ps(b_row + jj * 2);
                        __m256 v1 = _mm256_loadu_ps(b_row + jj * 2 + 8);
                        __m256 packed = _mm256_shuffle_ps(v0, v1, 0x88);
                        packed = _mm256_permutevar8x32_ps(packed, _mm256_setr_epi32(0,1,4,5,2,3,6,7));
                        _mm256_storeu_ps(dst + jj, packed);
                    }
                    for (; jj < n; ++jj) dst[jj] = b_row[jj * 2];
                }

                __m256 va[BLOCK_K];
                for (int pp = 0; pp < BLOCK_K; ++pp) va[pp] = _mm256_set1_ps(alpha * A[p + pp]);

                for (int j = 0; j + 8 <= n; j += 8) {
                    __m256 vc = _mm256_loadu_ps(C + j);
                    for (int pp = 0; pp < BLOCK_K; ++pp) {
                        vc = _mm256_fmadd_ps(va[pp], _mm256_loadu_ps(b_packed + pp * n + j), vc);
                    }
                    _mm256_storeu_ps(C + j, vc);
                }
                for (int j = (n / 8) * 8; j < n; ++j) {
                    float sum = C[j];
                    for (int pp = 0; pp < BLOCK_K; ++pp) {
                        sum += alpha * A[p + pp] * b_packed[pp * n + j];
                    }
                    C[j] = sum;
                }
            }

            for (; p < k; ++p) {
                float a_val = alpha * A[p];
                const float* b_row = B + p * rsb;
                __m256 va = _mm256_set1_ps(a_val);

                float* dst = b_packed;
                int jj = 0;
                for (; jj + 8 <= n; jj += 8) {
                    __m256 v0 = _mm256_loadu_ps(b_row + jj * 2);
                    __m256 v1 = _mm256_loadu_ps(b_row + jj * 2 + 8);
                    __m256 packed = _mm256_shuffle_ps(v0, v1, 0x88);
                    packed = _mm256_permutevar8x32_ps(packed, _mm256_setr_epi32(0,1,4,5,2,3,6,7));
                    _mm256_storeu_ps(dst + jj, packed);
                }
                for (; jj < n; ++jj) dst[jj] = b_row[jj * 2];

                jj = 0;
                for (; jj + 8 <= n; jj += 8) {
                    __m256 vc = _mm256_loadu_ps(C + jj);
                    vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_packed + jj), vc);
                    _mm256_storeu_ps(C + jj, vc);
                }
                for (; jj < n; ++jj) C[jj] += a_val * b_packed[jj];
            }
            return;
        }

        constexpr int BLOCK_N = 4096;
        alignas(32) float b_row_packed[BLOCK_N];

        for (int p = 0; p < k; ++p) {
            float a_val = alpha * A[p];
            const float* b_row = B + p * rsb;
            __m256 va = _mm256_set1_ps(a_val);

            for (int j0 = 0; j0 < n; j0 += BLOCK_N) {
                int j_end = std::min(j0 + BLOCK_N, n);
                int block_size = j_end - j0;

                for (int jj = 0; jj < block_size; ++jj) {
                    b_row_packed[jj] = b_row[(j0 + jj) * csb];
                }

                float* c_ptr = C + j0;
                int jj = 0;
                for (; jj + 32 <= block_size; jj += 32) {
                    __m256 vc0 = _mm256_loadu_ps(c_ptr + jj);
                    __m256 vc1 = _mm256_loadu_ps(c_ptr + jj + 8);
                    __m256 vc2 = _mm256_loadu_ps(c_ptr + jj + 16);
                    __m256 vc3 = _mm256_loadu_ps(c_ptr + jj + 24);
                    vc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row_packed + jj), vc0);
                    vc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row_packed + jj + 8), vc1);
                    vc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row_packed + jj + 16), vc2);
                    vc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row_packed + jj + 24), vc3);
                    _mm256_storeu_ps(c_ptr + jj, vc0);
                    _mm256_storeu_ps(c_ptr + jj + 8, vc1);
                    _mm256_storeu_ps(c_ptr + jj + 16, vc2);
                    _mm256_storeu_ps(c_ptr + jj + 24, vc3);
                }
                for (; jj + 8 <= block_size; jj += 8) {
                    __m256 vc = _mm256_loadu_ps(c_ptr + jj);
                    vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row_packed + jj), vc);
                    _mm256_storeu_ps(c_ptr + jj, vc);
                }
                for (; jj < block_size; ++jj) {
                    c_ptr[jj] += a_val * b_row_packed[jj];
                }
            }
        }
        return;
    }

    for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        const float* b_col = B + j * csb;
        for (int p = 0; p < k; ++p) sum += A[p * csa] * b_col[p * rsb];
        C[j * csc] = alpha * sum + beta * C[j * csc];
    }
}

inline void gemv_col_simd(const float* A, const float* B, float* C,
                          int m, int k, float alpha, float beta,
                          int64_t rsa, int64_t csa, int64_t rsb, int64_t rsc) {
    if (csa == 1 && rsb == 1 && rsc == 1) {
        int i = 0;
        for (; i + 8 <= m; i += 8) {
            __m256 acc[8];
            for (int ii = 0; ii < 8; ++ii) acc[ii] = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= k; p += 8) {
                __m256 vb = _mm256_loadu_ps(B + p);
                for (int ii = 0; ii < 8; ++ii)
                    acc[ii] = _mm256_fmadd_ps(_mm256_loadu_ps(A + (i + ii) * rsa + p), vb, acc[ii]);
            }
            for (int ii = 0; ii < 8; ++ii) {
                float sum = hsum_ps(acc[ii]);
                for (int pp = p; pp < k; ++pp) sum += A[(i + ii) * rsa + pp] * B[pp];
                C[i + ii] = alpha * sum + beta * C[i + ii];
            }
        }
        for (; i < m; ++i) {
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= k; p += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(A + i * rsa + p), _mm256_loadu_ps(B + p), acc);
            float sum = hsum_ps(acc);
            for (; p < k; ++p) sum += A[i * rsa + p] * B[p];
            C[i] = alpha * sum + beta * C[i];
        }
        return;
    }
    if (rsa == 1 && rsb == 1 && rsc == 1) {
        if (beta == 0.0f) for (int i = 0; i < m; ++i) C[i] = 0.0f;
        else if (beta != 1.0f) for (int i = 0; i < m; ++i) C[i] *= beta;
        for (int p = 0; p < k; ++p) {
            float b_val = alpha * B[p];
            const float* a_col = A + p * csa;
            __m256 vb = _mm256_set1_ps(b_val);
            int i = 0;
            for (; i + 8 <= m; i += 8)
                _mm256_storeu_ps(C + i, _mm256_fmadd_ps(_mm256_loadu_ps(a_col + i), vb, _mm256_loadu_ps(C + i)));
            for (; i < m; ++i) C[i] += b_val * a_col[i];
        }
        return;
    }
    for (int i = 0; i < m; ++i) {
        float sum = 0.0f;
        for (int p = 0; p < k; ++p) sum += A[i * rsa + p * csa] * B[p * rsb];
        C[i * rsc] = alpha * sum + beta * C[i * rsc];
    }
}

} // namespace yt::kernel::avx2
