#pragma once
/***************
 * @file hgemm.hpp
 * @brief 半精度矩阵乘法（HGEMM）实现
 * @author SnifferCaptain
 * @date 2025-12-31
 *
 * 策略:
 * - 若支持F16C指令集: 在微内核内部用 _mm256_cvtph_ps / _mm256_cvtps_ph 做流式转换
 * - 否则: 在打包阶段批量转换 float16 -> float32，调用 sgemm 微内核，输出时转回 float16
 *
 * 输入输出均为 yt::float16 类型
 ***************/

#include "gemm_utils.hpp"
#include "../../types/float_spec.hpp"

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

// ============================================================================
// float16 <-> float32 批量转换工具
// ============================================================================

#if __F16C__

// F16C 硬件加速转换
inline void f16_to_f32_block(const yt::float16* src, float* dst, int count) {
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i h8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f8 = _mm256_cvtph_ps(h8);
        _mm256_storeu_ps(dst + i, f8);
    }
    for (; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

inline void f32_to_f16_block(const float* src, yt::float16* dst, int count) {
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 f8 = _mm256_loadu_ps(src + i);
        __m128i h8 = _mm256_cvtps_ph(f8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h8);
    }
    for (; i < count; ++i) {
        dst[i] = yt::float16(src[i]);
    }
}

#else

// 软件回退转换
inline void f16_to_f32_block(const yt::float16* src, float* dst, int count) {
    for (int i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

inline void f32_to_f16_block(const float* src, yt::float16* dst, int count) {
    for (int i = 0; i < count; ++i) {
        dst[i] = yt::float16(src[i]);
    }
}

#endif // __F16C__

// ============================================================================
// 半精度打包：打包阶段直接转换为 float32
// ============================================================================

inline void pack_a_row_f16(const yt::float16* A, float* packed, int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) packed[ii] = static_cast<float>(A[(i + ii) * lda + p]);
            for (int ii = mr; ii < MR_ROW; ++ii) packed[ii] = 0.0f;
            packed += MR_ROW;
        }
    }
}

inline void pack_b_row_f16(const yt::float16* B, float* packed, int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        float* dest = packed + (j / NR_ROW) * NR_ROW * kc;
#if __F16C__
        if (nr == NR_ROW) {
            for (int p = 0; p < kc; ++p) {
                const yt::float16* src = B + p * ldb + j;
                __m128i h_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
                __m128i h_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 8));
                _mm256_storeu_ps(dest, _mm256_cvtph_ps(h_lo));
                _mm256_storeu_ps(dest + 8, _mm256_cvtph_ps(h_hi));
                dest += NR_ROW;
            }
        } else
#endif
        {
            for (int p = 0; p < kc; ++p) {
                for (int jj = 0; jj < nr; ++jj) dest[jj] = static_cast<float>(B[p * ldb + j + jj]);
                for (int jj = nr; jj < NR_ROW; ++jj) dest[jj] = 0.0f;
                dest += NR_ROW;
            }
        }
    }
}

inline void pack_a_generic_f16(const yt::float16* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa) {
    for (int i = 0; i < mc; i += MR_ROW) {
        int mr = std::min(MR_ROW, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) packed[ii] = static_cast<float>(A[(i + ii) * rsa + p * csa]);
            for (int ii = mr; ii < MR_ROW; ++ii) packed[ii] = 0.0f;
            packed += MR_ROW;
        }
    }
}

inline void pack_b_generic_f16(const yt::float16* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb) {
    for (int j = 0; j < nc; j += NR_ROW) {
        int nr = std::min(NR_ROW, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) packed[jj] = static_cast<float>(B[p * rsb + (j + jj) * csb]);
            for (int jj = nr; jj < NR_ROW; ++jj) packed[jj] = 0.0f;
            packed += NR_ROW;
        }
    }
}

// ============================================================================
// 输出写回：float32 -> float16
// ============================================================================

inline void store_c_f16(const float* c_buf, yt::float16* C, int mr, int nr, int ldc) {
#if __F16C__
    for (int i = 0; i < mr; ++i) {
        int j = 0;
        for (; j + 8 <= nr; j += 8) {
            __m256 f8 = _mm256_loadu_ps(c_buf + i * NR_ROW + j);
            __m128i h8 = _mm256_cvtps_ph(f8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(C + i * ldc + j), h8);
        }
        for (; j < nr; ++j) {
            C[i * ldc + j] = yt::float16(c_buf[i * NR_ROW + j]);
        }
    }
#else
    for (int i = 0; i < mr; ++i)
        for (int j = 0; j < nr; ++j)
            C[i * ldc + j] = yt::float16(c_buf[i * NR_ROW + j]);
#endif
}

inline void store_c_generic_f16(const float* c_buf, yt::float16* C, int mr, int nr, int64_t rsc, int64_t csc) {
    for (int i = 0; i < mr; ++i)
        for (int j = 0; j < nr; ++j)
            C[i * rsc + j * csc] = yt::float16(c_buf[i * NR_ROW + j]);
}

// ============================================================================
// HGEMM 主入口
// 输入输出均为 float16，内部以 float32 计算
// ============================================================================

inline void hgemm(
    const yt::float16* A, const yt::float16* B, yt::float16* C,
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
                for (int j = 0; j < n; ++j) C[i * rsc + j * csc] = yt::float16(0.0f);
        } else if (beta != 1.0f) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    C[i * rsc + j * csc] = yt::float16(beta * static_cast<float>(C[i * rsc + j * csc]));
        }
        return;
    }

    bool c_rowmajor = (csc == 1 && rsc >= n);

    static thread_local AlignedBuffer buf_a, buf_b;
    size_t a_sz = ((size_t)(GEMM_MC + MR_ROW - 1) / MR_ROW) * MR_ROW * GEMM_KC;
    size_t b_sz = ((size_t)(GEMM_NC + NR_ROW - 1) / NR_ROW) * NR_ROW * GEMM_KC;
    buf_a.ensure(a_sz);
    buf_b.ensure(b_sz);

    // 临时 float32 缓冲区用于输出
    alignas(32) float c_tmp[MR_ROW * NR_ROW];

    for (int jj = 0; jj < n; jj += GEMM_NC) {
        int nc = std::min(GEMM_NC, n - jj);
        for (int pp = 0; pp < k; pp += GEMM_KC) {
            int kc = std::min(GEMM_KC, k - pp);
            bool first = (pp == 0);
            pack_b_generic_f16(B + pp * rsb + jj * csb, buf_b.data, kc, nc, rsb, csb);
            for (int ii = 0; ii < m; ii += GEMM_MC) {
                int mc = std::min(GEMM_MC, m - ii);
                pack_a_generic_f16(A + ii * rsa + pp * csa, buf_a.data, mc, kc, rsa, csa);
                for (int ir = 0; ir < mc; ir += MR_ROW) {
                    int mr = std::min(MR_ROW, mc - ir);
                    for (int jr = 0; jr < nc; jr += NR_ROW) {
                        int nr = std::min(NR_ROW, nc - jr);
                        float* pa = buf_a.data + (ir / MR_ROW) * MR_ROW * kc;
                        float* pb = buf_b.data + (jr / NR_ROW) * NR_ROW * kc;
                        yt::float16* C_ij = C + (ii + ir) * rsc + (jj + jr) * csc;

                        // 计算微内核到 c_tmp（始终零初始化，alpha/beta 单独处理）
                        __m256 c_reg[MR_ROW][NR_BLOCKS] = {};
                        dispatch_fma_row<MR_ROW>(pa, pb, c_reg, mr, kc);
                        for (int i = 0; i < mr; ++i)
                            for (int jj2 = 0; jj2 < NR_BLOCKS; ++jj2)
                                _mm256_storeu_ps(&c_tmp[i * NR_ROW + jj2 * 8], c_reg[i][jj2]);

                        // alpha/beta 写回
                        if (first) {
                            // 首次: result = alpha * fma + beta * C_old
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    c_tmp[i * NR_ROW + j] = alpha * c_tmp[i * NR_ROW + j]
                                        + beta * static_cast<float>(C_ij[i * rsc + j * csc]);
                        } else {
                            // 后续: result = alpha * fma + C_partial
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    c_tmp[i * NR_ROW + j] = alpha * c_tmp[i * NR_ROW + j]
                                        + static_cast<float>(C_ij[i * rsc + j * csc]);
                        }

                        // 写回 float16
                        if (c_rowmajor)
                            store_c_f16(c_tmp, C_ij, mr, nr, static_cast<int>(rsc));
                        else
                            store_c_generic_f16(c_tmp, C_ij, mr, nr, rsc, csc);
                    }
                }
            }
        }
    }
}

// 便捷重载
inline void hgemm(const yt::float16* A, const yt::float16* B, yt::float16* C,
                  int m, int n, int k, float alpha = 1.0f, float beta = 0.0f) {
    hgemm(A, B, C, m, n, k, alpha, beta, k, 1, n, 1, n, 1);
}

inline void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C,
                    int m, int n, int k,
                    int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc, int64_t csc) {
    hgemm(A, B, C, m, n, k, 1.0f, 0.0f, rsa, csa, rsb, csb, rsc, csc);
}

inline void hmatmul(const yt::float16* A, const yt::float16* B, yt::float16* C, int m, int n, int k) {
    hgemm(A, B, C, m, n, k, 1.0f, 0.0f, k, 1, n, 1, n, 1);
}

} // namespace yt::kernel::avx2

#endif // __AVX2__ && __FMA__
