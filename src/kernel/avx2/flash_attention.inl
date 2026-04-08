#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::kernel::avx2 {

struct _FlashAllMask {
    inline bool operator()(int, int) const { return true; }
    inline bool tileAllTrue(int, int, int, int) const { return true; }
    inline bool tileAllFalse(int, int, int, int) const { return false; }
};

struct _FlashBoolMask {
    const bool* data = nullptr;
    int64_t stride = 0;

    inline bool operator()(int row, int col) const {
        return data[row * stride + col];
    }

    inline bool tileAllTrue(int row0, int col0, int mr, int nr) const {
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                if (!data[(row0 + i) * stride + (col0 + j)]) {
                    return false;
                }
            }
        }
        return true;
    }

    inline bool tileAllFalse(int row0, int col0, int mr, int nr) const {
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                if (data[(row0 + i) * stride + (col0 + j)]) {
                    return false;
                }
            }
        }
        return true;
    }
};

inline float _flashBiasValue(const float* bias, int64_t rsbias, int64_t csbias, int row, int col) {
    if (bias == nullptr) {
        return 0.0f;
    }
    return bias[row * rsbias + col * csbias];
}

inline void _flashZeroRow(float* row, int value_dim, int64_t stride) {
    if (stride == 1) {
        std::memset(row, 0, static_cast<size_t>(value_dim) * sizeof(float));
        return;
    }
    for (int d = 0; d < value_dim; ++d) {
        row[d * stride] = 0.0f;
    }
}

template <typename MaskOp>
inline bool _flashTileAllTrue(const MaskOp& mask, int row0, int col0, int mr, int nr) {
    if constexpr (requires { { mask.tileAllTrue(row0, col0, mr, nr) } -> std::convertible_to<bool>; }) {
        return mask.tileAllTrue(row0, col0, mr, nr);
    } else {
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                if (!mask(row0 + i, col0 + j)) {
                    return false;
                }
            }
        }
        return true;
    }
}

template <typename MaskOp>
inline bool _flashTileAllFalse(const MaskOp& mask, int row0, int col0, int mr, int nr) {
    if constexpr (requires { { mask.tileAllFalse(row0, col0, mr, nr) } -> std::convertible_to<bool>; }) {
        return mask.tileAllFalse(row0, col0, mr, nr);
    } else {
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                if (mask(row0 + i, col0 + j)) {
                    return false;
                }
            }
        }
        return true;
    }
}

inline void _flashScaleRow(float* row, int value_dim, int64_t stride, float scale) {
    if (scale == 1.0f) {
        return;
    }

    if (stride == 1) {
        __m256 scale_vec = _mm256_set1_ps(scale);
        int d = 0;
        for (; d + 8 <= value_dim; d += 8) {
            __m256 out = _mm256_loadu_ps(row + d);
            out = _mm256_mul_ps(out, scale_vec);
            _mm256_storeu_ps(row + d, out);
        }
        for (; d < value_dim; ++d) {
            row[d] *= scale;
        }
        return;
    }

    for (int d = 0; d < value_dim; ++d) {
        row[d * stride] *= scale;
    }
}

inline void _flashAccumulateRow(
    float* out_row,
    int value_dim,
    int64_t out_stride,
    const float* V,
    int value_row,
    int64_t rsv,
    int64_t csv,
    float weight
) {
    if (weight == 0.0f) {
        return;
    }

    const float* value_ptr = V + value_row * rsv;
    __m256 weight_vec = _mm256_set1_ps(weight);

    if (out_stride == 1 && csv == 1) {
        int d = 0;
        for (; d + 8 <= value_dim; d += 8) {
            __m256 out = _mm256_loadu_ps(out_row + d);
            __m256 val = _mm256_loadu_ps(value_ptr + d);
            out = _mm256_fmadd_ps(weight_vec, val, out);
            _mm256_storeu_ps(out_row + d, out);
        }
        for (; d < value_dim; ++d) {
            out_row[d] += weight * value_ptr[d];
        }
        return;
    }

    for (int d = 0; d < value_dim; ++d) {
        out_row[d * out_stride] += weight * value_ptr[d * csv];
    }
}

inline void _flashAccumulateRowContiguous(
    float* out_row,
    int value_dim,
    const float* V,
    int64_t rsv,
    const float* probs,
    int nr,
    float rescale,
    float inv_new_l
) {
    int d = 0;
    __m256 rescale_vec = _mm256_set1_ps(rescale);

    for (; d + 8 <= value_dim; d += 8) {
        __m256 acc = (rescale == 0.0f)
            ? _mm256_setzero_ps()
            : _mm256_mul_ps(_mm256_loadu_ps(out_row + d), rescale_vec);

        for (int j = 0; j < nr; ++j) {
            const float prob = probs[j];
            if (prob == 0.0f) {
                continue;
            }
            __m256 val = _mm256_loadu_ps(V + j * rsv + d);
            __m256 prob_vec = _mm256_set1_ps(prob * inv_new_l);
            acc = _mm256_fmadd_ps(prob_vec, val, acc);
        }
        _mm256_storeu_ps(out_row + d, acc);
    }

    for (; d < value_dim; ++d) {
        float acc = (rescale == 0.0f) ? 0.0f : out_row[d] * rescale;
        for (int j = 0; j < nr; ++j) {
            const float prob = probs[j];
            if (prob != 0.0f) {
                acc += (prob * inv_new_l) * V[j * rsv + d];
            }
        }
        out_row[d] = acc;
    }
}

template <typename MaskOp>
void _flashAttentionDecodeGemvImpl(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t cso,
    MaskOp&& mask,
    const float* bias,
    int64_t rsbias,
    int64_t csbias
) {
    static thread_local AlignedBuffer score_buf;
    score_buf.ensure(static_cast<size_t>(kv_len));
    float* scores = score_buf.data;

    gemv_row_simd(Q, K, scores, kv_len, head_dim, scale, 0.0f, csq, csk, rsk, 1);

    float row_max = -std::numeric_limits<float>::infinity();
    bool any_visible = false;
    for (int col = 0; col < kv_len; ++col) {
        if (!mask(0, col)) {
            scores[col] = -std::numeric_limits<float>::infinity();
            continue;
        }
        scores[col] += _flashBiasValue(bias, rsbias, csbias, 0, col);
        row_max = std::max(row_max, scores[col]);
        any_visible = true;
    }

    if (!any_visible) {
        _flashZeroRow(O, value_dim, cso);
        return;
    }

    float row_sum = 0.0f;
    for (int col = 0; col < kv_len; ++col) {
        if (!mask(0, col)) {
            scores[col] = 0.0f;
            continue;
        }
        scores[col] = std::exp(scores[col] - row_max);
        row_sum += scores[col];
    }

    if (row_sum == 0.0f) {
        _flashZeroRow(O, value_dim, cso);
        return;
    }

    const float inv_row_sum = 1.0f / row_sum;
    __m256 vinv = _mm256_set1_ps(inv_row_sum);
    int col = 0;
    for (; col + 8 <= kv_len; col += 8) {
        _mm256_storeu_ps(scores + col, _mm256_mul_ps(_mm256_loadu_ps(scores + col), vinv));
    }
    for (; col < kv_len; ++col) {
        scores[col] *= inv_row_sum;
    }

    gemv_row_simd(scores, V, O, value_dim, kv_len, 1.0f, 0.0f, 1, rsv, csv, cso);
}

template <typename MaskOp>
void _flashAttentionImpl(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int q_len,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t rsq,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t rso,
    int64_t cso,
    MaskOp&& mask,
    const float* bias,
    int64_t rsbias,
    int64_t csbias
) {
    if (q_len == 0 || kv_len == 0 || head_dim == 0 || value_dim == 0) {
        return;
    }

    AlignedBuffer k_pack_buf;
    const size_t packed_k_size = ((static_cast<size_t>(kv_len) + NR_ROW - 1) / NR_ROW) * static_cast<size_t>(NR_ROW) * static_cast<size_t>(head_dim);
    k_pack_buf.ensure(packed_k_size);
    pack_b_generic(K, k_pack_buf.data, head_dim, kv_len, csk, rsk);

    const int q_blocks = (q_len + MR_ROW - 1) / MR_ROW;

    #pragma omp parallel for schedule(static) if(q_blocks > 1) proc_bind(close)
    for (int q_block = 0; q_block < q_blocks; ++q_block) {
        static thread_local AlignedBuffer q_pack_buf;
        q_pack_buf.ensure(static_cast<size_t>(MR_ROW) * static_cast<size_t>(head_dim));
        float* packed_q = q_pack_buf.data;

        const int qi = q_block * MR_ROW;
        const int mr = std::min(MR_ROW, q_len - qi);
        pack_a_generic(Q + qi * rsq, packed_q, mr, head_dim, rsq, csq);

        alignas(32) float score_tile[MR_ROW][NR_ROW];
        alignas(32) float row_prob[MR_ROW][NR_ROW];
        alignas(32) float row_max[MR_ROW];
        alignas(32) float row_l[MR_ROW];
        alignas(32) float row_new_max[MR_ROW];
        alignas(32) float row_new_l[MR_ROW];
        alignas(32) float row_rescale[MR_ROW];

        for (int i = 0; i < mr; ++i) {
            row_max[i] = -std::numeric_limits<float>::infinity();
            row_l[i] = 0.0f;
            _flashZeroRow(O + (qi + i) * rso, value_dim, cso);
        }

        for (int kj = 0; kj < kv_len; kj += NR_ROW) {
            const int nr = std::min(NR_ROW, kv_len - kj);
            if (_flashTileAllFalse(mask, qi, kj, mr, nr)) {
                continue;
            }

            float* packed_k = k_pack_buf.data + (kj / NR_ROW) * NR_ROW * head_dim;
            __m256 c_reg[MR_ROW][NR_BLOCKS] = {};
            dispatch_fma_row<MR_ROW>(packed_q, packed_k, c_reg, mr, head_dim);
            for (int i = 0; i < MR_ROW; ++i) {
                for (int block = 0; block < NR_BLOCKS; ++block) {
                    _mm256_storeu_ps(&score_tile[i][block * 8], c_reg[i][block]);
                }
            }

            for (int i = 0; i < mr; ++i) {
                row_new_max[i] = row_max[i];
                row_new_l[i] = row_l[i];
                row_rescale[i] = 0.0f;
            }

            const bool tile_all_true = _flashTileAllTrue(mask, qi, kj, mr, nr);

            for (int i = 0; i < mr; ++i) {
                float tile_row_max = -std::numeric_limits<float>::infinity();
                bool has_valid = false;

                for (int j = 0; j < nr; ++j) {
                    if (!tile_all_true && !mask(qi + i, kj + j)) {
                        row_prob[i][j] = 0.0f;
                        continue;
                    }

                    const float score = score_tile[i][j] * scale + _flashBiasValue(bias, rsbias, csbias, qi + i, kj + j);
                    row_prob[i][j] = score;
                    tile_row_max = std::max(tile_row_max, score);
                    has_valid = true;
                }

                if (!has_valid) {
                    continue;
                }

                const float prev_max = row_max[i];
                const float new_max = std::max(prev_max, tile_row_max);
                const float prev_scale = std::isfinite(prev_max) ? std::exp(prev_max - new_max) : 0.0f;

                float tile_sum = 0.0f;
                for (int j = 0; j < nr; ++j) {
                    if (!tile_all_true && !mask(qi + i, kj + j)) {
                        continue;
                    }
                    const float prob = std::exp(row_prob[i][j] - new_max);
                    row_prob[i][j] = prob;
                    tile_sum += prob;
                }

                const float new_l = row_l[i] * prev_scale + tile_sum;
                if (new_l == 0.0f) {
                    continue;
                }

                row_new_max[i] = new_max;
                row_new_l[i] = new_l;
                row_rescale[i] = row_l[i] == 0.0f ? 0.0f : (row_l[i] * prev_scale / new_l);
            }

            for (int i = 0; i < mr; ++i) {
                if (row_new_l[i] == row_l[i] && row_new_max[i] == row_max[i]) {
                    continue;
                }

                float* out_row = O + (qi + i) * rso;
                const float inv_new_l = 1.0f / row_new_l[i];

                if (cso == 1 && csv == 1) {
                    _flashAccumulateRowContiguous(
                        out_row,
                        value_dim,
                        V + kj * rsv,
                        rsv,
                        row_prob[i],
                        nr,
                        row_rescale[i],
                        inv_new_l
                    );
                } else {
                    _flashScaleRow(out_row, value_dim, cso, row_rescale[i]);
                    for (int j = 0; j < nr; ++j) {
                        if (!tile_all_true && !mask(qi + i, kj + j)) {
                            continue;
                        }
                        _flashAccumulateRow(
                            out_row,
                            value_dim,
                            cso,
                            V,
                            kj + j,
                            rsv,
                            csv,
                            row_prob[i][j] * inv_new_l
                        );
                    }
                }

                row_max[i] = row_new_max[i];
                row_l[i] = row_new_l[i];
            }
        }
    }
}

inline void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int q_len,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t rsq,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t rso,
    int64_t cso,
    const bool* mask,
    int64_t mask_stride,
    const float* bias,
    int64_t rsbias,
    int64_t csbias
) {
    if (mask == nullptr) {
        if (q_len == 1) {
            _flashAttentionDecodeGemvImpl(
                Q, K, V, O,
                kv_len, head_dim, value_dim, scale,
                csq, rsk, csk, rsv, csv, cso,
                _FlashAllMask{},
                bias, rsbias, csbias
            );
            return;
        }
        _flashAttentionImpl(
            Q, K, V, O,
            q_len, kv_len, head_dim, value_dim, scale,
            rsq, csq, rsk, csk, rsv, csv, rso, cso,
            _FlashAllMask{},
            bias, rsbias, csbias
        );
        return;
    }

    if (q_len == 1) {
        _flashAttentionDecodeGemvImpl(
            Q, K, V, O,
            kv_len, head_dim, value_dim, scale,
            csq, rsk, csk, rsv, csv, cso,
            _FlashBoolMask{mask, mask_stride},
            bias, rsbias, csbias
        );
        return;
    }

    _flashAttentionImpl(
        Q, K, V, O,
        q_len, kv_len, head_dim, value_dim, scale,
        rsq, csq, rsk, csk, rsv, csv, rso, cso,
        _FlashBoolMask{mask, mask_stride},
        bias, rsbias, csbias
    );
}

template <typename Func>
inline void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int q_len,
    int kv_len,
    int head_dim,
    int value_dim,
    float scale,
    int64_t rsq,
    int64_t csq,
    int64_t rsk,
    int64_t csk,
    int64_t rsv,
    int64_t csv,
    int64_t rso,
    int64_t cso,
    Func&& mask,
    const float* bias,
    int64_t rsbias,
    int64_t csbias
) {
    static_assert(std::is_invocable_r_v<bool, std::decay_t<Func>, int, int>, "flash_attention mask func must be callable as bool(int, int)");
    if (q_len == 1) {
        _flashAttentionDecodeGemvImpl(
            Q, K, V, O,
            kv_len, head_dim, value_dim, scale,
            csq, rsk, csk, rsv, csv, cso,
            std::forward<Func>(mask),
            bias, rsbias, csbias
        );
        return;
    }
    _flashAttentionImpl(
        Q, K, V, O,
        q_len, kv_len, head_dim, value_dim, scale,
        rsq, csq, rsk, csk, rsv, csv, rso, cso,
        std::forward<Func>(mask),
        bias, rsbias, csbias
    );
}

}  // namespace yt::kernel::avx2

#endif  // __AVX2__ && __FMA__
