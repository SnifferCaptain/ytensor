#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace yt::blas {

struct FlashAllVisibleMask {
    inline bool operator()(int, int) const { return true; }
    inline bool tileAllTrue(int, int, int, int) const { return true; }
    inline bool tileAllFalse(int, int, int, int) const { return false; }
};

struct FlashBooleanMask {
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

inline float attentionBiasValue(const float* bias, int64_t rsbias, int64_t csbias, int row, int col) {
    if (bias == nullptr) {
        return 0.0f;
    }
    return bias[row * rsbias + col * csbias];
}

inline void zeroAttentionOutputRow(float* row, int value_dim, int64_t stride) {
    setv(value_dim, 0.0f, row, stride);
}

template <typename MaskOp>
inline bool attentionMaskTileAllTrue(const MaskOp& mask, int row0, int col0, int mr, int nr) {
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
inline bool attentionMaskTileAllFalse(const MaskOp& mask, int row0, int col0, int mr, int nr) {
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

inline void accumulateAttentionValueRows(
    float* out_row,
    int value_dim,
    int64_t cso,
    const float* V,
    int64_t rsv,
    int64_t csv,
    const float* probs,
    int nr,
    float rescale,
    float inv_new_l
) {
    scalv(value_dim, rescale, out_row, cso);
    const BlasContext& context = defaultBlasContext();
    const int fusion_width = std::max(1, context.af);
    int j = 0;
    while (j < nr) {
        while (j < nr && probs[j] == 0.0f) {
            ++j;
        }
        const int first = j;
        while (j < nr && probs[j] != 0.0f && j - first < fusion_width) {
            ++j;
        }
        const int f = j - first;
        if (f == 0) {
            continue;
        }
        axpyf(
            context,
            value_dim,
            f,
            inv_new_l,
            V + first * rsv,
            csv,
            rsv,
            probs + first,
            1,
            out_row,
            cso
        );
    }
}

template <typename MaskOp>
void computeFlashAttentionDecodeGemv(
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
    if (score_buf.capacity < static_cast<size_t>(kv_len)) {
        throw std::bad_alloc();
    }
    float* scores = score_buf.data;

    gemv_row_simd(Q, K, scores, kv_len, head_dim, scale, 0.0f, csq, csk, rsk, 1);

    float row_max = -std::numeric_limits<float>::infinity();
    bool any_visible = false;
    for (int col = 0; col < kv_len; ++col) {
        if (!mask(0, col)) {
            scores[col] = -std::numeric_limits<float>::infinity();
            continue;
        }
        scores[col] += attentionBiasValue(bias, rsbias, csbias, 0, col);
        row_max = std::max(row_max, scores[col]);
        any_visible = true;
    }

    if (!any_visible) {
        zeroAttentionOutputRow(O, value_dim, cso);
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
        zeroAttentionOutputRow(O, value_dim, cso);
        return;
    }

    scalv(kv_len, 1.0f / row_sum, scores, 1);

    gemv_row_simd(scores, V, O, value_dim, kv_len, 1.0f, 0.0f, 1, rsv, csv, cso);
}

template <
    typename MaskOp,
    GemmKernelSpec Spec = default_gemm_kernel_spec,
    typename Dispatcher = typename DefaultGemmDispatcher<float, float, float, float, Spec>::type>
void computeTiledFlashAttention(
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

    constexpr int row_mr = Spec.row_mr;
    constexpr int row_nr = Spec.row_nr;
    const int q_blocks = (q_len + row_mr - 1) / row_mr;
    const int kv_blocks = (kv_len + row_nr - 1) / row_nr;
    const int worker_count = std::max(1, std::min(q_blocks, get_num_threads()));
    const size_t packed_k_size = static_cast<size_t>(kv_blocks) * static_cast<size_t>(row_nr) *
                                 static_cast<size_t>(head_dim);
    const size_t packed_q_size = static_cast<size_t>(worker_count) * static_cast<size_t>(row_mr) *
                                 static_cast<size_t>(head_dim);

    AlignedBuffer k_pack_buf;
    k_pack_buf.ensure(packed_k_size);
    if (k_pack_buf.capacity < packed_k_size) {
        throw std::bad_alloc();
    }
    Dispatcher::template packB<float>(K, k_pack_buf.data, head_dim, kv_len, csk, rsk);

    AlignedBuffer q_pack_buf;
    q_pack_buf.ensure(packed_q_size);
    if (q_pack_buf.capacity < packed_q_size) {
        throw std::bad_alloc();
    }

    #pragma omp parallel for schedule(static) if(worker_count > 1) \
        num_threads(worker_count) proc_bind(close)
    for (int q_block = 0; q_block < q_blocks; ++q_block) {
        const int qi = q_block * row_mr;
        const int mr = std::min(row_mr, q_len - qi);
        int worker_index = 0;
#ifdef _OPENMP
        worker_index = omp_get_thread_num();
#endif
        float* packed_q = q_pack_buf.data + static_cast<size_t>(worker_index) * row_mr * head_dim;
        Dispatcher::template packA<float>(Q + qi * rsq, packed_q, mr, head_dim, rsq, csq);

        alignas(64) float score_tile[row_mr][row_nr];
        alignas(64) float row_prob[row_mr][row_nr];
        alignas(64) float row_max[row_mr];
        alignas(64) float row_l[row_mr];
        alignas(64) float row_new_max[row_mr];
        alignas(64) float row_new_l[row_mr];
        alignas(64) float row_rescale[row_mr];

        for (int i = 0; i < mr; ++i) {
            row_max[i] = -std::numeric_limits<float>::infinity();
            row_l[i] = 0.0f;
            zeroAttentionOutputRow(O + (qi + i) * rso, value_dim, cso);
        }

        for (int kj = 0; kj < kv_len; kj += row_nr) {
            const int nr = std::min(row_nr, kv_len - kj);
            if (attentionMaskTileAllFalse(mask, qi, kj, mr, nr)) {
                continue;
            }

            const float* packed_k =
                k_pack_buf.data + static_cast<size_t>(kj / row_nr) * row_nr * head_dim;
            Dispatcher::template compute<false>(packed_q, packed_k, &score_tile[0][0], head_dim, row_nr);

            for (int i = 0; i < mr; ++i) {
                row_new_max[i] = row_max[i];
                row_new_l[i] = row_l[i];
                row_rescale[i] = 0.0f;
            }

            const bool tile_all_true = attentionMaskTileAllTrue(mask, qi, kj, mr, nr);

            for (int i = 0; i < mr; ++i) {
                float tile_row_max = -std::numeric_limits<float>::infinity();
                bool has_valid = false;

                for (int j = 0; j < nr; ++j) {
                    if (!tile_all_true && !mask(qi + i, kj + j)) {
                        row_prob[i][j] = 0.0f;
                        continue;
                    }

                    const float score = score_tile[i][j] * scale +
                                        attentionBiasValue(bias, rsbias, csbias, qi + i, kj + j);
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

                accumulateAttentionValueRows(
                    out_row,
                    value_dim,
                    cso,
                    V + kj * rsv,
                    rsv,
                    csv,
                    row_prob[i],
                    nr,
                    row_rescale[i],
                    inv_new_l
                );

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
            computeFlashAttentionDecodeGemv(
                Q, K, V, O,
                kv_len, head_dim, value_dim, scale,
                csq, rsk, csk, rsv, csv, cso,
                FlashAllVisibleMask{},
                bias, rsbias, csbias
            );
            return;
        }
        computeTiledFlashAttention(
            Q, K, V, O,
            q_len, kv_len, head_dim, value_dim, scale,
            rsq, csq, rsk, csk, rsv, csv, rso, cso,
            FlashAllVisibleMask{},
            bias, rsbias, csbias
        );
        return;
    }

    if (q_len == 1) {
        computeFlashAttentionDecodeGemv(
            Q, K, V, O,
            kv_len, head_dim, value_dim, scale,
            csq, rsk, csk, rsv, csv, cso,
            FlashBooleanMask{mask, mask_stride},
            bias, rsbias, csbias
        );
        return;
    }

    computeTiledFlashAttention(
        Q, K, V, O,
        q_len, kv_len, head_dim, value_dim, scale,
        rsq, csq, rsk, csk, rsv, csv, rso, cso,
        FlashBooleanMask{mask, mask_stride},
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
        computeFlashAttentionDecodeGemv(
            Q, K, V, O,
            kv_len, head_dim, value_dim, scale,
            csq, rsk, csk, rsv, csv, cso,
            std::forward<Func>(mask),
            bias, rsbias, csbias
        );
        return;
    }
    computeTiledFlashAttention(
        Q, K, V, O,
        q_len, kv_len, head_dim, value_dim, scale,
        rsq, csq, rsk, csk, rsv, csv, rso, cso,
        std::forward<Func>(mask),
        bias, rsbias, csbias
    );
}

}  // namespace yt::blas
