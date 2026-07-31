#pragma once
/***************
 * file: blas/gemm.inl
 * purpose: typed GEMM packing and blocking frames.
 ***************/

#include <algorithm>
#include <concepts>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace yt::blas {

enum class GemmMaskTileState { AllFalse, AllTrue, Partial };

inline void validateGemmWorkspace(const AlignedBuffer& buffer, size_t required) {
    if (buffer.data == nullptr || buffer.capacity < required) throw std::bad_alloc();
}

struct GemmAllTrueMaskPolicy {
    constexpr bool operator()(int, int) const { return true; }
    constexpr GemmMaskTileState tileState(int, int, int, int) const { return GemmMaskTileState::AllTrue; }
};

template <typename MaskType>
inline GemmMaskTileState classifyGemmMaskTile(const MaskType& mask, int row0, int column0, int mr, int nr) {
    if constexpr (requires {
                      { mask.tileState(row0, column0, mr, nr) } -> std::same_as<GemmMaskTileState>;
                  }) {
        return mask.tileState(row0, column0, mr, nr);
    } else if constexpr (requires {
                             { mask.tileAllFalse(row0, column0, mr, nr) } -> std::convertible_to<bool>;
                             { mask.tileAllTrue(row0, column0, mr, nr) } -> std::convertible_to<bool>;
                         }) {
        if (mask.tileAllFalse(row0, column0, mr, nr)) return GemmMaskTileState::AllFalse;
        if (mask.tileAllTrue(row0, column0, mr, nr)) return GemmMaskTileState::AllTrue;
        return GemmMaskTileState::Partial;
    } else {
        bool any_true = false;
        bool all_true = true;
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                const bool value = mask(row0 + i, column0 + j);
                any_true = any_true || value;
                all_true = all_true && value;
            }
        }
        if (!any_true) return GemmMaskTileState::AllFalse;
        return all_true ? GemmMaskTileState::AllTrue : GemmMaskTileState::Partial;
    }
}

template <
    typename CType, typename ComputeType, GemmKernelSpec Spec, typename Dispatcher, typename MaskType,
    typename PackA, typename PackB>
inline void packedGemmFrame(
    const BlasContext& context, CType* C, int m, int n, int k, ComputeType alpha, ComputeType beta,
    int64_t rsc, int64_t csc, const MaskType& mask, PackA&& pack_a, PackB&& pack_b
) {
    static_assert(std::is_same_v<ComputeType, float>);
    static_assert(Spec.row_mr > 0 && Spec.row_nr > 0 && Spec.vector_lanes > 0);
    static_assert(Spec.row_nr % Spec.vector_lanes == 0);

    static thread_local AlignedBuffer a_workspace, b_workspace, c_workspace;
    const int block_mc = align_down_to(gemm_mc(context), Spec.row_mr);
    const int block_kc = std::max(1, gemm_kc(context));
    const int block_nc = align_down_to(gemm_nc(context), Spec.row_nr);
    const size_t a_size =
        ((static_cast<size_t>(block_mc) + Spec.row_mr - 1) / Spec.row_mr) * Spec.row_mr * block_kc;
    const size_t b_size =
        ((static_cast<size_t>(block_nc) + Spec.row_nr - 1) / Spec.row_nr) * Spec.row_nr * block_kc;
    a_workspace.ensure(a_size);
    b_workspace.ensure(b_size);
    validateGemmWorkspace(a_workspace, a_size);
    validateGemmWorkspace(b_workspace, b_size);

#ifdef _OPENMP
    const int thread_count = get_num_threads();
    std::unique_ptr<AlignedBuffer[]> thread_a_workspaces;
    const size_t thread_a_size = static_cast<size_t>(Spec.row_mr) * block_kc;
    if (thread_count > 1) {
        thread_a_workspaces = std::make_unique<AlignedBuffer[]>(thread_count);
        for (int thread = 0; thread < thread_count; ++thread) {
            thread_a_workspaces[thread].ensure(thread_a_size);
            validateGemmWorkspace(thread_a_workspaces[thread], thread_a_size);
        }
    }
#endif

    ComputeType* deferred_output = nullptr;
    if constexpr (!std::is_same_v<CType, ComputeType>) {
        if (static_cast<size_t>(m) > std::numeric_limits<size_t>::max() / static_cast<size_t>(n)) {
            throw std::overflow_error("gemm: output size overflow");
        }
        const size_t output_size = static_cast<size_t>(m) * static_cast<size_t>(n);
        c_workspace.ensure(output_size);
        validateGemmWorkspace(c_workspace, output_size);
        deferred_output = c_workspace.data;
    }

    const bool direct_row_output =
        std::is_same_v<CType, ComputeType> && csc == 1 && rsc >= n && rsc <= std::numeric_limits<int>::max();

    auto computeBlock = [&](int ii, int jj, int pp, int mc, int nc, int kc, bool first, ComputeType* packed_a,
                            ComputeType* packed_b) {
        pack_a(packed_a, ii, pp, mc, kc);
        for (int ir = 0; ir < mc;) {
            const int mr = std::min(Spec.row_mr, mc - ir);
            for (int jr = 0; jr < nc;) {
                const int nr = std::min(Spec.row_nr, nc - jr);
                const int next_jr = jr + nr;
                const ComputeType* pa = packed_a + static_cast<size_t>(ir / Spec.row_mr) * Spec.row_mr * kc;
                const ComputeType* pb = packed_b + static_cast<size_t>(jr / Spec.row_nr) * Spec.row_nr * kc;
                const bool full_tile = mr == Spec.row_mr && nr == Spec.row_nr;
                const int row0 = ii + ir;
                const int column0 = jj + jr;
                const GemmMaskTileState mask_state = classifyGemmMaskTile(mask, row0, column0, mr, nr);
                if (mask_state == GemmMaskTileState::AllFalse) {
                    jr = next_jr;
                    continue;
                }
                const bool all_true = mask_state == GemmMaskTileState::AllTrue;

                if constexpr (std::is_same_v<CType, ComputeType>) {
                    ComputeType* cij = C + row0 * rsc + column0 * csc;
                    if (all_true && full_tile && direct_row_output && alpha == ComputeType(1)) {
                        if (first && beta == ComputeType(0)) {
                            Dispatcher::template compute<false>(pa, pb, cij, kc, static_cast<int>(rsc));
                        } else {
                            if (first && beta != ComputeType(1)) {
                                for (int i = 0; i < mr; ++i)
                                    for (int j = 0; j < nr; ++j) cij[i * rsc + j] *= beta;
                            }
                            Dispatcher::template compute<true>(pa, pb, cij, kc, static_cast<int>(rsc));
                        }
                        jr = next_jr;
                        continue;
                    }

                    alignas(64) ComputeType tile[Spec.row_mr * Spec.row_nr];
                    Dispatcher::template compute<false>(pa, pb, tile, kc, Spec.row_nr);
                    for (int i = 0; i < mr; ++i) {
                        for (int j = 0; j < nr; ++j) {
                            if (!mask(row0 + i, column0 + j)) continue;
                            ComputeType* cp = cij + i * rsc + j * csc;
                            ComputeType value = alpha * tile[i * Spec.row_nr + j];
                            if (first && beta != ComputeType(0))
                                value += beta * (*cp);
                            else if (!first)
                                value += *cp;
                            *cp = value;
                        }
                    }
                } else {
                    ComputeType* cij =
                        deferred_output + static_cast<size_t>(row0) * n + static_cast<size_t>(column0);
                    if (full_tile) {
                        if (first)
                            Dispatcher::template compute<false>(pa, pb, cij, kc, n);
                        else
                            Dispatcher::template compute<true>(pa, pb, cij, kc, n);
                        jr = next_jr;
                        continue;
                    }

                    alignas(64) ComputeType tile[Spec.row_mr * Spec.row_nr];
                    Dispatcher::template compute<false>(pa, pb, tile, kc, Spec.row_nr);
                    for (int i = 0; i < mr; ++i) {
                        for (int j = 0; j < nr; ++j) {
                            ComputeType& value = cij[static_cast<size_t>(i) * n + j];
                            if (first)
                                value = tile[i * Spec.row_nr + j];
                            else
                                value += tile[i * Spec.row_nr + j];
                        }
                    }
                }
                jr = next_jr;
            }
            ir += mr;
        }
    };

    for (int jj = 0; jj < n;) {
        const int nc = std::min(block_nc, n - jj);
        for (int pp = 0; pp < k;) {
            const int kc = std::min(block_kc, k - pp);
            const bool first = pp == 0;
            pack_b(b_workspace.data, jj, pp, nc, kc);
            ComputeType* packed_b = b_workspace.data;

#ifdef _OPENMP
            if (thread_count > 1) {
                const int m_tiles = m / Spec.row_mr + (m % Spec.row_mr != 0);
                std::exception_ptr worker_error;
#pragma omp parallel num_threads(thread_count) proc_bind(spread)
                {
                    ComputeType* thread_a_workspace = thread_a_workspaces[omp_get_thread_num()].data;
#pragma omp for schedule(static)
                    for (int tile = 0; tile < m_tiles; ++tile) {
                        const int ii = tile * Spec.row_mr;
                        const int mc = std::min(Spec.row_mr, m - ii);
                        try {
                            computeBlock(ii, jj, pp, mc, nc, kc, first, thread_a_workspace, packed_b);
                        } catch (...) {
#pragma omp critical(yt_packed_gemm_exception)
                            {
                                if (worker_error == nullptr) worker_error = std::current_exception();
                            }
                        }
                    }
                }
                if (worker_error != nullptr) std::rethrow_exception(worker_error);
            } else
#endif
            {
                for (int ii = 0; ii < m;) {
                    const int mc = std::min(block_mc, m - ii);
                    computeBlock(ii, jj, pp, mc, nc, kc, first, a_workspace.data, packed_b);
                    ii += mc;
                }
            }
            pp += kc;
        }
        jj += nc;
    }

    if constexpr (!std::is_same_v<CType, ComputeType>) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!mask(i, j)) continue;
                ComputeType value = alpha * deferred_output[static_cast<size_t>(i) * n + j];
                if (beta != ComputeType(0)) value += beta * static_cast<ComputeType>(C[i * rsc + j * csc]);
                C[i * rsc + j * csc] = static_cast<CType>(value);
            }
        }
    }
}

template <
    typename AType, typename BType, typename CType, typename ComputeType, GemmKernelSpec Spec,
    typename Dispatcher, typename MaskType>
inline void typedGemmFrame(
    const BlasContext& context, const AType* A, const BType* B, CType* C, int m, int n, int k,
    ComputeType alpha, ComputeType beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc, const MaskType& mask
) {
    auto pack_a = [&](ComputeType* packed, int ii, int pp, int mc, int kc) {
        Dispatcher::template packA<AType>(A + ii * rsa + pp * csa, packed, mc, kc, rsa, csa);
    };
    auto pack_b = [&](ComputeType* packed, int jj, int pp, int nc, int kc) {
        Dispatcher::template packB<BType>(B + pp * rsb + jj * csb, packed, kc, nc, rsb, csb);
    };
    packedGemmFrame<CType, ComputeType, Spec, Dispatcher>(
        context, C, m, n, k, alpha, beta, rsc, csc, mask, pack_a, pack_b
    );
}

template <
    typename AType, typename BType, typename CType, typename ComputeType, GemmKernelSpec Spec,
    typename Dispatcher>
inline void gemm(
    const BlasContext& context, const AType* A, const BType* B, CType* C, int m, int n, int k,
    ComputeType alpha, ComputeType beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc
) {
    gemm_masked<AType, BType, CType, ComputeType, Spec, Dispatcher>(
        context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc, GemmAllTrueMaskPolicy{}
    );
}

}  // namespace yt::blas
