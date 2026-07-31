#pragma once
/***************
 * file: blas/gemm_masked.inl
 * purpose: GEMM mask policies and masked typed GEMM frontend.
 ***************/

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace yt::blas {

struct GemmDenseBoolMaskPolicy {
    const bool* values;
    int columns;

    bool operator()(int row, int column) const { return values[row * columns + column]; }

    GemmMaskTileState tileState(int row0, int column0, int mr, int nr) const {
        bool any_true = false;
        bool all_true = true;
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                const bool value = (*this)(row0 + i, column0 + j);
                any_true = any_true || value;
                all_true = all_true && value;
            }
        }
        if (!any_true) return GemmMaskTileState::AllFalse;
        return all_true ? GemmMaskTileState::AllTrue : GemmMaskTileState::Partial;
    }
};

template <
    typename AType, typename BType, typename CType, typename ComputeType, GemmKernelSpec Spec,
    typename Dispatcher, typename MaskType>
inline void gemm_masked(
    const BlasContext& context, const AType* A, const BType* B, CType* C, int m, int n, int k,
    ComputeType alpha, ComputeType beta, int64_t rsa, int64_t csa, int64_t rsb, int64_t csb, int64_t rsc,
    int64_t csc, MaskType&& mask
) {
    static_assert(is_supported_gemm_storage_v<AType>);
    static_assert(is_supported_gemm_storage_v<BType>);
    static_assert(is_supported_gemm_storage_v<CType>);
    static_assert(std::is_same_v<ComputeType, float>);
    static_assert(std::is_invocable_r_v<bool, const std::decay_t<MaskType>&, int, int>);
    auto&& mask_policy = std::forward<MaskType>(mask);
    if (m < 0 || n < 0 || k < 0) throw std::invalid_argument("gemm: dimensions must be non-negative");
    if (m == 0 || n == 0) return;
    if (k == 0 || alpha == ComputeType(0)) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!mask_policy(i, j)) continue;
                CType* cp = C + i * rsc + j * csc;
                if (beta == ComputeType(0))
                    *cp = CType(0);
                else if (beta != ComputeType(1))
                    *cp = static_cast<CType>(beta * static_cast<ComputeType>(*cp));
            }
        }
        return;
    }
    typedGemmFrame<AType, BType, CType, ComputeType, Spec, Dispatcher>(
        context, A, B, C, m, n, k, alpha, beta, rsa, csa, rsb, csb, rsc, csc, mask_policy
    );
}

}  // namespace yt::blas
