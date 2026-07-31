#pragma once
/***************
 * file: blas/kernels/avx2/pack_a.inl
 * purpose: typed GEMM A panel packing与storage转换。
 ***************/

namespace yt::blas {

template <typename SourceType, typename ComputeType, GemmKernelSpec Spec>
inline void pack_a_typed_panel(
    const SourceType* A, ComputeType* packed, int mc, int kc, int64_t rsa, int64_t csa
) {
    static_assert(Spec.row_mr > 0);
    for (int i = 0; i < mc; i += Spec.row_mr) {
        const int mr = std::min(Spec.row_mr, mc - i);
        for (int p = 0; p < kc; ++p) {
            for (int ii = 0; ii < mr; ++ii) {
                packed[ii] = static_cast<ComputeType>(A[(i + ii) * rsa + p * csa]);
            }
            for (int ii = mr; ii < Spec.row_mr; ++ii) packed[ii] = ComputeType(0);
            packed += Spec.row_mr;
        }
    }
}

inline void pack_a_generic(const float* A, float* packed, int mc, int kc, int64_t rsa, int64_t csa) {
    pack_a_typed_panel<float, float, default_gemm_kernel_spec>(A, packed, mc, kc, rsa, csa);
}

}  // namespace yt::blas
