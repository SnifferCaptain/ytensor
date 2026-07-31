#pragma once
/***************
 * file: blas/kernels/avx2/pack_b.inl
 * purpose: typed GEMM B panel packing与storage转换。
 ***************/

namespace yt::blas {

template <typename SourceType, typename ComputeType, GemmKernelSpec Spec>
inline void pack_b_typed_panel(
    const SourceType* B, ComputeType* packed, int kc, int nc, int64_t rsb, int64_t csb
) {
    static_assert(Spec.row_nr > 0);
    for (int j = 0; j < nc; j += Spec.row_nr) {
        const int nr = std::min(Spec.row_nr, nc - j);
        for (int p = 0; p < kc; ++p) {
            for (int jj = 0; jj < nr; ++jj) {
                packed[jj] = static_cast<ComputeType>(B[p * rsb + (j + jj) * csb]);
            }
            for (int jj = nr; jj < Spec.row_nr; ++jj) packed[jj] = ComputeType(0);
            packed += Spec.row_nr;
        }
    }
}

inline void pack_b_generic(const float* B, float* packed, int kc, int nc, int64_t rsb, int64_t csb) {
    pack_b_typed_panel<float, float, default_gemm_kernel_spec>(B, packed, kc, nc, rsb, csb);
}

}  // namespace yt::blas
