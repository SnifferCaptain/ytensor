#pragma once
/***************
 * file: blas/kernels/avx2/gemm.inl
 * purpose: AVX2 GEMM物理微内核。
 ***************/

#if defined(__AVX2__) && defined(__FMA__)

namespace yt::blas {

// ==================== FMA核心 ====================

template <int NB, int Row, int... Blocks>
__attribute__((always_inline)) inline void
fma_row_blocks(__m256 a, const __m256 b[], __m256 c[][NB], std::integer_sequence<int, Blocks...>) {
    ((c[Row][Blocks] = _mm256_fmadd_ps(a, b[Blocks], c[Row][Blocks])), ...);
}

template <int MR, int NB, int... Rows>
__attribute__((always_inline)) inline void
fma_row_step(const float* __restrict A, const __m256 b[], __m256 c[][NB], std::integer_sequence<int, Rows...>) {
    ((fma_row_blocks<NB, Rows>(_mm256_broadcast_ss(A + Rows), b, c, std::make_integer_sequence<int, NB>{})),
     ...);
}

template <GemmKernelSpec Spec, int... Blocks>
__attribute__((always_inline)) inline void
load_b_vectors(__m256 b[], const float* B, std::integer_sequence<int, Blocks...>) {
    ((b[Blocks] = _mm256_loadu_ps(B + Blocks * Spec.vector_lanes)), ...);
}

template <int MR>
__attribute__((hot)) inline void fma_loop_row(
    const float* __restrict A, const float* __restrict B, __m256 c[MR_ROW][NR_BLOCKS], int kc
) {
    static_assert(MR >= 1 && MR <= MR_ROW, "MR must be in [1, MR_ROW]");
    auto seq = std::make_integer_sequence<int, MR>{};
    for (int p = 0; p < kc; ++p) {
        __m256 b[NR_BLOCKS];
        for (int jj = 0; jj < NR_BLOCKS; ++jj) b[jj] = _mm256_loadu_ps(B + p * NR_ROW + jj * 8);
        fma_row_step<MR, NR_BLOCKS>(A + p * MR_ROW, b, c, seq);
    }
}

template <GemmKernelSpec Spec, int... Indices>
__attribute__((always_inline)) inline void
load_row_accumulators(__m256 c[][Spec.row_nr / Spec.vector_lanes], const float* C, int ldc, std::integer_sequence<int, Indices...>) {
    constexpr int blocks = Spec.row_nr / Spec.vector_lanes;
    ((c[Indices / blocks][Indices % blocks] =
          _mm256_loadu_ps(C + (Indices / blocks) * ldc + (Indices % blocks) * Spec.vector_lanes)),
     ...);
}

template <GemmKernelSpec Spec, int... Indices>
__attribute__((always_inline)) inline void
store_row_accumulators(const __m256 c[][Spec.row_nr / Spec.vector_lanes], float* C, int ldc, std::integer_sequence<int, Indices...>) {
    constexpr int blocks = Spec.row_nr / Spec.vector_lanes;
    ((_mm256_storeu_ps(
         C + (Indices / blocks) * ldc + (Indices % blocks) * Spec.vector_lanes,
         c[Indices / blocks][Indices % blocks]
     )),
     ...);
}

template <typename ComputeType, GemmKernelSpec Spec, bool Accumulate>
__attribute__((always_inline, hot)) inline void row_microkernel(
    const ComputeType* __restrict A, const ComputeType* __restrict B, ComputeType* __restrict C, int kc,
    int ldc
) {
    static_assert(std::is_same_v<ComputeType, float>, "AVX2 row microkernel requires float compute");
    static_assert(Spec.row_mr > 0 && Spec.row_nr > 0, "microkernel tile must be non-empty");
    static_assert(Spec.vector_lanes == 8, "AVX2 float microkernel requires eight lanes");
    static_assert(
        Spec.row_nr % Spec.vector_lanes == 0, "microkernel columns must be divisible by vector lanes"
    );
    constexpr int blocks = Spec.row_nr / Spec.vector_lanes;
    constexpr int accumulators = Spec.row_mr * blocks;
    __m256 c[Spec.row_mr][blocks] = {};
    if constexpr (Accumulate) {
        load_row_accumulators<Spec>(c, C, ldc, std::make_integer_sequence<int, accumulators>{});
    }
    for (int p = 0; p < kc; ++p) {
        __m256 b[blocks];
        load_b_vectors<Spec>(b, B + p * Spec.row_nr, std::make_integer_sequence<int, blocks>{});
        fma_row_step<Spec.row_mr, blocks>(
            A + p * Spec.row_mr, b, c, std::make_integer_sequence<int, Spec.row_mr>{}
        );
    }
    store_row_accumulators<Spec>(c, C, ldc, std::make_integer_sequence<int, accumulators>{});
}

// ==================== 编译期边界分发 ====================

template <int MAX_MR>
inline void dispatch_fma_row(const float* A, const float* B, __m256 c[MR_ROW][NR_BLOCKS], int mr, int kc) {
    if (mr == MAX_MR)
        fma_loop_row<MAX_MR>(A, B, c, kc);
    else if constexpr (MAX_MR > 1)
        dispatch_fma_row<MAX_MR - 1>(A, B, c, mr, kc);
}

}  // namespace yt::blas

#endif  // __AVX2__ && __FMA__
