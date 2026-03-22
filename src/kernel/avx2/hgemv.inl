#pragma once

namespace yt::kernel::avx2 {

struct HgemvConvertCache {
    const yt::float16* src = nullptr;
    int rows = 0;
    int cols = 0;
    int64_t rs = 0;
    int64_t cs = 0;
    AlignedBuffer buf;

    bool match(const yt::float16* s, int r, int c, int64_t rs_, int64_t cs_) const {
        return src == s && rows == r && cols == c && rs == rs_ && cs == cs_;
    }

    void update(const yt::float16* s, int r, int c, int64_t rs_, int64_t cs_) {
        src = s; rows = r; cols = c; rs = rs_; cs = cs_;
    }
};

inline void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y,
    int m, int k,
    float alpha, float beta,
    int64_t rsa, int64_t csa,
    int64_t rsx,
    int64_t rsy
) {
    if (m <= 0 || k <= 0) return;

#if __F16C__
    if (rsa == k && csa == 1 && rsx == 1 && rsy == 1) {
        static thread_local AlignedBuffer x_buf;
        x_buf.ensure((size_t)k);
        float* fx = x_buf.data;
        f16_to_f32_block(x, fx, k);

#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(g_num_threads) if(g_num_threads > 1 && m >= 128)
#endif
        for (int i = 0; i < m; ++i) {
            const yt::float16* a_row_h = A + (size_t)i * (size_t)k;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= k; p += 8) {
                __m128i ah8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_row_h + p));
                __m256 af8 = _mm256_cvtph_ps(ah8);
                __m256 xf8 = _mm256_loadu_ps(fx + p);
                acc = _mm256_fmadd_ps(af8, xf8, acc);
            }
            float sum = hsum_ps(acc);
            for (; p < k; ++p) sum += static_cast<float>(a_row_h[p]) * fx[p];
            y[i] = yt::float16(alpha * sum + (beta != 0.0f ? beta * static_cast<float>(y[i]) : 0.0f));
        }
        return;
    }
#endif

    static thread_local HgemvConvertCache a_cache;
    static thread_local AlignedBuffer x_buf, y_buf;

    if (!a_cache.match(A, m, k, rsa, csa)) {
        a_cache.buf.ensure((size_t)m * (size_t)k);
        float* fA_build = a_cache.buf.data;
        if (csa == 1 && rsa == k) {
            for (int i = 0; i < m; ++i)
                f16_to_f32_block(A + (size_t)i * (size_t)k, fA_build + (size_t)i * (size_t)k, k);
        } else {
            for (int i = 0; i < m; ++i)
                for (int p = 0; p < k; ++p)
                    fA_build[(size_t)i * (size_t)k + (size_t)p] = static_cast<float>(A[(size_t)i * (size_t)rsa + (size_t)p * (size_t)csa]);
        }
        a_cache.update(A, m, k, rsa, csa);
    }

    x_buf.ensure((size_t)k);
    y_buf.ensure((size_t)m);
    float* fx = x_buf.data;
    float* fy = y_buf.data;

    if (rsx == 1) f16_to_f32_block(x, fx, k);
    else {
        for (int p = 0; p < k; ++p) fx[p] = static_cast<float>(x[p * rsx]);
    }

    gemv_col_simd(a_cache.buf.data, fx, fy, m, k, alpha, 0.0f, k, 1, 1, 1);

    for (int i = 0; i < m; ++i) {
        float r = fy[i] + (beta != 0.0f ? beta * static_cast<float>(y[i * rsy]) : 0.0f);
        y[i * rsy] = yt::float16(r);
    }
}

inline void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* z,
    int n, int k,
    float alpha, float beta,
    int64_t csx,
    int64_t rsb, int64_t csb,
    int64_t csz
) {
    if (n <= 0 || k <= 0) return;

    static thread_local HgemvConvertCache b_cache;
    static thread_local AlignedBuffer x_buf, z_buf;

    if (!b_cache.match(B, k, n, rsb, csb)) {
        b_cache.buf.ensure((size_t)k * (size_t)n);
        float* fB_build = b_cache.buf.data;
        if (csb == 1 && rsb == n) {
            for (int p = 0; p < k; ++p)
                f16_to_f32_block(B + (size_t)p * (size_t)n, fB_build + (size_t)p * (size_t)n, n);
        } else {
            for (int p = 0; p < k; ++p)
                for (int j = 0; j < n; ++j)
                    fB_build[(size_t)p * (size_t)n + (size_t)j] = static_cast<float>(B[(size_t)p * (size_t)rsb + (size_t)j * (size_t)csb]);
        }
        b_cache.update(B, k, n, rsb, csb);
    }

    x_buf.ensure((size_t)k);
    z_buf.ensure((size_t)n);
    float* fx = x_buf.data;
    float* fz = z_buf.data;

    if (csx == 1) f16_to_f32_block(x, fx, k);
    else {
        for (int p = 0; p < k; ++p) fx[p] = static_cast<float>(x[p * csx]);
    }

    gemv_row_simd(fx, b_cache.buf.data, fz, n, k, alpha, 0.0f, 1, n, 1, 1);

    for (int j = 0; j < n; ++j) {
        float r = fz[j] + (beta != 0.0f ? beta * static_cast<float>(z[j * csz]) : 0.0f);
        z[j * csz] = yt::float16(r);
    }
}

inline void hgemv(const yt::float16* A, const yt::float16* x, yt::float16* y,
                  int m, int k, float alpha, float beta) {
    hgemv_col(A, x, y, m, k, alpha, beta, k, 1, 1, 1);
}

} // namespace yt::kernel::avx2
