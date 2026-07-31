#pragma once
/***************
 * file: blas/gemv_row.inl
 * purpose: typed行式GEMV实现。
 ***************/

namespace yt::blas {

template <typename T>
inline void gemv_row(
    const T* x, const T* B, T* y, int n, int k, float alpha, float beta, int64_t incx, int64_t rsb,
    int64_t csb, int64_t incy, const BlasContext& context
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, yt::float16>);
    if (n <= 0) return;

    if (rsb == 1) {
        int df = std::max(1, context.df);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(g_num_threads) if (g_num_threads > 1 && n >= 64)
#endif
        for (int j = 0; j < n; j += df) {
            int fuse = std::min(df, n - j);
            if constexpr (std::is_same_v<T, float>) {
                dotxf(context, k, fuse, alpha, x, incx, B + j * csb, rsb, csb, beta, y + j * incy, incy);
            } else {
                // dotxf快速实例最多融合四个输出，更大的df逐输出回退。
                float values[4] = {};
                if (fuse <= 4) {
                    if (beta != 0.0f) {
                        for (int jj = 0; jj < fuse; ++jj) {
                            values[jj] = static_cast<float>(y[(j + jj) * incy]);
                        }
                    }
                    dotxf(context, k, fuse, alpha, x, incx, B + j * csb, rsb, csb, beta, values, 1);
                    for (int jj = 0; jj < fuse; ++jj) y[(j + jj) * incy] = T(values[jj]);
                } else {
                    for (int jj = 0; jj < fuse; ++jj) {
                        float value = beta == 0.0f ? 0.0f : static_cast<float>(y[(j + jj) * incy]);
                        value = dotxv(k, alpha, x, incx, B + (j + jj) * csb, rsb, beta, value);
                        y[(j + jj) * incy] = T(value);
                    }
                }
            }
        }
        return;
    }

    constexpr int outputBlock = 256;
    int af = std::max(1, context.af);
    static thread_local std::vector<float> output;
    float* outputData;
    int64_t outputStride;
    if constexpr (std::is_same_v<T, float>) {
        outputData = y;
        outputStride = incy;
    } else {
        output.resize(static_cast<size_t>(n));
        outputData = output.data();
        outputStride = 1;
        if (beta == 0.0f) {
            setv(n, 0.0f, outputData, 1);
        } else {
            for (int j = 0; j < n; ++j) outputData[j] = static_cast<float>(y[j * incy]);
            scalv(n, beta, outputData, 1);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(g_num_threads) if (g_num_threads > 1 && n >= outputBlock)
#endif
    for (int j0 = 0; j0 < n; j0 += outputBlock) {
        int width = std::min(outputBlock, n - j0);
        float* yBlock = outputData + j0 * outputStride;
        if constexpr (std::is_same_v<T, float>) scalv(width, beta, yBlock, outputStride);
        if (k <= 0 || alpha == 0.0f) continue;
        for (int p = 0; p < k; p += af) {
            int fuse = std::min(af, k - p);
            axpyf(
                context, width, fuse, alpha, B + p * rsb + j0 * csb, csb, rsb, x + p * incx, incx,
                yBlock, outputStride
            );
        }
    }

    if constexpr (std::is_same_v<T, yt::float16>) {
        for (int j = 0; j < n; ++j) y[j * incy] = T(outputData[j]);
    }
}

inline void gemv_row_simd(
    const float* x, const float* B, float* y, int n, int k, float alpha, float beta, int64_t incx,
    int64_t rsb, int64_t csb, int64_t incy, const BlasContext& context
) {
    gemv_row(x, B, y, n, k, alpha, beta, incx, rsb, csb, incy, context);
}

inline void hgemv_row(
    const yt::float16* x, const yt::float16* B, yt::float16* y, int n, int k, float alpha, float beta,
    int64_t incx, int64_t rsb, int64_t csb, int64_t incy, const BlasContext& context
) {
    gemv_row(x, B, y, n, k, alpha, beta, incx, rsb, csb, incy, context);
}

}  // namespace yt::blas
