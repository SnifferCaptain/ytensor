#pragma once
/***************
 * file: blas/gemv_col.inl
 * purpose: typed列式GEMV实现。
 ***************/

namespace yt::blas {

template <typename T>
inline void gemv_col(
    const T* A, const T* x, T* y, int m, int k, float alpha, float beta, int64_t rsa, int64_t csa,
    int64_t incx, int64_t incy, const BlasContext& context
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, yt::float16>);
    if (m <= 0) return;

    if (rsa == 1) {
        constexpr int outputBlock = 256;
        int af = std::max(1, context.af);
        static thread_local std::vector<float> output;
        float* outputData;
        int64_t outputStride;
        if constexpr (std::is_same_v<T, float>) {
            outputData = y;
            outputStride = incy;
        } else {
            output.resize(static_cast<size_t>(m));
            outputData = output.data();
            outputStride = 1;
            if (beta == 0.0f) {
                setv(m, 0.0f, outputData, 1);
            } else {
                for (int i = 0; i < m; ++i) outputData[i] = static_cast<float>(y[i * incy]);
                scalv(m, beta, outputData, 1);
            }
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    num_threads(g_num_threads) if (g_num_threads > 1 && m >= outputBlock)
#endif
        for (int i0 = 0; i0 < m; i0 += outputBlock) {
            int height = std::min(outputBlock, m - i0);
            float* yBlock = outputData + i0 * outputStride;
            if constexpr (std::is_same_v<T, float>) scalv(height, beta, yBlock, outputStride);
            for (int p = 0; p < k; p += af) {
                int fuse = std::min(af, k - p);
                axpyf(
                    context, height, fuse, alpha, A + i0 * rsa + p * csa, rsa, csa, x + p * incx, incx,
                    yBlock, outputStride
                );
            }
        }

        if constexpr (std::is_same_v<T, yt::float16>) {
            for (int i = 0; i < m; ++i) y[i * incy] = T(outputData[i]);
        }
        return;
    }

    int df = std::max(1, context.df);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(g_num_threads) if (g_num_threads > 1 && m >= 128)
#endif
    for (int i = 0; i < m; i += df) {
        int fuse = std::min(df, m - i);
        if constexpr (std::is_same_v<T, float>) {
            dotxf(context, k, fuse, alpha, x, incx, A + i * rsa, csa, rsa, beta, y + i * incy, incy);
        } else {
            // dotxf快速实例最多融合四个输出，更大的df逐输出回退。
            float values[4] = {};
            if (fuse <= 4) {
                if (beta != 0.0f) {
                    for (int j = 0; j < fuse; ++j) {
                        values[j] = static_cast<float>(y[(i + j) * incy]);
                    }
                }
                dotxf(context, k, fuse, alpha, x, incx, A + i * rsa, csa, rsa, beta, values, 1);
                for (int j = 0; j < fuse; ++j) y[(i + j) * incy] = T(values[j]);
            } else {
                for (int j = 0; j < fuse; ++j) {
                    float value = beta == 0.0f ? 0.0f : static_cast<float>(y[(i + j) * incy]);
                    value = dotxv(k, alpha, x, incx, A + (i + j) * rsa, csa, beta, value);
                    y[(i + j) * incy] = T(value);
                }
            }
        }
    }
}

inline void gemv_col_simd(
    const float* A, const float* x, float* y, int m, int k, float alpha, float beta, int64_t rsa, int64_t csa,
    int64_t incx, int64_t incy, const BlasContext& context
) {
    gemv_col(A, x, y, m, k, alpha, beta, rsa, csa, incx, incy, context);
}

inline void hgemv_col(
    const yt::float16* A, const yt::float16* x, yt::float16* y, int m, int k, float alpha, float beta,
    int64_t rsa, int64_t csa, int64_t incx, int64_t incy, const BlasContext& context
) {
    gemv_col(A, x, y, m, k, alpha, beta, rsa, csa, incx, incy, context);
}

inline void hgemv(
    const yt::float16* A, const yt::float16* x, yt::float16* y, int m, int k, float alpha, float beta
) {
    gemv_col(A, x, y, m, k, alpha, beta, k, 1, 1, 1);
}

}  // namespace yt::blas
