#pragma once
/***************
 * file: blas/ger.inl
 * purpose: typed GER实现。
 ***************/

namespace yt::blas {

template <typename T>
inline void ger(
    int m, int n, float alpha, float beta, const T* x, int64_t incx, const T* y, int64_t incy, T* C,
    int64_t rsc, int64_t csc
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, yt::float16>);
    if (m <= 0 || n <= 0) return;

    if (csc == 1 || std::is_same_v<T, yt::float16>) {
        if constexpr (std::is_same_v<T, float>) {
            for (int i = 0; i < m; ++i) {
                T* row = C + i * rsc;
                scalv(n, beta, row, csc);
                axpyv(n, alpha * x[i * incx], y, incy, row, csc);
            }
        } else {
            static thread_local std::vector<float> output;
            output.resize(static_cast<size_t>(n));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) output[j] = static_cast<float>(C[i * rsc + j * csc]);
                scalv(n, beta, output.data(), 1);
                axpyv(n, alpha * static_cast<float>(x[i * incx]), y, incy, output.data(), 1);
                for (int j = 0; j < n; ++j) C[i * rsc + j * csc] = T(output[j]);
            }
        }
        return;
    }

    if constexpr (std::is_same_v<T, float>) {
        for (int j = 0; j < n; ++j) {
            T* column = C + j * csc;
            scalv(m, beta, column, rsc);
            axpyv(m, alpha * y[j * incy], x, incx, column, rsc);
        }
    } else {
        static thread_local std::vector<float> output;
        output.resize(static_cast<size_t>(m));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) output[i] = static_cast<float>(C[i * rsc + j * csc]);
            scalv(m, beta, output.data(), 1);
            axpyv(m, alpha * static_cast<float>(y[j * incy]), x, incx, output.data(), 1);
            for (int i = 0; i < m; ++i) C[i * rsc + j * csc] = T(output[i]);
        }
    }
}

inline void sger(
    int m, int n, float alpha, float beta, const float* x, int64_t incx, const float* y, int64_t incy,
    float* C, int64_t rsc, int64_t csc
) {
    ger(m, n, alpha, beta, x, incx, y, incy, C, rsc, csc);
}

inline void sger(int m, int n, float alpha, const float* x, const float* y, float* C, int lda) {
    ger(m, n, alpha, 1.0f, x, 1, y, 1, C, lda, 1);
}

inline void hger(
    int m, int n, float alpha, float beta, const yt::float16* x, int64_t incx, const yt::float16* y,
    int64_t incy, yt::float16* C, int64_t rsc, int64_t csc
) {
    ger(m, n, alpha, beta, x, incx, y, incy, C, rsc, csc);
}

inline void hger(
    int m, int n, float alpha, const yt::float16* x, const yt::float16* y, yt::float16* C, int lda
) {
    ger(m, n, alpha, 1.0f, x, 1, y, 1, C, lda, 1);
}

}  // namespace yt::blas
