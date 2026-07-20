#pragma once
/***************
 * file: strided/matmul.inl
 * purpose: strided layout 的 matmul 职责实现。
 ***************/

#include <bit>

#include "../../include/type/type_dispatch.hpp"
#if YT_USE_AVX2
#include "../../include/blas/avx2/hgemm.hpp"
#include "../../include/blas/avx2/sgemm.hpp"
#endif

namespace yt::strided {

// ==================== common shape and arithmetic contracts ====================

// 执行一个乘加；整数使用同宽模语义，避免signed overflow未定义行为。
template <typename T>
YT_IMPL_INLINE T matmulAccumulate(T accum, const T& left, const T& right) {
    if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        using Wide = std::conditional_t<(sizeof(T) < sizeof(uint64_t)), uint64_t, U>;
        U wrapped = static_cast<U>(
            static_cast<Wide>(static_cast<U>(accum)) +
            static_cast<Wide>(static_cast<U>(left)) * static_cast<Wide>(static_cast<U>(right))
        );
        if constexpr (std::is_signed_v<T>) return std::bit_cast<T>(wrapped);
        return wrapped;
    } else {
        return accum + left * right;
    }
}

// matmul owner修改typed view metadata所需的最小特权接口。
template <typename T, int dim>
struct TensorAccess {
    static const YMemory& memoryOf(const YTensor<T, dim>& tensor) { return tensor._memory; }

    static int offsetOf(const YTensor<T, dim>& tensor) { return tensor.stridedOffset(); }

    static void setView(
        YTensor<T, dim>& tensor, const std::vector<int>& shape, const std::vector<int>& stride, int offset,
        const YMemory& memory
    ) {
        BaseViewAccess::setView(
            static_cast<YTensorBase&>(tensor), shape, stride, offset, memory, sizeof(T),
            yt::type::getTypeName<T>()
        );
    }
};

// 按batch广播和rank-1提升规则构造runtime matmul输出shape。
inline std::vector<int> makeMatmulOutputShape(
    const YTensorBase& left, const YTensorBase& right, const YTensorBase& leftMatView,
    const YTensorBase& rightMatView
) {
    int ah = (left.ndim() >= 2) ? left.shape(left.ndim() - 2) : 1;
    int bw = right.shape(right.ndim() - 1);
    std::vector<int> opBatchShape =
        yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
    int opBatchDim = std::max(std::max(0, left.ndim() - 2), std::max(0, right.ndim() - 2));
    // matView为rank-1/2保留一个wrapper轴；只取真实batch rank对应的尾部维度。
    std::vector<int> opShape;
    for (int i = static_cast<int>(opBatchShape.size()) - opBatchDim;
         i < static_cast<int>(opBatchShape.size()); ++i) {
        opShape.push_back(opBatchShape[i]);
    }
    opShape.push_back(ah);
    opShape.push_back(bw);
    return opShape;
}

// 返回矩阵行数；rank-1按项目历史合同提升为1xN。
template <typename T, int dim>
YT_IMPL_INLINE int matrixRows(const YTensor<T, dim>& tensor) {
    // 项目历史合同把rank-1 tensor提升为1xN行矩阵。
    if constexpr (dim == 1) return 1;
    return tensor.shape(dim - 2);
}

// typed入口在选择backend前统一验证inner dimension合同。
template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE void requireTypedMatmulInputs(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, const std::string& opName
) {
    const int rightRows = rightDim == 1 ? 1 : right.shape(rightDim - 2);
    if (left.shape(leftDim - 1) != rightRows) {
        throw std::invalid_argument(
            opName + ": inner dimensions mismatch: " + std::to_string(left.shape(leftDim - 1)) +
            " vs " + std::to_string(rightRows)
        );
    }
}

// 判断最后两个matrix轴能否安全映射为Eigen non-overlapping strided map。
// 注意：extent>1时拒绝zero/negative stride；二维span必须满足行主序或列主序不重叠条件。
template <typename Tensor>
YT_IMPL_INLINE bool eigenMatrixLayoutSupported(const Tensor& tensor) {
    const int rank = [&]() {
        if constexpr (yt::utils::is_ytensor_template_v<Tensor>) {
            return std::decay_t<Tensor>::ndim;
        } else {
            return tensor.ndim();
        }
    }();
    const int colDim = rank - 1;
    const int cols = tensor.shape(colDim);
    const int64_t colStride = tensor.stride_(colDim);
    if (cols > 1 && colStride <= 0) return false;
    if (rank == 1) return true;

    const int rowDim = rank - 2;
    const int rows = tensor.shape(rowDim);
    const int64_t rowStride = tensor.stride_(rowDim);
    if (rows > 1 && rowStride <= 0) return false;
    if (rows <= 1 || cols <= 1) return true;

    if (rowStride <= colStride) {
        return rowStride > 0 && colStride >= rowStride * rows;
    }
    return colStride > 0 && rowStride >= colStride * cols;
}

// 计算半开shape区间乘积并约束到当前int-indexed backend范围。
template <typename Tensor>
YT_IMPL_INLINE int checkedShapeProduct(
    const Tensor& tensor, int begin, int end, const std::string& context
) {
    size_t product = 1;
    for (int i = begin; i < end; ++i) {
        int extent = tensor.shape(i);
        if (extent != 0 && product > static_cast<size_t>(std::numeric_limits<int>::max()) / extent) {
            throw std::overflow_error(context + ": shape product exceeds int range");
        }
        product *= static_cast<size_t>(extent);
    }
    return static_cast<int>(product);
}

// 将宽类型view offset安全收窄为Strided metadata使用的int。
YT_IMPL_INLINE int checkedMatmulOffset(int64_t offset, const std::string& context) {
    if (offset < std::numeric_limits<int>::min() || offset > std::numeric_limits<int>::max()) {
        throw std::overflow_error(context + ": view offset exceeds int range");
    }
    return static_cast<int>(offset);
}

// runtime naive kernel：把batch包装为二维view，再由broadcast owner逐batch计算。
template <typename DType>
YT_IMPL_INLINE void stridedMatmulKernel(YTensorBase& out, const YTensorBase& left, const YTensorBase& right) {
    auto leftMatView = yt::strided::matView(left);
    auto rightMatView = yt::strided::matView(right);
    int ah = (left.ndim() >= 2) ? left.shape(left.ndim() - 2) : 1;
    int aw = left.shape(left.ndim() - 1);
    int bw = right.shape(right.ndim() - 1);

    auto outMatView = yt::strided::matView(out);
    yt::strided::broadcastInplaceBase(
        outMatView,
        [ah, aw, bw](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
            for (int i = 0; i < ah; ++i) {
                for (int j = 0; j < bw; ++j) {
                    DType sum{};
                    for (int k = 0; k < aw; ++k) {
                        sum = matmulAccumulate(
                            sum, A.template at<DType>({i, k}), B.template at<DType>({k, j})
                        );
                    }
                    C.template at<DType>({i, j}) = sum;
                }
            }
        },
        leftMatView, rightMatView
    );
}

// runtime masked naive kernel；同一个二维mask复用于所有广播batch。
template <typename DType>
YT_IMPL_INLINE void stridedMaskedMatmulKernel(
    YTensorBase& out, const YTensorBase& left, const YTensorBase& right, const YTensorBase& mask,
    double maskedValue
) {
    auto leftMatView = yt::strided::matView(left);
    auto rightMatView = yt::strided::matView(right);
    int ah = (left.ndim() >= 2) ? left.shape(left.ndim() - 2) : 1;
    int aw = left.shape(left.ndim() - 1);
    int bw = right.shape(right.ndim() - 1);

    const DType masked = static_cast<DType>(maskedValue);
    auto outMatView = yt::strided::matView(out);
    yt::strided::broadcastInplaceBase(
        outMatView,
        [ah, aw, bw, &mask, masked](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
            for (int i = 0; i < ah; ++i) {
                for (int j = 0; j < bw; ++j) {
                    // masked位置不读取任何operand，直接写入调用方指定值。
                    if (!mask.at<bool>({i, j})) {
                        C.template at<DType>({i, j}) = masked;
                        continue;
                    }
                    DType sum{};
                    for (int k = 0; k < aw; ++k) {
                        sum = matmulAccumulate(
                            sum, A.template at<DType>({i, k}), B.template at<DType>({k, j})
                        );
                    }
                    C.template at<DType>({i, j}) = sum;
                }
            }
        },
        leftMatView, rightMatView
    );
}

#if YT_USE_EIGEN
template <typename T>
using EigenStridedMap = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenConstStridedMap = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
YT_IMPL_INLINE auto toEigenMap(YTensorBase& mat) {
    return EigenStridedMap<T>(
        mat.template data<T>(), mat.shape(0), mat.shape(1),
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(mat.stride_(0), mat.stride_(1))
    );
}

template <typename T>
YT_IMPL_INLINE auto toEigenConstMap(const YTensorBase& mat) {
    return EigenConstStridedMap<T>(
        mat.template data<T>(), mat.shape(0), mat.shape(1),
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(mat.stride_(0), mat.stride_(1))
    );
}

#endif

// 合并一个builtin dtype的naive matmul能力，不覆盖用户已注册kernel。
inline void registerBuiltinMatmulKernel(
    const std::string& dtype, void (*matmul)(YTensorBase&, const YTensorBase&, const YTensorBase&),
    void (*maskedMatmul)(YTensorBase&, const YTensorBase&, const YTensorBase&, const YTensorBase&, double)
) {
    yt::type::YDTypeKernels kernels;
    kernels.matmul = matmul;
    kernels.maskedMatmul = maskedMatmul;
    yt::type::mergeDTypeKernels(dtype, kernels);
}

// 首次runtime matmul时惰性安装builtin kernel table。
inline void ensureBuiltinMatmulKernels() {
    static const bool initialized = []() {
        registerBuiltinMatmulKernel("float32", &stridedMatmulKernel<float>, &stridedMaskedMatmulKernel<float>);
        registerBuiltinMatmulKernel("float64", &stridedMatmulKernel<double>, &stridedMaskedMatmulKernel<double>);
        registerBuiltinMatmulKernel("int8", &stridedMatmulKernel<int8_t>, &stridedMaskedMatmulKernel<int8_t>);
        registerBuiltinMatmulKernel("int16", &stridedMatmulKernel<int16_t>, &stridedMaskedMatmulKernel<int16_t>);
        registerBuiltinMatmulKernel("int32", &stridedMatmulKernel<int32_t>, &stridedMaskedMatmulKernel<int32_t>);
        registerBuiltinMatmulKernel("int64", &stridedMatmulKernel<int64_t>, &stridedMaskedMatmulKernel<int64_t>);
        registerBuiltinMatmulKernel("uint8", &stridedMatmulKernel<uint8_t>, &stridedMaskedMatmulKernel<uint8_t>);
        registerBuiltinMatmulKernel("uint16", &stridedMatmulKernel<uint16_t>, &stridedMaskedMatmulKernel<uint16_t>);
        registerBuiltinMatmulKernel("uint32", &stridedMatmulKernel<uint32_t>, &stridedMaskedMatmulKernel<uint32_t>);
        registerBuiltinMatmulKernel("uint64", &stridedMatmulKernel<uint64_t>, &stridedMaskedMatmulKernel<uint64_t>);
        registerBuiltinMatmulKernel("bfloat16", &stridedMatmulKernel<yt::bfloat16>, &stridedMaskedMatmulKernel<yt::bfloat16>);
        registerBuiltinMatmulKernel("float16", &stridedMatmulKernel<yt::float16>, &stridedMaskedMatmulKernel<yt::float16>);
        registerBuiltinMatmulKernel("float8_e5m2", &stridedMatmulKernel<yt::float8_e5m2>, &stridedMaskedMatmulKernel<yt::float8_e5m2>);
        registerBuiltinMatmulKernel("float8_e4m3", &stridedMatmulKernel<yt::float8_e4m3>, &stridedMaskedMatmulKernel<yt::float8_e4m3>);
        registerBuiltinMatmulKernel("float8_e8m0", &stridedMatmulKernel<yt::float8_e8m0>, &stridedMaskedMatmulKernel<yt::float8_e8m0>);
        registerBuiltinMatmulKernel("float8_ue8m0", &stridedMatmulKernel<yt::float8_ue8m0>, &stridedMaskedMatmulKernel<yt::float8_ue8m0>);
        return true;
    }();
    (void)initialized;
}

// runtime入口的统一layout、dtype、rank和inner-dimension校验。
inline void requireStridedMatmulInputs(
    const YTensorBase& left, const YTensorBase& right, const std::string& opName
) {
    if (!left.isStrided() || !right.isStrided()) {
        throw std::runtime_error(opName + ": layout not implemented");
    }
    if (left.ndim() < 1 || right.ndim() < 1) {
        throw std::runtime_error(opName + ": both tensors must have at least 1 dimension");
    }
    if (left.dtype() != right.dtype()) {
        throw std::runtime_error(opName + ": dtype mismatch: " + left.dtype() + " vs " + right.dtype());
    }
    int leftCols = left.shape(left.ndim() - 1);
    int rightRows = (right.ndim() >= 2) ? right.shape(right.ndim() - 2) : 1;
    if (leftCols != rightRows) {
        throw std::runtime_error(
            opName + ": inner dimensions mismatch: " + std::to_string(leftCols) + " vs " +
            std::to_string(rightRows)
        );
    }
}

// ==================== typed backend implementations ====================

// typed naive实现；二维直接输出matrix，高rank输出batch广播后的matrix tensor。
template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMatmulNaive(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right
) {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int ah = matrixRows(left);
    int aw = left.shape(-1);
    int bw = right.shape(-1);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {ah, bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(ah);
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [ah, aw, bw](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
#pragma omp parallel for collapse(2) proc_bind(close)
            for (int y = 0; y < ah; ++y) {
                for (int x = 0; x < bw; ++x) {
                    T sum = static_cast<T>(0);
                    for (int k = 0; k < aw; ++k) {
                        sum = matmulAccumulate(sum, A.at(y, k), B.at(k, x));
                    }
                    C.at(y, x) = sum;
                }
            }
        },
        leftMatView, rightMatView
    );
    return out;
}

// typed二维mask naive实现；先填充maskedValue，仅计算mask允许的位置。
template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMaskedMatmulNaive(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, const YTensor<bool, 2>& mask,
    const T& maskedValue
) {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int bw = right.shape(-1);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {matrixRows(left), bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(matrixRows(left));
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    out.fill(maskedValue);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [&mask](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
            int m = A.shape(0);
            int k = A.shape(1);
            int n = B.shape(1);
#pragma omp parallel for collapse(2) proc_bind(close)
            for (int y = 0; y < m; ++y) {
                for (int x = 0; x < n; ++x) {
                    if (!mask.at(y, x)) {
                        continue;
                    }
                    T sum = static_cast<T>(0);
                    for (int p = 0; p < k; ++p) {
                        sum = matmulAccumulate(sum, A.at(y, p), B.at(p, x));
                    }
                    C.at(y, x) = sum;
                }
            }
        },
        leftMatView, rightMatView
    );
    return out;
}

// callable mask naive实现；predicate可能在OpenMP区域并发调用，调用方需保证线程安全。
template <typename T, int leftDim, int rightDim, typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMaskedMatmulNaive(
        const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, Func&& func, const T& maskedValue
    ) {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    auto&& predicate = std::forward<Func>(func);
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int bw = right.shape(-1);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {matrixRows(left), bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(matrixRows(left));
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    out.fill(maskedValue);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [&predicate](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
            int m = A.shape(0);
            int k = A.shape(1);
            int n = B.shape(1);
#pragma omp parallel for collapse(2) proc_bind(close)
            for (int y = 0; y < m; ++y) {
                for (int x = 0; x < n; ++x) {
                    if (!predicate(y, x)) {
                        continue;
                    }
                    T sum = static_cast<T>(0);
                    for (int p = 0; p < k; ++p) {
                        sum = matmulAccumulate(sum, A.at(y, p), B.at(p, x));
                    }
                    C.at(y, x) = sum;
                }
            }
        },
        leftMatView, rightMatView
    );
    return out;
}

#if YT_USE_EIGEN
// Eigen实现；能合并连续batch-row轴时使用大矩阵乘法，否则逐batch映射二维矩阵。
template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMatmulEigen(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right
) {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    int aw = left.shape(-1);
    int bw = right.shape(-1);

    if constexpr (leftDim > 2) {
        bool rightIs2D = (rightDim <= 2);
        if (!rightIs2D) {
            rightIs2D = true;
            for (int i = 0; i < rightDim - 2; ++i) {
                if (right.shape(i) != 1) {
                    rightIs2D = false;
                    break;
                }
            }
        }
        if (rightIs2D) {
            // leading singleton轴不改变right matrix，可与真正2-D right使用同一flatten优化。
            int contiguousStart = left.isContiguousFrom(0, -1);
            if (contiguousStart < leftDim - 1) {
                // 连续suffix中的batch轴与row轴合并为innerRows；非连续prefix逐outerIdx处理。
                int outerSize = checkedShapeProduct(left, 0, contiguousStart, "typed Eigen matmul");
                int innerRows =
                    checkedShapeProduct(left, contiguousStart, leftDim - 1, "typed Eigen matmul");

                std::vector<int> outShape;
                for (int i = 0; i < leftDim - 1; ++i) {
                    outShape.push_back(left.shape(i));
                }
                outShape.push_back(bw);
                YTensor<T, outDim> out(outShape);

                YTensor<T, 2> right2D;
                TensorAccess<T, 2>::setView(
                    right2D, {aw, bw}, {right.stride_(-2), right.stride_(-1)},
                    TensorAccess<T, rightDim>::offsetOf(right), TensorAccess<T, rightDim>::memoryOf(right)
                );

                int innerStride = (contiguousStart == 0) ? aw : left.stride_(contiguousStart);
                int outInnerStride = (contiguousStart == 0) ? bw : out.stride_(contiguousStart);

                for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                    // 解码非连续prefix坐标，分别定位输入和新分配连续输出中的二维flat view。
                    int64_t leftOffset = 0;
                    int64_t outOffset = 0;
                    if (contiguousStart > 0) {
                        int idx = outerIdx;
                        for (int i = contiguousStart - 1; i >= 0; --i) {
                            int coord = idx % left.shape(i);
                            idx /= left.shape(i);
                            leftOffset += coord * left.stride_(i);
                            outOffset += coord * out.stride_(i);
                        }
                    }

                    YTensor<T, 2> leftFlat;
                    YTensor<T, 2> outFlat;
                    TensorAccess<T, 2>::setView(
                        leftFlat, {innerRows, aw}, {innerStride, left.stride_(-1)},
                        checkedMatmulOffset(
                            static_cast<int64_t>(TensorAccess<T, leftDim>::offsetOf(left)) + leftOffset,
                            "typed Eigen matmul"
                        ),
                        TensorAccess<T, leftDim>::memoryOf(left)
                    );
                    TensorAccess<T, 2>::setView(
                        outFlat, {innerRows, bw}, {outInnerStride, 1},
                        checkedMatmulOffset(outOffset, "typed Eigen matmul"),
                        TensorAccess<T, outDim>::memoryOf(out)
                    );

                    auto mapA = leftFlat.matViewEigen();
                    auto mapB = right2D.matViewEigen();
                    auto mapC = outFlat.matViewEigen();
                    mapC.noalias() = mapA * mapB;
                }
                return out;
            }
        }
    }

    // flatten条件不满足时保留完整batch广播语义，逐二维matrix调用Eigen。
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int ah = matrixRows(left);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {ah, bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(ah);
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
            auto mapA = A.matViewEigen();
            auto mapB = B.matViewEigen();
            auto mapC = C.matViewEigen();
            mapC.noalias() = mapA * mapB;
        },
        leftMatView, rightMatView
    );
    return out;
}
#endif

#if YT_USE_AVX2
// float AVX2实现；flatten条件和offset计算必须与typed Eigen路径保持同步。
template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMatmulAvx2(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right
) requires std::is_same_v<T, float> {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    int aw = left.shape(-1);
    int bw = right.shape(-1);

    if constexpr (leftDim > 2) {
        bool rightIs2D = (rightDim <= 2);
        if (!rightIs2D) {
            rightIs2D = true;
            for (int i = 0; i < rightDim - 2; ++i) {
                if (right.shape(i) != 1) {
                    rightIs2D = false;
                    break;
                }
            }
        }
        if (rightIs2D) {
            // 将连续batch-row suffix折叠成更大的M维，减少小型GEMM调用和调度开销。
            int contiguousStart = left.isContiguousFrom(0, -1);
            if (contiguousStart < leftDim - 1) {
                int outerSize = checkedShapeProduct(left, 0, contiguousStart, "typed AVX2 matmul");
                int innerRows =
                    checkedShapeProduct(left, contiguousStart, leftDim - 1, "typed AVX2 matmul");

                std::vector<int> outShape;
                for (int i = 0; i < leftDim - 1; ++i) {
                    outShape.push_back(left.shape(i));
                }
                outShape.push_back(bw);
                YTensor<T, outDim> out(outShape);

                YTensor<T, 2> right2D;
                TensorAccess<T, 2>::setView(
                    right2D, {aw, bw}, {right.stride_(-2), right.stride_(-1)},
                    TensorAccess<T, rightDim>::offsetOf(right), TensorAccess<T, rightDim>::memoryOf(right)
                );

                int innerStride = (contiguousStart == 0) ? aw : left.stride_(contiguousStart);
                int outInnerStride = (contiguousStart == 0) ? bw : out.stride_(contiguousStart);

                for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                    int64_t leftOffset = 0;
                    int64_t outOffset = 0;
                    if (contiguousStart > 0) {
                        int idx = outerIdx;
                        for (int i = contiguousStart - 1; i >= 0; --i) {
                            int coord = idx % left.shape(i);
                            idx /= left.shape(i);
                            leftOffset += coord * left.stride_(i);
                            outOffset += coord * out.stride_(i);
                        }
                    }

                    YTensor<T, 2> leftFlat;
                    YTensor<T, 2> outFlat;
                    TensorAccess<T, 2>::setView(
                        leftFlat, {innerRows, aw}, {innerStride, left.stride_(-1)},
                        checkedMatmulOffset(
                            static_cast<int64_t>(TensorAccess<T, leftDim>::offsetOf(left)) + leftOffset,
                            "typed AVX2 matmul"
                        ),
                        TensorAccess<T, leftDim>::memoryOf(left)
                    );
                    TensorAccess<T, 2>::setView(
                        outFlat, {innerRows, bw}, {outInnerStride, 1},
                        checkedMatmulOffset(outOffset, "typed AVX2 matmul"),
                        TensorAccess<T, outDim>::memoryOf(out)
                    );

                    yt::blas::matmul(
                        leftFlat.data(), right2D.data(), outFlat.data(), innerRows, bw, aw,
                        static_cast<int64_t>(leftFlat.stride_(0)), static_cast<int64_t>(leftFlat.stride_(1)),
                        static_cast<int64_t>(right2D.stride_(0)), static_cast<int64_t>(right2D.stride_(1)),
                        static_cast<int64_t>(outFlat.stride_(0)), static_cast<int64_t>(outFlat.stride_(1))
                    );
                }
                return out;
            }
        }
    }

    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int ah = matrixRows(left);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {ah, bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(ah);
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
            auto aStride = A.stride_();
            auto bStride = B.stride_();
            auto cStride = C.stride_();
            yt::blas::matmul(
                A.data(), B.data(), C.data(), A.shape(0), B.shape(1), A.shape(1),
                static_cast<int64_t>(aStride[0]), static_cast<int64_t>(aStride[1]),
                static_cast<int64_t>(bStride[0]), static_cast<int64_t>(bStride[1]),
                static_cast<int64_t>(cStride[0]), static_cast<int64_t>(cStride[1])
            );
        },
        leftMatView, rightMatView
    );
    return out;
}

template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMaskedMatmulAvx2(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, const YTensor<bool, 2>& mask,
    const T& maskedValue
) requires std::is_same_v<T, float> {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    int bw = right.shape(-1);
    std::vector<int> outShape;
    if constexpr (outDim == 2) {
        outShape = {matrixRows(left), bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(matrixRows(left));
        outShape.push_back(bw);
    }

    YTensor<T, outDim> out(outShape);
    out.fill(maskedValue);
    auto outMatView = out.matView();
    // sgemm_masked按线性bool pointer读取mask；只物化mask，operand仍保留合法strided访问。
    YTensor<bool, 2> maskContiguous = mask.isContiguous() ? mask : mask.contiguous();
    const bool* maskPtr = maskContiguous.data();

    outMatView.broadcastInplace(
        [maskPtr](YTensor<T, 2>& C, const YTensor<T, 2>& A, const YTensor<T, 2>& B) {
            auto aStride = A.stride_();
            auto bStride = B.stride_();
            auto cStride = C.stride_();
            yt::blas::sgemm_masked(
                A.data(), B.data(), C.data(), A.shape(0), B.shape(1), A.shape(1), 1.0f, 0.0f,
                static_cast<int64_t>(aStride[0]), static_cast<int64_t>(aStride[1]),
                static_cast<int64_t>(bStride[0]), static_cast<int64_t>(bStride[1]),
                static_cast<int64_t>(cStride[0]), static_cast<int64_t>(cStride[1]), maskPtr
            );
        },
        leftMatView, rightMatView
    );
    return out;
}

// dense 2-D callable mask专用入口，避免构造batch wrapper和broadcast调度。
template <typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    YT_IMPL_INLINE YTensor<float, 2> typedMaskedMatmulAvx2Dense2D(
        const YTensor<float, 2>& left, const YTensor<float, 2>& right, Func&& func, const float& maskedValue
    ) {
    auto&& predicate = std::forward<Func>(func);
    YTensor<float, 2> out({left.shape(-2), right.shape(-1)});
    out.fill(maskedValue);
    yt::blas::sgemm_masked(
        left.data(), right.data(), out.data(), left.shape(0), right.shape(1), left.shape(1), 1.0f, 0.0f,
        static_cast<int64_t>(left.stride_(0)), static_cast<int64_t>(left.stride_(1)),
        static_cast<int64_t>(right.stride_(0)), static_cast<int64_t>(right.stride_(1)),
        static_cast<int64_t>(out.stride_(0)), static_cast<int64_t>(out.stride_(1)), predicate
    );
    return out;
}

template <int leftDim, int rightDim, typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    YT_IMPL_INLINE YTensor<float, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> typedMaskedMatmulAvx2(
        const YTensor<float, leftDim>& left, const YTensor<float, rightDim>& right, Func&& func,
        const float& maskedValue
    ) {
    constexpr int outDim = yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2});
    auto&& predicate = std::forward<Func>(func);
    auto leftMatView = left.matView();
    auto rightMatView = right.matView();
    std::vector<int> outShape;
    int bw = right.shape(-1);
    if constexpr (outDim == 2) {
        outShape = {matrixRows(left), bw};
    } else {
        outShape = yt::strided::computeBroadcastShape({leftMatView.shape(), rightMatView.shape()});
        outShape.push_back(matrixRows(left));
        outShape.push_back(bw);
    }

    YTensor<float, outDim> out(outShape);
    out.fill(maskedValue);
    auto outMatView = out.matView();
    outMatView.broadcastInplace(
        [&predicate](YTensor<float, 2>& C, const YTensor<float, 2>& A, const YTensor<float, 2>& B) {
            yt::blas::sgemm_masked(
                A.data(), B.data(), C.data(), A.shape(0), B.shape(1), A.shape(1), 1.0f, 0.0f,
                static_cast<int64_t>(A.stride_(0)), static_cast<int64_t>(A.stride_(1)),
                static_cast<int64_t>(B.stride_(0)), static_cast<int64_t>(B.stride_(1)),
                static_cast<int64_t>(C.stride_(0)), static_cast<int64_t>(C.stride_(1)), predicate
            );
        },
        leftMatView, rightMatView
    );
    return out;
}
#endif

template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> matmul(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, yt::info::MatmulBackend backend
) {
    requireTypedMatmulInputs(left, right, "yt::strided::matmul");
    if constexpr (!yt::type::is_builtin_numeric_v<T>) {
        (void)backend;
        return typedMatmulNaive(left, right);
    } else if constexpr (std::is_integral_v<T>) {
        // Integer matmul must use the owner-defined same-width modular accumulator.
        (void)backend;
        return typedMatmulNaive(left, right);
    } else {
        // backend是偏好：AVX2不支持当前T时尝试Eigen，layout不满足Eigen时回退naive。
        switch (backend) {
#if YT_USE_AVX2
            case yt::info::MatmulBackend::AVX2:
                if constexpr (std::is_same_v<T, float>) {
                    return typedMatmulAvx2(left, right);
                }
                [[fallthrough]];
#endif
#if YT_USE_EIGEN
            case yt::info::MatmulBackend::Eigen:
                if (eigenMatrixLayoutSupported(left) && eigenMatrixLayoutSupported(right)) {
                    return typedMatmulEigen(left, right);
                }
                return typedMatmulNaive(left, right);
#endif
            case yt::info::MatmulBackend::Naive:
            default:
                return typedMatmulNaive(left, right);
        }
    }
}

template <typename T, int leftDim, int rightDim>
YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> masked_matmul(
    const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, const YTensor<bool, 2>& mask,
    const T& maskedValue, yt::info::MatmulBackend backend
) {
    requireTypedMatmulInputs(left, right, "yt::strided::masked_matmul");
    const std::vector<int> expectedMaskShape = {matrixRows(left), right.shape(rightDim - 1)};
    if (!mask.shapeMatch(expectedMaskShape)) {
        throw std::invalid_argument("yt::strided::masked_matmul: mask shape mismatch");
    }
    if constexpr (!yt::type::is_builtin_numeric_v<T>) {
        (void)backend;
        return typedMaskedMatmulNaive(left, right, mask, maskedValue);
    } else {
        // 当前没有masked Eigen实现，Eigen偏好和不支持的AVX2 dtype都安全回退naive。
        switch (backend) {
#if YT_USE_AVX2
            case yt::info::MatmulBackend::AVX2:
                if constexpr (std::is_same_v<T, float>) {
                    return typedMaskedMatmulAvx2(left, right, mask, maskedValue);
                }
                [[fallthrough]];
#endif
            case yt::info::MatmulBackend::Eigen:
            case yt::info::MatmulBackend::Naive:
            default:
                return typedMaskedMatmulNaive(left, right, mask, maskedValue);
        }
    }
}

template <typename T, int leftDim, int rightDim, typename Func>
requires(!yt::utils::is_ytensor_v<std::decay_t<Func>>)
    YT_IMPL_INLINE YTensor<T, yt::utils::CONSTEXPR_MAX({leftDim, rightDim, 2})> masked_matmul(
        const YTensor<T, leftDim>& left, const YTensor<T, rightDim>& right, Func&& func, const T& maskedValue,
        yt::info::MatmulBackend backend
    ) {
    requireTypedMatmulInputs(left, right, "yt::strided::masked_matmul");
    if constexpr (yt::type::is_builtin_numeric_v<T>) {
        switch (backend) {
#if YT_USE_AVX2
            case yt::info::MatmulBackend::AVX2:
                if constexpr (std::is_same_v<T, float>) {
                    return typedMaskedMatmulAvx2(left, right, std::forward<Func>(func), maskedValue);
                }
                [[fallthrough]];
#endif
            case yt::info::MatmulBackend::Eigen:
            case yt::info::MatmulBackend::Naive:
            default:
                return typedMaskedMatmulNaive(left, right, std::forward<Func>(func), maskedValue);
        }
    } else {
        (void)backend;
        return typedMaskedMatmulNaive(left, right, std::forward<Func>(func), maskedValue);
    }
}

// ==================== runtime backend orchestration ====================

// runtime matmul选择cast策略、backend fast path或注册dtype naive kernel。
YT_IMPL_INLINE YTensorBase
matmul(const YTensorBase& left, const YTensorBase& right, yt::info::MatmulBackend backend) {
    requireStridedMatmulInputs(left, right, "yt::strided::matmul");

    bool canUseFp16Avx2 = false;
#if YT_USE_AVX2
    canUseFp16Avx2 = (left.dtype() == "float16" && backend == yt::info::MatmulBackend::AVX2);
#endif

    // low precision默认以float32累加再cast回原dtype；仅AVX2 float16保留native hgemm路径。
    bool needsCastToFloat32 =
        (left.dtype() == "bfloat16" || (left.dtype() == "float16" && !canUseFp16Avx2) ||
         left.dtype() == "float8_e5m2" || left.dtype() == "float8_e4m3" || left.dtype() == "float8_e8m0" ||
         left.dtype() == "float8_ue8m0");
    if (needsCastToFloat32) {
        YTensorBase leftF32 = left.cast("float32");
        YTensorBase rightF32 = right.cast("float32");
        return yt::strided::matmul(leftF32, rightF32, backend).cast(left.dtype());
    }

    ensureBuiltinMatmulKernels();
    const auto& kernels = yt::type::getDTypeKernels(left.dtype());

    auto runNaive = [&]() {
        if (!kernels.matmul) {
            throw std::runtime_error("yt::strided::matmul: dtype kernel not implemented");
        }
        auto leftMatView = yt::strided::matView(left);
        auto rightMatView = yt::strided::matView(right);
        YTensorBase out(makeMatmulOutputShape(left, right, leftMatView, rightMatView), left.dtype());
        kernels.matmul(out, left, right);
        return out;
    };

#if YT_USE_AVX2
    // runtime版flatten优化与typedMatmulAvx2共享相同shape/stride不变量；修改时必须同步两处。
    auto runAvx2TypedMatmul = [&]<typename T>(const char* dtypeName, auto&& kernel) -> YTensorBase {
        int leftDim = left.ndim();
        int rightDim = right.ndim();
        int aw = left.shape(leftDim - 1);
        int bw = right.shape(rightDim - 1);
        if (leftDim > 2) {
            bool rightIs2D = (rightDim <= 2);
            if (!rightIs2D) {
                rightIs2D = true;
                for (int i = 0; i < rightDim - 2; ++i) {
                    if (right.shape(i) != 1) {
                        rightIs2D = false;
                        break;
                    }
                }
            }
            if (rightIs2D) {
                int contiguousStart = left.isContiguousFrom(0, -1);
                if (contiguousStart < leftDim - 1) {
                    int outerSize = checkedShapeProduct(left, 0, contiguousStart, "runtime AVX2 matmul");
                    int innerRows =
                        checkedShapeProduct(left, contiguousStart, leftDim - 1, "runtime AVX2 matmul");

                    std::vector<int> opShape;
                    for (int i = 0; i < leftDim - 1; ++i) opShape.push_back(left.shape(i));
                    opShape.push_back(bw);
                    YTensorBase out(opShape, dtypeName);

                    YTensorBase right2D;
                    BaseViewAccess::setView(
                        right2D, {aw, bw}, {right.stride_(rightDim - 2), right.stride_(rightDim - 1)},
                        right.stridedOffset(), right._memory, sizeof(T), dtypeName
                    );

                    int innerStride = (contiguousStart == 0) ? aw : left.stride_(contiguousStart);
                    int outInnerStride = (contiguousStart == 0) ? bw : out.stride_(contiguousStart);

                    for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                        int64_t leftOffset = 0;
                        int64_t outOffset = 0;
                        if (contiguousStart > 0) {
                            int idx = outerIdx;
                            for (int i = contiguousStart - 1; i >= 0; --i) {
                                int coord = idx % left.shape(i);
                                idx /= left.shape(i);
                                leftOffset += coord * left.stride_(i);
                                outOffset += coord * out.stride_(i);
                            }
                        }

                        YTensorBase leftFlat;
                        BaseViewAccess::setView(
                            leftFlat, {innerRows, aw}, {innerStride, left.stride_(leftDim - 1)},
                            checkedMatmulOffset(
                                static_cast<int64_t>(left.stridedOffset()) + leftOffset,
                                "runtime AVX2 matmul"
                            ),
                            left._memory, sizeof(T), dtypeName
                        );

                        YTensorBase outFlat;
                        BaseViewAccess::setView(
                            outFlat, {innerRows, bw}, {outInnerStride, 1},
                            checkedMatmulOffset(outOffset, "runtime AVX2 matmul"), out._memory, sizeof(T),
                            dtypeName
                        );

                        kernel(
                            leftFlat.data<T>(), right2D.data<T>(), outFlat.data<T>(), innerRows, bw, aw,
                            static_cast<int64_t>(leftFlat.stride_(0)),
                            static_cast<int64_t>(leftFlat.stride_(1)),
                            static_cast<int64_t>(right2D.stride_(0)),
                            static_cast<int64_t>(right2D.stride_(1)),
                            static_cast<int64_t>(outFlat.stride_(0)), static_cast<int64_t>(outFlat.stride_(1))
                        );
                    }
                    return out;
                }
            }
        }

        auto leftMatView = yt::strided::matView(left);
        auto rightMatView = yt::strided::matView(right);
        YTensorBase out(makeMatmulOutputShape(left, right, leftMatView, rightMatView), dtypeName);
        auto outMatView = yt::strided::matView(out);
        yt::strided::broadcastInplaceBase(
            outMatView,
            [&kernel](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
                kernel(
                    A.data<T>(), B.data<T>(), C.data<T>(), A.shape(0), B.shape(1), A.shape(1),
                    static_cast<int64_t>(A.stride_(0)), static_cast<int64_t>(A.stride_(1)),
                    static_cast<int64_t>(B.stride_(0)), static_cast<int64_t>(B.stride_(1)),
                    static_cast<int64_t>(C.stride_(0)), static_cast<int64_t>(C.stride_(1))
                );
            },
            leftMatView, rightMatView
        );
        return out;
    };
#endif

#if YT_USE_EIGEN
    // runtime Eigen路径先做一次dtype dispatch；每个选定类型内部不再进行逐元素dtype判断。
    auto runEigen = [&]() {
        int leftDim = left.ndim();
        int rightDim = right.ndim();
        int aw = left.shape(leftDim - 1);
        int bw = right.shape(rightDim - 1);
        YTensorBase result;

        yt::type::dispatch<yt::type::EigenNativeTypes>(
            left.dtype(),
            [&]<typename DType>() {
                if (leftDim > 2) {
                    // 与typed Eigen路径一致：连续batch-row suffix可折叠为一个更大的二维map。
                    bool rightIs2D = (rightDim <= 2);
                    if (!rightIs2D) {
                        rightIs2D = true;
                        for (int i = 0; i < rightDim - 2; ++i) {
                            if (right.shape(i) != 1) {
                                rightIs2D = false;
                                break;
                            }
                        }
                    }
                    if (rightIs2D) {
                        int contiguousStart = left.isContiguousFrom(0, -1);
                        if (contiguousStart < leftDim - 1) {
                            int outerSize =
                                checkedShapeProduct(left, 0, contiguousStart, "runtime Eigen matmul");
                            int innerRows = checkedShapeProduct(
                                left, contiguousStart, leftDim - 1, "runtime Eigen matmul"
                            );

                            std::vector<int> opShape;
                            for (int i = 0; i < leftDim - 1; ++i) opShape.push_back(left.shape(i));
                            opShape.push_back(bw);
                            YTensorBase out(opShape, left.dtype());

                            YTensorBase right2D;
                            BaseViewAccess::setView(
                                right2D, {aw, bw},
                                {right.stride_(rightDim - 2), right.stride_(rightDim - 1)},
                                right.stridedOffset(), right._memory, sizeof(DType), left.dtype()
                            );

                            int innerStride = (contiguousStart == 0) ? aw : left.stride_(contiguousStart);
                            int outInnerStride = (contiguousStart == 0) ? bw : out.stride_(contiguousStart);
                            auto eigenB = toEigenConstMap<DType>(right2D);

                            for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                                int64_t leftOffset = 0;
                                int64_t outOffset = 0;
                                if (contiguousStart > 0) {
                                    int idx = outerIdx;
                                    for (int i = contiguousStart - 1; i >= 0; --i) {
                                        int coord = idx % left.shape(i);
                                        idx /= left.shape(i);
                                        leftOffset += coord * left.stride_(i);
                                        outOffset += coord * out.stride_(i);
                                    }
                                }

                                EigenConstStridedMap<DType> eigenA(
                                    left.data<DType>() + checkedMatmulOffset(leftOffset, "runtime Eigen matmul"),
                                    innerRows, aw,
                                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                        innerStride, left.stride_(leftDim - 1)
                                    )
                                );
                                EigenStridedMap<DType> eigenC(
                                    out.data<DType>() + checkedMatmulOffset(outOffset, "runtime Eigen matmul"),
                                    innerRows, bw,
                                    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outInnerStride, 1)
                                );
                                eigenC.noalias() = eigenA * eigenB;
                            }
                            result = out;
                            return;
                        }
                    }
                }

                auto leftMatView = yt::strided::matView(left);
                auto rightMatView = yt::strided::matView(right);
                YTensorBase out(
                    makeMatmulOutputShape(left, right, leftMatView, rightMatView), left.dtype()
                );
                auto outMatView = yt::strided::matView(out);
                yt::strided::broadcastInplaceBase(
                    outMatView,
                    [](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
                        toEigenMap<DType>(C).noalias() =
                            toEigenConstMap<DType>(A) * toEigenConstMap<DType>(B);
                    },
                    leftMatView, rightMatView
                );
                result = out;
            },
            "strided::matmul_eigen"
        );

        return result;
    };
#endif

#if YT_USE_EIGEN
    const bool integerDtype =
        left.dtype() == "int8" || left.dtype() == "int16" || left.dtype() == "int32" ||
        left.dtype() == "int64" || left.dtype() == "uint8" || left.dtype() == "uint16" ||
        left.dtype() == "uint32" || left.dtype() == "uint64";
#endif

    switch (backend) {
        case yt::info::MatmulBackend::Naive:
            return runNaive();
        case yt::info::MatmulBackend::AVX2:
#if YT_USE_AVX2
            if (left.dtype() == "float16") {
                auto hKernel = [](auto... args) { yt::blas::hmatmul(args...); };
                return runAvx2TypedMatmul.template operator()<yt::float16>("float16", hKernel);
            }
            if (left.dtype() == "float32") {
                auto sKernel = [](auto... args) { yt::blas::matmul(args...); };
                return runAvx2TypedMatmul.template operator()<float>("float32", sKernel);
            }
#endif
            [[fallthrough]];
        case yt::info::MatmulBackend::Eigen:
#if YT_USE_EIGEN
            // Eigen整数乘加不提供owner规定的同宽模累加语义，因此必须走naive kernel。
            if (integerDtype) return runNaive();
            if (eigenMatrixLayoutSupported(left) && eigenMatrixLayoutSupported(right)) {
                return runEigen();
            }
            return runNaive();
#else
            [[fallthrough]];
#endif
        default:
            return runNaive();
    }
}

// runtime masked matmul；mask为所有batch共享的二维可见性矩阵。
YT_IMPL_INLINE YTensorBase masked_matmul(
    const YTensorBase& left, const YTensorBase& right, const YTensorBase& mask, double maskedValue,
    yt::info::MatmulBackend backend
) {
    requireStridedMatmulInputs(left, right, "yt::strided::masked_matmul");
    if (!mask.isStrided()) {
        throw std::runtime_error("yt::strided::masked_matmul: layout not implemented");
    }
    if (mask.dtype() != "bool") {
        throw std::runtime_error("yt::strided::masked_matmul: mask dtype must be bool");
    }
    if (mask.ndim() != 2) {
        throw std::runtime_error("yt::strided::masked_matmul: mask must be 2D");
    }

    int outRows = (left.ndim() >= 2) ? left.shape(left.ndim() - 2) : 1;
    int outCols = right.shape(right.ndim() - 1);
    if (mask.shape(0) != outRows || mask.shape(1) != outCols) {
        throw std::runtime_error(
            "yt::strided::masked_matmul: mask shape mismatch, expected [" + std::to_string(outRows) + ", " +
            std::to_string(outCols) + "] but got [" + std::to_string(mask.shape(0)) + ", " +
            std::to_string(mask.shape(1)) + "]"
        );
    }

    // masked AVX2只有float32 kernel，所有low precision输入统一提升后再cast回原dtype。
    bool needsCastToFloat32 =
        (left.dtype() == "bfloat16" || left.dtype() == "float16" || left.dtype() == "float8_e5m2" ||
         left.dtype() == "float8_e4m3" || left.dtype() == "float8_e8m0" || left.dtype() == "float8_ue8m0");
    if (needsCastToFloat32) {
        YTensorBase leftF32 = left.cast("float32");
        YTensorBase rightF32 = right.cast("float32");
        return yt::strided::masked_matmul(leftF32, rightF32, mask, maskedValue, backend).cast(left.dtype());
    }

    ensureBuiltinMatmulKernels();
    const auto& kernels = yt::type::getDTypeKernels(left.dtype());

    auto runNaive = [&]() {
        if (!kernels.maskedMatmul) {
            throw std::runtime_error("yt::strided::masked_matmul: dtype kernel not implemented");
        }
        auto leftMatView = yt::strided::matView(left);
        auto rightMatView = yt::strided::matView(right);
        YTensorBase out(makeMatmulOutputShape(left, right, leftMatView, rightMatView), left.dtype());
        kernels.maskedMatmul(out, left, right, mask, maskedValue);
        return out;
    };

#if YT_USE_AVX2
    auto runMaskedAvx2 = [&]() {
        auto leftMatView = yt::strided::matView(left);
        auto rightMatView = yt::strided::matView(right);
        YTensorBase out(makeMatmulOutputShape(left, right, leftMatView, rightMatView), "float32");
        float* outData = out.data<float>();
        float masked = static_cast<float>(maskedValue);
        yt::utils::parallelFor(
            0, yt::utils::checkedIntSize(out.size(), "strided::masked_matmul"),
            [&](int i) { outData[i] = masked; }
        );

        // sgemm_masked要求线性mask；operand/output仍通过各自stride传入backend。
        YTensorBase maskContiguous = mask.isContiguous() ? mask : mask.contiguous();
        const bool* maskPtr = maskContiguous.data<bool>();
        auto outMatView = yt::strided::matView(out);
        yt::strided::broadcastInplaceBase(
            outMatView,
            [maskPtr](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
                yt::blas::sgemm_masked(
                    A.data<float>(), B.data<float>(), C.data<float>(), A.shape(0), B.shape(1), A.shape(1),
                    1.0f, 0.0f, static_cast<int64_t>(A.stride_(0)), static_cast<int64_t>(A.stride_(1)),
                    static_cast<int64_t>(B.stride_(0)), static_cast<int64_t>(B.stride_(1)),
                    static_cast<int64_t>(C.stride_(0)), static_cast<int64_t>(C.stride_(1)), maskPtr
                );
            },
            leftMatView, rightMatView
        );
        return out;
    };
#endif

    // 无masked Eigen实现；Eigen偏好或不支持的AVX2 dtype最终使用注册naive kernel。
    switch (backend) {
        case yt::info::MatmulBackend::Naive:
            return runNaive();
        case yt::info::MatmulBackend::AVX2:
#if YT_USE_AVX2
            if (left.dtype() == "float32") {
                return runMaskedAvx2();
            }
#endif
            [[fallthrough]];
        case yt::info::MatmulBackend::Eigen:
        default:
            return runNaive();
    }
}

}  // namespace yt::strided
