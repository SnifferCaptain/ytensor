/***************
* @file: ytensor_math.hpp [inline]
* @brief: YTensor 类内置的数学运算功能
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

public:

/// @brief 最大子元素中量遍历父张量阈值，超过则使用stride遍历法，否则使用布尔掩码遍历底层存储。
static constexpr double MAX_SUBELEMENT_RATIO = 2.5;

/// @brief 统一的广播原地操作函数，支持N元张量/标量操作（转发到yt::kernel::broadcastInplace）
/// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
/// @tparam Args 参数类型，可以是YTensor或标量T
/// @param func 操作函数，第一个参数为this的元素引用
/// @param tensors 输入的张量或标量
/// @return 返回自身引用
template <typename Func, typename... Args>
yt::YTensor<T, dim>& broadcastInplace(Func&& func, Args&&... tensors);

/// @brief YTensor的算术运算符，一次性支持Tensor op Scalar 或者 Tensor op Tensor 的原地以及非原地操作。
#define YT_YTENSOR_OPERATOR_DEF(op)                                                   \
    template <int dim1>                                                               \
    yt::YTensor<T, std::max(dim, dim1)> operator op(const yt::YTensor<T, dim1> &other) const; \
    template <int dim1>                                                               \
    yt::YTensor<T, std::max(dim, dim1)>& operator op##=(const yt::YTensor<T, dim1> &other);   \
    yt::YTensor<T, dim> operator op(const T &other) const;                                \
    yt::YTensor<T, dim>& operator op##=(const T &other);

YT_YTENSOR_OPERATOR_DEF(+)
YT_YTENSOR_OPERATOR_DEF(-)
YT_YTENSOR_OPERATOR_DEF(*)
YT_YTENSOR_OPERATOR_DEF(/)
YT_YTENSOR_OPERATOR_DEF(%)
YT_YTENSOR_OPERATOR_DEF(&)
YT_YTENSOR_OPERATOR_DEF(|)
YT_YTENSOR_OPERATOR_DEF(^)

#undef YT_YTENSOR_OPERATOR_DEF

/// @brief 矩阵视图，将张量的最后两个维度视为YTensor<T, 2>的矩阵作为标量。
/// @return 矩阵视图
/// @note 仅支持dim>=1的张量调用此方法。默认为行主序。
yt::YTensor<yt::YTensor<T, 2>, std::max(1, dim - 2)> matView() const;

/// @brief 对张量的最后两个维度进行广播矩阵乘法运算。
/// @param other: 右张量输入。
/// @param backend: 矩阵乘法后端，默认使用编译时自动选择的最优后端。
/// @return 矩阵乘法结果张量。
template <int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul(
    const yt::YTensor<T, dim1> &other, 
    yt::infos::MatmulBackend backend = yt::infos::defaultMatmulBackend) const;

/// @brief 对指定轴求和
/// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
/// @return 求和结果
yt::YTensor<T, dim> sum(int axis) const requires(dim > 1);
yt::YTensor<T, dim> sum(std::vector<int> axes) const requires (dim > 1);
T sum(int axis = 0) const requires (dim == 1);

/// @brief 对指定轴求均值
/// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
/// @return 均值结果
yt::YTensor<T, dim> mean(int axis) const requires(dim > 1);
yt::YTensor<T, dim> mean(std::vector<int> axes) const requires (dim > 1);
T mean(int axis = 0) const requires (dim == 1);

/// @brief 对指定轴求最大值
/// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
/// @return 最大值及其索引
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> max(int axis) const requires (dim > 1);
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> max(std::vector<int> axes) const requires (dim > 1);
std::pair<T, int> max(int axis = 0) const requires (dim == 1);

// protected:

/// @brief 矩阵乘法的无优化后端实现，只保证规则正确，相当低效。
/// @param other: 右张量输入。
/// @return 矩阵乘法结果张量。
template<int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_zero_backend(const yt::YTensor<T, dim1>& other) const;

/////////////// Eigen support ///////////////
#if YT_USE_EIGEN
/// @brief Eigen类型转换
using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>;  

/// @brief 转换为Eigen矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
yt::YTensor<EigenMatrixMap, std::max(1, dim - 2)> matViewEigen() const requires(dim > 2);

/// @brief 矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
EigenMatrixMap matViewEigen() const requires(dim <= 2);    

/// @brief 矩阵乘法的Eigen后端实现，开启Eigen时为默认的后端。
template<int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_eigen_backend(const yt::YTensor<T, dim1>& other) const;

#endif // YT_USE_EIGEN

/////////////// AVX2 GEMM 支持 ///////////////
// 仅在x86架构且支持AVX2时可用，否则回退到Eigen或naive实现
#if YT_USE_AVX2
/// @brief 矩阵乘法的AVX2后端实现，仅支持float类型。
/// @note 使用高性能GEMM内核，支持自动布局检测。
template<int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_avx2_backend(const yt::YTensor<T, dim1>& other) const requires std::is_same_v<T, float>;
#endif // YT_USE_AVX2

public: // end of ytensor_math.hpp

// ********************************
// TODO:
// 1. <<左右移运算符仍然未支持
// 2. 返回布尔值的运算符尚未支持