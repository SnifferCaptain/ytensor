/***************
* @file: ytensor_base_math.hpp [inline]
* @brief: YTensorBase 类内置的数学运算功能
* @author: SnifferCaptain
* @date: 2025-12-1
* @version 1.0
* @email: 3586554865@qq.com
***************/

public:

/// @brief 最大子元素中量遍历父张量阈值，超过则使用stride遍历法，否则使用布尔掩码遍历底层存储。
static constexpr double MAX_SUBELEMENT_RATIO = 2.5;

/// @brief 统一的广播原地操作函数，支持N元张量/标量操作
/// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
/// @tparam Args 参数类型，可以是YTensorBase或标量T
/// @param func 操作函数，第一个参数为this的元素引用
/// @param tensors 输入的张量或标量
/// @return 返回自身引用
template<typename Func, typename... Args>
YTensorBase& broadcastInplace(Func&& func, Args&&... tensors);

/// @brief YTensorBase的算术运算符，使用宏同时定义Tensor op Tensor和Tensor op Scalar的原地及非原地版本
/// @note Tensor op Tensor: 支持广播，输出dtype与this相同
/// @note Tensor op Scalar: 输出dtype与this相同
#define YT_YTENSORBASE_OPERATOR_DEF(op)                         \
    YTensorBase operator op(const YTensorBase& other) const;    \
    YTensorBase& operator op##=(const YTensorBase& other);      \
    template<typename T>                                        \
    YTensorBase operator op(const T& scalar) const;             \
    template<typename T>                                        \
    YTensorBase& operator op##=(const T& scalar);

YT_YTENSORBASE_OPERATOR_DEF(+)
YT_YTENSORBASE_OPERATOR_DEF(-)
YT_YTENSORBASE_OPERATOR_DEF(*)
YT_YTENSORBASE_OPERATOR_DEF(/)
YT_YTENSORBASE_OPERATOR_DEF(%)
YT_YTENSORBASE_OPERATOR_DEF(&)
YT_YTENSORBASE_OPERATOR_DEF(|)
YT_YTENSORBASE_OPERATOR_DEF(^)
YT_YTENSORBASE_OPERATOR_DEF(<<)
YT_YTENSORBASE_OPERATOR_DEF(>>)

#undef YT_YTENSORBASE_OPERATOR_DEF

/// @brief YTensorBase的比较运算符，返回dtype="bool"的YTensorBase
#define YT_YTENSORBASE_CMP_OPERATOR_DEF(op)                         \
    YTensorBase operator op(const YTensorBase& other) const;        \
    template<typename T>                                            \
    YTensorBase operator op(const T& scalar) const;

YT_YTENSORBASE_CMP_OPERATOR_DEF(<)
YT_YTENSORBASE_CMP_OPERATOR_DEF(<=)
YT_YTENSORBASE_CMP_OPERATOR_DEF(>)
YT_YTENSORBASE_CMP_OPERATOR_DEF(>=)
YT_YTENSORBASE_CMP_OPERATOR_DEF(==)
YT_YTENSORBASE_CMP_OPERATOR_DEF(!=)

#undef YT_YTENSORBASE_CMP_OPERATOR_DEF

/// @brief 对张量的最后两个维度进行广播矩阵乘法运算
/// @param other 右张量输入，最后两个维度的列数必须与this最后两个维度的行数相等
/// @return 矩阵乘法结果张量
/// @note 广播规则应用于除最后两维之外的维度
YTensorBase matmul(const YTensorBase& other) const;

/// @brief 矩阵视图，将张量的最后两个维度视为2D矩阵作为"标量"
/// @return 返回一个scalar类型为YTensorBase的YTensorBase，每个"元素"是一个2D子张量视图
/// @note 仅支持ndim>=2的张量调用此方法。默认为行主序。
/// @note 返回的YTensorBase的dtype="YTensorBase"，element_size=sizeof(YTensorBase)
/// @note 可配合broadcast等函数使用，实现批量矩阵操作
/// @example shape=[3,4,5,6] -> 返回shape=[3,4]的YTensorBase，每个"元素"是[5,6]的2D张量视图
YTensorBase matView() const;

/// @brief 对指定轴求和
/// @param axis 轴索引，支持负数索引（-1表示最后一轴）
/// @return 求和结果张量，维度减少1
YTensorBase sum(int axis) const;

/// @brief 对多个轴求和
/// @param axes 轴索引向量
/// @return 求和结果张量，维度减少axes.size()
YTensorBase sum(const std::vector<int>& axes) const;

/// @brief 对指定轴求最大值
/// @param axis 轴索引，支持负数索引（-1表示最后一轴）
/// @return pair<最大值张量, 最大值索引张量(dtype=int32)>
std::pair<YTensorBase, YTensorBase> max(int axis) const;

/// @brief 对多个轴求最大值
/// @param axes 轴索引向量
/// @return pair<最大值张量, 最大值索引张量(dtype=int32)>
std::pair<YTensorBase, YTensorBase> max(const std::vector<int>& axes) const;

protected:

/// @brief 矩阵乘法的无优化后端实现，只保证规则正确
/// @param other 右张量输入
/// @return 矩阵乘法结果张量
YTensorBase matmul_naive_backend(const YTensorBase& other) const;

/// @brief 抛出类型不支持的运算异常
/// @param typeName 类型名称
/// @param opName 操作名称
static void throwOperatorNotSupport(const std::string& typeName, const std::string& opName);

/// @brief 抛出形状不匹配异常
/// @param opName 操作名称
/// @param otherShape 另一个张量的形状
void throwShapeNotMatch(const std::string& opName, const std::vector<int>& otherShape) const;

public: // end of naive math

/////////////// Eigen support ///////////////
#if YT_USE_EIGEN

protected:

/// @brief 矩阵乘法的Eigen后端实现，开启Eigen时为默认的后端
/// @param other 右张量输入
/// @return 矩阵乘法结果张量
YTensorBase matmul_eigen_backend(const YTensorBase& other) const;

/// @brief 对张量的最后两个维度应用Eigen矩阵运算（广播版本）
/// @tparam Func Eigen操作函数，签名为 EigenMatrix func(const EigenMatrix& mat) 或类似
/// @param func Eigen矩阵操作函数
/// @param opName 操作名称
/// @return 结果张量，dtype与this相同
/// @note 将最后两维视为矩阵，对每个矩阵应用func，其他维度遵循广播规则
/// @example tensor.applyEigenOp([](auto& mat) { return mat.transpose(); }, "transpose");
template<typename Func>
YTensorBase applyEigenOp(Func&& func, const std::string& opName = "") const;

/// @brief 对两个张量的最后两个维度应用Eigen矩阵二元运算（广播版本）
/// @tparam Func Eigen二元操作函数，签名为 EigenMatrix func(const EigenMatrix& a, const EigenMatrix& b)
/// @param other 另一个张量
/// @param func Eigen矩阵二元操作函数
/// @param opName 操作名称
/// @return 结果张量
/// @note 将最后两维视为矩阵，对每对矩阵应用func，其他维度遵循广播规则
/// @example a.applyEigenBinaryOp(b, [](auto& a, auto& b) { return a * b; }, "matmul");
template<typename Func>
YTensorBase applyEigenBinaryOp(const YTensorBase& other, Func&& func, const std::string& opName = "") const;

#endif // YT_USE_EIGEN

/////////////// AVX2 support ///////////////
#if YT_USE_AVX2
protected:
/// @brief 矩阵乘法的AVX2后端实现
/// @param other 右张量输入
/// @return 矩阵乘法结果张量
YTensorBase matmul_avx2_backend(const YTensorBase& other) const;
#endif // YT_USE_AVX2

public: // end of ytensor_base_math.hpp
