#pragma once
/***************
* @file: ytensor.hpp
* @brief: 易于使用的张量类，主要用作容器。
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <vector>
#include <cstddef>
#include <utility>
#include <iostream>
#include <functional>
#include <fstream>
#include <memory>
#include <random>

#pragma once
#include <concepts>
#include <initializer_list>
#include <iostream>

namespace yt::concepts {
    // binary operation concepts
    template<typename T> concept HAVE_ADD = requires(T a, T b) { { a + b } -> std::same_as<T>; };
    template<typename T> concept HAVE_SUB = requires(T a, T b) { { a - b } -> std::same_as<T>; };
    template<typename T> concept HAVE_MUL = requires(T a, T b) { { a * b } -> std::same_as<T>; };
    template<typename T> concept HAVE_DIV = requires(T a, T b) { { a / b } -> std::same_as<T>; };
    template<typename T> concept HAVE_MOD = requires(T a, T b) { { a % b } -> std::same_as<T>; };
    template<typename T> concept HAVE_AND = requires(T a, T b) { { a & b } -> std::same_as<T>; };
    template<typename T> concept HAVE_OR = requires(T a, T b) { { a | b } -> std::same_as<T>; };
    template<typename T> concept HAVE_XOR = requires(T a, T b) { { a ^ b } -> std::same_as<T>; };
    template<typename T> concept HAVE_LSHIFT = requires(T a, int b) { { a << b } -> std::same_as<T>; };
    template<typename T> concept HAVE_RSHIFT = requires(T a, int b) { { a >> b } -> std::same_as<T>; };

    // inplace binary operation concepts
    template<typename T> concept HAVE_ADD_INPLACE = requires(T a, T b) { { a += b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_SUB_INPLACE = requires(T a, T b) { { a -= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_MUL_INPLACE = requires(T a, T b) { { a *= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_DIV_INPLACE = requires(T a, T b) { { a /= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_MOD_INPLACE = requires(T a, T b) { { a %= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_AND_INPLACE = requires(T a, T b) { { a &= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_OR_INPLACE = requires(T a, T b) { { a |= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_XOR_INPLACE = requires(T a, T b) { { a ^= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_LSHIFT_INPLACE = requires(T a, int b) { { a <<= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_RSHIFT_INPLACE = requires(T a, int b) { { a >>= b } -> std::same_as<T&>; };

    // unary operation concepts
    template<typename T> concept HAVE_NOT = requires(T a) { { ~a } -> std::same_as<T>; };

    template<typename T> concept HAVE_EQ = requires(T a, T b) { { a == b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_NEQ = requires(T a, T b) { { a != b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_LT = requires(T a, T b) { { a < b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_LE = requires(T a, T b) { { a <= b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_GT = requires(T a, T b) { { a > b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_GE = requires(T a, T b) { { a >= b } -> std::same_as<bool>; };

    // other
    template <typename T>
    concept HAVE_OSTREAM = requires(std::ostream &os, const T &value) {
        { os << value } -> std::same_as<std::ostream &>;
    };

    // array min/max constexpr
    template <typename T>
    constexpr inline T CONSTEXPR_MAX(std::initializer_list<T> list) {
        return *std::max_element(list.begin(), list.end());
    }
};
#pragma once
/***************
* @file: ytensor_infos.hpp
* @brief: 存储一些全局静态信息的命名空间
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <random>
#include <atomic>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <string>
#include <cstdint>

namespace yt::infos{
    static constexpr double minParOps = 29609.;
    static constexpr double flopAdd = 1.;
    static constexpr double flopSub = 1.;
    static constexpr double flopMul = 1.;
    static constexpr double flopDiv = 1.;
    static constexpr double flopMod = 0.003;
    static constexpr double ilopAdd = 3.38;
    static constexpr double ilopSub = 3.38;
    static constexpr double ilopMul = 3.38;
    static constexpr double ilopDiv = 3.38;
    static constexpr double ilopMod = 0.03;
    static constexpr double ilopAnd = 1.39;
    static constexpr double ilopOr = 1.39;
    static constexpr double ilopXor = 1.466;

    /// @brief 矩阵乘法多核并行条件判断函数
    inline double flopMatmul(int m, int n, int k) {
        // 64k 16*16 b=32 op=128k
        // 32k 32*32 b=16 op=128k
        // 256k 64*64 b=8 op=2m
        // 1m 128*128 b=8 op=16m
        constexpr float scale = 5e-4;// 5e-2 if debug
        return static_cast<double>(std::max(m * n + n * k, m * n * k)) * scale;
    }

    /// @brief 随机数生成器，调用时必须上锁
    inline std::mt19937 gen = std::mt19937(std::random_device{}());

    /// @brief 随机数生成锁
    inline std::mutex rngMutex;

    /// @brief 设置数据类型转换方式
    static constexpr enum class RoundMode{
        nearest = 0,    // 四舍五入，偏差还行
        nearestEven = 1,// 标准转换，最小偏差
        truncate = 2    // 直接截断，速度最快，偏低。
    } roundMode = RoundMode::nearestEven;

    /// @brief 类型注册表
    /// @return 返回类型注册表的引用
    inline std::unordered_map<std::string, std::pair<std::string, int32_t>>& getTypeRegistry() {
        static std::unordered_map<std::string, std::pair<std::string, int32_t>> registry;
        return registry;
    }

    /// @brief 文件头标识
    static constexpr std::string_view YTENSOR_FILE_MAGIC = "YTENSORF";

    /// @brief 文件版本
    static constexpr uint8_t YTENSOR_FILE_VERSION = 0;

    /// @brief 控制是否启用Eigen库的宏，默认启用
    #ifndef YT_USE_EIGEN
        #if __has_include(<Eigen/Dense>)
            #define YT_USE_EIGEN 1
        #else
            #define YT_USE_EIGEN 0
        #endif
    #endif
}// namespace yt::infos

/////////////// extern includes ///////////////

#if YT_USE_EIGEN
#include <Eigen/Dense>
#endif // YT_USE_EIGEN


/**
 * @brief 易于使用的张量类。可以处理任意维度的张量。
 * @tparam T 张量元素的数据类型。
 * @tparam dim 张量的维度数。
 */
template <typename T=float, int dim=1>
class YTensor{
protected:
    /// @brief 抛出维度不匹配的异常
    void throwShapeNotMatch(const std::string& funcName, const std::vector<int>& otherShape) const;
    static void throwShapeNotMatch(const std::string& funcName, const std::vector<int>& thisShape,  const std::vector<int>& otherShape);

    /// @brief 抛出维度数量不匹配的异常
    static void throwShapeSizeNotMatch(const std::string& funcName, int otherDim);

    /// @brief 抛出操作符不支持的异常
    static void throwOperatorNotSupport(const std::string& typeName, const std::string& opName);

    /// @brief 随机数生成器辅助结构体
    struct _RandnGenerator {
        _RandnGenerator(std::mt19937& gen_p): gen(gen_p) {}
        std::mt19937& gen;
        template<typename... Args>
        auto operator()(Args... args) const -> YTensor<T, sizeof...(args)>{
            std::normal_distribution<double> dist(0.0, 1.0);
            auto op = YTensor<T, sizeof...(args)>(args...);
            auto max = op.size();
            std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
            #pragma omp simd
            for (size_t i = 0; i < max; i++){
                op.atData_(i) = static_cast<T>(dist(gen));
            }
            return op;
        }
    };

    /// @brief 随机数生成器辅助结构体
    struct _RanduGenerator {
        _RanduGenerator(std::mt19937& gen_p): gen(gen_p) {}
        std::mt19937& gen;
        template<typename... Args>
        auto operator()(Args... args) const -> YTensor<T, sizeof...(args)>{
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            auto op = YTensor<T, sizeof...(args)>(args...);
            auto max = op.size();
            std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
            #pragma omp simd
            for (size_t i = 0; i < max; i++){
                op.atData_(i) = static_cast<T>(dist(gen));
            }
            return op;
        }
    };

public:
    ~YTensor() = default;
    YTensor();
    template <typename U, int dim2> friend class YTensor;
    using scalarType = T;
    static constexpr int ndim = dim;

    /// @brief 构造函数。
    /// @param dims 张量的形状。
    /// @example YTensor<float, 3> a({3, 4, 5});
    YTensor(const std::vector<int> shape);

    /// @brief 构造函数。
    /// @param args 张量的形状。
    /// @example YTensor<float, 3> a(3, 4, 5);
    template <typename... Args>
    YTensor(Args... args);

    /// @brief 构造函数。
    /// @param list 张量的形状。
    /// @example YTensor<float, 3> a={3, 4, 5};
    YTensor(std::initializer_list<int> list);

    /// @brief 拷贝构造函数。默认行为是浅拷贝。
    /// @param other 源张量。
    YTensor(const YTensor& other);

    /// @brief 赋值操作符。默认行为是浅拷贝。
    /// @param other 源张量。
    YTensor<T, dim> &operator=(const YTensor& other);

    /// @brief 浅拷贝
    /// @param other 被赋值的张量。
    void shallowCopyTo(YTensor& other) const;

    /// @brief 浅拷贝
    /// @param other 被赋值的张量。
    void shareTo(YTensor& other) const;

    /// @brief 浅拷贝
    /// @param other 源张量
    /// @return 返回自身的引用。
    YTensor& shallowCopyFrom(const YTensor& src);

    /// @brief 浅拷贝
    /// @param other 源张量
    /// @return 返回自身的引用。
    YTensor& shareFrom(const YTensor& src);

    /// @brief 深拷贝
    /// @param other 源张量
    /// @return 返回深拷贝后的独立张量。
    YTensor clone() const;

    /// @brief 预留连续空间。注意原本的数据会被清空。与构造函数的逻辑相同。
    /// @param shape 张量的形状。
    /// @return 返回当前的张量的引用。
    YTensor& reserve(const std::vector<int>& shape);

    /// @brief 获取张量的数据指针，可以使用stride_()的步长进行安全访问。已经包含偏移量。
    /// @return 返回张量数据的指针。
    /// @note 如果需要获取内存段的数据指针，请使用data_()方法。
    T* data();
    const T *data() const;

    /// @brief 获取张量数据区的指针（在非contiguous情况下不建议使用）
    /// @return 返回张量数据的指针。
    T *data_();
    const T *data_() const;

    /// @brief 获取张量的数据智能指针（在非contiguous情况下不建议使用）
    /// @return 返回张量数据的指针。
    std::vector<T>& dataVector();
    const std::vector<const T>& dataVector() const;

    /// @brief 获取张量的元素个数
    /// @return 返回张量的元素个数。
    /// @example YTensor<float, 3> a(3, 4, 5); auto size = a.size(); // size = 60
    /// @note 如果张量是非contiguous的，size()方法返回的是逻辑个数，而不是实际个数。
    size_t size() const;

    /// @brief 获取张量的形状。可以根据这个形状信息进行数据的访问和操作。如at()方法或者下标(operator[])。
    /// @return 返回张量的形状。
    /// @example YTensor<float, 3> a(3, 4, 5); auto shape = a.shape(); // shape = [3, 4, 5]
    std::vector<int> shape() const;

    /// @brief 获取张量的形状在某个轴上的大小
    /// @param atDim 维度索引，从0开始。
    /// @return 返回指定维度的大小。
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimSize = a.shape(1); // dimSize = 4
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimSize = a.shape(-1); // dimSize = 5
    int shape(int atDim) const;

    /// @brief 检查张量的形状是否与另一个形状匹配
    /// @param otherShape 另一个张量的形状。
    /// @return 如果形状匹配则返回true，否则返回false。
    bool shapeMatch(const std::vector<int> &otherShape) const;

    /// @brief 获取张量的形状在某个轴上的大小（无安全检查、循环版的高效实现）
    /// @param atDim 维度索引，从0开始。
    /// @return 返回指定维度的大小。
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimSize = a.shape(1); // dimSize = 4
    int shape_(int atDim) const;

    constexpr int shapeSize() const;

    /// @brief 获取张量的步长
    /// @return 返回张量的步长。
    /// @example YTensor<float, 3> a(3, 4, 5); auto stride = a.stride();// stride = [20, 5, 1]
    /// @note 如果张量是非contiguous的，stride()方法返回的是逻辑步长，而不是实际的内存步长。
    //       如果需要获取实际的内存步长，请使用stride_()方法。
    std::vector<int> stride() const;

    /// @brief 获取张量在某个轴上的步长
    /// @param atDim 维度索引
    /// @return 返回指定维度的步长。
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimStride = a.stride(1); // dimStride = 4
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimStride = a.stride(-1); // dimStride = 5
    /// @note 如果张量是非contiguous的，stride(int)方法返回的是逻辑步长，而不是实际的内存步长。
    //       如果需要获取实际的内存步长，请使用stride_(int)方法。
    int stride(int atDim) const;

    /// @brief 获取张量的真实步长
    /// @return 返回张量的实际内存步长。
    /// @example YTensor<float, 3> a(3, 4, 5); auto stride = a.stride_(); // stride = [20, 5, 1]
    std::vector<int> stride_() const;

    /// @brief 获取张量在某个轴上的真实步长
    /// @param atDim 维度索引
    /// @return 返回指定维度的真实步长。
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimStride = a.stride(1); // dimStride = 4
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimStride = a.stride(-1); // dimStride = 5
    int stride_(int atDim) const;

    /// @brief 获取张量相对数据指针的真实偏移量，或者对应子张量或元素的偏移量
    /// @param index 元素的索引
    /// @return 返回张量、子张量或者对应元素的偏移量。
    template<typename... Args> int offset(Args... index) const;
    int offset(const std::vector<int>& index) const;

    /// @brief 获取张量的连续版本
    /// @return 返回连续张量。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    YTensor contiguous() const;

    /// @brief 原地操作，使张量连续
    /// @return 返回自身的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    /// @note 这会导致原先的“引用”失效
    YTensor &contiguous_();

    /// @brief 张量是否是连续的
    /// @param fromDim 从该维度开始检查
    /// @return 如果张量是连续的，返回true；否则返回false。
    /// @example YTensor<float, 3> a(3, 4, 5); bool is_contiguous = a.isContiguous(); // is_contiguous = true
    bool isContiguous(int fromDim = 0) const;

    /// @brief 获取张量从哪个维度开始是连续的
    /// @return 返回从哪个维度开始是连续的。如果完全不连续的话，就返回dim
    int isContiguousFrom() const;

    /// @brief 获取这个张量尽可能连续的视图，注意张量的形状会发生变化。
    /// @return 返回一个张量，表示当前张量最大可能的连续排布的视图。
    /// @note 这个张量并不能保证连续。
    YTensor mostContinuousView() const;

    /// @brief 张量是否是不重叠的
    /// @return 如果张量是不重叠的，返回true；否则返回false。
    bool isDisjoint() const;

    /// @brief 获取坐标对应的位置，可以使用atData()方法获取数据。
    /// @return 偏移量
    /// @example YTensor<float, 3> a(3, 4, 5); auto offset = a.toIndex(1, 2, 3); // offset = 31
    /// @note 如果张量是非contiguous的，toIndex()方法返回的是逻辑索引，而不是实际的内存索引。
    //       如果需要获取实际的内存索引，请使用toIndex_()方法
    template <typename... Args> size_t toIndex(const Args... args) const;
    size_t toIndex(const std::vector<int> &pos) const;
    size_t toIndex(const int pos[]) const;

    /// @brief 获取坐标对应的物理位置，可以使用atData_()方法获取数据。
    /// @return 相对data的偏移量
    /// @example YTensor<float, 3> a(3, 4, 5); auto offset = a.toIndex_(1, 2, 3); // offset = 31
    template <typename... Args> size_t toIndex_(const Args... args) const;
    size_t toIndex_(const std::vector<int> &pos) const;
    size_t toIndex_(const int pos[]) const;

    /// @brief 获取位置对应的坐标
    /// @param index 位置
    /// @return 坐标
    /// @note 输入必须是逻辑位置。
    std::vector<int> toCoord(size_t index) const;

    /// @brief 获取对应地址的数据
    /// @return 返回张量元素的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.at(1, 2, 3) = 10.0f; auto value = a.atData(31);// value = 10.0f
    /// @note 这个方法没有at高效，只是用作对原始逻辑的兼容
    T& atData(int index);
    const T& atData(int index) const;

    /// @brief 获取对应实际地址的数据
    /// @return 返回张量元素的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.at(1, 2, 3) = 10.0f; auto value = a.atData_(31);// value = 10.0f
    /// @note 可以与用作element-wise操作，相对高效，但是要求contiguous，且不进行安全检查。计算方式是 id=数据指针+偏移量+index
    T& atData_(int index);
    const T& atData_(int index) const;

    /// @brief 通过坐标获取张量的元素
    /// @return 返回张量元素的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.at(1, 2, 3) = 10.0f; auto value = a.at(1, 2, 3);
    template <typename... Args> T& at(const Args... args);
    T& at(const std::vector<int> &pos);
    T& at(const int pos[]);
    template <typename... Args> const T& at(const Args... args) const;
    const T& at(const std::vector<int> &pos) const;
    const T& at(const int pos[]) const;

    /// @brief 通过坐标获取张量的元素，或者获取子张量
    /// @return 返回张量元素的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.at(1, 2, 3) = 10.0f; a[1][2][-2] = 10.0f;
    /// @note 如果较高的性能需求或者频繁访问，推荐使用at()的方法，而非下标（[]）访问元素
    YTensor<T, dim - 1> operator[](int index) requires(dim > 1);
    const YTensor<T, dim - 1> operator[](int index) const requires(dim > 1);
    T& operator[](int index) requires(dim == 1);
    const T& operator[](int index) const requires(dim == 1);

    /// @brief 对指定轴进行切片操作，返回一个新的张量。
    /// @param atDim 切片的维度索引，从0开始。
    /// @param start 切片的起始位置，默认为0。
    /// @param end 切片的结束位置，默认为0，表示到最后一个元素。非包含
    /// @param step 切片的步长，默认为1。
    /// @param autoFix 是否自动修正切片范围，默认为true。
    /// @return 返回切片后的张量。
    /// @example YTensor<float, 3> a(3, 4, 5); auto slice = a.slice(1, 1, 3, 1); // slice:[3, 2(1, 2), 5]
    /// @note 返回的张量依然指向原始数据。可以使用等号或者contiguous()得到新的张量。如果原地切片，使用slice_()方法。
    /// @note 与python的切片操作不同，无论step是正是负，start都须小于end，否则会返回空张量，如果启用了autoFix，则会自动修正切片范围。
    YTensor<T, dim> slice(int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true) const;

    /// @brief 对指定轴进行原地切片操作，返回一个新的张量。
    /// @param atDim 切片的维度索引，从0开始。
    /// @param start 切片的起始位置，默认为0。
    /// @param end 切片的结束位置，默认为0，表示到最后一个元素。非包含
    /// @param step 切片的步长，默认为1。
    /// @param autoFix 是否自动修正切片范围，默认为true。
    /// @return 返回自身的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.slice_(1, 1, 3, 1); // a:[3, 4, 5] --> [3, 2(1, 2), 5]
    /// @note 与python的切片操作不同，无论step是正是负，start都须小于end，否则会返回空张量，如果启用了autoFix，则会自动修正切片范围。
    YTensor<T, dim>& slice_(int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true);

    /// @brief 改变张量排布顺序
    /// @param newOrder 新的维度顺序
    /// @return 返回自身的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.permute(2, 0, 1); // a:[5, 3, 4]
    /// @note 这个方法会创建新的张量，如果需要原地操作，请使用permute_()方法。
    template <typename... Args> YTensor<T, dim> permute(const Args... newOrder) const;
    YTensor<T, dim> permute(const std::vector<int>& newOrder) const;
    YTensor<T, dim> permute(const int newOrder[]) const;

    /// @brief 交换张量的维度顺序
    /// @param dim0, dim1 交换的维度，默认值是转置最后两个维度
    /// @return 返回交换后的张量
    YTensor<T, dim> transpose(int dim0 = -2, int dim1 = -1) const;

    /// @brief 自动推导合适形状
    /// @param shape 形状
    /// @return 返回一个vector<int>，表示推断出的张量的形状。
    /// @example YTensor<float, 3> a(3, 4, 5); auto inferredShape = a.autoShape({-1, 2, 2, -1}); // inferredShape = [3, 2, 2, 5]
    /// @note 自动推导逻辑，按优先级排序：1、形状相同，返回形状。2、存在一个-1，则在-1维度填充。
    //          3、存在多个-1，表示与原形状对应位置相同，最后一个-1依然是自动填充
    template<typename... Args> std::vector<int> autoShape(const Args... shape) const;
    std::vector<int> autoShape(const std::vector<int>& shape) const;

    /// @brief 设置视图，返回一个新的张量，指向原始数据的不同视图，要求张量contiguous
    /// @param newShape 新的形状
    /// @return 返回新的张量视图。
    /// @example YTensor<float, 3> a(3, 4, 5); auto view = a.view({3, 4, 5}); // view:[3, 4, 5]
    template <typename... Args> auto view(const Args... newShape) const -> YTensor<T, sizeof...(Args)>;
    template <int newdim> YTensor<T, newdim> view(const std::vector<int>& newShape) const;
    template <int newdim> YTensor<T, newdim> view(const int newShape[]) const;

    /// @brief 张量的维度进行重复，要求会被重复的维度，其长度必须为1。
    /// @param times 重复的个数，如果是小于等于1表示当前维度不变。
    /// @return 返回新张量
    template <typename... Args> YTensor<T, dim> repeat(const Args... times) const;
    YTensor<T, dim> repeat(const std::vector<int>& times) const;
    YTensor<T, dim> repeat(const int times[]) const;

    /// @brief 沿维度进行滑动窗口展开，展开方式是将dim维度原地拆分为[滑动窗口大小， 这个维度上可容纳的窗口数量]，其余维度不变。
    /// @param atDim 展开的维度索引
    /// @param kernel 滑动窗口大小
    /// @param stride 步长
    /// @param dilation 膨胀
    /// @return 返回展开后的张量
    /// @note 按照标准卷积操作的方式来看，Y = Kernel @ Unfolded 是更符合权重排布的。
    YTensor<T, dim + 1> unfold(int atDim, int kernel, int stride = 1, int dilation = 1) const;

    /// @brief 创建指定大小的，零张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensor<T, dim> zeros(const std::vector<int>& shape);
    static YTensor<T, dim> zeros(const int shape[]);
    template<typename... Args> static YTensor<T, sizeof...(Args)> zeros(Args... args);
    static YTensor<T, dim> zeros(const std::initializer_list<int>& shape);

    /// @brief 创建指定大小的，全1张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensor<T, dim> ones(const std::vector<int>& shape);
    static YTensor<T, dim> ones(const int shape[]);
    template<typename... Args> static YTensor<T, sizeof...(Args)> ones(const Args... args);
    static YTensor<T, dim> ones(const std::initializer_list<int>& shape);

    /// @brief 创建指定大小的，高斯分布U(0, 1)的张量
    /// @param shape 张量的形状
    /// @return 返回张量
    /// @note 注意多线程环境下可能会出现互斥锁导致的性能瓶颈。
    static _RandnGenerator randn;

    /// @brief 创建指定大小的，均匀分布N(0, 1)的张量
    /// @param shape 张量的形状
    /// @return 返回张量
    /// @note 注意多线程环境下可能会出现互斥锁导致的性能瓶颈。
    static _RanduGenerator randu;

    /// @brief 设置随机数种子，全局共享
    /// @param seed 随机数种子，默认为真随机种子
    /// @example YTensor<>::seed(42);
    static void seed(unsigned int seed = std::random_device{}());

    /// @brief 按置进行操作
    /// @param func 对每个元素进行操作的函数，函数签名为void func(T& value, const std::vector<int>& coord) -> T/void
    /// @param flop 每次操作的运算量，用于判断是否需要多线程计算，默认值为1e-11即禁用多线程
    /// @return 返回自身引用
    /// @note foreach的速度没有使用binaryOpTransformInplace快。但是它提供了coord的接口。
    template<typename Func>
    YTensor<T, dim>& foreach(Func&& func, double flop = 1e-11);

    /// @brief 填充张量
    /// @param value 填充的值
    /// @return 返回自身引用
    YTensor<T, dim>& fill(T value);

    /// @brief 标准cout输出流
    template<typename U, int d>
    friend std::ostream& operator<<(std::ostream& os, const YTensor<U, d>& tensor);

/////////////////// externs ////////////////
    /***************
    * @file: ytensor_math.hpp [inline]
    * @brief: YTensor 类内置的数学运算功能
    * @author: SnifferCaptain
    * @date: 2025-10-24
    * @version 1.0
    * @email: 3586554865@qq.com
    ***************/

    public:

    /// @brief 最大子元素中量遍历父张量阈值，超过则使用遍历法，否则使用布尔掩码。
    static constexpr int MAX_SUBELEMENT_RATIO = 2;

    /// @brief 获取两个张量形状的广播计算后的输出形状
    /// @param otherShape: 形状向量
    /// @return 返回广播后的形状
    std::vector<int> broadcastShape(std::vector<int> otherShape) const;

    /// @brief 对两个张量广播二元运算
    /// @param other: 另一个张量
    /// @param func: 二元运算函数，函数签名为T func(const T& a, const T& b)。也可以是T func(const T& a, const T& b, T& dest)。其中a、b为操作数，dest为结果。
    /// @param opName: 操作名称，用于报错
    /// @param result: 可指定结果张量的指针，nullptr表示返回值为结果张量，默认值为nullptr
    /// @param flop: 用于控制开启多核计算的阈值，值越大表示操作的开销越大，越容易触发多核计算，默认值为1.0（接近于单次浮点加法的计算量）
    /// @return 返回结果张量，维度数为输入两个张量的最大值。
    template <int dim1, typename Func>
    YTensor<T, std::max(dim, dim1)> binaryOpBroadcast(const YTensor<T, dim1> &other, Func &&func, std::string opName = "",
        YTensor<T, std::max(dim, dim1)>* result = nullptr, double flop = 1.) const;

    /// @brief 对两个张量广播二元原地运算
    /// @param other: 另一个张量
    /// @param func: 二元运算函数，函数签名为T func(T& a, const T& b)。或者void func(T& a, const T& b)。其中默认返回值会被赋值给a对应的位置。
    /// @param opName: 操作名称，用于报错
    /// @param result: 可指定结果张量的指针，nullptr表示返回值为结果张量，默认值为nullptr
    /// @param flop: 用于控制开启多核计算的阈值，值越大表示操作的开销越大，越容易触发多核计算，默认值为1.0（接近于单次浮点加法的计算量）
    /// @return 返回结果张量，维度数为输入两个张量最大值。
    template <int dim1, typename Func>
    YTensor<T, dim> &binaryOpBroadcastInplace(const YTensor<T, dim1> &other, Func &&func, std::string opName = "", double flop = 1.);

    /// @brief 对张量进行逐元素二元运算。
    /// @param other: 标量值。
    /// @param func: 二元运算函数，函数签名为T func(const T& a, const T& b)。也可以是T func(T& a, const T& b, T& dest)。其中a、b为操作数，dest为结果。
    /// @param result: 可指定结果张量的指针，nullptr表示返回值为结果张量，默认值为nullptr
    /// @param flop: 用于控制开启多核计算的阈值，值越大表示操作的开销越大，越容易触发多核计算，默认值为1.0（接近于单次浮点加法的计算量）
    /// @return 返回结果张量，维度数为输入两个张量最大值。
    template <typename Func>
    YTensor<T, dim> binaryOpTransform(const T &other, Func &&func, YTensor<T, dim> *result = nullptr, double flop = 1.) const;

    /// @brief 对张量进行逐元素二元原地运算。
    /// @param other: 标量值。
    /// @param func: 二元运算函数，函数签名为T func(T& a, const T& b)。或者void func(T& a, const T& b)。其中默认返回值会被赋值给a对应位置。
    /// @param result: 可指定结果张量的指针，nullptr表示返回值为结果张量，默认值为nullptr
    /// @param flop: 用于控制开启多核计算的阈值，值越大表示操作的开销越大，越容易触发多核计算，默认值为1.0（接近于单次浮点加法的计算量）
    /// @return 返回结果张量，维度数为输入两个张量最大值。
    template <typename Func>
    YTensor<T, dim>& binaryOpTransformInplace(const T &other, Func &&func, double flop = 1.);

    /// @brief YTensor的算术运算符，一次性支持Tensor op Scalar 或者 Tensor op Tensor 的原地以及非原地操作。
    #define YT_YTENSOR_OPERATOR_DEF(op)                                                   \
        template <int dim1>                                                               \
        YTensor<T, std::max(dim, dim1)> operator op(const YTensor<T, dim1> &other) const; \
        template <int dim1>                                                               \
        YTensor<T, std::max(dim, dim1)>& operator op##=(const YTensor<T, dim1> &other);   \
        YTensor<T, dim> operator op(const T &other) const;                                \
        YTensor<T, dim>& operator op##=(const T &other);

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
    YTensor<YTensor<T, 2>, std::max(1, dim - 2)> matView() const;

    /// @brief 对张量的最后两个维度进行广播矩阵乘法运算。
    /// @param other: 右张量输入。
    /// @return 矩阵乘法结果张量。
    template <int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul(const YTensor<T, dim1> &other) const;

    /// @brief 对指定轴求和
    /// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
    /// @return 求和结果
    YTensor<T, dim> sum(int axis) const requires(dim > 1);
    YTensor<T, dim> sum(std::vector<int> axes) const requires (dim > 1);
    T sum(int axis = 0) const requires (dim == 1);

    /// @brief 对指定轴求最大值
    /// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
    /// @return 最大值及其索引
    std::pair<YTensor<T, dim>, YTensor<int, dim>> max(int axis) const requires (dim > 1);
    std::pair<YTensor<T, dim>, YTensor<int, dim>> max(std::vector<int> axes) const requires (dim > 1);
    std::pair<T, int> max(int axis = 0) const requires (dim == 1);

    protected:

    /// @brief 矩阵乘法的无优化后端实现，只保证规则正确，相当低效。
    /// @param other: 右张量输入。
    /// @return 矩阵乘法结果张量。
    template<int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_zero_backend(const YTensor<T, dim1>& other) const;

    public: // end of naive math

    /////////////// Eigen support ///////////////
    #if YT_USE_EIGEN
    protected:

    /// @brief Eigen类型转换
    using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>;

    /// @brief 转换为Eigen矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
    YTensor<EigenMatrixMap, std::max(1, dim - 2)> matViewEigen() const requires(dim > 2);

    /// @brief 矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
    EigenMatrixMap matViewEigen() const requires(dim <= 2);

    /// @brief 矩阵乘法的Eigen后端实现，开启Eigen时为默认的后端。
    template<int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_eigen_backend(const YTensor<T, dim1>& other) const;

    public:
    #endif // YT_USE_EIGEN
    public: // end of ytensor_math.hpp

    // ********************************
    // TODO:
    // 1. <<左右移运算符仍然未支持

protected:
    std::shared_ptr<std::vector<T>> _data; // 张量数据
    std::vector<int> _shape;    // 张量的形状
    std::vector<int> _stride;   // 张量的步长
    int _offset;                // 张量距离指针的偏移量
};

//////////// implementation /////////////

#include <cstddef>
#include <algorithm>
#include <map>
#include <deque>
#include <cassert>
#include <iostream>
#include <cstdarg>
#include <ranges>
#include <omp.h>

#pragma once
/***************
* @file: ytensor_types.hpp
* @brief: YTensor 数据类型定义
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <cstdint>
#include <string>
#include <unordered_map>
#include <type_traits>
#pragma once
/***************
* @file: bfloat16.hpp
* @brief: bfloat16 数据类型定义。
* @author: https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/core/type/bfloat16.hpp
* @date: 2025-10-24
* @note 复制自openvino的bf16实现，做了少量修改以适配ytensor。
***************/

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>

namespace yt {
class bfloat16 {
public:
    bfloat16() = default;
    bfloat16(float value): m_value{round(value)} {}

    template <typename I>
    explicit bfloat16(I value) : m_value{bfloat16{static_cast<float>(value)}.m_value} {}

    std::string to_string() const;
    size_t size() const;
    template <typename T>
    bool operator==(const T& other) const;
    template <typename T>
    bool operator!=(const T& other) const {
        return !(*this == other);
    }
    template <typename T>
    bool operator<(const T& other) const;
    template <typename T>
    bool operator<=(const T& other) const;
    template <typename T>
    bool operator>(const T& other) const;
    template <typename T>
    bool operator>=(const T& other) const;
    template <typename T>
    bfloat16 operator+(const T& other) const;
    template <typename T>
    bfloat16 operator+=(const T& other);
    template <typename T>
    bfloat16 operator-(const T& other) const;
    template <typename T>
    bfloat16 operator-=(const T& other);
    template <typename T>
    bfloat16 operator*(const T& other) const;
    template <typename T>
    bfloat16 operator*=(const T& other);
    template <typename T>
    bfloat16 operator/(const T& other) const;
    template <typename T>
    bfloat16 operator/=(const T& other);
    operator float() const;

    static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
    static std::vector<bfloat16> from_float_vector(const std::vector<float>&);
    static constexpr bfloat16 from_bits(uint16_t bits) {
        return bfloat16(bits, true);
    }
    uint16_t to_bits() const;
    friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj) {
        out << static_cast<float>(obj);
        return out;
    }

    static uint16_t round_to_nearest_even(float x) {
        return static_cast<uint16_t>((cu32(x) + ((cu32(x) & 0x00010000) >> 1)) >> 16);
    }

    static uint16_t round_to_nearest(float x) {
        return static_cast<uint16_t>((cu32(x) + 0x8000) >> 16);
    }

    static uint16_t truncate(float x) {
        return static_cast<uint16_t>((cu32(x)) >> 16);
    }

    static uint16_t round(float x){
        if constexpr(yt::infos::roundMode == yt::infos::RoundMode::nearestEven){
            return round_to_nearest_even(x);
        } else if constexpr(yt::infos::roundMode == yt::infos::RoundMode::nearest){
            return round_to_nearest(x);
        } else{
            return truncate(x);
        }
    }

private:
    constexpr bfloat16(uint16_t x, bool) : m_value{x} {}
    union F32 {
        F32(float val) : f{val} {}
        F32(uint32_t val) : i{val} {}
        float f;
        uint32_t i;
    };

    static inline uint32_t cu32(float x) {
        return F32(x).i;
    }

    uint16_t m_value;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool bfloat16::operator==(const T& other) const {
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
}

template <typename T>
bool bfloat16::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
bfloat16 bfloat16::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
bfloat16 bfloat16::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
bfloat16 bfloat16::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
bfloat16 bfloat16::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace yt

namespace std {
template <>
class numeric_limits<yt::bfloat16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr yt::bfloat16 min() noexcept {
        return yt::bfloat16::from_bits(0x007F);
    }
    static constexpr yt::bfloat16 max() noexcept {
        return yt::bfloat16::from_bits(0x7F7F);
    }
    static constexpr yt::bfloat16 lowest() noexcept {
        return yt::bfloat16::from_bits(0xFF7F);
    }
    static constexpr int digits = 7;
    static constexpr int digits10 = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr yt::bfloat16 epsilon() noexcept {
        return yt::bfloat16::from_bits(0x3C00);
    }
    static constexpr yt::bfloat16 round_error() noexcept {
        return yt::bfloat16::from_bits(0x3F00);
    }
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr yt::bfloat16 infinity() noexcept {
        return yt::bfloat16::from_bits(0x7F80);
    }
    static constexpr yt::bfloat16 quiet_NaN() noexcept {
        return yt::bfloat16::from_bits(0x7FC0);
    }
    static constexpr yt::bfloat16 signaling_NaN() noexcept {
        return yt::bfloat16::from_bits(0x7FC0);
    }
    static constexpr yt::bfloat16 denorm_min() noexcept {
        return yt::bfloat16::from_bits(0);
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style =
        yt::infos::roundMode == yt::infos::RoundMode::truncate ?
            round_toward_zero
        /*else*/:
            round_to_nearest;
};
}  // namespace std



static_assert(sizeof(yt::bfloat16) == 2, "class bfloat16 must be exactly 2 bytes");

inline bool float_isnan(const float& x) {
    return std::isnan(x);
}

inline std::vector<float> yt::bfloat16::to_float_vector(const std::vector<yt::bfloat16>& v_bf16) {
    std::vector<float> v_f32(v_bf16.begin(), v_bf16.end());
    return v_f32;
}

inline std::vector<yt::bfloat16> yt::bfloat16::from_float_vector(const std::vector<float>& v_f32) {
    std::vector<yt::bfloat16> v_bf16;
    v_bf16.reserve(v_f32.size());
    for (float a : v_f32) {
        v_bf16.push_back(static_cast<yt::bfloat16>(a));
    }
    return v_bf16;
}

inline std::string yt::bfloat16::to_string() const {
    return std::to_string(static_cast<float>(*this));
}

inline size_t yt::bfloat16::size() const {
    return sizeof(m_value);
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuninitialized"
#endif

inline yt::bfloat16::operator float() const {
    uint32_t tmp = 0;
    uint32_t* ptmp = &tmp;
    *ptmp = (static_cast<uint32_t>(m_value) << 16);
    const float* f = reinterpret_cast<const float*>(ptmp);
    return *f;
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic pop
#endif

inline uint16_t yt::bfloat16::to_bits() const {
    return m_value;
}

namespace yt::types {
    /// @brief 获取数据类型名称
    /// @tparam T 数据类型
    /// @return 数据类型名称字符串
    template<typename T>
    std::string getTypeName() {
        // 首先检查是否已注册自定义名称
        auto& registry = yt::infos::getTypeRegistry();
        auto it = registry.find(typeid(T).name());
        if (it != registry.end()) {
            return it->second.first;
        }

        // 使用默认的类型名称
        if constexpr (std::is_same_v<T, float>) return "float32";
        else if constexpr (std::is_same_v<T, double>) return "float64";
        else if constexpr (std::is_same_v<T, int8_t>) return "int8";
        else if constexpr (std::is_same_v<T, int16_t>) return "int16";
        else if constexpr (std::is_same_v<T, int32_t>) return "int32";
        else if constexpr (std::is_same_v<T, int64_t>) return "int64";
        else if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
        else if constexpr (std::is_same_v<T, uint16_t>) return "uint16";
        else if constexpr (std::is_same_v<T, uint32_t>) return "uint32";
        else if constexpr (std::is_same_v<T, uint64_t>) return "uint64";
        else if constexpr (std::is_same_v<T, bool>) return "bool";
        // non std
        else if constexpr (std::is_same_v<T, yt::bfloat16>) return "bfloat16";
        else {
            throw std::runtime_error(std::string("Type ") + typeid(T).name() + " is not registered.");
            return "unregistered";
        }
    }

    /// @brief 获取数据类型大小（模板版本）
    /// @tparam T 数据类型
    /// @return 类型大小（字节）
    template<typename T>
    constexpr int32_t getTypeSize() {
        return static_cast<int32_t>(sizeof(T));
    }

    /// @brief 根据类型名称获取类型大小
    /// @param typeName 类型名称
    /// @return 类型大小（字节），未知类型返回0
    inline int32_t getTypeSize(const std::string& typeName) {
        if (typeName == "float32") return 4;
        else if (typeName == "float64") return 8;
        else if (typeName == "int8") return 1;
        else if (typeName == "int16") return 2;
        else if (typeName == "int32") return 4;
        else if (typeName == "int64") return 8;
        else if (typeName == "uint8") return 1;
        else if (typeName == "uint16") return 2;
        else if (typeName == "uint32") return 4;
        else if (typeName == "uint64") return 8;
        else if (typeName == "bool") return 1;
        // non std
        else if (typeName == "bfloat16") return 2;
        else return 0;
    }

    /// @brief 注册自定义类型
    /// @tparam T 要注册的类型
    /// @param typeName 自定义类型名称
    template<typename T>
    void registerType(const std::string& typeName) {
        auto& registry = yt::infos::getTypeRegistry();
        int32_t typeSize = getTypeSize<T>();
        registry[typeid(T).name()] = {typeName, typeSize};
    }
} // namespace yt::types

template <typename T, int dim>
void YTensor<T, dim>::throwShapeNotMatch(const std::string& funcName, const std::vector<int>& otherShape) const {
    std::string errorMsg = "Function \"" + funcName + "\" shape not match: YTensor[";
    for (int i = 0; i < dim; ++i) {
        errorMsg += std::to_string(_shape[i]);
        if( i < dim - 1) {
            errorMsg += ", ";
        }
    }
    errorMsg += "] and YTensor[";
    for (size_t i = 0; i < otherShape.size(); ++i) {
        errorMsg += std::to_string(otherShape[i]);
        if (i < otherShape.size() - 1) {
            errorMsg += ", ";
        }
    }
    errorMsg += "]";
    throw std::invalid_argument(errorMsg);
}

template <typename T, int dim>
void YTensor<T, dim>::throwShapeNotMatch(const std::string& funcName, const std::vector<int>& thisShape, const std::vector<int>& otherShape) {
    std::string errorMsg = "Function \"" + funcName + "\" shape not match: YTensor[";
    for (int i = 0; i < dim; ++i) {
        errorMsg += std::to_string(thisShape[i]);
        if( i < dim - 1) {
            errorMsg += ", ";
        }
    }
    errorMsg += "] and YTensor[";
    for (size_t i = 0; i < otherShape.size(); ++i) {
        errorMsg += std::to_string(otherShape[i]);
        if (i < otherShape.size() - 1) {
            errorMsg += ", ";
        }
    }
    errorMsg += "]";
    throw std::invalid_argument(errorMsg);
}

template <typename T, int dim>
void YTensor<T, dim>::throwShapeSizeNotMatch(const std::string& funcName, int otherDim) {
    std::string errorMsg = "Function \"" + funcName + "\" shape size not match: YTensor<T, " + std::to_string(dim) + "> and dim size " + std::to_string(otherDim);
    throw std::invalid_argument(errorMsg);
}

template <typename T, int dim>
void YTensor<T, dim>::throwOperatorNotSupport(const std::string& typeName, const std::string& opName) {
    std::string errorMsg = "Operator " + opName + " not supported for type " + typeName;
    throw std::runtime_error(errorMsg);
}

///////////////// ytensor ///////////////

template <typename T, int dim>
YTensor<T, dim>::YTensor():
    _data(nullptr), _shape(dim, 0), _stride(dim, 0), _offset(0){}

template <typename T, int dim>
YTensor<T, dim>::YTensor(const std::vector<int> shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("init", shape.size());
    }
    _shape = shape;
    _stride = this->stride();
    _offset = 0;
    _data = std::make_shared<std::vector<T>>(this->size());
}

template <typename T, int dim>
template <typename... Args>
YTensor<T, dim>::YTensor(Args... args): _shape(dim) {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int a = 0;
    ((_shape[a++] = args), ...);
    _stride = this->stride();
    _offset = 0;
    _data = std::make_shared<std::vector<T>>(this->size());
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::initializer_list<int> list){
    if (list.size() != dim) {
        throwShapeSizeNotMatch("init", list.size());
    }
    _shape = std::vector<int>(list);
    _stride = this->stride();
    _offset = 0;
    _data = std::make_shared<std::vector<T>>(this->size());
}

template<typename T, int dim>
YTensor<T, dim>::YTensor(const YTensor& other){
    if(this == &other){
        return;
    }
    if (dim != other.shapeSize()){
        throwShapeSizeNotMatch("<copy>", other.shapeSize());
    }
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;
    _data = other._data;
}

template <typename T, int dim>
YTensor<T, dim> &YTensor<T, dim>::operator=(const YTensor<T, dim> &other){
    if(this == &other){
        return *this;
    }
    if (dim != other.shapeSize()){
        throwShapeSizeNotMatch("=", other.shapeSize());
    }
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;
    _data = other._data;
    return *this;
}

template <typename T, int dim>
void YTensor<T, dim>::shallowCopyTo(YTensor<T, dim> &other) const {
    if (dim != other.shapeSize()) {
        throwShapeSizeNotMatch("shallowCopyTo", other.shapeSize());
    }
    other._shape = _shape;
    other._stride = this->stride();
    other._offset = _offset;
    other._data = _data;
}

template <typename T, int dim>
void YTensor<T, dim>::shareTo(YTensor<T, dim> &other) const {
    shallowCopyTo(other);
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::shallowCopyFrom(const YTensor<T, dim> &src) {
    if (dim != src.shapeSize()) {
        throwShapeSizeNotMatch("shallowCopyFrom", src.shapeSize());
    }
    _shape = src._shape;
    _stride = src._stride;
    _offset = src._offset;
    _data = src._data;
    return *this;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::shareFrom(const YTensor<T, dim> &src) {
    shallowCopyFrom(src);
    return *this;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::clone() const {
    // fused coord 2 index
    YTensor<T, dim> op(this->shape());
    std::vector<int> indices(dim, 0);
    size_t max = this->size();
    #pragma omp simd
    for(size_t dst = 0; dst < max; dst++){
        size_t thisIndex = _offset;
        size_t index = dst;
        #pragma omp simd reduction(+:thisIndex)
        for (int i = dim - 1; i >= 0; i--) {
            thisIndex += (index % _shape[i]) * _stride[i];
            index /= _shape[i];
        }
        (*op._data)[dst] = (*_data)[thisIndex];
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::reserve(const std::vector<int>& shape) {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("reserve", shape.size());
    }
    _shape = shape;
    _stride = this->stride();
    _offset = 0;
    _data = std::make_shared<std::vector<T>>(this->size());
    return *this;
}

template <typename T, int dim>
T* YTensor<T, dim>::data() {
    return data_() + _offset;
}

template <typename T, int dim>
const T* YTensor<T, dim>::data() const {
    return data_() + _offset;
}

template <typename T, int dim>
T* YTensor<T, dim>::data_() {
    return _data.get()->data();
}

template <typename T, int dim>
const T* YTensor<T, dim>::data_() const {
    return _data.get()->data();
}

template <typename T, int dim>
std::vector<T>& YTensor<T, dim>::dataVector() {
    return *_data;
}

template <typename T, int dim>
const std::vector<const T>& YTensor<T, dim>::dataVector() const {
    return *_data;
}

template <typename T, int dim>
size_t YTensor<T, dim>::size() const {
    size_t total_size = 1;
    for (int i = 0; i < dim; ++i) {
        total_size *= _shape[i];
    }
    return total_size;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::shape() const {
    return _shape;
}

template <typename T, int dim>
int YTensor<T, dim>::shape(int atDim) const {
    atDim = (atDim % dim + dim) % dim;// 循环索引
    return _shape[atDim];
}

template <typename T, int dim>
bool YTensor<T, dim>::shapeMatch(const std::vector<int> &otherShape) const{
    if (otherShape.size() != dim) {
        return false;
    }
    if(_shape.size() != otherShape.size()){
        return false;
    }
    for (int i = 0; i < dim; i++) {
        if (_shape[i] != otherShape[i]) {
            return false;
        }
    }
    return true;
}

template <typename T, int dim>
int YTensor<T, dim>::shape_(int atDim) const {
    return _shape[atDim];
}

template<typename T, int dim>
constexpr int YTensor<T, dim>::shapeSize() const {
    return dim;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::stride() const {
    std::vector<int> op(dim);
    int strd = 1;
    for (int i = dim - 1; i >= 0; --i){
        op[i] = strd;
        strd *= _shape[i];
    }
    return op;
}

template <typename T, int dim>
int YTensor<T, dim>::stride(int atDim) const {
    atDim = (atDim % dim + dim) % dim; // 循环索引
    int strd = 1;
    for (int i = dim - 1; i > atDim; --i) {
        strd *= _shape[i];
    }
    return strd;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::stride_() const {
    return _stride;
}

template <typename T, int dim>
int YTensor<T, dim>::stride_(int atDim) const {
    atDim = (atDim % dim + dim) % dim; // 循环索引
    return _stride[atDim];
}

template <typename T, int dim> template <typename... Args>
int YTensor<T, dim>::offset(Args... index) const {
    static_assert(sizeof...(index) <= dim, "Number of arguments must match the dimension");
    // zero pad to dim
    std::vector<int> indices(dim, 0);
    int a = 0;
    ((indices[a++] = index), ...);
    return toIndex_(indices);
}

template <typename T, int dim>
int YTensor<T, dim>::offset(const std::vector<int>& index) const {
    if (index.size() > dim) {
        throwShapeSizeNotMatch("offset", index.size());
    }
    std::vector<int> indices(dim, 0);
    for (size_t i = 0; i < index.size(); i++) {
        indices[i] = index[i];
    }
    return toIndex_(indices);
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::contiguous() const {
    if (_data == nullptr) {
        return YTensor<T, dim>(this->shape());
    }

    if(this->isContiguous()){
        return *this;
    }
    else{
        return this->clone();
    }
}


template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::contiguous_() {
    if(this->isContiguous()){
        return *this;
    }
    else{
        auto t = this->contiguous();
        _data = t._data;
        _shape = t._shape;
        _stride = t._stride;
        _offset = t._offset;
        return *this;
    }
}

template <typename T, int dim>
bool YTensor<T, dim>::isContiguous(int fromDim) const {
    if (_data == nullptr) {
        return false;
    }
    auto logStride = this->stride();
    if(logStride.size() != dim){
        return false;
    }
    // 检查数据是否是连续的
    fromDim = (fromDim % dim + dim) % dim; // 循环索引
    for (int i = fromDim; i < dim; ++i) {
        if(logStride[i] != _stride[i] && _shape[i] > 1){
            // 步长不匹配
            return false;
        }
    }
    return true;
}

template <typename T, int dim>
int YTensor<T, dim>::isContiguousFrom() const {
    if (_data == nullptr) {
        return dim;
    }
    auto logStride = this->stride();
    if (logStride.size() != dim) {
        return dim;
    }
    // 检查数据从哪个维度开始是连续的
    for (int a = dim - 1; a >= 0; a--) {
        if(logStride[a] != _stride[a] && _shape[a] > 1){
            return a + 1;
        }
    }
    return 0; // 全部连续
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::mostContinuousView() const {
    if (_data == nullptr) {
        return YTensor<T, dim>(this->shape());
    }
    // 按照stride的大小顺序进行排序
    std::vector<std::pair<int, int>> mapper(dim);// shape[i], i
    for (int i = 0; i < dim; ++i) {
        mapper[i] = std::make_pair(_stride[i], i);
    }
    std::sort(mapper.begin(), mapper.end(), [](const auto& a, const auto& b) {
        return std::abs(a.first) > std::abs(b.first);
    });
    std::vector<int> perm(dim);
    for (int i = 0; i < dim; ++i) {
        perm[i] = mapper[i].second;
    }
    auto op = this->permute(perm);
    for(int i = 0; i < dim; i++){
        if(op._stride[i] < 0){
            // 负步长改为正步长，并调整offset
            op._stride[i] = -op._stride[i];
            op._offset += (op._shape[i] - 1) * op._stride[i];
        }
    }
    return op;
}

template <typename T, int dim>
bool YTensor<T, dim>::isDisjoint() const {
    if (_data == nullptr) {
        return false;
    }
    if (dim <= 1) {
        return !(dim == 1 && _shape[0] > 1 && _stride[0] == 0);
    }
    std::vector<int> span_lower(dim, 1);
    for (int d = dim - 2; d >= 0; d--) {
        span_lower[d] = span_lower[d + 1] * _shape[d + 1];
    }
    for (int d = 0; d < dim; d++) {
        if (_shape[d] <= 1) continue;
        int abs_stride = std::abs(_stride[d]);
        // 关键判断：|stride| < 理论最小跨度 → 重叠
        if (abs_stride < span_lower[d]) {
            return false;
        }
    }
    return true;
}

template <typename T, int dim>
template <typename... Args>
size_t YTensor<T, dim>::toIndex(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    return toIndex(indices);
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(const std::vector<int> &pos) const {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("toIndex", pos.size());
    }
    size_t index = 0;
    auto logStride = this->stride();
    for (int i = 0; i < dim; ++i) {
        index += pos[i] * logStride[i];
    }
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex(const int pos[]) const {
    size_t index = 0;
    auto logStride = this->stride();
    for (int i = 0; i < dim; ++i) {
        index += pos[i] * logStride[i];
    }
    return index;
}

template <typename T, int dim>
template <typename... Args>
size_t YTensor<T, dim>::toIndex_(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex_(const std::vector<int> &pos) const {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("toIndex_", pos.size());
    }
    size_t index = 0;
    for (int i = 0; i < dim; ++i) {
        index += pos[i] * _stride[i];
    }
    return index;
}

template <typename T, int dim>
size_t YTensor<T, dim>::toIndex_(const int pos[]) const {
    size_t index = 0;
    for (int i = 0; i < dim; ++i) {
        index += pos[i] * _stride[i];
    }
    return index;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::toCoord(size_t index) const {
    std::vector<int> pos(dim);
    #pragma omp simd
    for (int i = dim - 1; i >= 0; i--) {
        pos[i] = (index % _shape[i]);
        index /= _shape[i];
    }
    return pos;
}

template <typename T, int dim>
T& YTensor<T, dim>::atData(int index) {
    auto coord = toCoord(index);
    return at(coord);
}

template <typename T, int dim>
const T& YTensor<T, dim>::atData(int index) const {
    auto coord = toCoord(index);
    return at(coord);
}

template <typename T, int dim>
T& YTensor<T, dim>::atData_(int index) {
    return (*_data)[_offset + index];
}

template <typename T, int dim>
const T& YTensor<T, dim>::atData_(int index) const {
    return (*_data)[_offset + index];
}

template <typename T, int dim>
template <typename... Args>
T& YTensor<T, dim>::at(const Args... args){
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return atData_(index);
}

template <typename T, int dim>
T& YTensor<T, dim>::at(const std::vector<int>& pos){
    if(pos.size() != dim){
        throwShapeSizeNotMatch("at", pos.size());
    }
    auto index = toIndex_(pos);
    return atData_(index);
}

template <typename T, int dim>
T& YTensor<T, dim>::at(const int pos[]){
    auto index = toIndex_(pos);
    return atData_(index);
}

template <typename T, int dim>
template <typename... Args>
const T& YTensor<T, dim>::at(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return atData_(index);
}

template <typename T, int dim>
const T& YTensor<T, dim>::at(const std::vector<int>& pos) const {
    if(pos.size() != dim){
        throwShapeSizeNotMatch("at", pos.size());
    }
    auto index = toIndex_(pos);
    return atData_(index);
}

template <typename T, int dim>
const T& YTensor<T, dim>::at(const int pos[]) const {
    auto index = toIndex_(pos);
    return atData_(index);
}

template<typename T, int dim>
YTensor<T, dim - 1> YTensor<T, dim>::operator[](int index) requires (dim > 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    std::vector<int> newShape(dim - 1);
    std::vector<int> newStride(dim - 1);
    for (int i = 1; i < dim; ++i) {
        newShape[i - 1] = _shape[i];
        newStride[i - 1] = _stride[i];
    }
    YTensor<T, dim - 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset + index * _stride[0];
    return op;
}

template<typename T, int dim>
const YTensor<T, dim - 1> YTensor<T, dim>::operator[](int index) const requires (dim > 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    std::vector<int> newShape(dim - 1);
    std::vector<int> newStride(dim - 1);
    for (int i = 1; i < dim; ++i) {
        newShape[i - 1] = _shape[i];
        newStride[i - 1] = _stride[i];
    }
    YTensor<T, dim - 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset + index * _stride[0];
    return op;
}

template <typename T, int dim>
T& YTensor<T, dim>::operator[](int index) requires (dim == 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    int bias = index * _stride[0];
    return this->data()[bias];
}

template <typename T, int dim>
const T& YTensor<T, dim>::operator[](int index) const requires (dim == 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    int bias = index * _stride[0];
    return this->data()[bias];
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::slice(int atDim, int start, int end, int step, bool autoFix) const {
    atDim = (atDim % dim + dim) % dim; // 循环索引
    start = (start % _shape[atDim] + _shape[atDim]) % _shape[atDim]; // 循环索引
    int last = end - 1;
    last = (last % _shape[atDim] + _shape[atDim]) % _shape[atDim]; // 循环索引指向最后一个元素
    std::vector<int> newShape = _shape;
    std::vector<int> newStride = _stride;
    int newOffset = _offset;
    if(autoFix && last < start){
        std::swap(start, last);
        last--;
        start++;
    }
    newShape[atDim] = std::max(0, (last - start) / std::abs(step) + 1);
    newStride[atDim] = step * _stride[atDim];
    if(step > 0){
        newOffset = _offset + start * _stride[atDim];
    }
    else if(step < 0){
        // 指向最后一个元素
        newOffset = _offset + last * _stride[atDim];
    }
    else{
        throw std::invalid_argument("Step cannot be 0 in slice operation.");
    }
    YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = newOffset;
    return op;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::slice_(int atDim, int start, int end, int step, bool autoFix){
    atDim = (atDim % dim + dim) % dim; // 循环索引
    start = (start % _shape[atDim] + _shape[atDim]) % _shape[atDim]; // 循环索引
    int last = end - 1;
    last = (last % _shape[atDim] + _shape[atDim]) % _shape[atDim]; // 循环索引指向最后一个元素
    std::vector<int> newShape = _shape;
    std::vector<int> newStride = _stride;
    int newOffset = _offset;
    if(autoFix && last < start){
        std::swap(start, last);
        last--;
        start++;
    }
    newShape[atDim] = std::max(0, (last - start) / std::abs(step) + 1);
    newStride[atDim] = step * _stride[atDim];
    if(step > 0){
        newOffset = _offset + start * _stride[atDim];
    }
    else if(step < 0){
        // 指向最后一个元素
        newOffset = _offset + last * _stride[atDim];
    }
    else{
        throw std::invalid_argument("Step cannot be 0 in slice operation.");
    }
    _offset = newOffset;
    _shape = newShape;
    _stride = newStride;
    return *this;
}

template <typename T, int dim> template <typename... Args>
YTensor<T, dim> YTensor<T, dim>::permute(const Args... args) const{
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    auto newShape = std::vector<int>(dim);
    auto newStride = std::vector<int>(dim);
    for (int i = 0; i < dim; ++i){
        int rotate = (indices[i] % dim + dim) % dim; // 循环索引
        newShape[i] = _shape[rotate];
        newStride[i] = _stride[rotate];
    }
    YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::permute(const std::vector<int> &newOrder) const {
    if (newOrder.size() != dim) {
        throwShapeSizeNotMatch("permute", newOrder.size());
    }
    auto newShape = std::vector<int>(dim);
    auto newStride = std::vector<int>(dim);
    for (int i = 0; i < dim; ++i) {
        int rotate = (newOrder[i] % dim + dim) % dim; // 循环索引
        newShape[i] = _shape[rotate];
        newStride[i] = _stride[rotate];
    }
    YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::permute(const int newOrder[]) const {
    auto newShape = std::vector<int>(dim);
    auto newStride = std::vector<int>(dim);
    for (int i = 0; i < dim; ++i) {
        int rotate = (newOrder[i] % dim + dim) % dim; // 循环索引
        newShape[i] = _shape[rotate];
        newStride[i] = _stride[rotate];
    }
    YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::transpose(int dim0, int dim1) const {
    dim0 = (dim0 % dim + dim) % dim; // 循环索引
    dim1 = (dim1 % dim + dim) % dim; // 循环索引
    if(dim0 == dim1){
        return *this;
    }
    std::vector<int> newShape = _shape;
    std::vector<int> newStride = _stride;
    std::swap(newShape[dim0], newShape[dim1]);
    std::swap(newStride[dim0], newStride[dim1]);
    YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <typename... Args>
std::vector<int> YTensor<T, dim>::autoShape(const Args... shape0) const {
    // constexpr int opShapeSize = sizeof...(shape0);
    std::vector<int> shape({shape0...});
    std::vector<int> op(shape.size());
    int autoDim = -1; // 自动推导的维度
    for (int i = 0; i < static_cast<int>(shape.size()); i++) {
        if (shape[i] < 0) {
            if (autoDim != -1) {
                // switch
                if(autoDim >= static_cast<int>(shape.size())){
                    std::string error = "auto shape cannot infer out of range: found -1 at dimension " + std::to_string(autoDim) +
                        " wh shapeSize " + std::to_string(this->shapeSize());
                    throw std::invalid_argument(error);
                }else{
                    op[i] = shape[autoDim];
                    autoDim = i;
                }
            }else{
                autoDim = i;
            }
        } else {
            op[i] = shape[i];
        }
    }
    if (autoDim != -1) {
        int thisSize = this->size();
        int newSize = 1;
        for (int i = 0; i < dim; ++i) {
            if (i != autoDim) {
                newSize *= op[i];
            }
        }
        if(thisSize % newSize != 0){
            std::string error = "auto shape cannot infer shape: " + std::to_string(thisSize) +
                " is not divisible by " + std::to_string(newSize) + " at dimension ";
            throw std::invalid_argument(error);
        }
        op[autoDim] = thisSize / newSize;
    } else {
        op = _shape;
    }
    return op;
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::autoShape(const std::vector<int> &shape) const {
    std::vector<int> op(shape.size());
    int autoDim = -1; // 自动推导的维度
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            if (autoDim != -1) {
                // switch
                if(autoDim >= static_cast<int>(shape.size())){
                    std::string error = "auto shape cannot infer out of range: found -1 at dimension " + std::to_string(autoDim) +
                        " wh shapeSize " + std::to_string(this->shapeSize());
                    throw std::invalid_argument(error);
                }else{
                    op[autoDim] = _shape[autoDim];
                    autoDim = i;
                }
            }else{
                autoDim = i;
            }
        } else {
            op[i] = shape[i];
        }
    }
    if (autoDim != -1) {
        int thisSize = this->size();
        int newSize = 1;
        for (int i = 0; i < static_cast<int>(op.size()); ++i) {
            if (i != autoDim) {
                newSize *= op[i];
            }
        }
        if(thisSize % newSize != 0){
            std::string error = "auto shape cannot infer shape: " + std::to_string(thisSize) +
                " is not divisible by " + std::to_string(newSize) + " at dimension ";
            throw std::invalid_argument(error);
        }
        op[autoDim] = thisSize / newSize;
    }
    return op;
}

template <typename T, int dim> template <typename... Args>
auto YTensor<T, dim>::view(const Args... newShape) const -> YTensor<T, sizeof...(Args)> {
    constexpr int newdim = sizeof...(newShape);
    static_assert(sizeof...(newShape) == newdim, "Number of arguments must match the dimension");
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    std::vector<int> shape({newShape...});
    shape = autoShape(shape);
    YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <int newdim>
YTensor<T, newdim> YTensor<T, dim>::view(const std::vector<int> &newShape) const {
    if (newShape.size() != newdim){
        throwShapeSizeNotMatch("view", newShape.size());
    }
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    std::vector<int> shape = autoShape(newShape);
    YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
}

template <typename T, int dim> template <int newdim>
YTensor<T, newdim> YTensor<T, dim>::view(const int newShape[]) const {
    std::vector<int> shape = std::vector<int>(newdim);
    for (int i = 0; i < newdim; ++i){
        shape[i] = newShape[i];
    }
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    shape = autoShape(shape);
    YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <typename... Args>
YTensor<T, dim> YTensor<T, dim>::repeat(const Args... times) const {
    static_assert(sizeof...(times) == dim, "Number of arguments must match the dimension");
    std::vector<int> reps = {times...};
    YTensor<T, dim> op;
    op._shape = _shape;
    op._stride = _stride;
    op._data = _data;
    op._offset = _offset;
    for (int i = 0; i < dim; ++i) {
        if(reps[i] <= 1) continue;
        if(_shape[i] != 1){
            throwShapeNotMatch("repeat", reps);
        }
        op._shape[i] = reps[i];
        op._stride[i] = 0;
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::repeat(const std::vector<int> &times) const {
    if (times.size() != dim) {
        throwShapeSizeNotMatch("repeat", times.size());
    }
    YTensor<T, dim> op;
    op._shape = _shape;
    op._stride = _stride;
    op._data = _data;
    op._offset = _offset;
    for (int i = 0; i < dim; ++i) {
        if(times[i] <= 1) continue;
        if(_shape[i] != 1){
            throwShapeNotMatch("repeat", times);
        }
        op._shape[i] = times[i];
        op._stride[i] = 0;
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::repeat(const int times[]) const {
    std::vector<int> reps = std::vector<int>(dim);
    for (int i = 0; i < dim; ++i) {
        reps[i] = times[i];
    }
    YTensor<T, dim> op;
    op._shape = _shape;
    op._stride = _stride;
    op._data = _data;
    op._offset = _offset;
    for (int i = 0; i < dim; ++i) {
        if(reps[i] <= 1) continue;
        if(_shape[i] != 1){
            throwShapeNotMatch("repeat", reps);
        }
        op._shape[i] = reps[i];
        op._stride[i] = 0;
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim + 1> YTensor<T, dim>::unfold(int mdim, int mkernel, int mstride, int mdilation) const{
    if (mkernel <= 0 || mstride <= 0 || mdilation <= 0) {
        throw std::invalid_argument("Invalid kernel/stride/dilation");
    }

    mdim = (mdim % dim + dim) % dim; // 循环索引

    // 计算展开后的形状
    std::vector<int> newShape = _shape;
    int nums = (_shape[mdim] - (mkernel - 1) * mdilation - 1) / mstride + 1;
    newShape.insert(newShape.begin() + mdim, mkernel);
    newShape[mdim + 1] = nums;

    // 计算新的步长
    std::vector<int> newStride = _stride;
    newStride.insert(newStride.begin() + mdim, _stride[mdim] * mdilation);// 核内步长
    newStride[mdim + 1] = _stride[mdim] * mstride; // 窗口移动步长

    YTensor<T, dim + 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;

    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::zeros(const std::vector<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("zeros", shape.size());
    }
    YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::zeros(const int shape[]){
    std::vector<int> shp(dim);
    for (int i = 0; i < dim; ++i){
        shp[i] = shape[i];
    }
    YTensor<T, dim> op(shp);
    std::fill(op.data(), op.data() + op.size(), static_cast<T>(0));
    return op;
}

template <typename T, int dim> template <typename... Args>
YTensor<T, sizeof...(Args)> YTensor<T, dim>::zeros(Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::zeros(const std::initializer_list<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("zeros", shape.size());
    }
    YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::ones(const std::vector<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("ones", shape.size());
    }
    YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::ones(const int shape[]){
    std::vector<int> shp(dim);
    for (int i = 0; i < dim; ++i){
        shp[i] = shape[i];
    }
    YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim> template <typename... Args>
YTensor<T, sizeof...(Args)> YTensor<T, dim>::ones(const Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::ones(const std::initializer_list<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("ones", shape.size());
    }
    YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
typename YTensor<T, dim>::_RandnGenerator YTensor<T, dim>::randn = YTensor<T, dim>::_RandnGenerator(yt::infos::gen);

template <typename T, int dim>
typename YTensor<T, dim>::_RanduGenerator YTensor<T, dim>::randu = YTensor<T, dim>::_RanduGenerator(yt::infos::gen);

template <typename T, int dim>
void YTensor<T, dim>::seed(unsigned int seed) {
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
    yt::infos::gen.seed(seed);
}

template <typename T, int dim> template <typename Func>
YTensor<T, dim>& YTensor<T, dim>::foreach(Func&& func, double flop){
    // **************TODO：按照stride进行重排列，然后创建新的张量后permute到这个视图下，这样能够最大程度的查找contiguous**************
    auto wrappedFunc = [func](T& a, const std::vector<int>& indices) {
        using ResultType = std::invoke_result_t<Func, T &, const std::vector<int> &>;
        if constexpr (std::is_void_v<ResultType>) {
            func(a, indices);
        } else {
            a = func(a, indices);
        }
    };
    // 连续性优化检测：如果最连续视图连续，意味着原张量的范围内可以安全遍历全部。
    auto mcView = mostContinuousView();
    if(mcView.isContiguous()) {
        // fast path
        int max = this->size();
        T* thisPtr = (*this->_data).data() + this->_offset;
        if(max * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int index = 0; index < max; index++) {
                auto coord = this->toCoord(index);
                wrappedFunc(thisPtr[index], coord);
            }
        }
        else {
            // 串行使用里程表法
            std::vector<int> coord(dim, 0);
            #pragma omp simd
            for (int index = 0; index < max; index++) {
                wrappedFunc(thisPtr[index], coord);
                // 计算下一个坐标
                for (int d = dim - 1; d >= 0; d--) {
                    if(coord[d] < _shape[d] - 1) {
                        coord[d]++;
                        break;
                    }
                    coord[d] = 0;
                }
            }
        }
        return *this;
    }
    int thisSize = this->size();
    // 使用遍历法，便于计算坐标
    auto kernel = [this, wrappedFunc](int index) -> void{
        std::vector<int> coord = toCoord(index);
        wrappedFunc(this->at(coord), coord);
    };
    if(thisSize * flop >= yt::infos::minParOps) {
        #pragma omp parallel for simd proc_bind(close)
        for (int i = 0; i < thisSize; ++i) {
            kernel(i);
        }
    }
    else {
        // 串行使用里程表法
        std::vector<int> coord(dim, 0);
        #pragma omp simd
        for (int i = 0; i < thisSize; ++i) {
            wrappedFunc(at(coord), coord);
            for (int d = dim - 1; d >= 0; d--) {
                if(coord[d] < _shape[d] - 1) {
                    coord[d]++;
                    break;
                }
                coord[d] = 0;
            }
        }
    }
    return *this;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::fill(T value){
    // TODO：遍历非连续的轴，然后fill填充剩下的连续的内存块。
    auto mcView = mostContinuousView();
    // int cLast = mcView.isContiguousFrom();// 表示从这个维度开始，后面的维度都是连续的***************************
    if(mcView.isContiguous()){
        std::fill(mcView.data(), mcView.data() + mcView.size(), value);
    }else{
        binaryOpTransformInplace(value, [](T& item, const T& value){
            item = value;
        });
    }
    return *this;
}

template <typename T, int dim>
std::ostream &operator<<(std::ostream &out, const YTensor<T, dim> &tensor){
    out << "[YTensor]:<" << yt::types::getTypeName<T>() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = tensor.shape();
    for (int a = 0; a < static_cast<int> (dims.size() - 1); a++){
        out << dims[a] << ", ";
    }
    out << dims[static_cast<int> (dims.size()) - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;
    if constexpr(yt::concepts::HAVE_OSTREAM<T>){
        // 使用递归函数打印多维数据
        std::function<void(std::vector<int>&, int, int)> printRecursive;
        printRecursive = [&](std::vector<int>& indices, int currentDim, int indent) {
            // 添加缩进
            for (int i = 0; i < indent; i++) {
                out << "  ";
            }

            if (currentDim == dim - 1) {
                // 最后一个维度，打印行向量
                out << "[";
                for (int i = 0; i < dims[currentDim]; i++) {
                    indices[currentDim] = i;
                    out << tensor.at(indices);
                    if (i < dims[currentDim] - 1) {
                        out << " ";
                    }
                }
                out << "]";
                if (dim < 1) {
                    out << std::endl;
                }
            } else {
                // 不是最后一个维度，递归处理
                out << "[" << std::endl;
                for (int i = 0; i < dims[currentDim]; i++) {
                    indices[currentDim] = i;
                    printRecursive(indices, currentDim + 1, indent + 1);
                    if (i < dims[currentDim] - 1) {
                        out << std::endl;
                    }
                }
                out << std::endl;
                for (int i = 0; i < indent; i++) {
                    out << "  ";
                }
                out << "]";
            }
        };

        std::vector<int> indices(dim, 0);
        printRecursive(indices, 0, 0);
    } else {
        out << "[data]: ... (print data not supported)" << std::endl;
    }


    out << std::endl;

    return out;
}
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <omp.h>
#include <string>
#include <typeinfo>

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::broadcastShape(std::vector<int> otherShape) const {
    // 1、填充this与other到相同的维度
    auto thisShape = this->shape();
    int opdim = std::max(dim, static_cast<int>(otherShape.size()));
    int thisLack = opdim - dim;
    int otherLack = opdim - static_cast<int>(otherShape.size());
    if(thisLack > 0){
        thisShape.insert(thisShape.begin(), thisLack, 1);
    }
    if(otherLack > 0){
        otherShape.insert(otherShape.begin(), otherLack, 1);
    }
    // 2、检查维度是否匹配
    std::vector<int> op(opdim);
    for (int i = 0; i < opdim; ++i) {
        if(thisShape[i] != otherShape[i]) {
            if (thisShape[i] == 1) {
                op[i] = otherShape[i];
            } else if (otherShape[i] == 1) {
                op[i] = thisShape[i];
            } else {
                throwShapeNotMatch("broadcastShape", otherShape);
            }
        } else {
            op[i] = thisShape[i];
        }
    }
    return op;
}

template <typename T, int dim>
template<int dim1, typename Func>
YTensor<T, std::max(dim, dim1)> YTensor<T, dim>::binaryOpBroadcast(const YTensor<T, dim1> &other, Func&& func,
    std::string opName, YTensor<T, std::max(dim, dim1)>* result, double flop) const {
    // 1、填充this与other到相同的维度
    auto thisShape = this->shape();
    auto otherShape = other.shape();
    auto thisStride = this->stride_();
    auto otherStride = other.stride_();
    constexpr int opdim = std::max(dim, dim1);
    int thisLack = opdim - dim;
    int otherLack = opdim - dim1;
    bool equalShape = true;
    if(thisLack > 0){
        thisShape.insert(thisShape.begin(), thisLack, 1);
        thisStride.insert(thisStride.begin(), thisLack, 0);
        equalShape = false;
    }
    if(otherLack > 0){
        otherShape.insert(otherShape.begin(), otherLack, 1);
        otherStride.insert(otherStride.begin(), otherLack, 0);
        equalShape = false;
    }
    // 2、检查维度是否匹配
    std::vector<int> opShape(opdim);
    for (int i = 0; i < opdim; ++i) {
        if(thisShape[i] != otherShape[i]) {
            if (thisShape[i] == 1) {
                opShape[i] = otherShape[i];
                thisStride[i] = 0;
            } else if (otherShape[i] == 1) {
                opShape[i] = thisShape[i];
                otherStride[i] = 0;
            } else {
                throwShapeNotMatch(opName, otherShape);
            }
            equalShape = false;
        } else {
            opShape[i] = thisShape[i];
        }
    }
    YTensor<T, opdim> op;
    if(result != nullptr) {
        if(!result->shapeMatch(opShape)){
            result->reserve(opShape);
        }
        op.shallowCopyFrom(*result);
    } else {
        op.reserve(opShape);
    }

    if constexpr (std::is_invocable_v<Func, const T&, const T&, T&>) {
        if(equalShape && this->isContiguous() && other.isContiguous()) {
            // fast path
            int max = op.size();
            T* thisPtr = (*this->_data).data() + this->_offset;
            T* otherPtr = (*other._data).data() + other._offset;
            if(max * flop >= yt::infos::minParOps) {
                #pragma omp parallel for simd proc_bind(close)
                for (int index = 0; index < max; index++) {
                    func(thisPtr[index], otherPtr[index], op.atData(index));
                }
            }
            else {
                #pragma omp simd
                for (int index = 0; index < max; index++) {
                    func(thisPtr[index], otherPtr[index], op.atData(index));
                }
            }
            return op;
        }
        // 3、定义并行计算的哈希函数
        auto logicStride = op.stride();
        auto kernel = [&thisStride, &otherStride, this, &other, &op, func, &logicStride](int index) {
            int thisIndex = 0, otherIndex = 0;
            #pragma omp simd reduction(+:thisIndex, otherIndex)
            for (int i = 0; i < op.shapeSize(); ++i) {
                int posi = (index / logicStride[i]) % op._shape[i];
                thisIndex += posi * thisStride[i];
                otherIndex += posi * otherStride[i];
            }
            return func(this->atData_(thisIndex), other.atData_(otherIndex), (*op._data)[index]);
        };

        // 4、并行计算
        int max = op.size();
        if(max * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int index = 0; index < max; index++) {
                kernel(index);
            }
        }
        else {
            #pragma omp simd
            for (int index = 0; index < max; index++) {
                kernel(index);
            }
        }
    }
    else{
        if(equalShape && this->isContiguous() && other.isContiguous()) {
            // fast path
            int max = op.size();
            T* opPtr = (*op._data).data();
            T* thisPtr = (*this->_data).data() + this->_offset;
            T* otherPtr = (*other._data).data() + other._offset;
            if(max * flop >= yt::infos::minParOps) {
                #pragma omp parallel for simd proc_bind(close)
                for (int index = 0; index < max; index++) {
                    opPtr[index] = func(thisPtr[index], otherPtr[index]);
                }
            }
            else {
                #pragma omp simd
                for (int index = 0; index < max; index++) {
                    opPtr[index] = func(thisPtr[index], otherPtr[index]);
                }
            }
            return op;
        }
        // 3、定义并行计算的哈希函数
        auto logicStride = op.stride();
        auto kernel = [&thisStride, &otherStride, this, &other, &op, func, &logicStride](int index) {
            int thisIndex = 0, otherIndex = 0;
            #pragma omp simd reduction(+:thisIndex, otherIndex)
            for (int i = 0; i < op.shapeSize(); ++i) {
                int posi = (index / logicStride[i]) % op._shape[i];
                thisIndex += posi * thisStride[i];
                otherIndex += posi * otherStride[i];
            }
            return func(this->atData_(thisIndex), other.atData_(otherIndex));
        };

        // 4、并行计算
        int max = op.size();
        if(max * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int index = 0; index < max; index++) {
                (*op._data)[index] = kernel(index);
            }
        }
        else {
            #pragma omp simd
            for (int index = 0; index < max; index++) {
                (*op._data)[index] = kernel(index);
            }
        }
    }
    return op;
}

template <typename T, int dim>
template <int dim1, typename Func>
YTensor<T, dim> &YTensor<T, dim>::binaryOpBroadcastInplace(const YTensor<T, dim1> &other, Func &&func, std::string opName, double flop){
    // 1、填充this与other到相同的维度
    auto thisShape = this->shape();
    auto otherShape = other.shape();
    auto otherStride = other.stride_();
    constexpr int thisLack = std::max(dim, dim1) - dim;
    constexpr int otherLack = std::max(dim, dim1) - dim1;
    bool equalShape = true;
    if(thisLack > 0){
        throwShapeNotMatch(opName, thisShape);
        equalShape = false;
    }
    if(otherLack > 0){
        otherShape.insert(otherShape.begin(), otherLack, 1);
        otherStride.insert(otherStride.begin(), otherLack, 0);
        equalShape = false;
    }

    // 2、检查维度是否匹配
    for (int i = 0; i < dim; ++i) {
        if(thisShape[i] != otherShape[i]) {
            if (otherShape[i] == 1) {
                otherStride[i] = 0;
            } else {
                throwShapeNotMatch(opName, otherShape);
            }
            equalShape = false;
        }
    }

    if(equalShape && this->isContiguous() && other.isContiguous()) {
        // fast path
        int max = this->size();
        T* thisPtr = (*this->_data).data() + this->_offset;
        T* otherPtr = (*other._data).data() + other._offset;
        if(max * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int index = 0; index < max; index++) {
                func(thisPtr[index], otherPtr[index]);
            }
        }
        else {
            #pragma omp simd
            for (int index = 0; index < max; index++) {
                func(thisPtr[index], otherPtr[index]);
            }
        }
        return *this;
    }

    // 3、定义并行计算的哈希函数
    auto logicStride = this->stride();
    auto kernel = [&otherStride, &logicStride, this, &other, func](int index) -> void {
        int thisIndex = 0, otherIndex = 0;
        #pragma omp simd reduction(+:thisIndex, otherIndex)
        for (int i = 0; i < dim; i++) {
            int posi = (index / logicStride[i]) % _shape[i];
            thisIndex += posi * _stride[i];
            otherIndex += posi * otherStride[i];
        }
        func(this->atData_(thisIndex), other.atData_(otherIndex));
        return;
    };

    // 3、并行计算
    int max = this->size();
    if(max * flop >= yt::infos::minParOps) {
        #pragma omp parallel for simd proc_bind(close)
        for (int index = 0; index < max; index++) {
            kernel(index);
        }
    }
    else {
        #pragma omp simd
        for (int index = 0; index < max; index++) {
            kernel(index);
        }
    }
    return *this;
}

template<typename T, int dim> template<typename Func>
YTensor<T, dim> YTensor<T, dim>::binaryOpTransform(const T& other, Func&& func,  YTensor<T, dim>* result, double flop) const{
    auto totalSize = this->size();
    YTensor<T, dim> op;
    if(result != nullptr){
        if(!result->shapeMatch(this->shape())){
            result->reserve(this->shape());
        }
        op = *result;
    } else {
        op.reserve(this->shape());
    }

    // 连续性优化检测
    auto mcView = this->mostContinuousView();

    int thisSize = this->size();
    if constexpr (std::is_invocable_v<Func, const T&, const T&, T&>){
        if (mcView.isContiguous()) {
            // fast path
            int max = mcView.size();
            T* thisPtr = (*mcView._data).data() + mcView._offset;
            T* opPtr = (*op._data).data() + op._offset;
            if(max * flop >= yt::infos::minParOps) {
                #pragma omp parallel for simd proc_bind(close)
                for (int index = 0; index < max; index++) {
                    func(thisPtr[index], other, opPtr[index]);
                }
            }
            else {
                #pragma omp simd
                for (int index = 0; index < max; index++) {
                    func(thisPtr[index], other, opPtr[index]);
                }
            }
            return op;
        }
        // 创建核函数
        auto logicStride = this->stride();
        auto kernel = [this, &other, func, &op, &logicStride](int index){
            int thisIndex = 0;
            #pragma omp simd reduction(+:thisIndex)
            for (int i = 0; i < dim; i++) {
                int posi = (index / logicStride[i]) % _shape[i];
                thisIndex += posi * _stride[i];
            }
            return func(this->atData_(thisIndex), other, op.atData_(index));
        };

        // 并行计算
        if(thisSize * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int i = 0; i < thisSize; ++i) {
                kernel(i);
            }
        }
        else {
            #pragma omp simd
            for (int i = 0; i < thisSize; ++i) {
                kernel(i);
            }
        }
    }else{
        if(mcView.isContiguous()) {
            // fast path
            int max = mcView.size();
            T* thisPtr = (*mcView._data).data() + mcView._offset;
            T* opPtr = (*op._data).data() + op._offset;
            if(max * flop >= yt::infos::minParOps) {
                #pragma omp parallel for simd proc_bind(close)
                for (int index = 0; index < max; index++) {
                    opPtr[index] = func(thisPtr[index], other);
                }
            }
            else {
                #pragma omp simd
                for (int index = 0; index < max; index++) {
                    opPtr[index] = func(thisPtr[index], other);
                }
            }
            return op;
        }
        // 创建核函数
        auto logicStride = this->stride();
        auto kernel = [this, &other, func, &logicStride](int index){
            int thisIndex = 0;
            #pragma omp simd reduction(+:thisIndex)
            for (int i = 0; i < dim; i++) {
                int posi = (index / logicStride[i]) % _shape[i];
                thisIndex += posi * _stride[i];
            }
            return func(this->atData_(thisIndex), other);
        };

        // 并行计算
        if(thisSize * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int i = 0; i < thisSize; ++i) {
                op.atData_(i) = kernel(i);
            }
        }
        else {
            #pragma omp simd
            for (int i = 0; i < thisSize; ++i) {
                op.atData_(i) = kernel(i);
            }
        }
    }
    return op;
}

template<typename T, int dim> template<typename Func>
YTensor<T, dim>& YTensor<T, dim>::binaryOpTransformInplace(const T& other, Func&& func, double flop){
    auto wrappedFunc = [func](T& a, const T& b) {
        using ResultType = std::invoke_result_t<Func, T &, const T &>;
        if constexpr (std::is_void_v<ResultType>) {
            func(a, b);
        } else {
            a = func(a, b);
        }
    };
    // 连续性优化检测
    YTensor<T, dim> mcView = this->mostContinuousView();

    if(mcView.isContiguous()) {
        // fast path
        int max = mcView.size();
        T* thisPtr = (*mcView._data).data() + mcView._offset;
        if(max * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int index = 0; index < max; index++) {
                wrappedFunc(thisPtr[index], other);
            }
        }
        else {
            #pragma omp simd
            for (int index = 0; index < max; index++) {
                wrappedFunc(thisPtr[index], other);
            }
        }
        return *this;
    }
    int thisSize = this->size();
    int oriSize = this->_data->size();
    if(oriSize / thisSize > MAX_SUBELEMENT_RATIO){
        // 使用遍历法
        auto logicStride = stride();
        auto kernel = [this, &other, &logicStride, wrappedFunc](int index) -> void{
            int thisIndex = 0;
            #pragma omp simd reduction(+:thisIndex)
            for (int i = 0; i < dim; i++) {
                int posi = (index / logicStride[i]) % _shape[i];
                thisIndex += posi * _stride[i];
            }
            wrappedFunc(this->atData_(thisIndex), other);
            return;
        };
        if(thisSize * flop >= yt::infos::minParOps) {
            #pragma omp parallel for simd proc_bind(close)
            for (int i = 0; i < thisSize; ++i) {
                kernel(i);
            }
        }
        else {
            // 串行使用里程表法依然较慢
            #pragma omp simd
            for (int i = 0; i < thisSize; ++i) {
                kernel(i);
            }
        }
    }
    else {
        // 使用布尔掩码
        T* parPtr = (*(this->_data)).data();
        #pragma omp simd
        for(int a = 0; a < oriSize; a++) {
            int delta = a - _offset;// 相对于基地址的偏移量
            // 算法：delta需要可以被stride在shape范围内表示
            // 使用mcView从大stride到小遍历。
            // 内部无需并行，整数计算simd价值不高
            for(int b = 0; b < dim; b++){
                if(mcView._shape[b] == 1){
                    if (mcView._shape[b] != 1)
                        break;// unfold
                    // else shape = 1(不影响)
                } else if(mcView._stride[b] != 0){
                    int step = delta / mcView._stride[b];// 负数向0取整，因此不影响
                    if(step < 0 || step >= mcView._shape[b]){
                        // 越界
                        break;
                    }
                    delta -= step * mcView._stride[b];
                }
            }
            if(!delta){
                wrappedFunc(*(parPtr + a), other);
            }
        }
    }
    return *this;
}

// 运算符生成规则
#define YT_YTENSOR_OPERATOR(OP, ENABLE_IF_T)                                                          \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    YTensor<T, std::max(dim, dim1)>                                                                   \
        YTensor<T, dim>::operator OP(const YTensor<T, dim1>& other) const {                           \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return binaryOpBroadcast(                                                                 \
                other, [](const T& a, const T& b) {                                                   \
                    return a OP b;                                                                    \
                },                                                                                    \
                #OP);                                                                                 \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    YTensor<T, std::max(dim, dim1)>& YTensor<T, dim>::operator OP##=(const YTensor<T, dim1>& other) { \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return binaryOpBroadcastInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a OP## = b;                                                                \
                },                                                                                    \
                "+=");                                                                                \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return binaryOpBroadcastInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a = a OP b;                                                                \
                },                                                                                    \
                "+=");                                                                                \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                        \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    YTensor<T, dim> YTensor<T, dim>::operator OP(const T& other) const {                              \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return binaryOpTransform(                                                                 \
                other, [](const T& a, const T& b) {                                                   \
                    return a OP b;                                                                    \
                },                                                                                    \
                nullptr);                                                                             \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    YTensor<T, dim>& YTensor<T, dim>::operator OP##=(const T& other) {                                \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return binaryOpTransformInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a OP## = b;                                                                \
                });                                                                                   \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return binaryOpTransformInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a = a OP b;                                                                \
                });                                                                                   \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                        \
        }                                                                                             \
    }


YT_YTENSOR_OPERATOR(+, yt::concepts::HAVE_ADD)
YT_YTENSOR_OPERATOR(-, yt::concepts::HAVE_SUB)
YT_YTENSOR_OPERATOR(*, yt::concepts::HAVE_MUL)
YT_YTENSOR_OPERATOR(/, yt::concepts::HAVE_DIV)
// %有特殊处理
YT_YTENSOR_OPERATOR(&, yt::concepts::HAVE_AND)
YT_YTENSOR_OPERATOR(|, yt::concepts::HAVE_MOD)
YT_YTENSOR_OPERATOR(^, yt::concepts::HAVE_XOR)

#undef YT_YTENSOR_OPERATOR

template <typename T, int dim> template<int dim1>
YTensor<T, std::max(dim, dim1)> YTensor<T, dim>::operator%(const YTensor<T, dim1>& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return binaryOpBroadcast(other, [](const T& a, const T& b) {
            return a % b;
        }, "%");
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpBroadcast(other, [](const T& a, const T& b) {
            return std::fmod(a, b);
        }, "%");
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, std::max(dim, dim1)>& YTensor<T, dim>::operator%=(const YTensor<T, dim1>& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a %= b;
        }, "%=");
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a = a % b;
        }, "%=");
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a = fmod(a, b);
        }, "%=");
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator%(const T& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return binaryOpTransform(other, [](const T& a, const T& b) {
            return a % b;
        });
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpTransform(other, [](const T& a, const T& b) {
            return std::fmod(a, b);
        });
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::operator%=(const T& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a %= b;
        });
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a = a % b;
        });
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a = fmod(a, b);
        });
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
YTensor<YTensor<T, 2>, std::max(1, dim - 2)> YTensor<T, dim>::matView() const {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    if constexpr (dim == 1){
        YTensor<T, 2> mat;
        mat._shape = std::vector<int>({1, this->_shape[0]});
        mat._stride = std::vector<int>({0, this->_stride[0]});
        mat._offset = this->_offset;
        mat._data = this->_data;
        YTensor<YTensor<T, 2>, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._data = std::make_shared<std::vector<YTensor<T, 2>>>(1, mat);
        return op;
    }else if constexpr (dim == 2){
        YTensor<YTensor<T, 2>, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._data = std::make_shared<std::vector<YTensor<T, 2>>>(1, *this);
        return op;
    }else{
        auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
        YTensor<YTensor<T, 2>, std::max(1, dim - 2)> op;
        op._shape = newShape;
        op._stride = op.stride();
        op._offset = 0;
        int batchSize = op.size();
        op._data = std::make_shared<std::vector<YTensor<T, 2>>>(batchSize);
        YTensor<T, 2>* dataptr = op._data->data();

        if(batchSize * 5. > yt::infos::minParOps){
            #pragma omp parallel for simd  proc_bind(close)
            for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
                auto coord = op.toCoord(batchIdx);
                YTensor<T, 2> mat;
                mat._shape = {this->_shape[dim-2], this->_shape[dim-1]};
                mat._stride = {this->_stride[dim-2], this->_stride[dim-1]};
                mat._offset = this->offset(coord);
                mat._data = this->_data;
                dataptr[batchIdx] = mat;
            }
        }else{
            #pragma omp simd
            for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
                auto coord = op.toCoord(batchIdx);
                YTensor<T, 2> mat;
                mat._shape = {this->_shape[dim-2], this->_shape[dim-1]};
                mat._stride = {this->_stride[dim-2], this->_stride[dim-1]};
                mat._offset = this->offset(coord);
                mat._data = this->_data;
                dataptr[batchIdx] = mat;
            }
        }

        return op;
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul(const YTensor<T, dim1>& other)const{
    static_assert(yt::concepts::HAVE_ADD<T> && yt::concepts::HAVE_MUL<T>, "Type must have add and mul in matmul");
    static_assert(dim >= 1 && dim1 >= 1, "matmul only support dim >= 1");
    int lw = this->shape(-1);
    int rw = other.shape(-2);
    if(lw != rw){
        throwShapeNotMatch("matmul", other.shape());
    }
    // 如果是数字，就用 eigen
    if constexpr (std::is_arithmetic_v<T> && YT_USE_EIGEN) {
        return matmul_eigen_backend(other);
        // return matmul_zero_backend(other);
    }else{
        return matmul_zero_backend(other);
    }
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::sum(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    size_t max = op.size();
    if (max * _shape[axis] > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                sum += this->at(subCoord);
            }
            op.atData_(i) = sum;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                sum += this->at(subCoord);
            }
            op.atData_(i) = sum;
        }
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::sum(std::vector<int> axis) const requires (dim > 1) {
    for (auto& ax : axis) {
        ax = (ax % dim + dim) % dim;
    }
    auto newShape = this->shape();
    int targetSize = 1;
    for (auto& ax : axis) {
        newShape[ax] = 1;
        targetSize *= shape(ax);
    }
    // 偏移量列表，使用里程表法构建
    std::vector<int> offsets(targetSize);
    std::vector<int> records(axis.size(), 0);
    int offset0 = 0;
    for (int i = 0; i < targetSize; i++) {
        offsets[i] = offset0;
        // 更新里程计
        for (int j = axis.size() - 1; j >= 0; j--) {
            if (records[j] < _shape[axis[j]] - 1) {
                records[j]++;
                break;
            } else {
                records[j] = 0;
            }
        }
        // 更新offset
        offset0 = 0;
        for (int j = axis.size() - 1; j >= 0; j--) {
            offset0 += records[j] * _stride[axis[j]];
        }
    }
    // 现在，offsets已经构建完成

    YTensor<T, dim> op(newShape);
    size_t max = op.size();
    if (max > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < offsets.size(); j++) {
                sum += this->atData_(base + offsets[j]);
            }
            op.atData_(i) = sum;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < offsets.size(); j++) {
                sum += this->atData_(base + offsets[j]);
            }
            op.atData_(i) = sum;
        }
    }
    return op;
}

template <typename T, int dim>
T YTensor<T, dim>::sum(int) const requires (dim == 1) {
    T sum = 0;
    int max = this->size();
    if (max * 1. > yt::infos::minParOps){
        #pragma omp parallel for simd reduction(+:sum)  proc_bind(close)
        for (int i = 0; i < max; i++) {
            sum += this->at(i);
        }
    }else{
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < max; i++) {
            sum += this->at(i);
        }
    }
    return sum;
}

template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> YTensor<T, dim>::max(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    YTensor<int, dim> opi(newShape);
    size_t max = op.size();
    if (max * _shape[axis] > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T maxer = this->at(coord);
            int maxerIndex = 0;
            #pragma omp simd
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                const T& value = this->at(subCoord);
                if (value > maxer) {
                    maxer = value;
                    maxerIndex = j;
                }
            }
            op.atData_(i) = maxer;
            opi.atData_(i) = maxerIndex;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T maxer = this->at(coord);
            int maxerIndex = 0;
            #pragma omp simd
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                const T& value = this->at(subCoord);
                if (value > maxer) {
                    maxer = value;
                    maxerIndex = j;
                }
            }
            op.atData_(i) = maxer;
            opi.atData_(i) = maxerIndex;
        }
    }
    return std::make_pair(op, opi);
}

template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> YTensor<T, dim>::max(std::vector<int> axis) const requires (dim > 1) {
    for (auto& ax : axis) {
        ax = (ax % dim + dim) % dim;
    }
    auto newShape = this->shape();
    int targetSize = 1;
    for (auto& ax : axis) {
        newShape[ax] = 1;
        targetSize *= shape(ax);
    }
    // 偏移量列表，使用里程表法构建
    std::vector<int> offsets(targetSize);
    std::vector<int> records(axis.size(), 0);
    int offset0 = 0;
    for (int i = 0; i < targetSize; i++) {
        offsets[i] = offset0;
        // 更新里程计
        for (int j = axis.size() - 1; j >= 0; j--) {
            if (records[j] < _shape[axis[j]] - 1) {
                records[j]++;
                break;
            } else {
                records[j] = 0;
            }
        }
        // 更新offset
        offset0 = 0;
        for (int j = axis.size() - 1; j >= 0; j--) {
            offset0 += records[j] * _stride[axis[j]];
        }
    }
    // 现在，offsets已经构建完成

    YTensor<T, dim> op(newShape);
    YTensor<int, dim> opi(newShape);
    size_t max = op.size();
    if (max > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T maxer = this->at(coord);
            int maxerIndex = 0;
            #pragma omp simd
            for (size_t j = 0; j < offsets.size(); j++) {
                const T& value = this->atData_(base + offsets[j]);
                if (value > maxer) {
                    maxer = value;
                    maxerIndex = j;
                }
            }
            op.atData_(i) = maxer;
            opi.atData_(i) = maxerIndex;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T maxer = this->at(coord);
            int maxerIndex = 0;
            #pragma omp simd
            for (size_t j = 0; j < offsets.size(); j++) {
                const T& value = this->atData_(base + offsets[j]);
                if (value > maxer) {
                    maxer = value;
                    maxerIndex = j;
                }
            }
            op.atData_(i) = maxer;
            opi.atData_(i) = maxerIndex;
        }
    }
    return std::make_pair(op, opi);
}

template<typename T, int dim>
std::pair<T, int> YTensor<T, dim>::max(int)const requires (dim == 1) {
    T maxer = this->at(0);
    int maxerIndex = 0;
    int max = this->size();
    if (max * 1. > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (int i = 0; i < max; i++) {
            const T& value = this->at(i);
            if (value > maxer) {
                maxer = value;
                maxerIndex = i;
            }
        }
    }else{
        #pragma omp simd
        for (int i = 0; i < max; i++) {
            const T& value = this->at(i);
            if (value > maxer) {
                maxer = value;
                maxerIndex = i;
            }
        }
    }
    return std::make_pair(maxer, maxerIndex);
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul_zero_backend(const YTensor<T, dim1>& other) const{
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    int aw = this->shape(-1);
    // int bh = other.shape(-2);
    int bw = other.shape(-1);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        // 如果是二维矩阵，直接返回
        opShape = {ah, bw};
    } else {
        opShape = thisMatView.broadcastShape(otherMatView.shape());
        opShape.push_back(ah); opShape.push_back(bw);
    }
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    auto mulop = thisMatView.binaryOpBroadcast(otherMatView, [&ah, &aw, &bw](const YTensor<T, 2>& a, const YTensor<T, 2>& b, YTensor<T, 2>& o) {
        #pragma omp parallel for simd collapse(2) proc_bind(close)
        // #pragma omp simd
        for (int y=0; y<ah; ++y) {
            for (int x=0; x<bw; ++x) {
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int k=0; k<aw; ++k) {
                    sum = sum + a.at(y, k) * b.at(k, x);
                }
                o.at(y, x) = sum;
            }
        }
        return;
    }, "matmul_zero_backend", &opMatView, yt::infos::flopMatmul(ah, aw, bw));
    return op;
}

///////////// Eigen support ////////////////

#if YT_USE_EIGEN
template <typename T, int dim>
YTensor<typename YTensor<T, dim>::EigenMatrixMap, std::max(1, dim - 2)> YTensor<T, dim>::matViewEigen() const requires (dim > 2) {
    auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
    YTensor<EigenMatrixMap, std::max(1, dim - 2)> op;
    op._shape = newShape;
    op._stride = op.stride();
    op._offset = 0;
    int batchSize = op.size();// ************************* 这里要修复，串行emplace back***********
    op._data = std::make_shared<std::vector<EigenMatrixMap>>();
    auto& opDataVec = op.dataVector();
    opDataVec.reserve(batchSize);

    if(batchSize * 5. > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
            auto coord = op.toCoord(batchIdx);
            Eigen::Stride<-1, -1> mstride(this->_stride[dim - 2], this->_stride[dim - 1]);
            T *matDataPtr = this->_data.get()->data() + this->offset(coord);
            opDataVec.emplace_back(matDataPtr, this->_shape[dim - 2], this->_shape[dim - 1], mstride);
        }
    }else{
        #pragma omp simd
        for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
            auto coord = op.toCoord(batchIdx);
            Eigen::Stride<-1, -1> mstride(this->_stride[dim - 2], this->_stride[dim - 1]);
            T *matDataPtr = this->_data.get()->data() + this->offset(coord);
            opDataVec.emplace_back(matDataPtr, this->_shape[dim - 2], this->_shape[dim - 1], mstride);
        }
    }
    return op;
}

template <typename T, int dim> typename
YTensor<T, dim>::EigenMatrixMap YTensor<T, dim>::matViewEigen() const requires (dim <= 2) {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    if constexpr (dim == 1){
        Eigen::Stride<-1, -1> mstride(0, this->_stride[0]);
        T* dataptr = _data.get()->data() + this->_offset;
        EigenMatrixMap op(dataptr, this->_shape[0], 1, mstride);
        return op;
    }else{
        Eigen::Stride<-1, -1> mstride(this->_stride[0], this->_stride[1]);
        T* dataptr = _data.get()->data() + this->_offset;
        EigenMatrixMap op(dataptr, this->_shape[0], this->_shape[1], mstride);
        return op;
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul_eigen_backend(const YTensor<T, dim1>& other) const{
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    int aw = this->shape(-1);
    // int bh = other.shape(-2);
    int bw = other.shape(-1);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        // 如果是二维矩阵，直接返回
        opShape = {ah, bw};
    } else {
        opShape = thisMatView.broadcastShape(otherMatView.shape());
        opShape.push_back(ah); opShape.push_back(bw);
    }
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    auto mulop = thisMatView.binaryOpBroadcast(otherMatView, [](const YTensor<T, 2>& a, const YTensor<T, 2>& b, YTensor<T, 2>& o) {
        auto mapA = a.matViewEigen();
        auto mapB = b.matViewEigen();
        auto mapO = o.matViewEigen();
        mapO.noalias() = mapA * mapB;
        return;
    }, "matmul_eigen_backend", &opMatView, yt::infos::flopMatmul(ah, aw, bw));
    return op;
}
#endif //YT_USE_EIGEN

//////////// external /////////////
//////////////// hpp file "ytensor_math.hpp" //////////////////
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * @brief 基于YTensor的常用函数库，对YTensor的常用操作进行封装。
 */
namespace yt::function{
    template<typename T, int dim0, int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> matmul(const YTensor<T, dim0>& a, const YTensor<T, dim1>& b);

    template<typename T, int dim>
    YTensor<T, dim> relu(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim>& relu_(YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> exp(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> sigmoid(const YTensor<T, dim>& x, int order = 0);

    template<typename T, int dim>
    YTensor<T, dim> softmax(const YTensor<T, dim>& x, int axis = -1);

    enum struct sdpaBackend{
        MATH
    };

    template<typename T, int dim>
    YTensor<T, dim> scaledDotProductAttention(
        YTensor<T, dim>& query,
        YTensor<T, dim>& key,
        YTensor<T, dim>& value,
        T scale = static_cast<T>(0.0),
        YTensor<T, dim>* mask = nullptr,
        sdpaBackend backend = sdpaBackend::MATH
    );

    inline void throwNotSupport(const std::string& funcName, const std::string& caseDiscription);
}// namespace yt::function


template <typename T,int dim>
class YTensor;

template <typename T, int dim0, int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> yt::function::matmul(const YTensor<T, dim0>& a, const YTensor<T, dim1>& b) {
    return a.matmul(b);
}

template <typename T, int dim>
YTensor<T, dim> yt::function::relu(const YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    YTensor<T, dim> op;
    if(order == 0){
        op = x.binaryOpTransform(0, [](const T& a, const T&) {
            return std::max(a, static_cast<T>(0));
        });
    }
    else if(order == 1){
        op = x.binaryOpTransform(0, [](const T& a, const T&) {
            return static_cast<T>(a > 0);
        });
    }
    else if(order > 1){
        // op = YTensor<T, dim>::zeros(x.shape());
        throwNotSupport("yt::function::relu", "order > 1");
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        op = x.binaryOpTransform(0, [&pow, &fact](const T& a, const T&) {
            if(a > 0){
                return std::pow(a, pow) / static_cast<T>(fact);
            }
            else{
                return static_cast<T>(0);
            }
        });
    }
}

template <typename T, int dim>
YTensor<T, dim>& yt::function::relu_(YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    if(order == 0){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = std::max(a, static_cast<T>(0));
        });
    }
    else if(order == 1){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = static_cast<T>(a > 0);
        });
    }
    else if(order > 1){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = static_cast<T>(0); // 这里可以根据需要修改
        });
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        x.binaryOpTransformInplace(0, [&pow, &fact](T& a, const T&) {
            if(a > 0){
                a = std::pow(a, pow) / static_cast<T>(fact);
            }
            else{
                a = static_cast<T>(0);
            }
        });
    }
    return x;
}

template <typename T, int dim>
YTensor<T, dim> yt::function::exp(const YTensor<T, dim>& x, int) {
    return x.unaryOpTransform(0, [](const T& a, const T&){
        return std::exp(a);
    });
}

template <typename T, int dim>
YTensor<T, dim> yt::function::sigmoid(const YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::sigmoid()");
    if(order == 0){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-a));
        });
    }
    else if(order == 1){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            T sig = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-a));
            return sig * (static_cast<T>(1) - sig);
        });
    }
    else if(order == -1){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            return std::log(static_cast<T>(1) + std::exp(a));
        });
    }
    else{
        throwNotSupport("yt::function::sigmoid", "order != 0, 1, -1");
    }
}

template<typename T, int dim>
YTensor<T, dim> yt::function::softmax(const YTensor<T, dim>& x, int axis) {
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();
    YTensor<T, dim> output(shape);

    // 快速路径：连续张量 - 使用优化实现（Flash Attention 风格）
    if (x.isContiguous() && output.isContiguous()) {
        int64_t dim_size = shape[axis];

        // 计算外层大小和内层大小
        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }

        int64_t inner_size = 1;
        for (int i = axis + 1; i < dim; ++i) {
            inner_size *= shape[i];
        }

        int64_t dim_stride = inner_size;
        int64_t outer_stride = dim_size * dim_stride;

        const T* input_data_base = x.data();
        T* output_data_base = output.data();

        // 针对连续数据的优化循环
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                const T* input_data = input_data_base + outer_idx * outer_stride + inner_idx;
                T* output_data = output_data_base + outer_idx * outer_stride + inner_idx;

                // 步骤 1: 获取最大值（数值稳定性）
                T max_val = input_data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, input_data[d * dim_stride]);
                }

                // 步骤 2: 计算 exp(x - max) 并累加和（融合操作）
                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(input_data[d * dim_stride] - max_val);
                    output_data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }

                // 步骤 3: 用和归一化
                for (int64_t d = 0; d < dim_size; ++d) {
                    output_data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        // 通用路径：支持非连续张量
        // 使用迭代器方式，融合 max、exp、sum 操作

        // 构建迭代形状（除了归约轴之外的所有维度）
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) {
            total_iterations *= s;
        }

        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            // 将平铺索引转换为多维索引（不包括 axis 维度）
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = iter_shape.size() - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            // 插入 axis 维度，转换为完整索引
            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0); // 占位符
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 融合传递：max、exp 和 sum 在一次循环中完成
            T max_val = std::numeric_limits<T>::lowest();

            // 第一遍：找最大值
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            // 第二遍：计算 exp(x - max) 并求和
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                output.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            // 第三遍：归一化
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) /= sum_exp;
            }
        }
    }

    return output;
}

template<typename T, int dim>
YTensor<T, dim> yt::function::scaledDotProductAttention(
    YTensor<T, dim>& query,// [b, n0, c0]
    YTensor<T, dim>& key,// [b, n1, c0]
    YTensor<T, dim>& value,// [b, n1, c1]
    T scale,
    YTensor<T, dim>* mask,
    sdpaBackend backend
) {
    if(static_cast<T>(0.0) == scale){
        // auto
        scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(query.shape(-1)));
    }
    if(backend == sdpaBackend::MATH){
        auto score = yt::function::matmul(query, key.transpose());// [b, n0, n1]
        score.binaryOpTransformInplace(scale, [](T& a, const T& b) {
            a *= b; // scale
        });
        score = yt::function::softmax(score, -1);// [b, n0, n1]
        if(mask != nullptr){
            if(mask->shapeSize() != 2 || mask->shape(-1) != score.shape(-1) || mask->shape(-2) != score.shape(-2)){
                throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
            }
            score *= *mask;
        }
        auto op = yt::function::matmul(score, value);// [b, n0, c1]
        return op;
    }
    else{
        throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
        return YTensor<T, dim>();
    }
}


void yt::function::throwNotSupport(const std::string& funcName, const std::string& caseDiscription) {
    throw std::invalid_argument("Function " + funcName + " is not supported for case: " + caseDiscription);
}
#pragma once
/***************
* @file: ytensor_io.hpp
* @brief: YTensor 类的文件输入/输出功能。
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

/*
文件结构：
二进制数据均使用char格式
string = {
4   [uint32_t]  文本长度，单位为字节，不包含结尾的'\0'（cpp规则）
... [char]      utf8编码的文本内容
}

array[T] = {
8   [uint64_t]  数组长度，单位为元素个数
... [T]         数组内容，T为元素类型
}

file:{
8   [char]                  文件类型校验码，固定为"YTENSORF"
1   [uint8_t]               版本号，当前版本为0
4   [uint32_t]              matadata长度，单位为字节
    matadata:{
...     [string]            json格式的元数据。使用utf8编码
    }
    data:[
        tensor:{
...         [string]        张量名称
...         [string]        张量数据类型名称
4           [int32_t]       张量数据类型的字节大小
...         [string]        张量类型，如dense, sparse等，决定了data的排布方式
...         [array[int32_t]]张量形状
...         [string]        张量压缩算法。可选项为：""、"zlib"
...         [array[byte]]   张量数据，按照张量类型进行排布
        },
        ...
    ]
    index:{
        [[uint64_t]*n]      张量在文件中的偏移位置
4       [uint32_t]          张量个数（即索引个数）
    }
}
*/

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <span>

#include <unordered_map>
#include <cstring>
#include <zlib.h>
// #include <zstd.h>

// 为避免循环包含，这里前向声明YTensor
template <typename T, int dim> class YTensor;


namespace yt::io{
    /// @brief 文件头标识
    using yt::infos::YTENSOR_FILE_MAGIC;

    /// @brief 文件版本
    using yt::infos::YTENSOR_FILE_VERSION;

    /// @brief 压缩级别，在zlib压缩方式中使用
    static int8_t compressLevel = Z_DEFAULT_COMPRESSION;

    /// @brief 是否输出警告信息
    static bool verbose = false; // 是否输出警告信息

    static constexpr int Closed = 0;
    static constexpr int Read = 1;
    static constexpr int Write = 2;
    static constexpr int Append = 3;

    /// @brief 检查压缩方法
    /// @enum "": 不压缩
    /// @enum "zlib": 使用zlib压缩，compressLevel控制压缩级别
    /// @enum "zlibfloat": 优化后的浮点数专用的压缩方法
    static std::string compressMethod = "";

    /// @brief 检查压缩方法
    /// @param method 压缩方法
    /// @return 返回检查后的方法，如果方法无效，回退为空字符串，表示不压缩
    std::string chechCompressMethod(const std::string& method);

    /// @brief 将字符串转换为写入文件的字节流
    /// @param str 需要转换的字符串
    /// @return 返回字节流
    std::vector<char> string2data(const std::string& str);

    /// @brief 将文件中的字节流转换为字符串
    /// @param file 文件流
    /// @param seek 是否在一边读取文件，一边向后移动文件指针，默认为true
    /// @return 返回转换后的字符串
    std::string data2string(std::fstream& file, bool seek = true);

    /// @brief 将数组转换为写入文件的字节流
    /// @param data 数组数据
    /// @return 返回字节流
    template<typename T>
    std::vector<char> array2data(const std::vector<T>& data);

    /// @brief 从文件中读取数组数据
    /// @param file 文件流
    /// @param seek 是否移动文件指针，默认为true
    /// @return 返回读取的char数组，需要自行reinterpret_cast为所需类型
    std::vector<char> data2array(std::fstream& file, bool seek = true);

    /// @brief 压缩数据
    /// @param input 输入数据
    /// @return 压缩后的数据，失败返回空向量
    template<typename T>
    std::vector<char> compressData(const std::vector<T>& input);

    /// @brief 解压数据
    /// @param input 压缩数据
    /// @param decompressedSize 期望的解压后数据大小，0表示未知大小
    /// @return 解压后的数据，失败返回空向量
    std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize = 0, const std::string& method = "");

    /// @brief 从文件中解压数据
    /// @param file 文件流
    /// @param compressedSize 压缩数据大小
    /// @param decompressedSize 预估的解压后大小，0表示未知
    /// @param seek 是否移动文件指针，默认为true
    /// @return 解压后的数据，失败返回空向量
    std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize = 0, bool seek = true, const std::string& method = "");

    /// @brief 张量信息结构体
    struct TensorInfo {
        std::string name;                   // 张量名称
        std::string typeName;               // 元素类型名称
        int32_t typeSize;                   // 元素类型的字节大小
        std::string tensorType;             // dense, sparse等
        std::vector<int> shape;             // 张量形状
        uint64_t dataOffset;                // 在文件中的偏移
        uint64_t compressedSize;            // 压缩后的数据大小
        std::string compressMethod;         // 压缩算法
        uint64_t uncompressedSize;          // 原始数据大小（通过shape计算得到，单位是字节）
        std::vector<char> compressedData;   // 为空表示数据在磁盘，使用dataOffset读取；不为空表示数据在内存中，且原dataOffset失效。
    };

    class YTensorIO{
    public:
        YTensorIO() = default;
        ~YTensorIO();

        /// @brief 打开文件
        /// @param fileName 文件名
        /// @param forWrite 是否为写入模式，true为写入，false为读取
        /// @return 如果文件打开成功，返回true；否则返回false。
        bool open(const std::string& fileName, int fileMode = yt::io::Read);

        /// @brief 关闭文件
        void close();

        /// @brief 获取文件中所有张量的名称
        /// @return 张量名称列表
        std::vector<std::string> getTensorNames() const;

        /// @brief 获取张量信息
        /// @param name 张量名称，缺省表示获取第一个张量
        /// @return 张量信息，如果不存在返回nullptr
        TensorInfo getTensorInfo(const std::string& name = "") const;

        /// @brief 保存张量数据
        /// @param tensor 需要保存的张量
        /// @param name 张量名称
        /// @return 如果保存成功，返回true；否则返回false。
        template<typename T, int dim>
        bool save(const YTensor<T, dim>& tensor, const std::string& name);

        /// @brief 加载张量数据
        /// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
        /// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
        /// @return 如果加载成功，返回true；否则返回false。
        template<typename T, int dim>
        bool load(YTensor<T, dim>& tensor, const std::string& name = "");

        /// @brief 检查文件格式是否正确
        /// @return 如果文件格式正确，返回true；否则返回false。
        bool validateFile();

    protected:
        /// @brief 读取文件头信息
        /// @return 如果成功，返回true；否则返回false。
        bool readHeader();

        /// @brief 写入文件头信息
        /// @return 如果成功，返回true；否则返回false。
        bool writeHeader();

        /// @brief 读取索引信息
        /// @return 如果成功，返回true；否则返回false。
        bool readIndex();

        /// @brief 写入索引信息
        /// @param tensorStructureOffsets 张量结构体的文件偏移量
        /// @return 如果成功，返回true；否则返回false。
        bool writeIndex(std::vector<uint64_t> tensorStructureOffsets);

    protected:
        std::fstream _file;                     // 文件流
        std::string _fileName;                  // 文件名
        int _fileMode = yt::io::Closed;         // 当前文件模式
        bool _isHeaderRead = false;             // 是否已读取文件头
        uint8_t _version = 0;                   // 版本号
        std::string _metadata;                  // JSON格式的元数据
        std::vector<TensorInfo> _tensorInfos;   // 保存张量信息和顺序
    };

    /// @brief 便利函数用于快速保存和加载
    /// @param fileName 文件名
    /// @param tensor 需要保存的张量
    /// @param name 张量名称
    /// @return 如果保存成功，返回true；否则返回false。
    template<typename T, int dim>
    bool saveTensor(const std::string& fileName, const YTensor<T, dim>& tensor, const std::string& name = "");

    /// @brief 便利函数用于快速保存和加载
    /// @param fileName 文件名
    /// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
    /// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
    /// @return 如果加载成功，返回true；否则返回false。
    template<typename T, int dim>
    bool loadTensor(const std::string& fileName, YTensor<T, dim>& tensor, const std::string& name = "");
}; // namespace yt::io

#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <zlib.h>

namespace yt::io {

std::string checkCompressMethod(const std::string& method) {
    if (method == "zlib" ||
        method.empty()
    ) {
        return method;
    } else {
        if (yt::io::verbose) {
            std::cerr << "Warning: Unknown compression method '" << method << "'. Falling back to no compression." << std::endl;
        }
        return ""; // 回退到不压缩
    }
}

std::vector<char> string2data(const std::string& str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    std::vector<char> op(sizeof(uint32_t) + length);
    std::memcpy(op.data(), &length, sizeof(uint32_t));
    if (!str.empty()) {
        std::memcpy(op.data() + sizeof(uint32_t), str.c_str(), length);
    }

    return op;
}

std::string data2string(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint32_t length;
    if (!file.read(reinterpret_cast<char*>(&length), sizeof(uint32_t))) {
        throw std::runtime_error("Failed to read string length");
    }
    std::string op;
    if (length > 0) {
        op.resize(length);
        if (!file.read(op.data(), length)) {
            throw std::runtime_error("Failed to read string data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }
    return op;
}

template<typename T>
std::vector<char> array2data(const std::vector<T>& data) {
    uint64_t count = static_cast<uint64_t>(data.size() * sizeof(T));
    std::vector<char> op(sizeof(uint64_t) + count);
    std::memcpy(op.data(), &count, sizeof(uint64_t));
    if (!data.empty()) {
        std::memcpy(op.data() + sizeof(uint64_t), data.data(), count);
    }

    return op;
}

std::vector<char> data2array(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint64_t count;
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t))) {
        throw std::runtime_error("Failed to read array count");
    }
    std::vector<char> op;
    if (count > 0) {
        op.resize(count);
        if (!file.read(op.data(), count)) {
            throw std::runtime_error("Failed to read array data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }
    return op;
}

template<typename T>
std::vector<char> compressData(const std::vector<T>& input) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(yt::io::compressMethod);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        // 初始化压缩流zlib
        if (deflateInit2(&stream, yt::io::compressLevel, Z_DEFLATED, 15, 8, Z_FILTERED) != Z_OK) {
            return {}; // 初始化失败
        }
        stream.avail_in = input.size() * sizeof(T);
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        // 预估压缩后大小，通常比原数据稍大一些作为缓冲
        size_t estimated_size = deflateBound(&stream, stream.avail_in);
        std::vector<char> compressed(estimated_size);
        stream.avail_out = compressed.size();
        stream.next_out = reinterpret_cast<Bytef*>(compressed.data());
        int result = deflate(&stream, Z_FINISH);
        deflateEnd(&stream);

        if (result != Z_STREAM_END) {
            return {}; // 压缩失败
        }
        compressed.resize(stream.total_out);
        return compressed;
    } else {
        // 缺省值/fallback，不压缩
        std::vector<char> op(input.size() * sizeof(T));
        std::memcpy(op.data(), input.data(), input.size() * sizeof(T));
        return op;
    }
}

std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize, const std::string& method) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        if (inflateInit2(&stream, 15 + 32) != Z_OK) {
            return {}; // 初始化失败
        }
        stream.avail_in = input.size();
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        std::vector<std::vector<char>> chunks;
        const size_t chunk_size = 65536; // 64KB chunks
        // 第一个chunk：如果已知解压大小，使用该大小；否则使用默认chunk大小
        size_t first_chunk_size = (decompressedSize > 0) ? decompressedSize : chunk_size;
        chunks.emplace_back(first_chunk_size);
        std::vector<char>& current_chunk = chunks.back();
        stream.avail_out = first_chunk_size;
        stream.next_out = reinterpret_cast<Bytef*>(current_chunk.data());
        int result = inflate(&stream, Z_NO_FLUSH);
        if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
            // 不继续处理 & 没有完成 & 非缓冲区错误
            inflateEnd(&stream);
            if (yt::io::verbose) {
                std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
            }
            return {};
        }
        size_t chunk_used = first_chunk_size - stream.avail_out;
        current_chunk.resize(chunk_used);// 真实chunk大小=size
        // 如果还有剩余数据需要解压，继续处理
        while (result != Z_STREAM_END && stream.avail_out == 0) {
            chunks.emplace_back(chunk_size);
            std::vector<char>& next_chunk = chunks.back();

            stream.avail_out = chunk_size;
            stream.next_out = reinterpret_cast<Bytef*>(next_chunk.data());

            result = inflate(&stream, Z_NO_FLUSH);

            // 检查是否为真正的错误
            if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
                inflateEnd(&stream);
                if (yt::io::verbose) {
                    std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
                }
                return {}; // 解压失败
            }

            // 调整当前chunk大小
            chunk_used = chunk_size - stream.avail_out;
            next_chunk.resize(chunk_used);
        }
        inflateEnd(&stream);
        if (result != Z_STREAM_END) {
            // 解压失败
            return {};
        }
        // 合并所有chunks
        size_t total_size = 0;
        for (const auto& chunk : chunks) {
            total_size += chunk.size();
        }
        std::vector<char> decompressed;
        decompressed.reserve(total_size);
        for (auto& chunk : chunks) {
            decompressed.insert(decompressed.end(),
                               std::make_move_iterator(chunk.begin()),
                               std::make_move_iterator(chunk.end()));
        }
        return decompressed;
    } else {
        // 缺省值/fallback，不压缩
        return input;
    }

}

std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize, bool seek, const std::string& method) {
    if (compressedSize == 0) {
        return {};
    }

    std::streampos originalPos = file.tellg();
    std::vector<char> output;

    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));

        if (inflateInit2(&stream, 15 + 32) != Z_OK) {
            return {};
        }

        // 我们使用单一连续输出缓冲区来避免多chunk管理时的指针混淆。
        const size_t io_chunk = 65536; // 64KB 输入缓冲
        std::vector<char> input_buffer(io_chunk);
        size_t remaining = compressedSize;

        // 如果已知解压后的大小，预分配；否则按需增长
        if (decompressedSize > 0) {
            output.resize(decompressedSize);
        } else {
            // 先分配一个chunk作为起始空间
            output.resize(io_chunk);
        }

        size_t out_pos = 0; // 当前写入位置

        // 初始化 zlib 的输出指针指向输出缓冲的可用空间
        stream.avail_out = static_cast<uInt>(output.size() - out_pos);
        stream.next_out = reinterpret_cast<Bytef*>(output.data() + out_pos);

        int result = Z_OK;
        while (result != Z_STREAM_END) {
            // 当没有输入可用并且还有剩余压缩字节时，从文件读取下一块压缩数据
            if (stream.avail_in == 0 && remaining > 0) {
                size_t to_read = std::min(remaining, io_chunk);
                if (!file.read(input_buffer.data(), to_read)) {
                    inflateEnd(&stream);
                    if (!seek) file.seekg(originalPos);
                    if (yt::io::verbose) {
                        std::cerr << "Error: Failed to read compressed data from file" << std::endl;
                    }
                    return {};
                }
                stream.next_in = reinterpret_cast<Bytef*>(input_buffer.data());
                stream.avail_in = static_cast<uInt>(to_read);
                remaining -= to_read;
            }

            // 如果输出缓冲已满，需要扩展输出缓冲并更新 next_out
            if (stream.avail_out == 0) {
                // 增加额外空间，扩展大小为 io_chunk 或剩余预知大小
                size_t add = io_chunk;
                if (decompressedSize > 0) {
                    // 如果知道总大小，尽量仅扩展到剩余需要的大小
                    if (decompressedSize > out_pos) {
                        size_t need = decompressedSize - out_pos;
                        add = std::min(add, need);
                    } else {
                        add = io_chunk;
                    }
                }
                size_t old_size = output.size();
                output.resize(old_size + add);
                stream.next_out = reinterpret_cast<Bytef*>(output.data() + out_pos);
                stream.avail_out = static_cast<uInt>(output.size() - out_pos);
            }

            result = inflate(&stream, Z_NO_FLUSH);
            if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
                inflateEnd(&stream);
                if (!seek) file.seekg(originalPos);
                if (yt::io::verbose) {
                    std::cerr << "Error: Failed to decompress data from file" << std::endl;
                }
                return {};
            }

            // 计算本次写入了多少字节并推进 out_pos
            size_t wrote = (output.size() - out_pos) - stream.avail_out;
            out_pos += wrote;

            // 如果流已经结束，跳出循环
            if (result == Z_STREAM_END) break;

            // 如果没有更多输入也没有更多输出空间但尚未结束，继续循环以触发更多读取或扩展
            if (stream.avail_in == 0 && remaining == 0 && stream.avail_out > 0 && result == Z_BUF_ERROR) {
                // 没有更多输入，inflate 需要更多输入才可能完成
                break;
            }
        }

        inflateEnd(&stream);

        // 截断输出到实际写入长度
        output.resize(out_pos);

    } else {
        // 缺省值/fallback，不压缩。以compressed size为准，因为后者可以是缺省值
        output.resize(compressedSize);
        if (!file.read(output.data(), compressedSize)) {
            file.seekg(originalPos);
            if (yt::io::verbose) {
                std::cerr << "Error: Failed to read compressed data from file" << std::endl;
            }
            return {};
        }
    }

    if (!seek) {
        file.seekg(originalPos);
    }
    return output;
}

// YTensorIO class implementation
YTensorIO::~YTensorIO() {
    close();
}

bool YTensorIO::open(const std::string& fileName, int fileMode) {
    close();
    _fileName = fileName;
    _fileMode = fileMode;
    if (fileMode == yt::io::Write || fileMode == yt::io::Append) {
        // 写入模式：先尝试读取现有文件的张量信息
        std::ifstream checkFile(fileName, std::ios::binary);
        bool fileExists = checkFile.good();
        checkFile.close();

        if (fileExists) {
            // 如果文件存在，先读取现有的张量信息和数据到内存
            _file.open(fileName, std::ios::binary | std::ios::in);
            if (_file.is_open() && readHeader() && readIndex()) {
                _file.close();
            } else {
                // 读取失败，当作新文件处理
                _file.close();
                _tensorInfos.clear();
            }
        } else {
            // 文件不存在，创建新的文件
            _file.open(fileName, std::ios::binary | std::ios::out | std::ios::trunc);
            _file.close();
        }

        // 在关闭文件之前，都以只读模式打开
        _file.open(fileName, std::ios::binary | std::ios::in);
        return true;
    } else {
        // 读取模式
        _file.open(fileName, std::ios::binary | std::ios::in);
        if (!_file.is_open()) {
            if (verbose) {
                std::cerr << "Error: Failed to open file for reading: " << fileName << std::endl;
            }
            return false;
        }
        // 读取并验证文件
        if (!readHeader()) {
            if (verbose) {
                std::cerr << "Error: Failed to read file header" << std::endl;
            }
            close();
            return false;
        }
        if (!readIndex()) {
            if (verbose) {
                std::cerr << "Error: Failed to read file index" << std::endl;
            }
            close();
            return false;
        }
        return true;
    }
}

void YTensorIO::close() {
    if (_file.is_open()) {
        if (_fileMode == yt::io::Write || _fileMode == yt::io::Append) {
            if (_fileMode == yt::io::Append) {
                // 如果是附加模式，先加载所有的张量数据进内存。无需解压。
                for (size_t i = 0; i < _tensorInfos.size(); ++i) {
                    auto& info = _tensorInfos[i];
                    if(!info.compressedData.empty()) {
                        // 已经在内存中，无需重复加载
                        info.compressedSize = info.compressedData.size();
                        continue;
                    }
                    _file.seekg(info.dataOffset, std::ios::beg);

                    // 读取压缩数据到内存
                    info.compressedData.resize(info.compressedSize);
                    if (info.compressedSize > 0) {
                        if (!_file.read(info.compressedData.data(), info.compressedSize)){
                            if (verbose) {
                                std::cerr << "Error: Failed to read compressed data for tensor '"
                                          << info.name << "'" << std::endl;
                            }
                            info.compressedData.clear();
                        }
                    }
                }
            } else {
                // 如果是写入模式，去除所有不在内存中的张量
                std::erase_if(_tensorInfos, [](const TensorInfo& info) {
                    return info.compressedData.empty();
                });
            }

            // 重新打开文件进行写入
            std::string tempFileName = _fileName;
            _file.close();
            _file.open(tempFileName, std::ios::binary | std::ios::out | std::ios::trunc);
            if (_file.is_open()) {
                // 写入头部（包含 metadata）
                if (!writeHeader()) {
                    if (verbose) {
                        std::cerr << "Error: Failed to write header during close" << std::endl;
                    }
                }

                // 写入所有张量数据
                std::vector<uint64_t> offsets;
                for (size_t i = 0; i < _tensorInfos.size(); ++i) {
                    const auto& info = _tensorInfos[i];

                    // 记录数据偏移量（张量结构数据的开始位置）
                    uint64_t dataOffset = _file.tellp();
                    offsets.push_back(dataOffset);
                    // 写入张量元数据
                    auto nameData = string2data(info.name);
                    auto typeNameData = string2data(info.typeName);
                    int32_t typeSize = info.typeSize;
                    auto tensorTypeData = string2data(info.tensorType);
                    auto shapeData = array2data(info.shape);
                    auto compressMethodData = string2data(info.compressMethod);

                    if (!_file.write(nameData.data(), nameData.size()) ||
                        !_file.write(typeNameData.data(), typeNameData.size()) ||
                        !_file.write(reinterpret_cast<const char*>(&typeSize), sizeof(int32_t)) ||
                        !_file.write(tensorTypeData.data(), tensorTypeData.size()) ||
                        !_file.write(shapeData.data(), shapeData.size()) ||
                        !_file.write(compressMethodData.data(), compressMethodData.size())) {
                        if (verbose) {
                            std::cerr << "Error: Failed to write tensor metadata for '" << info.name << "'" << std::endl;
                        }
                        break;
                    }
                    // 写入压缩数据大小
                    if (!_file.write(reinterpret_cast<const char*>(&info.compressedSize), sizeof(uint64_t))) {
                        if (verbose) {
                            std::cerr << "Error: Failed to write compressed size for '" << info.name << "'" << std::endl;
                        }
                        break;
                    }
                    if (!info.compressedData.empty()) {
                        if (!_file.write(info.compressedData.data(), info.compressedData.size())) {
                            if (verbose) {
                                std::cerr << "Error: Failed to write tensor data for '" << info.name << "'" << std::endl;
                            }
                            break;
                        }
                    }
                }
                // 写入索引到文件末尾
                if (!writeIndex(offsets)) {
                    if (verbose) {
                        std::cerr << "Error: Failed to write index during close" << std::endl;
                    }
                }
            }
        }
        _file.close();
    }
    _fileMode = yt::io::Closed;
    _isHeaderRead = false;
    _version = 0;
    _tensorInfos.clear();
    _metadata.clear();
    _fileName.clear();
}

std::vector<std::string> YTensorIO::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& tensorInfo : _tensorInfos) {
        names.push_back(tensorInfo.name);
    }
    return names;
}

TensorInfo YTensorIO::getTensorInfo(const std::string& name) const {
    if (name.empty()) {
        // 缺省取第一个张量
        if (_tensorInfos.empty()) {
            throw std::runtime_error("No tensors available");
        }
        return _tensorInfos[0];
    }

    // 按名称查找
    for (const auto& tensorInfo : _tensorInfos) {
        if (tensorInfo.name == name) {
            return tensorInfo;
        }
    }
    throw std::runtime_error("Tensor not found: " + name);
}

bool YTensorIO::validateFile() {
    if (!_file.is_open() || !_fileMode) {
        return false;
    }
    _file.seekg(0, std::ios::beg);
    // Check magic
    std::string magic(8, '\0');
    if (!_file.read(magic.data(), magic.size())) {
        return false;
    }
    return magic == yt::io::YTENSOR_FILE_MAGIC;
}

bool YTensorIO::readHeader() {
    if (!_file.is_open()) {
        return false;
    }
    _file.seekg(0, std::ios::beg);
    // Read magic
    std::string magic(8, '\0');
    if (!_file.read(magic.data(), magic.size()) || magic != yt::io::YTENSOR_FILE_MAGIC) {
        return false;
    }
    // Read version
    if (!_file.read(reinterpret_cast<char*>(&_version), sizeof(uint8_t))) {
        return false;
    }
    if (_version > yt::io::YTENSOR_FILE_VERSION) {
        // 文件版本过高时输出警告，但不阻止读取（向下兼容）
        if (verbose) {
            std::cerr << "Warning: YTensor file version is newer than supported. "
                      << "Current version " << static_cast<int>(yt::io::YTENSOR_FILE_VERSION)
                      << ", file version " << static_cast<int>(_version)
                      << ". Reading may fail or produce unexpected results." << std::endl;
        }
    }
    // 读取 metadata（紧接着版本号）
    _metadata = data2string(_file);
    _isHeaderRead = true;
    return true;
}

bool YTensorIO::writeHeader() {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        return false;
    }
    _file.seekp(0, std::ios::beg);
    // magic
    if (!_file.write(yt::io::YTENSOR_FILE_MAGIC.data(), yt::io::YTENSOR_FILE_MAGIC.size())) {
        return false;
    }
    // version
    if (!_file.write(reinterpret_cast<const char*>(&yt::io::YTENSOR_FILE_VERSION), sizeof(uint8_t))) {
        return false;
    }
    // metadata
    auto metadataData = string2data(_metadata);
    if (!_file.write(metadataData.data(), metadataData.size())) {
        return false;
    }
    return true;
}

bool YTensorIO::readIndex() {
    if (!_file.is_open()) {
        return false;
    }
    _tensorInfos.clear();
    _file.seekg(-static_cast<int>(sizeof(uint32_t)), std::ios::end);
    uint32_t tensorCount;
    if (!_file.read(reinterpret_cast<char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    // 读取索引
    const std::streamoff indexSize =
        static_cast<std::streamoff>(sizeof(uint32_t)) +
        static_cast<std::streamoff>(tensorCount) * sizeof(uint64_t);
    _file.seekg(-indexSize, std::ios::end);

    std::vector<uint64_t> offsets(tensorCount);
    if (!_file.read(reinterpret_cast<char*>(offsets.data()), tensorCount * sizeof(uint64_t))) {
        return false;
    }

    // Read tensor information from each offset
    for (size_t i = 0; i < tensorCount; ++i) {
        _file.seekg(offsets[i], std::ios::beg);
        TensorInfo info;
        info.name = data2string(_file);
        info.typeName = data2string(_file);
        if (!_file.read(reinterpret_cast<char*>(&info.typeSize), sizeof(int32_t))) {
            return false;
        }
        info.tensorType = data2string(_file);
        auto shapeData = data2array(_file);
        const int32_t* shapePtr = reinterpret_cast<const int32_t*>(shapeData.data());
        size_t shapeCount = shapeData.size() / sizeof(int32_t);
        info.shape.assign(shapePtr, shapePtr + shapeCount);
        info.compressMethod = data2string(_file);
        uint64_t compressedSize;
        if (!_file.read(reinterpret_cast<char*>(&compressedSize), sizeof(uint64_t))) {
            return false;
        }
        info.compressedSize = compressedSize;
        // data offset是指向压缩数据的偏移量
        info.dataOffset = _file.tellg();
        info.uncompressedSize = info.typeSize;
        for (auto dim : info.shape) {
            info.uncompressedSize *= dim;
        }
        info.compressedData.clear(); // 初始时不加载数据
        _tensorInfos.push_back(info);
    }
    return true;
}

bool YTensorIO::writeIndex(std::vector<uint64_t> offsets) {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        return false;
    }
    if (!_file.write(reinterpret_cast<const char*>(offsets.data()),
                     offsets.size() * sizeof(uint64_t))) {
        return false;
    }
    uint32_t tensorCount = static_cast<uint32_t>(offsets.size());
    if (!_file.write(reinterpret_cast<const char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    return true;
}

template<typename T, int dim>
bool YTensorIO::save(const YTensor<T, dim>& tensor, const std::string& name) {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        if (verbose) {
            std::cerr << "Error: File not open for writing" << std::endl;
        }
        return false;
    }

    // 如果 name 为空，自动命名为 tensorInfos 的 size
    std::string tensorName = name.empty() ? std::to_string(_tensorInfos.size()) : name;

    // 检查是否存在同名张量，如果存在则覆盖
    int existingIndex = -1;
    for (size_t i = 0; i < _tensorInfos.size(); ++i) {
        if (_tensorInfos[i].name == tensorName) {
            existingIndex = static_cast<int>(i);
            if (verbose) {
                std::cerr << "Warning: Tensor '" << tensorName << "' already exists, will be overwritten" << std::endl;
            }
            break;
        }
    }
    // 需要保证是单独存在的，否则dataVector指向的数据offset会不正确
    YTensor<T, dim> contiguousTensor = tensor.clone();

    // 创建张量信息
    TensorInfo info;
    info.name = tensorName;
    info.typeName = yt::types::getTypeName<T>();
    info.typeSize = sizeof(T);
    info.tensorType = "dense";
    // Convert shape to int32_t vector
    auto shape = contiguousTensor.shape();
    info.shape.resize(shape.size());
    std::transform(shape.begin(), shape.end(), info.shape.begin(), [](int s) {
        return static_cast<int32_t>(s);
    });

    // 准备并压缩张量数据
    info.compressMethod = checkCompressMethod(yt::io::compressMethod);
    auto compressedData = compressData(contiguousTensor.dataVector());
    if (compressedData.empty()) {
        if (verbose) {
            std::cerr << "Error: Failed to compress tensor data" << std::endl;
        }
        return false;
    }

    info.compressedSize = static_cast<uint32_t>(compressedData.size());
    info.uncompressedSize = contiguousTensor.size() * sizeof(T);

    // dataOffset 将在 close 函数中重新计算

    if (existingIndex >= 0) {
        // 覆盖现有张量
        _tensorInfos[existingIndex] = info;
        _tensorInfos[existingIndex].compressedData = std::move(compressedData);
    } else {
        // 添加新张量
        info.compressedData = std::move(compressedData);
        _tensorInfos.push_back(info);
    }
    return true;
}

template<typename T, int dim>
bool YTensorIO::load(YTensor<T, dim>& tensor, const std::string& name) {
    if (!_file.is_open() || !_fileMode) {
        if (verbose) {
            std::cerr << "Error: File not open for reading" << std::endl;
        }
        return false;
    }

    // 获取张量信息
    TensorInfo info;
    try {
        info = getTensorInfo(name);
    } catch (const std::runtime_error& e) {
        if (verbose) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        return false;
    }

    // 检查类型兼容性
    if (info.typeName != yt::types::getTypeName<T>()) {
        if (verbose) {
            std::cerr << "Warning: Type mismatch. Expected '" << yt::types::getTypeName<T>()
                      << "', got '" << info.typeName <<
                      "'. but will still try to load tensor if element size matches" << std::endl;
        }
    }
    // 检查元素大小
    if (info.typeSize != sizeof(T)) {
        if (verbose) {
            std::cerr << "Error: Element size mismatch. Expected " << sizeof(T)
                      << ", got " << info.typeSize << "." << std::endl;
        }
        return false;
    }
    // 检查张量维度
    if (static_cast<int>(info.shape.size()) != dim) {
        if (verbose) {
            std::cerr << "Error: Dimension mismatch. Expected " << dim
                      << "D, got " << info.shape.size() <<
                      "D, use load(YTensorBase<T>) or load(YTensor<T, " << info.shape.size() << ">) instead" << std::endl;
        }
        return false; // Dimension mismatch
    }

    std::vector<int> shape(info.shape.begin(), info.shape.end());

    std::vector<char> rawData;

    // 根据数据位置选择读取方式
    if (!info.compressedData.empty()) {
        // 数据在内存中，直接解压内存中的压缩数据
        rawData = decompressData(info.compressedData, info.uncompressedSize, info.compressMethod);
    } else {
        // 数据在磁盘中，从文件读取并解压
        _file.seekg(info.dataOffset, std::ios::beg);
        rawData = decompressData(_file, info.compressedSize, info.uncompressedSize, true, info.compressMethod);
    }

    if (rawData.empty() && info.uncompressedSize > 0) {
        if (verbose) {
            std::cerr << "Error: File corrupted or incomplete" << std::endl;
        }
        return false; // Decompression failed
    }
    // 检查tensorSize是否与解压的数据大小匹配
    if (info.uncompressedSize != rawData.size()) {
        if (verbose) {
            std::cerr << "Error: YTensor data size mismatch. Expected " << info.uncompressedSize
                      << ", got " << rawData.size() <<
                       ". Please check your file." << std::endl;
        }
        return false; // Decompressed data size mismatch
    }

    tensor = YTensor<T, dim>(shape);

    // Copy decompressed data to tensor
    if (!rawData.empty()) {
        std::memcpy(tensor.data(), rawData.data(), rawData.size());
    }
    return true;
}

// 便利函数实现
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const YTensor<T, dim>& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, true)) {
        return false;
    }
    if (!io.save(tensor, name)) {
        return false;
    }
    io.close();
    return true;
}

template<typename T, int dim>
bool loadTensor(const std::string& fileName, YTensor<T, dim>& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, false)) {
        return false;
    }
    return io.load(tensor, name);
}

} // namespace yt::io

// **********************************
// TODO:
// 1、需要tensorBase类的支持，实现对外部文件读取创建类对象的支持
// 2、将decompressData改成模板函数，以减少被std::vector恶心的机会，实现load的零拷贝
