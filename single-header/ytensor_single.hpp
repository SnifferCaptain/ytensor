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

#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// 前向声明
namespace yt {
class YTensorBase;
}
template<typename T, int dim> class YTensor;


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

namespace yt::traits {
    /// @brief 判断类型是否为YTensor或YTensorBase（广义tensor类型）
    template<typename U>
    struct is_ytensor : std::bool_constant<std::is_base_of_v<yt::YTensorBase, std::decay_t<U>>> {};

    template<typename U>
    inline constexpr bool is_ytensor_v = is_ytensor<U>::value;

    /// @brief 判断类型是否为精确的YTensor<U, d>模板实例
    template<typename U>
    struct is_ytensor_template : std::false_type {};

    template<typename U, int d>
    struct is_ytensor_template<YTensor<U, d>> : std::true_type {};

    template<typename U>
    inline constexpr bool is_ytensor_template_v = is_ytensor_template<std::decay_t<U>>::value;

    /// @brief 获取YTensor的维度
    template<typename U>
    struct ytensor_dim { static constexpr int value = 0; };

    template<typename U, int d>
    struct ytensor_dim<YTensor<U, d>> { static constexpr int value = d; };

    template<typename U>
    inline constexpr int ytensor_dim_v = ytensor_dim<std::decay_t<U>>::value;

    /// @brief 获取参数包中张量的最大维度
    template<typename... Args>
    constexpr int max_dim() {
        int dims[] = {ytensor_dim_v<Args>...};
        int maxd = 1;
        for (int d : dims) if (d > maxd) maxd = d;
        return maxd;
    }
}
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
#include <tuple>
#include <functional>
#include <memory>

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

    struct TypeRegItem{
        std::string name;
        int32_t size;
        std::function<std::string(const void*)> toString;
        // 非POD类型支持：析构和拷贝构造
        bool isPOD = true;  // POD类型不需要特殊处理
        std::function<void(void*)> destructor = nullptr;           // 调用析构函数
        std::function<void(void*, const void*)> copyConstruct = nullptr;  // placement new + 拷贝构造
        std::function<void(void*)> defaultConstruct = nullptr;     // placement new + 默认构造
    };

    /// @brief 类型注册表
    /// @return 返回类型注册表的引用
    inline auto& getTypeRegistry() {
        static std::unordered_map<std::string, yt::infos::TypeRegItem> registry;
        return registry;
    }

    /// @brief 文件头标识
    static constexpr std::string_view YTENSOR_FILE_MAGIC = "YTENSORF";

    /// @brief 文件版本
    static constexpr uint8_t YTENSOR_FILE_VERSION = 0;

    /// @brief 控制是否启用Eigen库的宏，默认启用
    #ifndef YT_USE_EIGEN
        #if __has_include(<Eigen/Core>)
            #define YT_USE_EIGEN 1
        #else
            #define YT_USE_EIGEN 0
        #endif
    #endif

    /// @brief 控制是否启用YTensorBase模板显式实例化（预创建常用类型模板）
    /// 优点：减少编译时间，可能提升运行时性能（减少模板实例化开销）
    /// 设为0则不预创建，所有模板按需实例化
    #ifndef YT_PREINSTANTIATE_TEMPLATES
        #define YT_PREINSTANTIATE_TEMPLATES 1
    #endif
}// namespace yt::infos

/////////////// extern includes ///////////////

#if YT_USE_EIGEN
#include <Eigen/Core>
#endif // YT_USE_EIGEN


/***************
* @file: ytensor_base.hpp
* @brief: YTensorBase基类的功能实现，YTensorBase类不包含模板参数，提供运行时接口。
***************/
#include <memory>
#include <vector>
#include <string>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <random>
#include <cstring>


namespace yt{

class YTensorBase {
public:
    YTensorBase() = default;

    /// @brief 构造函数
    /// @param shape 张量的形状
    /// @param dtype 数据类型
    /// @example YTensorBase t0({3, 4, 5}, "float32");
    YTensorBase(const std::vector<int>& shape, const std::string& dtype = "float32");

    /// @brief 拷贝构造函数
    /// @param other 另一个张量
    YTensorBase(const YTensorBase& other);

    /// @brief 拷贝赋值运算符
    /// @param other 另一个张量
    /// @return 返回自身的引用
    YTensorBase& operator=(const YTensorBase& other);

    virtual ~YTensorBase() = default;

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

    /// @brief 获取张量的元素个数
    /// @return 返回张量的元素个数。
    /// @example YTensor<float, 3> a(3, 4, 5); auto size = a.size(); // size = 60
    /// @note 如果张量是非contiguous的，size()方法返回的是逻辑个数，而不是实际个数。
    size_t size() const;

    /// @brief 获取张量的维度数。
    /// @return 返回张量的维度数量。
    int ndim() const;

    /// @brief 获取张量的数据偏移量（以元素为单位）。
    /// @return 返回当前视图相对于数据存储区头部的偏移量。
    template <typename... Args> int offset(Args... index) const;
    int offset(const std::vector<int>& index) const;

    /// @brief 获取张量的物理偏移量，考虑了张量自身的_offset
    /// @param index 元素的索引
    /// @return 返回 _offset + offset(index)
    template <typename... Args> int offset_(Args... index) const;
    int offset_(const std::vector<int>& index) const;

    /// @brief 获取张量的数据指针（按元素类型 T 返回指针）。已经包含偏移量。
    /// @tparam T 元素类型，例如 float
    /// @return 返回张量数据的指针（T*）。
    template <typename T>
    T* data();
    template <typename T>
    const T* data() const;
    /// @brief 便捷重载：当 dtype 为 float32 时可直接使用无模板 data()
    float* data();
    const float* data() const;

    /// @brief 获取张量的连续版本
    /// @return 返回连续张量。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    YTensorBase contiguous() const;

    /// @brief 原地操作，使张量连续
    /// @return 返回自身的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    /// @note 这会导致原先的“引用”失效
    YTensorBase& contiguous_();

    /// @brief 检查张量的形状是否与另一个形状匹配
    /// @param otherShape 另一个张量的形状。
    /// @return 如果形状匹配则返回true，否则返回false。
    bool shapeMatch(const std::vector<int> &otherShape) const;

    /// @brief 获取张量的形状在某个轴上的大小（无安全检查、循环版的高效实现）
    /// @param atDim 维度索引，从0开始。
    /// @return 返回指定维度的大小。
    /// @example YTensor<float, 3> a(3, 4, 5); auto dimSize = a.shape(1); // dimSize = 4
    int shape_(int atDim) const;

    int shapeSize() const;

    /// @brief 张量是否是连续的
    /// @param fromDim 从该维度开始检查
    /// @return 如果张量是连续的，返回true；否则返回false。
    /// @example YTensor<float, 3> a(3, 4, 5); bool is_contiguous = a.isContiguous(); // is_contiguous = true
    bool isContiguous(int fromDim = 0) const;

    /// @brief 获取张量从哪个维度开始是连续的
    /// @return 返回从哪个维度开始是连续的。如果完全不连续的话，就返回dim
    int isContiguousFrom() const;

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

    /// @brief 获取坐标对应的物理位置，可以使用atData_()方法获取数据。
    /// @return 相对data的偏移量
    /// @example YTensor<float, 3> a(3, 4, 5); auto offset = a.toIndex_(1, 2, 3); // offset = 31
    template <typename... Args> size_t toIndex_(const Args... args) const;
    size_t toIndex_(const std::vector<int> &pos) const;

    /// @brief 获取位置对应的坐标
    /// @param index 位置
    /// @return 坐标
    /// @note 输入必须是逻辑位置。
    std::vector<int> toCoord(size_t index) const;

    /// @brief 通过位置索引访问元素（模板版本，返回引用）
    template <typename T, typename... Args>
    T& at(Args... args);
    template <typename T>
    T& at(const std::vector<int>& pos);
    template <typename T>
    const T& at(const std::vector<int>& pos) const;

    /// @brief 按逻辑下标访问（模板版本），index是由shape计算出来的逻辑索引
    /// @note 这个方法没有at高效，只是用作对原始逻辑的兼容
    template <typename T>
    T& atData(int index);
    template <typename T>
    const T& atData(int index) const;

    /// @brief 按物理下标访问（模板版本），返回底层元素引用
    /// @note 可以用作element-wise操作，相对高效，但是要求contiguous，且不进行安全检查
    template <typename T>
    T& atData_(int index);
    template <typename T>
    const T& atData_(int index) const;

    /// @brief 浅拷贝（共享底层数据）
    void shallowCopyTo(YTensorBase &other) const;

    /// @brief 深拷贝：返回一个独立拥有自己数据的 YTensorBase
    YTensorBase clone() const;

    /// @brief 从源张量复制元素到本张量（原地操作，不重新分配内存）
    /// @param src 源张量，shape必须与本张量一致
    /// @return 返回自身引用
    /// @note 支持非连续张量，会尽可能地复制连续内存块以提高效率
    /// @note 目前不支持src与dst的内存重叠，若存在重叠则行为未定义
    YTensorBase& copy_(const YTensorBase& src);

    /// @brief 自动推导合适形状
    /// @param shape 形状
    /// @return 返回一个vector<int>，表示推断出的张量的形状。
    /// @example YTensor<float, 3> a(3, 4, 5); auto inferredShape = a.autoShape({-1, 2, 2, -1}); // inferredShape = [3, 2, 2, 5]
    /// @note 自动推导逻辑，按优先级排序：1、形状相同，返回形状。2、存在一个-1，则在-1维度填充。
    //          3、存在多个-1，表示与原形状对应位置相同，最后一个-1依然是自动填充
    template<typename... Args> std::vector<int> autoShape(const Args... shape) const;
    std::vector<int> autoShape(const std::vector<int>& shape) const;

    /// @brief 对指定轴进行切片操作。
    /// @param atDim 切片的维度索引。
    /// @param start 切片的起始位置（包含）。
    /// @param end 切片的结束位置（不包含）。
    /// @param step 切片的步长。
    /// @param autoFix 如果 start > end，是否自动交换它们。
    /// @return 返回一个新的 YTensorBase 视图，指向原始数据。
    YTensorBase slice(int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true) const;

    /// @brief 对指定轴进行原地切片操作。
    /// @return 返回自身的引用。
    YTensorBase& slice_(int atDim, int start = 0, int end = 0, int step = 1, bool autoFix = true);

    /// @brief 改变张量维度顺序。
    /// @param newOrder 包含新维度顺序的 vector。
    /// @return 返回一个新的 YTensorBase 视图。
    YTensorBase permute(const std::vector<int>& newOrder) const;
    template<typename... Args> YTensorBase permute(const Args... newOrder) const;

    /// @brief 原地改变张量维度顺序
    /// @param newOrder 包含新维度顺序的 vector。
    /// @return 返回自身的引用。
    YTensorBase& permute_(const std::vector<int>& newOrder);

    /// @brief 交换张量的两个维度。
    /// @param dim0, dim1 要交换的两个维度。
    /// @return 返回一个新的 YTensorBase 视图。
    YTensorBase transpose(int dim0 = -2, int dim1 = -1) const;

    /// @brief 创建一个新的张量视图。
    /// @param newShape 新的形状。
    /// @return 返回一个新的 YTensorBase 视图。
    /// @note 要求张量必须是连续的。
    YTensorBase view(const std::vector<int>& newShape) const;
    template<typename... Args> YTensorBase view(const Args... newShape) const;

    /// @brief 重塑张量形状
    /// @param newShape 新的形状
    /// @return 返回一个新的张量（可能是视图，也可能是拷贝）
    /// @note 等价于 contiguous().view(newShape)
    YTensorBase reshape(const std::vector<int>& newShape) const;
    template<typename... Args> YTensorBase reshape(const Args... newShape) const;

    /// @brief 在指定位置插入一个大小为1的维度（零拷贝）
    /// @param dim 插入的位置（支持负索引）
    /// @return 返回一个新的 YTensorBase 视图
    YTensorBase unsqueeze(int dim) const;

    /// @brief 原地在指定位置插入一个大小为1的维度
    /// @param dim 插入的位置（支持负索引）
    /// @return 返回自身的引用
    YTensorBase& unsqueeze_(int dim);

    /// @brief 移除指定位置的大小为1的维度（零拷贝）
    /// @param dim 要移除的维度（支持负索引），如果 dim < 0 则移除所有大小为1的维度
    /// @return 返回一个新的 YTensorBase 视图
    YTensorBase squeeze(int dim = -1) const;

    /// @brief 原地移除指定位置的大小为1的维度
    /// @param dim 要移除的维度（支持负索引），如果 dim < 0 则移除所有大小为1的维度
    /// @return 返回自身的引用
    YTensorBase& squeeze_(int dim = -1);

    /// @brief 沿指定维度重复张量。
    /// @param times 每个维度重复的次数。
    /// @return 返回一个新的 YTensorBase 视图。
    /// @note 只能在大小为 1 的维度上进行重复。
    YTensorBase repeat(const std::vector<int>& times) const;

    /// @brief 沿指定维度重复张量（Args...版本）
    /// @param times 每个维度重复的次数
    /// @return 返回一个新的 YTensorBase 视图。
    template<typename... Args> YTensorBase repeat(const Args... times) const;

    /// @brief 原地沿指定维度重复张量（配合 repeat 的行为）
    /// @param times 每个维度重复的次数。
    /// @return 返回自身的引用。
    YTensorBase& repeat_(const std::vector<int>& times);

    /// @brief 沿维度进行滑动窗口展开。
    /// @param atDim 展开的维度索引。
    /// @param kernel 滑动窗口大小。
    /// @param stride 步长。
    /// @param dilation 膨胀系数。
    /// @return 返回一个新的 YTensorBase 视图。
    YTensorBase unfold(int atDim, int kernel, int stride = 1, int dilation = 1) const;

    /// @brief 原地展开（unfold）的版本，修改自身并返回引用
    YTensorBase& unfold_(int atDim, int kernel, int stride = 1, int dilation = 1);

    /// @brief 获取这个张量尽可能连续的视图，注意张量的形状会发生变化。
    /// @return 返回一个张量，表示当前张量最大可能的连续排布的视图。
    /// @note 这个张量并不能保证连续。
    YTensorBase mostContinuousView() const;

    /// @brief 设置随机数种子，全局共享
    /// @param seed 随机数种子，默认为真随机种子
    /// @example YTensor<>::seed(42);
    static void seed(unsigned int seed = std::random_device{}());

    /// @brief 高级随机生成器：正态分布
    struct _RandnGenerator {
        _RandnGenerator(std::mt19937& gen_p): gen(gen_p) {}
        std::mt19937& gen;
        YTensorBase operator()(const std::vector<int>& shape, std::string dtype = "float32") const;
    };

    /// @brief 高级随机生成器：均匀分布
    struct _RanduGenerator {
        _RanduGenerator(std::mt19937& gen_p): gen(gen_p) {}
        std::mt19937& gen;
        YTensorBase operator()(const std::vector<int>& shape, std::string dtype = "float32") const;
    };

    /// @brief 运行时可用的随机生成器实例
    inline static _RandnGenerator randn{yt::infos::gen};
    inline static _RanduGenerator randu{yt::infos::gen};

    /// @brief 创建指定大小的，零张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensorBase zeros(const std::vector<int>& shape, std::string dtype = "float32");

    /// @brief 创建指定大小的，全1张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensorBase ones(const std::vector<int>& shape, std::string dtype = "float32");

    /// @brief 沿指定轴拼接多个张量
    /// @param tensors 要拼接的张量列表
    /// @param axis 拼接的轴，默认为0
    /// @return 返回拼接后的新张量
    /// @note 所有张量除了拼接轴外，其他维度必须相同
    static YTensorBase concat(const std::vector<YTensorBase>& tensors, int axis = 0);

    /// @brief 沿指定轴拼接两个张量
    /// @param other 要拼接的另一个张量
    /// @param axis 拼接的轴，默认为0
    /// @return 返回拼接后的新张量
    YTensorBase concat(const YTensorBase& other, int axis = 0) const;

    /// @brief 沿指定轴将张量分割成多个部分
    /// @param splitSizes 每个部分的大小列表
    /// @param axis 分割的轴，默认为0
    /// @return 返回分割后的张量列表（视图）
    std::vector<YTensorBase> split(const std::vector<int>& splitSizes, int axis = 0) const;

    /// @brief 沿指定轴将张量平均分割成n份
    /// @param n 分割的份数
    /// @param axis 分割的轴，默认为0
    /// @return 返回分割后的张量列表（视图）
    std::vector<YTensorBase> split(int n, int axis = 0) const;

    /// @brief 获取数据类型字符串（例如 "float32"）
    std::string dtype() const;

    /// @brief 获取元素字节大小
    size_t elementSize() const;

    /// @brief 标准cout输出流
    friend std::ostream& operator<<(std::ostream& os, const YTensorBase& tensor);
/////////////////// externs ////////////////
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

    /// @brief 对两个张量广播二元原地运算 (this op= other)
    /// @tparam T 计算类型 - 作为中间运算类型。this和other的元素会被static_cast<T>转换后运算
    /// @tparam Func 二元运算函数，签名为T func(const T& a, const T& b)。或者void func(T& a, const T& b)。以a是否为const判定。
    /// @param other 另一个张量，会被广播到this的形状
    /// @param func 二元运算函数
    /// @param opName 操作名称，用于错误报告
    /// @param flop 每次运算的浮点运算量估计，用于并行控制
    /// @return 返回自身引用
    /// @note other的形状必须可以广播到this的形状（即other维度<=this维度，且对应维度为1或相等）
    /// @note 结果类型跟随this的dtype（不重新分配内存），运算结果会被转换回this的dtype
    /// @example int32.binaryOpBroadcastInplace<float>(bfloat16, func)
    ///          -> int32 = (float)int32 op (float)bfloat16 转回int32
    template<typename T, typename Func>
    YTensorBase& binaryOpBroadcastInplace(const YTensorBase& other, Func&& func, const std::string& opName = "", double flop = 1.);

    /// @brief 对两个张量广播二元运算 (result = this op other)
    /// @tparam T 计算类型 - 作为中间运算类型，同时也是输出张量的dtype
    /// @tparam Func 二元运算函数，签名为T func(const T& a, const T& b)。也可以是T func(const T& a, const T& b, T& dest)。其中a、b为操作数，dest为结果。
    /// @param other 另一个张量
    /// @param func 二元运算函数
    /// @param opName 操作名称
    /// @param result 可选的结果张量指针，若为nullptr则创建dtype为T的新张量
    /// @param flop 每次运算的浮点运算量估计
    /// @return 返回结果张量（dtype为T）
    /// @note 实现：先将this广播到结果张量（dtype=T），再调用 result.binaryOpBroadcastInplace<T>(other, func, ...)
    /// @example int32.binaryOpBroadcast<float>(bfloat16, func)
    ///          -> 结果float = (float)int32 op (float)bfloat16
    template<typename T, typename Func>
    YTensorBase binaryOpBroadcast(const YTensorBase& other, Func&& func, const std::string& opName = "",
        YTensorBase* result = nullptr, double flop = 1.) const;

    /// @brief 统一的广播原地操作函数，支持N元张量/标量操作
    /// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
    /// @tparam Args 参数类型，可以是YTensorBase或标量T
    /// @param func 操作函数，第一个参数为this的元素引用
    /// @param tensors 输入的张量或标量
    /// @return 返回自身引用
    template<typename Func, typename... Args>
    YTensorBase& broadcastInplace(Func&& func, Args&&... tensors);

    /// @brief 对张量进行逐元素标量原地运算 (this op= scalar)
    /// @tparam T 计算类型 - 作为中间运算类型。this的元素和scalar会被static_cast<T>转换后运算
    /// @tparam Func 二元运算函数，签名为 void func(T& a, const T& b) 或 T func(T a, T b)
    /// @param scalar 标量值（会被转换为T类型）
    /// @param func 二元运算函数
    /// @param flop 每次运算的浮点运算量估计
    /// @return 返回自身引用
    /// @note 结果类型跟随this的dtype（不重新分配内存），运算结果会被转换回this的dtype
    template<typename T, typename Func>
    YTensorBase& binaryOpTransformInplace(const T& scalar, Func&& func, double flop = 1.);

    /// @brief 对张量进行逐元素标量运算 (result = this op scalar)
    /// @tparam T 计算类型 - 作为中间运算类型，同时也是输出张量的dtype
    /// @tparam Func 二元运算函数，签名为 void func(T& a, const T& b) 或 T func(T a, T b)
    /// @param scalar 标量值
    /// @param func 二元运算函数
    /// @param result 可选的结果张量指针，若为nullptr则创建dtype为T的新张量
    /// @param flop 每次运算的浮点运算量估计
    /// @return 返回结果张量（dtype为T）
    /// @note 实现：先克隆this到结果张量（dtype=T），再调用 result.binaryOpTransformInplace<T>(scalar, func, ...)
    template<typename T, typename Func>
    YTensorBase binaryOpTransform(const T& scalar, Func&& func, YTensorBase* result = nullptr, double flop = 1.) const;

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

    #undef YT_YTENSORBASE_OPERATOR_DEF

    /// @brief 对张量的最后两个维度进行广播矩阵乘法运算
    /// @param other 右张量输入，最后两个维度的列数必须与this最后两个维度的行数相等
    /// @return 矩阵乘法结果张量
    /// @note 广播规则应用于除最后两维之外的维度
    YTensorBase matmul(const YTensorBase& other) const;

    /// @brief 矩阵视图，将张量的最后两个维度视为2D矩阵作为"标量"
    /// @return 返回一个scalar类型为YTensorBase的YTensorBase，每个"元素"是一个2D子张量视图
    /// @note 仅支持ndim>=2的张量调用此方法。默认为行主序。
    /// @note 返回的YTensorBase的dtype="YTensorBase"，element_size=sizeof(YTensorBase)
    /// @note 可配合binaryOpBroadcast等函数使用，实现批量矩阵操作
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
    YTensorBase matmul_zero_backend(const YTensorBase& other) const;

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

    public:

    #endif // YT_USE_EIGEN

    public: // end of ytensor_base_math.hpp
protected:
    std::shared_ptr<char[]> _data;  // 存储数据
    int _offset = 0;                // 数据偏移 (以元素为单位)
    std::vector<int> _shape;        // 形状
    std::vector<int> _stride;       // 步长 (以元素为单位)
    size_t _element_size = 0;       // 元素大小（字节）
    std::string _dtype;             // 用于序列化/反序列化友好名称
};

} // namespace yt

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
#include <iomanip>
#include <optional>
#include <functional>
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
/***************
* @file: float_spec.hpp
* @brief: 通用浮点数模板类型，支持任意指数位和尾数位配置。
* @author: SnifferCaptain
* @date: 2025.12.26
* @note 通用浮点数模板生成器，支持ieee754标准下的任意浮点数生成。
***************/

#include <cstdint>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <limits>
#include <iostream>
#include <string>

namespace yt {

// ==================== 存储类型选择 ====================
// 根据总位数选择对齐的存储类型（2的n次方字节）
template <int TotalBits>
struct StorageTypeSelector {
    static_assert(TotalBits > 0 && TotalBits <= 64, "TotalBits must be between 1 and 64");
    using type = std::conditional_t<TotalBits <= 8, uint8_t,
                 std::conditional_t<TotalBits <= 16, uint16_t,
                 std::conditional_t<TotalBits <= 32, uint32_t,
                 uint64_t>>>;
};

// ==================== 计算类型选择 ====================
// 如果 exp_bits <= 8 && mantissa_bits <= 23，使用float；否则使用double
template <int ExpBits, int MantissaBits>
struct ComputeTypeSelector {
    using type = std::conditional_t<(ExpBits <= 8 && MantissaBits <= 23), float, double>;
};

// ==================== FloatSpec 模板类 ====================
/**
 * @brief 通用浮点数模板类
 * @tparam SignBits 符号位数（0或1）
 * @tparam ExpBits 指数位数
 * @tparam MantissaBits 尾数位数
 *
 * 示例：
 * - float16:    FloatSpec<1, 5, 10>
 * - bfloat16:   FloatSpec<1, 8, 7>
 * - float32:    FloatSpec<1, 8, 23>
 * - TF32:       FloatSpec<1, 8, 10>  (存储占用4字节)
 * - float8_e5m2: FloatSpec<1, 5, 2>
 * - float8_e4m3: FloatSpec<1, 4, 3>
 * - float8_e8m0: FloatSpec<0, 8, 0>  (无符号位)
 */
template <int SignBits, int ExpBits, int MantissaBits>
class FloatSpec {
    static_assert(SignBits >= 0 && SignBits <= 1, "SignBits must be 0 or 1");
    static_assert(ExpBits > 0, "ExpBits must be positive");
    static_assert(MantissaBits >= 0, "MantissaBits must be non-negative");

public:
    // 类型常量
    static constexpr int sign_bits = SignBits;
    static constexpr int exp_bits = ExpBits;
    static constexpr int mantissa_bits = MantissaBits;
    static constexpr int total_bits = SignBits + ExpBits + MantissaBits;
    static constexpr int bias = (1 << (ExpBits - 1)) - 1;  // IEEE 754 偏置

    // 存储类型
    using StorageType = typename StorageTypeSelector<total_bits>::type;
    // 计算类型
    using ComputeType = typename ComputeTypeSelector<ExpBits, MantissaBits>::type;

    static constexpr int storage_bits = sizeof(StorageType) * 8;

private:
    StorageType m_bits{0};

    // 掩码
    static constexpr StorageType mantissa_mask = (static_cast<StorageType>(1) << MantissaBits) - 1;
    static constexpr StorageType exp_mask = ((static_cast<StorageType>(1) << ExpBits) - 1) << MantissaBits;
    static constexpr StorageType sign_mask = SignBits ? (static_cast<StorageType>(1) << (ExpBits + MantissaBits)) : 0;

public:
    // ==================== 构造函数 ====================
    FloatSpec() = default;

    // 从计算类型构造
    explicit FloatSpec(ComputeType value) {
        fromComputeType(value);
    }

    // 从任意数值类型构造
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    explicit FloatSpec(T value) : FloatSpec(static_cast<ComputeType>(value)) {}

    // 从位表示构造
    static constexpr FloatSpec from_bits(StorageType bits) {
        FloatSpec f;
        f.m_bits = bits;
        return f;
    }

    // ==================== 类型转换（与 bfloat16 对齐）====================
    // 隐式转换到计算类型（float 或 double）
    operator ComputeType() const {
        return toComputeType();
    }

    // 隐式转换到 float（如果 ComputeType 不是 float）
    template <typename U = ComputeType, typename = std::enable_if_t<!std::is_same_v<U, float>>>
    operator float() const {
        return static_cast<float>(toComputeType());
    }

    // 隐式转换到 double（如果 ComputeType 不是 double）
    template <typename U = ComputeType, typename = std::enable_if_t<!std::is_same_v<U, double>>>
    operator double() const {
        return static_cast<double>(toComputeType());
    }

    // ==================== 与 bfloat16 对齐的方法 ====================
    // to_string: 返回浮点数的字符串表示
    std::string to_string() const {
        return std::to_string(static_cast<double>(toComputeType()));
    }

    // size: 返回存储大小（字节）
    constexpr size_t size() const {
        return sizeof(StorageType);
    }

    // ==================== 位操作 ====================
    StorageType to_bits() const { return m_bits; }

    int sign() const {
        if constexpr (SignBits == 0) return 0;
        else return (m_bits >> (ExpBits + MantissaBits)) & 1;
    }

    int exponent() const {
        return (m_bits >> MantissaBits) & ((1 << ExpBits) - 1);
    }

    StorageType mantissa() const {
        return m_bits & mantissa_mask;
    }

    /// @brief 返回零值
    static constexpr FloatSpec zero() {
        return from_bits(0);
    }

    /// @brief 返回 1.0
    static constexpr FloatSpec one() {
        constexpr StorageType bits = static_cast<StorageType>(bias) << MantissaBits;
        return from_bits(bits);
    }

    /// @brief 返回正无穷大
    static constexpr FloatSpec inf() {
        constexpr StorageType bits = static_cast<StorageType>((1 << ExpBits) - 1) << MantissaBits;
        return from_bits(bits);
    }

    /// @brief 返回 NaN
    static constexpr FloatSpec nan() {
        constexpr StorageType bits =
            (static_cast<StorageType>((1 << ExpBits) - 1) << MantissaBits) | 1;
        return from_bits(bits);
    }

    // ==================== 特殊值检查 ====================
    bool isZero() const {
        if constexpr (SignBits == 0) {
            return (m_bits & (exp_mask | mantissa_mask)) == 0;
        } else {
            return (m_bits & ~sign_mask) == 0;
        }
    }

    bool isInf() const {
        return exponent() == ((1 << ExpBits) - 1) && mantissa() == 0;
    }

    bool isNaN() const {
        return exponent() == ((1 << ExpBits) - 1) && mantissa() != 0;
    }

    bool isDenormal() const {
        return exponent() == 0 && mantissa() != 0;
    }

    // ==================== 比较运算符 ====================
    template <typename T>
    bool operator==(const T& other) const {
        return toComputeType() == static_cast<ComputeType>(other);
    }

    template <typename T>
    bool operator!=(const T& other) const { return !(*this == other); }

    template <typename T>
    bool operator<(const T& other) const {
        return toComputeType() < static_cast<ComputeType>(other);
    }

    template <typename T>
    bool operator<=(const T& other) const {
        return toComputeType() <= static_cast<ComputeType>(other);
    }

    template <typename T>
    bool operator>(const T& other) const {
        return toComputeType() > static_cast<ComputeType>(other);
    }

    template <typename T>
    bool operator>=(const T& other) const {
        return toComputeType() >= static_cast<ComputeType>(other);
    }

    // ==================== 算术运算符 ====================
    template <typename T>
    FloatSpec operator+(const T& other) const {
        return FloatSpec(toComputeType() + static_cast<ComputeType>(other));
    }

    template <typename T>
    FloatSpec& operator+=(const T& other) {
        fromComputeType(toComputeType() + static_cast<ComputeType>(other));
        return *this;
    }

    template <typename T>
    FloatSpec operator-(const T& other) const {
        return FloatSpec(toComputeType() - static_cast<ComputeType>(other));
    }

    template <typename T>
    FloatSpec& operator-=(const T& other) {
        fromComputeType(toComputeType() - static_cast<ComputeType>(other));
        return *this;
    }

    template <typename T>
    FloatSpec operator*(const T& other) const {
        return FloatSpec(toComputeType() * static_cast<ComputeType>(other));
    }

    template <typename T>
    FloatSpec& operator*=(const T& other) {
        fromComputeType(toComputeType() * static_cast<ComputeType>(other));
        return *this;
    }

    template <typename T>
    FloatSpec operator/(const T& other) const {
        return FloatSpec(toComputeType() / static_cast<ComputeType>(other));
    }

    template <typename T>
    FloatSpec& operator/=(const T& other) {
        fromComputeType(toComputeType() / static_cast<ComputeType>(other));
        return *this;
    }

    FloatSpec operator-() const {
        if constexpr (SignBits == 0) {
            return FloatSpec(-toComputeType());
        } else {
            return from_bits(m_bits ^ sign_mask);
        }
    }

    // ==================== 输出 ====================
    friend std::ostream& operator<<(std::ostream& out, const FloatSpec& f) {
        out << static_cast<double>(f.toComputeType());
        return out;
    }

private:
    // ==================== 内部转换函数 ====================
    ComputeType toComputeType() const {
        if constexpr (std::is_same_v<ComputeType, float>) {
            return toFloat32();
        } else {
            return toFloat64();
        }
    }

    void fromComputeType(ComputeType value) {
        if constexpr (std::is_same_v<ComputeType, float>) {
            fromFloat32(value);
        } else {
            fromFloat64(value);
        }
    }

    float toFloat32() const {
        // 提取各部分
        int s = sign();
        int e = exponent();
        uint32_t m = mantissa();

        uint32_t result;

        if (e == 0) {
            if (m == 0) {
                // 零
                result = static_cast<uint32_t>(s) << 31;
            } else {
                // 非规格化数：转换为规格化数
                // 对于MantissaBits=0的情况，非规格化数就是0
                if constexpr (MantissaBits == 0) {
                    result = static_cast<uint32_t>(s) << 31;
                } else {
                    // 找到最高位1的位置并规格化
                    int shift = 0;
                    while ((m & (static_cast<uint32_t>(1) << (MantissaBits - 1))) == 0 && shift < MantissaBits) {
                        m <<= 1;
                        shift++;
                    }
                    m = (m << 1) & mantissa_mask;  // 移除隐含的1
                    e = 1 - shift;

                    // 转换指数到float32的偏置
                    int newExp = e - bias + 127;
                    if (newExp <= 0) {
                        // 仍然是非规格化数
                        result = (static_cast<uint32_t>(s) << 31) |
                                 (static_cast<uint32_t>(m) << (23 - MantissaBits));
                    } else {
                        result = (static_cast<uint32_t>(s) << 31) |
                             (static_cast<uint32_t>(newExp) << 23) |
                             (static_cast<uint32_t>(m) << (23 - MantissaBits));
                    }
                }
            }
        } else if (e == ((1 << ExpBits) - 1)) {
            // 无穷大或NaN
            result = (static_cast<uint32_t>(s) << 31) |
                     (static_cast<uint32_t>(0xFF) << 23) |
                     (m != 0 ? 1 : 0);  // 保留NaN标志
        } else {
            // 规格化数
            int newExp = e - bias + 127;
            if (newExp <= 0) {
                // 下溢到非规格化数
                result = static_cast<uint32_t>(s) << 31;
            } else if (newExp >= 255) {
                // 上溢到无穷大
                result = (static_cast<uint32_t>(s) << 31) | (static_cast<uint32_t>(0xFF) << 23);
            } else {
                // 正常转换
                uint32_t newMantissa;
                if constexpr (MantissaBits <= 23) {
                    newMantissa = static_cast<uint32_t>(m) << (23 - MantissaBits);
                } else {
                    newMantissa = static_cast<uint32_t>(m >> (MantissaBits - 23));
                }
                result = (static_cast<uint32_t>(s) << 31) |
                         (static_cast<uint32_t>(newExp) << 23) |
                         newMantissa;
            }
        }

        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }

    double toFloat64() const {
        // 类似toFloat32，但目标是float64
        int s = sign();
        int e = exponent();
        uint64_t m = mantissa();

        uint64_t result;

        if (e == 0 && m == 0) {
            // 零
            result = static_cast<uint64_t>(s) << 63;
        } else if (e == ((1 << ExpBits) - 1)) {
            // 无穷大或NaN
            result = (static_cast<uint64_t>(s) << 63) |
                     (static_cast<uint64_t>(0x7FF) << 52) |
                     (m != 0 ? 1 : 0);
        } else if (e == 0) {
            // 非规格化数转换
            int shift = 0;
            while ((m & (static_cast<uint64_t>(1) << (MantissaBits - 1))) == 0 && shift < MantissaBits) {
                m <<= 1;
                shift++;
            }
            int newExp = 1 - shift - bias + 1023;
            if (newExp > 0) {
                m = (m << 1) & ((static_cast<uint64_t>(1) << MantissaBits) - 1);
                uint64_t newMantissa;
                if constexpr (MantissaBits <= 52) {
                    newMantissa = m << (52 - MantissaBits);
                } else {
                    newMantissa = m >> (MantissaBits - 52);
                }
                result = (static_cast<uint64_t>(s) << 63) |
                         (static_cast<uint64_t>(newExp) << 52) |
                         newMantissa;
            } else {
                result = static_cast<uint64_t>(s) << 63;
            }
        } else {
            // 规格化数
            int newExp = e - bias + 1023;
            if (newExp <= 0) {
                result = static_cast<uint64_t>(s) << 63;
            } else if (newExp >= 2047) {
                result = (static_cast<uint64_t>(s) << 63) | (static_cast<uint64_t>(0x7FF) << 52);
            } else {
                uint64_t newMantissa;
                if constexpr (MantissaBits <= 52) {
                    newMantissa = static_cast<uint64_t>(m) << (52 - MantissaBits);
                } else {
                    newMantissa = static_cast<uint64_t>(m >> (MantissaBits - 52));
                }
                result = (static_cast<uint64_t>(s) << 63) |
                         (static_cast<uint64_t>(newExp) << 52) |
                         newMantissa;
            }
        }

        double d;
        std::memcpy(&d, &result, sizeof(double));
        return d;
    }

    void fromFloat32(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));

        uint32_t s = (bits >> 31) & 1;
        uint32_t e = (bits >> 23) & 0xFF;
        uint32_t m = bits & 0x7FFFFF;

        StorageType newSign = SignBits ? static_cast<StorageType>(s) : 0;
        StorageType newExp, newMantissa;

        if (e == 0) {
            // 零或非规格化数
            newExp = 0;
            newMantissa = 0;
        } else if (e == 0xFF) {
            // 无穷大或NaN
            newExp = (1 << ExpBits) - 1;
            newMantissa = m != 0 ? 1 : 0;
        } else {
            // 规格化数
            int realExp = static_cast<int>(e) - 127 + bias;
            if (realExp <= 0) {
                // 下溢
                newExp = 0;
                newMantissa = 0;
            } else if (realExp >= (1 << ExpBits) - 1) {
                // 上溢
                newExp = (1 << ExpBits) - 1;
                newMantissa = 0;
            } else {
                newExp = static_cast<StorageType>(realExp);
                if constexpr (MantissaBits <= 23) {
                    // 截断高位
                    newMantissa = static_cast<StorageType>(m >> (23 - MantissaBits));
                    // 四舍五入到最近偶数
                    uint32_t remainder = m & ((1 << (23 - MantissaBits)) - 1);
                    uint32_t halfBit = 1 << (22 - MantissaBits);
                    if (remainder > halfBit || (remainder == halfBit && (newMantissa & 1))) {
                        newMantissa++;
                        if (newMantissa >= (static_cast<StorageType>(1) << MantissaBits)) {
                            newMantissa = 0;
                            newExp++;
                            if (newExp >= (1 << ExpBits) - 1) {
                                newExp = (1 << ExpBits) - 1;
                                newMantissa = 0;
                            }
                        }
                    }
                } else {
                    newMantissa = static_cast<StorageType>(m) << (MantissaBits - 23);
                }
            }
        }

        m_bits = (newSign << (ExpBits + MantissaBits)) | (newExp << MantissaBits) | newMantissa;
    }

    void fromFloat64(double value) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(double));

        uint64_t s = (bits >> 63) & 1;
        uint64_t e = (bits >> 52) & 0x7FF;
        uint64_t m = bits & 0xFFFFFFFFFFFFF;

        StorageType newSign = SignBits ? static_cast<StorageType>(s) : 0;
        StorageType newExp, newMantissa;

        if (e == 0) {
            newExp = 0;
            newMantissa = 0;
        } else if (e == 0x7FF) {
            newExp = (1 << ExpBits) - 1;
            newMantissa = m != 0 ? 1 : 0;
        } else {
            int realExp = static_cast<int>(e) - 1023 + bias;
            if (realExp <= 0) {
                newExp = 0;
                newMantissa = 0;
            } else if (realExp >= (1 << ExpBits) - 1) {
                newExp = (1 << ExpBits) - 1;
                newMantissa = 0;
            } else {
                newExp = static_cast<StorageType>(realExp);
                if constexpr (MantissaBits <= 52) {
                    newMantissa = static_cast<StorageType>(m >> (52 - MantissaBits));
                } else {
                    newMantissa = static_cast<StorageType>(m) << (MantissaBits - 52);
                }
            }
        }

        m_bits = (newSign << (ExpBits + MantissaBits)) | (newExp << MantissaBits) | newMantissa;
    }
};

// ==================== 类型别名 ====================
using float16 = FloatSpec<1, 5, 10>;       // IEEE 754 half precision
using float8_e5m2 = FloatSpec<1, 5, 2>;    // E5M2 format (8-bit float, 1 sign, 5 exp, 2 mantissa)
using float8_e4m3 = FloatSpec<1, 4, 3>;    // E4M3 format (8-bit float, 1 sign, 4 exp, 3 mantissa)
using float8_e8m0 = FloatSpec<0, 8, 0>;    // E8M0 format (unsigned, 8-bit exponent only, no mantissa)
using float8_ue8m0 = FloatSpec<0, 8, 0>;   // 别名: unsigned E8M0

} // namespace yt

// ==================== std::numeric_limits 特化 ====================
namespace std {

template <int S, int E, int M>
class numeric_limits<yt::FloatSpec<S, E, M>> {
    using F = yt::FloatSpec<S, E, M>;
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = (S == 1);
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm = true;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = M + 1;
    static constexpr int digits10 = static_cast<int>((M + 1) * 0.30103);
    static constexpr int max_digits10 = static_cast<int>((M + 1) * 0.30103) + 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 2 - ((1 << (E - 1)) - 1);
    static constexpr int min_exponent10 = static_cast<int>(min_exponent * 0.30103);
    static constexpr int max_exponent = (1 << (E - 1));
    static constexpr int max_exponent10 = static_cast<int>(max_exponent * 0.30103);

    static constexpr F min() noexcept {
        // 最小正规格化数
        return F::from_bits(static_cast<typename F::StorageType>(1) << M);
    }
    static constexpr F max() noexcept {
        // 最大有限数
        constexpr typename F::StorageType maxExp = ((1 << E) - 2);
        constexpr typename F::StorageType maxMant = (static_cast<typename F::StorageType>(1) << M) - 1;
        return F::from_bits((maxExp << M) | maxMant);
    }
    static constexpr F lowest() noexcept {
        if constexpr (S == 0) return F::from_bits(0);
        else return F::from_bits(max().to_bits() | (static_cast<typename F::StorageType>(1) << (E + M)));
    }
    static constexpr F epsilon() noexcept {
        // 1.0和下一个可表示数之间的差
        constexpr int bias = (1 << (E - 1)) - 1;
        constexpr int eps_exp = bias - M;
        if constexpr (eps_exp > 0) {
            return F::from_bits(static_cast<typename F::StorageType>(eps_exp) << M);
        } else {
            return F::from_bits(1);  // 非规格化最小
        }
    }
    static constexpr F round_error() noexcept { return F(0.5f); }
    static constexpr F infinity() noexcept { return F::inf(); }
    static constexpr F quiet_NaN() noexcept { return F::nan(); }
    static constexpr F signaling_NaN() noexcept { return F::nan(); }
    static constexpr F denorm_min() noexcept { return F::from_bits(1); }
};

} // namespace std

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
            return it->second.name;
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
        else if constexpr (std::is_same_v<T, yt::float16>) return "float16";
        else if constexpr (std::is_same_v<T, yt::float8_e5m2>) return "float8_e5m2";
        else if constexpr (std::is_same_v<T, yt::float8_e4m3>) return "float8_e4m3";
        else if constexpr (std::is_same_v<T, yt::float8_e8m0>) return "float8_e8m0";
        else if constexpr (std::is_same_v<T, yt::float8_ue8m0>) return "float8_ue8m0";
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
        else if (typeName == "float16") return 2;
        else if (typeName == "float8_e5m2") return 1;
        else if (typeName == "float8_e4m3") return 1;
        else if (typeName == "float8_e8m0" || typeName == "float8_ue8m0") return 1;
        else {
            // registered custom types
            auto& registry = yt::infos::getTypeRegistry();// std::unordered_map<std::string, std::pair<std::string, int32_t>>
            for(auto& [key, value] : registry) {
                if (value.name == typeName) {
                    return value.size;
                }
            }
            // unk
            throw std::runtime_error(std::string("Type ") + typeName + " is not registered.");
            return 0;
        }
    }

    /// @brief 根据类型名称获取类型注册信息
    /// @param typeName 类型名称
    /// @return 类型注册信息的optional引用，未找到返回std::nullopt
    inline std::optional<std::reference_wrapper<const yt::infos::TypeRegItem>> getTypeInfo(const std::string& typeName) {
        auto& registry = yt::infos::getTypeRegistry();
        for (auto& [key, value] : registry) {
            if (value.name == typeName) {
                return std::cref(value);
            }
        }
        return std::nullopt;  // 内置类型或未注册类型
    }

    /// @brief 检查类型是否为POD（或内置类型）
    /// @param typeName 类型名称
    /// @return true=POD类型，不需要特殊析构处理
    inline bool isPODType(const std::string& typeName) {
        // 内置类型都是POD
        if (typeName == "float32" || typeName == "float64" ||
            typeName == "int8" || typeName == "int16" || typeName == "int32" || typeName == "int64" ||
            typeName == "uint8" || typeName == "uint16" || typeName == "uint32" || typeName == "uint64" ||
            typeName == "bool" || typeName == "bfloat16" ||
            typeName == "float16" || typeName == "float8_e5m2" ||
            typeName == "float8_e4m3" || typeName == "float8_e8m0" || typeName == "float8_ue8m0") {
            return true;
        }
        // 检查注册的自定义类型
        auto info = getTypeInfo(typeName);
        return info ? info->get().isPOD : true;  // 未知类型假设为POD
    }

    /// @brief 注册自定义类型
    /// @tparam T 要注册的类型
    /// @param typeName 自定义类型名称
    template<typename T>
    void registerType(const std::string& typeName) {
        auto& registry = yt::infos::getTypeRegistry();
        int32_t typeSize = getTypeSize<T>();
        // default formatter: if type has operator<< then use that, else nullptr
        auto makeDefaultFormatter = []() -> std::function<std::string(const void*)> {
            if constexpr (yt::concepts::HAVE_OSTREAM<T>) {
                return [](const void* data) {
                    std::ostringstream oss;
                    const T* p = reinterpret_cast<const T*>(data);
                    oss << *p;
                    return oss.str();
                };
            } else {
                return nullptr;
            }
        };

        // 非POD类型支持
        yt::infos::TypeRegItem item;
        item.name = typeName;
        item.size = typeSize;
        item.toString = makeDefaultFormatter();
        item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

        if (!item.isPOD) {
            // 析构函数
            item.destructor = [](void* ptr) {
                reinterpret_cast<T*>(ptr)->~T();
            };
            // 拷贝构造
            item.copyConstruct = [](void* dest, const void* src) {
                new (dest) T(*reinterpret_cast<const T*>(src));
            };
            // 默认构造
            if constexpr (std::is_default_constructible_v<T>) {
                item.defaultConstruct = [](void* dest) {
                    new (dest) T();
                };
            }
        }

        registry[typeid(T).name()] = std::move(item);
    }

    /// @brief registerType overload that accepts an explicit formatter function
    template<typename T>
    void registerType(const std::string& typeName, std::function<std::string(const void*)> formatter) {
        auto& registry = yt::infos::getTypeRegistry();
        int32_t typeSize = getTypeSize<T>();
        if (!formatter) {
            if constexpr (yt::concepts::HAVE_OSTREAM<T>) {
                formatter = [](const void* data) {
                    std::ostringstream oss;
                    const T* p = reinterpret_cast<const T*>(data);
                    oss << *p;
                    return oss.str();
                };
            } else {
                throw std::invalid_argument("Formatter function cannot be null for type without ostream support.");
            }
        }

        // 非POD类型支持
        yt::infos::TypeRegItem item;
        item.name = typeName;
        item.size = typeSize;
        item.toString = formatter;
        item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

        if (!item.isPOD) {
            item.destructor = [](void* ptr) {
                reinterpret_cast<T*>(ptr)->~T();
            };
            item.copyConstruct = [](void* dest, const void* src) {
                new (dest) T(*reinterpret_cast<const T*>(src));
            };
            if constexpr (std::is_default_constructible_v<T>) {
                item.defaultConstruct = [](void* dest) {
                    new (dest) T();
                };
            }
        }

        registry[typeid(T).name()] = std::move(item);
    }

    /// @brief 将任意 dtype 的单个元素（原始数据指针）格式化为字符串，用于打印
    /// @param data 指向元素起始位置的原始指针
    /// @param dtype 元素类型名称（如 "float32"）
    /// @return 返回格式化后的字符串
    inline std::string formatValue(const void* data, const std::string& dtype) {
        if (!data) return std::string("null");
        std::ostringstream oss;
        // use default formatting; decide casting based on dtype
        if (dtype == "float32") {
            const float* p = reinterpret_cast<const float*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "int32") {
            const int32_t* p = reinterpret_cast<const int32_t*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "int64") {
            const int64_t* p = reinterpret_cast<const int64_t*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "int16") {
            const int16_t* p = reinterpret_cast<const int16_t*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "int8") {
            const int8_t* p = reinterpret_cast<const int8_t*>(data);
            // print numeric, not character
            oss << static_cast<int>(*p);
            return oss.str();
        } else if (dtype == "uint8") {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            oss << static_cast<unsigned int>(*p);
            return oss.str();
        } else if (dtype == "uint16") {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "uint32") {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(data);
            oss << *p;
            return oss.str();
        } else if (dtype == "bool") {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
            oss << ((*p) ? "true" : "false");
            return oss.str();
        } else if (dtype == "bfloat16") {
            const yt::bfloat16* p = reinterpret_cast<const yt::bfloat16*>(data);
            oss << static_cast<float>(*p);
            return oss.str();
        }
        else if (dtype == "float16") {
            const yt::float16* p = reinterpret_cast<const yt::float16*>(data);
            oss << static_cast<float>(*p);
            return oss.str();
        } else if (dtype == "float8_e5m2") {
            const yt::float8_e5m2* p = reinterpret_cast<const yt::float8_e5m2*>(data);
            oss << static_cast<float>(*p);
            return oss.str();
        } else if (dtype == "float8_e4m3") {
            const yt::float8_e4m3* p = reinterpret_cast<const yt::float8_e4m3*>(data);
            oss << static_cast<float>(*p);
            return oss.str();
        } else if (dtype == "float8_e8m0" || dtype == "float8_ue8m0") {
            const yt::float8_e8m0* p = reinterpret_cast<const yt::float8_e8m0*>(data);
            oss << static_cast<float>(*p);
            return oss.str();
        }
        // 查看自定义注册类型
        auto& registry = yt::infos::getTypeRegistry();
        for (auto& [key, value] : registry) {
            if (value.name == dtype) {
                if (value.toString) {
                    return value.toString(data);
                }
                break; // no formatter, fallback
            }
        }
        // fallback打印单字节
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
        oss << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*bytes);
        return oss.str();
    }
} // namespace yt::types

namespace yt{

inline std::vector<int> YTensorBase::shape() const {
    return _shape;
}

inline YTensorBase::YTensorBase(const std::vector<int>& shape, const std::string& dtype) {
    _shape = shape;
    _offset = 0;
    int d = ndim();
    _stride.assign(d, 0);
    if (d > 0) {
        _stride[d - 1] = 1;
        for (int i = d - 2; i >= 0; --i) {
            _stride[i] = _stride[i + 1] * _shape[i + 1];
        }
    }
    _dtype = dtype;
    _element_size = static_cast<size_t>(yt::types::getTypeSize(dtype));
    // allocate contiguous storage for given shape
    size_t total = 1;
    for (int v : _shape) total *= std::max(0, v);
    if (total == 0) total = 1;

    // 检查是否为非POD类型
    auto typeInfoOpt = yt::types::getTypeInfo(dtype);
    if (typeInfoOpt && !typeInfoOpt->get().isPOD) {
        // 非POD类型：需要使用自定义删除器
        const auto& typeInfo = typeInfoOpt->get();
        size_t elemSize = _element_size;
        size_t numElems = total;
        auto destructor = typeInfo.destructor;
        auto defaultConstruct = typeInfo.defaultConstruct;

        // 分配内存
        char* rawPtr = new char[total * _element_size];

        // 调用默认构造函数（如果有）
        if (defaultConstruct) {
            for (size_t i = 0; i < numElems; ++i) {
                defaultConstruct(rawPtr + i * elemSize);
            }
        }

        // 使用自定义删除器
        _data = std::shared_ptr<char[]>(rawPtr, [destructor, elemSize, numElems](char* ptr) {
            if (destructor) {
                for (size_t i = 0; i < numElems; ++i) {
                    destructor(ptr + i * elemSize);
                }
            }
            delete[] ptr;
        });
    } else {
        // POD类型：简单分配即可
        _data = std::shared_ptr<char[]>(new char[total * _element_size]);
    }
    _offset = 0;
}

inline YTensorBase::YTensorBase(const YTensorBase& other) {
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;
    _data = other._data;
    _element_size = other._element_size;
    _dtype = other._dtype;
}

inline YTensorBase& YTensorBase::operator=(const YTensorBase& other) {
    if (this != &other) {
        _shape = other._shape;
        _stride = other._stride;
        _offset = other._offset;
        _data = other._data;
        _element_size = other._element_size;
        _dtype = other._dtype;
    }
    return *this;
}

inline int YTensorBase::shape(int atDim) const {
    // 循环索引，与 YTensor::shape 相同
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::shape] Cannot access shape of a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d; // 循环索引
    return _shape[atDim];
}

inline std::vector<int> YTensorBase::stride() const {
    std::vector<int> op(ndim());
    int d = ndim();
    if (d > 0) {
        op[d - 1] = 1;
        for (int i = d - 2; i >= 0; --i) {
            op[i] = op[i + 1] * _shape[i + 1];
        }
    }
    return op;
}

inline std::vector<int> YTensorBase::stride_() const {
    return _stride;
}

inline int YTensorBase::stride_(int atDim) const {
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::stride_] Cannot access stride of a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d;
    return _stride[atDim];
}

inline int YTensorBase::stride(int atDim) const {
    // 循环索引处理，与 YTensor::stride 保持一致
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::stride] Cannot access stride of a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d; // 循环索引
    // 直接返回物理步长数组中的值
    return _stride[atDim];
}

inline size_t YTensorBase::size() const {
    size_t total_size = 1;
    for (int i = 0; i < ndim(); ++i) {
        total_size *= _shape[i];
    }
    return total_size;
}

inline int YTensorBase::ndim() const {
    return static_cast<int>(_shape.size());
}

inline int YTensorBase::shapeSize() const {
    return ndim();
}

// element_size() and dtype() are intentionally not implemented here; they are
// not part of the YTensor interface that YTensorBase mirrors.

template <typename... Args>
inline int YTensorBase::offset(Args... index) const {
    static_assert(sizeof...(index) <= 0 || sizeof...(index) >= 0, "offset template forwarded to toIndex_");
    // zero-pad / forward behavior is provided in YTensor; here forward to vector overload
    std::vector<int> indices = {index...};
    return this->toIndex_(indices);
}

inline int YTensorBase::offset(const std::vector<int>& index) const {
    // forward to toIndex_
    return static_cast<int>(this->toIndex_(index));
}

template <typename... Args>
inline int YTensorBase::offset_(Args... index) const {
    return _offset + offset(index...);
}

inline int YTensorBase::offset_(const std::vector<int>& index) const {
    return _offset + offset(index);
}

// 数据访问：提供模板化的按类型访问，以及 float 的便捷重载
#include <cstring>

template <typename T>
inline T* YTensorBase::data() {
    if (!_data) return nullptr;
    // _data 是 char[]，按元素类型 T 返回指针
    return reinterpret_cast<T*>(_data.get()) + _offset;
}

template <typename T>
inline const T* YTensorBase::data() const {
    if (!_data) return nullptr;
    return reinterpret_cast<const T*>(_data.get()) + _offset;
}

inline float* YTensorBase::data() {
    return data<float>();
}

inline const float* YTensorBase::data() const {
    return data<float>();
}

inline bool YTensorBase::shapeMatch(const std::vector<int> &otherShape) const {
    if (static_cast<int>(otherShape.size()) != ndim()) return false;
    if (_shape.size() != otherShape.size()) return false;
    for (int i = 0; i < ndim(); ++i) {
        if (_shape[i] != otherShape[i]) return false;
    }
    return true;
}

inline void YTensorBase::shallowCopyTo(YTensorBase &other) const {
    other._shape = _shape;
    other._stride = this->stride();
    other._offset = _offset;
    other._data = _data;
    other._element_size = _element_size;
    other._dtype = _dtype;
}

inline YTensorBase YTensorBase::clone() const {
    YTensorBase op;
    op._shape = _shape;
    op._dtype = _dtype;
    op._element_size = _element_size;
    op._offset = 0;

    // 计算连续排布的stride
    int ndim = static_cast<int>(_shape.size());
    op._stride.resize(ndim);
    if (ndim > 0) {
        op._stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            op._stride[i] = op._stride[i + 1] * _shape[i + 1];
        }
    }

    // 计算总元素数
    size_t total = 1;
    for (int v : _shape) total *= std::max(0, v);
    if (total == 0) total = 1;

    // 检查是否为非POD类型
    auto typeInfoOpt = yt::types::getTypeInfo(_dtype);
    size_t elemSize = _element_size;

    if (typeInfoOpt && !typeInfoOpt->get().isPOD) {
        // 非POD类型：需要使用拷贝构造和自定义删除器
        const auto& typeInfo = typeInfoOpt->get();
        size_t numElems = total;
        auto destructor = typeInfo.destructor;
        auto copyConstruct = typeInfo.copyConstruct;

        // 分配内存
        char* rawPtr = new char[total * elemSize];

        // 调用拷贝构造函数（处理非连续情况）
        if (copyConstruct && _data) {
            for (size_t dst = 0; dst < numElems; ++dst) {
                // 计算源张量中对应元素的物理位置
                size_t srcIndex = _offset;
                size_t index = dst;
                for (int i = ndim - 1; i >= 0; --i) {
                    srcIndex += (index % _shape[i]) * _stride[i];
                    index /= _shape[i];
                }
                copyConstruct(rawPtr + dst * elemSize, _data.get() + srcIndex * elemSize);
            }
        }

        // 使用自定义删除器
        op._data = std::shared_ptr<char[]>(rawPtr, [destructor, elemSize, numElems](char* ptr) {
            if (destructor) {
                for (size_t i = 0; i < numElems; ++i) {
                    destructor(ptr + i * elemSize);
                }
            }
            delete[] ptr;
        });
    } else {
        // POD类型
        op._data = std::shared_ptr<char[]>(new char[total * elemSize]);
        if (_data) {
            // 检查是否连续
            if (isContiguous()) {
                // 连续：直接memcpy
                std::memcpy(op._data.get(), _data.get() + _offset * elemSize, total * elemSize);
            } else {
                // 非连续：逐元素复制
                char* dstPtr = op._data.get();
                const char* srcBase = _data.get();
                for (size_t dst = 0; dst < total; ++dst) {
                    // 计算源张量中对应元素的物理位置
                    size_t srcIndex = _offset;
                    size_t index = dst;
                    for (int i = ndim - 1; i >= 0; --i) {
                        srcIndex += (index % _shape[i]) * _stride[i];
                        index /= _shape[i];
                    }
                    std::memcpy(dstPtr + dst * elemSize, srcBase + srcIndex * elemSize, elemSize);
                }
            }
        }
    }
    return op;
}

inline YTensorBase& YTensorBase::copy_(const YTensorBase& src) {
    // 验证shape一致
    if (!this->shapeMatch(src.shape())) {
        throw std::runtime_error("copy_: source and destination shapes must match");
    }
    // 验证dtype一致
    if (this->_dtype != src._dtype) {
        throw std::runtime_error("copy_: source and destination dtypes must match");
    }

    size_t elemSize = _element_size;
    int d = ndim();
    size_t total = size();

    // 如果两者都是完全连续的，直接memcpy
    if (this->isContiguous() && src.isContiguous()) {
        std::memcpy(_data.get() + _offset * elemSize,
                    src._data.get() + src._offset * elemSize,
                    total * elemSize);
        return *this;
    }

    // 找到从哪个维度开始两者都是连续的
    int dstCFrom = this->isContiguousFrom();
    int srcCFrom = src.isContiguousFrom();

    // 检查两者stride是否完全匹配
    bool strideMatch = (d == src.ndim());
    for (int i = 0; i < d && strideMatch; ++i) {
        if (_stride[i] != src._stride[i]) {
            strideMatch = false;
        }
    }

    // 只有当两者从相同维度开始都连续，且连续部分的stride也匹配时，才能优化
    if (dstCFrom == srcCFrom && dstCFrom < d) {
        // 检查从dstCFrom开始的stride是否匹配
        bool contiguousMatch = true;
        for (int i = dstCFrom; i < d && contiguousMatch; ++i) {
            if (_stride[i] != src._stride[i]) {
                contiguousMatch = false;
            }
        }

        if (contiguousMatch) {
            // 连续部分stride匹配，可以分块memcpy
            size_t contiguousSize = 1;
            for (int i = dstCFrom; i < d; i++) {
                contiguousSize *= _shape[i];
            }

            size_t outerSize = 1;
            for (int i = 0; i < dstCFrom; i++) {
                outerSize *= _shape[i];
            }

            char* dstBasePtr = _data.get();
            const char* srcBasePtr = src._data.get();

            #pragma omp parallel for if(outerSize > 64)
            for (size_t outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                // 计算非连续部分的坐标
                std::vector<int> outerCoord(dstCFrom);
                size_t remaining = outerIdx;
                for (int i = dstCFrom - 1; i >= 0; i--) {
                    outerCoord[i] = remaining % _shape[i];
                    remaining /= _shape[i];
                }

                // 计算dst的偏移
                size_t dstOffset = _offset;
                for (int i = 0; i < dstCFrom; i++) {
                    dstOffset += outerCoord[i] * _stride[i];
                }

                // 计算src的偏移
                size_t srcOffset = src._offset;
                for (int i = 0; i < dstCFrom; i++) {
                    srcOffset += outerCoord[i] * src._stride[i];
                }

                // 复制连续部分
                std::memcpy(dstBasePtr + dstOffset * elemSize,
                            srcBasePtr + srcOffset * elemSize,
                            contiguousSize * elemSize);
            }

            return *this;
        }
    }

    // 通用情况：逐元素复制
    auto thisLogicStride = this->stride();
    auto srcLogicStride = src.stride();

    char* dstBasePtr = _data.get();
    const char* srcBasePtr = src._data.get();

    #pragma omp parallel for if(total > 1024)
    for (size_t index = 0; index < total; ++index) {
        // 计算逻辑坐标
        std::vector<int> coord(d);
        size_t remaining = index;
        for (int i = 0; i < d; i++) {
            coord[i] = (remaining / thisLogicStride[i]) % _shape[i];
        }

        // 计算dst的物理索引
        size_t dstIndex = _offset;
        for (int i = 0; i < d; i++) {
            dstIndex += coord[i] * _stride[i];
        }

        // 计算src的物理索引
        size_t srcIndex = src._offset;
        for (int i = 0; i < d; i++) {
            srcIndex += coord[i] * src._stride[i];
        }

        std::memcpy(dstBasePtr + dstIndex * elemSize,
                    srcBasePtr + srcIndex * elemSize,
                    elemSize);
    }

    return *this;
}

inline std::ostream &operator<<(std::ostream &out, const YTensorBase &tensor){
    out << "[YTensorBase]:<" << tensor.dtype() << ">" << std::endl;
    out << "[itemSize]: " << tensor.size() << std::endl;
    out << "[byteSize]: " << tensor.size() * tensor.elementSize() << std::endl;
    out << "[shape]: [";
    for (int i = 0; i < tensor.ndim(); ++i){
        out << tensor.shape(i) << (i + 1 == tensor.ndim() ? "" : ", ");
    }
    out << "]" << std::endl;
    out << "[data]:" << std::endl;

    // Print data using runtime dtype and a centralized formatting helper
    std::vector<int> dims = tensor.shape();
    if (dims.size() == 0) {
        // scalar case
        if (!tensor._data) {
            out << "[data]: null" << std::endl;
        } else {
            size_t phys = 0; // scalar
            size_t addressIndex = static_cast<size_t>(tensor._offset) + phys;
            const void* valPtr = static_cast<const void*>(tensor._data.get() + addressIndex * tensor.elementSize());
            out << yt::types::formatValue(valPtr, tensor.dtype());
        }
    } else {
        std::function<void(std::vector<int>&, int, int)> printRecursive;
        printRecursive = [&](std::vector<int>& indices, int currentDim, int indent) {
            for (int i = 0; i < indent; ++i) out << "  ";
            if (currentDim == static_cast<int>(dims.size()) - 1) {
                out << "[";
                for (int i = 0; i < dims[currentDim]; ++i) {
                    indices[currentDim] = i;
                    try {
                        size_t phys = tensor.toIndex_(indices);
                        size_t addressIndex = static_cast<size_t>(tensor._offset) + phys;
                        const void* valPtr = static_cast<const void*>(tensor._data.get() + addressIndex * tensor.elementSize());
                        out << yt::types::formatValue(valPtr, tensor.dtype());
                    } catch (...) {
                        out << "...";
                    }
                    if (i < dims[currentDim] - 1) out << " ";
                }
                out << "]";
                if (dims.size() < 1) out << std::endl;
            } else {
                out << "[" << std::endl;
                for (int i = 0; i < dims[currentDim]; ++i) {
                    indices[currentDim] = i;
                    printRecursive(indices, currentDim + 1, indent + 1);
                    if (i < dims[currentDim] - 1) out << std::endl;
                }
                out << std::endl;
                for (int i = 0; i < indent; ++i) out << "  ";
                out << "]";
            }
        };
        std::vector<int> indices(static_cast<int>(dims.size()), 0);
        printRecursive(indices, 0, 0);
    }

    out << std::endl;
    return out;
}

inline std::string YTensorBase::dtype() const { return _dtype; }
inline size_t YTensorBase::elementSize() const { return _element_size; }

inline bool YTensorBase::isContiguous(int fromDim) const {
    if (_data == nullptr) {
        return false;
    }
    auto logStride = this->stride();
    if (logStride.size() != static_cast<size_t>(this->ndim())) {
        return false;
    }
    // 检查数据是否是连续的
    int d = ndim();
    fromDim = (fromDim % d + d) % d; // 循环索引
    for (int i = fromDim; i < d; ++i) {
        if (logStride[i] != _stride[i] && _shape[i] > 1) {
            // 步长不匹配
            return false;
        }
    }
    return true;
}

inline int YTensorBase::isContiguousFrom() const {
    if (_data == nullptr) {
        return ndim();
    }
    auto logStride = this->stride();
    if (logStride.size() != static_cast<size_t>(this->ndim())) {
        return ndim();
    }
    // 检查数据从哪个维度开始是连续的
    for (int a = ndim() - 1; a >= 0; --a) {
        if (logStride[a] != _stride[a] && _shape[a] > 1) {
            return a + 1;
        }
    }
    return 0; // 全部连续
}

inline size_t YTensorBase::toIndex_(const std::vector<int> &pos) const {
    // 与 YTensor::toIndex_ 对应
    if (static_cast<int>(pos.size()) != ndim()) {
        throw std::invalid_argument("toIndex_: position dimensions do not match ndim");
    }
    size_t index = 0;
    for (int i = 0; i < ndim(); ++i) {
        index += static_cast<size_t>(pos[i]) * static_cast<size_t>(_stride[i]);
    }
    return index;
}
template <typename... Args>
inline size_t YTensorBase::toIndex(const Args... args) const {
    std::vector<int> pos = {args...};
    return toIndex(pos);
}

template <typename... Args>
inline size_t YTensorBase::toIndex_(const Args... args) const {
    std::vector<int> pos = {args...};
    return toIndex_(pos);
}
inline size_t YTensorBase::toIndex(const std::vector<int> &pos) const {
    // 与 YTensor::toIndex 对应（逻辑索引）
    if (static_cast<int>(pos.size()) != ndim()) {
        throw std::invalid_argument("toIndex: position dimensions do not match ndim");
    }
    size_t index = 0;
    auto logical_strides = this->stride();
    for (size_t i = 0; i < pos.size(); ++i) {
        index += static_cast<size_t>(pos[i]) * static_cast<size_t>(logical_strides[i]);
    }
    return index;
}

inline std::vector<int> YTensorBase::toCoord(size_t index) const {
    std::vector<int> pos(ndim());
    for (int i = ndim() - 1; i >= 0; --i) {
        pos[i] = (index % _shape[i]);
        index /= _shape[i];
    }
    return pos;
}

template <typename T, typename... Args>
inline T& YTensorBase::at(Args... args) {
    std::vector<int> pos = {args...};
    size_t phys = this->toIndex_(pos);
    return this->data<T>()[phys];
}

template <typename T>
inline T& YTensorBase::at(const std::vector<int>& pos) {
    size_t phys = this->toIndex_(pos);
    return this->data<T>()[phys];
}

template <typename T>
inline const T& YTensorBase::at(const std::vector<int>& pos) const {
    size_t phys = this->toIndex_(pos);
    return this->data<T>()[phys];
}

template <typename T>
inline T& YTensorBase::atData(int index) {
    auto coord = toCoord(index);
    return at<T>(coord);
}

template <typename T>
inline const T& YTensorBase::atData(int index) const {
    auto coord = toCoord(index);
    return at<T>(coord);
}

template <typename T>
inline T& YTensorBase::atData_(int index) {
    return this->data<T>()[index + _offset];
}

template <typename T>
inline const T& YTensorBase::atData_(int index) const {
    return this->data<T>()[index + _offset];
}

// note: calculate_logical_stride removed; stride() returns logical strides

inline bool YTensorBase::isDisjoint() const {
    if (_data == nullptr) {
        return false;
    }
    if (ndim() <= 1) {
        return !(ndim() == 1 && _shape[0] > 1 && _stride[0] == 0);
    }
    std::vector<int> span_lower(ndim(), 1);
    for (int d = ndim() - 2; d >= 0; --d) {
        span_lower[d] = span_lower[d + 1] * _shape[d + 1];
    }
    for (int d = 0; d < ndim(); ++d) {
        if (_shape[d] <= 1) continue;
        int abs_stride = std::abs(_stride[d]);
        // 关键判断：|stride| < 理论最小跨度 → 重叠
        if (abs_stride < span_lower[d]) {
            return false;
        }
    }
    return true;
}

inline std::vector<int> YTensorBase::autoShape(const std::vector<int>& shape) const {
    // 参考 ytensor.inl 中的 autoShape 实现（支持一个 -1 推断）
    std::vector<int> op(shape.size());
    int autoDim = -1; // 自动推导的维度
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            if (autoDim != -1) {
                // switch
                if(autoDim >= static_cast<int>(shape.size())){
                    std::string error = "auto shape cannot infer out of range: found -1 at dimension " + std::to_string(autoDim) +
                        " wh shapeSize " + std::to_string(static_cast<int>(_shape.size()));
                    throw std::invalid_argument(error);
                }else{
                    op[autoDim] = _shape[autoDim];
                    autoDim = static_cast<int>(i);
                }
            }else{
                autoDim = static_cast<int>(i);
            }
        } else {
            op[i] = shape[i];
        }
    }
    if (autoDim != -1) {
        int thisSize = static_cast<int>(this->size());
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

template<typename... Args>
std::vector<int> YTensorBase::autoShape(const Args... shape0) const {
    std::vector<int> shape({shape0...});
    return autoShape(shape);  // 委托给 vector 版本
}

inline YTensorBase YTensorBase::slice(int atDim, int start, int end, int step, bool autoFix) const {
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::slice] Cannot slice a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d; // 循环索引
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
    YTensorBase op = *this;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = newOffset;
    return op;
}

inline YTensorBase& YTensorBase::slice_(int atDim, int start, int end, int step, bool autoFix) {
    *this = slice(atDim, start, end, step, autoFix);
    return *this;
}

inline YTensorBase YTensorBase::permute(const std::vector<int>& newOrder) const {
    // 参考 ytensor.inl: 支持循环索引（负数或超出范围会被映射回有效维度）
    if (newOrder.size() != static_cast<size_t>(ndim())) {
        throw std::invalid_argument("permute: order size must match ndim");
    }
    std::vector<int> new_shape(ndim()), new_stride(ndim());
    int d = ndim();
    for (int i = 0; i < d; ++i) {
        int rotate = (newOrder[i] % d + d) % d;
        new_shape[i] = _shape[rotate];
        new_stride[i] = _stride[rotate];
    }
    YTensorBase op = *this;
    op._shape = new_shape;
    op._stride = new_stride;
    return op;
}

template<typename... Args>
inline YTensorBase YTensorBase::permute(const Args... newOrder) const {
    return permute(std::vector<int>{static_cast<int>(newOrder)...});
}

inline YTensorBase& YTensorBase::permute_(const std::vector<int>& newOrder) {
    if (newOrder.size() != static_cast<size_t>(ndim())) {
        throw std::invalid_argument("permute_: order size must match ndim");
    }
    std::vector<int> new_shape(ndim()), new_stride(ndim());
    int d = ndim();
    for (int i = 0; i < d; ++i) {
        int rotate = (newOrder[i] % d + d) % d;
        new_shape[i] = _shape[rotate];
        new_stride[i] = _stride[rotate];
    }
    _shape = std::move(new_shape);
    _stride = std::move(new_stride);
    return *this;
}

inline YTensorBase YTensorBase::transpose(int dim0, int dim1) const {
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::transpose] Cannot transpose a 0-dim tensor.");
    }
    dim0 = (dim0 % d + d) % d;
    dim1 = (dim1 % d + d) % d;
    if (dim0 == dim1) return *this;
    std::vector<int> newShape = _shape;
    std::vector<int> newStride = _stride;
    std::swap(newShape[dim0], newShape[dim1]);
    std::swap(newStride[dim0], newStride[dim1]);
    YTensorBase op = *this;
    op._shape = newShape;
    op._stride = newStride;
    return op;
}

inline YTensorBase YTensorBase::view(const std::vector<int>& newShape) const {
    if (!isContiguous()) {
        throw std::runtime_error("\"view\" requires contiguous tensor.");
    }
    YTensorBase op = *this;
    op._shape = autoShape(newShape);
    op._stride = op.stride();
    return op;
}

template<typename... Args>
inline YTensorBase YTensorBase::view(const Args... newShape) const {
    return view(std::vector<int>{static_cast<int>(newShape)...});
}

inline YTensorBase YTensorBase::reshape(const std::vector<int>& newShape) const {
    return contiguous().view(newShape);
}

template<typename... Args>
inline YTensorBase YTensorBase::reshape(const Args... newShape) const {
    return reshape(std::vector<int>{static_cast<int>(newShape)...});
}

inline YTensorBase YTensorBase::unsqueeze(int dim) const {
    int d = ndim();
    // 循环索引：dim 的有效范围是 [0, d]（共 d+1 个位置）
    dim = ((dim % (d + 1)) + (d + 1)) % (d + 1);
    YTensorBase op = *this;
    op._shape.insert(op._shape.begin() + dim, 1);
    // 新维度的 stride 可以设为任意值（因为 size=1），通常设为下一维度的 stride * size
    int newStride = (dim < d) ? op._stride[dim] * op._shape[dim + 1] : 1;
    op._stride.insert(op._stride.begin() + dim, newStride);
    return op;
}

inline YTensorBase& YTensorBase::unsqueeze_(int dim) {
    int d = ndim();
    // 循环索引
    dim = ((dim % (d + 1)) + (d + 1)) % (d + 1);
    int newStride = (dim < d) ? _stride[dim] * _shape[dim] : 1;
    _shape.insert(_shape.begin() + dim, 1);
    _stride.insert(_stride.begin() + dim, newStride);
    return *this;
}

inline YTensorBase YTensorBase::squeeze(int dim) const {
    YTensorBase op = *this;
    if (dim >= 0) {
        // 移除指定维度
        int d = ndim();
        dim = (dim % d + d) % d;
        if (_shape[dim] != 1) {
            throw std::runtime_error("squeeze: can only squeeze dimensions of size 1");
        }
        op._shape.erase(op._shape.begin() + dim);
        op._stride.erase(op._stride.begin() + dim);
    } else {
        // dim < 0：移除所有大小为1的维度
        std::vector<int> newShape, newStride;
        for (int i = 0; i < ndim(); ++i) {
            if (_shape[i] != 1) {
                newShape.push_back(_shape[i]);
                newStride.push_back(_stride[i]);
            }
        }
        if (newShape.empty()) {
            // 如果全部维度都是1，保留一个
            newShape.push_back(1);
            newStride.push_back(1);
        }
        op._shape = newShape;
        op._stride = newStride;
    }
    return op;
}

inline YTensorBase& YTensorBase::squeeze_(int dim) {
    if (dim >= 0) {
        int d = ndim();
        dim = (dim % d + d) % d;
        if (_shape[dim] != 1) {
            throw std::runtime_error("squeeze_: can only squeeze dimensions of size 1");
        }
        _shape.erase(_shape.begin() + dim);
        _stride.erase(_stride.begin() + dim);
    } else {
        std::vector<int> newShape, newStride;
        for (int i = 0; i < ndim(); ++i) {
            if (_shape[i] != 1) {
                newShape.push_back(_shape[i]);
                newStride.push_back(_stride[i]);
            }
        }
        if (newShape.empty()) {
            newShape.push_back(1);
            newStride.push_back(1);
        }
        _shape = newShape;
        _stride = newStride;
    }
    return *this;
}

inline YTensorBase YTensorBase::repeat(const std::vector<int>& times) const {
    if (times.size() != static_cast<size_t>(ndim())) {
        throw std::invalid_argument("repeat: times size must match ndim");
    }
    YTensorBase op = *this;
    for (int i = 0; i < ndim(); ++i) {
        if (times[i] <= 1) continue;
        if (_shape[i] != 1) {
            throw std::runtime_error("Can only repeat on dimensions of size 1.");
        }
        op._shape[i] = times[i];
        op._stride[i] = 0;
    }
    return op;
}

template<typename... Args>
inline YTensorBase YTensorBase::repeat(const Args... times) const {
    return repeat(std::vector<int>{static_cast<int>(times)...});
}

inline YTensorBase& YTensorBase::repeat_(const std::vector<int>& times) {
    if (times.size() != static_cast<size_t>(ndim())) {
        throw std::invalid_argument("repeat_: times size must match ndim");
    }
    for (int i = 0; i < ndim(); ++i) {
        if (times[i] <= 1) continue;
        if (_shape[i] != 1) {
            throw std::runtime_error("Can only repeat on dimensions of size 1.");
        }
        _shape[i] = times[i];
        _stride[i] = 0;
    }
    return *this;
}

inline YTensorBase YTensorBase::unfold(int atDim, int kernel, int stride, int dilation) const {
    // 参考 ytensor.inl 的实现，保证插入维度与步长计算一致
    if (kernel <= 0 || stride <= 0 || dilation <= 0) {
        throw std::invalid_argument("Invalid kernel/stride/dilation");
    }
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::unfold] Cannot unfold a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d;
    if (_shape[atDim] < ((kernel - 1) * dilation + 1)) {
        throw std::invalid_argument("Dimension size is too small for unfold.");
    }
    int nums = (_shape[atDim] - (kernel - 1) * dilation - 1) / stride + 1;

    YTensorBase op = *this;
    std::vector<int> newShape = _shape;
    // 保持与模板一致的 ordering：nums 放在 kernel 之前
    newShape[atDim] = nums;
    newShape.insert(newShape.begin() + atDim + 1, kernel);

    std::vector<int> newStride = _stride;
    // nums 维度的步长为原步长 * stride，kernel 维度的步长为原步长 * dilation
    newStride[atDim] = _stride[atDim] * stride;
    newStride.insert(newStride.begin() + atDim + 1, _stride[atDim] * dilation);

    op._shape = newShape;
    op._stride = newStride;
    return op;
}

inline YTensorBase& YTensorBase::unfold_(int atDim, int kernel, int stride, int dilation) {
    if (kernel <= 0 || stride <= 0 || dilation <= 0) {
        throw std::invalid_argument("Invalid kernel/stride/dilation");
    }
    int d = ndim();
    if (d == 0) {
        throw std::out_of_range("[YTensorBase::unfold_] Cannot unfold a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d;
    if (_shape[atDim] < ((kernel - 1) * dilation + 1)) {
        throw std::invalid_argument("Dimension size is too small for unfold.");
    }
    int nums = (_shape[atDim] - (kernel - 1) * dilation - 1) / stride + 1;
    std::vector<int> newShape = _shape;
    newShape[atDim] = nums;
    newShape.insert(newShape.begin() + atDim + 1, kernel);
    std::vector<int> newStride = _stride;
    newStride[atDim] = _stride[atDim] * stride;
    newStride.insert(newStride.begin() + atDim + 1, _stride[atDim] * dilation);
    _shape = std::move(newShape);
    _stride = std::move(newStride);
    return *this;
}
// non-inplace overloads removed; inplace versions with trailing '_' are provided above

inline YTensorBase YTensorBase::mostContinuousView() const {
    // 按照stride的大小顺序进行排序
    if (_data == nullptr) {
        YTensorBase op;
        op._shape = this->shape();
        op._stride = op.stride();
        op._data = nullptr;
        op._offset = 0;
        return op;
    }
    std::vector<std::pair<int, int>> mapper(ndim());// shape[i], i
    for (int i = 0; i < ndim(); ++i) {
        mapper[i] = std::make_pair(_stride[i], i);
    }
    std::sort(mapper.begin(), mapper.end(), [](const auto& a, const auto& b) {
        return std::abs(a.first) > std::abs(b.first);
    });
    std::vector<int> perm(ndim());
    for (int i = 0; i < ndim(); ++i) {
        perm[i] = mapper[i].second;
    }
    YTensorBase op = this->permute(perm);
    for(int i = 0; i < ndim(); i++){
        if(op._stride[i] < 0){
            // 负步长改为正步长，并调整offset
            op._stride[i] = -op._stride[i];
            op._offset += (op._shape[i] - 1) * op._stride[i];
        }
    }
    return op;
}

inline void YTensorBase::seed(unsigned int seed) {
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
    yt::infos::gen.seed(seed);
}

inline YTensorBase YTensorBase::_RandnGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
    YTensorBase op(shape, dtype);
    size_t max = op.size();
    std::normal_distribution<double> dist(0.0, 1.0);
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);

    // dtype 由 op 构造时设定，用对应类型写入
    if (op.elementSize() == static_cast<size_t>(yt::types::getTypeSize<float>())) {
        float* ptr = op.data<float>();
        for (size_t i = 0; i < max; ++i) ptr[i] = static_cast<float>(dist(gen));
    } else if (op.dtype() == "float64" ) {
        double* ptr = op.data<double>();
        for (size_t i = 0; i < max; ++i) ptr[i] = static_cast<double>(dist(gen));
    } else {
        // fallback: generate float values and cast-copy per element
        float* tmp = new float[max];
        for (size_t i = 0; i < max; ++i) tmp[i] = static_cast<float>(dist(gen));
        // copy into op according to dtype
        if (op.dtype() == "int32") {
            int32_t* p = op.data<int32_t>(); for (size_t i=0;i<max;i++) p[i] = static_cast<int32_t>(tmp[i]);
        } else if (op.dtype() == "int64") {
            int64_t* p = op.data<int64_t>(); for (size_t i=0;i<max;i++) p[i] = static_cast<int64_t>(tmp[i]);
        } else if (op.dtype() == "bfloat16") {
            yt::bfloat16* p = op.data<yt::bfloat16>(); for (size_t i=0;i<max;i++) p[i] = static_cast<yt::bfloat16>(tmp[i]);
        } else {
            // last resort: write bytes by memcpy of float
            std::memcpy(op._data.get(), tmp, std::min<size_t>(max * sizeof(float), max * op.elementSize()));
        }
        delete[] tmp;
    }
    return op;
}

inline YTensorBase YTensorBase::_RanduGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
    YTensorBase op(shape, dtype);
    size_t max = op.size();
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);

    if (op.elementSize() == static_cast<size_t>(yt::types::getTypeSize<float>())) {
        float* ptr = op.data<float>();
        for (size_t i = 0; i < max; ++i) ptr[i] = static_cast<float>(dist(gen));
    } else if (op.dtype() == "float64") {
        double* ptr = op.data<double>();
        for (size_t i = 0; i < max; ++i) ptr[i] = static_cast<double>(dist(gen));
    } else {
        float* tmp = new float[max];
        for (size_t i = 0; i < max; ++i) tmp[i] = static_cast<float>(dist(gen));
        if (op.dtype() == "int32") {
            int32_t* p = op.data<int32_t>(); for (size_t i=0;i<max;i++) p[i] = static_cast<int32_t>(tmp[i]);
        } else if (op.dtype() == "int64") {
            int64_t* p = op.data<int64_t>(); for (size_t i=0;i<max;i++) p[i] = static_cast<int64_t>(tmp[i]);
        } else if (op.dtype() == "bfloat16") {
            yt::bfloat16* p = op.data<yt::bfloat16>(); for (size_t i=0;i<max;i++) p[i] = static_cast<yt::bfloat16>(tmp[i]);
        } else {
            std::memcpy(op._data.get(), tmp, std::min<size_t>(max * sizeof(float), max * op.elementSize()));
        }
        delete[] tmp;
    }
    return op;
}

inline YTensorBase YTensorBase::zeros(const std::vector<int>& shape, std::string dtype) {
    YTensorBase op(shape, dtype);
    size_t total = op.size();
    size_t bytes = total * op.elementSize();
    if (op._data) std::memset(op._data.get(), 0, bytes);
    return op;
}

inline YTensorBase YTensorBase::ones(const std::vector<int>& shape, std::string dtype) {
    YTensorBase op(shape, dtype);
    size_t total = op.size();
    const std::string dt = op.dtype();
    if (dt == "float32") {
        float* p = op.data<float>(); for (size_t i=0;i<total;i++) p[i] = 1.0f;
    } else if (dt == "float64") {
        double* p = op.data<double>(); for (size_t i=0;i<total;i++) p[i] = 1.0;
    } else if (dt == "int32") {
        int32_t* p = op.data<int32_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else if (dt == "int64") {
        int64_t* p = op.data<int64_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else if (dt == "int16") {
        int16_t* p = op.data<int16_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else if (dt == "int8") {
        int8_t* p = op.data<int8_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else if (dt == "uint8") {
        uint8_t* p = op.data<uint8_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else if (dt == "bfloat16") {
        yt::bfloat16* p = op.data<yt::bfloat16>(); for (size_t i=0;i<total;i++) p[i] = static_cast<yt::bfloat16>(1.0f);
    } else if (dt == "bool") {
        uint8_t* p = op.data<uint8_t>(); for (size_t i=0;i<total;i++) p[i] = 1;
    } else {
        // fallback: set first byte of each element to 1
        char* bytes_ptr = op._data.get();
        size_t es = op.elementSize();
        for (size_t i=0;i<total;i++) bytes_ptr[i*es] = 1;
    }
    return op;
}

inline YTensorBase YTensorBase::contiguous() const {
    if (_data == nullptr) {
        return YTensorBase(_shape, _dtype);
    }
    if (isContiguous()) {
        return *this;  // 已经连续，直接返回浅拷贝
    }
    // 非连续，返回深拷贝（clone会生成连续排布）
    return clone();
}

inline YTensorBase& YTensorBase::contiguous_() {
    if (_data == nullptr) return *this;
    if (isContiguous()) {
        return *this;  // 已经连续，无需操作
    }
    // 非连续，用clone替换自己
    YTensorBase cloned = clone();
    _data = cloned._data;
    _shape = cloned._shape;
    _stride = cloned._stride;
    _offset = cloned._offset;
    return *this;
}

inline YTensorBase YTensorBase::concat(const std::vector<YTensorBase>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("[YTensorBase::concat] Empty tensor list");
    }
    if (tensors.size() == 1) {
        return tensors[0].clone();
    }

    const auto& first = tensors[0];
    int d = first.ndim();
    axis = (axis % d + d) % d;

    // 验证所有张量的形状兼容性
    std::vector<int> resultShape = first.shape();
    int totalAxisSize = resultShape[axis];

    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (t.ndim() != d) {
            throw std::invalid_argument("[YTensorBase::concat] Dimension mismatch");
        }
        if (t.dtype() != first.dtype()) {
            throw std::invalid_argument("[YTensorBase::concat] dtype mismatch");
        }
        for (int dim = 0; dim < d; ++dim) {
            if (dim != axis && t.shape(dim) != resultShape[dim]) {
                throw std::invalid_argument("[YTensorBase::concat] Shape mismatch on non-concat axis");
            }
        }
        totalAxisSize += t.shape(axis);
    }
    resultShape[axis] = totalAxisSize;

    // 创建结果张量
    YTensorBase result(resultShape, first.dtype());

    // 逐个复制数据 - 需要正确处理非连续的 slice
    int offset = 0;
    size_t elemSize = first._element_size;
    for (const auto& t : tensors) {
        int axisSize = t.shape(axis);
        YTensorBase src = t.contiguous();

        // 计算每个 axis 维度块的大小
        size_t blockSize = 1;
        for (int i = axis + 1; i < d; ++i) {
            blockSize *= resultShape[i];
        }

        // 计算有多少个块（axis 前面的维度乘积）
        size_t numBlocks = 1;
        for (int i = 0; i < axis; ++i) {
            numBlocks *= resultShape[i];
        }

        // 逐块复制
        char* resultData = result._data.get();
        const char* srcData = src._data.get() + src._offset * elemSize;

        for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
            // 计算 result 中的起始位置
            // 在 result 中，每个块的起始偏移 = blockIdx * (totalAxisSize * blockSize) + offset * blockSize
            size_t resultOffset = blockIdx * resultShape[axis] * blockSize + offset * blockSize;

            // 在 src 中，每个块的大小 = axisSize * blockSize
            size_t srcBlockOffset = blockIdx * axisSize * blockSize;

            std::memcpy(resultData + resultOffset * elemSize,
                        srcData + srcBlockOffset * elemSize,
                        axisSize * blockSize * elemSize);
        }

        offset += axisSize;
    }

    return result;
}

inline YTensorBase YTensorBase::concat(const YTensorBase& other, int axis) const {
    return YTensorBase::concat({*this, other}, axis);
}

inline std::vector<YTensorBase> YTensorBase::split(const std::vector<int>& splitSizes, int axis) const {
    int d = ndim();
    axis = (axis % d + d) % d;

    // 验证分割大小总和
    int total = 0;
    for (int s : splitSizes) {
        if (s <= 0) {
            throw std::invalid_argument("[YTensorBase::split] Split size must be positive");
        }
        total += s;
    }
    if (total != _shape[axis]) {
        throw std::invalid_argument("[YTensorBase::split] Split sizes sum doesn't match axis size");
    }

    // 使用 slice 创建视图
    std::vector<YTensorBase> result;
    result.reserve(splitSizes.size());
    int offset = 0;
    for (int s : splitSizes) {
        result.push_back(slice(axis, offset, offset + s));
        offset += s;
    }
    return result;
}

inline std::vector<YTensorBase> YTensorBase::split(int n, int axis) const {
    int d = ndim();
    axis = (axis % d + d) % d;
    int axisSize = _shape[axis];

    if (n <= 0) {
        throw std::invalid_argument("[YTensorBase::split] n must be positive");
    }
    if (axisSize % n != 0) {
        throw std::invalid_argument("[YTensorBase::split] Axis size not divisible by n");
    }

    int chunkSize = axisSize / n;
    std::vector<int> splitSizes(n, chunkSize);
    return split(splitSizes, axis);
}

}//namespace yt
/***************
* @file: ytensor_base_math.inl
* @brief: YTensorBase 数学运算的实现
* @author: SnifferCaptain
* @date: 2025-12-1
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>


namespace yt::kernel{
/// @brief 并行for循环，根据任务量自动选择并行或串行执行
/// @param from 起始索引（包含）
/// @param to 结束索引（不包含）
/// @param func 可调用对象，接受一个int参数，表示当前索引
/// @param flop 每次迭代的浮点运算量估计。当问题规模大于minParOps时，开启多核并行执行。以单次浮点运算为单位1。
template<typename Func>
void parallelFor(int from, int to, Func&& func, double flop = 1.){
    if((to - from) * flop >= yt::infos::minParOps) {
        #pragma omp parallel for simd proc_bind(close)
        for (int i = from; i < to; i++) {
            func(i);
        }
    } else {
        #pragma omp simd
        for (int i = from; i < to; i++) {
            func(i);
        }
    }
}

}// namespace yt::kernel

namespace yt::kernel {

// 数学工具函数（暂无）

} // namespace yt::kernel

#include <memory>
#include <cstddef>

namespace yt::kernel {

/// @brief 使用placement new为非POD类型分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param obj 要拷贝构造的对象
/// @return 返回管理内存的shared_ptr<char[]>
template<typename T>
inline std::shared_ptr<char[]> makeSharedPlacement(const T& obj) {
    char* rawMemory = new char[sizeof(T)];
    new (rawMemory) T(obj);
    return std::shared_ptr<char[]>(rawMemory, [](char* ptr) {
        reinterpret_cast<T*>(ptr)->~T();
        delete[] ptr;
    });
}

/// @brief 使用placement new为非POD类型数组分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param count 数组元素个数
/// @return 返回管理内存的shared_ptr<char[]>，内存未初始化，需要手动使用placement new
/// @example auto ptr = makeSharedPlacementArray<MatType>(10);
///          for(int i = 0; i < 10; i++) { new (&reinterpret_cast<MatType*>(ptr.get())[i]) MatType(...); }
template<typename T>
inline std::shared_ptr<char[]> makeSharedPlacementArray(size_t count) {
    char* rawMemory = new char[count * sizeof(T)];
    return std::shared_ptr<char[]>(rawMemory, [count](char* ptr) {
        T* arr = reinterpret_cast<T*>(ptr);
        #pragma omp simd
        for (size_t i = 0; i < count; ++i) {
            arr[i].~T();
        }
        delete[] ptr;
    });
}

} // namespace yt::kernel
#include <vector>
#include <stdexcept>
#include <array>
#include <utility>

namespace yt::kernel {

/// @brief 使用模板递归在编译期展开N个索引的累加
/// @tparam N 张量数量
/// @tparam I 当前处理的张量索引
template<size_t N, size_t I = 0>
struct IndexAccumulator {
    template<typename StridesArray, typename IndicesArray>
    static inline void accumulate(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
        indices[I] += posi * strides[I][dimIdx];
        if constexpr (I + 1 < N) {
            IndexAccumulator<N, I + 1>::accumulate(indices, posi, strides, dimIdx);
        }
    }
};

/// @brief 编译期展开的N元索引计算器（使用编译期dim）
/// @tparam N 张量数量
/// @tparam Dim 维度（编译期常量）
template<size_t N, int Dim>
struct NaryIndexComputer {
    /// @brief 计算N个张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param strides N个张量的stride数组，每个是Dim维的数组
    /// @return 包含N个数据索引的数组
    template<typename LogicStrideArray, typename ShapeArray, typename StridesArray>
    static inline std::array<int, N> compute(
        int index,
        const LogicStrideArray& logicStride,
        const ShapeArray& shape,
        const StridesArray& strides)
    {
        std::array<int, N> indices = {};
        // 外层循环使用编译期常量Dim
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            // 内层循环使用编译期常量N，通过递归展开
            accumulateAll<0>(indices, posi, strides, d);
        }
        return indices;
    }

    /// @brief 同时计算this和N个其他张量的数据索引
    /// @param index 线性索引
    /// @param logicStride 逻辑stride数组
    /// @param shape 广播shape数组
    /// @param thisStride this的stride数组
    /// @param strides N个张量的stride数组
    /// @param thisDataIdx 输出：this的数据索引
    /// @return 包含N个数据索引的数组
    template<typename LogicStrideArray, typename ShapeArray, typename ThisStrideArray, typename StridesArray>
    static inline std::array<int, N> computeWithThis(
        int index,
        const LogicStrideArray& logicStride,
        const ShapeArray& shape,
        const ThisStrideArray& thisStride,
        const StridesArray& strides,
        int& thisDataIdx)
    {
        std::array<int, N> indices = {};
        thisDataIdx = 0;
        // 外层循环使用编译期常量Dim，计算一次posi同时更新所有索引
        for (int d = 0; d < Dim; ++d) {
            int posi = (index / logicStride[d]) % shape[d];
            thisDataIdx += posi * thisStride[d];
            // 内层循环使用编译期常量N，通过递归展开
            accumulateAll<0>(indices, posi, strides, d);
        }
        return indices;
    }

private:
    template<size_t I, typename IndicesArray, typename StridesArray>
    static inline void accumulateAll(IndicesArray& indices, int posi, const StridesArray& strides, int dimIdx) {
        indices[I] += posi * strides[I][dimIdx];
        if constexpr (I + 1 < N) {
            accumulateAll<I + 1>(indices, posi, strides, dimIdx);
        }
    }
};

/// @brief 计算广播索引（运行时ndim版本）
/// @param index 线性索引
/// @param logicStride 逻辑stride
/// @param shape 广播shape
/// @param thisStride this张量的stride
/// @param otherStrides 其他张量的stride数组
/// @param thisIdx 输出：this的数据索引
/// @param otherIndices 输出：其他张量的数据索引
/// @param ndim 维度数
inline void computeBroadcastIndicesRuntime(
    int index,
    const std::vector<int>& logicStride,
    const std::vector<int>& shape,
    const std::vector<int>& thisStride,
    const std::vector<std::vector<int>>& otherStrides,
    int& thisIdx,
    std::vector<int>& otherIndices,
    int ndim)
{
    thisIdx = 0;
    std::fill(otherIndices.begin(), otherIndices.end(), 0);
    for (int i = 0; i < ndim; ++i) {
        int posi = (index / logicStride[i]) % shape[i];
        thisIdx += posi * thisStride[i];
        for (size_t t = 0; t < otherStrides.size(); ++t) {
            otherIndices[t] += posi * otherStrides[t][i];
        }
    }
}

/// @brief 计算N个张量的广播索引（编译期展开）
/// @tparam N 张量数量
/// @param index 线性索引
/// @param logicStride 逻辑stride
/// @param broadcastShape 广播shape
/// @param strides 每个张量的广播stride数组
/// @param opdim 操作维度
/// @return 包含N个数据索引的数组
template<size_t N>
inline std::array<int, N> computeBroadcastIndices(
    int index,
    const std::vector<int>& logicStride,
    const std::vector<int>& broadcastShape,
    const std::array<const int*, N>& strides,
    int opdim)
{
    std::array<int, N> indices = {};
    for (int i = 0; i < opdim; ++i) {
        int posi = (index / logicStride[i]) % broadcastShape[i];
        IndexAccumulator<N>::accumulate(indices, posi, strides, i);
    }
    return indices;
}

// ==================== 原有函数 ====================

/// @brief 计算多个张量的广播shape
/// @param shapes 所有参与广播的张量的shape列表
/// @return 广播后的shape
/// @throw std::runtime_error 如果shapes无法广播
inline std::vector<int> computeBroadcastShape(const std::vector<std::vector<int>>& shapes) {
    if (shapes.empty()) return {};

    // 找最大维度
    size_t maxDim = 0;
    for (const auto& s : shapes) {
        maxDim = std::max(maxDim, s.size());
    }

    std::vector<int> result(maxDim, 1);
    for (const auto& shape : shapes) {
        size_t offset = maxDim - shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            size_t ri = i + offset;
            if (result[ri] == 1) {
                result[ri] = shape[i];
            } else if (shape[i] != 1 && shape[i] != result[ri]) {
                throw std::runtime_error("Broadcast: shapes cannot be broadcast together");
            }
        }
    }
    return result;
}

/// @brief 获取张量在广播shape下的stride
/// @param shape 原始张量的shape
/// @param stride 原始张量的stride
/// @param broadcastShape 广播后的shape
/// @return 广播stride（对于被广播的维度，stride为0）
inline std::vector<int> getBroadcastStride(const std::vector<int>& shape,
                                           const std::vector<int>& stride,
                                           const std::vector<int>& broadcastShape) {
    size_t offset = broadcastShape.size() - shape.size();
    std::vector<int> result(broadcastShape.size(), 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == broadcastShape[i + offset]) {
            result[i + offset] = stride[i];
        }
        // else: shape[i] == 1, stride stays 0 (broadcast)
    }
    return result;
}

/// @brief 计算张量在给定索引处的实际数据索引
/// @param linearIndex 线性索引
/// @param logicStride 逻辑stride（连续存储）
/// @param broadcastStride 广播stride
/// @param broadcastShape 广播shape
/// @return 实际数据索引
inline int computeDataIndex(int linearIndex,
                            const std::vector<int>& logicStride,
                            const std::vector<int>& broadcastStride,
                            const std::vector<int>& broadcastShape) {
    int dataIndex = 0;
    int opdim = static_cast<int>(broadcastShape.size());
    #pragma omp simd reduction(+:dataIndex)
    for (int i = 0; i < opdim; ++i) {
        int posi = (linearIndex / logicStride[i]) % broadcastShape[i];
        dataIndex += posi * broadcastStride[i];
    }
    return dataIndex;
}

/// @brief 统一的广播操作函数（非原地），支持N元张量/标量操作
/// @tparam Func 函数类型，签名为 ReturnType func(const T&, const T&, ...) 或 ReturnType func(T, T, ...)
/// @tparam Args 参数类型，可以是YTensor或标量T
/// @param func 操作函数，返回类型用于推断结果张量的标量类型
/// @param tensors 输入的张量或标量
/// @return 返回结果张量，形状为所有输入张量广播后的形状，返回类型由func的返回值推断
template <typename Func, typename... Args>
auto broadcast(Func&& func, Args&&... tensors) {
    using namespace yt::traits;

    // 收集所有张量的shape
    std::vector<std::vector<int>> shapes;
    auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    if (shapes.empty()) {
        throw std::runtime_error("broadcast: at least one tensor argument required");
    }

    // 推断标量类型（从第一个张量参数）
    using ScalarType = typename std::decay_t<decltype(std::get<0>(std::tuple<Args...>(tensors...)))>::scalarType;

    // 计算广播shape
    auto broadcastShape = computeBroadcastShape(shapes);
    int opdim = static_cast<int>(broadcastShape.size());

    // 计算逻辑stride（连续存储）
    std::vector<int> logicStride(opdim);
    int stride = 1;
    for (int i = opdim - 1; i >= 0; --i) {
        logicStride[i] = stride;
        stride *= broadcastShape[i];
    }
    int totalSize = stride;

    // 收集每个张量参数的广播stride
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const ScalarType*> dataPtrs;
    bool allContiguous = true;
    bool allShapeEqual = true;

    auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            // 区分YTensor<U, d>和YTensorBase
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                dataPtrs.push_back(arg.data());
                allContiguous = allContiguous && arg.isContiguous();
            } else {
                dataPtrs.push_back(arg.template data<ScalarType>());
                allContiguous = allContiguous && arg.isContiguous();
            }
            // 检查shape是否与broadcastShape相同
            auto argShape = arg.shape();
            if (argShape.size() != broadcastShape.size()) {
                allShapeEqual = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != broadcastShape[i]) {
                        allShapeEqual = false;
                        break;
                    }
                }
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);

    // 推断返回类型
    using ReturnType = std::invoke_result_t<Func, decltype(std::declval<std::conditional_t<is_ytensor_v<Args>, ScalarType, Args>>())...>;

    // 创建结果张量（使用最大维度）
    constexpr int resultDim = max_dim<Args...>() > 0 ? max_dim<Args...>() : 1;
    YTensor<ReturnType, resultDim> result;

    // 根据broadcastShape的实际维度调整
    if (opdim == resultDim) {
        result.reserve(broadcastShape);
    } else if (opdim < resultDim) {
        std::vector<int> paddedShape(resultDim - opdim, 1);
        paddedShape.insert(paddedShape.end(), broadcastShape.begin(), broadcastShape.end());
        result.reserve(paddedShape);
    } else {
        std::vector<int> trimmedShape(broadcastShape.end() - resultDim, broadcastShape.end());
        result.reserve(trimmedShape);
    }

    // Fastpath: 所有张量都是连续的且shape相同
    if (allContiguous && allShapeEqual) {
        ReturnType* resultPtr = result.data_();
        // 使用直接指针访问
        auto getValueDirect = [&](auto&& arg, int index, int& tensorIdx) -> decltype(auto) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                return dataPtrs[tensorIdx++][index];
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };

        parallelFor(0, totalSize, [&](int index) {
            int tensorIdx = 0;
            resultPtr[index] = func(getValueDirect(tensors, index, tensorIdx)...);
        });
        return result;
    }

    // Slowpath: 需要广播的情况
    ReturnType* resultPtr = result.data_();
    constexpr size_t numTensors = (static_cast<size_t>(is_ytensor_v<Args>) + ...);

    // 使用编译期dim优化 - 提取所有必要的std::array
    std::array<std::array<int, resultDim>, numTensors> tensorStrides;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int i = 0; i < resultDim; ++i) {
            tensorStrides[t][i] = (i < static_cast<int>(broadcastStrides[t].size())) ? broadcastStrides[t][i] : 0;
        }
    }

    // 提取结果shape和stride为std::array用于编译期循环
    std::array<int, resultDim> resultShape;
    std::array<int, resultDim> resultLogicStride;
    auto resShape = result.shape();
    for (int i = 0; i < resultDim; ++i) {
        resultShape[i] = resShape[i];
    }
    // 计算连续stride
    stride = 1;
    for (int i = resultDim - 1; i >= 0; --i) {
        resultLogicStride[i] = stride;
        stride *= resultShape[i];
    }

    parallelFor(0, totalSize, [&](int index) {
        // 使用NaryIndexComputer进行N元索引计算
        auto indices = NaryIndexComputer<numTensors, resultDim>::compute(
            index, resultLogicStride, resultShape, tensorStrides);

        // 获取值的lambda，使用预计算的索引
        size_t tensorIdx = 0;
        auto getValue = [&](auto&& arg) -> decltype(auto) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][indices[idx]];
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };
        resultPtr[index] = func(getValue(tensors)...);
    });

    return result;
}

/// @brief 统一的广播原地操作函数，支持N元张量/标量操作
/// @tparam TensorType 目标张量类型（YTensor<T, dim>）
/// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
/// @tparam Args 参数类型，可以是YTensor或标量T
/// @param target 目标张量（将被原地修改）
/// @param func 操作函数，第一个参数为target的元素引用
/// @param tensors 输入的张量或标量
/// @return 返回target的引用
template <typename TensorType, typename Func, typename... Args>
TensorType& broadcastInplace(TensorType& target, Func&& func, Args&&... tensors) {
    using namespace yt::traits;
    using T = typename TensorType::scalarType;
    constexpr int dim = TensorType::ndim;

    // 收集所有张量的shape（包括target）
    std::vector<std::vector<int>> shapes;
    shapes.push_back(target.shape());

    auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    // 计算广播shape
    auto broadcastShape = computeBroadcastShape(shapes);

    // 验证target的shape与广播shape兼容（inplace要求target的shape必须等于广播shape）
    auto targetShapeVec = target.shape();
    if (static_cast<int>(broadcastShape.size()) != dim) {
        throw std::runtime_error("broadcastInplace: result dimension mismatch");
    }
    for (int i = 0; i < dim; ++i) {
        if (targetShapeVec[i] != broadcastShape[i]) {
            throw std::runtime_error("broadcastInplace: target tensor shape must match broadcast shape");
        }
    }

    int totalSize = target.size();
    bool allContiguous = target.isContiguous();
    bool allEqualShape = true;

    auto checkContiguousAndShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (!arg.isContiguous()) {
                allContiguous = false;
            }
            auto argShape = arg.shape();
            if (argShape.size() != shapes[0].size()) {
                allEqualShape = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != shapes[0][i]) {
                        allEqualShape = false;
                        break;
                    }
                }
            }
        }
    };
    (checkContiguousAndShape(tensors), ...);

    if (allContiguous && allEqualShape) {
        // 所有张量是否连续且shape相同
        T* targetDataPtr = target.data();

        // 收集所有张量的数据指针
        std::vector<const T*> dataPtrs;
        auto collectPtrs = [&](auto&& arg) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                if constexpr (is_ytensor_template_v<decltype(arg)>) {
                    dataPtrs.push_back(arg.data());
                } else {
                    dataPtrs.push_back(arg.template data<T>());
                }
            }
        };
        (collectPtrs(tensors), ...);

        // 创建fastpath的getValue lambda
        auto getValueFast = [&](auto&& arg, int index, int& tensorIdx) -> decltype(auto) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                ++tensorIdx;
                return dataPtrs[tensorIdx - 1][index];
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };

        parallelFor(0, totalSize, [&](int index) {
            int tensorIdx = 0;
            func(targetDataPtr[index], getValueFast(tensors, index, tensorIdx)...);
        });

        return target;
    }

    // 计算逻辑stride
    auto logicStride = target.stride();

    // 收集每个张量参数的广播stride
    std::vector<std::vector<int>> broadcastStrides;
    std::vector<const T*> dataPtrs;

    auto collectBroadcastInfo = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            broadcastStrides.push_back(getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
            // 区分YTensor<U, d>和YTensorBase
            if constexpr (is_ytensor_template_v<decltype(arg)>) {
                // YTensor
                dataPtrs.push_back(arg.data());
            } else {
                // YTensorBase
                dataPtrs.push_back(arg.template data<T>());
            }
        }
    };
    (collectBroadcastInfo(tensors), ...);

    // 使用编译期展开优化多参数内核
    constexpr size_t numTensors = (static_cast<size_t>(is_ytensor_v<Args>) + ...);

    // N元通用优化：使用编译期常量dim和numTensors
    // 将所有张量的stride存入编译期大小的数组
    std::array<std::array<int, dim>, numTensors> strideArrays;
    for (size_t t = 0; t < numTensors; ++t) {
        for (int d = 0; d < dim; ++d) {
            strideArrays[t][d] = broadcastStrides[t][d];
        }
    }

    // 将target的shape和stride也存入编译期数组
    std::array<int, dim> targetShape, targetStride;
    auto targetShapeV = target.shape();
    auto targetStrideV = target.stride_();
    for (int d = 0; d < dim; ++d) {
        targetShape[d] = targetShapeV[d];
        targetStride[d] = targetStrideV[d];
    }

    parallelFor(0, totalSize, [&](int index) {
        // 使用NaryIndexComputer同时计算target和所有其他张量的索引
        int targetDataIdx;
        auto tensorIndices = NaryIndexComputer<numTensors, dim>::computeWithThis(
            index, logicStride, targetShape, targetStride, strideArrays, targetDataIdx);

        // 获取各张量的值
        size_t tensorIdx = 0;
        auto getValue = [&](auto&& arg) -> decltype(auto) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                size_t idx = tensorIdx++;
                return dataPtrs[idx][tensorIndices[idx]];
            } else {
                return std::forward<decltype(arg)>(arg);
            }
        };

        // 使用atData_处理偏移量
        func(target.atData_(targetDataIdx), getValue(tensors)...);
    });

    return target;
}

/// @brief 编译期版本：判断底层存储的某个位置是否属于当前view
/// @tparam Dim 维度数
/// @param delta 相对于view起始offset的偏移量
/// @param shape view的shape数组
/// @param stride view的stride数组
/// @return 如果该位置属于view返回true，否则返回false
template<int Dim>
inline bool isPositionInView(int delta, const std::array<int, Dim>& shape, const std::array<int, Dim>& stride) {
    for (int b = 0; b < Dim; ++b) {
        if (shape[b] == 1) {
            continue;
        } else if (stride[b] != 0) {
            int step = delta / stride[b];
            if (step < 0 || step >= shape[b]) {
                return false;
            }
            delta -= step * stride[b];
        }
    }
    return (delta == 0);
}

/// @brief 运行时版本：判断底层存储的某个位置是否属于当前view
/// @param delta 相对于view起始offset的偏移量
/// @param shape view的shape
/// @param stride view的stride
/// @return 如果该位置属于view返回true，否则返回false
inline bool isPositionInViewRuntime(int delta, const std::vector<int>& shape, const std::vector<int>& stride) {
    int ndim = static_cast<int>(shape.size());
    for (int b = 0; b < ndim; ++b) {
        if (shape[b] == 1) {
            continue;
        } else if (stride[b] != 0) {
            int step = delta / stride[b];
            if (step < 0 || step >= shape[b]) {
                return false;
            }
            delta -= step * stride[b];
        }
    }
    return (delta == 0);
}

} // namespace yt::kernel

namespace yt{

template<typename T, typename Func>
YTensorBase& YTensorBase::binaryOpBroadcastInplace(const YTensorBase& other, Func&& func,
    const std::string& opName, double flop) {
    auto thisShape = this->shape();
    auto otherShape = other.shape();
    int thisDim = ndim();
    int otherDim = other.ndim();

    // other维度不能大于this
    if (otherDim > thisDim) {
        throwShapeNotMatch(opName, otherShape);
    }

    // 填充other的shape和stride到this的维度
    std::vector<int> paddedOtherShape(thisDim, 1);
    std::vector<int> paddedOtherStride(thisDim, 0);
    int otherOffset = thisDim - otherDim;
    for (int i = 0; i < otherDim; ++i) {
        paddedOtherShape[otherOffset + i] = otherShape[i];
        paddedOtherStride[otherOffset + i] = other._stride[i];
    }

    // 检查形状兼容性并调整stride
    bool equalShape = true;
    for (int i = 0; i < thisDim; ++i) {
        if (thisShape[i] != paddedOtherShape[i]) {
            if (paddedOtherShape[i] == 1) {
                paddedOtherStride[i] = 0;  // 广播维度stride设为0
            } else {
                throwShapeNotMatch(opName, otherShape);
            }
            equalShape = false;
        }
    }

    size_t totalSize = this->size();
    T* thisData = this->data<T>();
    const T* otherData = other.data<T>();

    // 快速路径：形状相同且连续
    if (equalShape && this->isContiguous() && other.isContiguous()) {
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {
            T thisVal = static_cast<T>(thisData[i]);
            T otherVal = static_cast<T>(otherData[i]);
            // 检查func是否有返回值（返回非void类型）
            if constexpr (std::is_void_v<std::invoke_result_t<Func, T&, const T&>>) {
                // void返回类型：func直接修改thisVal
                func(thisVal, otherVal);
                thisData[i] = thisVal;
            } else {
                // 非void返回类型：使用返回值
                thisData[i] = func(thisVal, otherVal);
            }
        }, flop);
        return *this;
    }

    // 通用路径：需要计算索引
    auto logicStride = this->stride();
    yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int index) {
        int thisIdx = 0, otherIdx = 0;
        int remaining = index;
        for (int i = 0; i < thisDim; ++i) {
            int coord = remaining / logicStride[i];
            remaining = remaining % logicStride[i];
            thisIdx += coord * _stride[i];
            otherIdx += coord * paddedOtherStride[i];
        }

        T thisVal = static_cast<T>(*(reinterpret_cast<const T*>(_data.get() + (_offset + thisIdx) * _element_size)));
        T otherVal = static_cast<T>(*(reinterpret_cast<const T*>(other._data.get() + (other._offset + otherIdx) * other._element_size)));

        T* destPtr = reinterpret_cast<T*>(_data.get() + (_offset + thisIdx) * _element_size);
        // 检查func是否有返回值（返回非void类型）
        if constexpr (std::is_void_v<std::invoke_result_t<Func, T&, const T&>>) {
            // void返回类型：func直接修改thisVal
            func(thisVal, otherVal);
            *destPtr = thisVal;
        } else {
            // 非void返回类型：使用返回值
            *destPtr = func(thisVal, otherVal);
        }
    }, flop);

    return *this;
}

template<typename T, typename Func>
YTensorBase YTensorBase::binaryOpBroadcast(const YTensorBase& other, Func&& func,
    [[maybe_unused]] const std::string& opName, YTensorBase* result, double flop) const {
    // opName保留用于将来的错误消息扩展（如broadcastShape错误时）
    // 计算输出形状
    auto opShape = yt::kernel::computeBroadcastShape({this->shape(), other.shape()});

    // 准备结果张量
    YTensorBase op;
    if (result != nullptr) {
        if (result->shape() != opShape) {
            *result = YTensorBase(opShape, yt::types::getTypeName<T>());
        }
        op = *result;
    } else {
        op = YTensorBase(opShape, yt::types::getTypeName<T>());
    }

    // 将this广播到结果张量
    // 这里简化实现：先复制this的数据到op（带广播），然后调用inplace版本
    int opDim = static_cast<int>(opShape.size());
    int thisDim = ndim();
    int otherDim = other.ndim();

    // 计算this和other相对于op的stride
    std::vector<int> thisStride(opDim, 0);
    std::vector<int> otherStride(opDim, 0);

    int thisLack = opDim - thisDim;
    int otheLack = opDim - otherDim;

    for (int i = 0; i < thisDim; ++i) {
        if (_shape[i] != 1) {
            thisStride[thisLack + i] = _stride[i];
        }
    }
    for (int i = 0; i < otherDim; ++i) {
        if (other._shape[i] != 1) {
            otherStride[otheLack + i] = other._stride[i];
        }
    }

    size_t totalSize = op.size();
    auto opLogicStride = op.stride();
    T* opData = op.data<T>();

    // 快速路径：this和other形状相同且都连续
    if(this->isContiguous() && other.isContiguous() && this->shapeMatch(other.shape())){
        const T* thisData = this->data<T>();
        const T* otherData = other.data<T>();
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {
            T thisVal = static_cast<T>(thisData[i]);
            T otherVal = static_cast<T>(otherData[i]);
            // 检查func是否有返回值（返回非void类型）
            if constexpr (std::is_void_v<std::invoke_result_t<Func, T&, const T&>>) {
                // void返回类型：func直接修改thisVal
                func(thisVal, otherVal);
                opData[i] = thisVal;
            } else {
                // 非void返回类型：使用返回值
                opData[i] = func(thisVal, otherVal);
            }
        }, flop);
        return op;
    }

    // 通用路径：需要计算索引
    yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int index) {
        int thisIdx = 0, otherIdx = 0;
        int remaining = index;
        for (int i = 0; i < opDim; ++i) {
            int coord = remaining / opLogicStride[i];
            remaining = remaining % opLogicStride[i];
            thisIdx += coord * thisStride[i];
            otherIdx += coord * otherStride[i];
        }

        T thisVal = static_cast<T>(*(reinterpret_cast<const T*>(_data.get() + (_offset + thisIdx) * _element_size)));
        T otherVal = static_cast<T>(*(reinterpret_cast<const T*>(other._data.get() + (other._offset + otherIdx) * other._element_size)));

        // 检查func是否有返回值（返回非void类型）
        if constexpr (std::is_void_v<std::invoke_result_t<Func, T&, const T&>>) {
            // void返回类型：func直接修改thisVal
            func(thisVal, otherVal);
            opData[index] = thisVal;
        } else {
            // 非void返回类型：使用返回值
            opData[index] = func(thisVal, otherVal);
        }
    }, flop);

    return op;
}

#define YT_DISPATCH_BY_DTYPE(dtype, BLOCK)                                                 \
    if (dtype == "float32") { using DType = float; BLOCK }                                 \
    else if (dtype == "float64") { using DType = double; BLOCK }                           \
    else if (dtype == "int8") { using DType = int8_t; BLOCK }                              \
    else if (dtype == "int16") { using DType = int16_t; BLOCK }                            \
    else if (dtype == "int32") { using DType = int32_t; BLOCK }                            \
    else if (dtype == "int64") { using DType = int64_t; BLOCK }                            \
    else if (dtype == "uint8") { using DType = uint8_t; BLOCK }                            \
    else if (dtype == "uint16") { using DType = uint16_t; BLOCK }                          \
    else if (dtype == "uint32") { using DType = uint32_t; BLOCK }                          \
    else if (dtype == "uint64") { using DType = uint64_t; BLOCK }                          \
    else if (dtype == "bfloat16") { using DType = yt::bfloat16; BLOCK }                    \
    else { throwOperatorNotSupport(dtype, "dispatch"); }

template<typename Func, typename... Args>
YTensorBase& YTensorBase::broadcastInplace(Func&& func, Args&&... tensors) {
    using namespace yt::traits;

    // 收集所有张量的shape（包括this）
    std::vector<std::vector<int>> shapes;
    shapes.push_back(this->shape());

    auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);

    // 计算广播shape
    auto broadcastShape = yt::kernel::computeBroadcastShape(shapes);
    int thisDim = this->ndim();

    // 验证this的shape与广播shape兼容
    auto thisShapeVec = this->shape();
    if (static_cast<int>(broadcastShape.size()) != thisDim) {
        throw std::runtime_error("broadcastInplace: result dimension mismatch");
    }
    for (int i = 0; i < thisDim; ++i) {
        if (thisShapeVec[i] != broadcastShape[i]) {
            throw std::runtime_error("broadcastInplace: this tensor shape must match broadcast shape");
        }
    }

    int totalSize = this->size();
    bool allContiguous = this->isContiguous();
    bool allEqualShape = true;

    auto checkContiguousAndShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (!arg.isContiguous()) {
                allContiguous = false;
            }
            auto argShape = arg.shape();
            if (argShape.size() != shapes[0].size()) {
                allEqualShape = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != shapes[0][i]) {
                        allEqualShape = false;
                        break;
                    }
                }
            }
        }
    };
    (checkContiguousAndShape(tensors), ...);

    // 根据dtype分发类型
    auto _dtype = this->dtype();
    YT_DISPATCH_BY_DTYPE(_dtype, {
        if (allContiguous && allEqualShape) {
            // 快速路径：所有张量连续且shape相同
            DType* thisDataPtr = this->data<DType>();

            // 收集所有张量的数据指针
            std::vector<const DType*> dataPtrs;
            auto collectPtrs = [&](auto&& arg) {
                if constexpr (is_ytensor_v<decltype(arg)>) {
                    if constexpr (is_ytensor_template_v<decltype(arg)>) {
                        dataPtrs.push_back(reinterpret_cast<const DType*>(arg.data()));
                    } else {
                        dataPtrs.push_back(arg.template data<DType>());
                    }
                }
            };
            (collectPtrs(tensors), ...);

            // 创建fastpath的getValue lambda
            auto getValueFast = [&](auto&& arg, int index, int& tensorIdx) -> decltype(auto) {
                if constexpr (is_ytensor_v<decltype(arg)>) {
                    ++tensorIdx;
                    return dataPtrs[tensorIdx - 1][index];
                } else {
                    return static_cast<DType>(std::forward<decltype(arg)>(arg));
                }
            };

            yt::kernel::parallelFor(0, totalSize, [&](int index) {
                int tensorIdx = 0;
                func(thisDataPtr[index], getValueFast(tensors, index, tensorIdx)...);
            });
        } else {
            // 慢速路径：需要计算广播索引
            DType* thisDataPtr = this->data<DType>();
            auto logicStride = this->stride();
            auto thisStride = this->stride_();

            // 收集每个张量的广播stride和数据指针
            std::vector<std::vector<int>> broadcastStrides;
            std::vector<const DType*> dataPtrs;

            auto collectBroadcastInfo = [&](auto&& arg) {
                if constexpr (is_ytensor_v<decltype(arg)>) {
                    broadcastStrides.push_back(yt::kernel::getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape));
                    if constexpr (is_ytensor_template_v<decltype(arg)>) {
                        dataPtrs.push_back(reinterpret_cast<const DType*>(arg.data()));
                    } else {
                        dataPtrs.push_back(arg.template data<DType>());
                    }
                }
            };
            (collectBroadcastInfo(tensors), ...);

            size_t numTensors = dataPtrs.size();

            yt::kernel::parallelFor(0, totalSize, [&](int index) {
                // 运行时计算索引
                int thisIdx = 0;
                std::vector<int> otherIndices(numTensors, 0);
                yt::kernel::computeBroadcastIndicesRuntime(index, logicStride, thisShapeVec, thisStride,
                    broadcastStrides, thisIdx, otherIndices, thisDim);

                // 获取值并调用func
                size_t tensorIdx = 0;
                auto getValue = [&](auto&& arg) -> decltype(auto) {
                    if constexpr (is_ytensor_v<decltype(arg)>) {
                        size_t idx = tensorIdx++;
                        return dataPtrs[idx][otherIndices[idx]];
                    } else {
                        return static_cast<DType>(std::forward<decltype(arg)>(arg));
                    }
                };

                func(thisDataPtr[thisIdx + _offset], getValue(tensors)...);
            });
        }
    });

    return *this;
}

// ======================== binaryOpTransformInplace ========================

template<typename T, typename Func>
YTensorBase& YTensorBase::binaryOpTransformInplace(const T& scalar, Func&& func, double flop) {
    size_t totalSize = this->size();

    if (this->isContiguous()) {
        T* dataPtr = this->data<T>();
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {
            T val = dataPtr[i];
            if constexpr (std::is_invocable_v<Func, T&, const T&>) {
                func(val, scalar);
                dataPtr[i] = val;
            } else {
                dataPtr[i] = func(val, scalar);
            }
        }, flop);
    } else {
        auto logicStride = this->stride();
        int dim = ndim();
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int index) {
            int physIdx = 0;
            int remaining = index;
            for (int i = 0; i < dim; ++i) {
                int coord = remaining / logicStride[i];
                remaining = remaining % logicStride[i];
                physIdx += coord * _stride[i];
            }

            T* ptr = reinterpret_cast<T*>(_data.get() + (_offset + physIdx) * _element_size);
            T val = *ptr;
            if constexpr (std::is_invocable_v<Func, T&, const T&>) {
                func(val, scalar);
                *ptr = val;
            } else {
                *ptr = func(val, scalar);
            }
        }, flop);
    }
    return *this;
}

// ======================== binaryOpTransform ========================

template<typename T, typename Func>
YTensorBase YTensorBase::binaryOpTransform(const T& scalar, Func&& func,
    YTensorBase* result, double flop) const {

    YTensorBase op;
    if (result != nullptr) {
        if (result->shape() != _shape) {
            *result = YTensorBase(_shape, yt::types::getTypeName<T>());
        }
        op = *result;
    } else {
        op = YTensorBase(_shape, yt::types::getTypeName<T>());
    }

    size_t totalSize = this->size();
    T* opData = op.data<T>();

    if (this->isContiguous()) {
        const T* thisData = this->data<T>();
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {
            T val = thisData[i];
            if constexpr (std::is_invocable_v<Func, T&, const T&>) {
                func(val, scalar);
                opData[i] = val;
            } else {
                opData[i] = func(val, scalar);
            }
        }, flop);
    } else {
        auto logicStride = this->stride();
        int dim = ndim();
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int index) {
            int physIdx = 0;
            int remaining = index;
            for (int i = 0; i < dim; ++i) {
                int coord = remaining / logicStride[i];
                remaining = remaining % logicStride[i];
                physIdx += coord * _stride[i];
            }

            const T* srcPtr = reinterpret_cast<const T*>(_data.get() + (_offset + physIdx) * _element_size);
            T val = *srcPtr;
            if constexpr (std::is_invocable_v<Func, T&, const T&>) {
                func(val, scalar);
                opData[index] = val;
            } else {
                opData[index] = func(val, scalar);
            }
        }, flop);
    }
    return op;
}

// ======================== 类型分发宏 ========================
// 用于在运行时根据dtype调用对应类型的模板函数

#define YT_DISPATCH_NUMERIC_TYPES(dtype, FUNC, ...)                          \
    do {                                                                      \
        if (dtype == "float32") { FUNC<float>(__VA_ARGS__); }                \
        else if (dtype == "float64") { FUNC<double>(__VA_ARGS__); }          \
        else if (dtype == "int8") { FUNC<int8_t>(__VA_ARGS__); }             \
        else if (dtype == "int16") { FUNC<int16_t>(__VA_ARGS__); }           \
        else if (dtype == "int32") { FUNC<int32_t>(__VA_ARGS__); }           \
        else if (dtype == "int64") { FUNC<int64_t>(__VA_ARGS__); }           \
        else if (dtype == "uint8") { FUNC<uint8_t>(__VA_ARGS__); }           \
        else if (dtype == "uint16") { FUNC<uint16_t>(__VA_ARGS__); }         \
        else if (dtype == "uint32") { FUNC<uint32_t>(__VA_ARGS__); }         \
        else if (dtype == "uint64") { FUNC<uint64_t>(__VA_ARGS__); }         \
        else if (dtype == "bfloat16") { FUNC<yt::bfloat16>(__VA_ARGS__); }   \
        else { YTensorBase::throwOperatorNotSupport(dtype, "dispatch"); }    \
    } while(0)

#define YT_DISPATCH_NUMERIC_TYPES_RET(dtype, FUNC, result, ...)              \
    do {                                                                      \
        if (dtype == "float32") { result = FUNC<float>(__VA_ARGS__); }       \
        else if (dtype == "float64") { result = FUNC<double>(__VA_ARGS__); } \
        else if (dtype == "int8") { result = FUNC<int8_t>(__VA_ARGS__); }    \
        else if (dtype == "int16") { result = FUNC<int16_t>(__VA_ARGS__); }  \
        else if (dtype == "int32") { result = FUNC<int32_t>(__VA_ARGS__); }  \
        else if (dtype == "int64") { result = FUNC<int64_t>(__VA_ARGS__); }  \
        else if (dtype == "uint8") { result = FUNC<uint8_t>(__VA_ARGS__); }  \
        else if (dtype == "uint16") { result = FUNC<uint16_t>(__VA_ARGS__); }\
        else if (dtype == "uint32") { result = FUNC<uint32_t>(__VA_ARGS__); }\
        else if (dtype == "uint64") { result = FUNC<uint64_t>(__VA_ARGS__); }\
        else if (dtype == "bfloat16") { result = FUNC<yt::bfloat16>(__VA_ARGS__); }\
        else { YTensorBase::throwOperatorNotSupport(dtype, "dispatch"); }    \
    } while(0)

// ======================== 算术运算符实现 ========================

// Eigen原生支持的类型分发宏（不包含bfloat16）
// 原因：Eigen的矩阵乘法需要完整的表达式模板支持，bfloat16的运算符模板与Eigen内部类型不兼容
#define YT_DISPATCH_EIGEN_NATIVE_TYPES(dtype, BLOCK)                                       \
    if (dtype == "float32") { using DType = float; BLOCK }                                 \
    else if (dtype == "float64") { using DType = double; BLOCK }                           \
    else if (dtype == "int8") { using DType = int8_t; BLOCK }                              \
    else if (dtype == "int16") { using DType = int16_t; BLOCK }                            \
    else if (dtype == "int32") { using DType = int32_t; BLOCK }                            \
    else if (dtype == "int64") { using DType = int64_t; BLOCK }                            \
    else if (dtype == "uint8") { using DType = uint8_t; BLOCK }                            \
    else if (dtype == "uint16") { using DType = uint16_t; BLOCK }                          \
    else if (dtype == "uint32") { using DType = uint32_t; BLOCK }                          \
    else if (dtype == "uint64") { using DType = uint64_t; BLOCK }                          \
    else { throwOperatorNotSupport(dtype, "eigen_native_dispatch"); }

// 带概念约束的类型分发宏 - 使用 if constexpr 在编译时检查类型是否支持运算符
// ConceptCheck: yt::concepts中的concept（如 yt::concepts::HAVE_MUL）
// BLOCK: 需要执行的代码块
// FALLBACK: 类型不支持时的处理
#define YT_DISPATCH_IF_TRAIT(dtype, ConceptCheck, BLOCK, FALLBACK)                         \
    if (dtype == "float32") {                                                              \
        using DType = float;                                                               \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "float64") {                                                         \
        using DType = double;                                                              \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "int8") {                                                            \
        using DType = int8_t;                                                              \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "int16") {                                                           \
        using DType = int16_t;                                                             \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "int32") {                                                           \
        using DType = int32_t;                                                             \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "int64") {                                                           \
        using DType = int64_t;                                                             \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "uint8") {                                                           \
        using DType = uint8_t;                                                             \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "uint16") {                                                          \
        using DType = uint16_t;                                                            \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "uint32") {                                                          \
        using DType = uint32_t;                                                            \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "uint64") {                                                          \
        using DType = uint64_t;                                                            \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else if (dtype == "bfloat16") {                                                        \
        using DType = yt::bfloat16;                                                        \
        if constexpr (ConceptCheck<DType>) { BLOCK } else { FALLBACK }                     \
    }                                                                                      \
    else { throwOperatorNotSupport(dtype, "trait_dispatch"); }

// 简化版：不支持时抛出异常
#define YT_DISPATCH_WITH_TRAIT(dtype, ConceptCheck, BLOCK)                                 \
    YT_DISPATCH_IF_TRAIT(dtype, ConceptCheck, BLOCK, { throwOperatorNotSupport(dtype, "op_not_supported"); })

// 整数类型分发宏
#define YT_DISPATCH_INT_TYPES(dtype, BLOCK)                                                \
    if (dtype == "int8") { using DType = int8_t; BLOCK }                                   \
    else if (dtype == "int16") { using DType = int16_t; BLOCK }                            \
    else if (dtype == "int32") { using DType = int32_t; BLOCK }                            \
    else if (dtype == "int64") { using DType = int64_t; BLOCK }                            \
    else if (dtype == "uint8") { using DType = uint8_t; BLOCK }                            \
    else if (dtype == "uint16") { using DType = uint16_t; BLOCK }                          \
    else if (dtype == "uint32") { using DType = uint32_t; BLOCK }                          \
    else if (dtype == "uint64") { using DType = uint64_t; BLOCK }                          \
    else { throwOperatorNotSupport(dtype, "int_op"); }

// 统一运算符宏 - 同时生成 Tensor op Tensor 和 Tensor op Scalar 的4个版本
#define YT_IMPL_BINARY_OP(OP, OP_NAME, DISPATCH_MACRO)                                     \
/* Tensor op Tensor */                                                                     \
inline YTensorBase YTensorBase::operator OP(const YTensorBase& other) const {              \
    YTensorBase result;                                                                    \
    DISPATCH_MACRO(_dtype, {                                                               \
        result = binaryOpBroadcast<DType>(other,                                           \
            [](DType& a, const DType& b) { a = a OP b; }, OP_NAME);                        \
    });                                                                                    \
    return result;                                                                         \
}                                                                                          \
inline YTensorBase& YTensorBase::operator OP##=(const YTensorBase& other) {                \
    DISPATCH_MACRO(_dtype, {                                                               \
        binaryOpBroadcastInplace<DType>(other,                                             \
            [](DType& a, const DType& b) { a = a OP b; }, OP_NAME "=");                    \
    });                                                                                    \
    return *this;                                                                          \
}                                                                                          \
/* Tensor op Scalar */                                                                     \
template<typename T>                                                                       \
YTensorBase YTensorBase::operator OP(const T& scalar) const {                              \
    return binaryOpTransform<T>(scalar, [](T& a, const T& b) { a = a OP b; });             \
}                                                                                          \
template<typename T>                                                                       \
YTensorBase& YTensorBase::operator OP##=(const T& scalar) {                                \
    return binaryOpTransformInplace<T>(scalar, [](T& a, const T& b) { a = a OP b; });      \
}

// 实例化所有运算符 - 数值类型
YT_IMPL_BINARY_OP(+, "+", YT_DISPATCH_BY_DTYPE)
YT_IMPL_BINARY_OP(-, "-", YT_DISPATCH_BY_DTYPE)
YT_IMPL_BINARY_OP(*, "*", YT_DISPATCH_BY_DTYPE)
YT_IMPL_BINARY_OP(/, "/", YT_DISPATCH_BY_DTYPE)

// 实例化所有运算符 - 仅整数类型
YT_IMPL_BINARY_OP(%, "%", YT_DISPATCH_INT_TYPES)
YT_IMPL_BINARY_OP(&, "&", YT_DISPATCH_INT_TYPES)
YT_IMPL_BINARY_OP(|, "|", YT_DISPATCH_INT_TYPES)
YT_IMPL_BINARY_OP(^, "^", YT_DISPATCH_INT_TYPES)

// 清理宏
#undef YT_IMPL_BINARY_OP
#undef YT_DISPATCH_INT_TYPES

// ======================== sum ========================

inline YTensorBase YTensorBase::sum(int axis) const {
    int dim = ndim();
    if (dim == 0) {
        throw std::runtime_error("[YTensorBase::sum] Cannot sum a 0-dim tensor");
    }
    axis = (axis % dim + dim) % dim;

    // 计算输出形状（对应轴设为1）
    auto newShape = this->shape();
    int axisSize = newShape[axis];
    newShape[axis] = 1;

    YTensorBase op(newShape, _dtype);
    size_t outSize = op.size();

    YT_DISPATCH_BY_DTYPE(_dtype, {
        DType* opData = op.data<DType>();
        yt::kernel::parallelFor(0, static_cast<int>(outSize), [&](int i) {
            auto coord = op.toCoord(i);
            DType sum = 0;
            for (int j = 0; j < axisSize; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                int physIdx = 0;
                for (int k = 0; k < dim; ++k) {
                    physIdx += subCoord[k] * _stride[k];
                }
                sum += *(reinterpret_cast<const DType*>(_data.get() + (_offset + physIdx) * _element_size));
            }
            opData[i] = sum;
        }, static_cast<double>(axisSize));
    });

    return op;
}

inline YTensorBase YTensorBase::sum(const std::vector<int>& axes) const {
    // 简单实现：逐个轴求和
    YTensorBase result = *this;
    // 排序axes，从大到小，避免轴索引变化问题
    std::vector<int> sortedAxes = axes;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    for (int ax : sortedAxes) {
        result = result.sum(ax);
    }
    return result;
}

// ======================== max ========================

inline std::pair<YTensorBase, YTensorBase> YTensorBase::max(int axis) const {
    int dim = ndim();
    if (dim == 0) {
        throw std::runtime_error("[YTensorBase::max] Cannot max a 0-dim tensor");
    }

    // 处理负索引
    axis = (axis % dim + dim) % dim;

    // 计算输出形状（对应轴设为1）
    auto newShape = this->shape();
    int axisSize = newShape[axis];
    newShape[axis] = 1;

    YTensorBase values(newShape, _dtype);
    YTensorBase indices(newShape, "int32");
    size_t outSize = values.size();

    YT_DISPATCH_BY_DTYPE(_dtype, {
        DType* valData = values.data<DType>();
        int32_t* idxData = indices.data<int32_t>();
        yt::kernel::parallelFor(0, static_cast<int>(outSize), [&](int i) {
            auto coord = values.toCoord(i);
            DType maxVal = std::numeric_limits<DType>::lowest();
            int32_t maxIdx = 0;
            for (int j = 0; j < axisSize; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                int physIdx = 0;
                for (int k = 0; k < dim; ++k) {
                    physIdx += subCoord[k] * _stride[k];
                }
                DType val = *(reinterpret_cast<const DType*>(_data.get() + (_offset + physIdx) * _element_size));
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = j;
                }
            }
            valData[i] = maxVal;
            idxData[i] = maxIdx;
        }, static_cast<double>(axisSize));
    });

    return {values, indices};
}

inline std::pair<YTensorBase, YTensorBase> YTensorBase::max(const std::vector<int>& axes) const {
    // 简单实现：逐个轴求最大值
    YTensorBase values = *this;
    YTensorBase indices;
    std::vector<int> sortedAxes = axes;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    for (int ax : sortedAxes) {
        auto [v, idx] = values.max(ax);
        values = v;
        indices = idx;  // 只保留最后一次的索引
    }
    return {values, indices};
}

// ======================== matView ========================

// 注意：YTensorBase的matView实现与YTensor不同
// YTensorBase是类型擦除的，我们返回一个特殊的YTensorBase，其"标量"是YTensorBase子视图
//
// 为什么这里需要手动处理非POD（YTensorBase）？
// 1. ytensor_base.inl 中的非POD支持是针对"用户通过registerType注册的自定义类型"
// 2. 但 YTensorBase 本身不能被注册（循环依赖：YTensorBase 在 ytensor_types.hpp 之前定义）
// 3. matView 返回的张量的元素类型就是 YTensorBase 本身，这是一个特殊情况
// 4. 因此必须手动使用 placement new 和自定义删除器来正确管理 YTensorBase 元素的生命周期

inline YTensorBase YTensorBase::matView() const {
    int dim = ndim();
    if (dim < 1) {
        throw std::runtime_error("[YTensorBase::matView] Tensor must have at least 1 dimension");
    }

    // 对于1D张量，视为1xN矩阵
    if (dim == 1) {
        // 创建一个2D视图 [1, shape[0]]
        YTensorBase mat2d;
        mat2d._shape = {1, _shape[0]};
        mat2d._stride = {0, _stride[0]};  // 第0维stride=0因为只有1行
        mat2d._offset = _offset;
        mat2d._data = _data;
        mat2d._element_size = _element_size;
        mat2d._dtype = _dtype;

        // 返回一个shape=[1]的YTensorBase，元素是YTensorBase类型
        // 由于YTensorBase不在类型注册表中，我们用特殊方式处理
        YTensorBase result;
        result._shape = {1};
        result._stride = {1};
        result._offset = 0;
        result._element_size = sizeof(YTensorBase);
        result._dtype = "YTensorBase";

        // 分配存储并放置构造YTensorBase元素
        result._data = std::shared_ptr<char[]>(
            new char[sizeof(YTensorBase)],
            [](char* p) {
                // 析构YTensorBase对象
                reinterpret_cast<YTensorBase*>(p)->~YTensorBase();
                delete[] p;
            }
        );
        new (result._data.get()) YTensorBase(mat2d);
        return result;
    }

    // 对于2D张量，返回shape=[1]的matView
    if (dim == 2) {
        YTensorBase result;
        result._shape = {1};
        result._stride = {1};
        result._offset = 0;
        result._element_size = sizeof(YTensorBase);
        result._dtype = "YTensorBase";

        result._data = std::shared_ptr<char[]>(
            new char[sizeof(YTensorBase)],
            [](char* p) {
                reinterpret_cast<YTensorBase*>(p)->~YTensorBase();
                delete[] p;
            }
        );
        // 直接复制当前张量作为2D视图
        new (result._data.get()) YTensorBase(*this);
        return result;
    }

    // 对于高维张量，前dim-2维是batch维度
    std::vector<int> batchShape(_shape.begin(), _shape.end() - 2);
    int matRows = _shape[dim - 2];
    int matCols = _shape[dim - 1];
    int matRowStride = _stride[dim - 2];
    int matColStride = _stride[dim - 1];

    // 计算batch数量
    size_t batchSize = 1;
    for (int s : batchShape) batchSize *= s;

    // 创建结果tensor
    YTensorBase result;
    result._shape = batchShape;
    result._element_size = sizeof(YTensorBase);
    result._dtype = "YTensorBase";

    // 计算stride（按连续存储）
    result._stride.resize(batchShape.size());
    if (!batchShape.empty()) {
        result._stride.back() = 1;
        for (int i = static_cast<int>(batchShape.size()) - 2; i >= 0; --i) {
            result._stride[i] = result._stride[i + 1] * batchShape[i + 1];
        }
    }
    result._offset = 0;

    // 使用封装函数分配存储
    result._data = yt::kernel::makeSharedPlacementArray<YTensorBase>(batchSize);

    // 为每个batch创建2D视图
    YTensorBase* dataPtr = reinterpret_cast<YTensorBase*>(result._data.get());

    // 计算原始tensor的batch stride（前dim-2维）
    std::vector<int> batchStride(_stride.begin(), _stride.end() - 2);

    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        // 计算坐标
        std::vector<int> coord(batchShape.size());
        int remaining = batchIdx;
        for (int i = static_cast<int>(batchShape.size()) - 1; i >= 0; --i) {
            coord[i] = remaining % batchShape[i];
            remaining /= batchShape[i];
        }

        // 计算偏移
        int batchOffset = 0;
        for (size_t i = 0; i < batchShape.size(); ++i) {
            batchOffset += coord[i] * batchStride[i];
        }

        // 创建2D视图
        YTensorBase mat2d;
        mat2d._shape = {matRows, matCols};
        mat2d._stride = {matRowStride, matColStride};
        mat2d._offset = _offset + batchOffset;
        mat2d._data = _data;  // 共享数据
        mat2d._element_size = _element_size;
        mat2d._dtype = _dtype;

        // placement new
        new (&dataPtr[batchIdx]) YTensorBase(std::move(mat2d));
    });

    return result;
}

// ======================== matmul ========================

// 模板化的naive matmul实现（零后端）
template<typename DType>
inline YTensorBase matmul_naive_impl(const YTensorBase& self, const YTensorBase& other) {
    // 获取matView
    auto thisMatView = self.matView();
    auto otherMatView = other.matView();

    // 获取矩阵维度
    int ah = (self.ndim() >= 2) ? self.shape(self.ndim() - 2) : 1;
    int aw = self.shape(self.ndim() - 1);
    int bw = other.shape(other.ndim() - 1);

    // 计算广播后的batch形状
    std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});

    // 计算输出的batch维度数量
    int thisBatchDim = std::max(0, self.ndim() - 2);
    int otherBatchDim = std::max(0, other.ndim() - 2);
    int opBatchDim = std::max(thisBatchDim, otherBatchDim);

    // 计算输出形状
    std::vector<int> opShape;
    int skipDims = static_cast<int>(opBatchShape.size()) - opBatchDim;
    for (int i = skipDims; i < static_cast<int>(opBatchShape.size()); ++i) {
        opShape.push_back(opBatchShape[i]);
    }
    opShape.push_back(ah);
    opShape.push_back(bw);

    // 创建输出tensor
    YTensorBase op(opShape, self.dtype());
    auto opMatView = op.matView();

    // 广播矩阵乘法
    size_t batchSize = opMatView.size();
    YTensorBase* opMats = opMatView.template data<YTensorBase>();
    YTensorBase* thisMats = thisMatView.template data<YTensorBase>();
    YTensorBase* otherMats = otherMatView.template data<YTensorBase>();

    auto getBroadcastIdx = [](int idx, const std::vector<int>& shape, const std::vector<int>& targetShape) -> int {
        if (shape.size() == 0) return 0;
        std::vector<int> coord(targetShape.size());
        int remaining = idx;
        for (int i = static_cast<int>(targetShape.size()) - 1; i >= 0; --i) {
            coord[i] = remaining % targetShape[i];
            remaining /= targetShape[i];
        }
        int result = 0;
        int stride = 1;
        int offset = static_cast<int>(targetShape.size()) - static_cast<int>(shape.size());
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            int c = coord[i + offset];
            if (shape[i] == 1) c = 0;
            result += c * stride;
            stride *= shape[i];
        }
        return result;
    };

    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        int thisIdx = getBroadcastIdx(batchIdx, thisMatView.shape(), opBatchShape);
        int otherIdx = getBroadcastIdx(batchIdx, otherMatView.shape(), opBatchShape);

        YTensorBase& A = thisMats[thisIdx];
        YTensorBase& B = otherMats[otherIdx];
        YTensorBase& C = opMats[batchIdx];

        for (int i = 0; i < ah; ++i) {
            for (int j = 0; j < bw; ++j) {
                DType sum = 0;
                for (int k = 0; k < aw; ++k) {
                    sum += A.template at<DType>({i, k}) * B.template at<DType>({k, j});
                }
                C.template at<DType>({i, j}) = sum;
            }
        }
    }, static_cast<double>(ah * bw * aw));

    return op;
}

inline YTensorBase YTensorBase::matmul(const YTensorBase& other) const {
    // 验证维度
    if (ndim() < 1 || other.ndim() < 1) {
        throw std::runtime_error("[YTensorBase::matmul] Both tensors must have at least 1 dimension");
    }

    // 检查类型匹配
    if (_dtype != other._dtype) {
        throw std::runtime_error("[YTensorBase::matmul] dtype mismatch: " + _dtype + " vs " + other._dtype);
    }

    // 获取矩阵维度
    [[maybe_unused]] int thisRows = (ndim() >= 2) ? _shape[ndim() - 2] : 1;
    int thisCols = _shape[ndim() - 1];
    int otherRows = (other.ndim() >= 2) ? other._shape[other.ndim() - 2] : 1;
    [[maybe_unused]] int otherCols = other._shape[other.ndim() - 1];

    if (thisCols != otherRows) {
        throw std::runtime_error("[YTensorBase::matmul] Inner dimensions mismatch: " +
            std::to_string(thisCols) + " vs " + std::to_string(otherRows));
    }

    // 在matmul层完成类型分发，直接调用模板化后端
#if YT_USE_EIGEN
    return matmul_eigen_backend(other);
#else
    // 无Eigen时使用naive实现
    YT_DISPATCH_BY_DTYPE(_dtype, {
        return matmul_naive_impl<DType>(*this, other);
    });
#endif

    // 不应该到达这里
    throw std::runtime_error("[YTensorBase::matmul] Unsupported dtype: " + _dtype);
}

// 保留原始的零后端接口（直接调用模板实现）
// 使用yt::concepts::HAVE_MUL概念检查，确保类型支持乘法运算
inline YTensorBase YTensorBase::matmul_zero_backend(const YTensorBase& other) const {
    YT_DISPATCH_WITH_TRAIT(_dtype, yt::concepts::HAVE_MUL, {
        return matmul_naive_impl<DType>(*this, other);
    });
    throw std::runtime_error("[YTensorBase::matmul_zero_backend] Unsupported dtype: " + _dtype);
}


inline void YTensorBase::throwOperatorNotSupport(const std::string& typeName, const std::string& opName) {
    throw std::runtime_error("[YTensorBase] Operator " + opName + " not support for type " + typeName);
}

inline void YTensorBase::throwShapeNotMatch(const std::string& opName, const std::vector<int>& otherShape) const {
    std::string thisShapeStr = "[";
    for (size_t i = 0; i < _shape.size(); ++i) {
        thisShapeStr += std::to_string(_shape[i]);
        if (i + 1 < _shape.size()) thisShapeStr += ", ";
    }
    thisShapeStr += "]";

    std::string otherShapeStr = "[";
    for (size_t i = 0; i < otherShape.size(); ++i) {
        otherShapeStr += std::to_string(otherShape[i]);
        if (i + 1 < otherShape.size()) otherShapeStr += ", ";
    }
    otherShapeStr += "]";

    throw std::runtime_error("[YTensorBase::" + opName + "] Shape mismatch: " + thisShapeStr + " vs " + otherShapeStr);
}

// ======================== Eigen Support ========================
#if YT_USE_EIGEN

// Eigen类型映射宏 - 使用Stride支持非连续数据
// OuterStride: 行间距离（对于RowMajor）
// InnerStride: 列间距离（对于RowMajor）
#define YT_EIGEN_STRIDED_MAP(Scalar, rows, cols, data, outerStride, innerStride) \
    Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, \
               0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>( \
        data, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outerStride, innerStride))

#define YT_EIGEN_CONST_STRIDED_MAP(Scalar, rows, cols, data, outerStride, innerStride) \
    Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, \
               0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>( \
        data, rows, cols, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(outerStride, innerStride))

// 辅助函数：获取广播索引
inline int _getBroadcastIdx(int idx, const std::vector<int>& shape, const std::vector<int>& targetShape) {
    if (shape.size() == 0) return 0;
    std::vector<int> coord(targetShape.size());
    int remaining = idx;
    for (int i = static_cast<int>(targetShape.size()) - 1; i >= 0; --i) {
        coord[i] = remaining % targetShape[i];
        remaining /= targetShape[i];
    }
    int result = 0;
    int stride = 1;
    int offset = static_cast<int>(targetShape.size()) - static_cast<int>(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        int c = coord[i + offset];
        if (shape[i] == 1) c = 0;
        result += c * stride;
        stride *= shape[i];
    }
    return result;
}

// 模板化的matmul Eigen批量乘法实现
template<typename T>
inline void matmul_eigen_batch_impl(
    YTensorBase* thisMats, YTensorBase* otherMats, YTensorBase* opMats,
    const std::vector<int>& thisMatViewShape, const std::vector<int>& otherMatViewShape,
    const std::vector<int>& opBatchShape,
    int ah, int aw, int bw, size_t batchSize
) {
    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        int thisIdx = _getBroadcastIdx(batchIdx, thisMatViewShape, opBatchShape);
        int otherIdx = _getBroadcastIdx(batchIdx, otherMatViewShape, opBatchShape);

        YTensorBase& A = thisMats[thisIdx];
        YTensorBase& B = otherMats[otherIdx];
        YTensorBase& C = opMats[batchIdx];

        // 使用Stride Map支持非连续数据
        auto eigenA = YT_EIGEN_CONST_STRIDED_MAP(T, ah, aw, A.template data<T>(), A.stride_(0), A.stride_(1));
        auto eigenB = YT_EIGEN_CONST_STRIDED_MAP(T, aw, bw, B.template data<T>(), B.stride_(0), B.stride_(1));
        auto eigenC = YT_EIGEN_STRIDED_MAP(T, ah, bw, C.template data<T>(), C.stride_(0), C.stride_(1));

        eigenC.noalias() = eigenA * eigenB;
    }, static_cast<double>(ah * bw * aw));
}

// 模板化的Eigen matmul后端实现
template<typename DType>
inline YTensorBase matmul_eigen_impl(const YTensorBase& self, const YTensorBase& other) {
    // 获取matView
    auto thisMatView = self.matView();
    auto otherMatView = other.matView();

    // 获取矩阵维度
    int ah = (self.ndim() >= 2) ? self.shape(self.ndim() - 2) : 1;
    int aw = self.shape(self.ndim() - 1);
    int bw = other.shape(other.ndim() - 1);

    // 计算广播后的batch形状
    std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});

    // 计算输出的batch维度数量
    int thisBatchDim = std::max(0, self.ndim() - 2);
    int otherBatchDim = std::max(0, other.ndim() - 2);
    int opBatchDim = std::max(thisBatchDim, otherBatchDim);

    // 计算输出形状
    std::vector<int> opShape;
    int skipDims = static_cast<int>(opBatchShape.size()) - opBatchDim;
    for (int i = skipDims; i < static_cast<int>(opBatchShape.size()); ++i) {
        opShape.push_back(opBatchShape[i]);
    }
    opShape.push_back(ah);
    opShape.push_back(bw);

    // 创建输出tensor
    YTensorBase op(opShape, self.dtype());
    auto opMatView = op.matView();

    size_t batchSize = opMatView.size();
    YTensorBase* opMats = opMatView.template data<YTensorBase>();
    YTensorBase* thisMats = thisMatView.template data<YTensorBase>();
    YTensorBase* otherMats = otherMatView.template data<YTensorBase>();

    auto thisShape = thisMatView.shape();
    auto otherShape = otherMatView.shape();

    matmul_eigen_batch_impl<DType>(thisMats, otherMats, opMats, thisShape, otherShape, opBatchShape, ah, aw, bw, batchSize);

    return op;
}

// 保留原始接口（直接调用模板实现）
inline YTensorBase YTensorBase::matmul_eigen_backend(const YTensorBase& other) const {
    // bfloat16需要特殊处理：转换为float32执行Eigen matmul，再转回bfloat16
    // 原因：Eigen内部使用表达式模板，需要与Eigen兼容的operator+=等，
    // 而yt::bfloat16的模板化运算符无法处理Eigen的表达式类型
    if (_dtype == "bfloat16") {
        YTensorBase thisF32(this->shape(), "float32");
        YTensorBase otherF32(other.shape(), "float32");

        const yt::bfloat16* thisBf16 = this->data<yt::bfloat16>();
        const yt::bfloat16* otherBf16 = other.data<yt::bfloat16>();
        float* thisF32Ptr = thisF32.data<float>();
        float* otherF32Ptr = otherF32.data<float>();

        for (size_t i = 0; i < this->size(); ++i) thisF32Ptr[i] = static_cast<float>(thisBf16[i]);
        for (size_t i = 0; i < other.size(); ++i) otherF32Ptr[i] = static_cast<float>(otherBf16[i]);

        YTensorBase opF32 = matmul_eigen_impl<float>(thisF32, otherF32);

        YTensorBase result(opF32.shape(), "bfloat16");
        const float* opF32Ptr = opF32.data<float>();
        yt::bfloat16* resultPtr = result.data<yt::bfloat16>();
        for (size_t i = 0; i < opF32.size(); ++i) resultPtr[i] = yt::bfloat16(opF32Ptr[i]);
        return result;
    }

    YT_DISPATCH_EIGEN_NATIVE_TYPES(_dtype, {
        return matmul_eigen_impl<DType>(*this, other);
    });

    throw std::runtime_error("[YTensorBase::matmul_eigen_backend] Unsupported dtype: " + _dtype);
}

// 模板化的applyEigenOp实现
template<typename T, typename Func>
inline YTensorBase _applyEigenOpImpl(
    const YTensorBase& self, Func&& func, int rows, int cols
) {
    auto thisMatView = self.matView();
    size_t batchSize = thisMatView.size();
    YTensorBase* thisMats = thisMatView.data<YTensorBase>();

    // 推断输出形状
    auto eigenMat0 = YT_EIGEN_CONST_STRIDED_MAP(T, rows, cols, thisMats[0].template data<T>(),
                                                thisMats[0].stride_(0), thisMats[0].stride_(1));
    auto result0 = func(eigenMat0);
    int outRows = static_cast<int>(result0.rows());
    int outCols = static_cast<int>(result0.cols());

    // 构建输出形状
    std::vector<int> outShape;
    for (int i = 0; i < self.ndim() - 2; ++i) {
        outShape.push_back(self.shape(i));
    }
    outShape.push_back(outRows);
    outShape.push_back(outCols);

    YTensorBase op(outShape, yt::types::getTypeName<T>());
    auto opMatView = op.matView();
    YTensorBase* opMats = opMatView.data<YTensorBase>();

    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        YTensorBase& A = thisMats[batchIdx];
        YTensorBase& C = opMats[batchIdx];

        auto eigenA = YT_EIGEN_CONST_STRIDED_MAP(T, rows, cols, A.template data<T>(), A.stride_(0), A.stride_(1));
        auto eigenC = YT_EIGEN_STRIDED_MAP(T, outRows, outCols, C.template data<T>(), C.stride_(0), C.stride_(1));

        eigenC = func(eigenA);
    }, static_cast<double>(rows * cols));

    return op;
}

template<typename Func>
inline YTensorBase YTensorBase::applyEigenOp(Func&& func, const std::string& opName) const {
    int dim = ndim();
    if (dim < 2) {
        throw std::runtime_error("[YTensorBase::" + opName + "] Tensor must have at least 2 dimensions");
    }

    int rows = _shape[dim - 2];
    int cols = _shape[dim - 1];

    // 类型分发
    if (_dtype == "float32") {
        return _applyEigenOpImpl<float>(*this, std::forward<Func>(func), rows, cols);
    } else if (_dtype == "float64") {
        return _applyEigenOpImpl<double>(*this, std::forward<Func>(func), rows, cols);
    } else if (_dtype == "int32") {
        return _applyEigenOpImpl<int32_t>(*this, std::forward<Func>(func), rows, cols);
    } else if (_dtype == "int64") {
        return _applyEigenOpImpl<int64_t>(*this, std::forward<Func>(func), rows, cols);
    } else if (_dtype == "bfloat16") {
        // bfloat16: 转换为float32执行
        YTensorBase thisF32 = YTensorBase(this->shape(), "float32");
        size_t thisSize = this->size();
        const yt::bfloat16* thisBf16 = this->data<yt::bfloat16>();
        float* thisF32Data = thisF32.data<float>();
        for (size_t i = 0; i < thisSize; ++i) thisF32Data[i] = static_cast<float>(thisBf16[i]);

        YTensorBase opF32 = _applyEigenOpImpl<float>(thisF32, std::forward<Func>(func), rows, cols);

        // 转回bfloat16
        YTensorBase op(opF32.shape(), "bfloat16");
        size_t opSize = op.size();
        const float* opF32Data = opF32.data<float>();
        yt::bfloat16* opBf16Data = op.data<yt::bfloat16>();
        for (size_t i = 0; i < opSize; ++i) opBf16Data[i] = yt::bfloat16(opF32Data[i]);
        return op;
    } else {
        throw std::runtime_error("[YTensorBase::" + opName + "] Unsupported dtype: " + _dtype);
    }
}

// 模板化的applyEigenBinaryOp实现
template<typename T, typename Func>
inline YTensorBase _applyEigenBinaryOpImpl(
    const YTensorBase& self, const YTensorBase& other, Func&& func,
    int aRows, int aCols, int bRows, int bCols
) {
    auto thisMatView = self.matView();
    auto otherMatView = other.matView();

    std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});

    int thisBatchDim = std::max(0, self.ndim() - 2);
    int otherBatchDim = std::max(0, other.ndim() - 2);
    int opBatchDim = std::max(thisBatchDim, otherBatchDim);

    size_t batchSize = 1;
    for (int s : opBatchShape) batchSize *= s;

    YTensorBase* thisMats = thisMatView.data<YTensorBase>();
    YTensorBase* otherMats = otherMatView.data<YTensorBase>();

    auto thisShape = thisMatView.shape();
    auto otherShape = otherMatView.shape();

    // 推断输出形状
    int thisIdx0 = _getBroadcastIdx(0, thisShape, opBatchShape);
    int otherIdx0 = _getBroadcastIdx(0, otherShape, opBatchShape);
    YTensorBase& A0 = thisMats[thisIdx0];
    YTensorBase& B0 = otherMats[otherIdx0];

    auto eigenA0 = YT_EIGEN_CONST_STRIDED_MAP(T, aRows, aCols, A0.template data<T>(), A0.stride_(0), A0.stride_(1));
    auto eigenB0 = YT_EIGEN_CONST_STRIDED_MAP(T, bRows, bCols, B0.template data<T>(), B0.stride_(0), B0.stride_(1));
    auto result0 = func(eigenA0, eigenB0);
    int outRows = static_cast<int>(result0.rows());
    int outCols = static_cast<int>(result0.cols());

    // 构建输出形状
    std::vector<int> opShape;
    int skipDims = static_cast<int>(opBatchShape.size()) - opBatchDim;
    for (int i = skipDims; i < static_cast<int>(opBatchShape.size()); ++i) {
        opShape.push_back(opBatchShape[i]);
    }
    opShape.push_back(outRows);
    opShape.push_back(outCols);

    YTensorBase op(opShape, yt::types::getTypeName<T>());
    auto opMatView = op.matView();
    YTensorBase* opMats = opMatView.data<YTensorBase>();

    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        int thisIdx = _getBroadcastIdx(batchIdx, thisShape, opBatchShape);
        int otherIdx = _getBroadcastIdx(batchIdx, otherShape, opBatchShape);

        YTensorBase& A = thisMats[thisIdx];
        YTensorBase& B = otherMats[otherIdx];
        YTensorBase& C = opMats[batchIdx];

        auto eigenA = YT_EIGEN_CONST_STRIDED_MAP(T, aRows, aCols, A.template data<T>(), A.stride_(0), A.stride_(1));
        auto eigenB = YT_EIGEN_CONST_STRIDED_MAP(T, bRows, bCols, B.template data<T>(), B.stride_(0), B.stride_(1));
        auto eigenC = YT_EIGEN_STRIDED_MAP(T, outRows, outCols, C.template data<T>(), C.stride_(0), C.stride_(1));

        eigenC = func(eigenA, eigenB);
    }, static_cast<double>(aRows * aCols + bRows * bCols));

    return op;
}

template<typename Func>
inline YTensorBase YTensorBase::applyEigenBinaryOp(const YTensorBase& other, Func&& func, const std::string& opName) const {
    int dim = ndim();
    int otherDim = other.ndim();
    if (dim < 2 || otherDim < 2) {
        throw std::runtime_error("[YTensorBase::" + opName + "] Both tensors must have at least 2 dimensions");
    }

    if (_dtype != other._dtype) {
        throw std::runtime_error("[YTensorBase::" + opName + "] dtype mismatch: " + _dtype + " vs " + other._dtype);
    }

    int aRows = _shape[dim - 2];
    int aCols = _shape[dim - 1];
    int bRows = other._shape[otherDim - 2];
    int bCols = other._shape[otherDim - 1];

    // 类型分发
    if (_dtype == "float32") {
        return _applyEigenBinaryOpImpl<float>(*this, other, std::forward<Func>(func), aRows, aCols, bRows, bCols);
    } else if (_dtype == "float64") {
        return _applyEigenBinaryOpImpl<double>(*this, other, std::forward<Func>(func), aRows, aCols, bRows, bCols);
    } else if (_dtype == "int32") {
        return _applyEigenBinaryOpImpl<int32_t>(*this, other, std::forward<Func>(func), aRows, aCols, bRows, bCols);
    } else if (_dtype == "int64") {
        return _applyEigenBinaryOpImpl<int64_t>(*this, other, std::forward<Func>(func), aRows, aCols, bRows, bCols);
    } else if (_dtype == "bfloat16") {
        // bfloat16: 转换为float32执行
        YTensorBase thisF32 = YTensorBase(this->shape(), "float32");
        YTensorBase otherF32 = YTensorBase(other.shape(), "float32");

        size_t thisSize = this->size();
        size_t otherSize = other.size();
        const yt::bfloat16* thisBf16 = this->data<yt::bfloat16>();
        const yt::bfloat16* otherBf16 = other.data<yt::bfloat16>();
        float* thisF32Data = thisF32.data<float>();
        float* otherF32Data = otherF32.data<float>();

        for (size_t i = 0; i < thisSize; ++i) thisF32Data[i] = static_cast<float>(thisBf16[i]);
        for (size_t i = 0; i < otherSize; ++i) otherF32Data[i] = static_cast<float>(otherBf16[i]);

        YTensorBase opF32 = _applyEigenBinaryOpImpl<float>(thisF32, otherF32, std::forward<Func>(func), aRows, aCols, bRows, bCols);

        // 转回bfloat16
        YTensorBase op(opF32.shape(), "bfloat16");
        size_t opSize = op.size();
        const float* opF32Data = opF32.data<float>();
        yt::bfloat16* opBf16Data = op.data<yt::bfloat16>();
        for (size_t i = 0; i < opSize; ++i) opBf16Data[i] = yt::bfloat16(opF32Data[i]);
        return op;
    } else {
        throw std::runtime_error("[YTensorBase::" + opName + "] Unsupported dtype: " + _dtype);
    }
}

#undef YT_EIGEN_STRIDED_MAP
#undef YT_EIGEN_CONST_STRIDED_MAP

#endif // YT_USE_EIGEN

} // namespace yt
/***************
* @file: ytensor_io.hpp
* @brief: YTensor/YTensorBase 类的文件输入/输出功能。
***************/

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <zlib.h>


// 前向声明 YTensor
template<typename T, int dim>
class YTensor;

namespace yt::io {

/// @brief 文件头标识
using yt::infos::YTENSOR_FILE_MAGIC;

/// @brief 文件版本
using yt::infos::YTENSOR_FILE_VERSION;

/// @brief 压缩级别，在zlib压缩方式中使用
static int8_t compressLevel = Z_DEFAULT_COMPRESSION;

/// @brief 是否输出警告信息
static bool verbose = false;

static constexpr int Closed = 0;
static constexpr int Read = 1;
static constexpr int Write = 2;
static constexpr int Append = 3;

/// @brief 压缩方法
/// @enum "": 不压缩
/// @enum "zlib": 使用zlib压缩，compressLevel控制压缩级别
/// @enum "zlibfloat": 优化后的浮点数专用的压缩方法
static std::string compressMethod = "";

/// @brief 检查压缩方法
/// @param method 压缩方法
/// @return 返回检查后的方法，如果方法无效，回退为空字符串，表示不压缩
std::string checkCompressMethod(const std::string& method);

/// @brief 将字符串转换为写入文件的字节流
std::vector<char> string2data(const std::string& str);

/// @brief 将文件中的字节流转换为字符串
std::string data2string(std::fstream& file, bool seek = true);

/// @brief 将数组转换为写入文件的字节流
template<typename T>
std::vector<char> array2data(const std::vector<T>& data);

/// @brief 从文件中读取数组数据
std::vector<char> data2array(std::fstream& file, bool seek = true);

/// @brief 压缩数据
template<typename T>
std::vector<char> compressData(const std::vector<T>& input);

/// @brief 解压数据
std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize = 0, const std::string& method = "");

/// @brief 从文件中解压数据
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

/// @brief YTensor/YTensorBase 专用的 IO 类
class YTensorIO {
public:
    YTensorIO() = default;
    ~YTensorIO();

    /// @brief 打开文件
    /// @param fileName 文件名
    /// @param fileMode 文件模式：Read, Write, Append
    /// @return 如果文件打开成功，返回true；否则返回false。
    bool open(const std::string& fileName, int fileMode = yt::io::Read);

    /// @brief 关闭文件
    void close();

    /// @brief 获取文件中所有张量的名称
    /// @return 张量名称列表
    std::vector<std::string> getTensorNames() const;

    /// @brief 获取张量信息
    /// @param name 张量名称，缺省表示获取第一个张量
    /// @return 张量信息，如果不存在抛出异常
    TensorInfo getTensorInfo(const std::string& name = "") const;

    /// @brief 保存张量数据
    /// @param tensor 需要保存的张量
    /// @param name 张量名称
    /// @return 如果保存成功，返回true；否则返回false。
    bool save(const yt::YTensorBase& tensor, const std::string& name);

    /// @brief 保存张量数据 (模板版本)
    /// @tparam T 张量元素类型
    /// @tparam dim 张量维度
    /// @param tensor 需要保存的张量
    /// @param name 张量名称
    /// @return 如果保存成功，返回true；否则返回false。
    template<typename T, int dim>
    bool save(const YTensor<T, dim>& tensor, const std::string& name);

    /// @brief 加载张量数据
    /// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
    /// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
    /// @return 如果加载成功，返回true；否则返回false。
    bool load(yt::YTensorBase& tensor, const std::string& name = "");

    /// @brief 加载张量数据 (模板版本)
    /// @tparam T 张量元素类型
    /// @tparam dim 张量维度
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

/// @brief 便利函数用于快速保存 YTensorBase
/// @param fileName 文件名
/// @param tensor 需要保存的张量
/// @param name 张量名称
/// @return 如果保存成功，返回true；否则返回false。
bool saveTensorBase(const std::string& fileName, const yt::YTensorBase& tensor, const std::string& name = "");

/// @brief 便利函数用于快速加载 YTensorBase
/// @param fileName 文件名
/// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
/// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
/// @return 如果加载成功，返回true；否则返回false。
bool loadTensorBase(const std::string& fileName, yt::YTensorBase& tensor, const std::string& name = "");

/// @brief 便利函数用于快速保存 YTensor (模板版本)
/// @tparam T 张量元素类型
/// @tparam dim 张量维度
/// @param fileName 文件名
/// @param tensor 需要保存的张量
/// @param name 张量名称
/// @return 如果保存成功，返回true；否则返回false。
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const YTensor<T, dim>& tensor, const std::string& name = "");

/// @brief 便利函数用于快速加载 YTensor (模板版本)
/// @tparam T 张量元素类型
/// @tparam dim 张量维度
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

inline std::string checkCompressMethod(const std::string& method) {
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

inline std::vector<char> string2data(const std::string& str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    std::vector<char> op(sizeof(uint32_t) + length);
    std::memcpy(op.data(), &length, sizeof(uint32_t));
    if (!str.empty()) {
        std::memcpy(op.data() + sizeof(uint32_t), str.c_str(), length);
    }
    return op;
}

inline std::string data2string(std::fstream& file, bool seek) {
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

inline std::vector<char> data2array(std::fstream& file, bool seek) {
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

inline std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize, const std::string& method) {
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
            inflateEnd(&stream);
            if (yt::io::verbose) {
                std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
            }
            return {};
        }
        size_t chunk_used = first_chunk_size - stream.avail_out;
        current_chunk.resize(chunk_used);
        // 如果还有剩余数据需要解压，继续处理
        while (result != Z_STREAM_END && stream.avail_out == 0) {
            chunks.emplace_back(chunk_size);
            std::vector<char>& next_chunk = chunks.back();

            stream.avail_out = chunk_size;
            stream.next_out = reinterpret_cast<Bytef*>(next_chunk.data());

            result = inflate(&stream, Z_NO_FLUSH);

            if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
                inflateEnd(&stream);
                if (yt::io::verbose) {
                    std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
                }
                return {};
            }

            chunk_used = chunk_size - stream.avail_out;
            next_chunk.resize(chunk_used);
        }
        inflateEnd(&stream);
        if (result != Z_STREAM_END) {
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

inline std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize, bool seek, const std::string& method) {
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

        const size_t io_chunk = 65536; // 64KB 输入缓冲
        std::vector<char> input_buffer(io_chunk);
        size_t remaining = compressedSize;

        if (decompressedSize > 0) {
            output.resize(decompressedSize);
        } else {
            output.resize(io_chunk);
        }

        size_t out_pos = 0;

        stream.avail_out = static_cast<uInt>(output.size() - out_pos);
        stream.next_out = reinterpret_cast<Bytef*>(output.data() + out_pos);

        int result = Z_OK;
        while (result != Z_STREAM_END) {
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

            if (stream.avail_out == 0) {
                size_t add = io_chunk;
                if (decompressedSize > 0) {
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

            size_t wrote = (output.size() - out_pos) - stream.avail_out;
            out_pos += wrote;

            if (result == Z_STREAM_END) break;

            if (stream.avail_in == 0 && remaining == 0 && stream.avail_out > 0 && result == Z_BUF_ERROR) {
                break;
            }
        }

        inflateEnd(&stream);
        output.resize(out_pos);

    } else {
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

inline YTensorIO::~YTensorIO() {
    close();
}

inline bool YTensorIO::open(const std::string& fileName, int fileMode) {
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

inline void YTensorIO::close() {
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

inline std::vector<std::string> YTensorIO::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& tensorInfo : _tensorInfos) {
        names.push_back(tensorInfo.name);
    }
    return names;
}

inline TensorInfo YTensorIO::getTensorInfo(const std::string& name) const {
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

inline bool YTensorIO::validateFile() {
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

inline bool YTensorIO::readHeader() {
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

inline bool YTensorIO::writeHeader() {
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

inline bool YTensorIO::readIndex() {
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

inline bool YTensorIO::writeIndex(std::vector<uint64_t> offsets) {
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

inline bool YTensorIO::save(const yt::YTensorBase& tensor, const std::string& name) {
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

    // 需要保证是连续的，否则数据排布不正确
    yt::YTensorBase contiguousTensor = tensor.clone();

    // 创建张量信息
    TensorInfo info;
    info.name = tensorName;
    info.typeName = tensor.dtype();
    info.typeSize = static_cast<int32_t>(tensor.elementSize());
    info.tensorType = "dense";

    // 获取形状
    auto shape = contiguousTensor.shape();
    info.shape.resize(shape.size());
    std::transform(shape.begin(), shape.end(), info.shape.begin(), [](int s) {
        return static_cast<int32_t>(s);
    });

    // 准备并压缩张量数据
    info.compressMethod = checkCompressMethod(yt::io::compressMethod);

    // 获取原始数据
    size_t dataSize = contiguousTensor.size() * contiguousTensor.elementSize();
    std::vector<char> rawData(dataSize);
    std::memcpy(rawData.data(), contiguousTensor.data(), dataSize);

    // 压缩数据
    auto compressedData = compressData(rawData);
    if (compressedData.empty() && dataSize > 0) {
        if (verbose) {
            std::cerr << "Error: Failed to compress tensor data" << std::endl;
        }
        return false;
    }

    info.compressedSize = static_cast<uint64_t>(compressedData.size());
    info.uncompressedSize = dataSize;

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

// 模板方法实现：保存 YTensor<T, dim>
template<typename T, int dim>
bool YTensorIO::save(const YTensor<T, dim>& tensor, const std::string& name) {
    // YTensor 继承自 YTensorBase，直接调用基类版本
    return save(static_cast<const yt::YTensorBase&>(tensor), name);
}

inline bool YTensorIO::load(yt::YTensorBase& tensor, const std::string& name) {
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
            std::cerr << "Error: YTensorBase data size mismatch. Expected " << info.uncompressedSize
                      << ", got " << rawData.size() <<
                       ". Please check your file." << std::endl;
        }
        return false; // Decompressed data size mismatch
    }

    // 使用文件中的 dtype 创建张量
    tensor = yt::YTensorBase(shape, info.typeName);

    // Copy decompressed data to tensor
    if (!rawData.empty()) {
        std::memcpy(tensor.data(), rawData.data(), rawData.size());
    }
    return true;
}

// 模板方法实现：加载 YTensor<T, dim>
template<typename T, int dim>
bool YTensorIO::load(YTensor<T, dim>& tensor, const std::string& name) {
    // 先加载为 YTensorBase
    yt::YTensorBase base;
    if (!load(base, name)) {
        return false;
    }
    // 转换为 YTensor<T, dim>
    tensor = YTensor<T, dim>(base);
    return true;
}

// 便利函数实现
inline bool saveTensorBase(const std::string& fileName, const yt::YTensorBase& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Write)) {
        return false;
    }
    if (!io.save(tensor, name)) {
        return false;
    }
    io.close();
    return true;
}

inline bool loadTensorBase(const std::string& fileName, yt::YTensorBase& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Read)) {
        return false;
    }
    return io.load(tensor, name);
}

// 便利函数模板实现
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const YTensor<T, dim>& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Write)) {
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
    if (!io.open(fileName, yt::io::Read)) {
        return false;
    }
    return io.load(tensor, name);
}

} // namespace yt::io


/**
 * @brief 易于使用的张量类。可以处理任意维度的张量。
 * @tparam T 张量元素的数据类型。
 * @tparam dim 张量的维度数。
 */
template <typename T=float, int dim=1>
class YTensor : public yt::YTensorBase {
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
    /// @example YTensor<float, 3> a = {3, 4, 5};
    YTensor(std::initializer_list<int> list);

    /// @brief 从 YTensorBase 构造。
    /// @param base 源 YTensorBase 对象。
    /// @note 类型和维度必须匹配，否则行为未定义。
    explicit YTensor(const YTensorBase& base);

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

    /// @brief 预留连续空间。注意原本的数据会被清空。与构造函数的逻辑相同。
    /// @param args 张量的形状。
    /// @return 返回当前的张量的引用。
    /// @example auto t = YTensor<float, 3>().reserve(3, 4, 5);
    template<typename... Args>
    YTensor& reserve(Args... args);

    /// @brief 获取张量的数据指针，可以使用stride_()的步长进行安全访问。已经包含偏移量。
    /// @return 返回张量数据的指针。
    /// @note 如果需要获取内存段的数据指针，请使用data_()方法。
    T* data();
    const T *data() const;

    /// @brief 获取张量数据区的指针（在非contiguous情况下不建议使用）
    /// @return 返回张量数据的指针。
    T *data_();
    const T *data_() const;

    using YTensorBase::size;
    using YTensorBase::shape;
    using YTensorBase::shapeMatch;
    using YTensorBase::shape_;
    using YTensorBase::stride;
    using YTensorBase::stride_;
    using YTensorBase::isContiguous;
    using YTensorBase::isContiguousFrom;
    using YTensorBase::isDisjoint;
    using YTensorBase::toCoord;

    /// @brief 获取张量的维度数（编译时常量，区别于基类的运行时版本）
    constexpr int shapeSize() const;

    /// @brief 获取张量相对数据指针的真实偏移量，或者对应子张量或元素的偏移量
    /// @param index 元素的索引
    /// @return 返回张量、子张量或者对应元素的偏移量。
    template<typename... Args> int offset(Args... index) const;
    int offset(const std::vector<int>& index) const;

    /// @brief 获取张量的物理偏移量，考虑了张量自身的_offset
    /// @param index 元素的索引
    /// @return 返回 _offset + offset(index)
    template<typename... Args> int offset_(Args... index) const;
    int offset_(const std::vector<int>& index) const;

    /// @brief 获取张量的连续版本
    /// @return 返回连续张量。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    YTensor contiguous() const;

    /// @brief 原地操作，使张量连续
    /// @return 返回自身的引用。
    /// @example YTensor<float, 3> a(3, 4, 5); a.contiguous();
    /// @note 这会导致原先的“引用”失效
    YTensor &contiguous_();

    /// @brief 获取这个张量尽可能连续的视图，注意张量的形状会发生变化。
    /// @return 返回一个张量，表示当前张量最大可能的连续排布的视图。
    /// @note 这个张量并不能保证连续。适用于处理elementwise操作加速。
    YTensor mostContinuousView() const;

    /// @brief 获取坐标对应的位置，可以使用atData()方法获取数据。
    /// @return 偏移量
    /// @example YTensor<float, 3> a(3, 4, 5); auto offset = a.toIndex(1, 2, 3); // offset = 31
    /// @note 如果张量是非contiguous的，toIndex()方法返回的是逻辑索引，而不是实际的内存索引。
    //       如果需要获取实际的内存索引，请使用toIndex_()方法
    template <typename... Args> size_t toIndex(const Args... args) const;
    size_t toIndex(const std::vector<int> &pos) const;

    /// @brief 获取坐标对应的物理位置，可以使用atData_()方法获取数据。
    /// @return 相对data的偏移量
    /// @example YTensor<float, 3> a(3, 4, 5); auto offset = a.toIndex_(1, 2, 3); // offset = 31
    template <typename... Args> size_t toIndex_(const Args... args) const;
    size_t toIndex_(const std::vector<int> &pos) const;

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
    template <typename... Args> const T& at(const Args... args) const;
    const T& at(const std::vector<int> &pos) const;

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

    /// @brief 重塑张量形状
    /// @param newShape 新的形状
    /// @return 返回一个新的张量（可能是视图，也可能是拷贝）
    template <typename... Args> auto reshape(const Args... newShape) const -> YTensor<T, sizeof...(Args)>;
    template <int newdim> YTensor<T, newdim> reshape(const std::vector<int>& newShape) const;

    /// @brief 在指定位置插入一个大小为1的维度（零拷贝）
    /// @param d 插入的位置（支持负索引）
    /// @return 返回新维度的张量视图
    YTensor<T, dim + 1> unsqueeze(int d) const;

    /// @brief 移除指定位置的大小为1的维度（零拷贝）
    /// @param d 要移除的维度（支持负索引）
    /// @return 返回减少维度后的张量视图
    /// @note dim 必须大于1，否则编译失败
    YTensor<T, dim - 1> squeeze(int d) const requires (dim > 1);

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
    template<typename... Args> static YTensor<T, sizeof...(Args)> zeros(Args... args);

    /// @brief 创建指定大小的，全1张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensor<T, dim> ones(const std::vector<int>& shape);
    template<typename... Args> static YTensor<T, sizeof...(Args)> ones(const Args... args);

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

    /// @brief 按位置进行操作
    /// @param func 对每个元素进行操作的函数：
    ///        - 带坐标版本：void/T func(T& value, const std::vector<int>& coord)
    ///        - 无坐标版本：void/T func(T& value)
    /// @param flop 每次操作的运算量，用于判断是否需要多线程计算
    /// @return 返回自身引用
    /// @note 当func只接受一个参数时，会自动使用更高效的实现
    template<typename Func>
    YTensor<T, dim>& foreach(Func&& func, double flop = 1e-11);

    /// @brief 填充张量
    /// @param value 填充的值
    /// @return 返回自身引用
    YTensor<T, dim>& fill(T value);

    /// @brief 从源张量复制元素到本张量（原地操作，不重新分配内存）
    /// @param src 源张量，shape和dtype必须与本张量一致
    /// @return 返回自身引用
    /// @note 目前不支持src与dst的内存重叠，若存在重叠则行为未定义
    YTensor<T, dim>& copy_(const yt::YTensorBase& src);

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

    /// @brief 最大子元素中量遍历父张量阈值，超过则使用stride遍历法，否则使用布尔掩码遍历底层存储。
    static constexpr double MAX_SUBELEMENT_RATIO = 2.5;

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

    /// @brief 统一的广播原地操作函数，支持N元张量/标量操作（转发到yt::kernel::broadcastInplace）
    /// @tparam Func 函数类型，签名为 void func(T&, const T&, ...) 或返回值被忽略
    /// @tparam Args 参数类型，可以是YTensor或标量T
    /// @param func 操作函数，第一个参数为this的元素引用
    /// @param tensors 输入的张量或标量
    /// @return 返回自身引用
    template <typename Func, typename... Args>
    YTensor<T, dim>& broadcastInplace(Func&& func, Args&&... tensors);

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

    /// @brief 对指定轴求均值
    /// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
    /// @return 均值结果
    YTensor<T, dim> mean(int axis) const requires(dim > 1);
    YTensor<T, dim> mean(std::vector<int> axes) const requires (dim > 1);
    T mean(int axis = 0) const requires (dim == 1);

    /// @brief 对指定轴求最大值
    /// @param axis: 轴索引，当张量的维度为1时，取值无关结果。
    /// @return 最大值及其索引
    std::pair<YTensor<T, dim>, YTensor<int, dim>> max(int axis) const requires (dim > 1);
    std::pair<YTensor<T, dim>, YTensor<int, dim>> max(std::vector<int> axes) const requires (dim > 1);
    std::pair<T, int> max(int axis = 0) const requires (dim == 1);

    // protected:

    /// @brief 矩阵乘法的无优化后端实现，只保证规则正确，相当低效。
    /// @param other: 右张量输入。
    /// @return 矩阵乘法结果张量。
    template<int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_zero_backend(const YTensor<T, dim1>& other) const;

    /////////////// Eigen support ///////////////
    #if YT_USE_EIGEN
    /// @brief Eigen类型转换
    using EigenMatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>>;

    /// @brief 转换为Eigen矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
    YTensor<EigenMatrixMap, std::max(1, dim - 2)> matViewEigen() const requires(dim > 2);

    /// @brief 矩阵视图，将张量的最后两个维度视为EigenMatrixMap标量。
    EigenMatrixMap matViewEigen() const requires(dim <= 2);

    /// @brief 矩阵乘法的Eigen后端实现，开启Eigen时为默认的后端。
    template<int dim1>
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> matmul_eigen_backend(const YTensor<T, dim1>& other) const;

    #endif // YT_USE_EIGEN
    public: // end of ytensor_math.hpp

    // ********************************
    // TODO:
    // 1. <<左右移运算符仍然未支持
    // 2. 返回布尔值的运算符尚未支持
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

// 辅助函数：安全获取类型名
template<typename T>
inline std::string safeGetTypeName() {
    if constexpr (std::is_arithmetic_v<T>) {
        return yt::types::getTypeName<T>();
    } else {
        return "complex_type";
    }
}

///////////////// ytensor ///////////////

template <typename T, int dim>
YTensor<T, dim>::YTensor():
    YTensorBase() {
    _shape.resize(dim, 0);
    _stride.resize(dim, 0);
    _element_size = sizeof(T);
    _dtype = safeGetTypeName<T>();
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(const std::vector<int> shape): YTensorBase() {
    if(shape.size() != dim){
        throwShapeSizeNotMatch("init", shape.size());
    }
    _shape = shape;
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = safeGetTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
template <typename... Args>
YTensor<T, dim>::YTensor(Args... args): YTensorBase() {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    _shape.resize(dim);
    int a = 0;
    ((_shape[a++] = args), ...);
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = safeGetTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(std::initializer_list<int> list): YTensorBase() {
    if (list.size() != dim) {
        throwShapeSizeNotMatch("init", list.size());
    }
    _shape = std::vector<int>(list);
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = safeGetTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
YTensor<T, dim>::YTensor(const YTensorBase& base): YTensorBase(base) {
    // 从 YTensorBase 构造 YTensor
    // 用户需要确保类型和维度匹配
    if (static_cast<int>(base.ndim()) != dim) {
        throwShapeSizeNotMatch("YTensorBase", base.ndim());
    }
}

template<typename T, int dim>
YTensor<T, dim>::YTensor(const YTensor& other): YTensorBase() {
    if(this == &other){
        return;
    }
    if (dim != other.shapeSize()){
        throwShapeSizeNotMatch("<copy>", other.shapeSize());
    }
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;
    _element_size = other._element_size;
    _dtype = other._dtype;
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
    _element_size = other._element_size;
    _dtype = other._dtype;
    _data = other._data;
    return *this;
}

template <typename T, int dim>
void YTensor<T, dim>::shallowCopyTo(YTensor<T, dim> &other) const {
    YTensorBase::shallowCopyTo(other);
}

template <typename T, int dim>
void YTensor<T, dim>::shareTo(YTensor<T, dim> &other) const {
    shallowCopyTo(other);
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::shallowCopyFrom(const YTensor<T, dim> &src) {
    src.YTensorBase::shallowCopyTo(*this);
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
    T* opData = op.data_();
    const T* thisData = this->data_();
    #pragma omp simd
    for(size_t dst = 0; dst < max; dst++){
        size_t thisIndex = _offset;
        size_t index = dst;
        #pragma omp simd reduction(+:thisIndex)
        for (int i = dim - 1; i >= 0; i--) {
            thisIndex += (index % _shape[i]) * _stride[i];
            index /= _shape[i];
        }
        opData[dst] = thisData[thisIndex];
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
    _element_size = sizeof(T);
    _dtype = safeGetTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
    return *this;
}

template <typename T, int dim>
template<typename... Args>
YTensor<T, dim>& YTensor<T, dim>::reserve(Args... args) {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return reserve(std::vector<int>{args...});
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
    return reinterpret_cast<T*>(_data.get());
}

template <typename T, int dim>
const T* YTensor<T, dim>::data_() const {
    return reinterpret_cast<const T*>(_data.get());
}

template<typename T, int dim>
constexpr int YTensor<T, dim>::shapeSize() const {
    return dim;
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

template <typename T, int dim> template <typename... Args>
int YTensor<T, dim>::offset_(Args... index) const {
    return _offset + this->offset(index...);
}

template <typename T, int dim>
int YTensor<T, dim>::offset_(const std::vector<int>& index) const {
    return _offset + this->offset(index);
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
        _element_size = t._element_size;
        _dtype = t._dtype;
        return *this;
    }
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::mostContinuousView() const {
    return YTensor<T, dim>(YTensorBase::mostContinuousView());
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

// toCoord() 已移至 YTensorBase 基类

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
    return data_()[_offset + index];
}

template <typename T, int dim>
const T& YTensor<T, dim>::atData_(int index) const {
    return data_()[_offset + index];
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
    return YTensor<T, dim>(YTensorBase::slice(atDim, start, end, step, autoFix));
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::slice_(int atDim, int start, int end, int step, bool autoFix){
    YTensorBase::slice_(atDim, start, end, step, autoFix);
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
    // 委托给 YTensorBase::permute
    return YTensor<T, dim>(YTensorBase::permute(newOrder));
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::permute(const int newOrder[]) const {
    // 转换为 vector 并委托
    std::vector<int> order(newOrder, newOrder + dim);
    return YTensor<T, dim>(YTensorBase::permute(order));
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::transpose(int dim0, int dim1) const {
    // 委托给 YTensorBase::transpose
    return YTensor<T, dim>(YTensorBase::transpose(dim0, dim1));
}

template <typename T, int dim> template <typename... Args>
std::vector<int> YTensor<T, dim>::autoShape(const Args... shape0) const {
    // 委托给 vector 版本
    return autoShape(std::vector<int>{shape0...});
}

template <typename T, int dim>
std::vector<int> YTensor<T, dim>::autoShape(const std::vector<int> &shape) const {
    return YTensorBase::autoShape(shape);
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
    return op;
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
auto YTensor<T, dim>::reshape(const Args... newShape) const -> YTensor<T, sizeof...(Args)> {
    return contiguous().template view<sizeof...(Args)>(std::vector<int>{newShape...});
}

template <typename T, int dim> template <int newdim>
YTensor<T, newdim> YTensor<T, dim>::reshape(const std::vector<int>& newShape) const {
    return contiguous().template view<newdim>(newShape);
}

template <typename T, int dim>
YTensor<T, dim + 1> YTensor<T, dim>::unsqueeze(int d) const {
    d = ((d % (dim + 1)) + (dim + 1)) % (dim + 1);
    YTensor<T, dim + 1> op;
    op._shape = _shape;
    op._stride = _stride;
    op._shape.insert(op._shape.begin() + d, 1);
    // 新维度的 stride 可以设为任意值（因为 size=1），设为下一维度的 stride * size
    int newStride = (d < dim) ? _stride[d] * _shape[d] : 1;
    op._stride.insert(op._stride.begin() + d, newStride);
    op._data = _data;
    op._offset = _offset;
    op._element_size = _element_size;
    op._dtype = _dtype;
    return op;
}

template <typename T, int dim>
YTensor<T, dim - 1> YTensor<T, dim>::squeeze(int d) const requires (dim > 1) {
    int actualDim = (d % dim + dim) % dim;
    if (_shape[actualDim] != 1) {
        throw std::runtime_error("squeeze: can only squeeze dimensions of size 1");
    }
    YTensor<T, dim - 1> op;
    op._shape = _shape;
    op._stride = _stride;
    op._shape.erase(op._shape.begin() + actualDim);
    op._stride.erase(op._stride.begin() + actualDim);
    op._data = _data;
    op._offset = _offset;
    op._element_size = _element_size;
    op._dtype = _dtype;
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
    // 委托给 YTensorBase::repeat
    return YTensor<T, dim>(YTensorBase::repeat(times));
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::repeat(const int times[]) const {
    // 转换为 vector 并委托
    std::vector<int> reps(times, times + dim);
    return YTensor<T, dim>(YTensorBase::repeat(reps));
}

template <typename T, int dim>
YTensor<T, dim + 1> YTensor<T, dim>::unfold(int mdim, int mkernel, int mstride, int mdilation) const{
    if (mkernel <= 0 || mstride <= 0 || mdilation <= 0) {
        throw std::invalid_argument("Invalid kernel/stride/dilation");
    }

    mdim = (mdim % dim + dim) % dim; // 循环索引

    // 计算展开后的形状（nums 放在 kernel 之前：[..., nums, kernel, ...]）
    std::vector<int> newShape = _shape;
    int nums = (_shape[mdim] - (mkernel - 1) * mdilation - 1) / mstride + 1;
    // 将原维度替换为 nums，然后在其后插入 kernel，这样最终顺序为 nums, kernel
    newShape[mdim] = nums;
    newShape.insert(newShape.begin() + mdim + 1, mkernel);

    // 计算新的步长：nums 维度步长是原步长 * stride，kernel 维度步长是原步长 * dilation
    std::vector<int> newStride = _stride;
    newStride[mdim] = _stride[mdim] * mstride; // 窗口移动步长，用于 nums
    newStride.insert(newStride.begin() + mdim + 1, _stride[mdim] * mdilation); // 核内步长

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

template <typename T, int dim> template <typename... Args>
YTensor<T, sizeof...(Args)> YTensor<T, dim>::zeros(Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    YTensor<T, dim> op(shp);
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

template <typename T, int dim> template <typename... Args>
YTensor<T, sizeof...(Args)> YTensor<T, dim>::ones(const Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
inline typename YTensor<T, dim>::_RandnGenerator YTensor<T, dim>::randn = YTensor<T, dim>::_RandnGenerator(yt::infos::gen);

template <typename T, int dim>
inline typename YTensor<T, dim>::_RanduGenerator YTensor<T, dim>::randu = YTensor<T, dim>::_RanduGenerator(yt::infos::gen);

template <typename T, int dim>
void YTensor<T, dim>::seed(unsigned int seed) {
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
    yt::infos::gen.seed(seed);
}

template <typename T, int dim> template <typename Func>
YTensor<T, dim>& YTensor<T, dim>::foreach(Func&& func, double flop){
    // 检测func是否只接受一个参数（不需要coord）
    constexpr bool oneArgFunc = std::is_invocable_v<Func, T&> && !std::is_invocable_v<Func, T&, const std::vector<int>&>;

    if constexpr (oneArgFunc) {
        // 无坐标版本：使用binaryOpTransformInplace的高效实现
        binaryOpTransformInplace(T{}, [&func](T& a, const T&) {
            using ResultType = std::invoke_result_t<Func, T&>;
            if constexpr (std::is_void_v<ResultType>) {
                func(a);
            } else {
                a = func(a);
            }
        }, flop);
    } else {
        // 带坐标版本：原始实现
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
            T* thisPtr = this->data_() + this->_offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                auto coord = this->toCoord(index);
                wrappedFunc(thisPtr[index], coord);
            }, flop);
            return *this;
        }
        int thisSize = this->size();
        // 使用遍历法，便于计算坐标
        yt::kernel::parallelFor(0, thisSize, [&](int i) {
            std::vector<int> coord = toCoord(i);
            wrappedFunc(this->at(coord), coord);
        }, flop);
    }
    return *this;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::fill(T value){
    auto mcView = mostContinuousView();
    int cFrom = mcView.isContiguousFrom();  // 从这个维度开始，后面的维度都是连续的

    if(cFrom == 0){
        // 完全连续，直接fill
        std::fill(mcView.data(), mcView.data() + mcView.size(), value);
    } else if(cFrom < dim) {
        // 部分连续：遍历非连续的前cFrom个维度，然后fill填充连续的后dim-cFrom个维度
        // 计算连续部分的大小
        size_t contiguousSize = 1;
        for(int i = cFrom; i < dim; i++){
            contiguousSize *= _shape[i];
        }

        // 计算非连续部分的迭代次数
        size_t outerSize = 1;
        for(int i = 0; i < cFrom; i++){
            outerSize *= _shape[i];
        }

        // 使用parallelFor遍历非连续维度
        T* basePtr = this->data_();
        yt::kernel::parallelFor(0, static_cast<int>(outerSize), [&](int outerIdx){
            // 计算非连续部分的坐标
            std::vector<int> outerCoord(cFrom);
            int remaining = outerIdx;
            for(int i = cFrom - 1; i >= 0; i--){
                outerCoord[i] = remaining % _shape[i];
                remaining /= _shape[i];
            }

            // 计算这个坐标对应的基地址偏移
            size_t offset = _offset;
            for(int i = 0; i < cFrom; i++){
                offset += outerCoord[i] * _stride[i];
            }

            // 对连续部分进行fill
            T* ptr = basePtr + offset;
            std::fill(ptr, ptr + contiguousSize, value);
        });
    } else {
        // 完全不连续，使用逐元素填充
        binaryOpTransformInplace(value, [](T& item, const T& value){
            item = value;
        });
    }
    return *this;
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::copy_(const yt::YTensorBase& src) {
    yt::YTensorBase::copy_(src);
    return *this;
}

template <typename T, int dim>
std::ostream &operator<<(std::ostream &out, const YTensor<T, dim> &tensor){
    out << "[YTensor]:<" << safeGetTypeName<T>() << ">" << std::endl;
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
            const T* thisPtr = this->data_() + this->_offset;
            const T* otherPtr = other.data_() + other._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                func(thisPtr[index], otherPtr[index], op.atData(index));
            }, flop);
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
            return func(this->atData_(thisIndex), other.atData_(otherIndex), op.atData_(index));
        };

        // 4、并行计算
        int max = op.size();
        yt::kernel::parallelFor(0, max, kernel, flop);
    }
    else{
        if(equalShape && this->isContiguous() && other.isContiguous()) {
            // fast path
            int max = op.size();
            T* opPtr = op.data_();
            const T* thisPtr = this->data_() + this->_offset;
            const T* otherPtr = other.data_() + other._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                opPtr[index] = func(thisPtr[index], otherPtr[index]);
            }, flop);
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
        yt::kernel::parallelFor(0, max, [&](int index) {
            op.atData_(index) = kernel(index);
        }, flop);
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
        T* thisPtr = this->data_() + this->_offset;
        const T* otherPtr = other.data_() + other._offset;
        yt::kernel::parallelFor(0, max, [&](int index) {
            func(thisPtr[index], otherPtr[index]);
        }, flop);
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
    yt::kernel::parallelFor(0, max, kernel, flop);
    return *this;
}

template <typename T, int dim>
template <typename Func, typename... Args>
YTensor<T, dim>& YTensor<T, dim>::broadcastInplace(Func&& func, Args&&... tensors) {
    return yt::kernel::broadcastInplace(*this, std::forward<Func>(func), std::forward<Args>(tensors)...);
}

template<typename T, int dim> template<typename Func>
YTensor<T, dim> YTensor<T, dim>::binaryOpTransform(const T& other, Func&& func,  YTensor<T, dim>* result, double flop) const{
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
            T* thisPtr = mcView.data_() + mcView._offset;
            T* opPtr = op.data_() + op._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                func(thisPtr[index], other, opPtr[index]);
            }, flop);
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
        yt::kernel::parallelFor(0, thisSize, kernel, flop);
    }else{
        if(mcView.isContiguous()) {
            // fast path
            int max = mcView.size();
            T* thisPtr = mcView.data_() + mcView._offset;
            T* opPtr = op.data_() + op._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                opPtr[index] = func(thisPtr[index], other);
            }, flop);
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
        yt::kernel::parallelFor(0, thisSize, [&](int i) {
            op.atData_(i) = kernel(i);
        }, flop);
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
        T* thisPtr = mcView.data_() + mcView._offset;
        yt::kernel::parallelFor(0, max, [&](int index) {
            wrappedFunc(thisPtr[index], other);
        }, flop);
        return *this;
    }
    int thisSize = this->size();
    // 对于非连续情况，使用遍历法
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
    yt::kernel::parallelFor(0, thisSize, kernel, flop);
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
    using MatType = YTensor<T, 2>;

    if constexpr (dim == 1){
        MatType mat;
        mat._shape = std::vector<int>({1, this->_shape[0]});
        mat._stride = std::vector<int>({0, this->_stride[0]});
        mat._offset = this->_offset;
        mat._element_size = sizeof(T);
        mat._dtype = safeGetTypeName<T>();
        mat._data = this->_data;

        YTensor<MatType, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";

        // 使用封装函数分配内存
        op._data = yt::kernel::makeSharedPlacement<MatType>(mat);
        return op;
    }else if constexpr (dim == 2){
        YTensor<MatType, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";

        // 使用封装函数分配内存
        MatType thisCopy = *this;  // 创建当前张量的副本
        op._data = yt::kernel::makeSharedPlacement<MatType>(thisCopy);
        return op;
    }else{
        auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
        YTensor<MatType, std::max(1, dim - 2)> op;
        op._shape = newShape;
        op._stride = op.stride();
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";
        int batchSize = op.size();

        // 使用封装函数分配数组内存
        op._data = yt::kernel::makeSharedPlacementArray<MatType>(batchSize);
        MatType* dataptr = reinterpret_cast<MatType*>(op._data.get());

        // 使用 placement new 构造每个 MatType
        for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
            auto coord = op.toCoord(batchIdx);
            MatType mat;
            mat._shape = {this->_shape[dim-2], this->_shape[dim-1]};
            mat._stride = {this->_stride[dim-2], this->_stride[dim-1]};
            mat._offset = this->offset_(coord);
            mat._element_size = sizeof(T);
            mat._dtype = safeGetTypeName<T>();
            mat._data = this->_data;
            new (&dataptr[batchIdx]) MatType(mat);
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
YTensor<T, dim> YTensor<T, dim>::mean(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    int axisLen = newShape[axis];
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    int max = static_cast<int>(op.size());

    yt::kernel::parallelFor(0, max, [&](int i){
        // 使用Welford算法进行均值计算
        auto coord = op.toCoord(i);
        T mean = 0;
        for (int j = 0; j < axisLen; j++) {
            auto subCoord = coord;
            subCoord[axis] = j;
            T x = this->at(subCoord);
            mean += (x - mean) / static_cast<T>(j + 1);
        }
        op.atData_(i) = mean;
    }, static_cast<double>(axisLen));

    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::mean(std::vector<int> axes) const requires (dim > 1) {
    // 规范化轴并计算每个轴的长度
    std::vector<int> normalizedAxes;
    int totalN = 1;
    for (int ax : axes) {
        ax = (ax % dim + dim) % dim;
        normalizedAxes.push_back(ax);
        totalN *= this->shape(ax);
    }

    // 依次对每个轴使用mean(int axis)
    YTensor<T, dim> result = *this;
    for (int ax : normalizedAxes) {
        result = result.mean(ax);
    }

    return result;
}

template <typename T, int dim>
T YTensor<T, dim>::mean(int) const requires (dim == 1) {
    int n = this->size();
    if (n == 0) return static_cast<T>(0);

    // 使用Welford算法进行均值计算，提高数值稳定性
    T mean_val = 0;
    for (int i = 0; i < n; i++) {
        T x = this->at(i);
        mean_val += (x - mean_val) / static_cast<T>(i + 1);
    }

    return mean_val;
}

template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> YTensor<T, dim>::max(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    YTensor<int, dim> opi(newShape);
    int max = static_cast<int>(op.size());
    int axisSize = _shape[axis];

    yt::kernel::parallelFor(0, max, [&](int i) {
        auto coord = op.toCoord(i);
        T maxer = this->at(coord);
        int maxerIndex = 0;
        for (int j = 0; j < axisSize; j++) {
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
    }, static_cast<double>(axisSize));

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
    int maxSize = static_cast<int>(op.size());
    int offsetsSize = static_cast<int>(offsets.size());

    yt::kernel::parallelFor(0, maxSize, [&](int i) {
        auto coord = op.toCoord(i);
        auto base = this->offset(coord);
        T maxer = this->at(coord);
        int maxerIndex = 0;
        for (int j = 0; j < offsetsSize; j++) {
            const T& value = this->atData_(base + offsets[j]);
            if (value > maxer) {
                maxer = value;
                maxerIndex = j;
            }
        }
        op.atData_(i) = maxer;
        opi.atData_(i) = maxerIndex;
    }, static_cast<double>(offsetsSize));

    return std::make_pair(op, opi);
}

template<typename T, int dim>
std::pair<T, int> YTensor<T, dim>::max(int)const requires (dim == 1) {
    T maxer = this->at(0);
    int maxerIndex = 0;
    int max = this->size();
    for (int i = 0; i < max; i++) {
        const T& value = this->at(i);
        if (value > maxer) {
            maxer = value;
            maxerIndex = i;
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
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        opShape.push_back(ah); opShape.push_back(bw);
    }
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    auto mulop = thisMatView.binaryOpBroadcast(otherMatView, [&ah, &aw, &bw](const YTensor<T, 2>& a, const YTensor<T, 2>& b, YTensor<T, 2>& o) {
        #pragma omp simd collapse(2)
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
    op._element_size = sizeof(EigenMatrixMap);
    op._dtype = "eigen_map";
    int batchSize = op.size();

    // 使用封装函数分配数组内存
    op._data = yt::kernel::makeSharedPlacementArray<EigenMatrixMap>(batchSize);
    EigenMatrixMap* opData = reinterpret_cast<EigenMatrixMap*>(op._data.get());

    // 使用 placement new 构造每个 EigenMatrixMap
    const T* thisData = this->data_();
    for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
        auto coord = op.toCoord(batchIdx);
        Eigen::Stride<-1, -1> mstride(this->_stride[dim - 2], this->_stride[dim - 1]);
        T* matDataPtr = const_cast<T*>(thisData) + this->offset_(coord);  // 使用 offset_ 考虑张量自身的 _offset
        new (&opData[batchIdx]) EigenMatrixMap(matDataPtr, this->_shape[dim - 2], this->_shape[dim - 1], mstride);
    }
    return op;
}

template <typename T, int dim> typename
YTensor<T, dim>::EigenMatrixMap YTensor<T, dim>::matViewEigen() const requires (dim <= 2) {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    if constexpr (dim == 1){
        Eigen::Stride<-1, -1> mstride(0, this->_stride[0]);
        T* dataptr = const_cast<T*>(this->data_()) + this->_offset;
        EigenMatrixMap op(dataptr, this->_shape[0], 1, mstride);
        return op;
    }else{
        Eigen::Stride<-1, -1> mstride(this->_stride[0], this->_stride[1]);
        T* dataptr = const_cast<T*>(this->data_()) + this->_offset;
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
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
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

    template<typename T, int dim>
    YTensor<T, dim>& softmax_(YTensor<T, dim>& x, int axis = -1);

    enum struct sdpaBackend{
        MATH
    };

    template<typename T, int dim>
    YTensor<T, dim> scaledDotProductAttention(
        YTensor<T, dim>& query,
        YTensor<T, dim>& key,
        YTensor<T, dim>& value,
        T scale = static_cast<T>(0.0),
        YTensor<T, 2>* mask = nullptr,
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
YTensor<T, dim>& yt::function::softmax_(YTensor<T, dim>& x, int axis) {
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();

    // 快速路径：连续张量 - 使用优化实现（Flash Attention 风格）
    if (x.isContiguous()) {
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

        T* data_base = x.data();

        // 针对连续数据的优化循环
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                T* data = data_base + outer_idx * outer_stride + inner_idx;

                // 步骤 1: 获取最大值（数值稳定性）
                T max_val = data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, data[d * dim_stride]);
                }

                // 步骤 2: 计算 exp(x - max) 并累加和（融合操作，原地更新）
                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(data[d * dim_stride] - max_val);
                    data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }

                // 步骤 3: 用和归一化
                for (int64_t d = 0; d < dim_size; ++d) {
                    data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        // 通用路径：支持非连续张量
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

            // 第二遍：计算 exp(x - max) 并求和（原地更新）
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                x.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            // 第三遍：归一化
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) /= sum_exp;
            }
        }
    }

    return x;
}

template<typename T, int dim>
YTensor<T, dim> yt::function::scaledDotProductAttention(
    YTensor<T, dim>& query,// [..., n0, c0]
    YTensor<T, dim>& key,// [..., n1, c0]
    YTensor<T, dim>& value,// [..., n1, c1]
    T scale,
    YTensor<T, 2>* mask,
    sdpaBackend backend
) {
    if(static_cast<T>(0.0) == scale){
        // auto
        scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(query.shape(-1)));
    }
    if(backend == sdpaBackend::MATH){
        auto t0 = std::chrono::high_resolution_clock::now();
        auto score = yt::function::matmul(query, key.transpose());// [..., n0, n1]
        auto t1 =  std::chrono::high_resolution_clock::now();
        score.binaryOpTransformInplace(scale, [](T& a, const T& b) {
            a *= b; // scale
        });
        auto t2 =  std::chrono::high_resolution_clock::now();
        if(mask != nullptr){
            if(mask->shape(0) != score.shape(-2) || mask->shape(1) != score.shape(-1)){
                throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
            }
            score += *mask;
        }
        auto t3 =  std::chrono::high_resolution_clock::now();
        yt::function::softmax_(score, -1);// [..., n0, n1] inplace
        auto t4 =  std::chrono::high_resolution_clock::now();
        auto op = yt::function::matmul(score, value);// [..., n0, c1]
        auto t5 =  std::chrono::high_resolution_clock::now();
        double dt0 = std::chrono::duration<double>(t1 - t0).count() * 1e6;
        double dt1 = std::chrono::duration<double>(t2 - t1).count() * 1e6;
        double dt2 = std::chrono::duration<double>(t3 - t2).count() * 1e6;
        double dt3 = std::chrono::duration<double>(t4 - t3).count() * 1e6;
        double dt4 = std::chrono::duration<double>(t5 - t4).count() * 1e6;
        std::cout << "QK: " << dt0 << "us" << std::endl;
        std::cout << "scale: " << dt1 << "us" << std::endl;
        std::cout << "mask: " << dt2 << "us" << std::endl;
        std::cout << "softmax: " << dt3 << "us" << std::endl;
        std::cout << "V: " << dt4 << "us" << std::endl;
        return op;

        // auto score = yt::function::matmul(query, key.transpose());// [..., n0, n1]
        // score.binaryOpTransformInplace(scale, [](T& a, const T& b) {
        //     a *= b; // scale
        // });
        // if(mask != nullptr){
        //     if(mask->shape(0) != score.shape(-2) || mask->shape(1) != score.shape(-1)){
        //         throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
        //     }
        //     score += *mask;
        // }
        // yt::function::softmax_(score, -1);// [..., n0, n1] inplace
        // auto op = yt::function::matmul(score, value);// [..., n0, c1]
        // return op;
    }
    else{
        throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
        return YTensor<T, dim>();
    }
}


void yt::function::throwNotSupport(const std::string& funcName, const std::string& caseDiscription) {
    throw std::invalid_argument("Function " + funcName + " is not supported for case: " + caseDiscription);
}
