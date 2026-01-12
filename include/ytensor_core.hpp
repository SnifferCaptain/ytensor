#pragma once
/***************
* @file: ytensor.hpp
* @brief: 易于使用的张量类，主要用作容器。
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 0.3
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
#include "ytensor_concepts.hpp"
#include "ytensor_infos.hpp"
#include "ytensor_base.hpp"

namespace yt{

/**
 * @brief 易于使用的张量类。可以处理任意维度的张量。
 * @tparam T 张量元素的数据类型。
 * @tparam dim 张量的维度数。
 */
template <typename T=float, int dim=1>
class YTensor : public YTensorBase {
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
    YTensor<T, dim>& copy_(const YTensorBase& src);

/////////////////// externs ////////////////
    #include "ytensor_math.hpp"

protected:
    /// @brief 标准cout输出流
    std::ostream& _cout(std::ostream& os) const override;
};

} // namespace yt