#pragma once
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

#include "ytensor_concepts.hpp"
#include "ytensor_infos.hpp"

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

    /// @brief 通过位置索引访问元素（模板版本，返回引用）
    template <typename T, typename... Args>
    T& at(Args... args);
    template <typename T>
    T& at(const std::vector<int>& pos);
    template <typename T>
    const T& at(const std::vector<int>& pos) const;

    /// @brief 按元素下标访问（模板版本），返回底层元素引用（物理索引）
    template <typename T>
    T& atData(int index);
    template <typename T>
    const T& atData(int index) const;

    /// @brief 浅拷贝（共享底层数据）
    void shallowCopyTo(YTensorBase &other) const;

    /// @brief 深拷贝：返回一个独立拥有自己数据的 YTensorBase
    YTensorBase clone() const;

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

    /// @brief 沿指定维度重复张量。
    /// @param times 每个维度重复的次数。
    /// @return 返回一个新的 YTensorBase 视图。
    /// @note 只能在大小为 1 的维度上进行重复。
    YTensorBase repeat(const std::vector<int>& times) const;

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
    static _RandnGenerator randn;
    static _RanduGenerator randu;

    /// @brief 创建指定大小的，零张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensorBase zeros(const std::vector<int>& shape, std::string dtype = "float32");

    /// @brief 创建指定大小的，全1张量
    /// @param shape 张量的形状
    /// @return 返回张量
    static YTensorBase ones(const std::vector<int>& shape, std::string dtype = "float32");

    /// @brief 获取数据类型字符串（例如 "float32"）
    std::string dtype() const;

    /// @brief 获取元素字节大小
    size_t elementSize() const;

protected:
    std::shared_ptr<char[]> _data;  // 存储数据
    int _offset = 0;                // 数据偏移 (以元素为单位)
    std::vector<int> _shape;        // 形状
    std::vector<int> _stride;       // 步长 (以元素为单位)
    size_t _element_size = 0;       // 元素大小（字节）
    std::string _dtype;             // 用于序列化/反序列化友好名称

    // removed helper calculate_logical_stride to match ytensor.hpp declarations
};

#include "../src/ytensor_base.inl"