/***************
 * file: ytensor_core.inl
 * purpose: YTensor<T, dim> typed facade、直接访问热路径和view包装实现。
 ***************/

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <ranges>

#include "../include/ytensor_core.hpp"
#include "../include/ytensor_types.hpp"

template <typename T, int dim>
void yt::YTensor<T, dim>::throwShapeNotMatch(const std::string& funcName, const std::vector<int>& otherShape)
    const {
    std::string errorMsg = "Function \"" + funcName + "\" shape not match: YTensor[";
    for (int i = 0; i < dim; ++i) {
        errorMsg += std::to_string(stridedShape()[i]);
        if (i < dim - 1) {
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
void yt::YTensor<T, dim>::throwShapeNotMatch(
    const std::string& funcName, const std::vector<int>& thisShape, const std::vector<int>& otherShape
) {
    std::string errorMsg = "Function \"" + funcName + "\" shape not match: YTensor[";
    for (int i = 0; i < dim; ++i) {
        errorMsg += std::to_string(thisShape[i]);
        if (i < dim - 1) {
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
void yt::YTensor<T, dim>::throwShapeSizeNotMatch(const std::string& funcName, int otherDim) {
    std::string errorMsg = "Function \"" + funcName + "\" shape size not match: YTensor<T, " +
                           std::to_string(dim) + "> and dim size " + std::to_string(otherDim);
    throw std::invalid_argument(errorMsg);
}

template <typename T, int dim>
void yt::YTensor<T, dim>::throwOperatorNotSupport(const std::string& typeName, const std::string& opName) {
    std::string errorMsg = "Operator " + opName + " not supported for type " + typeName;
    throw std::runtime_error(errorMsg);
}

// ==================== construction and shared ownership ====================

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor() : YTensorBase() {
    stridedShape().resize(dim, 0);
    stridedStride().resize(dim, 0);
    _element_size = sizeof(T);
    _dtype = yt::type::getYTensorDtype<T, dim>();
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(const std::vector<int> shape) : YTensorBase() {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("init", shape.size());
    }
    reserve(shape);
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, dim>::YTensor(Args... args) : YTensorBase() {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    std::vector<int> shapeVec(dim);
    int a = 0;
    ((shapeVec[a++] = args), ...);
    reserve(shapeVec);
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(std::initializer_list<int> list) : YTensorBase() {
    if (list.size() != dim) {
        throwShapeSizeNotMatch("init", list.size());
    }
    reserve(std::vector<int>(list));
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(const YTensorBase& base) : YTensorBase(base) {
    // 转换保持base的共享storage语义，但必须在发布typed facade前校验固定rank/dtype/size。
    if (static_cast<int>(base.ndim()) != dim) {
        throwShapeSizeNotMatch("YTensorBase", base.ndim());
    }
    const std::string expectedDtype = yt::type::getYTensorDtype<T, dim>();
    const bool float8Alias = std::is_same_v<T, yt::float8_e8m0> && expectedDtype == "float8_e8m0" &&
                             base.dtype() == "float8_ue8m0";
    if ((!float8Alias && base.dtype() != expectedDtype) || base.elementSize() != sizeof(T)) {
        throw std::invalid_argument(
            "YTensorBase dtype mismatch: expected " + expectedDtype + ", got " + base.dtype()
        );
    }
    // 接受历史alias后立即canonicalize，后续typed操作只观察一种dtype拼写。
    if (float8Alias) _dtype = expectedDtype;
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(const yt::YTensor<T, dim>& other) : YTensorBase(other) {}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator=(const yt::YTensor<T, dim>& other) {
    if (this == &other) {
        return *this;
    }
    YTensorBase::operator=(other);
    return *this;
}

template <typename T, int dim>
void yt::YTensor<T, dim>::shallowCopyTo(yt::YTensor<T, dim>& other) const {
    YTensorBase::shallowCopyTo(other);
}

template <typename T, int dim>
void yt::YTensor<T, dim>::shareTo(yt::YTensor<T, dim>& other) const {
    shallowCopyTo(other);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::shallowCopyFrom(const yt::YTensor<T, dim>& src) {
    src.YTensorBase::shallowCopyTo(*this);
    return *this;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::shareFrom(const yt::YTensor<T, dim>& src) {
    shallowCopyFrom(src);
    return *this;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::clone() const {
    if (!isStrided()) throw std::runtime_error("YTensor::clone: layout not implemented");
    return yt::strided::clone(*this);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::reserve(const std::vector<int>& shape) {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("reserve", shape.size());
    }
    const std::string dtype = yt::type::getYTensorDtype<T, dim>();
    if constexpr (yt::utils::is_ytensor_template_v<T>) {
        // Nested typed元素必须建立真实C++对象生命周期，不能交给runtime raw-byte allocator。
        const auto stride = yt::strided::contiguousStrideForShape(shape, "YTensor::reserve");
        size_t total = 1;
        for (int extent : shape) total *= static_cast<size_t>(extent);
        // makeSharedPlacementArray在构造失败时回滚已构造前缀；object-backed storage禁止字节clone。
        std::shared_ptr<char[]> objects;
        if (total > 0) {
            objects = yt::utils::makeSharedPlacementArray<T>(
                total, [](T* destination, size_t) { new (destination) T(); }
            );
        }
        yt::strided::ViewTensorAccess<T, dim>::setView(
            *this, shape, stride, 0,
            total == 0 ? YMemory() : YMemory(objects, total * sizeof(T), "cpu", false), dtype
        );
    } else {
        YTensorBase::operator=(YTensorBase(shape, dtype));
    }
    return *this;
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::reserve(Args... args) {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return reserve(std::vector<int>{args...});
}

// ==================== typed pointer and offset hot paths ====================

template <typename T, int dim>
T* yt::YTensor<T, dim>::data() {
    return _memory ? reinterpret_cast<T*>(_memory.get()) + stridedOffset() : nullptr;
}

template <typename T, int dim>
const T* yt::YTensor<T, dim>::data() const {
    return _memory ? reinterpret_cast<const T*>(_memory.get()) + stridedOffset() : nullptr;
}

template <typename T, int dim>
T* yt::YTensor<T, dim>::data_() {
    return reinterpret_cast<T*>(_memory.get());
}

template <typename T, int dim>
const T* yt::YTensor<T, dim>::data_() const {
    return reinterpret_cast<const T*>(_memory.get());
}

template <typename T, int dim>
constexpr int yt::YTensor<T, dim>::shapeSize() const {
    return dim;
}

template <typename T, int dim>
template <typename... Args>
int yt::YTensor<T, dim>::offset(Args... index) const {
    // 允许partial coordinate；未提供的尾部坐标值初始化为0，保持旧版切片访问合同。
    static_assert(sizeof...(index) <= dim, "Number of arguments must match the dimension");
    std::array<int, dim> indices{};
    size_t a = 0;
    ((indices[a++] = static_cast<int>(index)), ...);
    int64_t result = 0;
    for (int i = 0; i < dim; ++i) {
        if (indices[i] < 0 || indices[i] >= stridedShape()[i]) {
            throw std::out_of_range("YTensor::offset: coordinate out of range");
        }
        result += static_cast<int64_t>(indices[i]) * stridedStride()[i];
    }
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
        throw std::overflow_error("YTensor::offset: physical offset exceeds int range");
    }
    return static_cast<int>(result);
}

template <typename T, int dim>
int yt::YTensor<T, dim>::offset(const std::vector<int>& index) const {
    if (index.size() > dim) {
        throwShapeSizeNotMatch("offset", index.size());
    }
    // vector同样允许少于dim个坐标，缺失尾轴按0处理。
    int64_t result = 0;
    for (int i = 0; i < dim; ++i) {
        const int coordinate = i < static_cast<int>(index.size()) ? index[i] : 0;
        if (coordinate < 0 || coordinate >= stridedShape()[i]) {
            throw std::out_of_range("YTensor::offset: coordinate out of range");
        }
        result += static_cast<int64_t>(coordinate) * stridedStride()[i];
    }
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
        throw std::overflow_error("YTensor::offset: physical offset exceeds int range");
    }
    return static_cast<int>(result);
}

template <typename T, int dim>
template <typename... Args>
int yt::YTensor<T, dim>::offset_(Args... index) const {
    static_assert(sizeof...(index) <= dim, "Number of arguments must match the dimension");
    const int64_t result = static_cast<int64_t>(stridedOffset()) + offset(index...);
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
        throw std::overflow_error("YTensor::offset_: storage offset exceeds int range");
    }
    return static_cast<int>(result);
}

template <typename T, int dim>
int yt::YTensor<T, dim>::offset_(const std::vector<int>& index) const {
    const int64_t result = static_cast<int64_t>(stridedOffset()) + offset(index);
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max()) {
        throw std::overflow_error("YTensor::offset_: storage offset exceeds int range");
    }
    return static_cast<int>(result);
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::contiguous() const {
    if (!isStrided()) throw std::runtime_error("YTensor::contiguous: layout not implemented");
    return yt::strided::contiguous(*this);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::contiguous_() {
    if (!isStrided()) throw std::runtime_error("YTensor::contiguous_: layout not implemented");
    return yt::strided::contiguous_(*this);
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::mostContinuousView() const {
    return yt::YTensor<T, dim>(YTensorBase::mostContinuousView());
}

template <typename T, int dim>
template <typename... Args>
size_t yt::YTensor<T, dim>::toIndex(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    return toIndex(indices);
}

template <typename T, int dim>
size_t yt::YTensor<T, dim>::toIndex(const std::vector<int>& pos) const {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("toIndex", pos.size());
    }
    return YTensorBase::toIndex(pos);
}

template <typename T, int dim>
template <typename... Args>
size_t yt::YTensor<T, dim>::toIndex_(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return YTensorBase::toIndex_(args...);
}

template <typename T, int dim>
size_t yt::YTensor<T, dim>::toIndex_(const std::vector<int>& pos) const {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("toIndex_", pos.size());
    }
    return YTensorBase::toIndex_(pos);
}

// toCoord()由runtime facade按layout分发；typed热路径只保留需要静态T/dim的访问。

template <typename T, int dim>
T& yt::YTensor<T, dim>::atData(int index) {
    if (index < 0 || static_cast<size_t>(index) >= size()) {
        throw std::out_of_range("YTensor::atData: logical index out of range");
    }
    // 手工分解row-major逻辑下标，避免构造coordinate vector，同时支持负/非连续physical stride。
    size_t remaining = static_cast<size_t>(index);
    std::ptrdiff_t physical = 0;
    for (int i = dim - 1; i >= 0; --i) {
        const int coordinate = static_cast<int>(remaining % static_cast<size_t>(stridedShape()[i]));
        remaining /= static_cast<size_t>(stridedShape()[i]);
        physical += static_cast<std::ptrdiff_t>(coordinate) * stridedStride()[i];
    }
    return data()[physical];
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::atData(int index) const {
    if (index < 0 || static_cast<size_t>(index) >= size()) {
        throw std::out_of_range("YTensor::atData: logical index out of range");
    }
    // const路径保持与mutable路径完全相同的逻辑到物理映射。
    size_t remaining = static_cast<size_t>(index);
    std::ptrdiff_t physical = 0;
    for (int i = dim - 1; i >= 0; --i) {
        const int coordinate = static_cast<int>(remaining % static_cast<size_t>(stridedShape()[i]));
        remaining /= static_cast<size_t>(stridedShape()[i]);
        physical += static_cast<std::ptrdiff_t>(coordinate) * stridedStride()[i];
    }
    return data()[physical];
}

template <typename T, int dim>
T& yt::YTensor<T, dim>::atData_(int index) {
    return data()[index];
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::atData_(int index) const {
    return data()[index];
}

template <typename T, int dim>
template <typename... Args>
T& yt::YTensor<T, dim>::at(const Args... args) {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return data()[offset(args...)];
}

template <typename T, int dim>
T& yt::YTensor<T, dim>::at(const std::vector<int>& pos) {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("at", pos.size());
    }
    return data()[offset(pos)];
}

template <typename T, int dim>
template <typename... Args>
const T& yt::YTensor<T, dim>::at(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return data()[offset(args...)];
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::at(const std::vector<int>& pos) const {
    if (pos.size() != dim) {
        throwShapeSizeNotMatch("at", pos.size());
    }
    return data()[offset(pos)];
}

// 首轴索引按shape(0)循环规范化；rank>1返回共享storage的slice view。
template <typename T, int dim>
yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::operator[](int index) requires(dim > 1) {
    if (shape(0) == 0) throw std::out_of_range("YTensor::operator[]: empty leading dimension");
    index = (index % shape(0) + shape(0)) % shape(0);
    return yt::YTensor<T, dim - 1>(YTensorBase::slice(0, index, index + 1).squeeze(0));
}

template <typename T, int dim>
const yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::operator[](int index) const requires(dim > 1) {
    if (shape(0) == 0) throw std::out_of_range("YTensor::operator[]: empty leading dimension");
    index = (index % shape(0) + shape(0)) % shape(0);
    return yt::YTensor<T, dim - 1>(YTensorBase::slice(0, index, index + 1).squeeze(0));
}

// rank-1索引同样循环规范化，负数和越界正数不会直接抛出。
template <typename T, int dim>
T& yt::YTensor<T, dim>::operator[](int index) requires(dim == 1) {
    if (shape(0) == 0) throw std::out_of_range("YTensor::operator[]: empty tensor");
    index = (index % shape(0) + shape(0)) % shape(0);
    return at(index);
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::operator[](int index) const requires(dim == 1) {
    if (shape(0) == 0) throw std::out_of_range("YTensor::operator[]: empty tensor");
    index = (index % shape(0) + shape(0)) % shape(0);
    return at(index);
}

// ==================== typed view transforms ====================

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::slice(int atDim, int start, int end, int step, bool autoFix) const {
    return yt::YTensor<T, dim>(YTensorBase::slice(atDim, start, end, step, autoFix));
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::slice_(int atDim, int start, int end, int step, bool autoFix) {
    YTensorBase::slice_(atDim, start, end, step, autoFix);
    return *this;
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, dim> yt::YTensor<T, dim>::permute(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return permute(std::vector<int>{args...});
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::permute(const std::vector<int>& newOrder) const {
    // 委托给 YTensorBase::permute
    return yt::YTensor<T, dim>(YTensorBase::permute(newOrder));
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::permute(const int newOrder[]) const {
    // 转换为 vector 并委托
    std::vector<int> order(newOrder, newOrder + dim);
    return yt::YTensor<T, dim>(YTensorBase::permute(order));
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::transpose(int dim0, int dim1) const {
    // 委托给 YTensorBase::transpose
    return yt::YTensor<T, dim>(YTensorBase::transpose(dim0, dim1));
}

template <typename T, int dim>
template <typename... Args>
std::vector<int> yt::YTensor<T, dim>::autoShape(const Args... shape0) const {
    // 委托给 vector 版本
    return autoShape(std::vector<int>{shape0...});
}

template <typename T, int dim>
std::vector<int> yt::YTensor<T, dim>::autoShape(const std::vector<int>& shape) const {
    return YTensorBase::autoShape(shape);
}

template <typename T, int dim>
template <typename... Args>
auto yt::YTensor<T, dim>::view(const Args... newShape) const -> yt::YTensor<T, sizeof...(Args)> {
    constexpr int newdim = sizeof...(newShape);
    return view<newdim>(std::vector<int>{newShape...});
}

template <typename T, int dim>
template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::view(const std::vector<int>& newShape) const {
    if (newShape.size() != newdim) {
        throwShapeSizeNotMatch("view", newShape.size());
    }
    if (!isStrided()) throw std::runtime_error("YTensor::view: layout not implemented");
    if (!isContiguous()) throw std::runtime_error("\"view\" requires contiguous tensor.");
    const auto shape = autoShape(newShape);
    const auto stride = yt::strided::contiguousStrideForShape(shape, "YTensor::view");
    // 直接构造newdim metadata以保留编译期rank；values storage仍与源tensor共享。
    yt::YTensor<T, newdim> out;
    yt::strided::ViewTensorAccess<T, newdim>::setView(
        out, shape, stride, stridedOffset(), _memory, yt::type::getYTensorDtype<T, newdim>()
    );
    return out;
}

template <typename T, int dim>
template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::view(const int newShape[]) const {
    std::vector<int> shape = std::vector<int>(newdim);
    for (int i = 0; i < newdim; ++i) {
        shape[i] = newShape[i];
    }
    return view<newdim>(shape);
}

template <typename T, int dim>
template <typename... Args>
auto yt::YTensor<T, dim>::reshape(const Args... newShape) const -> yt::YTensor<T, sizeof...(Args)> {
    return contiguous().template view<sizeof...(Args)>(std::vector<int>{newShape...});
}

template <typename T, int dim>
template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::reshape(const std::vector<int>& newShape) const {
    if (newShape.size() != newdim) {
        throwShapeSizeNotMatch("reshape", newShape.size());
    }
    return contiguous().template view<newdim>(newShape);
}

template <typename T, int dim>
yt::YTensor<T, dim + 1> yt::YTensor<T, dim>::unsqueeze(int d) const {
    return yt::YTensor<T, dim + 1>(YTensorBase::unsqueeze(d));
}

template <typename T, int dim>
yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::squeeze(int d) const requires(dim > 1) {
    int actualDim = (d % dim + dim) % dim;
    return yt::YTensor<T, dim - 1>(YTensorBase::squeeze(actualDim));
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, dim> yt::YTensor<T, dim>::repeat(const Args... times) const {
    static_assert(sizeof...(times) == dim, "Number of arguments must match the dimension");
    return repeat(std::vector<int>{times...});
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::repeat(const std::vector<int>& times) const {
    // 委托给 YTensorBase::repeat
    return yt::YTensor<T, dim>(YTensorBase::repeat(times));
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::repeat(const int times[]) const {
    // 转换为 vector 并委托
    std::vector<int> reps(times, times + dim);
    return yt::YTensor<T, dim>(YTensorBase::repeat(reps));
}

template <typename T, int dim>
yt::YTensor<T, dim + 1> yt::YTensor<T, dim>::unfold(int mdim, int mkernel, int mstride, int mdilation) const {
    return yt::YTensor<T, dim + 1>(YTensorBase::unfold(mdim, mkernel, mstride, mdilation));
}

// ==================== factories, random state and mutation ====================

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::zeros(const std::vector<int>& shape) {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("zeros", shape.size());
    }
    yt::YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, sizeof...(Args)> yt::YTensor<T, dim>::zeros(Args... shape) {
    std::vector<int> shp = {shape...};
    yt::YTensor<T, sizeof...(Args)> op(shp);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::ones(const std::vector<int>& shape) {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("ones", shape.size());
    }
    yt::YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, sizeof...(Args)> yt::YTensor<T, dim>::ones(const Args... shape) {
    std::vector<int> shp = {shape...};
    yt::YTensor<T, sizeof...(Args)> op(shp);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
inline typename yt::YTensor<T, dim>::_RandnGenerator yt::YTensor<T, dim>::randn =
    yt::YTensor<T, dim>::_RandnGenerator(yt::info::gen);

template <typename T, int dim>
inline typename yt::YTensor<T, dim>::_RanduGenerator yt::YTensor<T, dim>::randu =
    yt::YTensor<T, dim>::_RanduGenerator(yt::info::gen);

template <typename T, int dim>
void yt::YTensor<T, dim>::seed(unsigned int seed) {
    // 各模板特化的generator facade最终共享yt::info::gen；任一seed会影响所有dtype/rank。
    std::lock_guard<std::mutex> lock(yt::info::rngMutex);
    yt::info::gen.seed(seed);
}

template <typename T, int dim>
template <typename Func>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::foreach (Func&& func, double flop) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::forEach(*this, std::forward<Func>(func), flop);
        default:
            throw std::runtime_error("YTensor::foreach: layout not implemented");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::fill(T value) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::fill(*this, value);
        default:
            throw std::runtime_error("YTensor::fill: layout not implemented");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::copy_(const yt::YTensorBase& src) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::copy_(*this, src);
        default:
            throw std::runtime_error("YTensor::copy_: layout not implemented");
    }
}

// ==================== formatting ====================

template <typename T, int dim>
YT_IMPL_INLINE std::ostream& yt::YTensor<T, dim>::_cout(std::ostream& out) const {
    out << "[YTensor]:<" << yt::type::getTypeName<T>() << ">" << std::endl;
    out << "[itemSize]: " << this->size() << std::endl;
    out << "[byteSize]: " << this->size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = this->shape();
    for (int a = 0; a < static_cast<int>(dims.size() - 1); a++) {
        out << dims[a] << ", ";
    }
    out << dims[static_cast<int>(dims.size()) - 1] << "]" << std::endl;
    out << "[data]:" << std::endl;

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
                // 原生可流式类型直接输出；其他注册类型统一走runtime formatter callback。
                if constexpr (yt::utils::HAVE_OSTREAM<T>) {
                    out << this->at(indices);
                } else {
                    out << yt::type::formatValue(&this->at(indices), this->dtype());
                }
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
    out << std::endl;

    return out;
}
