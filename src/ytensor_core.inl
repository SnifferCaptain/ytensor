#include <cstddef>
#include <algorithm>
#include <map>
#include <deque>
#include <cassert>
#include <iostream>
#include <cstdarg>
#include <ranges>
#include <omp.h>

#include "../include/ytensor_types.hpp"
#include "../include/ytensor_core.hpp"

template <typename T, int dim>
void yt::YTensor<T, dim>::throwShapeNotMatch(const std::string& funcName, const std::vector<int>& otherShape) const {
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
void yt::YTensor<T, dim>::throwShapeNotMatch(const std::string& funcName, const std::vector<int>& thisShape, const std::vector<int>& otherShape) {
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
void yt::YTensor<T, dim>::throwShapeSizeNotMatch(const std::string& funcName, int otherDim) {
    std::string errorMsg = "Function \"" + funcName + "\" shape size not match: YTensor<T, " + std::to_string(dim) + "> and dim size " + std::to_string(otherDim);
    throw std::invalid_argument(errorMsg);
}

template <typename T, int dim>
void yt::YTensor<T, dim>::throwOperatorNotSupport(const std::string& typeName, const std::string& opName) {
    std::string errorMsg = "Operator " + opName + " not supported for type " + typeName;
    throw std::runtime_error(errorMsg);
}

///////////////// ytensor ///////////////

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor():
    YTensorBase() {
    _shape.resize(dim, 0);
    _stride.resize(dim, 0);
    _element_size = sizeof(T);
    _dtype = yt::types::getTypeName<T>();
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(const std::vector<int> shape): YTensorBase() {
    if(shape.size() != dim){
        throwShapeSizeNotMatch("init", shape.size());
    }
    _shape = shape;
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = yt::types::getTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
template <typename... Args>
yt::YTensor<T, dim>::YTensor(Args... args): YTensorBase() {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    _shape.resize(dim);
    int a = 0;
    ((_shape[a++] = args), ...);
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = yt::types::getTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(std::initializer_list<int> list): YTensorBase() {
    if (list.size() != dim) {
        throwShapeSizeNotMatch("init", list.size());
    }
    _shape = std::vector<int>(list);
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = yt::types::getTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
}

template <typename T, int dim>
yt::YTensor<T, dim>::YTensor(const YTensorBase& base): YTensorBase(base) {
    // 从 YTensorBase 构造 yt::YTensor
    // 用户需要确保类型和维度匹配
    if (static_cast<int>(base.ndim()) != dim) {
        throwShapeSizeNotMatch("YTensorBase", base.ndim());
    }
}

template<typename T, int dim>
yt::YTensor<T, dim>::YTensor(const yt::YTensor<T, dim>& other): YTensorBase() {
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
yt::YTensor<T, dim> &yt::YTensor<T, dim>::operator=(const yt::YTensor<T, dim> &other){
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
void yt::YTensor<T, dim>::shallowCopyTo(yt::YTensor<T, dim> &other) const {
    YTensorBase::shallowCopyTo(other);
}

template <typename T, int dim>
void yt::YTensor<T, dim>::shareTo(yt::YTensor<T, dim> &other) const {
    shallowCopyTo(other);
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::shallowCopyFrom(const yt::YTensor<T, dim> &src) {
    src.YTensorBase::shallowCopyTo(*this);
    return *this;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::shareFrom(const yt::YTensor<T, dim> &src) {
    shallowCopyFrom(src);
    return *this;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::clone() const {
    // fused coord 2 index
    yt::YTensor<T, dim> op(this->shape());
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
yt::YTensor<T, dim>& yt::YTensor<T, dim>::reserve(const std::vector<int>& shape) {
    if (shape.size() != dim) {
        throwShapeSizeNotMatch("reserve", shape.size());
    }
    _shape = shape;
    _stride = this->stride();
    _offset = 0;
    _element_size = sizeof(T);
    _dtype = yt::types::getTypeName<T>();
    _data = std::shared_ptr<char[]>(new char[this->size() * sizeof(T)]());
    return *this;
}

template <typename T, int dim>
template<typename... Args>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::reserve(Args... args) {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    return reserve(std::vector<int>{args...});
}

template <typename T, int dim>
T* yt::YTensor<T, dim>::data() {
    return data_() + _offset;
}

template <typename T, int dim>
const T* yt::YTensor<T, dim>::data() const {
    return data_() + _offset;
}

template <typename T, int dim>
T* yt::YTensor<T, dim>::data_() {
    return reinterpret_cast<T*>(_data.get());
}

template <typename T, int dim>
const T* yt::YTensor<T, dim>::data_() const {
    return reinterpret_cast<const T*>(_data.get());
}

template<typename T, int dim>
constexpr int yt::YTensor<T, dim>::shapeSize() const {
    return dim;
}

template <typename T, int dim> template <typename... Args>
int yt::YTensor<T, dim>::offset(Args... index) const {
    static_assert(sizeof...(index) <= dim, "Number of arguments must match the dimension");
    // zero pad to dim
    std::vector<int> indices(dim, 0);
    int a = 0;
    ((indices[a++] = index), ...);
    return toIndex_(indices);
}

template <typename T, int dim>
int yt::YTensor<T, dim>::offset(const std::vector<int>& index) const {
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
int yt::YTensor<T, dim>::offset_(Args... index) const {
    return _offset + this->offset(index...);
}

template <typename T, int dim>
int yt::YTensor<T, dim>::offset_(const std::vector<int>& index) const {
    return _offset + this->offset(index);
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::contiguous() const {
    if (_data == nullptr) {
        return yt::YTensor<T, dim>(this->shape());
    }
    if(this->isContiguous()){
        return *this;
    }
    else{
        return this->clone();
    }
}


template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::contiguous_() {
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
size_t yt::YTensor<T, dim>::toIndex(const std::vector<int> &pos) const {
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
size_t yt::YTensor<T, dim>::toIndex_(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return index;
}

template <typename T, int dim>
size_t yt::YTensor<T, dim>::toIndex_(const std::vector<int> &pos) const {
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
T& yt::YTensor<T, dim>::atData(int index) {
    auto coord = toCoord(index);
    return at(coord);
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::atData(int index) const {
    auto coord = toCoord(index);
    return at(coord);
}

template <typename T, int dim>
T& yt::YTensor<T, dim>::atData_(int index) {
    return data_()[_offset + index];
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::atData_(int index) const {
    return data_()[_offset + index];
}

template <typename T, int dim>
template <typename... Args>
T& yt::YTensor<T, dim>::at(const Args... args){
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return atData_(index);
}

template <typename T, int dim>
T& yt::YTensor<T, dim>::at(const std::vector<int>& pos){
    if(pos.size() != dim){
        throwShapeSizeNotMatch("at", pos.size());
    }
    auto index = toIndex_(pos);
    return atData_(index);
}

template <typename T, int dim>
template <typename... Args>
const T& yt::YTensor<T, dim>::at(const Args... args) const {
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    size_t index = 0;
    for (int i = 0; i < dim; ++i){
        index += indices[i] * _stride[i];
    }
    return atData_(index);
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::at(const std::vector<int>& pos) const {
    if(pos.size() != dim){
        throwShapeSizeNotMatch("at", pos.size());
    }
    auto index = toIndex_(pos);
    return atData_(index);
}

template<typename T, int dim>
yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::operator[](int index) requires (dim > 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    std::vector<int> newShape(dim - 1);
    std::vector<int> newStride(dim - 1);
    for (int i = 1; i < dim; ++i) {
        newShape[i - 1] = _shape[i];
        newStride[i - 1] = _stride[i];
    }
    yt::YTensor<T, dim - 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset + index * _stride[0];
    return op;
}

template<typename T, int dim>
const yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::operator[](int index) const requires (dim > 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    std::vector<int> newShape(dim - 1);
    std::vector<int> newStride(dim - 1);
    for (int i = 1; i < dim; ++i) {
        newShape[i - 1] = _shape[i];
        newStride[i - 1] = _stride[i];
    }
    yt::YTensor<T, dim - 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset + index * _stride[0];
    return op;
}

template <typename T, int dim>
T& yt::YTensor<T, dim>::operator[](int index) requires (dim == 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    int bias = index * _stride[0];
    return this->data()[bias];
}

template <typename T, int dim>
const T& yt::YTensor<T, dim>::operator[](int index) const requires (dim == 1) {
    index = (index % _shape[0] + _shape[0]) % _shape[0]; // 循环索引
    int bias = index * _stride[0];
    return this->data()[bias];
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::slice(int atDim, int start, int end, int step, bool autoFix) const {
    return yt::YTensor<T, dim>(YTensorBase::slice(atDim, start, end, step, autoFix));
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::slice_(int atDim, int start, int end, int step, bool autoFix){
    YTensorBase::slice_(atDim, start, end, step, autoFix);
    return *this;
}

template <typename T, int dim> template <typename... Args>
yt::YTensor<T, dim> yt::YTensor<T, dim>::permute(const Args... args) const{
    static_assert(sizeof...(args) == dim, "Number of arguments must match the dimension");
    int indices[dim] = {args...};
    auto newShape = std::vector<int>(dim);
    auto newStride = std::vector<int>(dim);
    for (int i = 0; i < dim; ++i){
        int rotate = (indices[i] % dim + dim) % dim; // 循环索引
        newShape[i] = _shape[rotate];
        newStride[i] = _stride[rotate];
    }
    yt::YTensor<T, dim> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::permute(const std::vector<int> &newOrder) const {
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

template <typename T, int dim> template <typename... Args>
std::vector<int> yt::YTensor<T, dim>::autoShape(const Args... shape0) const {
    // 委托给 vector 版本
    return autoShape(std::vector<int>{shape0...});
}

template <typename T, int dim>
std::vector<int> yt::YTensor<T, dim>::autoShape(const std::vector<int> &shape) const {
    return YTensorBase::autoShape(shape);
}

template <typename T, int dim> template <typename... Args>
auto yt::YTensor<T, dim>::view(const Args... newShape) const -> yt::YTensor<T, sizeof...(Args)> {
    constexpr int newdim = sizeof...(newShape);
    static_assert(sizeof...(newShape) == newdim, "Number of arguments must match the dimension");
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    std::vector<int> shape({newShape...});
    shape = autoShape(shape);
    yt::YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::view(const std::vector<int> &newShape) const { 
    if (newShape.size() != newdim){
        throwShapeSizeNotMatch("view", newShape.size());
    }
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    std::vector<int> shape = autoShape(newShape);
    yt::YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::view(const int newShape[]) const {
    std::vector<int> shape = std::vector<int>(newdim);
    for (int i = 0; i < newdim; ++i){
        shape[i] = newShape[i];
    }
    if(!isContiguous()){
        throw std::invalid_argument("\"view\" requires contiguous tensor.");
    }
    shape = autoShape(shape);
    yt::YTensor<T, newdim> op;
    op._shape = shape;
    op._stride = op.stride();
    op._data = _data;
    op._offset = _offset;
    return op;
}

template <typename T, int dim> template <typename... Args>
auto yt::YTensor<T, dim>::reshape(const Args... newShape) const -> yt::YTensor<T, sizeof...(Args)> {
    return contiguous().template view<sizeof...(Args)>(std::vector<int>{newShape...});
}

template <typename T, int dim> template <int newdim>
yt::YTensor<T, newdim> yt::YTensor<T, dim>::reshape(const std::vector<int>& newShape) const {
    return contiguous().template view<newdim>(newShape);
}

template <typename T, int dim>
yt::YTensor<T, dim + 1> yt::YTensor<T, dim>::unsqueeze(int d) const {
    d = ((d % (dim + 1)) + (dim + 1)) % (dim + 1);
    yt::YTensor<T, dim + 1> op;
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
yt::YTensor<T, dim - 1> yt::YTensor<T, dim>::squeeze(int d) const requires (dim > 1) {
    int actualDim = (d % dim + dim) % dim;
    if (_shape[actualDim] != 1) {
        throw std::runtime_error("squeeze: can only squeeze dimensions of size 1");
    }
    yt::YTensor<T, dim - 1> op;
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
yt::YTensor<T, dim> yt::YTensor<T, dim>::repeat(const Args... times) const {
    static_assert(sizeof...(times) == dim, "Number of arguments must match the dimension");
    std::vector<int> reps = {times...};
    yt::YTensor<T, dim> op;
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
yt::YTensor<T, dim> yt::YTensor<T, dim>::repeat(const std::vector<int> &times) const {
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
yt::YTensor<T, dim + 1> yt::YTensor<T, dim>::unfold(int mdim, int mkernel, int mstride, int mdilation) const{
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

    yt::YTensor<T, dim + 1> op;
    op._shape = newShape;
    op._stride = newStride;
    op._data = _data;
    op._offset = _offset;

    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::zeros(const std::vector<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("zeros", shape.size());
    }
    yt::YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim> template <typename... Args>
yt::YTensor<T, sizeof...(Args)> yt::YTensor<T, dim>::zeros(Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    yt::YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(0));
    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::ones(const std::vector<int>& shape){
    if(shape.size() != dim){
        throwShapeSizeNotMatch("ones", shape.size());
    }
    yt::YTensor<T, dim> op(shape);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim> template <typename... Args>
yt::YTensor<T, sizeof...(Args)> yt::YTensor<T, dim>::ones(const Args... shape) {
    static_assert(sizeof...(shape) == dim, "Number of arguments must match the dimension");
    std::vector<int> shp = {shape...};
    yt::YTensor<T, dim> op(shp);
    op.fill(static_cast<T>(1));
    return op;
}

template <typename T, int dim>
inline typename yt::YTensor<T, dim>::_RandnGenerator yt::YTensor<T, dim>::randn = yt::YTensor<T, dim>::_RandnGenerator(yt::infos::gen);

template <typename T, int dim>
inline typename yt::YTensor<T, dim>::_RanduGenerator yt::YTensor<T, dim>::randu = yt::YTensor<T, dim>::_RanduGenerator(yt::infos::gen);

template <typename T, int dim>
void yt::YTensor<T, dim>::seed(unsigned int seed) {
    std::lock_guard<std::mutex> lock(yt::infos::rngMutex);
    yt::infos::gen.seed(seed);
}

template <typename T, int dim> template <typename Func>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::foreach(Func&& func, double flop){
    // 检测func是否只接受一个参数（不需要coord）
    constexpr bool oneArgFunc = std::is_invocable_v<Func, T&> && !std::is_invocable_v<Func, T&, const std::vector<int>&>;
    
    if constexpr (oneArgFunc) {
        // 无坐标版本：使用broadcastInplace的高效实现
        broadcastInplace([&func](T& a) {
            using ResultType = std::invoke_result_t<Func, T&>;
            if constexpr (std::is_void_v<ResultType>) {
                func(a);
            } else {
                a = func(a);
            }
        });
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
yt::YTensor<T, dim>& yt::YTensor<T, dim>::fill(T value){
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
        broadcastInplace([value](T& item){
            item = value;
        });
    }
    return *this;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::copy_(const yt::YTensorBase& src) {
    yt::YTensorBase::copy_(src);
    return *this;
}

template <typename T, int dim>
inline std::ostream& yt::YTensor<T, dim>::_cout(std::ostream &out) const{
    out << "[YTensor]:<" << yt::types::getTypeName<T>() << ">" << std::endl;
    out << "[itemSize]: " << this->size() << std::endl;
    out << "[byteSize]: " << this->size() * sizeof(T) << std::endl;
    out << "[shape]: [";
    std::vector<int> dims = this->shape();
    for (int a = 0; a < static_cast<int> (dims.size() - 1); a++){
        out << dims[a] << ", ";
    }
    out << dims[static_cast<int> (dims.size()) - 1] << "]" << std::endl;
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
                if constexpr (yt::concepts::HAVE_OSTREAM<T>) {
                    out << this->at(indices);
                } else {
                    out << yt::types::formatValue(&this->at(indices), this->dtype());
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
