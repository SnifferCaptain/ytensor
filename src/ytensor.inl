
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
