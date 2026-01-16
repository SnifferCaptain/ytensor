#include "../include/ytensor_infos.hpp"
#include "../include/ytensor_types.hpp"
#include "../include/kernel/parallel_for.hpp"
#include "../include/kernel/type_dispatch.hpp"

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
    
    size_t dstElemSize = _element_size;
    size_t srcElemSize = src._element_size;
    int d = ndim();
    int total = static_cast<int>(size());
    
    // 检查是否存在内存重叠
    bool mayOverlap = (_data.get() == src._data.get());
    
    // 检查类型是否相同
    bool sameType = (_dtype == src._dtype);
    
    // 如果类型相同且两者都是完全连续且无重叠，直接memcpy
    if (sameType && this->isContiguous() && src.isContiguous() && !mayOverlap) {
        std::memcpy(_data.get() + _offset * dstElemSize, 
                    src._data.get() + src._offset * srcElemSize, 
                    static_cast<size_t>(total) * dstElemSize);
        return *this;
    }
    
    // 处理重叠情况：先复制源数据到临时缓冲区
    std::unique_ptr<char[]> tempBuffer;
    const char* srcBasePtr = src._data.get();
    bool needTemp = mayOverlap;
    
    if (needTemp) {
        // 分配临时缓冲区存储源数据
        tempBuffer = std::make_unique<char[]>(static_cast<size_t>(total) * srcElemSize);
        
        // 复制源数据到临时缓冲区（处理非连续情况）
        if (src.isContiguous()) {
            std::memcpy(tempBuffer.get(), srcBasePtr + src._offset * srcElemSize, 
                        static_cast<size_t>(total) * srcElemSize);
        } else {
            auto srcLogicStride = src.stride();
            yt::kernel::parallelFor(0, total, [&](int index) {
                size_t srcIndex = src._offset;
                size_t remaining = static_cast<size_t>(index);
                for (int i = 0; i < d; i++) {
                    int coord = static_cast<int>((remaining / srcLogicStride[i]) % src._shape[i]);
                    srcIndex += coord * src._stride[i];
                    remaining = remaining % srcLogicStride[i];
                }
                std::memcpy(tempBuffer.get() + index * srcElemSize, 
                            srcBasePtr + srcIndex * srcElemSize, 
                            srcElemSize);
            });
        }
        srcBasePtr = tempBuffer.get();
    }
    
    char* dstBasePtr = _data.get();
    auto thisLogicStride = this->stride();
    auto srcLogicStride = src.stride();
    
    // 辅助lambda：计算dst物理索引
    auto calcDstIndex = [&](int index) -> size_t {
        size_t dstIndex = _offset;
        size_t remaining = static_cast<size_t>(index);
        for (int i = 0; i < d; i++) {
            int coord = static_cast<int>((remaining / thisLogicStride[i]) % _shape[i]);
            dstIndex += coord * _stride[i];
            remaining = remaining % thisLogicStride[i];
        }
        return dstIndex;
    };
    
    // 辅助lambda：计算src物理索引
    auto calcSrcIndex = [&](int index) -> size_t {
        if (needTemp) {
            return static_cast<size_t>(index);
        }
        size_t srcIndex = src._offset;
        size_t remaining = static_cast<size_t>(index);
        for (int i = 0; i < d; i++) {
            int coord = static_cast<int>((remaining / srcLogicStride[i]) % src._shape[i]);
            srcIndex += coord * src._stride[i];
            remaining = remaining % srcLogicStride[i];
        }
        return srcIndex;
    };
    
    // 如果类型相同，使用memcpy
    if (sameType) {
        yt::kernel::parallelFor(0, total, [&](int index) {
            size_t dstIndex = calcDstIndex(index);
            size_t srcIndex = calcSrcIndex(index);
            std::memcpy(dstBasePtr + dstIndex * dstElemSize, 
                        srcBasePtr + srcIndex * srcElemSize, 
                        dstElemSize);
        });
    } else {
        // 类型不同，需要类型转换
        // 使用 dispatch2 进行双类型分发
        yt::kernel::dispatch2OrThrow<yt::types::AllNumericTypes, yt::types::AllNumericTypes>(
            src._dtype, _dtype,
            [&]<typename SrcType, typename DstType>() {
                const SrcType* srcPtr = reinterpret_cast<const SrcType*>(srcBasePtr);
                DstType* dstPtr = reinterpret_cast<DstType*>(dstBasePtr);
                yt::kernel::parallelFor(0, total, [&](int index) {
                    size_t dstIndex = calcDstIndex(index);
                    size_t srcIndex = calcSrcIndex(index);
                    dstPtr[dstIndex] = static_cast<DstType>(srcPtr[srcIndex]);
                });
            },
            "copy_"
        );
    }
    
    return *this;
}

inline std::string YTensorBase::dtype() const { return _dtype; }
inline size_t YTensorBase::elementSize() const { return _element_size; }

inline bool YTensorBase::isContiguous(int fromDim, int toDim) const {
    if (_data == nullptr) {
        return false;
    }
    int d = ndim();
    if (d == 0) {
        return true;
    }
    
    auto logStride = this->stride();
    if (logStride.size() != static_cast<size_t>(d)) {
        return false;
    }
    
    // 循环索引处理
    fromDim = (fromDim % d + d) % d;
    // toDim默认-1表示到最后一维（不含），即检查整个张量时等于d
    if (toDim < 0) {
        toDim = d + toDim + 1;  // -1 -> d, 即检查全部
    } else {
        toDim = (toDim % d + d) % d;
    }
    
    if (fromDim >= toDim) {
        return true; // 空范围视为连续
    }
    
    // 检查 [fromDim, toDim) 范围内的维度是否连续
    for (int i = fromDim; i < toDim; ++i) {
        if (logStride[i] != _stride[i] && _shape[i] > 1) {
            return false;
        }
    }
    return true;
}

inline int YTensorBase::isContiguousFrom(int fromDim, int toDim) const {
    if (_data == nullptr) {
        return ndim();
    }
    int d = ndim();
    if (d == 0) {
        return 0;
    }
    
    auto logStride = this->stride();
    if (logStride.size() != static_cast<size_t>(d)) {
        return d;
    }
    
    // 循环索引处理
    fromDim = (fromDim % d + d) % d;
    if (toDim < 0) {
        toDim = d + toDim + 1;
    } else {
        toDim = (toDim % d + d) % d;
    }
    
    if (fromDim >= toDim) {
        return fromDim;
    }
    
    // 从后往前检查，找到第一个不连续的维度
    for (int a = toDim - 1; a >= fromDim; --a) {
        if (logStride[a] != _stride[a] && _shape[a] > 1) {
            return a + 1;
        }
    }
    return fromDim; // 全部连续
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

inline std::ostream &operator<<(std::ostream &out, const YTensorBase &tensor){
    return tensor._cout(out);
}

inline std::ostream& YTensorBase::_cout(std::ostream& out) const {
    out << "[YTensorBase]:<" << this->dtype() << ">" << std::endl;
    out << "[itemSize]: " << this->size() << std::endl;
    out << "[byteSize]: " << this->size() * this->elementSize() << std::endl;
    out << "[shape]: [";
    for (int i = 0; i < this->ndim(); ++i){
        out << this->shape(i) << (i + 1 == this->ndim() ? "" : ", ");
    }
    out << "]" << std::endl;
    out << "[data]:" << std::endl;

    // Print data using runtime dtype and a centralized formatting helper
    std::vector<int> dims = this->shape();
    if (dims.size() == 0) {
        // scalar case
        if (!this->_data) {
            out << "[data]: null" << std::endl;
        } else {
            size_t phys = 0; // scalar
            size_t addressIndex = static_cast<size_t>(this->_offset) + phys;
            const void* valPtr = static_cast<const void*>(this->_data.get() + addressIndex * this->elementSize());
            out << yt::types::formatValue(valPtr, this->dtype());
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
                        size_t phys = this->toIndex_(indices);
                        size_t addressIndex = static_cast<size_t>(this->_offset) + phys;
                        const void* valPtr = static_cast<const void*>(this->_data.get() + addressIndex * this->elementSize());
                        out << yt::types::formatValue(valPtr, this->dtype());
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

}//namespace yt
