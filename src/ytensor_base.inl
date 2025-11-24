#include "include/ytensor_infos.hpp"
inline std::vector<int> YTensorBase::shape() const {
    return _shape;
}

#include "../include/ytensor_types.hpp"


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
    _data = std::shared_ptr<char[]>(new char[total * _element_size]);
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
    op._stride = _stride;
    op._offset = 0;
    op._dtype = _dtype;
    op._element_size = _element_size;
    // allocate and copy raw bytes
    size_t total = 1;
    for (int v : _shape) total *= std::max(0, v);
    if (total == 0) total = 1;
    op._data = std::shared_ptr<char[]>(new char[total * _element_size]);
    if (_data) {
        std::memcpy(op._data.get(), _data.get(), total * _element_size);
    }
    return op;
}

// non-template atData overloads removed; use template atData<T>() instead

inline std::ostream &operator<<(std::ostream &out, const YTensorBase &tensor){
    out << "YTensorBase(shape=[";
    for (int i = 0; i < tensor.ndim(); ++i){
        out << tensor.shape(i) << (i + 1 == tensor.ndim() ? "" : ", ");
    }
    out << "]";
    // use public accessors if available
    try {
        out << ", dtype=" << tensor.dtype() << ", element_size=" << tensor.elementSize();
    } catch (...) {}
    out << ")";
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

inline size_t YTensorBase::toIndex(const int pos[]) const {
    size_t index = 0;
    auto logical_strides = this->stride();
    for (int i = 0; i < ndim(); ++i) {
        index += static_cast<size_t>(pos[i]) * static_cast<size_t>(logical_strides[i]);
    }
    return index;
}

inline size_t YTensorBase::toIndex_(const int pos[]) const {
    size_t index = 0;
    for (int i = 0; i < ndim(); ++i) {
        index += static_cast<size_t>(pos[i]) * static_cast<size_t>(_stride[i]);
    }
    return index;
}

inline std::vector<int> YTensorBase::toCoord(size_t index) const {
    std::vector<int> pos(ndim());
    #pragma omp simd
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
    return this->data<T>()[index + _offset];
}

template <typename T>
inline const T& YTensorBase::atData(int index) const {
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

YTensorBase YTensorBase::_RandnGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
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

YTensorBase YTensorBase::_RanduGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
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

inline auto YTensorBase::randn = YTensorBase::_RandnGenerator(yt::infos::gen);
inline auto YTensorBase::randu = YTensorBase::_RanduGenerator(yt::infos::gen);

YTensorBase YTensorBase::zeros(const std::vector<int>& shape, std::string dtype) {
    YTensorBase op(shape, dtype);
    size_t total = op.size();
    size_t bytes = total * op.elementSize();
    if (op._data) std::memset(op._data.get(), 0, bytes);
    return op;
}

YTensorBase YTensorBase::ones(const std::vector<int>& shape, std::string dtype) {
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
    YTensorBase op = *this;
    if (_data == nullptr) return op;
    op._stride = op.stride();
    op._offset = 0;
    return op;
}

inline YTensorBase& YTensorBase::contiguous_() {
    if (_data == nullptr) return *this;
    _stride = stride();
    _offset = 0;
    return *this;
}
