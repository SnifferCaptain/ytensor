/***************
 * file: ytensor_base.inl
 * purpose: YTensorBase runtime facade、storage构造和layout路由实现。
 ***************/

#include "../include/type/type_dispatch.hpp"
#include "../include/utils/parallel_for.hpp"
#include "../include/ytensor_infos.hpp"
#include "../include/ytensor_types.hpp"

namespace yt {

// ==================== Strided metadata access ====================

// 这些accessor是facade内部唯一的Strided metadata入口；具体shape算法归yt::strided所有。
YT_IMPL_INLINE YMeta<YLayoutType::Strided>& YTensorBase::stridedMeta() {
    return _layout.as<YLayoutType::Strided>();
}

YT_IMPL_INLINE const YMeta<YLayoutType::Strided>& YTensorBase::stridedMeta() const {
    return _layout.as<YLayoutType::Strided>();
}

YT_IMPL_INLINE std::vector<int>& YTensorBase::stridedShape() { return stridedMeta().shape; }

YT_IMPL_INLINE const std::vector<int>& YTensorBase::stridedShape() const { return stridedMeta().shape; }

YT_IMPL_INLINE std::vector<int>& YTensorBase::stridedStride() { return stridedMeta().stride; }

YT_IMPL_INLINE const std::vector<int>& YTensorBase::stridedStride() const { return stridedMeta().stride; }

YT_IMPL_INLINE int& YTensorBase::stridedOffset() { return stridedMeta().offset; }

YT_IMPL_INLINE const int& YTensorBase::stridedOffset() const { return stridedMeta().offset; }

// ==================== construction and shared ownership ====================

YT_IMPL_INLINE std::vector<int> YTensorBase::shape() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::shape(*this);
        default:
            throw std::runtime_error("YTensorBase::shape: layout not implemented");
    }
}

// 构造连续Strided tensor，并为注册非POD dtype建立完整对象生命周期。
// scalar空shape包含一个元素；任一零extent产生空tensor且不构造非POD伪元素。
YT_IMPL_INLINE YTensorBase::YTensorBase(const std::vector<int>& shape, const std::string& dtype) {
    for (int extent : shape) {
        if (extent < 0) {
            throw std::invalid_argument("YTensorBase: shape extents must be non-negative");
        }
    }
    stridedShape() = shape;
    stridedOffset() = 0;
    int d = ndim();
    stridedStride().assign(d, 0);
    if (d > 0) {
        size_t logicalStride = 1;
        for (int i = d - 1; i >= 0; --i) {
            if (logicalStride > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error("YTensorBase: logical stride exceeds int range");
            }
            stridedStride()[i] = static_cast<int>(logicalStride);
            if (stridedShape()[i] != 0 &&
                logicalStride > std::numeric_limits<size_t>::max() / stridedShape()[i]) {
                throw std::overflow_error("YTensorBase: shape element count overflow");
            }
            logicalStride *= static_cast<size_t>(stridedShape()[i]);
        }
    }
    _dtype = dtype;
    _element_size = static_cast<size_t>(yt::type::getTypeSize(dtype));
    // shape/stride先完整验证；只有metadata可表示后才开始分配storage。
    size_t total = 1;
    for (int v : stridedShape()) {
        if (v != 0 && total > std::numeric_limits<size_t>::max() / static_cast<size_t>(v)) {
            throw std::overflow_error("YTensorBase: shape element count overflow");
        }
        total *= static_cast<size_t>(v);
    }
    yt::utils::checkedIntSize(total, "YTensorBase");
    if (_element_size != 0 && total > std::numeric_limits<size_t>::max() / _element_size) {
        throw std::overflow_error("YTensorBase: storage byte size overflow");
    }

    // 非POD数组的核心不变量：成功构造的元素形成前缀，且每个对象最终只析构一次。
    // _memory在全部元素构造成功前不发布，因此callback异常不会暴露半初始化tensor。
    auto typeInfoOpt = yt::type::getTypeInfo(dtype);
    if (typeInfoOpt && !typeInfoOpt->get().isPOD) {
        const auto& typeInfo = typeInfoOpt->get();
        size_t elemSize = _element_size;
        size_t numElems = total;
        auto destructor = typeInfo.destructor;
        auto defaultConstruct = typeInfo.defaultConstruct;
        if (!defaultConstruct) {
            throw std::runtime_error("YTensorBase: non-POD type has no defaultConstruct registered");
        }

        if (total == 0) {
            _memory = YMemory();  // 零大小非POD：不构造任何元素，不分配内存，避免"无中生有"的析构
            return;
        }
        char* rawPtr = new char[total * _element_size];
        size_t constructed = 0;
        try {
            for (; constructed < numElems; ++constructed) {
                defaultConstruct(rawPtr + constructed * elemSize);
            }
        } catch (...) {
            // 构造函数中途抛异常：回滚已构造的元素，然后释放裸内存
            if (destructor) {
                for (size_t i = 0; i < constructed; ++i) {
                    destructor(rawPtr + i * elemSize);
                }
            }
            delete[] rawPtr;
            throw;
        }

        // false禁止YMemory按字节clone对象；后续复制必须经过dtype lifecycle callback。
        _memory = YMemory(
            std::shared_ptr<char[]>(
                rawPtr,
                [destructor, elemSize, numElems](char* ptr) {
                    if (destructor) {
                        for (size_t i = 0; i < numElems; ++i) {
                            destructor(ptr + i * elemSize);
                        }
                    }
                    delete[] ptr;
                }
            ),
            total * _element_size, "cpu", false
        );
    } else {
        if (total > 0) {
            _memory = YMemory(
                std::shared_ptr<char[]>(new char[total * _element_size]), total * _element_size
            );
        }
    }
    stridedOffset() = 0;
}

// 复制metadata和dtype，同时共享values storage所有权。
YT_IMPL_INLINE YTensorBase::YTensorBase(const YTensorBase& other) {
    _layout = other._layout;
    _memory = other._memory;
    _element_size = other._element_size;
    _dtype = other._dtype;
}

YT_IMPL_INLINE YTensorBase& YTensorBase::operator=(const YTensorBase& other) {
    if (this != &other) {
        // 通过YTensorBase&给typed派生对象赋值时，必须先保护其固定dtype/rank不变量。
        validateReplacement(other);
        // 先完整构造replacement，再一次性提交；metadata/string复制失败时保持原tensor不变。
        YLayout replacementLayout = other._layout;
        YMemory replacementMemory = other._memory;
        std::string replacementDtype = other._dtype;

        _layout = std::move(replacementLayout);
        _memory = std::move(replacementMemory);
        _element_size = other._element_size;
        _dtype = std::move(replacementDtype);
    }
    return *this;
}

// ==================== layout facade queries ====================

// 以下函数只负责选择layout owner，不在facade中复制Strided算法。
YT_IMPL_INLINE int YTensorBase::shape(int atDim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::shape(*this, atDim);
        default:
            throw std::runtime_error("YTensorBase::shape: layout not implemented");
    }
}

YT_IMPL_INLINE std::vector<int> YTensorBase::stride() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::logicalStride(*this);
        default:
            throw std::runtime_error("YTensorBase::stride: layout not implemented");
    }
}

YT_IMPL_INLINE std::vector<int> YTensorBase::stride_() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::physicalStride(*this);
        default:
            throw std::runtime_error("YTensorBase::stride_: layout not implemented");
    }
}

YT_IMPL_INLINE int YTensorBase::stride_(int atDim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::physicalStride(*this, atDim);
        default:
            throw std::runtime_error("YTensorBase::stride_: layout not implemented");
    }
}

YT_IMPL_INLINE int YTensorBase::stride(int atDim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::logicalStride(*this, atDim);
        default:
            throw std::runtime_error("YTensorBase::stride: layout not implemented");
    }
}

YT_IMPL_INLINE size_t YTensorBase::size() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::size(*this);
        default:
            throw std::runtime_error("YTensorBase::size: layout not implemented");
    }
}

YT_IMPL_INLINE int YTensorBase::ndim() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::ndim(*this);
        default:
            throw std::runtime_error("YTensorBase::ndim: layout not implemented");
    }
}

YT_IMPL_INLINE int YTensorBase::shapeSize() const { return ndim(); }

YT_IMPL_INLINE bool YTensorBase::shapeMatch(const std::vector<int>& otherShape) const {
    return shape() == otherShape;
}

YT_IMPL_INLINE void YTensorBase::shallowCopyTo(YTensorBase& other) const {
    other = *this;
}

YT_IMPL_INLINE YTensorBase YTensorBase::clone() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::clone(*this);
        default:
            throw std::runtime_error("YTensorBase::clone: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::copy_(const YTensorBase& src) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::copy_(*this, src);
        default:
            throw std::runtime_error("YTensorBase::copy_: layout not implemented");
    }
}

YT_IMPL_INLINE std::string YTensorBase::dtype() const { return _dtype; }
YT_IMPL_INLINE size_t YTensorBase::elementSize() const { return _element_size; }
YT_IMPL_INLINE std::string YTensorBase::device() const { return _memory.device(); }
YT_IMPL_INLINE size_t YTensorBase::nbytes() const { return _memory.nbytes(); }
YT_IMPL_INLINE YLayoutType YTensorBase::layoutType() const { return _layout.type(); }
YT_IMPL_INLINE bool YTensorBase::isStrided() const { return layoutType() == YLayoutType::Strided; }
YT_IMPL_INLINE bool YTensorBase::isNested() const { return layoutType() == YLayoutType::Nested; }

YT_IMPL_INLINE bool YTensorBase::isContiguous(int fromDim, int toDim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::isContiguous(*this, fromDim, toDim);
        default:
            throw std::runtime_error("YTensorBase::isContiguous: layout not implemented");
    }
}

YT_IMPL_INLINE int YTensorBase::isContiguousFrom(int fromDim, int toDim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::isContiguousFrom(*this, fromDim, toDim);
        default:
            throw std::runtime_error("YTensorBase::isContiguousFrom: layout not implemented");
    }
}

YT_IMPL_INLINE size_t YTensorBase::toIndex_(const std::vector<int>& pos) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::physicalIndex(*this, pos);
        default:
            throw std::runtime_error("YTensorBase::toIndex_: layout not implemented");
    }
}
YT_IMPL_INLINE size_t YTensorBase::toIndex(const std::vector<int>& pos) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::logicalIndex(*this, pos);
        default:
            throw std::runtime_error("YTensorBase::toIndex: layout not implemented");
    }
}

YT_IMPL_INLINE std::vector<int> YTensorBase::toCoord(size_t index) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::coordinate(*this, index);
        default:
            throw std::runtime_error("YTensorBase::toCoord: layout not implemented");
    }
}

// note: calculate_logical_stride removed; stride() returns logical strides

YT_IMPL_INLINE bool YTensorBase::isDisjoint() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::isDisjoint(*this);
        default:
            throw std::runtime_error("YTensorBase::isDisjoint: layout not implemented");
    }
}

YT_IMPL_INLINE std::vector<int> YTensorBase::autoShape(const std::vector<int>& shape) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::autoShape(*this, shape);
        default:
            throw std::runtime_error("YTensorBase::autoShape: layout not implemented");
    }
}

// ==================== layout facade view transforms ====================

YT_IMPL_INLINE YTensorBase YTensorBase::slice(int atDim, int start, int end, int step, bool autoFix) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::slice(*this, atDim, start, end, step, autoFix);
        default:
            throw std::runtime_error("YTensorBase::slice: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::slice_(int atDim, int start, int end, int step, bool autoFix) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::slice_(*this, atDim, start, end, step, autoFix);
        default:
            throw std::runtime_error("YTensorBase::slice_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::permute(const std::vector<int>& newOrder) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::permute(*this, newOrder);
        default:
            throw std::runtime_error("YTensorBase::permute: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::permute_(const std::vector<int>& newOrder) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::permute_(*this, newOrder);
        default:
            throw std::runtime_error("YTensorBase::permute_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::transpose(int dim0, int dim1) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::transpose(*this, dim0, dim1);
        default:
            throw std::runtime_error("YTensorBase::transpose: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::view(const std::vector<int>& newShape) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::view(*this, newShape);
        default:
            throw std::runtime_error("YTensorBase::view: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::reshape(const std::vector<int>& newShape) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::reshape(*this, newShape);
        default:
            throw std::runtime_error("YTensorBase::reshape: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::unsqueeze(int dim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::unsqueeze(*this, dim);
        default:
            throw std::runtime_error("YTensorBase::unsqueeze: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::unsqueeze_(int dim) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::unsqueeze_(*this, dim);
        default:
            throw std::runtime_error("YTensorBase::unsqueeze_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::squeeze(int dim) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::squeeze(*this, dim);
        default:
            throw std::runtime_error("YTensorBase::squeeze: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::squeeze_(int dim) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::squeeze_(*this, dim);
        default:
            throw std::runtime_error("YTensorBase::squeeze_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::repeat(const std::vector<int>& times) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::repeat(*this, times);
        default:
            throw std::runtime_error("YTensorBase::repeat: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::repeat_(const std::vector<int>& times) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::repeat_(*this, times);
        default:
            throw std::runtime_error("YTensorBase::repeat_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::unfold(int atDim, int kernel, int stride, int dilation) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::unfold(*this, atDim, kernel, stride, dilation);
        default:
            throw std::runtime_error("YTensorBase::unfold: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::unfold_(int atDim, int kernel, int stride, int dilation) {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::unfold_(*this, atDim, kernel, stride, dilation);
        default:
            throw std::runtime_error("YTensorBase::unfold_: layout not implemented");
    }
}
// non-inplace overloads removed; inplace versions with trailing '_' are provided above

YT_IMPL_INLINE YTensorBase YTensorBase::mostContinuousView() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::mostContinuousView(*this);
        default:
            throw std::runtime_error("YTensorBase::mostContinuousView: layout not implemented");
    }
}

// ==================== random factories ====================

// 设置进程级随机引擎种子；所有typed/runtime generator共享该引擎。
YT_IMPL_INLINE void YTensorBase::seed(unsigned int seed) {
    std::lock_guard<std::mutex> lock(yt::info::rngMutex);
    yt::info::gen.seed(seed);
}

// 生成正态分布tensor；float64直接生成，其余dtype经float32和统一cast链转换。
YT_IMPL_INLINE YTensorBase
YTensorBase::_RandnGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::lock_guard<std::mutex> lock(yt::info::rngMutex);
    if (dtype == "float64") {
        YTensorBase op(shape, dtype);
        double* ptr = op.data<double>();
        for (size_t i = 0; i < op.size(); ++i) ptr[i] = dist(gen);
        return op;
    }
    YTensorBase values(shape, "float32");
    float* ptr = values.data<float>();
    for (size_t i = 0; i < values.size(); ++i) ptr[i] = static_cast<float>(dist(gen));
    return dtype == "float32" ? values : values.cast(dtype);
}

// 生成[0,1)均匀分布tensor；锁覆盖整个生成过程以保护共享engine。
YT_IMPL_INLINE YTensorBase
YTensorBase::_RanduGenerator::operator()(const std::vector<int>& shape, std::string dtype) const {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::lock_guard<std::mutex> lock(yt::info::rngMutex);
    if (dtype == "float64") {
        YTensorBase op(shape, dtype);
        double* ptr = op.data<double>();
        for (size_t i = 0; i < op.size(); ++i) ptr[i] = dist(gen);
        return op;
    }
    YTensorBase values(shape, "float32");
    float* ptr = values.data<float>();
    for (size_t i = 0; i < values.size(); ++i) ptr[i] = static_cast<float>(dist(gen));
    return dtype == "float32" ? values : values.cast(dtype);
}

// ==================== factories and copy-oriented operations ====================

YT_IMPL_INLINE YTensorBase YTensorBase::zeros(const std::vector<int>& shape, std::string dtype) {
    YTensorBase op(shape, dtype);

    // 检查是否为非POD类型
    auto typeInfoOpt = yt::type::getTypeInfo(dtype);
    if (typeInfoOpt && !typeInfoOpt->get().isPOD) {
        // 非POD类型：构造函数已调用defaultConstruct，不应再用memset覆盖
        // 保持默认构造状态即可
        return op;
    }

    // POD类型：直接memset清零
    size_t total = op.size();
    size_t bytes = total * op.elementSize();
    if (op._memory) std::memset(op._memory.get(), 0, bytes);
    return op;
}

YT_IMPL_INLINE YTensorBase YTensorBase::ones(const std::vector<int>& shape, std::string dtype) {
    // 用int32工厂produce + cast复用类型转换链，避免为每种dtype手写赋值循环
    YTensorBase values(shape, "int32");
    int32_t* data = values.data<int32_t>();
    for (size_t i = 0; i < values.size(); ++i) data[i] = 1;
    return dtype == "int32" ? values : values.cast(dtype);
}

YT_IMPL_INLINE YTensorBase YTensorBase::contiguous() const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::contiguous(*this);
        default:
            throw std::runtime_error("YTensorBase::contiguous: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase& YTensorBase::contiguous_() {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::contiguous_(*this);
        default:
            throw std::runtime_error("YTensorBase::contiguous_: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::concat(const std::vector<YTensorBase>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("[YTensorBase::concat] Empty tensor list");
    }
    switch (tensors[0].layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::concat(tensors, axis);
        default:
            throw std::runtime_error("YTensorBase::concat: layout not implemented");
    }
}

YT_IMPL_INLINE YTensorBase YTensorBase::concat(const YTensorBase& other, int axis) const {
    return YTensorBase::concat({*this, other}, axis);
}

YT_IMPL_INLINE std::vector<YTensorBase> YTensorBase::split(const std::vector<int>& splitSizes, int axis)
    const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::split(*this, splitSizes, axis);
        default:
            throw std::runtime_error("YTensorBase::split: layout not implemented");
    }
}

YT_IMPL_INLINE std::vector<YTensorBase> YTensorBase::split(int n, int axis) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::split(*this, n, axis);
        default:
            throw std::runtime_error("YTensorBase::split: layout not implemented");
    }
}

// ==================== runtime formatting ====================

YT_IMPL_INLINE std::ostream& operator<<(std::ostream& out, const YTensorBase& tensor) {
    return tensor._cout(out);
}

// 按逻辑坐标格式化runtime tensor，支持非连续view和注册custom dtype formatter。
YT_IMPL_INLINE std::ostream& YTensorBase::_cout(std::ostream& out) const {
    out << "[YTensorBase]:<" << this->dtype() << ">" << std::endl;
    out << "[itemSize]: " << this->size() << std::endl;
    out << "[byteSize]: " << this->size() * this->elementSize() << std::endl;
    out << "[shape]: [";
    for (int i = 0; i < this->ndim(); ++i) {
        out << this->shape(i) << (i + 1 == this->ndim() ? "" : ", ");
    }
    out << "]" << std::endl;
    out << "[data]:" << std::endl;

    // 逐逻辑坐标计算物理位置，不能假设transpose/slice后的values连续。
    std::vector<int> dims = this->shape();
    if (dims.size() == 0) {
        // scalar case
        if (!this->_memory) {
            out << "[data]: null" << std::endl;
        } else {
            size_t phys = 0;  // scalar
            size_t addressIndex = static_cast<size_t>(this->stridedOffset()) + phys;
            const void* valPtr =
                static_cast<const void*>(this->_memory.get() + addressIndex * this->elementSize());
            out << yt::type::formatValue(valPtr, this->dtype());
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
                        size_t addressIndex = static_cast<size_t>(this->stridedOffset()) + phys;
                        const void* valPtr = static_cast<const void*>(
                            this->_memory.get() + addressIndex * this->elementSize()
                        );
                        out << yt::type::formatValue(valPtr, this->dtype());
                    } catch (...) {
                        // 打印是诊断路径：单元素索引/formatter失败用省略号表示，不中断其余tensor信息。
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

}  // namespace yt
