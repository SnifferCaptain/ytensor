#pragma once
/***************
 * file: ytensor_base_templates.inl
 * purpose: YTensorBase模板访问器和参数包便捷重载实现。
 ***************/

#include <cstring>

namespace yt {

// ==================== coordinate offsets ====================

// 返回相对当前view origin的Strided元素偏移，不包含layout base offset。
template <typename... Args>
inline int YTensorBase::offset(Args... index) const {
    std::vector<int> indices = {index...};
    return offset(indices);
}

inline int YTensorBase::offset(const std::vector<int>& index) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::relativeOffset(*this, index);
        default:
            throw std::runtime_error("YTensorBase::offset: layout not implemented");
    }
}

// 返回相对storage起点的绝对元素偏移，包含layout base offset。
template <typename... Args>
inline int YTensorBase::offset_(Args... index) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::storageOffset(*this, std::vector<int>{index...});
        default:
            throw std::runtime_error("YTensorBase::offset_: layout not implemented");
    }
}

inline int YTensorBase::offset_(const std::vector<int>& index) const {
    switch (layoutType()) {
        case YLayoutType::Strided:
            return yt::strided::storageOffset(*this, index);
        default:
            throw std::runtime_error("YTensorBase::offset_: layout not implemented");
    }
}

// ==================== typed pointer access ====================

// 校验element size和runtime dtype后返回当前view origin的typed pointer。
// 注意：pointer不表示逻辑连续；调用方仍需检查isContiguous或使用stride访问。
template <typename T>
inline T* YTensorBase::data() {
    if (!_memory) return nullptr;
    if (_element_size != sizeof(T)) throw std::invalid_argument("YTensorBase::data: element size mismatch");
    if constexpr (!yt::utils::is_ytensor_v<T>) {
        const std::string expected = yt::type::getTypeName<std::remove_cv_t<T>>();
        // 兼容历史unsigned float8名称；typed facade对外使用canonical float8_e8m0。
        const bool float8Alias = expected == "float8_e8m0" && _dtype == "float8_ue8m0";
        if (_dtype != expected && !float8Alias) {
            throw std::invalid_argument("YTensorBase::data: dtype mismatch");
        }
    }
    // 指针仅在对应storage保持存活且未被当前tensor替换期间有效；共享句柄可延长storage生命周期。
    return reinterpret_cast<T*>(_memory.get()) + yt::strided::storageOffset(*this, {});
}

template <typename T>
inline const T* YTensorBase::data() const {
    if (!_memory) return nullptr;
    if (_element_size != sizeof(T)) throw std::invalid_argument("YTensorBase::data: element size mismatch");
    if constexpr (!yt::utils::is_ytensor_v<T>) {
        const std::string expected = yt::type::getTypeName<std::remove_cv_t<T>>();
        // 兼容历史unsigned float8名称；typed facade对外使用canonical float8_e8m0。
        const bool float8Alias = expected == "float8_e8m0" && _dtype == "float8_ue8m0";
        if (_dtype != expected && !float8Alias) {
            throw std::invalid_argument("YTensorBase::data: dtype mismatch");
        }
    }
    // 指针仅在对应storage保持存活且未被当前tensor替换期间有效；共享句柄可延长storage生命周期。
    return reinterpret_cast<const T*>(_memory.get()) + yt::strided::storageOffset(*this, {});
}

inline float* YTensorBase::data() { return data<float>(); }

inline const float* YTensorBase::data() const { return data<float>(); }

inline char* YTensorBase::rawData() {
    return _memory ? _memory.get() + static_cast<std::ptrdiff_t>(yt::strided::storageOffset(*this, {})) * _element_size : nullptr;
}

inline const char* YTensorBase::rawData() const {
    return _memory ? _memory.get() + static_cast<std::ptrdiff_t>(yt::strided::storageOffset(*this, {})) * _element_size : nullptr;
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

// ==================== checked element access ====================

template <typename T, typename... Args>
inline T& YTensorBase::at(Args... args) {
    // data<T>()的返回值无需使用；调用它是为了在reinterpret前统一校验dtype和element size。
    (void)data<T>();
    std::vector<int> pos = {args...};
    return reinterpret_cast<T*>(_memory.get())[offset_(pos)];
}

template <typename T>
inline T& YTensorBase::at(const std::vector<int>& pos) {
    // 先校验runtime dtype，再使用包含layout base offset的storage下标。
    (void)data<T>();
    return reinterpret_cast<T*>(_memory.get())[offset_(pos)];
}

template <typename T>
inline const T& YTensorBase::at(const std::vector<int>& pos) const {
    // 先校验runtime dtype，再使用包含layout base offset的storage下标。
    (void)data<T>();
    return reinterpret_cast<const T*>(_memory.get())[offset_(pos)];
}

// 将row-major逻辑下标转换为坐标后访问，因此适用于任意合法Strided view。
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

// 从view-origin pointer直接线性访问；仅用于调用方已确认物理布局的热路径。
template <typename T>
inline T& YTensorBase::atData_(int index) {
    return this->data<T>()[index];
}

template <typename T>
inline const T& YTensorBase::atData_(int index) const {
    return this->data<T>()[index];
}

template <typename... Args>
inline std::vector<int> YTensorBase::autoShape(const Args... shape0) const {
    std::vector<int> shape({shape0...});
    return autoShape(shape);
}

template <typename... Args>
inline YTensorBase YTensorBase::permute(const Args... newOrder) const {
    return permute(std::vector<int>{static_cast<int>(newOrder)...});
}

template <typename... Args>
inline YTensorBase YTensorBase::view(const Args... newShape) const {
    return view(std::vector<int>{static_cast<int>(newShape)...});
}

template <typename... Args>
inline YTensorBase YTensorBase::reshape(const Args... newShape) const {
    return reshape(std::vector<int>{static_cast<int>(newShape)...});
}

template <typename... Args>
inline YTensorBase YTensorBase::repeat(const Args... times) const {
    return repeat(std::vector<int>{static_cast<int>(times)...});
}

}  // namespace yt
