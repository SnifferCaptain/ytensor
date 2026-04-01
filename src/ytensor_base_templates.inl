#pragma once

#include <cstring>

namespace yt{

template <typename... Args>
inline int YTensorBase::offset(Args... index) const {
    static_assert(sizeof...(index) <= 0 || sizeof...(index) >= 0, "offset template forwarded to toIndex_");
    std::vector<int> indices = {index...};
    return this->toIndex_(indices);
}

inline int YTensorBase::offset(const std::vector<int>& index) const {
    return static_cast<int>(this->toIndex_(index));
}

template <typename... Args>
inline int YTensorBase::offset_(Args... index) const {
    return _offset + offset(index...);
}

inline int YTensorBase::offset_(const std::vector<int>& index) const {
    return _offset + offset(index);
}

template <typename T>
inline T* YTensorBase::data() {
    if (!_data) return nullptr;
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

template<typename... Args>
inline std::vector<int> YTensorBase::autoShape(const Args... shape0) const {
    std::vector<int> shape({shape0...});
    return autoShape(shape);
}

template<typename... Args>
inline YTensorBase YTensorBase::permute(const Args... newOrder) const {
    return permute(std::vector<int>{static_cast<int>(newOrder)...});
}

template<typename... Args>
inline YTensorBase YTensorBase::view(const Args... newShape) const {
    return view(std::vector<int>{static_cast<int>(newShape)...});
}

template<typename... Args>
inline YTensorBase YTensorBase::reshape(const Args... newShape) const {
    return reshape(std::vector<int>{static_cast<int>(newShape)...});
}

template<typename... Args>
inline YTensorBase YTensorBase::repeat(const Args... times) const {
    return repeat(std::vector<int>{static_cast<int>(times)...});
}

} // namespace yt

