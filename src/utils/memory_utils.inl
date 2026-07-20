#pragma once
/***************
 * file: memory_utils.inl
 * purpose: 内存工具函数实现
 ***************/

namespace yt::utils {

template <typename T>
inline std::shared_ptr<char[]> makeSharedPlacement(const T& obj) {
    char* rawMemory = static_cast<char*>(::operator new(sizeof(T), std::align_val_t{alignof(T)}));
    try {
        new (rawMemory) T(obj);
    } catch (...) {
        ::operator delete(rawMemory, std::align_val_t{alignof(T)});
        throw;
    }
    return std::shared_ptr<char[]>(rawMemory, [](char* ptr) {
        reinterpret_cast<T*>(ptr)->~T();
        ::operator delete(ptr, std::align_val_t{alignof(T)});
    });
}

template <typename T>
inline std::shared_ptr<char[]> makeSharedPlacementArray(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error("makeSharedPlacementArray: byte size overflow");
    }
    char* rawMemory = static_cast<char*>(::operator new(count * sizeof(T), std::align_val_t{alignof(T)}));
    return std::shared_ptr<char[]>(rawMemory, [count](char* ptr) {
        T* arr = reinterpret_cast<T*>(ptr);
        for (size_t i = 0; i < count; ++i) {
            arr[i].~T();
        }
        ::operator delete(ptr, std::align_val_t{alignof(T)});
    });
}

template <typename T, typename Initializer>
inline std::shared_ptr<char[]> makeSharedPlacementArray(size_t count, Initializer&& initializer) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error("makeSharedPlacementArray: byte size overflow");
    }
    char* rawMemory = static_cast<char*>(::operator new(count * sizeof(T), std::align_val_t{alignof(T)}));
    size_t constructed = 0;
    try {
        for (; constructed < count; ++constructed) {
            initializer(reinterpret_cast<T*>(rawMemory) + constructed, constructed);
        }
    } catch (...) {
        T* arr = reinterpret_cast<T*>(rawMemory);
        while (constructed > 0) arr[--constructed].~T();
        ::operator delete(rawMemory, std::align_val_t{alignof(T)});
        throw;
    }
    return std::shared_ptr<char[]>(rawMemory, [count](char* ptr) {
        T* arr = reinterpret_cast<T*>(ptr);
        for (size_t i = 0; i < count; ++i) arr[i].~T();
        ::operator delete(ptr, std::align_val_t{alignof(T)});
    });
}

}  // namespace yt::utils
