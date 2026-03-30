#pragma once
/***************
 * @file memory_utils.inl
 * @brief 内存工具函数实现
 ***************/

namespace yt::kernel {

template<typename T>
inline std::shared_ptr<char[]> makeSharedPlacement(const T& obj) {
    char* rawMemory = new char[sizeof(T)];
    new (rawMemory) T(obj);
    return std::shared_ptr<char[]>(rawMemory, [](char* ptr) {
        reinterpret_cast<T*>(ptr)->~T();
        delete[] ptr;
    });
}

template<typename T>
inline std::shared_ptr<char[]> makeSharedPlacementArray(size_t count) {
    char* rawMemory = new char[count * sizeof(T)];
    return std::shared_ptr<char[]>(rawMemory, [count](char* ptr) {
        T* arr = reinterpret_cast<T*>(ptr);
        for (size_t i = 0; i < count; ++i) {
            arr[i].~T();
        }
        delete[] ptr;
    });
}

} // namespace yt::kernel
