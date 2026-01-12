#pragma once

#include <memory>
#include <cstddef>

namespace yt::kernel {

/// @brief 使用placement new为非POD类型分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param obj 要拷贝构造的对象
/// @return 返回管理内存的shared_ptr<char[]>
template<typename T>
inline std::shared_ptr<char[]> makeSharedPlacement(const T& obj) {
    char* rawMemory = new char[sizeof(T)];
    new (rawMemory) T(obj);
    return std::shared_ptr<char[]>(rawMemory, [](char* ptr) {
        reinterpret_cast<T*>(ptr)->~T();
        delete[] ptr;
    });
}

/// @brief 使用placement new为非POD类型数组分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param count 数组元素个数
/// @return 返回管理内存的shared_ptr<char[]>，内存未初始化，需要手动使用placement new
/// @example auto ptr = makeSharedPlacementArray<MatType>(10);
///          for(int i = 0; i < 10; i++) { new (&reinterpret_cast<MatType*>(ptr.get())[i]) MatType(...); }
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
