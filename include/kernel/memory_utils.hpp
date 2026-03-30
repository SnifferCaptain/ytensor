#pragma once
/***************
 * @file memory_utils.hpp
 * @brief 内存工具函数声明
 ***************/

#include <memory>
#include <cstddef>

namespace yt::kernel {

/// @brief 使用placement new为非POD类型分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param obj 要拷贝构造的对象
/// @return 返回管理内存的shared_ptr<char[]>
template<typename T>
std::shared_ptr<char[]> makeSharedPlacement(const T& obj);

/// @brief 使用placement new为非POD类型数组分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param count 数组元素个数
/// @return 返回管理内存的shared_ptr<char[]>，内存未初始化，需要手动使用placement new
/// @example auto ptr = makeSharedPlacementArray<MatType>(10);
///          for(int i = 0; i < 10; i++) { new (&reinterpret_cast<MatType*>(ptr.get())[i]) MatType(...); }
template<typename T>
std::shared_ptr<char[]> makeSharedPlacementArray(size_t count);

} // namespace yt::kernel

#include "../../src/kernel/memory_utils.inl"
