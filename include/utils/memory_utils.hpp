#pragma once
/***************
 * @file memory_utils.hpp
 * @brief 内存工具函数声明
 ***************/

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

namespace yt::utils {

/// @brief 使用placement new为非POD类型分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param obj 要拷贝构造的对象
/// @return 返回管理内存的shared_ptr<char[]>
template <typename T>
std::shared_ptr<char[]> makeSharedPlacement(const T& obj);

/// @brief 使用placement new为非POD类型数组分配内存，并返回带自定义删除器的shared_ptr
/// @tparam T 要分配的类型
/// @param count 数组元素个数
/// @return 返回管理内存的shared_ptr<char[]>，内存未初始化，需要手动使用placement new
/// @warning 返回的 deleter 会析构全部 count 个元素；调用者必须在所有退出路径上完成全部元素构造。
///          可能发生部分构造或异常时应使用带 initializer 的重载。
/// @example auto ptr = makeSharedPlacementArray<MatType>(10);
///          for(int i = 0; i < 10; i++) { new (&reinterpret_cast<MatType*>(ptr.get())[i]) MatType(...); }
template <typename T>
std::shared_ptr<char[]> makeSharedPlacementArray(size_t count);

/// @brief 分配raw array并逐项调用initializer完成placement construction。
/// @tparam T 数组元素类型
/// @tparam Initializer placement construction callback 类型
/// @param count 数组元素个数
/// @param initializer 接收目标地址和索引，并在目标地址恰好构造一个 T
/// @return 管理所有成功构造对象的shared_ptr；完整成功后管理 count 个对象
/// @note initializer抛异常时只析构此前成功构造的元素。
template <typename T, typename Initializer>
std::shared_ptr<char[]> makeSharedPlacementArray(size_t count, Initializer&& initializer);

} // namespace yt::utils

#include "../../src/utils/memory_utils.inl"
