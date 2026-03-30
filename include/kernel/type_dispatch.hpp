#pragma once
/***************
 * @file type_dispatch.hpp
 * @brief 基于模板的运行时类型分发机制声明
 * @author SnifferCaptain
 * @date 2025-01-15
 * @version 2.0
 ***************/

#include <string>
#include <stdexcept>
#include "../ytensor_types.hpp"

namespace yt::kernel {

// ======================== 类型名称映射 ========================

/// @brief 获取类型 T 对应的 dtype 字符串（编译时）
template<typename T>
constexpr const char* getDTypeName();

// ======================== 分发实现（折叠表达式） ========================

/// @brief 使用折叠表达式进行类型分发（替代递归展开）
/// 将每个唯一Func类型的模板实例化从O(N)降低到O(1)，其中N为TypeList中的类型数量。
/// 对于15种类型的AllNumericTypes，这意味着约15倍的模板实例化减少。
template<typename Func, typename... Ts>
bool dispatchImpl(const std::string& dtype, Func&& func, yt::types::TypeList<Ts...>);

/// @brief 根据 dtype 字符串分发到对应类型的模板函数
/// @tparam TypeListT 要尝试匹配的类型列表（来自 yt::types）
/// @param dtype 数据类型字符串（如 "float32"）
/// @param func 模板 lambda，形如 [&]<typename DType>() { ... }
/// @return 是否成功匹配并执行
/// @note 如果dtype是嵌套类型（如"YTensorBase<float32>"），会自动解析并匹配内部基础类型
template<typename TypeListT, typename Func>
void dispatch(const std::string& dtype, Func&& func, const std::string& opName = "dispatch");

/// @brief 双类型分发：对 src 和 dst 分别分发
/// @param srcDtype 源类型
/// @param dstDtype 目标类型  
/// @param func 模板 lambda，形如 [&]<typename SrcType, typename DstType>() { ... }
template<typename SrcTypeList, typename DstTypeList, typename Func>
bool dispatch2(const std::string& srcDtype, const std::string& dstDtype, Func&& func);

template<typename SrcTypeList, typename DstTypeList, typename Func>
void dispatch2OrThrow(const std::string& srcDtype, const std::string& dstDtype, 
                       Func&& func, const std::string& opName = "dispatch2");

} // namespace yt::kernel

#include "../../src/kernel/type_dispatch.inl"
