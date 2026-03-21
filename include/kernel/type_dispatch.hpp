#pragma once
/***************
 * @file: type_dispatch.hpp
 * @brief: 基于模板的运行时类型分发机制
 * @author: SnifferCaptain
 * @date: 2025-01-15
 * @version 2.0
 ***************/

#include <string>
#include <stdexcept>
#include "../ytensor_types.hpp"

namespace yt::kernel {

// ======================== 类型名称映射 ========================

/// @brief 获取类型 T 对应的 dtype 字符串（编译时）
template<typename T>
constexpr const char* getDTypeName() {
    if constexpr (std::is_same_v<T, float>) return "float32";
    else if constexpr (std::is_same_v<T, double>) return "float64";
    else if constexpr (std::is_same_v<T, int8_t>) return "int8";
    else if constexpr (std::is_same_v<T, int16_t>) return "int16";
    else if constexpr (std::is_same_v<T, int32_t>) return "int32";
    else if constexpr (std::is_same_v<T, int64_t>) return "int64";
    else if constexpr (std::is_same_v<T, uint8_t>) return "uint8";
    else if constexpr (std::is_same_v<T, uint16_t>) return "uint16";
    else if constexpr (std::is_same_v<T, uint32_t>) return "uint32";
    else if constexpr (std::is_same_v<T, uint64_t>) return "uint64";
    else if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_same_v<T, yt::bfloat16>) return "bfloat16";
    else if constexpr (std::is_same_v<T, yt::float16>) return "float16";
    else if constexpr (std::is_same_v<T, yt::float8_e5m2>) return "float8_e5m2";
    else if constexpr (std::is_same_v<T, yt::float8_e4m3>) return "float8_e4m3";
    else if constexpr (std::is_same_v<T, yt::float8_e8m0>) return "float8_e8m0";
    else return nullptr;
}

// ======================== 分发实现（折叠表达式） ========================

/// @brief 使用折叠表达式进行类型分发（替代递归展开）
/// 将每个唯一Func类型的模板实例化从O(N)降低到O(1)，其中N为TypeList中的类型数量。
/// 对于15种类型的AllNumericTypes，这意味着约15倍的模板实例化减少。
template<typename Func, typename... Ts>
bool dispatchImpl(const std::string& dtype, Func&& func, yt::types::TypeList<Ts...>) {
    // || 折叠表达式：左到右短路求值，找到匹配后立即停止
    return ((dtype == getDTypeName<Ts>() && (func.template operator()<Ts>(), true)) || ...);
}

/// @brief 根据 dtype 字符串分发到对应类型的模板函数
/// @tparam TypeListT 要尝试匹配的类型列表（来自 yt::types）
/// @param dtype 数据类型字符串（如 "float32"）
/// @param func 模板 lambda，形如 [&]<typename DType>() { ... }
/// @return 是否成功匹配并执行
/// @note 如果dtype是嵌套类型（如"YTensorBase<float32>"），会自动解析并匹配内部基础类型
template<typename TypeListT, typename Func>
void dispatch(const std::string& dtype, Func&& func, const std::string& opName = "dispatch" ) {
    // 首先尝试直接匹配
    if (dispatchImpl(dtype, std::forward<Func>(func), TypeListT{})) {
        return;
    }
    // 如果直接匹配失败，尝试解析嵌套类型
    std::string baseDtype = yt::types::getBaseDtype(dtype);
    if (baseDtype != dtype) {
        // dtype是嵌套类型，尝试用基础类型匹配
        if(dispatchImpl(baseDtype, std::forward<Func>(func), TypeListT{})) {
            return;
        }
    }
    throw std::runtime_error(opName + ": unsupported dtype: " + dtype);
}

/// @brief 双类型分发：对 src 和 dst 分别分发
/// @param srcDtype 源类型
/// @param dstDtype 目标类型  
/// @param func 模板 lambda，形如 [&]<typename SrcType, typename DstType>() { ... }
template<typename SrcTypeList, typename DstTypeList, typename Func>
bool dispatch2(const std::string& srcDtype, const std::string& dstDtype, Func&& func) {
    bool matched = false;
    dispatch<SrcTypeList>(srcDtype, [&]<typename SrcType>() {
        dispatch<DstTypeList>(dstDtype, [&]<typename DstType>() {
            func.template operator()<SrcType, DstType>();
            matched = true;
        });
    });
    return matched;
}

template<typename SrcTypeList, typename DstTypeList, typename Func>
void dispatch2OrThrow(const std::string& srcDtype, const std::string& dstDtype, 
                       Func&& func, const std::string& opName = "dispatch2") {
    if (!dispatch2<SrcTypeList, DstTypeList>(srcDtype, dstDtype, std::forward<Func>(func))) {
        throw std::runtime_error(opName + ": unsupported dtype pair: " + srcDtype + " -> " + dstDtype);
    }
}
} // namespace yt::kernel
