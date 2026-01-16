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

// ======================== 分发实现（递归展开） ========================

/// @brief 分发递归终止
template<typename Func>
bool dispatchImpl(const std::string&, Func&&, yt::types::TypeList<>) {
    return false;
}

/// @brief 分发递归展开
template<typename Func, typename T, typename... Rest>
bool dispatchImpl(const std::string& dtype, Func&& func, yt::types::TypeList<T, Rest...>) {
    if (dtype == getDTypeName<T>()) {
        func.template operator()<T>();
        return true;
    }
    return dispatchImpl(dtype, std::forward<Func>(func), yt::types::TypeList<Rest...>{});
}

/// @brief 根据 dtype 字符串分发到对应类型的模板函数
/// @tparam TypeListT 要尝试匹配的类型列表（来自 yt::types）
/// @param dtype 数据类型字符串（如 "float32"）
/// @param func 模板 lambda，形如 [&]<typename DType>() { ... }
/// @return 是否成功匹配并执行
template<typename TypeListT, typename Func>
bool dispatch(const std::string& dtype, Func&& func) {
    return dispatchImpl(dtype, std::forward<Func>(func), TypeListT{});
}

/// @brief 带默认错误处理的分发（未匹配时抛出异常）
template<typename TypeListT, typename Func>
void dispatchOrThrow(const std::string& dtype, Func&& func, const std::string& opName = "dispatch") {
    if (!dispatch<TypeListT>(dtype, std::forward<Func>(func))) {
        throw std::runtime_error(opName + ": unsupported dtype: " + dtype);
    }
}

// ======================== 双类型分发 ========================

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

// ======================== 带 Trait 检查的分发 ========================

/// @brief 带 trait 检查的分发递归终止
template<template<typename> class Trait, typename Func, typename Fallback>
bool dispatchWithTraitImpl(const std::string&, Func&&, Fallback&&, yt::types::TypeList<>) {
    return false;
}

/// @brief 带 trait 检查的分发递归展开
template<template<typename> class Trait, typename Func, typename Fallback, typename T, typename... Rest>
bool dispatchWithTraitImpl(const std::string& dtype, Func&& func, Fallback&& fallback, yt::types::TypeList<T, Rest...>) {
    if (dtype == getDTypeName<T>()) {
        if constexpr (Trait<T>::value) {
            func.template operator()<T>();
        } else {
            fallback.template operator()<T>();
        }
        return true;
    }
    return dispatchWithTraitImpl<Trait>(dtype, std::forward<Func>(func), 
                                        std::forward<Fallback>(fallback), yt::types::TypeList<Rest...>{});
}

/// @brief 带 trait 检查的分发
/// @tparam Trait type trait（需要有 ::value）
/// @tparam TypeListT 类型列表
/// @param dtype 数据类型字符串
/// @param func 类型支持 trait 时执行的函数
/// @param fallback 类型不支持 trait 时执行的函数
template<template<typename> class Trait, typename TypeListT, typename Func, typename Fallback>
bool dispatchWithTrait(const std::string& dtype, Func&& func, Fallback&& fallback) {
    return dispatchWithTraitImpl<Trait>(dtype, std::forward<Func>(func), 
                                        std::forward<Fallback>(fallback), TypeListT{});
}

/// @brief 带 trait 检查的分发（不支持时抛出异常）
template<template<typename> class Trait, typename TypeListT, typename Func>
void dispatchWithTraitOrThrow(const std::string& dtype, Func&& func, const std::string& opName = "dispatchWithTrait") {
    auto fallback = [&]<typename T>() {
        throw std::runtime_error(opName + ": operation not supported for type: " + dtype);
    };
    if (!dispatchWithTrait<Trait, TypeListT>(dtype, std::forward<Func>(func), fallback)) {
        throw std::runtime_error(opName + ": unsupported dtype: " + dtype);
    }
}

} // namespace yt::kernel
