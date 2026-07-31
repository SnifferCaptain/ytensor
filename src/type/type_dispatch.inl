#pragma once
/***************
 * file: type_dispatch.inl
 * purpose: 类型分发机制实现
 ***************/

namespace yt::type {

// ======================== 类型名称映射 ========================

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

// ======================== 类型列表分发（折叠表达式） ========================

template <typename Func, typename... Ts>
bool dispatchRegisteredTypeList(const std::string& dtype, Func&& func, yt::type::TypeList<Ts...>) {
    // || 折叠表达式：左到右短路求值，找到匹配后立即停止
    return ((dtype == getDTypeName<Ts>() && (func.template operator()<Ts>(), true)) || ...);
}

template <typename TypeListT, typename Func>
void dispatch(const std::string& dtype, Func&& func, const std::string& opName) {
    // 首先尝试直接匹配
    if (dispatchRegisteredTypeList(dtype, std::forward<Func>(func), TypeListT{})) {
        return;
    }
    // 如果直接匹配失败，尝试解析嵌套类型
    std::string baseDtype = yt::type::getBaseDtype(dtype);
    if (baseDtype != dtype) {
        // dtype是嵌套类型，尝试用基础类型匹配
        if (dispatchRegisteredTypeList(baseDtype, std::forward<Func>(func), TypeListT{})) {
            return;
        }
    }
    throw std::runtime_error(opName + ": unsupported dtype: " + dtype);
}

}  // namespace yt::type
