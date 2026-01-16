#pragma once
/***************
* @file: ytensor_types.hpp
* @brief: YTensor 数据类型定义
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <cstdint>
#include <string>
#include <unordered_map>
#include <type_traits>
#include <iomanip>
#include <optional>
#include <functional>
#include "./ytensor_concepts.hpp"
#include "./ytensor_infos.hpp"
#include "./types/bfloat16.hpp"
#include "./types/float_spec.hpp"

namespace yt::types {

/// @brief 编译时类型列表模板
template<typename... Types>
struct TypeList {
    static constexpr size_t size = sizeof...(Types);
    
    /// @brief 追加类型到列表末尾
    template<typename... More>
    struct Append {
        using type = TypeList<Types..., More...>;
    };
};

/// @brief TypeList 合并辅助模板
template<typename List1, typename List2>
struct TypeListConcat;

template<typename... T1, typename... T2>
struct TypeListConcat<TypeList<T1...>, TypeList<T2...>> {
    using type = TypeList<T1..., T2...>;
};

/// @brief 标准整数类型（有符号 + 无符号）
using StandardIntTypes = TypeList<
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t
>;

/// @brief 标准浮点类型
using StandardFloatTypes = TypeList<float, double>;

/// @brief 标准数值类型（浮点 + 整数）
using StandardNumericTypes = typename TypeListConcat<StandardFloatTypes, StandardIntTypes>::type;

/// @brief 扩展浮点类型（bfloat16, float16, float8 变体）
using ExtendedFloatTypes = TypeList<
    yt::bfloat16, yt::float16, 
    yt::float8_e5m2, yt::float8_e4m3, yt::float8_e8m0
>;

/// @brief 所有数值类型（标准 + 扩展浮点）
using AllNumericTypes = typename TypeListConcat<StandardNumericTypes, ExtendedFloatTypes>::type;

/// @brief 仅整数类型（用于位运算等）
using IntegerTypes = StandardIntTypes;

/// @brief Eigen 原生支持的类型（不含扩展浮点类型）
using EigenNativeTypes = StandardNumericTypes;

/// @brief 获取数据类型名称
/// @tparam T 数据类型
/// @return 数据类型名称字符串
template<typename T>
std::string getTypeName() {
    // 首先检查是否已注册自定义名称
    auto& registry = yt::infos::getTypeRegistry();
    auto it = registry.find(typeid(T).name());
    if (it != registry.end()) {
        return it->second.name;
    }
    
    // 使用默认的类型名称
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
    else if constexpr (std::is_same_v<T, std::string>) return "string";
    // non std
    else if constexpr (std::is_same_v<T, yt::bfloat16>) return "bfloat16";
    else if constexpr (std::is_same_v<T, yt::float16>) return "float16";
    else if constexpr (std::is_same_v<T, yt::float8_e5m2>) return "float8_e5m2";
    else if constexpr (std::is_same_v<T, yt::float8_e4m3>) return "float8_e4m3";
    else if constexpr (std::is_same_v<T, yt::float8_e8m0>) return "float8_e8m0";
    else if constexpr (std::is_same_v<T, yt::float8_ue8m0>) return "float8_ue8m0";
    else {
        // 未注册，使用默认名称
        return typeid(T).name();
    }
}

/// @brief 获取数据类型大小（模板版本）
/// @tparam T 数据类型
/// @return 类型大小（字节）
template<typename T>
constexpr int32_t getTypeSize() {
    return static_cast<int32_t>(sizeof(T));
}

/// @brief 根据类型名称获取类型大小
/// @param typeName 类型名称
/// @return 类型大小（字节），未知类型返回0
inline int32_t getTypeSize(const std::string& typeName) {
    if (typeName == "float32") return 4;
    else if (typeName == "float64") return 8;
    else if (typeName == "int8") return 1;
    else if (typeName == "int16") return 2;
    else if (typeName == "int32") return 4;
    else if (typeName == "int64") return 8;
    else if (typeName == "uint8") return 1;
    else if (typeName == "uint16") return 2;
    else if (typeName == "uint32") return 4;
    else if (typeName == "uint64") return 8;
    else if (typeName == "bool") return 1;
    // non std
    else if (typeName == "bfloat16") return 2;
    else if (typeName == "float16") return 2;
    else if (typeName == "float8_e5m2") return 1;
    else if (typeName == "float8_e4m3") return 1;
    else if (typeName == "float8_e8m0" || typeName == "float8_ue8m0") return 1;
    else {
        // registered custom types
        auto& registry = yt::infos::getTypeRegistry();// std::unordered_map<std::string, std::pair<std::string, int32_t>>
        for(auto& [key, value] : registry) {
            if (value.name == typeName) {
                return value.size;
            }
        }
        // unk
        throw std::runtime_error(std::string("Type ") + typeName + " is not registered.");
        return 0;
    }
}

/// @brief 根据类型名称获取类型注册信息
/// @param typeName 类型名称
/// @return 类型注册信息的optional引用，未找到返回std::nullopt
inline std::optional<std::reference_wrapper<const yt::infos::TypeRegItem>> getTypeInfo(const std::string& typeName) {
    auto& registry = yt::infos::getTypeRegistry();
    for (auto& [key, value] : registry) {
        if (value.name == typeName) {
            return std::cref(value);
        }
    }
    return std::nullopt;  // 内置类型或未注册类型
}

/// @brief 检查类型是否为POD（或内置类型）
/// @param typeName 类型名称
/// @return true=POD类型，不需要特殊析构处理
inline bool isPODType(const std::string& typeName) {
    // 内置类型都是POD
    if (typeName == "float32" || typeName == "float64" ||
        typeName == "int8" || typeName == "int16" || typeName == "int32" || typeName == "int64" ||
        typeName == "uint8" || typeName == "uint16" || typeName == "uint32" || typeName == "uint64" ||
        typeName == "bool" || typeName == "bfloat16" ||
        typeName == "float16" || typeName == "float8_e5m2" || 
        typeName == "float8_e4m3" || typeName == "float8_e8m0" || typeName == "float8_ue8m0") {
        return true;
    }
    // 检查注册的自定义类型
    auto info = getTypeInfo(typeName);
    return info ? info->get().isPOD : true;  // 未知类型假设为POD
}

/// @brief 注册自定义类型
/// @tparam T 要注册的类型
/// @param typeName 自定义类型名称
template<typename T>
void registerType(const std::string& typeName) {
    auto& registry = yt::infos::getTypeRegistry();
    int32_t typeSize = getTypeSize<T>();
    // default formatter: if type has operator<< then use that, else nullptr
    auto makeDefaultFormatter = []() -> std::function<std::string(const void*)> {
        if constexpr (yt::concepts::HAVE_OSTREAM<T>) {
            return [](const void* data) {
                std::ostringstream oss;
                const T* p = reinterpret_cast<const T*>(data);
                oss << *p;
                return oss.str();
            };
        } else {
            return nullptr;
        }
    };
    
    // 非POD类型支持
    yt::infos::TypeRegItem item;
    item.name = typeName;
    item.size = typeSize;
    item.toString = makeDefaultFormatter();
    item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;
    
    if (!item.isPOD) {
        // 析构函数
        item.destructor = [](void* ptr) {
            reinterpret_cast<T*>(ptr)->~T();
        };
        // 拷贝构造
        item.copyConstruct = [](void* dest, const void* src) {
            new (dest) T(*reinterpret_cast<const T*>(src));
        };
        // 默认构造
        if constexpr (std::is_default_constructible_v<T>) {
            item.defaultConstruct = [](void* dest) {
                new (dest) T();
            };
        }
    }
    
    registry[typeid(T).name()] = std::move(item);
}

/// @brief registerType overload that accepts an explicit formatter function
template<typename T>
void registerType(const std::string& typeName, std::function<std::string(const void*)> formatter) {
    auto& registry = yt::infos::getTypeRegistry();
    int32_t typeSize = getTypeSize<T>();
    if (!formatter) {
        if constexpr (yt::concepts::HAVE_OSTREAM<T>) {
            formatter = [](const void* data) {
                std::ostringstream oss;
                const T* p = reinterpret_cast<const T*>(data);
                oss << *p;
                return oss.str();
            };
        } else {
            throw std::invalid_argument("Formatter function cannot be null for type without ostream support.");
        }
    }
    
    // 非POD类型支持
    yt::infos::TypeRegItem item;
    item.name = typeName;
    item.size = typeSize;
    item.toString = formatter;
    item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;
    
    if (!item.isPOD) {
        item.destructor = [](void* ptr) {
            reinterpret_cast<T*>(ptr)->~T();
        };
        item.copyConstruct = [](void* dest, const void* src) {
            new (dest) T(*reinterpret_cast<const T*>(src));
        };
        if constexpr (std::is_default_constructible_v<T>) {
            item.defaultConstruct = [](void* dest) {
                new (dest) T();
            };
        }
    }
    
    registry[typeid(T).name()] = std::move(item);
}

/// @brief 将任意 dtype 的单个元素（原始数据指针）格式化为字符串，用于打印
/// @param data 指向元素起始位置的原始指针
/// @param dtype 元素类型名称（如 "float32"）
/// @return 返回格式化后的字符串
inline std::string formatValue(const void* data, const std::string& dtype) {
    if (!data) return std::string("null");
    std::ostringstream oss;
    // use default formatting; decide casting based on dtype
    if (dtype == "float32") {
        const float* p = reinterpret_cast<const float*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int32") {
        const int32_t* p = reinterpret_cast<const int32_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int64") {
        const int64_t* p = reinterpret_cast<const int64_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int16") {
        const int16_t* p = reinterpret_cast<const int16_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int8") {
        const int8_t* p = reinterpret_cast<const int8_t*>(data);
        // print numeric, not character
        oss << static_cast<int>(*p);
        return oss.str();
    } else if (dtype == "uint8") {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        oss << static_cast<unsigned int>(*p);
        return oss.str();
    } else if (dtype == "uint16") {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "uint32") {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "bool") {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "string") {
        const std::string* p = reinterpret_cast<const std::string*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "bfloat16") {
        const yt::bfloat16* p = reinterpret_cast<const yt::bfloat16*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    }
    else if (dtype == "float16") {
        const yt::float16* p = reinterpret_cast<const yt::float16*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e5m2") {
        const yt::float8_e5m2* p = reinterpret_cast<const yt::float8_e5m2*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e4m3") {
        const yt::float8_e4m3* p = reinterpret_cast<const yt::float8_e4m3*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e8m0" || dtype == "float8_ue8m0") {
        const yt::float8_e8m0* p = reinterpret_cast<const yt::float8_e8m0*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    }
    // 查看自定义注册类型
    auto& registry = yt::infos::getTypeRegistry();
    for (auto& [key, value] : registry) {
        if (value.name == dtype) {
            if (value.toString) {
                return value.toString(data);
            }
            break; // no formatter, fallback
        }
    }
    // fallback打印单字节
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    oss << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*bytes);
    return oss.str();
}
} // namespace yt::types