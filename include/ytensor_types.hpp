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
#include "./ytensor_concepts.hpp"
#include "./ytensor_infos.hpp"
#include "./types/bfloat16.hpp"

namespace yt::types {
    /// @brief 获取数据类型名称
    /// @tparam T 数据类型
    /// @return 数据类型名称字符串
    template<typename T>
    std::string getTypeName() {
        // 首先检查是否已注册自定义名称
        auto& registry = yt::infos::getTypeRegistry();
        auto it = registry.find(typeid(T).name());
        if (it != registry.end()) {
            return it->second.first;
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
        // non std
        else if constexpr (std::is_same_v<T, yt::bfloat16>) return "bfloat16";
        else {
            throw std::runtime_error(std::string("Type ") + typeid(T).name() + " is not registered.");
            return "unregistered";
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
        else {
            // registered custom types
            auto& registry = yt::infos::getTypeRegistry();// std::unordered_map<std::string, std::pair<std::string, int32_t>>
            for(auto& [key, value] : registry) {
                if (value.first == typeName) {
                    return value.second;
                }
            }
            // unk
            throw std::runtime_error(std::string("Type ") + typeName + " is not registered.");
            return 0;
        }
    }

    /// @brief 注册自定义类型
    /// @tparam T 要注册的类型
    /// @param typeName 自定义类型名称
    template<typename T>
    void registerType(const std::string& typeName) {
        auto& registry = yt::infos::getTypeRegistry();
        int32_t typeSize = getTypeSize<T>();
        registry[typeid(T).name()] = {typeName, typeSize};
    }
} // namespace yt::types