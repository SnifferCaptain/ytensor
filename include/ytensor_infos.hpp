#pragma once
/***************
* @file: ytensor_infos.hpp
* @brief: 存储一些全局静态信息的命名空间
* @author: SnifferCaptain
* @date: 2025-10-24
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <random>
#include <atomic>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <tuple>
#include <functional>
#include <memory>

namespace yt::infos{
    static constexpr double minParOps = 29609.;
    static constexpr double flopAdd = 1.;
    static constexpr double flopSub = 1.;
    static constexpr double flopMul = 1.;
    static constexpr double flopDiv = 1.;
    static constexpr double flopMod = 0.003;
    static constexpr double ilopAdd = 3.38;
    static constexpr double ilopSub = 3.38;
    static constexpr double ilopMul = 3.38;
    static constexpr double ilopDiv = 3.38;
    static constexpr double ilopMod = 0.03;
    static constexpr double ilopAnd = 1.39;
    static constexpr double ilopOr = 1.39;
    static constexpr double ilopXor = 1.466;

    /// @brief 矩阵乘法多核并行条件判断函数
    inline double flopMatmul(int m, int n, int k) {
        // 64k 16*16 b=32 op=128k
        // 32k 32*32 b=16 op=128k
        // 256k 64*64 b=8 op=2m
        // 1m 128*128 b=8 op=16m
        constexpr float scale = 5e-4;// 5e-2 if debug
        return static_cast<double>(std::max(m * n + n * k, m * n * k)) * scale;
    }

    /// @brief 随机数生成器，调用时必须上锁
    inline std::mt19937 gen = std::mt19937(std::random_device{}());

    /// @brief 随机数生成锁
    inline std::mutex rngMutex;

    /// @brief 设置数据类型转换方式
    static constexpr enum class RoundMode{
        nearest = 0,    // 四舍五入，偏差还行
        nearestEven = 1,// 标准转换，最小偏差
        truncate = 2    // 直接截断，速度最快，偏低。
    } roundMode = RoundMode::nearestEven;

    struct TypeRegItem{
        std::string name;
        int32_t size;
        std::function<std::string(const void*)> toString;
        // 非POD类型支持：析构和拷贝构造
        bool isPOD = true;  // POD类型不需要特殊处理
        std::function<void(void*)> destructor = nullptr;           // 调用析构函数
        std::function<void(void*, const void*)> copyConstruct = nullptr;  // placement new + 拷贝构造
        std::function<void(void*)> defaultConstruct = nullptr;     // placement new + 默认构造
    };

    /// @brief 类型注册表
    /// @return 返回类型注册表的引用
    inline auto& getTypeRegistry() {
        static std::unordered_map<std::string, yt::infos::TypeRegItem> registry;
        return registry;
    }

    /// @brief 文件头标识
    static constexpr std::string_view YTENSOR_FILE_MAGIC = "YTENSORF";

    /// @brief 文件版本
    static constexpr uint8_t YTENSOR_FILE_VERSION = 0;

    /// @brief 控制是否启用Eigen库的宏，默认启用
    #ifndef YT_USE_EIGEN
        #if __has_include(<Eigen/Core>)
            #define YT_USE_EIGEN 1
        #else
            #define YT_USE_EIGEN 0
        #endif
    #endif
    
    /// @brief 控制是否启用YTensorBase模板显式实例化（预创建常用类型模板）
    /// 优点：减少编译时间，可能提升运行时性能（减少模板实例化开销）
    /// 设为0则不预创建，所有模板按需实例化
    #ifndef YT_PREINSTANTIATE_TEMPLATES
        #define YT_PREINSTANTIATE_TEMPLATES 1
    #endif
}// namespace yt::infos

/////////////// extern includes ///////////////

#if YT_USE_EIGEN
#include <Eigen/Core>
#endif // YT_USE_EIGEN
    