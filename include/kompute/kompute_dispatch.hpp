#pragma once

/***************
 * @file: kompute_dispatch.hpp
 * @brief: Kompute GPU着色器调度接口，提供通过Kompute运行SPIR-V着色器的功能
 ***************/

#include "../ytensor_infos.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace yt::kompute {

/// @brief 设置SPIR-V着色器文件所在目录，在使用GPU运算前必须调用
void setShaderDir(const std::string& dir);

/// @brief 获取当前着色器目录
const std::string& getShaderDir();

/// @brief 加载SPIR-V着色器二进制（带缓存）
std::vector<uint32_t> loadSPIRV(const std::string& spvFilename);

#if YT_USE_KOMPUTE

/// @brief 获取全局共享的Kompute Manager（单例，所有GPU张量共用同一Vulkan设备）
kp::Manager& getSharedManager();

/// @brief 二元逐元素操作的GPU调度（float32）
/// @param inputA  输入张量A的host指针
/// @param inputB  输入张量B的host指针
/// @param outputC 输出张量C的host指针（结果写入）
/// @param count   元素个数
/// @param spvFile SPIR-V文件名（如 "binary_arith_float_add.spv"）
void dispatchBinaryFloat(const float* inputA, const float* inputB, float* outputC,
                         size_t count, const std::string& spvFile);

/// @brief 二元逐元素操作的GPU调度（int32）
void dispatchBinaryInt(const int32_t* inputA, const int32_t* inputB, int32_t* outputC,
                       size_t count, const std::string& spvFile);

/// @brief 矩阵乘法的GPU调度（float32）
/// @param A  矩阵A的host指针 [M x K]
/// @param B  矩阵B的host指针 [K x N]
/// @param C  矩阵C的host指针 [M x N]（结果写入）
/// @param M  行数
/// @param N  列数
/// @param K  内维大小
void dispatchMatmul(const float* A, const float* B, float* C,
                    uint32_t M, uint32_t N, uint32_t K);

/// @brief 归约求和的GPU调度（float32）
/// @param input      输入数据的host指针
/// @param output     输出数据的host指针
/// @param axisLength 被归约轴的长度
/// @param outputSize 输出元素个数
void dispatchReductionSum(const float* input, float* output,
                          uint32_t axisLength, uint32_t outputSize);

/// @brief 归约求最大值的GPU调度（float32）
/// @param input     输入数据的host指针
/// @param maxValues 最大值输出的host指针
/// @param argmax    最大值索引输出的host指针
/// @param axisLength 被归约轴的长度
/// @param outputSize 输出元素个数
void dispatchReductionMax(const float* input, float* maxValues, uint32_t* argmax,
                          uint32_t axisLength, uint32_t outputSize);

/// @brief 比较操作的GPU调度（float32输入，uint32输出bool）
void dispatchCmpFloat(const float* inputA, const float* inputB, uint32_t* outputC,
                      size_t count, const std::string& spvFile);

#endif // YT_USE_KOMPUTE

} // namespace yt::kompute

#include "../../src/kompute/kompute_dispatch.inl"
