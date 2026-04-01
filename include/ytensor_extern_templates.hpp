#pragma once
/**********************************************************************
 * @file: ytensor_extern_templates.hpp
 * @brief: YTensor模板extern声明
 * @author: SnifferCaptain
 * @date: 2026-01-29
 * 
 * 说明：
 * 当定义了 YT_USE_LIB 宏时，此文件为预编译的模板类型提供
 * extern template 声明，避免重复实例化，从而加速编译。
 * 
 * 使用方式：
 * 1. 链接 libytensor.a 静态库
 * 2. 在包含 ytensor.hpp 之前定义 YT_USE_LIB：
 *    #define YT_USE_LIB
 *    #include "ytensor.hpp"
 * 
 * 或者通过CMake：
 *    target_compile_definitions(your_target PRIVATE YT_USE_LIB)
 **********************************************************************/

#ifdef YT_USE_LIB

namespace yt {

//=============================================================================
// YTensor extern template 声明
//=============================================================================

// 默认仅启用 YTensorBase 方法模板的 extern。
// 若要实验性启用 YTensor 整类 extern，请额外定义 YT_ENABLE_YTENSOR_CLASS_EXTERN。
#if defined(YT_ENABLE_YTENSOR_CLASS_EXTERN)

// 定义 YTensor extern 声明宏
#define EXTERN_YTENSOR(Type, Dim) \
    extern template class YTensor<Type, Dim>;

//=============================================================================
// 标准浮点类型 (1D - 5D)
//=============================================================================
EXTERN_YTENSOR(float, 1)
EXTERN_YTENSOR(float, 2)
EXTERN_YTENSOR(float, 3)
EXTERN_YTENSOR(float, 4)
EXTERN_YTENSOR(float, 5)

EXTERN_YTENSOR(double, 1)
EXTERN_YTENSOR(double, 2)
EXTERN_YTENSOR(double, 3)
EXTERN_YTENSOR(double, 4)
EXTERN_YTENSOR(double, 5)

//=============================================================================
// 标准有符号整数类型 (1D - 5D)
//=============================================================================
EXTERN_YTENSOR(int8_t, 1)
EXTERN_YTENSOR(int8_t, 2)
EXTERN_YTENSOR(int8_t, 3)
EXTERN_YTENSOR(int8_t, 4)
EXTERN_YTENSOR(int8_t, 5)

EXTERN_YTENSOR(int16_t, 1)
EXTERN_YTENSOR(int16_t, 2)
EXTERN_YTENSOR(int16_t, 3)
EXTERN_YTENSOR(int16_t, 4)
EXTERN_YTENSOR(int16_t, 5)

EXTERN_YTENSOR(int32_t, 1)
EXTERN_YTENSOR(int32_t, 2)
EXTERN_YTENSOR(int32_t, 3)
EXTERN_YTENSOR(int32_t, 4)
EXTERN_YTENSOR(int32_t, 5)

EXTERN_YTENSOR(int64_t, 1)
EXTERN_YTENSOR(int64_t, 2)
EXTERN_YTENSOR(int64_t, 3)
EXTERN_YTENSOR(int64_t, 4)
EXTERN_YTENSOR(int64_t, 5)

//=============================================================================
// 标准无符号整数类型 (1D - 5D)
//=============================================================================
EXTERN_YTENSOR(uint8_t, 1)
EXTERN_YTENSOR(uint8_t, 2)
EXTERN_YTENSOR(uint8_t, 3)
EXTERN_YTENSOR(uint8_t, 4)
EXTERN_YTENSOR(uint8_t, 5)

EXTERN_YTENSOR(uint16_t, 1)
EXTERN_YTENSOR(uint16_t, 2)
EXTERN_YTENSOR(uint16_t, 3)
EXTERN_YTENSOR(uint16_t, 4)
EXTERN_YTENSOR(uint16_t, 5)

EXTERN_YTENSOR(uint32_t, 1)
EXTERN_YTENSOR(uint32_t, 2)
EXTERN_YTENSOR(uint32_t, 3)
EXTERN_YTENSOR(uint32_t, 4)
EXTERN_YTENSOR(uint32_t, 5)

EXTERN_YTENSOR(uint64_t, 1)
EXTERN_YTENSOR(uint64_t, 2)
EXTERN_YTENSOR(uint64_t, 3)
EXTERN_YTENSOR(uint64_t, 4)
EXTERN_YTENSOR(uint64_t, 5)

//=============================================================================
// 扩展浮点类型 (1D - 4D)
//=============================================================================
EXTERN_YTENSOR(yt::bfloat16, 1)
EXTERN_YTENSOR(yt::bfloat16, 2)
EXTERN_YTENSOR(yt::bfloat16, 3)
EXTERN_YTENSOR(yt::bfloat16, 4)

EXTERN_YTENSOR(yt::float16, 1)
EXTERN_YTENSOR(yt::float16, 2)
EXTERN_YTENSOR(yt::float16, 3)
EXTERN_YTENSOR(yt::float16, 4)

EXTERN_YTENSOR(yt::float8_e5m2, 1)
EXTERN_YTENSOR(yt::float8_e5m2, 2)
EXTERN_YTENSOR(yt::float8_e5m2, 3)
EXTERN_YTENSOR(yt::float8_e5m2, 4)

EXTERN_YTENSOR(yt::float8_e4m3, 1)
EXTERN_YTENSOR(yt::float8_e4m3, 2)
EXTERN_YTENSOR(yt::float8_e4m3, 3)
EXTERN_YTENSOR(yt::float8_e4m3, 4)

#undef EXTERN_YTENSOR

#endif // YT_ENABLE_YTENSOR_CLASS_EXTERN

//=============================================================================
// YTensorBase 模板方法 extern 声明
//=============================================================================
#define EXTERN_YTENSOR_BASE_METHOD(T)                               \
    extern template T& YTensorBase::at<T>(const std::vector<int>&);             \
    extern template const T& YTensorBase::at<T>(const std::vector<int>&) const; \
    extern template T& YTensorBase::atData<T>(int);                             \
    extern template const T& YTensorBase::atData<T>(int) const;                 \
    extern template T* YTensorBase::data<T>();                                  \
    extern template const T* YTensorBase::data<T>() const;

// 标准类型
EXTERN_YTENSOR_BASE_METHOD(float)
EXTERN_YTENSOR_BASE_METHOD(double)
EXTERN_YTENSOR_BASE_METHOD(int8_t)
EXTERN_YTENSOR_BASE_METHOD(int16_t)
EXTERN_YTENSOR_BASE_METHOD(int32_t)
EXTERN_YTENSOR_BASE_METHOD(int64_t)
EXTERN_YTENSOR_BASE_METHOD(uint8_t)
EXTERN_YTENSOR_BASE_METHOD(uint16_t)
EXTERN_YTENSOR_BASE_METHOD(uint32_t)
EXTERN_YTENSOR_BASE_METHOD(uint64_t)

// 扩展类型
EXTERN_YTENSOR_BASE_METHOD(yt::bfloat16)
EXTERN_YTENSOR_BASE_METHOD(yt::float16)
EXTERN_YTENSOR_BASE_METHOD(yt::float8_e5m2)
EXTERN_YTENSOR_BASE_METHOD(yt::float8_e4m3)

#undef EXTERN_YTENSOR_BASE_METHOD

} // namespace yt

#endif // YT_USE_LIB
