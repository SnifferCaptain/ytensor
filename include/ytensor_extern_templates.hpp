#pragma once
/**********************************************************************
 * @file: ytensor_extern_templates.hpp
 * @brief: YTensor模板extern声明
 * @author: SnifferCaptain
 * @date: 2026-01-29
 * 
 * 说明：
 * 当定义了 YT_USE_LIB=1 时，此文件为预编译的模板类型提供
 * extern template 声明，避免重复实例化，从而加速编译。
 * 
 * 使用方式：
 * 1. 链接预编译的 libytensor 库
 * 2. 在包含 ytensor.hpp 之前定义 YT_USE_LIB=1：
 *    #define YT_USE_LIB 1
 *    #include "ytensor.hpp"
 * 
 * 或者通过CMake：
 *    target_compile_definitions(your_target PRIVATE YT_USE_LIB=1)
 **********************************************************************/

#if YT_USE_LIB

namespace yt {

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
EXTERN_YTENSOR_BASE_METHOD(yt::float8_e8m0)

#undef EXTERN_YTENSOR_BASE_METHOD

} // namespace yt

#endif // YT_USE_LIB
