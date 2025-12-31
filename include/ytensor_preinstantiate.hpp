#pragma once
/***************
* @file: ytensor_preinstantiate.hpp
* @brief: YTensorBase模板显式实例化（预创建常用类型模板）
* @author: SnifferCaptain
* @date: 2025-12-05
* @note: 通过显式实例化减少编译时间，并确保常用类型的模板在编译时就被创建。
*        使用 YT_PREINSTANTIATE_TEMPLATES 宏控制是否启用。
***************/

#include "ytensor_infos.hpp"
#include "ytensor_base.hpp"

#if YT_PREINSTANTIATE_TEMPLATES

namespace yt {
// 为单个类型生成所有YTensorBase的at、atData、data方法的显式实例化
#define YT_EXPLICIT_INSTANTIATE_TYPE(T)                                          \
    template T& YTensorBase::at<T>(const std::vector<int>&);                     \
    template const T& YTensorBase::at<T>(const std::vector<int>&) const;         \
    template T& YTensorBase::atData<T>(int);                                     \
    template const T& YTensorBase::atData<T>(int) const;                         \
    template T* YTensorBase::data<T>();                                          \
    template const T* YTensorBase::data<T>() const;

YT_EXPLICIT_INSTANTIATE_TYPE(float)
YT_EXPLICIT_INSTANTIATE_TYPE(double)
YT_EXPLICIT_INSTANTIATE_TYPE(int8_t)
YT_EXPLICIT_INSTANTIATE_TYPE(int16_t)
YT_EXPLICIT_INSTANTIATE_TYPE(int32_t)
YT_EXPLICIT_INSTANTIATE_TYPE(int64_t)
YT_EXPLICIT_INSTANTIATE_TYPE(uint8_t)
YT_EXPLICIT_INSTANTIATE_TYPE(uint16_t)
YT_EXPLICIT_INSTANTIATE_TYPE(uint32_t)
YT_EXPLICIT_INSTANTIATE_TYPE(uint64_t)

#undef YT_EXPLICIT_INSTANTIATE_TYPE

} // namespace yt

#endif // YT_PREINSTANTIATE_TEMPLATES
