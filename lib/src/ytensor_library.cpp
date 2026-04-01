#define YT_USE_LIB 1
#define YT_LIBRARY_IMPLEMENTATION
#include "../../ytensor.hpp"

namespace yt::infos {
std::unordered_map<std::string, yt::infos::TypeRegItem>& getTypeRegistry() {
    static std::unordered_map<std::string, yt::infos::TypeRegItem> registry;
    return registry;
}
}  // namespace yt::infos

namespace yt {

#define INSTANTIATE_YTENSOR_BASE_METHOD(T) \
    template T& YTensorBase::at<T>(const std::vector<int>&); \
    template const T& YTensorBase::at<T>(const std::vector<int>&) const; \
    template T& YTensorBase::atData<T>(int); \
    template const T& YTensorBase::atData<T>(int) const; \
    template T* YTensorBase::data<T>(); \
    template const T* YTensorBase::data<T>() const;

INSTANTIATE_YTENSOR_BASE_METHOD(float)
INSTANTIATE_YTENSOR_BASE_METHOD(double)
INSTANTIATE_YTENSOR_BASE_METHOD(int8_t)
INSTANTIATE_YTENSOR_BASE_METHOD(int16_t)
INSTANTIATE_YTENSOR_BASE_METHOD(int32_t)
INSTANTIATE_YTENSOR_BASE_METHOD(int64_t)
INSTANTIATE_YTENSOR_BASE_METHOD(uint8_t)
INSTANTIATE_YTENSOR_BASE_METHOD(uint16_t)
INSTANTIATE_YTENSOR_BASE_METHOD(uint32_t)
INSTANTIATE_YTENSOR_BASE_METHOD(uint64_t)
INSTANTIATE_YTENSOR_BASE_METHOD(yt::bfloat16)
INSTANTIATE_YTENSOR_BASE_METHOD(yt::float16)
INSTANTIATE_YTENSOR_BASE_METHOD(yt::float8_e5m2)
INSTANTIATE_YTENSOR_BASE_METHOD(yt::float8_e4m3)

#undef INSTANTIATE_YTENSOR_BASE_METHOD

}  // namespace yt
