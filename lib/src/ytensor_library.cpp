#define YT_USE_LIB 1
#define YT_LIBRARY_IMPLEMENTATION
#include "../../ytensor.hpp"

// Library mode集中持有process-wide registries，避免consumer TU各自产生header-local副本。
namespace yt::type {
namespace internal {
std::unordered_map<std::string, yt::type::TypeRegItem>& getMutableTypeRegistry() {
    static std::unordered_map<std::string, yt::type::TypeRegItem> registry;
    return registry;
}

std::unordered_map<std::string, yt::type::YDTypeKernels>& getDTypeKernelRegistry() {
    static std::unordered_map<std::string, yt::type::YDTypeKernels> registry;
    return registry;
}

std::mutex& getDTypeKernelRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, yt::type::YCastKernel>& getCastKernelRegistry() {
    static std::unordered_map<std::string, yt::type::YCastKernel> registry;
    return registry;
}

std::mutex& getCastKernelRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}
}  // namespace internal

std::mutex& getTypeRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, yt::type::TypeRegItem> getTypeRegistry() {
    std::lock_guard<std::mutex> lock(getTypeRegistryMutex());
    return internal::getMutableTypeRegistry();
}

}  // namespace yt::type

namespace yt {

#define INSTANTIATE_YTENSOR_BASE_METHOD(T)                               \
    template T& YTensorBase::at<T>(const std::vector<int>&);             \
    template const T& YTensorBase::at<T>(const std::vector<int>&) const; \
    template T& YTensorBase::atData<T>(int);                             \
    template const T& YTensorBase::atData<T>(int) const;                 \
    template T* YTensorBase::data<T>();                                  \
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
INSTANTIATE_YTENSOR_BASE_METHOD(yt::float8_e8m0)

#undef INSTANTIATE_YTENSOR_BASE_METHOD

}  // namespace yt
