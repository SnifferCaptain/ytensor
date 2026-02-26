#pragma once

#include "../ytensor_infos.hpp"
#include <memory>
#include <string>
#include <vector>

namespace yt {

class GPUMemory {
public:
    virtual ~GPUMemory() = default;
    virtual void allocate(size_t bytes) = 0;
    virtual void release() = 0;
    virtual void syncFromHost(const void* hostPtr, size_t bytes) = 0;
    virtual void syncToHost(void* hostPtr, size_t bytes) const = 0;
    virtual size_t bytes() const = 0;
    virtual std::string backend() const = 0;
};

class KomputeMemory final : public GPUMemory {
public:
    void allocate(size_t bytes) override;
    void release() override;
    void syncFromHost(const void* hostPtr, size_t bytes) override;
    void syncToHost(void* hostPtr, size_t bytes) const override;
    size_t bytes() const override;
    std::string backend() const override;

private:
    std::vector<uint32_t> _hostWords;
#if YT_USE_KOMPUTE
    std::shared_ptr<kp::Manager> _manager;
    std::shared_ptr<kp::TensorT<uint32_t>> _tensor;
#endif
    size_t _bytes = 0;
};

std::string normalizeDevice(const std::string& device);

} // namespace yt

#include "../../src/kompute/kompute_memory.inl"
