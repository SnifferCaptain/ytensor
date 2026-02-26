#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

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
    void allocate(size_t bytes) override {
        if (bytes == 0) bytes = 1;
        if (_bytes == bytes && _buffer) return;
        _buffer = std::shared_ptr<char[]>(new char[bytes]);
        _bytes = bytes;
    }

    void release() override {
        _buffer.reset();
        _bytes = 0;
    }

    void syncFromHost(const void* hostPtr, size_t bytes) override {
        if (bytes == 0) return;
        allocate(bytes);
        if (!hostPtr) {
            throw std::invalid_argument("[KomputeMemory::syncFromHost] hostPtr is null");
        }
        std::memcpy(_buffer.get(), hostPtr, bytes);
    }

    void syncToHost(void* hostPtr, size_t bytes) const override {
        if (bytes == 0) return;
        if (!hostPtr) {
            throw std::invalid_argument("[KomputeMemory::syncToHost] hostPtr is null");
        }
        if (!_buffer || bytes > _bytes) {
            throw std::runtime_error("[KomputeMemory::syncToHost] device buffer is invalid");
        }
        std::memcpy(hostPtr, _buffer.get(), bytes);
    }

    size_t bytes() const override { return _bytes; }
    std::string backend() const override { return "kompute"; }

private:
    std::shared_ptr<char[]> _buffer;
    size_t _bytes = 0;
};

inline std::string normalizeDevice(const std::string& device) {
    std::string d = device;
    std::transform(d.begin(), d.end(), d.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (d.empty()) d = "cpu";
    if (d == "gpu" || d == "kp") d = "kompute";
    return d;
}

} // namespace yt
