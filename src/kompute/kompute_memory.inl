#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace yt {

inline void KomputeMemory::allocate(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (_bytes == bytes && _buffer) return;
    _buffer = std::shared_ptr<char[]>(new char[bytes]);
    _bytes = bytes;
}

inline void KomputeMemory::release() {
    _buffer.reset();
    _bytes = 0;
}

inline void KomputeMemory::syncFromHost(const void* hostPtr, size_t bytes) {
    if (bytes == 0) return;
    allocate(bytes);
    if (!hostPtr) {
        throw std::invalid_argument("[KomputeMemory::syncFromHost] hostPtr is null");
    }
    std::memcpy(_buffer.get(), hostPtr, bytes);
}

inline void KomputeMemory::syncToHost(void* hostPtr, size_t bytes) const {
    if (bytes == 0) return;
    if (!hostPtr) {
        throw std::invalid_argument("[KomputeMemory::syncToHost] hostPtr is null");
    }
    if (!_buffer || bytes > _bytes) {
        throw std::runtime_error("[KomputeMemory::syncToHost] device buffer is invalid");
    }
    std::memcpy(hostPtr, _buffer.get(), bytes);
}

inline size_t KomputeMemory::bytes() const { return _bytes; }
inline std::string KomputeMemory::backend() const { return "kompute"; }

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
