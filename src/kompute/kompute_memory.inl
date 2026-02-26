#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>

namespace yt {

inline void KomputeMemory::allocate(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (_bytes == bytes && !_hostWords.empty()) return;
    _hostWords.assign((bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0u);
#if YT_USE_KOMPUTE
    if (!_manager) {
        _manager = std::make_shared<kp::Manager>();
    }
    _tensor = _manager->tensorT<uint32_t>(_hostWords);
#endif
    _bytes = bytes;
}

inline void KomputeMemory::release() {
    _hostWords.clear();
#if YT_USE_KOMPUTE
    _tensor.reset();
    _manager.reset();
#endif
    _bytes = 0;
}

inline void KomputeMemory::syncFromHost(const void* hostPtr, size_t bytes) {
    if (bytes == 0) return;
    allocate(bytes);
    if (!hostPtr) {
        throw std::invalid_argument("[KomputeMemory::syncFromHost] hostPtr is null");
    }
    std::memcpy(_hostWords.data(), hostPtr, bytes);
#if YT_USE_KOMPUTE
    std::memcpy(_tensor->vector().data(), _hostWords.data(), _hostWords.size() * sizeof(uint32_t));
    auto seq = _manager->sequence();
    std::vector<std::shared_ptr<kp::Memory>> params = {_tensor};
    seq->record<kp::OpSyncDevice>(params)->eval();
#endif
}

inline void KomputeMemory::syncToHost(void* hostPtr, size_t bytes) const {
    if (bytes == 0) return;
    if (!hostPtr) {
        throw std::invalid_argument("[KomputeMemory::syncToHost] hostPtr is null");
    }
    if (_hostWords.empty() || bytes > _bytes) {
        throw std::runtime_error("[KomputeMemory::syncToHost] device buffer is invalid");
    }
#if YT_USE_KOMPUTE
    auto seq = _manager->sequence();
    std::vector<std::shared_ptr<kp::Memory>> params = {_tensor};
    seq->record<kp::OpSyncLocal>(params)->eval();
    std::memcpy(hostPtr, _tensor->vector().data(), bytes);
#else
    std::memcpy(hostPtr, _hostWords.data(), bytes);
#endif
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
