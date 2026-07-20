#pragma once
/***************
 * file: ytensor_memory.inl
 * purpose: CPU storage和共享YMemory handle实现。
 ***************/

#include <cstring>
#include <stdexcept>

namespace yt{

// ==================== CPU storage ownership ====================

YT_IMPL_INLINE YCpuStorage::YCpuStorage(size_t nbytes):
    _data(nbytes == 0 ? nullptr : std::shared_ptr<char>(new char[nbytes], [](char* p) { delete[] p; })),
    _nbytes(nbytes) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(const std::shared_ptr<char>& data, size_t nbytes, bool byteCopyable):
    _data(data),
    _nbytes(nbytes),
    _byteCopyable(byteCopyable) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(const std::shared_ptr<char[]>& data, size_t nbytes, bool byteCopyable):
    // aliasing shared_ptr暴露char*接口，同时保留shared_ptr<char[]>原控制块和delete[]语义。
    _data(data ? std::shared_ptr<char>(data, data.get()) : std::shared_ptr<char>()),
    _nbytes(nbytes),
    _byteCopyable(byteCopyable) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(
    char* ptr, size_t nbytes, std::function<void(char*)> deleter, bool byteCopyable
):
    _data(ptr, deleter),
    _nbytes(nbytes),
    _byteCopyable(byteCopyable) {}

YT_IMPL_INLINE size_t YCpuStorage::nbytes() const {
    return _nbytes;
}

YT_IMPL_INLINE const std::string& YCpuStorage::device() const {
    return _device;
}

YT_IMPL_INLINE char* YCpuStorage::rawData() {
    return _data.get();
}

YT_IMPL_INLINE const char* YCpuStorage::rawData() const {
    return _data.get();
}

YT_IMPL_INLINE std::shared_ptr<YStorageBase> YCpuStorage::clone() const {
    // memcpy只对明确byte-copyable的storage合法；C++对象数组必须由tensor dtype owner逐对象clone。
    if (!_byteCopyable) {
        throw std::runtime_error("YCpuStorage::clone: object-backed storage requires tensor lifecycle clone");
    }
    std::shared_ptr<YCpuStorage> result(new YCpuStorage(_nbytes));
    if (_nbytes > 0 && _data) {
        std::memcpy(result->rawData(), rawData(), _nbytes);
    }
    return result;
}

YT_IMPL_INLINE std::shared_ptr<YStorageBase> YCpuStorage::to(const std::string& device) const {
    if (device == "cpu") {
        return clone();
    }
    throw std::runtime_error("YCpuStorage::to: unsupported target device: " + device);
}

// ==================== shared storage handle ====================

YT_IMPL_INLINE YMemory::YMemory(std::nullptr_t) {}

YT_IMPL_INLINE YMemory::YMemory(const std::shared_ptr<YStorageBase>& storage):
    _storage(storage) {}

YT_IMPL_INLINE YMemory::YMemory(
    const std::shared_ptr<char>& data, size_t nbytes, const std::string& device, bool byteCopyable
) {
    if (!data) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu shared_ptr storage is supported now");
    }
    _storage.reset(new YCpuStorage(data, nbytes, byteCopyable));
}

YT_IMPL_INLINE YMemory::YMemory(
    const std::shared_ptr<char[]>& data, size_t nbytes, const std::string& device, bool byteCopyable
) {
    if (!data) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu shared_ptr storage is supported now");
    }
    _storage.reset(new YCpuStorage(data, nbytes, byteCopyable));
}

YT_IMPL_INLINE YMemory::YMemory(
    char* ptr, size_t nbytes, const std::string& device, std::function<void(char*)> deleter,
    bool byteCopyable
) {
    if (!ptr) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu raw pointer storage is supported now");
    }
    _storage.reset(new YCpuStorage(ptr, nbytes, deleter, byteCopyable));
}

YT_IMPL_INLINE YMemory& YMemory::operator=(std::nullptr_t) {
    _storage.reset();
    return *this;
}

YT_IMPL_INLINE YMemory& YMemory::operator=(const std::shared_ptr<char>& data) {
    _storage = data ? std::shared_ptr<YStorageBase>(new YCpuStorage(data, 0)) : std::shared_ptr<YStorageBase>();
    return *this;
}

YT_IMPL_INLINE YMemory& YMemory::operator=(const std::shared_ptr<char[]>& data) {
    _storage = data ? std::shared_ptr<YStorageBase>(new YCpuStorage(data, 0)) : std::shared_ptr<YStorageBase>();
    return *this;
}

YT_IMPL_INLINE bool YMemory::empty() const {
    return !_storage || !_storage->rawData();
}

YT_IMPL_INLINE YMemory::operator bool() const {
    return !empty();
}

YT_IMPL_INLINE size_t YMemory::nbytes() const {
    return _storage ? _storage->nbytes() : 0;
}

YT_IMPL_INLINE const std::string& YMemory::device() const {
    // 空handle仍报告cpu，保持默认tensor和尚未分配tensor的历史device合同。
    static const std::string cpu = "cpu";
    return _storage ? _storage->device() : cpu;
}

YT_IMPL_INLINE char* YMemory::rawData() {
    return _storage ? _storage->rawData() : nullptr;
}

YT_IMPL_INLINE const char* YMemory::rawData() const {
    return _storage ? _storage->rawData() : nullptr;
}

YT_IMPL_INLINE char* YMemory::get() {
    return rawData();
}

YT_IMPL_INLINE const char* YMemory::get() const {
    return rawData();
}

YT_IMPL_INLINE YMemory YMemory::clone() const {
    return _storage ? YMemory(_storage->clone()) : YMemory();
}

YT_IMPL_INLINE YMemory YMemory::to(const std::string& device) const {
    return _storage ? YMemory(_storage->to(device)) : YMemory();
}

YT_IMPL_INLINE std::shared_ptr<YStorageBase> YMemory::storage() const {
    return _storage;
}

YT_IMPL_INLINE bool operator==(const YMemory& memory, std::nullptr_t) {
    return memory.empty();
}

YT_IMPL_INLINE bool operator==(std::nullptr_t, const YMemory& memory) {
    return memory.empty();
}

YT_IMPL_INLINE bool operator!=(const YMemory& memory, std::nullptr_t) {
    return !memory.empty();
}

YT_IMPL_INLINE bool operator!=(std::nullptr_t, const YMemory& memory) {
    return !memory.empty();
}

}  // namespace yt
