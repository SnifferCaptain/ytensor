#pragma once

#include <cstring>
#include <stdexcept>

namespace yt{

YT_IMPL_INLINE YCpuStorage::YCpuStorage(size_t nbytes):
    _data(nbytes == 0 ? nullptr : std::shared_ptr<char>(new char[nbytes], [](char* p) { delete[] p; })),
    _nbytes(nbytes) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(const std::shared_ptr<char>& data, size_t nbytes):
    _data(data),
    _nbytes(nbytes) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(const std::shared_ptr<char[]>& data, size_t nbytes):
    _data(data ? std::shared_ptr<char>(data, data.get()) : std::shared_ptr<char>()),
    _nbytes(nbytes) {}

YT_IMPL_INLINE YCpuStorage::YCpuStorage(char* ptr, size_t nbytes, std::function<void(char*)> deleter):
    _data(ptr, deleter),
    _nbytes(nbytes) {}

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

YT_IMPL_INLINE YMemory::YMemory(std::nullptr_t) {}

YT_IMPL_INLINE YMemory::YMemory(const std::shared_ptr<YStorageBase>& storage):
    _storage(storage) {}

YT_IMPL_INLINE YMemory::YMemory(const std::shared_ptr<char>& data, size_t nbytes, const std::string& device) {
    if (!data) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu shared_ptr storage is supported now");
    }
    _storage.reset(new YCpuStorage(data, nbytes));
}

YT_IMPL_INLINE YMemory::YMemory(const std::shared_ptr<char[]>& data, size_t nbytes, const std::string& device) {
    if (!data) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu shared_ptr storage is supported now");
    }
    _storage.reset(new YCpuStorage(data, nbytes));
}

YT_IMPL_INLINE YMemory::YMemory(char* ptr, size_t nbytes, const std::string& device, std::function<void(char*)> deleter) {
    if (!ptr) return;
    if (device != "cpu") {
        throw std::runtime_error("YMemory: only cpu raw pointer storage is supported now");
    }
    _storage.reset(new YCpuStorage(ptr, nbytes, deleter));
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

} // namespace yt
