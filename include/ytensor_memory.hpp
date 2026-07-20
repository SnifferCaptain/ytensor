#pragma once
/***************
* @file: ytensor_memory.hpp
* @brief: YTensor 的底层内存封装。只负责字节存储和设备信息，不处理形状、步长或类型语义。
***************/
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace yt{

/// @brief 底层存储的抽象基类。
/// @note 该类只描述一段线性字节内存及其所在设备，不负责张量的 shape、stride、offset 或 dtype。
class YStorageBase {
public:
    virtual ~YStorageBase() {}

    /// @brief 获取底层存储占用的字节数。
    virtual size_t nbytes() const = 0;

    /// @brief 获取底层存储所在设备，例如 "cpu"。
    virtual const std::string& device() const = 0;

    /// @brief 获取原始字节指针。
    /// @note 返回的是 storage 起点，不包含张量当前 layout 视图偏移。
    virtual char* rawData() = 0;
    virtual const char* rawData() const = 0;

    /// @brief 深拷贝当前存储。
    /// @return 返回一个拥有独立内存的新 storage。
    /// @throws std::runtime_error object-backed storage 禁止原始字节 clone 时抛出。
    virtual std::shared_ptr<YStorageBase> clone() const = 0;

    /// @brief 将当前存储复制到指定设备。
    /// @param device 目标设备名称。
    /// @return 返回目标设备上的新 storage。
    virtual std::shared_ptr<YStorageBase> to(const std::string& device) const = 0;
};

/// @brief CPU 线性字节存储。
class YCpuStorage : public YStorageBase {
public:
    /// @brief 分配指定字节数的 CPU 存储。
    explicit YCpuStorage(size_t nbytes);

    /// @brief 使用已有 shared_ptr 管理的内存构造 CPU 存储。
    /// @param byteCopyable 是否允许按原始字节 clone；对象生命周期由 tensor owner 管理时必须为 false。
    YCpuStorage(const std::shared_ptr<char>& data, size_t nbytes, bool byteCopyable = true);
    YCpuStorage(const std::shared_ptr<char[]>& data, size_t nbytes, bool byteCopyable = true);

    /// @brief 使用裸指针和自定义释放器构造 CPU 存储。
    /// @param byteCopyable 是否允许按原始字节 clone；对象生命周期由 tensor owner 管理时必须为 false。
    YCpuStorage(char* ptr, size_t nbytes, std::function<void(char*)> deleter, bool byteCopyable = true);

    size_t nbytes() const override;
    const std::string& device() const override;

    char* rawData() override;
    const char* rawData() const override;

    std::shared_ptr<YStorageBase> clone() const override;
    std::shared_ptr<YStorageBase> to(const std::string& device) const override;

private:
    std::shared_ptr<char> _data;     // CPU 字节内存的共享所有权。
    size_t _nbytes = 0;              // 存储区总字节数。
    std::string _device = "cpu";     // 当前 storage 所在设备。
    bool _byteCopyable = true;        // false表示storage中包含需要生命周期管理的C++对象。
};

/// @brief YTensor 使用的内存句柄。
/// @note YMemory 是轻量共享句柄，只封装 storage/dataptr/device，不处理 dtype、对象构造、shape、stride 或 offset。
class YMemory {
public:
    YMemory() = default;

    /// @brief 构造空内存句柄。
    YMemory(std::nullptr_t);

    /// @brief 从已有 storage 构造内存句柄。
    explicit YMemory(const std::shared_ptr<YStorageBase>& storage);

    /// @brief 从已有 shared_ptr 字节内存构造内存句柄。
    /// @param byteCopyable 是否允许按原始字节 clone；placement-constructed 对象 storage 必须为 false。
    YMemory(
        const std::shared_ptr<char>& data, size_t nbytes, const std::string& device = "cpu",
        bool byteCopyable = true
    );
    YMemory(
        const std::shared_ptr<char[]>& data, size_t nbytes = 0, const std::string& device = "cpu",
        bool byteCopyable = true
    );

    /// @brief 从裸指针构造内存句柄，并使用自定义释放器管理生命周期。
    /// @param byteCopyable 是否允许按原始字节 clone；placement-constructed 对象 storage 必须为 false。
    YMemory(
        char* ptr, size_t nbytes, const std::string& device, std::function<void(char*)> deleter,
        bool byteCopyable = true
    );

    /// @brief 置空当前内存句柄。
    YMemory& operator=(std::nullptr_t);

    /// @brief 兼容 shared_ptr 赋值的便捷接口。
    /// @note 该重载无法推断字节数，nbytes() 会返回 0。新代码应优先使用构造函数显式传入 nbytes。
    YMemory& operator=(const std::shared_ptr<char>& data);
    YMemory& operator=(const std::shared_ptr<char[]>& data);

    /// @brief 判断当前句柄是否为空。
    bool empty() const;

    /// @brief 判断当前句柄是否持有有效内存。
    explicit operator bool() const;

    /// @brief 获取底层存储的字节数。
    size_t nbytes() const;

    /// @brief 获取底层存储所在设备。
    const std::string& device() const;

    /// @brief 获取底层 storage 起点的原始字节指针。
    /// @note 不包含张量当前 layout 视图偏移；需要按张量元素访问时应通过 YTensor/YTensorBase 接口处理。
    char* rawData();
    const char* rawData() const;

    /// @brief 兼容旧代码的 rawData() 别名。
    /// @note 新代码应优先使用 rawData()，该接口只用于兼容原有 get() 调用点。
    char* get();
    const char* get() const;

    /// @brief 深拷贝底层存储，返回新的内存句柄。
    /// @throws std::runtime_error object-backed storage 必须通过 tensor 生命周期 owner clone 时抛出。
    YMemory clone() const;

    /// @brief 将底层存储复制到指定设备，返回新的内存句柄。
    YMemory to(const std::string& device) const;

    /// @brief 获取底层 storage 共享指针。
    std::shared_ptr<YStorageBase> storage() const;

    friend bool operator==(const YMemory& memory, std::nullptr_t);
    friend bool operator==(std::nullptr_t, const YMemory& memory);
    friend bool operator!=(const YMemory& memory, std::nullptr_t);
    friend bool operator!=(std::nullptr_t, const YMemory& memory);

private:
    std::shared_ptr<YStorageBase> _storage; // 底层线性存储的共享所有权。
};

}  // namespace yt
