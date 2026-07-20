#pragma once
/***************
* @file: ytensor_layout.hpp
* @brief: 张量layout元数据的运行时封装。
***************/

#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ytensor_memory.hpp"

namespace yt {

/// @brief 张量layout类型。用于运行时layout分发和YMeta模板特化。
enum class YLayoutType : uint8_t {
    Strided,
    Nested,
};

/// @brief layout元数据模板。每个layout通过特化绑定自己的metadata结构。
template <YLayoutType Type>
struct YMeta;

/// @brief 经典strided layout元数据，对应storage offset、shape和stride。
template <>
struct YMeta<YLayoutType::Strided> {
    int offset = 0;
    std::vector<int> shape;
    std::vector<int> stride;
};

/// @brief Nested/Jagged layout元数据，表示为offsets加strided values。
template <>
struct YMeta<YLayoutType::Nested> {
    int nested_axis = 1;
    YMemory offsets;
    YMeta<YLayoutType::Strided> values;
    std::vector<int> shape;
};

/// @brief 安全的layout tagged union，只管理metadata生命周期。
/// @note 拷贝赋值使用copy-then-move惯用法保证异常安全（构造临时副本，然后move赋值）。
///       被move的YLayout保持有效默认状态（Strided，offset=0，shape/stride为空）。
class YLayout {
public:
    /// @brief 构造默认 Strided metadata。
    YLayout();
    /// @brief 复制 active metadata；其中的 YMemory 成员保持共享 storage 语义。
    YLayout(const YLayout& other);
    /// @brief 以强异常保证复制 active metadata。
    YLayout& operator=(const YLayout& other);
    /// @brief 移动 active metadata，并将源对象重置为有效的默认 Strided 状态。
    YLayout(YLayout&& other) noexcept;
    /// @brief 移动赋值 active metadata，并将源对象重置为有效的默认 Strided 状态。
    YLayout& operator=(YLayout&& other) noexcept;
    /// @brief 析构当前 active metadata。
    ~YLayout() noexcept;

    /// @brief 返回当前激活的layout类型。
    YLayoutType type() const;

    /// @brief 按layout类型访问metadata，类型不匹配时抛出异常。
    template <YLayoutType Type>
    YMeta<Type>& as();

    /// @brief 按layout类型访问只读metadata，类型不匹配时抛出异常。
    template <YLayoutType Type>
    const YMeta<Type>& as() const;

    /// @brief 切换当前layout并默认构造对应metadata。
    /// @note 先构造临时 replacement，成功后才销毁 active metadata 并 placement-new 目标类型；
    ///       tag 在新对象构造完成后更新，保证构造失败时原 metadata 和 tag 仍一致。
    template <YLayoutType Type>
    YMeta<Type>& emplace();

private:
    YLayoutType _type = YLayoutType::Strided;

    union Storage {
        typename std::aligned_storage<
            sizeof(YMeta<YLayoutType::Strided>),
            std::alignment_of<YMeta<YLayoutType::Strided>>::value
        >::type strided;

        typename std::aligned_storage<
            sizeof(YMeta<YLayoutType::Nested>),
            std::alignment_of<YMeta<YLayoutType::Nested>>::value
        >::type nested;

        Storage() {}
        ~Storage() {}
    } _storage;

    void destroy() noexcept;
    void resetToDefaultStrided();
    void copyFrom(const YLayout& other);
    void moveFrom(YLayout&& other) noexcept;
};

}  // namespace yt
