#pragma once
/***************
* file: ytensor_layout.inl
* purpose: YLayout tagged union实现。
***************/

namespace yt {

// 所有private helper都操作未初始化或由_type准确标识的active union member。
// 调用方必须先destroy已有member，不能在active storage上直接copyFrom/moveFrom。

YT_IMPL_INLINE YLayout::YLayout() {
    resetToDefaultStrided();
}

YT_IMPL_INLINE YLayout::YLayout(const YLayout& other) {
    copyFrom(other);
}

YT_IMPL_INLINE YLayout& YLayout::operator=(const YLayout& other) {
    if (this != &other) {
        // copy-then-move：构造临时副本，若构造抛异常则*this不变；move赋值不抛异常
        YLayout replacement(other);
        *this = std::move(replacement);
    }
    return *this;
}

YT_IMPL_INLINE YLayout::YLayout(YLayout&& other) noexcept {
    moveFrom(std::move(other));
}

YT_IMPL_INLINE YLayout& YLayout::operator=(YLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        moveFrom(std::move(other));
    }
    return *this;
}

YT_IMPL_INLINE YLayout::~YLayout() noexcept {
    destroy();
}

YT_IMPL_INLINE YLayoutType YLayout::type() const {
    return _type;
}

YT_IMPL_INLINE void YLayout::destroy() noexcept {
    // _type始终标识当前唯一active member；析构后storage在下次placement-new前未初始化。
    switch (_type) {
        case YLayoutType::Strided:
            reinterpret_cast<YMeta<YLayoutType::Strided>*>(&_storage.strided)->~YMeta<YLayoutType::Strided>();
            break;
        case YLayoutType::Nested:
            reinterpret_cast<YMeta<YLayoutType::Nested>*>(&_storage.nested)->~YMeta<YLayoutType::Nested>();
            break;
    }
}

YT_IMPL_INLINE void YLayout::resetToDefaultStrided() {
    // 仅对未初始化storage调用，并建立可析构的默认moved-from状态。
    _type = YLayoutType::Strided;
    new (&_storage.strided) YMeta<YLayoutType::Strided>();
}

YT_IMPL_INLINE void YLayout::copyFrom(const YLayout& other) {
    // destination storage必须未初始化；复制构造成功后由复制的tag管理其生命周期。
    _type = other._type;
    switch (_type) {
        case YLayoutType::Strided:
            new (&_storage.strided) YMeta<YLayoutType::Strided>(
                *reinterpret_cast<const YMeta<YLayoutType::Strided>*>(&other._storage.strided));
            break;
        case YLayoutType::Nested:
            new (&_storage.nested) YMeta<YLayoutType::Nested>(
                *reinterpret_cast<const YMeta<YLayoutType::Nested>*>(&other._storage.nested));
            break;
        default:
            throw std::runtime_error("YLayout::copyFrom: layout not implemented");
    }
}

YT_IMPL_INLINE void YLayout::moveFrom(YLayout&& other) noexcept {
    // metadata成员的move constructor为noexcept，因而可先建立destination再重置source。
    _type = other._type;
    switch (_type) {
        case YLayoutType::Strided:
            new (&_storage.strided) YMeta<YLayoutType::Strided>(
                std::move(*reinterpret_cast<YMeta<YLayoutType::Strided>*>(&other._storage.strided)));
            break;
        case YLayoutType::Nested:
            new (&_storage.nested) YMeta<YLayoutType::Nested>(
                std::move(*reinterpret_cast<YMeta<YLayoutType::Nested>*>(&other._storage.nested)));
            break;
    }

    // 被move的other重置为默认Strided状态，保证其仍可析构和重新赋值
    other.destroy();
    other.resetToDefaultStrided();
}

template <>
YT_IMPL_INLINE YMeta<YLayoutType::Strided>& YLayout::as<YLayoutType::Strided>() {
    if (_type != YLayoutType::Strided) {
        throw std::runtime_error("YLayout::as: layout is not Strided");
    }
    return *reinterpret_cast<YMeta<YLayoutType::Strided>*>(&_storage.strided);
}

template <>
YT_IMPL_INLINE const YMeta<YLayoutType::Strided>& YLayout::as<YLayoutType::Strided>() const {
    if (_type != YLayoutType::Strided) {
        throw std::runtime_error("YLayout::as: layout is not Strided");
    }
    return *reinterpret_cast<const YMeta<YLayoutType::Strided>*>(&_storage.strided);
}

template <>
YT_IMPL_INLINE YMeta<YLayoutType::Nested>& YLayout::as<YLayoutType::Nested>() {
    if (_type != YLayoutType::Nested) {
        throw std::runtime_error("YLayout::as: layout is not Nested");
    }
    return *reinterpret_cast<YMeta<YLayoutType::Nested>*>(&_storage.nested);
}

template <>
YT_IMPL_INLINE const YMeta<YLayoutType::Nested>& YLayout::as<YLayoutType::Nested>() const {
    if (_type != YLayoutType::Nested) {
        throw std::runtime_error("YLayout::as: layout is not Nested");
    }
    return *reinterpret_cast<const YMeta<YLayoutType::Nested>*>(&_storage.nested);
}

template <>
YT_IMPL_INLINE YMeta<YLayoutType::Strided>& YLayout::emplace<YLayoutType::Strided>() {
    // replacement先成功构造，再销毁active member；metadata move为noexcept，tag不会与对象失配。
    YMeta<YLayoutType::Strided> replacement;
    destroy();
    new (&_storage.strided) YMeta<YLayoutType::Strided>(std::move(replacement));
    _type = YLayoutType::Strided;
    return *reinterpret_cast<YMeta<YLayoutType::Strided>*>(&_storage.strided);
}

template <>
YT_IMPL_INLINE YMeta<YLayoutType::Nested>& YLayout::emplace<YLayoutType::Nested>() {
    // replacement先成功构造，再销毁active member；metadata move为noexcept，tag不会与对象失配。
    YMeta<YLayoutType::Nested> replacement;
    destroy();
    new (&_storage.nested) YMeta<YLayoutType::Nested>(std::move(replacement));
    _type = YLayoutType::Nested;
    return *reinterpret_cast<YMeta<YLayoutType::Nested>*>(&_storage.nested);
}

}  // namespace yt
