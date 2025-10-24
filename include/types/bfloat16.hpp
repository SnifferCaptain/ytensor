#pragma once
/***************
* @file: bfloat16.hpp
* @brief: bfloat16 数据类型定义。
* @author: https://github.com/openvinotoolkit/openvino/blob/master/src/core/include/openvino/core/type/bfloat16.hpp
* @date: 2025-10-24
* @note 复制自openvino的bf16实现，做了少量修改以适配ytensor。
***************/

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <sys/types.h>
#include <vector>
#include "../ytensor_infos.hpp"

namespace yt {
class bfloat16 {
public:
    bfloat16() = default;
    bfloat16(float value): m_value{round(value)} {}

    template <typename I>
    explicit bfloat16(I value) : m_value{bfloat16{static_cast<float>(value)}.m_value} {}

    std::string to_string() const;
    size_t size() const;
    template <typename T>
    bool operator==(const T& other) const;
    template <typename T>
    bool operator!=(const T& other) const {
        return !(*this == other);
    }
    template <typename T>
    bool operator<(const T& other) const;
    template <typename T>
    bool operator<=(const T& other) const;
    template <typename T>
    bool operator>(const T& other) const;
    template <typename T>
    bool operator>=(const T& other) const;
    template <typename T>
    bfloat16 operator+(const T& other) const;
    template <typename T>
    bfloat16 operator+=(const T& other);
    template <typename T>
    bfloat16 operator-(const T& other) const;
    template <typename T>
    bfloat16 operator-=(const T& other);
    template <typename T>
    bfloat16 operator*(const T& other) const;
    template <typename T>
    bfloat16 operator*=(const T& other);
    template <typename T>
    bfloat16 operator/(const T& other) const;
    template <typename T>
    bfloat16 operator/=(const T& other);
    operator float() const;

    static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
    static std::vector<bfloat16> from_float_vector(const std::vector<float>&);
    static constexpr bfloat16 from_bits(uint16_t bits) {
        return bfloat16(bits, true);
    }
    uint16_t to_bits() const;
    friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj) {
        out << static_cast<float>(obj);
        return out;
    }

    static uint16_t round_to_nearest_even(float x) {
        return static_cast<uint16_t>((cu32(x) + ((cu32(x) & 0x00010000) >> 1)) >> 16);
    }

    static uint16_t round_to_nearest(float x) {
        return static_cast<uint16_t>((cu32(x) + 0x8000) >> 16);
    }

    static uint16_t truncate(float x) {
        return static_cast<uint16_t>((cu32(x)) >> 16);
    }

    static uint16_t round(float x){
        if constexpr(yt::infos::roundMode == yt::infos::RoundMode::nearestEven){
            return round_to_nearest_even(x);
        } else if constexpr(yt::infos::roundMode == yt::infos::RoundMode::nearest){
            return round_to_nearest(x);
        } else{
            return truncate(x);
        }
    }

private:
    constexpr bfloat16(uint16_t x, bool) : m_value{x} {}
    union F32 {
        F32(float val) : f{val} {}
        F32(uint32_t val) : i{val} {}
        float f;
        uint32_t i;
    };

    static inline uint32_t cu32(float x) {
        return F32(x).i;
    }

    uint16_t m_value;
};

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4756)
#endif
template <typename T>
bool bfloat16::operator==(const T& other) const {
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif
}

template <typename T>
bool bfloat16::operator<(const T& other) const {
    return (static_cast<float>(*this) < static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator<=(const T& other) const {
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator>(const T& other) const {
    return (static_cast<float>(*this) > static_cast<float>(other));
}

template <typename T>
bool bfloat16::operator>=(const T& other) const {
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

template <typename T>
bfloat16 bfloat16::operator+(const T& other) const {
    return {static_cast<float>(*this) + static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator+=(const T& other) {
    return *this = *this + other;
}

template <typename T>
bfloat16 bfloat16::operator-(const T& other) const {
    return {static_cast<float>(*this) - static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator-=(const T& other) {
    return *this = *this - other;
}

template <typename T>
bfloat16 bfloat16::operator*(const T& other) const {
    return {static_cast<float>(*this) * static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator*=(const T& other) {
    return *this = *this * other;
}

template <typename T>
bfloat16 bfloat16::operator/(const T& other) const {
    return {static_cast<float>(*this) / static_cast<float>(other)};
}

template <typename T>
bfloat16 bfloat16::operator/=(const T& other) {
    return *this = *this / other;
}
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif
}  // namespace yt

namespace std {
template <>
class numeric_limits<yt::bfloat16> {
public:
    static constexpr bool is_specialized = true;
    static constexpr yt::bfloat16 min() noexcept {
        return yt::bfloat16::from_bits(0x007F);
    }
    static constexpr yt::bfloat16 max() noexcept {
        return yt::bfloat16::from_bits(0x7F7F);
    }
    static constexpr yt::bfloat16 lowest() noexcept {
        return yt::bfloat16::from_bits(0xFF7F);
    }
    static constexpr int digits = 7;
    static constexpr int digits10 = 2;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr yt::bfloat16 epsilon() noexcept {
        return yt::bfloat16::from_bits(0x3C00);
    }
    static constexpr yt::bfloat16 round_error() noexcept {
        return yt::bfloat16::from_bits(0x3F00);
    }
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr yt::bfloat16 infinity() noexcept {
        return yt::bfloat16::from_bits(0x7F80);
    }
    static constexpr yt::bfloat16 quiet_NaN() noexcept {
        return yt::bfloat16::from_bits(0x7FC0);
    }
    static constexpr yt::bfloat16 signaling_NaN() noexcept {
        return yt::bfloat16::from_bits(0x7FC0);
    }
    static constexpr yt::bfloat16 denorm_min() noexcept {
        return yt::bfloat16::from_bits(0);
    }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = 
        yt::infos::roundMode == yt::infos::RoundMode::truncate ?
            round_toward_zero
        /*else*/:
            round_to_nearest;
};
}  // namespace std



static_assert(sizeof(yt::bfloat16) == 2, "class bfloat16 must be exactly 2 bytes");

inline bool float_isnan(const float& x) {
    return std::isnan(x);
}

inline std::vector<float> yt::bfloat16::to_float_vector(const std::vector<yt::bfloat16>& v_bf16) {
    std::vector<float> v_f32(v_bf16.begin(), v_bf16.end());
    return v_f32;
}

inline std::vector<yt::bfloat16> yt::bfloat16::from_float_vector(const std::vector<float>& v_f32) {
    std::vector<yt::bfloat16> v_bf16;
    v_bf16.reserve(v_f32.size());
    for (float a : v_f32) {
        v_bf16.push_back(static_cast<yt::bfloat16>(a));
    }
    return v_bf16;
}

inline std::string yt::bfloat16::to_string() const {
    return std::to_string(static_cast<float>(*this));
}

inline size_t yt::bfloat16::size() const {
    return sizeof(m_value);
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuninitialized"
#endif

inline yt::bfloat16::operator float() const {
    uint32_t tmp = 0;
    uint32_t* ptmp = &tmp;
    *ptmp = (static_cast<uint32_t>(m_value) << 16);
    const float* f = reinterpret_cast<const float*>(ptmp);
    return *f;
}

#if defined __GNUC__ && __GNUC__ == 11
#    pragma GCC diagnostic pop
#endif

inline uint16_t yt::bfloat16::to_bits() const {
    return m_value;
}