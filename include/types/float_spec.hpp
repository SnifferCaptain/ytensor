#pragma once
/***************
* @file: float_spec.hpp
* @brief: 通用浮点数模板类型，支持任意指数位和尾数位配置。
* @author: SnifferCaptain
* @date: 2025.12.26
* @note 通用浮点数模板生成器，支持ieee754标准下的任意浮点数生成。
***************/

#include <cstdint>
#include <cstring>
#include <cmath>
#include <type_traits>
#include <limits>
#include <iostream>
#include <string>

namespace yt {

// ==================== 存储类型选择 ====================
// 根据总位数选择对齐的存储类型（2的n次方字节）
template <int TotalBits>
struct StorageTypeSelector {
    static_assert(TotalBits > 0 && TotalBits <= 64, "TotalBits must be between 1 and 64");
    using type = std::conditional_t<TotalBits <= 8, uint8_t,
                 std::conditional_t<TotalBits <= 16, uint16_t,
                 std::conditional_t<TotalBits <= 32, uint32_t,
                 uint64_t>>>;
};

// ==================== 计算类型选择 ====================
// 如果 exp_bits <= 8 && mantissa_bits <= 23，使用float；否则使用double
template <int ExpBits, int MantissaBits>
struct ComputeTypeSelector {
    using type = std::conditional_t<(ExpBits <= 8 && MantissaBits <= 23), float, double>;
};

// ==================== FloatSpec 模板类 ====================
/**
 * @brief 通用浮点数模板类
 * @tparam SignBits 符号位数（0或1）
 * @tparam ExpBits 指数位数
 * @tparam MantissaBits 尾数位数
 * 
 * 示例：
 * - float16:    FloatSpec<1, 5, 10>
 * - bfloat16:   FloatSpec<1, 8, 7>
 * - float32:    FloatSpec<1, 8, 23>
 * - TF32:       FloatSpec<1, 8, 10>  (存储占用4字节)
 * - float8_e5m2: FloatSpec<1, 5, 2>
 * - float8_e4m3: FloatSpec<1, 4, 3>
 * - float8_e8m0: FloatSpec<0, 8, 0>  (无符号位)
 */
template <int SignBits, int ExpBits, int MantissaBits>
class FloatSpec {
    static_assert(SignBits >= 0 && SignBits <= 1, "SignBits must be 0 or 1");
    static_assert(ExpBits > 0, "ExpBits must be positive");
    static_assert(MantissaBits >= 0, "MantissaBits must be non-negative");
    
public:
    // 类型常量
    static constexpr int sign_bits = SignBits;
    static constexpr int exp_bits = ExpBits;
    static constexpr int mantissa_bits = MantissaBits;
    static constexpr int total_bits = SignBits + ExpBits + MantissaBits;
    static constexpr int bias = (1 << (ExpBits - 1)) - 1;  // IEEE 754 偏置
    
    // 存储类型
    using StorageType = typename StorageTypeSelector<total_bits>::type;
    // 计算类型
    using ComputeType = typename ComputeTypeSelector<ExpBits, MantissaBits>::type;
    
    static constexpr int storage_bits = sizeof(StorageType) * 8;
    
private:
    StorageType m_bits{0};
    
    // 掩码
    static constexpr StorageType mantissa_mask = (static_cast<StorageType>(1) << MantissaBits) - 1;
    static constexpr StorageType exp_mask = ((static_cast<StorageType>(1) << ExpBits) - 1) << MantissaBits;
    static constexpr StorageType sign_mask = SignBits ? (static_cast<StorageType>(1) << (ExpBits + MantissaBits)) : 0;
    
public:
    // ==================== 构造函数 ====================
    FloatSpec() = default;
    
    // 从计算类型构造
    explicit FloatSpec(ComputeType value) {
        fromComputeType(value);
    }
    
    // 从任意数值类型构造
    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    explicit FloatSpec(T value) : FloatSpec(static_cast<ComputeType>(value)) {}
    
    // 从位表示构造
    static constexpr FloatSpec from_bits(StorageType bits) {
        FloatSpec f;
        f.m_bits = bits;
        return f;
    }
    
    // ==================== 类型转换（与 bfloat16 对齐）====================
    // 隐式转换到计算类型（float 或 double）
    operator ComputeType() const {
        return toComputeType();
    }
    
    // 隐式转换到 float（如果 ComputeType 不是 float）
    template <typename U = ComputeType, typename = std::enable_if_t<!std::is_same_v<U, float>>>
    operator float() const {
        return static_cast<float>(toComputeType());
    }
    
    // 隐式转换到 double（如果 ComputeType 不是 double）
    template <typename U = ComputeType, typename = std::enable_if_t<!std::is_same_v<U, double>>>
    operator double() const {
        return static_cast<double>(toComputeType());
    }
    
    // ==================== 与 bfloat16 对齐的方法 ====================
    // to_string: 返回浮点数的字符串表示
    std::string to_string() const {
        return std::to_string(static_cast<double>(toComputeType()));
    }
    
    // size: 返回存储大小（字节）
    constexpr size_t size() const {
        return sizeof(StorageType);
    }
    
    // ==================== 位操作 ====================
    StorageType to_bits() const { return m_bits; }
    
    int sign() const {
        if constexpr (SignBits == 0) return 0;
        else return (m_bits >> (ExpBits + MantissaBits)) & 1;
    }
    
    int exponent() const {
        return (m_bits >> MantissaBits) & ((1 << ExpBits) - 1);
    }
    
    StorageType mantissa() const {
        return m_bits & mantissa_mask;
    }
    
    /// @brief 返回零值
    static constexpr FloatSpec zero() {
        return from_bits(0);
    }
    
    /// @brief 返回 1.0
    static constexpr FloatSpec one() {
        constexpr StorageType bits = static_cast<StorageType>(bias) << MantissaBits;
        return from_bits(bits);
    }
    
    /// @brief 返回正无穷大
    static constexpr FloatSpec inf() {
        constexpr StorageType bits = static_cast<StorageType>((1 << ExpBits) - 1) << MantissaBits;
        return from_bits(bits);
    }
    
    /// @brief 返回 NaN
    static constexpr FloatSpec nan() {
        constexpr StorageType bits = 
            (static_cast<StorageType>((1 << ExpBits) - 1) << MantissaBits) | 1;
        return from_bits(bits);
    }
    
    // ==================== 特殊值检查 ====================
    bool isZero() const {
        if constexpr (SignBits == 0) {
            return (m_bits & (exp_mask | mantissa_mask)) == 0;
        } else {
            return (m_bits & ~sign_mask) == 0;
        }
    }
    
    bool isInf() const {
        return exponent() == ((1 << ExpBits) - 1) && mantissa() == 0;
    }
    
    bool isNaN() const {
        return exponent() == ((1 << ExpBits) - 1) && mantissa() != 0;
    }
    
    bool isDenormal() const {
        return exponent() == 0 && mantissa() != 0;
    }
    
    // ==================== 比较运算符 ====================
    template <typename T>
    bool operator==(const T& other) const {
        return toComputeType() == static_cast<ComputeType>(other);
    }
    
    template <typename T>
    bool operator!=(const T& other) const { return !(*this == other); }
    
    template <typename T>
    bool operator<(const T& other) const {
        return toComputeType() < static_cast<ComputeType>(other);
    }
    
    template <typename T>
    bool operator<=(const T& other) const {
        return toComputeType() <= static_cast<ComputeType>(other);
    }
    
    template <typename T>
    bool operator>(const T& other) const {
        return toComputeType() > static_cast<ComputeType>(other);
    }
    
    template <typename T>
    bool operator>=(const T& other) const {
        return toComputeType() >= static_cast<ComputeType>(other);
    }
    
    // ==================== 算术运算符 ====================
    template <typename T>
    FloatSpec operator+(const T& other) const {
        return FloatSpec(toComputeType() + static_cast<ComputeType>(other));
    }
    
    template <typename T>
    FloatSpec& operator+=(const T& other) {
        fromComputeType(toComputeType() + static_cast<ComputeType>(other));
        return *this;
    }
    
    template <typename T>
    FloatSpec operator-(const T& other) const {
        return FloatSpec(toComputeType() - static_cast<ComputeType>(other));
    }
    
    template <typename T>
    FloatSpec& operator-=(const T& other) {
        fromComputeType(toComputeType() - static_cast<ComputeType>(other));
        return *this;
    }
    
    template <typename T>
    FloatSpec operator*(const T& other) const {
        return FloatSpec(toComputeType() * static_cast<ComputeType>(other));
    }
    
    template <typename T>
    FloatSpec& operator*=(const T& other) {
        fromComputeType(toComputeType() * static_cast<ComputeType>(other));
        return *this;
    }
    
    template <typename T>
    FloatSpec operator/(const T& other) const {
        return FloatSpec(toComputeType() / static_cast<ComputeType>(other));
    }
    
    template <typename T>
    FloatSpec& operator/=(const T& other) {
        fromComputeType(toComputeType() / static_cast<ComputeType>(other));
        return *this;
    }
    
    FloatSpec operator-() const {
        if constexpr (SignBits == 0) {
            return FloatSpec(-toComputeType());
        } else {
            return from_bits(m_bits ^ sign_mask);
        }
    }
    
    // ==================== 输出 ====================
    friend std::ostream& operator<<(std::ostream& out, const FloatSpec& f) {
        out << static_cast<double>(f.toComputeType());
        return out;
    }
    
private:
    // ==================== 内部转换函数 ====================
    ComputeType toComputeType() const {
        if constexpr (std::is_same_v<ComputeType, float>) {
            return toFloat32();
        } else {
            return toFloat64();
        }
    }
    
    void fromComputeType(ComputeType value) {
        if constexpr (std::is_same_v<ComputeType, float>) {
            fromFloat32(value);
        } else {
            fromFloat64(value);
        }
    }
    
    float toFloat32() const {
        // 提取各部分
        int s = sign();
        int e = exponent();
        uint32_t m = mantissa();
        
        uint32_t result;
        
        if (e == 0) {
            if (m == 0) {
                // 零
                result = static_cast<uint32_t>(s) << 31;
            } else {
                // 非规格化数：转换为规格化数
                // 对于MantissaBits=0的情况，非规格化数就是0
                if constexpr (MantissaBits == 0) {
                    result = static_cast<uint32_t>(s) << 31;
                } else {
                    // 找到最高位1的位置并规格化
                    int shift = 0;
                    while ((m & (static_cast<uint32_t>(1) << (MantissaBits - 1))) == 0 && shift < MantissaBits) {
                        m <<= 1;
                        shift++;
                    }
                    m = (m << 1) & mantissa_mask;  // 移除隐含的1
                    e = 1 - shift;
                    
                    // 转换指数到float32的偏置
                    int newExp = e - bias + 127;
                    if (newExp <= 0) {
                        // 仍然是非规格化数
                        result = (static_cast<uint32_t>(s) << 31) |
                                 (static_cast<uint32_t>(m) << (23 - MantissaBits));
                    } else {
                        result = (static_cast<uint32_t>(s) << 31) |
                             (static_cast<uint32_t>(newExp) << 23) |
                             (static_cast<uint32_t>(m) << (23 - MantissaBits));
                    }
                }
            }
        } else if (e == ((1 << ExpBits) - 1)) {
            // 无穷大或NaN
            result = (static_cast<uint32_t>(s) << 31) |
                     (static_cast<uint32_t>(0xFF) << 23) |
                     (m != 0 ? 1 : 0);  // 保留NaN标志
        } else {
            // 规格化数
            int newExp = e - bias + 127;
            if (newExp <= 0) {
                // 下溢到非规格化数
                result = static_cast<uint32_t>(s) << 31;
            } else if (newExp >= 255) {
                // 上溢到无穷大
                result = (static_cast<uint32_t>(s) << 31) | (static_cast<uint32_t>(0xFF) << 23);
            } else {
                // 正常转换
                uint32_t newMantissa;
                if constexpr (MantissaBits <= 23) {
                    newMantissa = static_cast<uint32_t>(m) << (23 - MantissaBits);
                } else {
                    newMantissa = static_cast<uint32_t>(m >> (MantissaBits - 23));
                }
                result = (static_cast<uint32_t>(s) << 31) |
                         (static_cast<uint32_t>(newExp) << 23) |
                         newMantissa;
            }
        }
        
        float f;
        std::memcpy(&f, &result, sizeof(float));
        return f;
    }
    
    double toFloat64() const {
        // 类似toFloat32，但目标是float64
        int s = sign();
        int e = exponent();
        uint64_t m = mantissa();
        
        uint64_t result;
        
        if (e == 0 && m == 0) {
            // 零
            result = static_cast<uint64_t>(s) << 63;
        } else if (e == ((1 << ExpBits) - 1)) {
            // 无穷大或NaN
            result = (static_cast<uint64_t>(s) << 63) |
                     (static_cast<uint64_t>(0x7FF) << 52) |
                     (m != 0 ? 1 : 0);
        } else if (e == 0) {
            // 非规格化数转换
            int shift = 0;
            while ((m & (static_cast<uint64_t>(1) << (MantissaBits - 1))) == 0 && shift < MantissaBits) {
                m <<= 1;
                shift++;
            }
            int newExp = 1 - shift - bias + 1023;
            if (newExp > 0) {
                m = (m << 1) & ((static_cast<uint64_t>(1) << MantissaBits) - 1);
                uint64_t newMantissa;
                if constexpr (MantissaBits <= 52) {
                    newMantissa = m << (52 - MantissaBits);
                } else {
                    newMantissa = m >> (MantissaBits - 52);
                }
                result = (static_cast<uint64_t>(s) << 63) |
                         (static_cast<uint64_t>(newExp) << 52) |
                         newMantissa;
            } else {
                result = static_cast<uint64_t>(s) << 63;
            }
        } else {
            // 规格化数
            int newExp = e - bias + 1023;
            if (newExp <= 0) {
                result = static_cast<uint64_t>(s) << 63;
            } else if (newExp >= 2047) {
                result = (static_cast<uint64_t>(s) << 63) | (static_cast<uint64_t>(0x7FF) << 52);
            } else {
                uint64_t newMantissa;
                if constexpr (MantissaBits <= 52) {
                    newMantissa = static_cast<uint64_t>(m) << (52 - MantissaBits);
                } else {
                    newMantissa = static_cast<uint64_t>(m >> (MantissaBits - 52));
                }
                result = (static_cast<uint64_t>(s) << 63) |
                         (static_cast<uint64_t>(newExp) << 52) |
                         newMantissa;
            }
        }
        
        double d;
        std::memcpy(&d, &result, sizeof(double));
        return d;
    }
    
    void fromFloat32(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(float));
        
        uint32_t s = (bits >> 31) & 1;
        uint32_t e = (bits >> 23) & 0xFF;
        uint32_t m = bits & 0x7FFFFF;
        
        StorageType newSign = SignBits ? static_cast<StorageType>(s) : 0;
        StorageType newExp, newMantissa;
        
        if (e == 0) {
            // 零或非规格化数
            newExp = 0;
            newMantissa = 0;
        } else if (e == 0xFF) {
            // 无穷大或NaN
            newExp = (1 << ExpBits) - 1;
            newMantissa = m != 0 ? 1 : 0;
        } else {
            // 规格化数
            int realExp = static_cast<int>(e) - 127 + bias;
            if (realExp <= 0) {
                // 下溢
                newExp = 0;
                newMantissa = 0;
            } else if (realExp >= (1 << ExpBits) - 1) {
                // 上溢
                newExp = (1 << ExpBits) - 1;
                newMantissa = 0;
            } else {
                newExp = static_cast<StorageType>(realExp);
                if constexpr (MantissaBits <= 23) {
                    // 截断高位
                    newMantissa = static_cast<StorageType>(m >> (23 - MantissaBits));
                    // 四舍五入到最近偶数
                    uint32_t remainder = m & ((1 << (23 - MantissaBits)) - 1);
                    uint32_t halfBit = 1 << (22 - MantissaBits);
                    if (remainder > halfBit || (remainder == halfBit && (newMantissa & 1))) {
                        newMantissa++;
                        if (newMantissa >= (static_cast<StorageType>(1) << MantissaBits)) {
                            newMantissa = 0;
                            newExp++;
                            if (newExp >= (1 << ExpBits) - 1) {
                                newExp = (1 << ExpBits) - 1;
                                newMantissa = 0;
                            }
                        }
                    }
                } else {
                    newMantissa = static_cast<StorageType>(m) << (MantissaBits - 23);
                }
            }
        }
        
        m_bits = (newSign << (ExpBits + MantissaBits)) | (newExp << MantissaBits) | newMantissa;
    }
    
    void fromFloat64(double value) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(double));
        
        uint64_t s = (bits >> 63) & 1;
        uint64_t e = (bits >> 52) & 0x7FF;
        uint64_t m = bits & 0xFFFFFFFFFFFFF;
        
        StorageType newSign = SignBits ? static_cast<StorageType>(s) : 0;
        StorageType newExp, newMantissa;
        
        if (e == 0) {
            newExp = 0;
            newMantissa = 0;
        } else if (e == 0x7FF) {
            newExp = (1 << ExpBits) - 1;
            newMantissa = m != 0 ? 1 : 0;
        } else {
            int realExp = static_cast<int>(e) - 1023 + bias;
            if (realExp <= 0) {
                newExp = 0;
                newMantissa = 0;
            } else if (realExp >= (1 << ExpBits) - 1) {
                newExp = (1 << ExpBits) - 1;
                newMantissa = 0;
            } else {
                newExp = static_cast<StorageType>(realExp);
                if constexpr (MantissaBits <= 52) {
                    newMantissa = static_cast<StorageType>(m >> (52 - MantissaBits));
                } else {
                    newMantissa = static_cast<StorageType>(m) << (MantissaBits - 52);
                }
            }
        }
        
        m_bits = (newSign << (ExpBits + MantissaBits)) | (newExp << MantissaBits) | newMantissa;
    }
};

// ==================== 类型别名 ====================
using float16 = FloatSpec<1, 5, 10>;       // IEEE 754 half precision
using float8_e5m2 = FloatSpec<1, 5, 2>;    // E5M2 format (8-bit float, 1 sign, 5 exp, 2 mantissa)
using float8_e4m3 = FloatSpec<1, 4, 3>;    // E4M3 format (8-bit float, 1 sign, 4 exp, 3 mantissa)
using float8_e8m0 = FloatSpec<0, 8, 0>;    // E8M0 format (unsigned, 8-bit exponent only, no mantissa)
using float8_ue8m0 = FloatSpec<0, 8, 0>;   // 别名: unsigned E8M0

} // namespace yt

// ==================== std::numeric_limits 特化 ====================
namespace std {

template <int S, int E, int M>
class numeric_limits<yt::FloatSpec<S, E, M>> {
    using F = yt::FloatSpec<S, E, M>;
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = (S == 1);
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = false;
    static constexpr bool has_denorm = true;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = false;
    static constexpr int digits = M + 1;
    static constexpr int digits10 = static_cast<int>((M + 1) * 0.30103);
    static constexpr int max_digits10 = static_cast<int>((M + 1) * 0.30103) + 2;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 2 - ((1 << (E - 1)) - 1);
    static constexpr int min_exponent10 = static_cast<int>(min_exponent * 0.30103);
    static constexpr int max_exponent = (1 << (E - 1));
    static constexpr int max_exponent10 = static_cast<int>(max_exponent * 0.30103);
    
    static constexpr F min() noexcept { 
        // 最小正规格化数
        return F::from_bits(static_cast<typename F::StorageType>(1) << M);
    }
    static constexpr F max() noexcept {
        // 最大有限数
        constexpr typename F::StorageType maxExp = ((1 << E) - 2);
        constexpr typename F::StorageType maxMant = (static_cast<typename F::StorageType>(1) << M) - 1;
        return F::from_bits((maxExp << M) | maxMant);
    }
    static constexpr F lowest() noexcept {
        if constexpr (S == 0) return F::from_bits(0);
        else return F::from_bits(max().to_bits() | (static_cast<typename F::StorageType>(1) << (E + M)));
    }
    static constexpr F epsilon() noexcept {
        // 1.0和下一个可表示数之间的差
        constexpr int bias = (1 << (E - 1)) - 1;
        constexpr int eps_exp = bias - M;
        if constexpr (eps_exp > 0) {
            return F::from_bits(static_cast<typename F::StorageType>(eps_exp) << M);
        } else {
            return F::from_bits(1);  // 非规格化最小
        }
    }
    static constexpr F round_error() noexcept { return F(0.5f); }
    static constexpr F infinity() noexcept { return F::inf(); }
    static constexpr F quiet_NaN() noexcept { return F::nan(); }
    static constexpr F signaling_NaN() noexcept { return F::nan(); }
    static constexpr F denorm_min() noexcept { return F::from_bits(1); }
};

} // namespace std
