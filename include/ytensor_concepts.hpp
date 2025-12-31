#pragma once
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// 前向声明
namespace yt {
class YTensorBase;
}
template<typename T, int dim> class YTensor;


namespace yt::concepts {
    // binary operation concepts
    template<typename T> concept HAVE_ADD = requires(T a, T b) { { a + b } -> std::same_as<T>; };
    template<typename T> concept HAVE_SUB = requires(T a, T b) { { a - b } -> std::same_as<T>; };
    template<typename T> concept HAVE_MUL = requires(T a, T b) { { a * b } -> std::same_as<T>; };
    template<typename T> concept HAVE_DIV = requires(T a, T b) { { a / b } -> std::same_as<T>; };
    template<typename T> concept HAVE_MOD = requires(T a, T b) { { a % b } -> std::same_as<T>; };
    template<typename T> concept HAVE_AND = requires(T a, T b) { { a & b } -> std::same_as<T>; };
    template<typename T> concept HAVE_OR = requires(T a, T b) { { a | b } -> std::same_as<T>; };
    template<typename T> concept HAVE_XOR = requires(T a, T b) { { a ^ b } -> std::same_as<T>; };
    template<typename T> concept HAVE_LSHIFT = requires(T a, int b) { { a << b } -> std::same_as<T>; };
    template<typename T> concept HAVE_RSHIFT = requires(T a, int b) { { a >> b } -> std::same_as<T>; };

    // inplace binary operation concepts
    template<typename T> concept HAVE_ADD_INPLACE = requires(T a, T b) { { a += b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_SUB_INPLACE = requires(T a, T b) { { a -= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_MUL_INPLACE = requires(T a, T b) { { a *= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_DIV_INPLACE = requires(T a, T b) { { a /= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_MOD_INPLACE = requires(T a, T b) { { a %= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_AND_INPLACE = requires(T a, T b) { { a &= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_OR_INPLACE = requires(T a, T b) { { a |= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_XOR_INPLACE = requires(T a, T b) { { a ^= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_LSHIFT_INPLACE = requires(T a, int b) { { a <<= b } -> std::same_as<T&>; };
    template<typename T> concept HAVE_RSHIFT_INPLACE = requires(T a, int b) { { a >>= b } -> std::same_as<T&>; };

    // unary operation concepts
    template<typename T> concept HAVE_NOT = requires(T a) { { ~a } -> std::same_as<T>; };

    template<typename T> concept HAVE_EQ = requires(T a, T b) { { a == b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_NEQ = requires(T a, T b) { { a != b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_LT = requires(T a, T b) { { a < b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_LE = requires(T a, T b) { { a <= b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_GT = requires(T a, T b) { { a > b } -> std::same_as<bool>; };
    template<typename T> concept HAVE_GE = requires(T a, T b) { { a >= b } -> std::same_as<bool>; };

    // other
    template <typename T>
    concept HAVE_OSTREAM = requires(std::ostream &os, const T &value) {
        { os << value } -> std::same_as<std::ostream &>;
    };

    // array min/max constexpr
    template <typename T>
    constexpr inline T CONSTEXPR_MAX(std::initializer_list<T> list) {
        return *std::max_element(list.begin(), list.end());
    }
};

namespace yt::traits {
    /// @brief 判断类型是否为YTensor或YTensorBase（广义tensor类型）
    template<typename U>
    struct is_ytensor : std::bool_constant<std::is_base_of_v<yt::YTensorBase, std::decay_t<U>>> {};

    template<typename U>
    inline constexpr bool is_ytensor_v = is_ytensor<U>::value;

    /// @brief 判断类型是否为精确的YTensor<U, d>模板实例
    template<typename U>
    struct is_ytensor_template : std::false_type {};

    template<typename U, int d>
    struct is_ytensor_template<YTensor<U, d>> : std::true_type {};

    template<typename U>
    inline constexpr bool is_ytensor_template_v = is_ytensor_template<std::decay_t<U>>::value;

    /// @brief 获取YTensor的维度
    template<typename U>
    struct ytensor_dim { static constexpr int value = 0; };

    template<typename U, int d>
    struct ytensor_dim<YTensor<U, d>> { static constexpr int value = d; };

    template<typename U>
    inline constexpr int ytensor_dim_v = ytensor_dim<std::decay_t<U>>::value;

    /// @brief 获取参数包中张量的最大维度
    template<typename... Args>
    constexpr int max_dim() {
        int dims[] = {ytensor_dim_v<Args>...};
        int maxd = 1;
        for (int d : dims) if (d > maxd) maxd = d;
        return maxd;
    }
}
