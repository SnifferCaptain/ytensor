#pragma once
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>

// 前向声明
namespace yt {
class YTensorBase;
template<typename T, int dim> class YTensor;
}

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
    struct is_ytensor_template<yt::YTensor<U, d>> : std::true_type {};

    template<typename U>
    inline constexpr bool is_ytensor_template_v = is_ytensor_template<std::decay_t<U>>::value;

    /// @brief 获取YTensor的维度
    template<typename U>
    struct ytensor_dim { static constexpr int value = 0; };

    template<typename U, int d>
    struct ytensor_dim<yt::YTensor<U, d>> { static constexpr int value = d; };

    template<typename U>
    inline constexpr int ytensor_dim_v = ytensor_dim<std::decay_t<U>>::value;

    // 辅助结构体
    template <typename T>
    struct first_arg_of : first_arg_of<decltype(&T::operator())> {};
    template <typename R, typename Arg0, typename... Rest>
    struct first_arg_of<R(*)(Arg0, Rest...)> {using type = Arg0;};
    template <typename C, typename R, typename Arg0, typename... Rest>
    struct first_arg_of<R(C::*)(Arg0, Rest...) const> {using type = Arg0;};
    template <typename C, typename R, typename Arg0, typename... Rest>
    struct first_arg_of<R(C::*)(Arg0, Rest...)> {using type = Arg0;};

    /// @brief 获取函数第一个参数类型
    template <typename Func> using first_arg_of_t = std::remove_cvref_t<typename first_arg_of<std::remove_cvref_t<Func>>::type>;


    /// @brief 获取参数包中张量的最大维度
    template<typename... Args>
    constexpr int max_dim() {
        int dims[] = {ytensor_dim_v<Args>...};
        int maxd = 1;
        for (int d : dims) if (d > maxd) maxd = d;
        return maxd;
    }

    template<typename U> struct is_ytensorbase_only : std::bool_constant<is_ytensor_v<U> && !is_ytensor_template_v<U>> {};
    /// @brief 判断类型是否是纯YTensorBase
    template<typename U> inline constexpr bool is_ytensorbase_only_v = is_ytensorbase_only<U>::value;

    template<typename... Args> constexpr bool has_ytensorbase_only() {return (is_ytensorbase_only_v<std::decay_t<Args>> || ...);}
    template<typename... Args> constexpr bool all_ytensor_template() {return !has_ytensorbase_only<Args...>();}
    template<typename T, typename ScalarType> struct is_valid_broadcast_arg : std::bool_constant<
        is_ytensor_v<std::decay_t<T>> || 
        std::is_convertible_v<std::decay_t<T>, ScalarType>
    > {};
    template<typename T, typename ScalarType> inline constexpr bool is_valid_broadcast_arg_v = is_valid_broadcast_arg<T, ScalarType>::value;

    /// @brief 检查参数包中所有参数是否都是有效的broadcast参数
    template<typename ScalarType, typename... Args>
    constexpr bool all_valid_broadcast_args() {return (is_valid_broadcast_arg_v<Args, ScalarType> && ...);}
}
