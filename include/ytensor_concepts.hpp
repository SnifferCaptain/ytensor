#pragma once
#include <concepts>
#include <initializer_list>
#include <iostream>

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
