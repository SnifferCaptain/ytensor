#pragma once
/***************
 * @file: ytensor_types.hpp
 * @brief: YTensor 数据类型定义
 * @author: SnifferCaptain
 ***************/

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "./type/bfloat16.hpp"
#include "./type/float_spec.hpp"
#include "./ytensor_concepts.hpp"
#include "./ytensor_infos.hpp"

namespace yt {
class YTensorBase;
}

namespace yt::type {

/// @brief 校验当前通用storage能否满足注册类型的对齐要求。
/// @note YTensor当前使用普通动态分配；over-aligned类型必须等专用allocator支持后再开放。
template <typename T>
inline void validateRegisteredTypeAlignment() {
    if constexpr (alignof(T) > alignof(std::max_align_t)) {
        throw std::invalid_argument("registerType: over-aligned types are not supported by current storage");
    }
}

/// @brief 运行时broadcast操作类型。
enum class YBroadcastOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    LShift,
    RShift,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
};

/// @brief 运行时reduction操作类型。
enum class YReduceOp : uint8_t {
    Sum,
    Max,
    Mean,
};

/// @brief dtype运行时kernel表。空字段表示该 dtype 不支持对应能力。
/// @details 所有回调接收已分配、layout/dtype/shape 已由 owner 校验的 tensor；broadcast 创建型回调写入 out，
///          broadcastInplace 修改既有 out。函数内部必须保证 dtype/op 判断在元素循环外完成。
struct YDTypeKernels {
    /// @brief 将 inputs 广播计算到独立 out。
    void (*broadcast
    )(YBroadcastOp op, YTensorBase& out, const std::vector<const YTensorBase*>& inputs) = nullptr;
    /// @brief 将 inputs 广播计算到既有 out；out 可能非连续但必须满足 owner 的可写合同。
    void (*broadcastInplace
    )(YBroadcastOp op, YTensorBase& out, const std::vector<const YTensorBase*>& inputs) = nullptr;
    /// @brief 单轴 reduction，out 已按 keep-dim shape 分配。
    void (*reduce)(YReduceOp op, YTensorBase& out, const YTensorBase& input, int axis) = nullptr;
    /// @brief 单轴 indexed reduction，同时写 values 和 indices。
    void (*reduceIndexed
    )(YReduceOp op, YTensorBase& values, YTensorBase& indices, const YTensorBase& input, int axis) = nullptr;
    /// @brief 已校验矩阵 shape/layout 的 runtime matmul 回调。
    void (*matmul)(YTensorBase& out, const YTensorBase& left, const YTensorBase& right) = nullptr;
    /// @brief 使用二维 bool mask 的 runtime masked-matmul 回调。
    void (*maskedMatmul
    )(YTensorBase& out, const YTensorBase& left, const YTensorBase& right, const YTensorBase& mask,
      double maskedValue) = nullptr;
};

/// @brief pairwise dtype cast 回调类型。
/// @note src 指向已构造源对象；dst 指向可写目标。非POD目标在调用前已经构造，kernel 必须赋值而非placement-new。
using YCastKernel = void (*)(void* dst, const void* src);

/// @brief 查找内置或已注册custom dtype的生命周期信息。
/// @warning 返回的custom registry引用可能在后续注册触发rehash后失效，不得长期保存。
inline std::optional<std::reference_wrapper<const yt::type::TypeRegItem>> getTypeInfo(
    const std::string& typeName
);

/// @brief 以 \\x1f 为分隔符拼接 srcDtype 与 dstDtype，用作 cast kernel 查找键。
inline std::string makeCastKernelKey(const std::string& srcDtype, const std::string& dstDtype) {
    return srcDtype + "\x1f" + dstDtype;
}

namespace internal {
    #if YT_USE_LIB
    std::unordered_map<std::string, YCastKernel>& getCastKernelRegistry();
    std::mutex& getCastKernelRegistryMutex();
    #else
    inline auto& getCastKernelRegistry() {
        static std::unordered_map<std::string, YCastKernel> registry;
        return registry;
    }
    inline std::mutex& getCastKernelRegistryMutex() {
        static std::mutex mutex;
        return mutex;
    }
    #endif
}

/// @brief 注册 cast kernel。若已有同名 key 则覆盖。
/// @throw std::invalid_argument kernel 为空时抛出。
/// @note registry 只保存函数指针，不管理外部状态；函数及其依赖必须在注册有效期内保持已加载。
inline void registerCastKernel(const std::string& srcDtype, const std::string& dstDtype, YCastKernel kernel) {
    if (!kernel) throw std::invalid_argument("registerCastKernel: kernel must not be null");
    if (srcDtype.empty() || dstDtype.empty() || !getTypeInfo(srcDtype) || !getTypeInfo(dstDtype)) {
        throw std::invalid_argument("registerCastKernel: source and destination dtypes must be registered");
    }
    std::lock_guard<std::mutex> lock(internal::getCastKernelRegistryMutex());
    internal::getCastKernelRegistry()[makeCastKernelKey(srcDtype, dstDtype)] = kernel;
}

/// @brief 查找 cast kernel，未注册时返回 nullptr。
/// @return 已注册函数指针的快照；registry mutex 在返回前释放。
inline YCastKernel getCastKernel(const std::string& srcDtype, const std::string& dstDtype) {
    std::lock_guard<std::mutex> lock(internal::getCastKernelRegistryMutex());
    auto& registry = internal::getCastKernelRegistry();
    auto it = registry.find(makeCastKernelKey(srcDtype, dstDtype));
    return it == registry.end() ? nullptr : it->second;
}

/// @brief dtype kernel注册表，按dtype字符串保存运行时计算入口。
namespace internal {
    #if YT_USE_LIB
    std::unordered_map<std::string, YDTypeKernels>& getDTypeKernelRegistry();
    std::mutex& getDTypeKernelRegistryMutex();
    #else
    inline auto& getDTypeKernelRegistry() {
        static std::unordered_map<std::string, YDTypeKernels> registry;
        return registry;
    }
    inline std::mutex& getDTypeKernelRegistryMutex() {
        static std::mutex mutex;
        return mutex;
    }
    #endif
}

/// @brief 注册指定dtype的运行时kernel表，用非空字段 **覆盖** 已有注册。
/// @note 与 mergeDTypeKernels 不同：只要 kernels 中某字段非空即覆盖，
///       而非仅填充空位。适用于显式全量注册。
inline void registerDTypeKernels(const std::string& dtype, const YDTypeKernels& kernels) {
    if (dtype.empty() || !getTypeInfo(dtype)) {
        throw std::invalid_argument("registerDTypeKernels: dtype must be registered");
    }
    std::lock_guard<std::mutex> lock(internal::getDTypeKernelRegistryMutex());
    auto& current = internal::getDTypeKernelRegistry()[dtype];
    if (kernels.broadcast) current.broadcast = kernels.broadcast;
    if (kernels.broadcastInplace) current.broadcastInplace = kernels.broadcastInplace;
    if (kernels.reduce) current.reduce = kernels.reduce;
    if (kernels.reduceIndexed) current.reduceIndexed = kernels.reduceIndexed;
    if (kernels.matmul) current.matmul = kernels.matmul;
    if (kernels.maskedMatmul) current.maskedMatmul = kernels.maskedMatmul;
}

/// @brief 合并指定dtype的运行时kernel表，**仅填充**当前为空的字段。
/// @note 与 registerDTypeKernels 不同：不会覆盖已有非空字段。
///       适用于增量注册，不破坏已有实现。
inline void mergeDTypeKernels(const std::string& dtype, const YDTypeKernels& kernels) {
    if (dtype.empty() || !getTypeInfo(dtype)) {
        throw std::invalid_argument("mergeDTypeKernels: dtype must be registered");
    }
    std::lock_guard<std::mutex> lock(internal::getDTypeKernelRegistryMutex());
    auto& current = internal::getDTypeKernelRegistry()[dtype];
    if (kernels.broadcast && !current.broadcast) current.broadcast = kernels.broadcast;
    if (kernels.broadcastInplace && !current.broadcastInplace)
        current.broadcastInplace = kernels.broadcastInplace;
    if (kernels.reduce && !current.reduce) current.reduce = kernels.reduce;
    if (kernels.reduceIndexed && !current.reduceIndexed) current.reduceIndexed = kernels.reduceIndexed;
    if (kernels.matmul && !current.matmul) current.matmul = kernels.matmul;
    if (kernels.maskedMatmul && !current.maskedMatmul) current.maskedMatmul = kernels.maskedMatmul;
}

/// @brief 判断指定dtype是否已注册运行时计算kernel。
inline bool hasDTypeKernels(const std::string& dtype) {
    std::lock_guard<std::mutex> lock(internal::getDTypeKernelRegistryMutex());
    auto& registry = internal::getDTypeKernelRegistry();
    auto it = registry.find(dtype);
    if (it == registry.end()) return false;
    const auto& kernels = it->second;
    return kernels.broadcast || kernels.broadcastInplace || kernels.reduce || kernels.reduceIndexed ||
           kernels.matmul || kernels.maskedMatmul;
}

/// @brief 获取指定dtype的运行时kernel表，未注册时抛出异常。
/// @return kernel table 的值拷贝，可在 registry mutex 释放后安全调用其中的函数指针。
inline YDTypeKernels getDTypeKernels(const std::string& dtype) {
    std::lock_guard<std::mutex> lock(internal::getDTypeKernelRegistryMutex());
    auto& registry = internal::getDTypeKernelRegistry();
    auto it = registry.find(dtype);
    if (it == registry.end()) {
        throw std::runtime_error("dtype kernels not registered for dtype: " + dtype);
    }
    return it->second;
}

/// @brief 获取内置基础类型的注册信息（POD属性、大小、构造/析构函数等）。
/// @param typeName 类型名，如 "float32"
/// @return 若为内置类型则返回 TypeRegItem 的引用包装，否则返回 nullopt。
inline std::optional<std::reference_wrapper<const yt::type::TypeRegItem>> getBuiltinTypeInfo(
    const std::string& typeName
) {
    static const std::unordered_map<std::string, yt::type::TypeRegItem> builtins = [] {
        std::unordered_map<std::string, yt::type::TypeRegItem> m;
        auto addPod = [&](const std::string& name, int32_t size) {
            yt::type::TypeRegItem item;
            item.name = name;
            item.size = size;
            item.isPOD = true;
            m.emplace(name, std::move(item));
        };
        addPod("float32", sizeof(float));
        addPod("float64", sizeof(double));
        addPod("int8", sizeof(int8_t));
        addPod("int16", sizeof(int16_t));
        addPod("int32", sizeof(int32_t));
        addPod("int64", sizeof(int64_t));
        addPod("uint8", sizeof(uint8_t));
        addPod("uint16", sizeof(uint16_t));
        addPod("uint32", sizeof(uint32_t));
        addPod("uint64", sizeof(uint64_t));
        addPod("bool", sizeof(bool));
        addPod("bfloat16", sizeof(yt::bfloat16));
        addPod("float16", sizeof(yt::float16));
        addPod("float8_e5m2", sizeof(yt::float8_e5m2));
        addPod("float8_e4m3", sizeof(yt::float8_e4m3));
        addPod("float8_e8m0", sizeof(yt::float8_e8m0));
        addPod("float8_ue8m0", sizeof(yt::float8_ue8m0));

        yt::type::TypeRegItem str;
        str.name = "string";
        str.size = static_cast<int32_t>(sizeof(std::string));
        str.isPOD = false;
        str.toString = [](const void* data) -> std::string {
            return *reinterpret_cast<const std::string*>(data);
        };
        str.defaultConstruct = [](void* dest) { new (dest) std::string(); };
        str.copyConstruct = [](void* dest, const void* src) {
            new (dest) std::string(*reinterpret_cast<const std::string*>(src));
        };
        str.copyAssign = [](void* dest, const void* src) {
            *reinterpret_cast<std::string*>(dest) = *reinterpret_cast<const std::string*>(src);
        };
        str.swap = [](void* left, void* right) noexcept {
            reinterpret_cast<std::string*>(left)->swap(*reinterpret_cast<std::string*>(right));
        };
        str.destructor = [](void* ptr) { reinterpret_cast<std::string*>(ptr)->~basic_string(); };
        str.serialize = [](const void* src) -> std::vector<char> {
            const auto& s = *reinterpret_cast<const std::string*>(src);
            return std::vector<char>(s.begin(), s.end());
        };
        str.deserialize = [](void* dst, const char* bytes, size_t len) {
            reinterpret_cast<std::string*>(dst)->assign(bytes, len);
        };
        m.emplace("string", std::move(str));
        return m;
    }();

    auto it = builtins.find(typeName);
    if (it != builtins.end()) {
        return std::cref(it->second);
    }
    return std::nullopt;
}

/// @brief 编译时类型列表模板
template <typename... Types>
struct TypeList {
    static constexpr size_t size = sizeof...(Types);

    /// @brief 追加类型到列表末尾
    template <typename... More>
    struct Append {
        using type = TypeList<Types..., More...>;
    };
};

/// @brief TypeList 合并辅助模板
template <typename List1, typename List2>
struct TypeListConcat;

template <typename... T1, typename... T2>
struct TypeListConcat<TypeList<T1...>, TypeList<T2...>> {
    using type = TypeList<T1..., T2...>;
};

/// @brief 判断类型是否属于某个 TypeList
template <typename T, typename List>
struct TypeListContains;

template <typename T>
struct TypeListContains<T, TypeList<>> : std::false_type {};

template <typename T, typename Head, typename... Tail>
struct TypeListContains<T, TypeList<Head, Tail...>>
    : std::bool_constant<
          std::is_same_v<std::remove_cv_t<T>, Head> || TypeListContains<T, TypeList<Tail...>>::value> {};

template <typename T, typename List>
inline constexpr bool TypeListContains_v = TypeListContains<T, List>::value;

/// @brief 标准整数类型（有符号 + 无符号）
using StandardIntTypes = TypeList<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>;

/// @brief 标准浮点类型
using StandardFloatTypes = TypeList<float, double>;

/// @brief 标准数值类型（浮点 + 整数）
using StandardNumericTypes = typename TypeListConcat<StandardFloatTypes, StandardIntTypes>::type;

/// @brief 扩展浮点类型（bfloat16, float16, float8 变体）
using ExtendedFloatTypes =
    TypeList<yt::bfloat16, yt::float16, yt::float8_e5m2, yt::float8_e4m3, yt::float8_e8m0>;

/// @brief 所有数值类型（标准 + 扩展浮点）
using AllNumericTypes = typename TypeListConcat<StandardNumericTypes, ExtendedFloatTypes>::type;

/// @brief 仅整数类型（用于位运算等）
using IntegerTypes = StandardIntTypes;

/// @brief Eigen 原生支持的类型（不含扩展浮点类型）
using EigenNativeTypes = StandardNumericTypes;

/// @brief 判断类型是否为库内置数值类型（用于数学编译后端）
template <typename T>
inline constexpr bool is_builtin_numeric_v = TypeListContains_v<std::remove_cv_t<T>, AllNumericTypes>;

// dtype 规范化命名

/// @brief 获取基础数据类型名称（不含张量嵌套）
/// @tparam T 数据类型
/// @return 数据类型名称字符串
template <typename T>
std::string getBaseTypeName() {
    if constexpr (std::is_same_v<T, float>)
        return "float32";
    else if constexpr (std::is_same_v<T, double>)
        return "float64";
    else if constexpr (std::is_same_v<T, int8_t>)
        return "int8";
    else if constexpr (std::is_same_v<T, int16_t>)
        return "int16";
    else if constexpr (std::is_same_v<T, int32_t>)
        return "int32";
    else if constexpr (std::is_same_v<T, int64_t>)
        return "int64";
    else if constexpr (std::is_same_v<T, uint8_t>)
        return "uint8";
    else if constexpr (std::is_same_v<T, uint16_t>)
        return "uint16";
    else if constexpr (std::is_same_v<T, uint32_t>)
        return "uint32";
    else if constexpr (std::is_same_v<T, uint64_t>)
        return "uint64";
    else if constexpr (std::is_same_v<T, bool>)
        return "bool";
    else if constexpr (std::is_same_v<T, std::string>)
        return "string";
    else if constexpr (std::is_same_v<T, yt::bfloat16>)
        return "bfloat16";
    else if constexpr (std::is_same_v<T, yt::float16>)
        return "float16";
    else if constexpr (std::is_same_v<T, yt::float8_e5m2>)
        return "float8_e5m2";
    else if constexpr (std::is_same_v<T, yt::float8_e4m3>)
        return "float8_e4m3";
    else if constexpr (std::is_same_v<T, yt::float8_e8m0>)
        return "float8_e8m0";
    else if constexpr (std::is_same_v<T, yt::float8_ue8m0>)
        return "float8_ue8m0";
    else {
        return typeid(T).name();
    }
}

/// @brief 获取数据类型名称（支持嵌套张量类型的前向声明）
/// @tparam T 数据类型
/// @return 数据类型名称字符串
template <typename T>
std::string getTypeName();

/// @brief 解析dtype字符串，提取内层类型名称
/// @param dtype 完整的dtype字符串，如 "YTensor<float32, 2>" 或 "YTensorBase<YTensor<float32, 2>>"
/// @return pair<外层类型名, 内层dtype字符串>，如果不是嵌套类型则返回 {"", dtype}
inline std::pair<std::string, std::string> parseDtypeInner(const std::string& dtype) {
    // 查找第一个 '<'
    auto pos = dtype.find('<');
    if (pos == std::string::npos) {
        return {"", dtype};  // 基础类型
    }

    std::string outerType = dtype.substr(0, pos);

    // 找到匹配的 '>'
    int depth = 1;
    size_t start = pos + 1;
    size_t end = start;
    while (end < dtype.size() && depth > 0) {
        if (dtype[end] == '<')
            depth++;
        else if (dtype[end] == '>')
            depth--;
        end++;
    }

    if (depth != 0) {
        return {"", dtype};  // 格式错误，作为基础类型处理
    }

    std::string inner = dtype.substr(start, end - start - 1);
    return {outerType, inner};
}

/// @brief 从嵌套dtype中提取最内层的基础类型名称
/// @param dtype dtype字符串
/// @return 基础类型名称，如 "float32"
inline std::string getBaseDtype(const std::string& dtype) {
    auto [outer, inner] = parseDtypeInner(dtype);
    if (outer.empty()) {
        return dtype;  // 已经是基础类型
    }

    // 对于YTensor<scalar, dim>格式，需要提取scalar部分
    if (outer == "YTensor") {
        // inner 是 "scalar_dtype, dim" 的形式
        // 我们需要找到最后一个逗号，取前面的部分作为scalar_dtype
        // 但要注意scalar_dtype本身可能包含逗号（如果是嵌套的YTensor）
        // 所以需要从后向前找，且要考虑括号匹配
        int depth = 0;
        int lastCommaPos = -1;
        for (int i = static_cast<int>(inner.size()) - 1; i >= 0; --i) {
            if (inner[i] == '>')
                depth++;
            else if (inner[i] == '<')
                depth--;
            else if (inner[i] == ',' && depth == 0) {
                lastCommaPos = i;
                break;
            }
        }

        if (lastCommaPos != -1) {
            std::string scalarPart = inner.substr(0, lastCommaPos);
            // 去除尾部空格
            scalarPart.erase(scalarPart.find_last_not_of(" ") + 1);
            return getBaseDtype(scalarPart);  // 递归解析scalar部分
        }
    }

    // 对于YTensorBase<inner>格式，inner就是内层dtype
    return getBaseDtype(inner);  // 递归解析
}

/// @brief 从YTensor dtype中提取维度
/// @param dtype dtype字符串，如 "YTensor<float32, 2>"
/// @return 维度值，如果不是YTensor类型则返回-1
inline int getDtypeDim(const std::string& dtype) {
    auto [outer, inner] = parseDtypeInner(dtype);
    if (outer != "YTensor") {
        return -1;
    }
    // inner 是 "float32, 2" 的形式，找到最后一个逗号后的数字
    auto commaPos = inner.rfind(',');
    if (commaPos == std::string::npos) {
        return -1;
    }
    std::string dimStr = inner.substr(commaPos + 1);
    // 去除空格
    dimStr.erase(0, dimStr.find_first_not_of(" "));
    dimStr.erase(dimStr.find_last_not_of(" ") + 1);
    return std::stoi(dimStr);
}

/// @brief 从YTensor dtype中提取scalar类型
/// @param dtype dtype字符串，如 "YTensor<float32, 2>"
/// @return scalar类型名，如 "float32"
inline std::string getDtypeScalar(const std::string& dtype) {
    auto [outer, inner] = parseDtypeInner(dtype);
    if (outer != "YTensor") {
        return dtype;  // 不是YTensor，返回自身
    }
    // inner 是 "float32, 2" 的形式，找到最后一个逗号前的部分
    auto commaPos = inner.rfind(',');
    if (commaPos == std::string::npos) {
        return inner;
    }
    std::string scalarStr = inner.substr(0, commaPos);
    // 去除尾部空格
    scalarStr.erase(scalarStr.find_last_not_of(" ") + 1);
    return scalarStr;
}

/// @brief 构建YTensor的规范化dtype字符串
/// @param scalarDtype 标量类型名称
/// @param dim 维度
/// @return 规范化的dtype字符串，如 "YTensor<float32, 2>"
inline std::string makeYTensorDtype(const std::string& scalarDtype, int dim) {
    return "YTensor<" + scalarDtype + ", " + std::to_string(dim) + ">";
}

/// @brief 构建YTensorBase的规范化dtype字符串
/// @param innerDtype 内层类型名称
/// @return 规范化的dtype字符串，如 "YTensorBase<float32>"
inline std::string makeYTensorBaseDtype(const std::string& innerDtype) {
    return "YTensorBase<" + innerDtype + ">";
}

/// @brief 获取数据类型名称
/// @tparam T 数据类型
/// @return 数据类型名称字符串
template <typename T>
std::string getTypeName() {
    // 首先检查是否已注册自定义名称
    std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
    const auto& registry = yt::type::internal::getMutableTypeRegistry();
    auto it = registry.find(typeid(T).name());
    if (it != registry.end()) {
        return it->second.name;
    }

    // 使用基础类型名称
    return getBaseTypeName<T>();
}

/// @brief 返回 T 的 canonical dtype 名称；YTensor 特化会递归保留内层 dtype 与 rank。
template <typename T>
struct CanonicalTypeName {
    static std::string get() { return getTypeName<T>(); }
};

template <typename T, int dim>
struct CanonicalTypeName<yt::YTensor<T, dim>> {
    static std::string get() {
        return makeYTensorDtype(CanonicalTypeName<T>::get(), dim);
    }
};

/// @brief 构造 YTensor<T, dim> facade 应使用的 canonical dtype。
template <typename T, int dim>
std::string getYTensorDtype() {
    if constexpr (yt::utils::is_ytensor_template_v<T>) {
        return makeYTensorDtype(CanonicalTypeName<T>::get(), dim);
    } else {
        return getTypeName<T>();
    }
}

/// @brief 获取数据类型大小（模板版本）
/// @tparam T 数据类型
/// @return 类型大小（字节）
template <typename T>
constexpr int32_t getTypeSize() {
    return static_cast<int32_t>(sizeof(T));
}

/// @brief 根据类型名称获取类型大小
/// @param typeName 类型名称
/// @return 类型大小（字节），未知类型返回0
inline int32_t getTypeSize(const std::string& typeName) {
    if (auto builtin = getBuiltinTypeInfo(typeName); builtin) {
        return builtin->get().size;
    }
    if (typeName == "float32")
        return 4;
    else if (typeName == "float64")
        return 8;
    else if (typeName == "int8")
        return 1;
    else if (typeName == "int16")
        return 2;
    else if (typeName == "int32")
        return 4;
    else if (typeName == "int64")
        return 8;
    else if (typeName == "uint8")
        return 1;
    else if (typeName == "uint16")
        return 2;
    else if (typeName == "uint32")
        return 4;
    else if (typeName == "uint64")
        return 8;
    else if (typeName == "bool")
        return 1;
    // non std
    else if (typeName == "bfloat16")
        return 2;
    else if (typeName == "float16")
        return 2;
    else if (typeName == "float8_e5m2")
        return 1;
    else if (typeName == "float8_e4m3")
        return 1;
    else if (typeName == "float8_e8m0" || typeName == "float8_ue8m0")
        return 1;
    else {
        // registered custom types
        std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
        const auto& registry = yt::type::internal::getMutableTypeRegistry();
        for (auto& [key, value] : registry) {
            if (value.name == typeName) {
                return value.size;
            }
        }
        // unk
        throw std::runtime_error(std::string("Type ") + typeName + " is not registered.");
        return 0;
    }
}

/// @brief 根据类型名称获取类型注册信息
/// @param typeName 类型名称
/// @return 类型注册信息的optional引用，未找到返回std::nullopt
/// @warning custom registry 引用不持有 registry 所有权；后续注册可能触发 rehash，调用方不得长期保存该引用。
inline std::optional<std::reference_wrapper<const yt::type::TypeRegItem>> getTypeInfo(
    const std::string& typeName
) {
    if (auto builtin = getBuiltinTypeInfo(typeName); builtin) {
        return builtin;
    }
    std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
    const auto& registry = yt::type::internal::getMutableTypeRegistry();
    for (auto& [key, value] : registry) {
        if (value.name == typeName) {
            return std::cref(value);
        }
    }
    return std::nullopt;  // 内置类型或未注册类型
}

/// @brief 在已持有 type registry mutex 的注册路径中验证 custom dtype 名称唯一性。
/// @throws std::invalid_argument 名称为空、与 builtin 冲突，或 C++ 类型/名称已注册时抛出。
template <typename T>
void validateRegisteredTypeName(const std::string& typeName) {
    if (typeName.empty()) {
        throw std::invalid_argument("Custom dtype name cannot be empty");
    }
    if (getBuiltinTypeInfo(typeName)) {
        throw std::invalid_argument("Custom dtype name conflicts with builtin dtype: " + typeName);
    }
    const std::string typeKey = typeid(T).name();
    for (const auto& [key, value] : yt::type::internal::getMutableTypeRegistry()) {
        if (key == typeKey) {
            throw std::invalid_argument("C++ type is already registered: " + typeName);
        }
        if (value.name == typeName) {
            throw std::invalid_argument("Custom dtype name is already registered: " + typeName);
        }
    }
}

/// @brief 检查类型是否为POD（或内置类型）
/// @param typeName 类型名称
/// @return true=POD类型，不需要特殊析构处理
inline bool isPODType(const std::string& typeName) {
    if (auto builtin = getBuiltinTypeInfo(typeName); builtin) {
        return builtin->get().isPOD;
    }
    // 检查注册的自定义类型
    auto info = getTypeInfo(typeName);
    return info ? info->get().isPOD : true;  // 未知类型假设为POD
}

/// @brief 注册自定义类型
/// @tparam T 要注册的类型
/// @param typeName 自定义类型名称
template <typename T>
void registerType(const std::string& typeName) {
    validateRegisteredTypeAlignment<T>();
    std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
    validateRegisteredTypeName<T>(typeName);
    auto& registry = yt::type::internal::getMutableTypeRegistry();
    int32_t typeSize = getTypeSize<T>();
    // default formatter: if type has operator<< then use that, else nullptr
    auto makeDefaultFormatter = []() -> std::function<std::string(const void*)> {
        if constexpr (yt::utils::HAVE_OSTREAM<T>) {
            return [](const void* data) {
                std::ostringstream oss;
                const T* p = reinterpret_cast<const T*>(data);
                oss << *p;
                return oss.str();
            };
        } else {
            return nullptr;
        }
    };

    // 非POD类型支持
    yt::type::TypeRegItem item;
    item.name = typeName;
    item.size = typeSize;
    item.toString = makeDefaultFormatter();
    item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

    if (!item.isPOD) {
        // 析构函数
        item.destructor = [](void* ptr) { reinterpret_cast<T*>(ptr)->~T(); };
        // 拷贝构造
        item.copyConstruct = [](void* dest, const void* src) {
            new (dest) T(*reinterpret_cast<const T*>(src));
        };
        if constexpr (std::is_copy_assignable_v<T>) {
            item.copyAssign = [](void* dest, const void* src) {
                *reinterpret_cast<T*>(dest) = *reinterpret_cast<const T*>(src);
            };
        }
        if constexpr (std::is_nothrow_swappable_v<T>) {
            item.swap = [](void* left, void* right) noexcept {
                using std::swap;
                swap(*reinterpret_cast<T*>(left), *reinterpret_cast<T*>(right));
            };
        }
        // 默认构造
        if constexpr (std::is_default_constructible_v<T>) {
            item.defaultConstruct = [](void* dest) { new (dest) T(); };
        }
    }

    registry[typeid(T).name()] = std::move(item);
}

/// @brief registerType overload that accepts an explicit formatter function
template <typename T>
void registerType(const std::string& typeName, std::function<std::string(const void*)> formatter) {
    validateRegisteredTypeAlignment<T>();
    std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
    validateRegisteredTypeName<T>(typeName);
    auto& registry = yt::type::internal::getMutableTypeRegistry();
    int32_t typeSize = getTypeSize<T>();
    if (!formatter) {
        if constexpr (yt::utils::HAVE_OSTREAM<T>) {
            formatter = [](const void* data) {
                std::ostringstream oss;
                const T* p = reinterpret_cast<const T*>(data);
                oss << *p;
                return oss.str();
            };
        } else {
            throw std::invalid_argument("Formatter function cannot be null for type without ostream support."
            );
        }
    }

    // 非POD类型支持
    yt::type::TypeRegItem item;
    item.name = typeName;
    item.size = typeSize;
    item.toString = formatter;
    item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;

    if (!item.isPOD) {
        item.destructor = [](void* ptr) { reinterpret_cast<T*>(ptr)->~T(); };
        item.copyConstruct = [](void* dest, const void* src) {
            new (dest) T(*reinterpret_cast<const T*>(src));
        };
        if constexpr (std::is_copy_assignable_v<T>) {
            item.copyAssign = [](void* dest, const void* src) {
                *reinterpret_cast<T*>(dest) = *reinterpret_cast<const T*>(src);
            };
        }
        if constexpr (std::is_nothrow_swappable_v<T>) {
            item.swap = [](void* left, void* right) noexcept {
                using std::swap;
                swap(*reinterpret_cast<T*>(left), *reinterpret_cast<T*>(right));
            };
        }
        if constexpr (std::is_default_constructible_v<T>) {
            item.defaultConstruct = [](void* dest) { new (dest) T(); };
        }
    }

    registry[typeid(T).name()] = std::move(item);
}

/// @brief registerType overload that accepts serialize/deserialize functions for IO support
/// @tparam T 要注册的类型
/// @param typeName 自定义类型名称
/// @param formatter 格式化函数（用于打印）
/// @param serialize 序列化函数：将对象转换为字节数组
/// @param deserialize 反序列化函数：从字节数组恢复对象
template <typename T>
void registerType(
    const std::string& typeName, std::function<std::string(const void*)> formatter,
    std::function<std::vector<char>(const void*)> serialize,
    std::function<void(void*, const char*, size_t)> deserialize
) {
    validateRegisteredTypeAlignment<T>();
    std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
    validateRegisteredTypeName<T>(typeName);
    auto& registry = yt::type::internal::getMutableTypeRegistry();
    int32_t typeSize = getTypeSize<T>();

    if (!formatter) {
        if constexpr (yt::utils::HAVE_OSTREAM<T>) {
            formatter = [](const void* data) {
                std::ostringstream oss;
                const T* p = reinterpret_cast<const T*>(data);
                oss << *p;
                return oss.str();
            };
        }
    }

    yt::type::TypeRegItem item;
    item.name = typeName;
    item.size = typeSize;
    item.toString = formatter;
    item.isPOD = std::is_trivially_destructible_v<T> && std::is_trivially_copyable_v<T>;
    item.serialize = serialize;
    item.deserialize = deserialize;

    if (!item.isPOD) {
        item.destructor = [](void* ptr) { reinterpret_cast<T*>(ptr)->~T(); };
        item.copyConstruct = [](void* dest, const void* src) {
            new (dest) T(*reinterpret_cast<const T*>(src));
        };
        if constexpr (std::is_copy_assignable_v<T>) {
            item.copyAssign = [](void* dest, const void* src) {
                *reinterpret_cast<T*>(dest) = *reinterpret_cast<const T*>(src);
            };
        }
        if constexpr (std::is_nothrow_swappable_v<T>) {
            item.swap = [](void* left, void* right) noexcept {
                using std::swap;
                swap(*reinterpret_cast<T*>(left), *reinterpret_cast<T*>(right));
            };
        }
        if constexpr (std::is_default_constructible_v<T>) {
            item.defaultConstruct = [](void* dest) { new (dest) T(); };
        }
    }

    registry[typeid(T).name()] = std::move(item);
}

/// @brief 将任意 dtype 的单个元素（原始数据指针）格式化为字符串，用于打印
/// @param data 指向元素起始位置的原始指针
/// @param dtype 元素类型名称（如 "float32"）
/// @return 返回格式化后的字符串
inline std::string formatValue(const void* data, const std::string& dtype) {
    if (!data) return std::string("null");
    std::ostringstream oss;
    // use default formatting; decide casting based on dtype
    if (dtype == "float32") {
        const float* p = reinterpret_cast<const float*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int32") {
        const int32_t* p = reinterpret_cast<const int32_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int64") {
        const int64_t* p = reinterpret_cast<const int64_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int16") {
        const int16_t* p = reinterpret_cast<const int16_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "int8") {
        const int8_t* p = reinterpret_cast<const int8_t*>(data);
        // print numeric, not character
        oss << static_cast<int>(*p);
        return oss.str();
    } else if (dtype == "uint8") {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        oss << static_cast<unsigned int>(*p);
        return oss.str();
    } else if (dtype == "uint16") {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "uint32") {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "bool") {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "string") {
        const std::string* p = reinterpret_cast<const std::string*>(data);
        oss << *p;
        return oss.str();
    } else if (dtype == "bfloat16") {
        const yt::bfloat16* p = reinterpret_cast<const yt::bfloat16*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float16") {
        const yt::float16* p = reinterpret_cast<const yt::float16*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e5m2") {
        const yt::float8_e5m2* p = reinterpret_cast<const yt::float8_e5m2*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e4m3") {
        const yt::float8_e4m3* p = reinterpret_cast<const yt::float8_e4m3*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    } else if (dtype == "float8_e8m0" || dtype == "float8_ue8m0") {
        const yt::float8_e8m0* p = reinterpret_cast<const yt::float8_e8m0*>(data);
        oss << static_cast<float>(*p);
        return oss.str();
    }
    // 查看自定义注册类型
    std::function<std::string(const void*)> formatter;
    {
        std::lock_guard<std::mutex> lock(yt::type::getTypeRegistryMutex());
        const auto& registry = yt::type::internal::getMutableTypeRegistry();
        for (auto& [key, value] : registry) {
            if (value.name == dtype) {
                formatter = value.toString;
                break;
            }
        }
    }
    // 用户formatter可能重入type API；复制回调后释放registry mutex再调用，避免死锁。
    if (formatter) return formatter(data);
    // fallback打印单字节
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
    oss << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(*bytes);
    return oss.str();
}
}  // namespace yt::type
