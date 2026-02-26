#include <algorithm>

namespace yt::kompute {

inline const std::vector<ShaderBinding>& ytensorBaseMathShaderBindings() {
    static const std::vector<ShaderBinding> bindings = {
        {"broadcastInplace", "include/kompute/shaders/broadcast_inplace.comp.hlsl", "main"},
        {"operator+", "include/kompute/shaders/binary_arith.comp.hlsl", "add_main"},
        {"operator+=", "include/kompute/shaders/binary_arith.comp.hlsl", "add_main"},
        {"operator-", "include/kompute/shaders/binary_arith.comp.hlsl", "sub_main"},
        {"operator-=", "include/kompute/shaders/binary_arith.comp.hlsl", "sub_main"},
        {"operator*", "include/kompute/shaders/binary_arith.comp.hlsl", "mul_main"},
        {"operator*=", "include/kompute/shaders/binary_arith.comp.hlsl", "mul_main"},
        {"operator/", "include/kompute/shaders/binary_arith.comp.hlsl", "div_main"},
        {"operator/=", "include/kompute/shaders/binary_arith.comp.hlsl", "div_main"},
        {"operator%", "include/kompute/shaders/binary_arith.comp.hlsl", "mod_main"},
        {"operator%=", "include/kompute/shaders/binary_arith.comp.hlsl", "mod_main"},
        {"operator&", "include/kompute/shaders/binary_arith.comp.hlsl", "and_main"},
        {"operator&=", "include/kompute/shaders/binary_arith.comp.hlsl", "and_main"},
        {"operator|", "include/kompute/shaders/binary_arith.comp.hlsl", "or_main"},
        {"operator|=", "include/kompute/shaders/binary_arith.comp.hlsl", "or_main"},
        {"operator^", "include/kompute/shaders/binary_arith.comp.hlsl", "xor_main"},
        {"operator^=", "include/kompute/shaders/binary_arith.comp.hlsl", "xor_main"},
        {"operator<<", "include/kompute/shaders/binary_arith.comp.hlsl", "shl_main"},
        {"operator<<=", "include/kompute/shaders/binary_arith.comp.hlsl", "shl_main"},
        {"operator>>", "include/kompute/shaders/binary_arith.comp.hlsl", "shr_main"},
        {"operator>>=", "include/kompute/shaders/binary_arith.comp.hlsl", "shr_main"},
        {"operator<", "include/kompute/shaders/binary_cmp.comp.hlsl", "lt_main"},
        {"operator<=", "include/kompute/shaders/binary_cmp.comp.hlsl", "le_main"},
        {"operator>", "include/kompute/shaders/binary_cmp.comp.hlsl", "gt_main"},
        {"operator>=", "include/kompute/shaders/binary_cmp.comp.hlsl", "ge_main"},
        {"operator==", "include/kompute/shaders/binary_cmp.comp.hlsl", "eq_main"},
        {"operator!=", "include/kompute/shaders/binary_cmp.comp.hlsl", "ne_main"},
        {"matmul", "include/kompute/shaders/matmul.comp.hlsl", "main"},
        {"matView", "include/kompute/shaders/matview.comp.hlsl", "main"},
        {"sum(int)", "include/kompute/shaders/reduction_sum.comp.hlsl", "main"},
        {"sum(vector)", "include/kompute/shaders/reduction_sum.comp.hlsl", "main"},
        {"max(int)", "include/kompute/shaders/reduction_max.comp.hlsl", "main"},
        {"max(vector)", "include/kompute/shaders/reduction_max.comp.hlsl", "main"},
    };
    return bindings;
}

inline bool hasYTensorBaseMathShader(const std::string& functionName) {
    const auto& bindings = ytensorBaseMathShaderBindings();
    return std::any_of(bindings.begin(), bindings.end(), [&](const ShaderBinding& item) {
        return item.functionName == functionName;
    });
}

} // namespace yt::kompute
