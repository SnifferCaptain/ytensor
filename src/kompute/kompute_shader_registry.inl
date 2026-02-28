#include <algorithm>

namespace yt::kompute {

inline const std::vector<ShaderBinding>& ytensorBaseMathShaderBindings() {
    static const std::vector<ShaderBinding> bindings = {
        {"broadcastInplace", "shaders/kompute/broadcast_inplace.comp.hlsl", "main"},
        {"operator+", "shaders/kompute/binary_arith_float.comp.hlsl", "fadd_main"},
        {"operator+=", "shaders/kompute/binary_arith_float.comp.hlsl", "fadd_main"},
        {"operator-", "shaders/kompute/binary_arith_float.comp.hlsl", "fsub_main"},
        {"operator-=", "shaders/kompute/binary_arith_float.comp.hlsl", "fsub_main"},
        {"operator*", "shaders/kompute/binary_arith_float.comp.hlsl", "fmul_main"},
        {"operator*=", "shaders/kompute/binary_arith_float.comp.hlsl", "fmul_main"},
        {"operator/", "shaders/kompute/binary_arith_float.comp.hlsl", "fdiv_main"},
        {"operator/=", "shaders/kompute/binary_arith_float.comp.hlsl", "fdiv_main"},
        {"operator%(int)", "shaders/kompute/binary_arith.comp.hlsl", "mod_main"},
        {"operator%=(int)", "shaders/kompute/binary_arith.comp.hlsl", "mod_main"},
        {"operator&(int)", "shaders/kompute/binary_arith.comp.hlsl", "and_main"},
        {"operator&=(int)", "shaders/kompute/binary_arith.comp.hlsl", "and_main"},
        {"operator|(int)", "shaders/kompute/binary_arith.comp.hlsl", "or_main"},
        {"operator|=(int)", "shaders/kompute/binary_arith.comp.hlsl", "or_main"},
        {"operator^(int)", "shaders/kompute/binary_arith.comp.hlsl", "xor_main"},
        {"operator^=(int)", "shaders/kompute/binary_arith.comp.hlsl", "xor_main"},
        {"operator<<(int)", "shaders/kompute/binary_arith.comp.hlsl", "shl_main"},
        {"operator<<=(int)", "shaders/kompute/binary_arith.comp.hlsl", "shl_main"},
        {"operator>>(int)", "shaders/kompute/binary_arith.comp.hlsl", "shr_main"},
        {"operator>>=(int)", "shaders/kompute/binary_arith.comp.hlsl", "shr_main"},
        {"operator<", "shaders/kompute/binary_cmp.comp.hlsl", "lt_main"},
        {"operator<=", "shaders/kompute/binary_cmp.comp.hlsl", "le_main"},
        {"operator>", "shaders/kompute/binary_cmp.comp.hlsl", "gt_main"},
        {"operator>=", "shaders/kompute/binary_cmp.comp.hlsl", "ge_main"},
        {"operator==", "shaders/kompute/binary_cmp.comp.hlsl", "eq_main"},
        {"operator!=", "shaders/kompute/binary_cmp.comp.hlsl", "ne_main"},
        {"matmul", "shaders/kompute/matmul.comp.hlsl", "main"},
        {"matView", "shaders/kompute/matview.comp.hlsl", "main"},
        {"sum(int)", "shaders/kompute/reduction_sum.comp.hlsl", "main"},
        {"sum(vector)", "shaders/kompute/reduction_sum.comp.hlsl", "main"},
        {"max(int)", "shaders/kompute/reduction_max.comp.hlsl", "main"},
        {"max(vector)", "shaders/kompute/reduction_max.comp.hlsl", "main"},
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
