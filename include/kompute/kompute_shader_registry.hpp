#pragma once

#include <string>
#include <vector>

namespace yt::kompute {

struct ShaderBinding {
    std::string functionName;
    std::string shaderFile;
    std::string entryPoint;
};

const std::vector<ShaderBinding>& ytensorBaseMathShaderBindings();
bool hasYTensorBaseMathShader(const std::string& functionName);

} // namespace yt::kompute

#include "../../src/kompute/kompute_shader_registry.inl"
