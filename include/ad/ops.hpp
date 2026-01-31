#pragma once
/***************
* @file: ops.hpp
* @brief: 具体算子实现
***************/

#include "operator.hpp"
#include "../../ytensor.hpp"
#include <cmath>

namespace yt {
namespace ad {
namespace ops {

// ==================== Linear 算子 ====================
class LinearOp : public Operator {
public:
    LinearOp() : Operator("Linear") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        // inputs: [input, weight] 或 [input, weight, bias]
        if (inputs.size() < 2) {
            throw std::runtime_error("Linear requires at least 2 inputs");
        }
        
        auto& input = inputs[0];
        auto& weight = inputs[1];
        
        // 执行 input @ weight.T
        // YTensor的matmul默认是行主序，weight需要转置
        auto output = std::make_shared<YTensorBase>(input->matmul(weight->transpose()));
        
        // 如果有bias，加上
        if (inputs.size() >= 3) {
            auto& bias = inputs[2];
            *output = *output + *bias;
        }
        
        // 保存用于反向传播
        if (ctx) {
            ctx->save("input", input);
            ctx->save("weight", weight);
        }
        
        return {output};
    }
    
    bool supports_backward() const override { return true; }
    
    std::vector<std::shared_ptr<YTensorBase>> backward(
        const std::vector<std::shared_ptr<YTensorBase>>& grad_outputs,
        OpContext* ctx
    ) override {
        auto grad_output = grad_outputs[0];
        auto input = ctx->get("input");
        auto weight = ctx->get("weight");
        
        // grad_input = grad_output @ weight
        auto grad_input = std::make_shared<YTensorBase>(grad_output->matmul(*weight));
        
        // grad_weight = grad_output.T @ input
        auto grad_weight = std::make_shared<YTensorBase>(grad_output->transpose().matmul(*input));
        
        return {grad_input, grad_weight};
    }
};

// ==================== RMSNorm 算子 ====================
class RMSNormOp : public Operator {
public:
    RMSNormOp() : Operator("RMSNorm") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        // inputs: [input, weight]
        if (inputs.size() < 2) {
            throw std::runtime_error("RMSNorm requires 2 inputs");
        }
        
        auto& x = inputs[0];
        auto& weight = inputs[1];
        
        float eps = 1e-8f;
        if (attrs.count("eps")) {
            eps = std::stof(attrs.at("eps"));
        }
        
        // RMS归一化：x / sqrt(mean(x^2) + eps) * weight
        auto output = std::make_shared<YTensorBase>(x->clone());
        
        // 简化实现：使用广播操作
        // 完整实现需要按最后一维计算RMS
        // TODO: 完整实现
        
        return {output};
    }
};

// ==================== Add 算子 ====================
class AddOp : public Operator {
public:
    AddOp() : Operator("Add") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        if (inputs.size() < 2) {
            throw std::runtime_error("Add requires at least 2 inputs");
        }
        
        auto result = std::make_shared<YTensorBase>(*inputs[0] + *inputs[1]);
        
        // 如果有更多输入，继续相加
        for (size_t i = 2; i < inputs.size(); ++i) {
            *result = *result + *inputs[i];
        }
        
        return {result};
    }
    
    bool supports_backward() const override { return true; }
    
    std::vector<std::shared_ptr<YTensorBase>> backward(
        const std::vector<std::shared_ptr<YTensorBase>>& grad_outputs,
        OpContext* ctx
    ) override {
        // 加法的梯度直接传递
        auto grad = grad_outputs[0];
        std::vector<std::shared_ptr<YTensorBase>> grads;
        for (size_t i = 0; i < 2; ++i) {  // 假设2个输入
            grads.push_back(grad);
        }
        return grads;
    }
};

// ==================== Multiply 算子 ====================
class MultiplyOp : public Operator {
public:
    MultiplyOp() : Operator("Multiply") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        if (inputs.size() < 2) {
            throw std::runtime_error("Multiply requires at least 2 inputs");
        }
        
        if (ctx) {
            ctx->save("input0", inputs[0]);
            ctx->save("input1", inputs[1]);
        }
        
        auto result = std::make_shared<YTensorBase>(*inputs[0] * *inputs[1]);
        return {result};
    }
    
    bool supports_backward() const override { return true; }
    
    std::vector<std::shared_ptr<YTensorBase>> backward(
        const std::vector<std::shared_ptr<YTensorBase>>& grad_outputs,
        OpContext* ctx
    ) override {
        auto grad_output = grad_outputs[0];
        auto input0 = ctx->get("input0");
        auto input1 = ctx->get("input1");
        
        // d(a*b)/da = b, d(a*b)/db = a
        auto grad_input0 = std::make_shared<YTensorBase>(*grad_output * *input1);
        auto grad_input1 = std::make_shared<YTensorBase>(*grad_output * *input0);
        
        return {grad_input0, grad_input1};
    }
};

// ==================== GELU 算子 ====================
class GELUOp : public Operator {
public:
    GELUOp() : Operator("GELU") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        if (inputs.empty()) {
            throw std::runtime_error("GELU requires 1 input");
        }
        
        auto& x = inputs[0];
        auto output = std::make_shared<YTensorBase>(x->clone());
        
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // 简化实现
        // TODO: 完整实现
        
        if (ctx) {
            ctx->save("input", x);
        }
        
        return {output};
    }
};

// ==================== Embedding 算子 ====================
class EmbeddingOp : public Operator {
public:
    EmbeddingOp() : Operator("Embedding") {}
    
    std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) override {
        // inputs: [indices, weight]
        if (inputs.size() < 2) {
            throw std::runtime_error("Embedding requires 2 inputs");
        }
        
        // 简化实现
        // TODO: 实际的embedding查找
        
        auto output = std::make_shared<YTensorBase>(inputs[1]->clone());
        return {output};
    }
};

} // namespace ops
} // namespace ad
} // namespace yt
