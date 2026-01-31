#pragma once
/***************
* @file: operator.hpp
* @brief: 算子注册系统 - 支持前向和反向传播
***************/

#include "../../ytensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace yt {
namespace ad {

// 前向声明
class OpContext;
class Operator;

// 算子上下文 - 存储前向传播中需要保存的中间结果（用于反向传播）
class OpContext {
public:
    std::unordered_map<std::string, std::shared_ptr<YTensorBase>> saved_tensors;
    std::unordered_map<std::string, std::string> attrs;
    
    void save(const std::string& name, const std::shared_ptr<YTensorBase>& tensor) {
        saved_tensors[name] = tensor;
    }
    
    std::shared_ptr<YTensorBase> get(const std::string& name) const {
        auto it = saved_tensors.find(name);
        if (it != saved_tensors.end()) {
            return it->second;
        }
        return nullptr;
    }
};

// 算子基类
class Operator {
public:
    std::string name;
    
    Operator(const std::string& name_) : name(name_) {}
    virtual ~Operator() = default;
    
    // 前向传播
    // inputs: 输入张量列表
    // attrs: 算子属性（如维度、epsilon等）
    // ctx: 上下文，用于保存反向传播需要的中间结果
    // 返回：输出张量列表
    virtual std::vector<std::shared_ptr<YTensorBase>> forward(
        const std::vector<std::shared_ptr<YTensorBase>>& inputs,
        const std::unordered_map<std::string, std::string>& attrs,
        OpContext* ctx
    ) = 0;
    
    // 反向传播（可选）
    // grad_outputs: 输出梯度
    // ctx: 前向传播保存的上下文
    // 返回：输入梯度列表
    virtual std::vector<std::shared_ptr<YTensorBase>> backward(
        const std::vector<std::shared_ptr<YTensorBase>>& grad_outputs,
        OpContext* ctx
    ) {
        // 默认实现：不支持反向传播
        throw std::runtime_error("Backward not implemented for operator: " + name);
    }
    
    // 是否支持反向传播
    virtual bool supports_backward() const { return false; }
};

// 算子注册表
class OpRegistry {
public:
    using OpCreator = std::function<std::shared_ptr<Operator>()>;
    
    static OpRegistry& instance() {
        static OpRegistry registry;
        return registry;
    }
    
    // 注册算子
    void registerOp(const std::string& name, OpCreator creator) {
        creators_[name] = creator;
    }
    
    // 创建算子实例
    std::shared_ptr<Operator> create(const std::string& name) {
        auto it = creators_.find(name);
        if (it != creators_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    // 检查算子是否已注册
    bool has(const std::string& name) const {
        return creators_.find(name) != creators_.end();
    }
    
private:
    std::unordered_map<std::string, OpCreator> creators_;
    OpRegistry() = default;
};

// 注册辅助宏
#define REGISTER_OPERATOR(name, OpClass) \
    namespace { \
        struct OpClass##Registrar { \
            OpClass##Registrar() { \
                OpRegistry::instance().registerOp(name, []() -> std::shared_ptr<Operator> { \
                    return std::make_shared<OpClass>(); \
                }); \
            } \
        }; \
        static OpClass##Registrar global_##OpClass##_registrar; \
    }

} // namespace ad
} // namespace yt
