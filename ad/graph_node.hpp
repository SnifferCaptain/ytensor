#pragma once
/***************
* @file: graph_node.hpp
* @brief: 计算图节点类，表示算子
***************/
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "../ytensor.hpp"

namespace yt {
namespace ad {

// 节点类型枚举
enum class NodeType {
    // 基础操作
    Input,          // 输入节点
    Parameter,      // 参数节点
    Constant,       // 常量节点
    
    // 线性层
    Linear,         // 线性变换 (矩阵乘法)
    Embedding,      // 词嵌入
    
    // 归一化
    RMSNorm,        // RMS归一化
    
    // 激活函数
    GELU,           // GELU激活
    
    // 注意力机制
    Attention,      // 注意力层 (包括RoPE等)
    
    // 前馈网络
    FFN,            // 前馈神经网络
    
    // 基本算术
    Add,            // 加法
    Multiply,       // 乘法
    
    // 张量操作
    Slice,          // 切片
    Reshape,        // 重塑
    Permute,        // 维度置换
    Concat,         // 拼接
    Squeeze,        // 压缩维度
    Unsqueeze,      // 扩展维度
    
    // 其他
    Output,         // 输出节点
    Custom          // 自定义操作
};

// 将NodeType转换为字符串
inline std::string nodeTypeToString(NodeType type) {
    static const std::unordered_map<NodeType, std::string> typeMap = {
        {NodeType::Input, "Input"},
        {NodeType::Parameter, "Parameter"},
        {NodeType::Constant, "Constant"},
        {NodeType::Linear, "Linear"},
        {NodeType::Embedding, "Embedding"},
        {NodeType::RMSNorm, "RMSNorm"},
        {NodeType::GELU, "GELU"},
        {NodeType::Attention, "Attention"},
        {NodeType::FFN, "FFN"},
        {NodeType::Add, "Add"},
        {NodeType::Multiply, "Multiply"},
        {NodeType::Slice, "Slice"},
        {NodeType::Reshape, "Reshape"},
        {NodeType::Permute, "Permute"},
        {NodeType::Concat, "Concat"},
        {NodeType::Squeeze, "Squeeze"},
        {NodeType::Unsqueeze, "Unsqueeze"},
        {NodeType::Output, "Output"},
        {NodeType::Custom, "Custom"}
    };
    auto it = typeMap.find(type);
    return (it != typeMap.end()) ? it->second : "Unknown";
}

// 从字符串转换为NodeType
inline NodeType stringToNodeType(const std::string& str) {
    static const std::unordered_map<std::string, NodeType> typeMap = {
        {"Input", NodeType::Input},
        {"Parameter", NodeType::Parameter},
        {"Constant", NodeType::Constant},
        {"Linear", NodeType::Linear},
        {"Embedding", NodeType::Embedding},
        {"RMSNorm", NodeType::RMSNorm},
        {"GELU", NodeType::GELU},
        {"Attention", NodeType::Attention},
        {"FFN", NodeType::FFN},
        {"Add", NodeType::Add},
        {"Multiply", NodeType::Multiply},
        {"Slice", NodeType::Slice},
        {"Reshape", NodeType::Reshape},
        {"Permute", NodeType::Permute},
        {"Concat", NodeType::Concat},
        {"Squeeze", NodeType::Squeeze},
        {"Unsqueeze", NodeType::Unsqueeze},
        {"Output", NodeType::Output},
        {"Custom", NodeType::Custom}
    };
    auto it = typeMap.find(str);
    return (it != typeMap.end()) ? it->second : NodeType::Custom;
}

// 计算图节点
class Node {
public:
    // 构造函数
    Node(const std::string& name, NodeType type)
        : name_(name), type_(type), id_(nextId_++) {}
    
    // 获取节点名称
    const std::string& name() const { return name_; }
    
    // 获取节点类型
    NodeType type() const { return type_; }
    
    // 获取节点ID
    int id() const { return id_; }
    
    // 添加输入边（来自其他节点）
    void addInput(int nodeId) {
        inputs_.push_back(nodeId);
    }
    
    // 添加输出边（指向其他节点）
    void addOutput(int nodeId) {
        outputs_.push_back(nodeId);
    }
    
    // 获取输入节点ID列表
    const std::vector<int>& inputs() const { return inputs_; }
    
    // 获取输出节点ID列表
    const std::vector<int>& outputs() const { return outputs_; }
    
    // 设置/获取节点属性
    void setAttribute(const std::string& key, const std::string& value) {
        attributes_[key] = value;
    }
    
    std::string getAttribute(const std::string& key, const std::string& defaultValue = "") const {
        auto it = attributes_.find(key);
        return (it != attributes_.end()) ? it->second : defaultValue;
    }
    
    const std::unordered_map<std::string, std::string>& attributes() const {
        return attributes_;
    }
    
    // 设置节点的数据（用于参数节点）
    void setData(const YTensorBase& data) {
        data_ = std::make_shared<YTensorBase>(data);
    }
    
    // 获取节点的数据
    std::shared_ptr<YTensorBase> data() const { return data_; }
    
private:
    std::string name_;                                  // 节点名称
    NodeType type_;                                     // 节点类型
    int id_;                                            // 节点唯一ID
    std::vector<int> inputs_;                           // 输入节点ID列表
    std::vector<int> outputs_;                          // 输出节点ID列表
    std::unordered_map<std::string, std::string> attributes_;  // 节点属性
    std::shared_ptr<YTensorBase> data_;                 // 节点数据（用于参数）
    
    static int nextId_;                                 // 全局节点ID计数器
};

} // namespace ad
} // namespace yt
