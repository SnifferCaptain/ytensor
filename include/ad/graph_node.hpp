#pragma once
/***************
* @file: graph_node.hpp
* @brief: 计算图节点定义，表示计算图中的算子
* @description: 节点代表算子操作，包含操作类型、参数和连接关系
***************/

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <any>

namespace yt{
namespace ad{

// 前向声明
class GraphEdge;
class ComputationGraph;

/// @brief 节点类型枚举
enum class NodeType {
    Operator,      // 算子节点（如add, mul, matmul等）
    Parameter,     // 参数节点（如权重、偏置）
    Input,         // 输入节点
    Constant       // 常量节点
};

/// @brief 计算图节点类，表示计算图中的一个节点（算子、参数、输入或常量）
class GraphNode {
public:
    /// @brief 默认构造函数
    GraphNode() = default;

    /// @brief 构造一个计算图节点
    /// @param nodeId 节点的唯一标识符
    /// @param opType 算子类型（如 "add", "mul", "matmul", "relu" 等）或节点类型
    /// @param nodeType 节点类型（默认为Operator）
    GraphNode(const std::string& nodeId, const std::string& opType, NodeType nodeType = NodeType::Operator);

    /// @brief 获取节点ID
    /// @return 节点的唯一标识符
    std::string getId() const { return nodeId_; }

    /// @brief 获取算子类型或节点类型描述
    /// @return 算子类型字符串
    std::string getOpType() const { return opType_; }

    /// @brief 获取节点类型
    /// @return 节点类型
    NodeType getNodeType() const { return nodeType_; }

    /// @brief 设置节点名称（可选，用于调试和可视化）
    /// @param name 节点名称
    void setName(const std::string& name) { name_ = name; }

    /// @brief 获取节点名称
    /// @return 节点名称
    std::string getName() const { return name_; }

    /// @brief 添加输入边
    /// @param edge 指向输入边的指针
    void addInputEdge(std::shared_ptr<GraphEdge> edge);

    /// @brief 添加输出边
    /// @param edge 指向输出边的指针
    void addOutputEdge(std::shared_ptr<GraphEdge> edge);

    /// @brief 获取所有输入边
    /// @return 输入边的vector
    const std::vector<std::shared_ptr<GraphEdge>>& getInputEdges() const { return inputEdges_; }

    /// @brief 获取所有输出边
    /// @return 输出边的vector
    const std::vector<std::shared_ptr<GraphEdge>>& getOutputEdges() const { return outputEdges_; }

    /// @brief 设置节点参数
    /// @param key 参数名
    /// @param value 参数值（使用std::any以支持多种类型）
    void setParameter(const std::string& key, const std::any& value);

    /// @brief 获取节点参数
    /// @param key 参数名
    /// @return 参数值
    std::any getParameter(const std::string& key) const;

    /// @brief 检查是否存在某个参数
    /// @param key 参数名
    /// @return 如果参数存在返回true，否则返回false
    bool hasParameter(const std::string& key) const;

    /// @brief 获取所有参数
    /// @return 参数映射表
    const std::unordered_map<std::string, std::any>& getParameters() const { return parameters_; }

    /// @brief 标记节点是否已执行
    /// @param executed 是否已执行
    void setExecuted(bool executed) { executed_ = executed; }

    /// @brief 检查节点是否已执行
    /// @return 如果已执行返回true
    bool isExecuted() const { return executed_; }

    /// @brief 重置节点执行状态
    void reset() { executed_ = false; }

    /// @brief 设置节点数据（用于参数节点和常量节点）
    /// @param tensor 张量数据
    void setData(const YTensorBase& tensor) { data_ = tensor; hasData_ = true; }

    /// @brief 获取节点数据
    /// @return 张量数据的引用
    YTensorBase& getData() { return data_; }

    /// @brief 获取节点数据（const版本）
    /// @return 张量数据的const引用
    const YTensorBase& getData() const { return data_; }

    /// @brief 检查节点是否有数据
    /// @return 如果有数据返回true
    bool hasData() const { return hasData_; }

private:
    std::string nodeId_;                                      // 节点唯一标识符
    std::string opType_;                                      // 算子类型或节点类型描述
    std::string name_;                                        // 节点名称（可选）
    NodeType nodeType_ = NodeType::Operator;                  // 节点类型
    std::vector<std::shared_ptr<GraphEdge>> inputEdges_;     // 输入边列表
    std::vector<std::shared_ptr<GraphEdge>> outputEdges_;    // 输出边列表
    std::unordered_map<std::string, std::any> parameters_;   // 节点参数（用于算子配置）
    YTensorBase data_;                                        // 节点数据（用于参数和常量节点）
    bool hasData_ = false;                                    // 是否有数据
    bool executed_ = false;                                   // 是否已执行
};

} // namespace ad
} // namespace yt
