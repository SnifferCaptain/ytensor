#pragma once
/***************
* @file: graph_edge.hpp
* @brief: 计算图边定义，表示计算图中的数据流
* @description: 边代表有向的数据流，连接源节点和目标节点，携带张量数据
***************/

#include <string>
#include <memory>
#include "../ytensor_base.hpp"

namespace yt{
namespace ad{

// 前向声明
class GraphNode;

/// @brief 计算图边类，表示节点之间的有向数据流
class GraphEdge {
public:
    /// @brief 默认构造函数
    GraphEdge() = default;

    /// @brief 构造一个计算图边
    /// @param edgeId 边的唯一标识符
    /// @param fromNode 源节点（可为nullptr，表示输入边）
    /// @param toNode 目标节点（可为nullptr，表示输出边）
    GraphEdge(const std::string& edgeId, 
              std::shared_ptr<GraphNode> fromNode = nullptr,
              std::shared_ptr<GraphNode> toNode = nullptr);

    /// @brief 获取边ID
    /// @return 边的唯一标识符
    std::string getId() const { return edgeId_; }

    /// @brief 设置边名称（可选）
    /// @param name 边名称
    void setName(const std::string& name) { name_ = name; }

    /// @brief 获取边名称
    /// @return 边名称
    std::string getName() const { return name_; }

    /// @brief 设置源节点
    /// @param node 源节点指针
    void setFromNode(std::shared_ptr<GraphNode> node) { fromNode_ = node; }

    /// @brief 设置目标节点
    /// @param node 目标节点指针
    void setToNode(std::shared_ptr<GraphNode> node) { toNode_ = node; }

    /// @brief 获取源节点
    /// @return 源节点指针
    std::shared_ptr<GraphNode> getFromNode() const { return fromNode_; }

    /// @brief 获取目标节点
    /// @return 目标节点指针
    std::shared_ptr<GraphNode> getToNode() const { return toNode_; }

    /// @brief 设置边上的张量数据
    /// @param tensor 张量数据
    void setTensor(const YTensorBase& tensor) { tensor_ = tensor; }

    /// @brief 获取边上的张量数据
    /// @return 张量数据的引用
    YTensorBase& getTensor() { return tensor_; }

    /// @brief 获取边上的张量数据（const版本）
    /// @return 张量数据的const引用
    const YTensorBase& getTensor() const { return tensor_; }

    /// @brief 检查边是否包含有效的张量数据
    /// @return 如果张量有效返回true
    bool hasTensor() const { return tensor_.size() > 0; }

    /// @brief 检查是否为输入边（没有源节点）
    /// @return 如果是输入边返回true
    bool isInputEdge() const { return fromNode_ == nullptr; }

    /// @brief 检查是否为输出边（没有目标节点）
    /// @return 如果是输出边返回true
    bool isOutputEdge() const { return toNode_ == nullptr; }

private:
    std::string edgeId_;                    // 边的唯一标识符
    std::string name_;                      // 边名称（可选）
    std::shared_ptr<GraphNode> fromNode_;   // 源节点
    std::shared_ptr<GraphNode> toNode_;     // 目标节点
    YTensorBase tensor_;                    // 边上携带的张量数据
};

} // namespace ad
} // namespace yt
