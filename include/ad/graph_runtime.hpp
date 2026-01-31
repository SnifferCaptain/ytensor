#pragma once
/***************
* @file: graph_runtime.hpp
* @brief: 计算图运行时，负责计算图的构建、执行和管理
* @description: 运行时引擎管理计算图的生命周期，提供节点和边的创建、执行、序列化等功能
***************/

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>

#include "graph_node.hpp"
#include "graph_edge.hpp"
#include "../ytensor_base.hpp"

namespace yt {
namespace ad {

// 算子执行函数类型定义
// 输入：节点指针，输入边列表，输出边列表
// 输出：void（结果写入输出边）
using OpExecutor = std::function<void(
    const GraphNode&,
    const std::vector<std::shared_ptr<GraphEdge>>&,
    std::vector<std::shared_ptr<GraphEdge>>&
)>;

/// @brief 计算图运行时类
class ComputationGraph {
public:
    /// @brief 默认构造函数
    ComputationGraph() = default;

    /// @brief 创建一个新节点
    /// @param nodeId 节点ID（必须唯一）
    /// @param opType 算子类型或节点描述
    /// @param nodeType 节点类型（默认为Operator）
    /// @return 节点指针
    std::shared_ptr<GraphNode> createNode(const std::string& nodeId, const std::string& opType, NodeType nodeType = NodeType::Operator);

    /// @brief 创建一个新边
    /// @param edgeId 边ID（必须唯一）
    /// @param fromNode 源节点（nullptr表示输入边）
    /// @param toNode 目标节点（nullptr表示输出边）
    /// @return 边指针
    std::shared_ptr<GraphEdge> createEdge(
        const std::string& edgeId,
        std::shared_ptr<GraphNode> fromNode = nullptr,
        std::shared_ptr<GraphNode> toNode = nullptr
    );

    /// @brief 连接两个节点
    /// @param fromNodeId 源节点ID
    /// @param toNodeId 目标节点ID
    /// @param edgeId 边ID（如果为空则自动生成）
    /// @return 创建的边指针
    std::shared_ptr<GraphEdge> connect(
        const std::string& fromNodeId,
        const std::string& toNodeId,
        const std::string& edgeId = ""
    );

    /// @brief 获取节点
    /// @param nodeId 节点ID
    /// @return 节点指针，如果不存在则返回nullptr
    std::shared_ptr<GraphNode> getNode(const std::string& nodeId) const;

    /// @brief 获取边
    /// @param edgeId 边ID
    /// @return 边指针，如果不存在则返回nullptr
    std::shared_ptr<GraphEdge> getEdge(const std::string& edgeId) const;

    /// @brief 获取所有节点
    /// @return 所有节点的map
    const std::unordered_map<std::string, std::shared_ptr<GraphNode>>& getNodes() const {
        return nodes_;
    }

    /// @brief 获取所有边
    /// @return 所有边的map
    const std::unordered_map<std::string, std::shared_ptr<GraphEdge>>& getEdges() const {
        return edges_;
    }

    /// @brief 注册算子执行器
    /// @param opType 算子类型
    /// @param executor 执行函数
    void registerOperator(const std::string& opType, OpExecutor executor);

    /// @brief 执行计算图
    /// @param inputs 输入张量映射（边ID -> 张量）
    /// @param outputs 输出边ID列表
    /// @return 输出张量映射（边ID -> 张量）
    std::unordered_map<std::string, YTensorBase> execute(
        const std::unordered_map<std::string, YTensorBase>& inputs,
        const std::vector<std::string>& outputs
    );

    /// @brief 执行单个节点
    /// @param nodeId 节点ID
    void executeNode(const std::string& nodeId);

    /// @brief 拓扑排序，返回执行顺序
    /// @return 节点ID的执行顺序
    std::vector<std::string> topologicalSort() const;

    /// @brief 重置所有节点的执行状态
    void reset();

    /// @brief 清空计算图
    void clear();

    /// @brief 获取计算图中节点的数量
    /// @return 节点数量
    size_t nodeCount() const { return nodes_.size(); }

    /// @brief 获取计算图中边的数量
    /// @return 边数量
    size_t edgeCount() const { return edges_.size(); }

    /// @brief 检查计算图是否为空
    /// @return 如果为空返回true
    bool empty() const { return nodes_.empty(); }

private:
    /// @brief 生成唯一的边ID
    /// @return 唯一的边ID
    std::string generateEdgeId();

    /// @brief 检查图是否有环（DFS辅助函数）
    /// @param nodeId 当前节点ID
    /// @param visited 访问标记
    /// @param recursionStack 递归栈标记
    /// @return 如果检测到环返回true
    bool hasCycleDFS(
        const std::string& nodeId,
        std::unordered_map<std::string, bool>& visited,
        std::unordered_map<std::string, bool>& recursionStack
    ) const;

    /// @brief 拓扑排序的DFS辅助函数
    /// @param nodeId 当前节点ID
    /// @param visited 访问标记
    /// @param stack 结果栈
    void topologicalSortDFS(
        const std::string& nodeId,
        std::unordered_map<std::string, bool>& visited,
        std::vector<std::string>& stack
    ) const;

private:
    std::unordered_map<std::string, std::shared_ptr<GraphNode>> nodes_;    // 节点映射表
    std::unordered_map<std::string, std::shared_ptr<GraphEdge>> edges_;    // 边映射表
    std::unordered_map<std::string, OpExecutor> operators_;                // 算子执行器映射表
    int edgeCounter_ = 0;                                                  // 边计数器，用于生成唯一ID
};

} // namespace ad
} // namespace yt
