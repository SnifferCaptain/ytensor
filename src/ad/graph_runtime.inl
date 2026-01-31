/***************
* @file: graph_runtime.inl
* @brief: 计算图运行时的实现
***************/

#include <algorithm>
#include <queue>
#include <sstream>

namespace yt{
namespace ad{

inline std::shared_ptr<GraphNode> ComputationGraph::createNode(
    const std::string& nodeId, const std::string& opType, NodeType nodeType) {
    
    if (nodes_.find(nodeId) != nodes_.end()) {
        throw std::runtime_error("Node with ID '" + nodeId + "' already exists");
    }
    
    auto node = std::make_shared<GraphNode>(nodeId, opType, nodeType);
    nodes_[nodeId] = node;
    return node;
}

inline std::shared_ptr<GraphEdge> ComputationGraph::createEdge(
    const std::string& edgeId,
    std::shared_ptr<GraphNode> fromNode,
    std::shared_ptr<GraphNode> toNode) {
    
    if (edges_.find(edgeId) != edges_.end()) {
        throw std::runtime_error("Edge with ID '" + edgeId + "' already exists");
    }
    
    auto edge = std::make_shared<GraphEdge>(edgeId, fromNode, toNode);
    edges_[edgeId] = edge;
    
    // 更新节点的输入输出边列表
    if (fromNode) {
        fromNode->addOutputEdge(edge);
    }
    if (toNode) {
        toNode->addInputEdge(edge);
    }
    
    return edge;
}

inline std::shared_ptr<GraphEdge> ComputationGraph::connect(
    const std::string& fromNodeId,
    const std::string& toNodeId,
    const std::string& edgeId) {
    
    auto fromNode = getNode(fromNodeId);
    auto toNode = getNode(toNodeId);
    
    if (!fromNode) {
        throw std::runtime_error("Source node '" + fromNodeId + "' not found");
    }
    if (!toNode) {
        throw std::runtime_error("Target node '" + toNodeId + "' not found");
    }
    
    std::string actualEdgeId = edgeId.empty() ? generateEdgeId() : edgeId;
    return createEdge(actualEdgeId, fromNode, toNode);
}

inline std::shared_ptr<GraphNode> ComputationGraph::getNode(const std::string& nodeId) const {
    auto it = nodes_.find(nodeId);
    return (it != nodes_.end()) ? it->second : nullptr;
}

inline std::shared_ptr<GraphEdge> ComputationGraph::getEdge(const std::string& edgeId) const {
    auto it = edges_.find(edgeId);
    return (it != edges_.end()) ? it->second : nullptr;
}

inline void ComputationGraph::registerOperator(const std::string& opType, OpExecutor executor) {
    operators_[opType] = executor;
}

inline std::unordered_map<std::string, YTensorBase> ComputationGraph::execute(
    const std::unordered_map<std::string, YTensorBase>& inputs,
    const std::vector<std::string>& outputs) {
    
    // 重置所有节点的执行状态
    reset();
    
    // 设置输入张量
    for (const auto& [edgeId, tensor] : inputs) {
        auto edge = getEdge(edgeId);
        if (!edge) {
            throw std::runtime_error("Input edge '" + edgeId + "' not found");
        }
        edge->setTensor(tensor);
    }
    
    // 获取执行顺序
    std::vector<std::string> execOrder = topologicalSort();
    
    // 按拓扑顺序执行节点
    for (const auto& nodeId : execOrder) {
        executeNode(nodeId);
    }
    
    // 收集输出张量
    std::unordered_map<std::string, YTensorBase> results;
    for (const auto& edgeId : outputs) {
        auto edge = getEdge(edgeId);
        if (!edge) {
            throw std::runtime_error("Output edge '" + edgeId + "' not found");
        }
        if (!edge->hasTensor()) {
            throw std::runtime_error("Output edge '" + edgeId + "' has no tensor data");
        }
        results[edgeId] = edge->getTensor();
    }
    
    return results;
}

inline void ComputationGraph::executeNode(const std::string& nodeId) {
    auto node = getNode(nodeId);
    if (!node) {
        throw std::runtime_error("Node '" + nodeId + "' not found");
    }
    
    if (node->isExecuted()) {
        return;  // 节点已执行，跳过
    }
    
    // 参数节点和常量节点不需要执行，直接标记为已执行
    if (node->getNodeType() == NodeType::Parameter || node->getNodeType() == NodeType::Constant) {
        // 将节点数据传递到输出边
        if (node->hasData() && !node->getOutputEdges().empty()) {
            for (auto& edge : node->getOutputEdges()) {
                edge->setTensor(node->getData());
            }
        }
        node->setExecuted(true);
        return;
    }
    
    // 输入节点：从输入边读取数据，传递到输出边
    if (node->getNodeType() == NodeType::Input) {
        auto inputEdges = node->getInputEdges();
        auto outputEdges = node->getOutputEdges();
        if (!inputEdges.empty() && !outputEdges.empty()) {
            if (inputEdges[0]->hasTensor()) {
                for (auto& edge : outputEdges) {
                    edge->setTensor(inputEdges[0]->getTensor());
                }
            }
        }
        node->setExecuted(true);
        return;
    }
    
    // 算子节点：执行计算
    auto it = operators_.find(node->getOpType());
    if (it == operators_.end()) {
        throw std::runtime_error("Operator '" + node->getOpType() + "' not registered");
    }
    
    // 获取输入输出边
    auto inputEdges = node->getInputEdges();
    auto outputEdges = node->getOutputEdges();
    
    // 检查输入是否就绪
    for (const auto& edge : inputEdges) {
        if (!edge->hasTensor()) {
            throw std::runtime_error("Input edge '" + edge->getId() + "' for node '" 
                                   + nodeId + "' has no tensor data");
        }
    }
    
    // 执行算子
    it->second(*node, inputEdges, outputEdges);
    
    // 标记节点已执行
    node->setExecuted(true);
}

inline std::vector<std::string> ComputationGraph::topologicalSort() const {
    std::vector<std::string> result;
    std::unordered_map<std::string, bool> visited;
    
    // 初始化访问标记
    for (const auto& [nodeId, _] : nodes_) {
        visited[nodeId] = false;
    }
    
    // 对每个未访问的节点执行DFS
    for (const auto& [nodeId, _] : nodes_) {
        if (!visited[nodeId]) {
            topologicalSortDFS(nodeId, visited, result);
        }
    }
    
    // 反转结果（DFS后序遍历需要反转才是拓扑序）
    std::reverse(result.begin(), result.end());
    return result;
}

inline void ComputationGraph::topologicalSortDFS(
    const std::string& nodeId,
    std::unordered_map<std::string, bool>& visited,
    std::vector<std::string>& stack) const {
    
    visited[nodeId] = true;
    
    auto node = getNode(nodeId);
    if (node) {
        // 访问所有输出边的目标节点
        for (const auto& edge : node->getOutputEdges()) {
            auto toNode = edge->getToNode();
            if (toNode) {
                const std::string& nextNodeId = toNode->getId();
                if (!visited[nextNodeId]) {
                    topologicalSortDFS(nextNodeId, visited, stack);
                }
            }
        }
    }
    
    // 将当前节点加入栈
    stack.push_back(nodeId);
}

inline void ComputationGraph::reset() {
    for (auto& [_, node] : nodes_) {
        node->reset();
    }
}

inline void ComputationGraph::clear() {
    nodes_.clear();
    edges_.clear();
    operators_.clear();
    edgeCounter_ = 0;
}

inline std::string ComputationGraph::generateEdgeId() {
    std::ostringstream oss;
    oss << "edge_" << edgeCounter_++;
    return oss.str();
}

inline bool ComputationGraph::hasCycleDFS(
    const std::string& nodeId,
    std::unordered_map<std::string, bool>& visited,
    std::unordered_map<std::string, bool>& recursionStack) const {
    
    visited[nodeId] = true;
    recursionStack[nodeId] = true;
    
    auto node = getNode(nodeId);
    if (node) {
        for (const auto& edge : node->getOutputEdges()) {
            auto toNode = edge->getToNode();
            if (toNode) {
                const std::string& nextNodeId = toNode->getId();
                
                if (!visited[nextNodeId]) {
                    if (hasCycleDFS(nextNodeId, visited, recursionStack)) {
                        return true;
                    }
                } else if (recursionStack[nextNodeId]) {
                    return true;  // 发现环
                }
            }
        }
    }
    
    recursionStack[nodeId] = false;
    return false;
}

} // namespace ad
} // namespace yt
