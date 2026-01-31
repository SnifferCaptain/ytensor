/***************
* @file: graph_node.inl
* @brief: 计算图节点的实现
***************/

namespace yt{
namespace ad{

inline GraphNode::GraphNode(const std::string& nodeId, const std::string& opType, NodeType nodeType)
    : nodeId_(nodeId), opType_(opType), name_(nodeId), nodeType_(nodeType) {
}

inline void GraphNode::addInputEdge(std::shared_ptr<GraphEdge> edge) {
    if (edge) {
        inputEdges_.push_back(edge);
    }
}

inline void GraphNode::addOutputEdge(std::shared_ptr<GraphEdge> edge) {
    if (edge) {
        outputEdges_.push_back(edge);
    }
}

inline void GraphNode::setParameter(const std::string& key, const std::any& value) {
    parameters_[key] = value;
}

inline std::any GraphNode::getParameter(const std::string& key) const {
    auto it = parameters_.find(key);
    if (it != parameters_.end()) {
        return it->second;
    }
    return std::any();
}

inline bool GraphNode::hasParameter(const std::string& key) const {
    return parameters_.find(key) != parameters_.end();
}

} // namespace ad
} // namespace yt
