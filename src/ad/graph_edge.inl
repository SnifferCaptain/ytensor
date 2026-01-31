/***************
* @file: graph_edge.inl
* @brief: 计算图边的实现
***************/

namespace yt{
namespace ad{

inline GraphEdge::GraphEdge(const std::string& edgeId,
                            std::shared_ptr<GraphNode> fromNode,
                            std::shared_ptr<GraphNode> toNode)
    : edgeId_(edgeId), name_(edgeId), fromNode_(fromNode), toNode_(toNode) {
}

} // namespace ad
} // namespace yt
