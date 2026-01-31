#include "../../include/ad/graph.hpp"
#include "../../example/ymodel2-s-2/json.hpp"
#include <sstream>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

namespace yt {
namespace ad {

int Graph::addNode(const std::string& name, NodeType type) {
    int id = next_node_id++;
    auto node = std::make_shared<Node>(id, name, type);
    nodes[id] = node;
    return id;
}

int Graph::addEdge(int from, int to) {
    int id = next_edge_id++;
    auto edge = std::make_shared<Edge>(id, from, to);
    edges[id] = edge;
    
    if (nodes.count(from)) {
        nodes[from]->output_ids.push_back(to);
    }
    if (nodes.count(to)) {
        nodes[to]->input_ids.push_back(from);
    }
    
    return id;
}

std::shared_ptr<Node> Graph::getNode(int id) {
    auto it = nodes.find(id);
    if (it == nodes.end()) {
        throw std::runtime_error("Node not found: " + std::to_string(id));
    }
    return it->second;
}

std::shared_ptr<Node> Graph::getNode(const std::string& name) {
    for (const auto& [id, node] : nodes) {
        if (node->name == name) {
            return node;
        }
    }
    throw std::runtime_error("Node not found: " + name);
}

std::vector<int> Graph::topologicalSort() const {
    std::vector<int> result;
    std::unordered_map<int, int> inDegree;
    
    for (const auto& [id, node] : nodes) {
        inDegree[id] = node->input_ids.size();
    }
    
    std::vector<int> queue;
    for (const auto& [id, degree] : inDegree) {
        if (degree == 0) {
            queue.push_back(id);
        }
    }
    
    size_t front = 0;
    while (front < queue.size()) {
        int current = queue[front++];
        result.push_back(current);
        
        auto node = nodes.at(current);
        for (int outputId : node->output_ids) {
            inDegree[outputId]--;
            if (inDegree[outputId] == 0) {
                queue.push_back(outputId);
            }
        }
    }
    
    if (result.size() != nodes.size()) {
        throw std::runtime_error("Graph contains cycles");
    }
    
    return result;
}

std::string nodeTypeToString(NodeType type) {
    switch (type) {
        case NodeType::Input: return "Input";
        case NodeType::Parameter: return "Parameter";
        case NodeType::Constant: return "Constant";
        case NodeType::Output: return "Output";
        case NodeType::Linear: return "Linear";
        case NodeType::RMSNorm: return "RMSNorm";
        case NodeType::Embedding: return "Embedding";
        case NodeType::Attention: return "Attention";
        case NodeType::FFN: return "FFN";
        case NodeType::Add: return "Add";
        default: return "Custom";
    }
}

NodeType stringToNodeType(const std::string& str) {
    if (str == "Input") return NodeType::Input;
    if (str == "Parameter") return NodeType::Parameter;
    if (str == "Constant") return NodeType::Constant;
    if (str == "Output") return NodeType::Output;
    if (str == "Linear") return NodeType::Linear;
    if (str == "RMSNorm") return NodeType::RMSNorm;
    if (str == "Embedding") return NodeType::Embedding;
    if (str == "Attention") return NodeType::Attention;
    if (str == "FFN") return NodeType::FFN;
    if (str == "Add") return NodeType::Add;
    return NodeType::Custom;
}

std::string Graph::toJSON() const {
    json j;
    
    json nodesArray = json::array();
    for (const auto& [id, node] : nodes) {
        json nodeJson;
        nodeJson["id"] = id;
        nodeJson["name"] = node->name;
        nodeJson["type"] = nodeTypeToString(node->type);
        nodeJson["attrs"] = node->attrs;
        nodeJson["inputs"] = node->input_ids;
        nodeJson["outputs"] = node->output_ids;
        nodesArray.push_back(nodeJson);
    }
    j["nodes"] = nodesArray;
    
    json edgesArray = json::array();
    for (const auto& [id, edge] : edges) {
        json edgeJson;
        edgeJson["id"] = id;
        edgeJson["from"] = edge->from_node;
        edgeJson["to"] = edge->to_node;
        edgeJson["name"] = edge->name;
        edgesArray.push_back(edgeJson);
    }
    j["edges"] = edgesArray;
    
    return j.dump(2);
}

Graph Graph::fromJSON(const std::string& jsonStr) {
    Graph graph;
    json j = json::parse(jsonStr);
    
    if (j.contains("nodes")) {
        for (const auto& nodeJson : j["nodes"]) {
            int id = nodeJson["id"];
            std::string name = nodeJson["name"];
            NodeType type = stringToNodeType(nodeJson["type"]);
            
            auto node = std::make_shared<Node>(id, name, type);
            if (nodeJson.contains("attrs")) {
                for (auto it = nodeJson["attrs"].begin(); it != nodeJson["attrs"].end(); ++it) {
                    node->attrs[it.key()] = it.value();
                }
            }
            graph.nodes[id] = node;
            graph.next_node_id = std::max(graph.next_node_id, id + 1);
        }
    }
    
    if (j.contains("edges")) {
        for (const auto& edgeJson : j["edges"]) {
            int id = edgeJson["id"];
            int from = edgeJson["from"];
            int to = edgeJson["to"];
            
            auto edge = std::make_shared<Edge>(id, from, to);
            if (edgeJson.contains("name")) {
                edge->name = edgeJson["name"];
            }
            
            graph.edges[id] = edge;
            if (graph.nodes.count(from)) {
                graph.nodes[from]->output_ids.push_back(to);
            }
            if (graph.nodes.count(to)) {
                graph.nodes[to]->input_ids.push_back(from);
            }
            
            graph.next_edge_id = std::max(graph.next_edge_id, id + 1);
        }
    }
    
    return graph;
}

// 执行图 - 这是核心执行逻辑
std::unordered_map<int, std::shared_ptr<YTensorBase>> Graph::execute(
    const std::unordered_map<std::string, std::shared_ptr<YTensorBase>>& inputs
) {
    std::unordered_map<int, std::shared_ptr<YTensorBase>> nodeOutputs;
    
    // 按拓扑顺序执行
    auto sorted = topologicalSort();
    
    for (int nodeId : sorted) {
        auto node = nodes[nodeId];
        
        // 处理输入节点
        if (node->type == NodeType::Input) {
            auto it = inputs.find(node->name);
            if (it != inputs.end()) {
                nodeOutputs[nodeId] = it->second;
            }
            continue;
        }
        
        // 处理参数/常量节点  
        if (node->type == NodeType::Parameter || node->type == NodeType::Constant) {
            if (node->data) {
                nodeOutputs[nodeId] = node->data;
            }
            continue;
        }
        
        // 收集输入
        std::vector<std::shared_ptr<YTensorBase>> nodeInputs;
        for (int inputId : node->input_ids) {
            if (nodeOutputs.count(inputId)) {
                nodeInputs.push_back(nodeOutputs[inputId]);
            }
        }
        
        // 根据节点类型执行操作
        // 注意：完整实现需要所有算子的执行逻辑
        // 这里只是框架，具体算子需要逐个实现
        
        if (node->type == NodeType::Output) {
            if (!nodeInputs.empty()) {
                nodeOutputs[nodeId] = nodeInputs[0];
            }
        }
        // TODO: 实现其他算子
    }
    
    return nodeOutputs;
}

} // namespace ad
} // namespace yt
