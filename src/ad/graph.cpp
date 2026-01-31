#include "../../include/ad/graph.hpp"
#include "../../include/ad/ops.hpp"
#include "../../example/ymodel2-s-2/json.hpp"
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

namespace yt {
namespace ad {

int Graph::addNode(const std::string& name, NodeType type) {
    int id = next_node_id++;
    auto node = std::make_shared<Node>(id, name, type);
    nodes[id] = node;
    return id;
}

int Graph::addOpNode(const std::string& name, const std::string& op_type) {
    int id = next_node_id++;
    auto node = std::make_shared<Node>(id, name, NodeType::Op);
    node->op_type = op_type;
    // 创建算子实例
    node->op = OpRegistry::instance().create(op_type);
    if (!node->op) {
        std::cerr << "Warning: Operator '" << op_type << "' not registered, node will not execute" << std::endl;
    }
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
        case NodeType::Op: return "Op";
        default: return "Custom";
    }
}

NodeType stringToNodeType(const std::string& str) {
    if (str == "Input") return NodeType::Input;
    if (str == "Parameter") return NodeType::Parameter;
    if (str == "Constant") return NodeType::Constant;
    if (str == "Output") return NodeType::Output;
    if (str == "Op") return NodeType::Op;
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
        if (node->type == NodeType::Op) {
            nodeJson["op_type"] = node->op_type;
        }
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
    
    // 创建ID映射
    std::unordered_map<int, int> oldToNew;
    
    if (j.contains("nodes")) {
        for (const auto& nodeJson : j["nodes"]) {
            int oldId = nodeJson["id"];
            std::string name = nodeJson["name"];
            NodeType type = stringToNodeType(nodeJson["type"]);
            
            int newId;
            if (type == NodeType::Op && nodeJson.contains("op_type")) {
                std::string op_type = nodeJson["op_type"];
                newId = graph.addOpNode(name, op_type);
            } else {
                newId = graph.addNode(name, type);
            }
            
            oldToNew[oldId] = newId;
            auto node = graph.nodes[newId];
            
            if (nodeJson.contains("attrs")) {
                for (auto it = nodeJson["attrs"].begin(); it != nodeJson["attrs"].end(); ++it) {
                    node->attrs[it.key()] = it.value();
                }
            }
        }
    }
    
    if (j.contains("edges")) {
        for (const auto& edgeJson : j["edges"]) {
            int oldFrom = edgeJson["from"];
            int oldTo = edgeJson["to"];
            
            int newFrom = oldToNew[oldFrom];
            int newTo = oldToNew[oldTo];
            
            graph.addEdge(newFrom, newTo);
        }
    }
    
    return graph;
}

// 执行图 - 前向传播
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
        
        // 处理输出节点
        if (node->type == NodeType::Output) {
            if (!nodeInputs.empty()) {
                nodeOutputs[nodeId] = nodeInputs[0];
            }
            continue;
        }
        
        // 处理算子节点
        if (node->type == NodeType::Op && node->op) {
            try {
                auto outputs = node->op->forward(nodeInputs, node->attrs, node->ctx.get());
                if (!outputs.empty()) {
                    nodeOutputs[nodeId] = outputs[0];
                }
            } catch (const std::exception& e) {
                std::cerr << "Error executing node " << node->name << " (" << node->op_type << "): " 
                         << e.what() << std::endl;
                throw;
            }
        }
    }
    
    return nodeOutputs;
}

// 反向传播
void Graph::backward(
    const std::unordered_map<std::string, std::shared_ptr<YTensorBase>>& grad_outputs
) {
    // TODO: 实现完整的反向传播
    // 1. 按反向拓扑顺序遍历
    // 2. 对每个节点调用其算子的backward方法
    // 3. 累积梯度
    throw std::runtime_error("Backward not yet fully implemented");
}

} // namespace ad
} // namespace yt
