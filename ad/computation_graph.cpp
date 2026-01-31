#include "computation_graph.hpp"
#include "../example/ymodel2-s-2/json.hpp"
#include <fstream>
#include <sstream>

using json = nlohmann::json;

namespace yt {
namespace ad {

std::string ComputationGraph::toJSON() const {
    json j;
    
    // 序列化节点
    json nodesArray = json::array();
    for (const auto& [id, node] : nodes_) {
        json nodeJson;
        nodeJson["id"] = id;
        nodeJson["name"] = node->name();
        nodeJson["type"] = nodeTypeToString(node->type());
        
        // 序列化属性
        json attrs = json::object();
        for (const auto& [key, value] : node->attributes()) {
            attrs[key] = value;
        }
        nodeJson["attributes"] = attrs;
        
        // 序列化输入和输出
        nodeJson["inputs"] = node->inputs();
        nodeJson["outputs"] = node->outputs();
        
        nodesArray.push_back(nodeJson);
    }
    j["nodes"] = nodesArray;
    
    // 序列化边
    json edgesArray = json::array();
    for (const auto& [id, edge] : edges_) {
        json edgeJson;
        edgeJson["id"] = id;
        edgeJson["from"] = edge->from();
        edgeJson["to"] = edge->to();
        edgeJson["name"] = edge->name();
        edgeJson["shape"] = edge->shape();
        edgeJson["dtype"] = edge->dtype();
        edgesArray.push_back(edgeJson);
    }
    j["edges"] = edgesArray;
    
    return j.dump(2);  // 格式化输出，缩进2个空格
}

ComputationGraph ComputationGraph::fromJSON(const std::string& jsonStr) {
    ComputationGraph graph;
    json j = json::parse(jsonStr);
    
    // 反序列化节点
    if (j.contains("nodes")) {
        for (const auto& nodeJson : j["nodes"]) {
            std::string name = nodeJson["name"];
            std::string typeStr = nodeJson["type"];
            NodeType type = stringToNodeType(typeStr);
            
            int id = graph.addNode(name, type);
            auto node = graph.getNode(id);
            
            // 反序列化属性
            if (nodeJson.contains("attributes")) {
                for (auto it = nodeJson["attributes"].begin(); it != nodeJson["attributes"].end(); ++it) {
                    node->setAttribute(it.key(), it.value());
                }
            }
        }
    }
    
    // 反序列化边
    if (j.contains("edges")) {
        for (const auto& edgeJson : j["edges"]) {
            int from = edgeJson["from"];
            int to = edgeJson["to"];
            std::string name = edgeJson.value("name", "");
            
            int edgeId = graph.addEdge(from, to, name);
            auto edge = graph.getEdge(edgeId);
            
            if (edgeJson.contains("shape")) {
                std::vector<int> shape = edgeJson["shape"];
                edge->setShape(shape);
            }
            if (edgeJson.contains("dtype")) {
                edge->setDtype(edgeJson["dtype"]);
            }
        }
    }
    
    return graph;
}

bool ComputationGraph::saveToFile(const std::string& filename) const {
    try {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        file << toJSON();
        file.close();
        return true;
    } catch (...) {
        return false;
    }
}

ComputationGraph ComputationGraph::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return fromJSON(buffer.str());
}

} // namespace ad
} // namespace yt
