#pragma once
/***************
* @file: computation_graph.hpp
* @brief: 计算图类，管理节点和边
***************/
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include "graph_node.hpp"
#include "graph_edge.hpp"
#include "../ytensor.hpp"

namespace yt {
namespace ad {

// 计算图类
class ComputationGraph {
public:
    ComputationGraph() = default;
    
    // 添加节点
    int addNode(const std::string& name, NodeType type) {
        auto node = std::make_shared<Node>(name, type);
        int id = node->id();
        nodes_[id] = node;
        nodeNameMap_[name] = id;
        return id;
    }
    
    // 添加边
    int addEdge(int from, int to, const std::string& name = "") {
        auto edge = std::make_shared<Edge>(from, to, name);
        int id = edge->id();
        edges_[id] = edge;
        
        // 更新节点的输入输出关系
        if (nodes_.find(from) != nodes_.end()) {
            nodes_[from]->addOutput(to);
        }
        if (nodes_.find(to) != nodes_.end()) {
            nodes_[to]->addInput(from);
        }
        
        return id;
    }
    
    // 通过ID获取节点
    std::shared_ptr<Node> getNode(int id) const {
        auto it = nodes_.find(id);
        if (it == nodes_.end()) {
            throw std::runtime_error("Node not found: " + std::to_string(id));
        }
        return it->second;
    }
    
    // 通过名称获取节点
    std::shared_ptr<Node> getNode(const std::string& name) const {
        auto it = nodeNameMap_.find(name);
        if (it == nodeNameMap_.end()) {
            throw std::runtime_error("Node not found: " + name);
        }
        return getNode(it->second);
    }
    
    // 获取边
    std::shared_ptr<Edge> getEdge(int id) const {
        auto it = edges_.find(id);
        if (it == edges_.end()) {
            throw std::runtime_error("Edge not found: " + std::to_string(id));
        }
        return it->second;
    }
    
    // 获取所有节点
    const std::unordered_map<int, std::shared_ptr<Node>>& nodes() const {
        return nodes_;
    }
    
    // 获取所有边
    const std::unordered_map<int, std::shared_ptr<Edge>>& edges() const {
        return edges_;
    }
    
    // 清空图
    void clear() {
        nodes_.clear();
        edges_.clear();
        nodeNameMap_.clear();
    }
    
    // 获取输入节点列表
    std::vector<std::shared_ptr<Node>> getInputNodes() const {
        std::vector<std::shared_ptr<Node>> inputs;
        for (const auto& [id, node] : nodes_) {
            if (node->type() == NodeType::Input) {
                inputs.push_back(node);
            }
        }
        return inputs;
    }
    
    // 获取输出节点列表
    std::vector<std::shared_ptr<Node>> getOutputNodes() const {
        std::vector<std::shared_ptr<Node>> outputs;
        for (const auto& [id, node] : nodes_) {
            if (node->type() == NodeType::Output) {
                outputs.push_back(node);
            }
        }
        return outputs;
    }
    
    // 拓扑排序
    std::vector<int> topologicalSort() const {
        std::vector<int> result;
        std::unordered_map<int, int> inDegree;
        
        // 计算入度
        for (const auto& [id, node] : nodes_) {
            inDegree[id] = node->inputs().size();
        }
        
        // 找到所有入度为0的节点
        std::vector<int> queue;
        for (const auto& [id, degree] : inDegree) {
            if (degree == 0) {
                queue.push_back(id);
            }
        }
        
        // 拓扑排序 (使用FIFO顺序)
        size_t front = 0;
        while (front < queue.size()) {
            int current = queue[front++];
            result.push_back(current);
            
            auto node = nodes_.at(current);
            for (int outputId : node->outputs()) {
                inDegree[outputId]--;
                if (inDegree[outputId] == 0) {
                    queue.push_back(outputId);
                }
            }
        }
        
        // 检查是否有环
        if (result.size() != nodes_.size()) {
            throw std::runtime_error("Graph contains cycles");
        }
        
        return result;
    }
    
    // 序列化为JSON字符串
    std::string toJSON() const;
    
    // 从JSON字符串反序列化
    static ComputationGraph fromJSON(const std::string& json);
    
    // 保存到JSON文件
    bool saveToFile(const std::string& filename) const;
    
    // 从JSON文件加载
    static ComputationGraph loadFromFile(const std::string& filename);
    
private:
    std::unordered_map<int, std::shared_ptr<Node>> nodes_;      // 节点映射
    std::unordered_map<int, std::shared_ptr<Edge>> edges_;      // 边映射
    std::unordered_map<std::string, int> nodeNameMap_;          // 节点名称到ID的映射
};

} // namespace ad
} // namespace yt
