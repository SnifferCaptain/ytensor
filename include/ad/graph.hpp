#pragma once
/***************
* @file: graph.hpp  
* @brief: 计算图核心类 - 节点、边、图
***************/

#include "../../ytensor.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace yt {
namespace ad {

// 前向声明
class Node;
class Edge;
class Graph;

// 节点类型
enum class NodeType {
    Input,      // 输入
    Parameter,  // 参数（可学习）
    Constant,   // 常量
    Output,     // 输出
    // 操作符
    Linear,
    RMSNorm,
    Embedding,
    Attention,
    FFN,
    Add,
    // 其他操作
    Custom
};

// 节点 - 表示算子
class Node {
public:
    int id;
    std::string name;
    NodeType type;
    std::unordered_map<std::string, std::string> attrs;
    
    // 数据存储（用于参数节点）
    std::shared_ptr<YTensorBase> data;
    
    // 连接
    std::vector<int> input_ids;   // 输入节点ID
    std::vector<int> output_ids;  // 输出节点ID
    
    Node(int id_, const std::string& name_, NodeType type_)
        : id(id_), name(name_), type(type_) {}
};

// 边 - 表示数据流
class Edge {
public:
    int id;
    int from_node;
    int to_node;
    std::string name;
    
    // 运行时数据
    std::shared_ptr<YTensorBase> data;
    
    Edge(int id_, int from_, int to_)
        : id(id_), from_node(from_), to_node(to_) {}
};

// 计算图
class Graph {
public:
    std::unordered_map<int, std::shared_ptr<Node>> nodes;
    std::unordered_map<int, std::shared_ptr<Edge>> edges;
    
    int next_node_id = 0;
    int next_edge_id = 0;
    
    // 添加节点
    int addNode(const std::string& name, NodeType type);
    
    // 添加边
    int addEdge(int from, int to);
    
    // 获取节点
    std::shared_ptr<Node> getNode(int id);
    std::shared_ptr<Node> getNode(const std::string& name);
    
    // 拓扑排序
    std::vector<int> topologicalSort() const;
    
    // JSON序列化
    std::string toJSON() const;
    static Graph fromJSON(const std::string& json);
    
    // 执行图
    std::unordered_map<int, std::shared_ptr<YTensorBase>> execute(
        const std::unordered_map<std::string, std::shared_ptr<YTensorBase>>& inputs
    );
};

} // namespace ad
} // namespace yt
