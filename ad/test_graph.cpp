#include "ad.hpp"
#include "ymodel2_graph_builder.hpp"
#include <iostream>

int main() {
    std::cout << "=== YTensor Computational Graph Test ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 构建YModel2-s-2的计算图
        std::cout << "Building YModel2-s-2 computational graph..." << std::endl;
        auto graph = yt::ad::YModel2GraphBuilder::buildYModel2S2Graph();
        std::cout << "Graph built successfully!" << std::endl;
        std::cout << "Number of nodes: " << graph.nodes().size() << std::endl;
        std::cout << "Number of edges: " << graph.edges().size() << std::endl;
        std::cout << std::endl;
        
        // 序列化为JSON
        std::cout << "Serializing graph to JSON..." << std::endl;
        std::string json = graph.toJSON();
        std::cout << "JSON serialization successful!" << std::endl;
        std::cout << "JSON length: " << json.length() << " characters" << std::endl;
        std::cout << std::endl;
        
        // 保存到文件
        std::string filename = "ymodel2_s2_graph.json";
        std::cout << "Saving graph to file: " << filename << std::endl;
        if (graph.saveToFile(filename)) {
            std::cout << "Graph saved successfully!" << std::endl;
        } else {
            std::cerr << "Failed to save graph to file!" << std::endl;
            return 1;
        }
        std::cout << std::endl;
        
        // 从文件加载
        std::cout << "Loading graph from file..." << std::endl;
        auto loadedGraph = yt::ad::ComputationGraph::loadFromFile(filename);
        std::cout << "Graph loaded successfully!" << std::endl;
        std::cout << "Number of nodes: " << loadedGraph.nodes().size() << std::endl;
        std::cout << "Number of edges: " << loadedGraph.edges().size() << std::endl;
        std::cout << std::endl;
        
        // 验证加载的图
        if (loadedGraph.nodes().size() == graph.nodes().size() &&
            loadedGraph.edges().size() == graph.edges().size()) {
            std::cout << "✓ Graph serialization and deserialization test passed!" << std::endl;
        } else {
            std::cerr << "✗ Graph serialization and deserialization test failed!" << std::endl;
            return 1;
        }
        std::cout << std::endl;
        
        // 打印部分JSON以供查看
        std::cout << "First 500 characters of JSON:" << std::endl;
        std::cout << json.substr(0, 500) << "..." << std::endl;
        std::cout << std::endl;
        
        // 拓扑排序测试
        std::cout << "Testing topological sort..." << std::endl;
        auto sortedNodes = loadedGraph.topologicalSort();
        std::cout << "Topological sort successful!" << std::endl;
        std::cout << "Number of sorted nodes: " << sortedNodes.size() << std::endl;
        std::cout << std::endl;
        
        // 显示前10个节点
        std::cout << "First 10 nodes in topological order:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), sortedNodes.size()); ++i) {
            auto node = loadedGraph.getNode(sortedNodes[i]);
            std::cout << "  " << i << ". [" << node->id() << "] " 
                     << node->name() << " (" 
                     << yt::ad::nodeTypeToString(node->type()) << ")" << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "=== All tests passed! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
