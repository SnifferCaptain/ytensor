#include "../include/ad/graph.hpp"
#include <iostream>
#include <fstream>

using namespace yt::ad;

int main() {
    std::cout << "=== YTensor AD Computational Graph Test ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "[重要说明]" << std::endl;
    std::cout << "要实现与ymodel2-s-2完全相同的数值输出，需要：" << std::endl;
    std::cout << "1. 实现所有算子的完整执行逻辑（Linear, RMSNorm, Attention, FFN等）" << std::endl;
    std::cout << "2. 实现权重加载机制" << std::endl;
    std::cout << "3. 确保相同的随机种子处理" << std::endl;
    std::cout << "4. 保证数值精度完全一致" << std::endl;
    std::cout << "这是一个需要大量工作的完整推理引擎实现。" << std::endl;
    std::cout << std::endl;
    
    try {
        // 测试基本图构建
        std::cout << "[1] 测试基本图构建..." << std::endl;
        Graph g;
        
        int input = g.addNode("input", NodeType::Input);
        int param = g.addNode("weight", NodeType::Parameter);
        int op = g.addNode("linear", NodeType::Linear);
        int output = g.addNode("output", NodeType::Output);
        
        g.addEdge(input, op);
        g.addEdge(param, op);
        g.addEdge(op, output);
        
        std::cout << "  ✓ 创建了 " << g.nodes.size() << " 个节点" << std::endl;
        std::cout << "  ✓ 创建了 " << g.edges.size() << " 条边" << std::endl;
        std::cout << std::endl;
        
        // 测试拓扑排序
        std::cout << "[2] 测试拓扑排序..." << std::endl;
        auto sorted = g.topologicalSort();
        std::cout << "  ✓ 排序成功，顺序: ";
        for (int id : sorted) {
            std::cout << g.getNode(id)->name << " ";
        }
        std::cout << std::endl << std::endl;
        
        // 测试JSON序列化
        std::cout << "[3] 测试JSON序列化..." << std::endl;
        std::string json = g.toJSON();
        std::cout << "  ✓ JSON长度: " << json.length() << " 字符" << std::endl;
        std::cout << "  JSON前200字符:" << std::endl;
        std::cout << "  " << json.substr(0, 200) << "..." << std::endl;
        std::cout << std::endl;
        
        // 保存到文件
        std::ofstream file("test_graph.json");
        file << json;
        file.close();
        std::cout << "  ✓ 保存到 test_graph.json" << std::endl;
        std::cout << std::endl;
        
        // 测试从JSON加载
        std::cout << "[4] 测试从JSON加载..." << std::endl;
        std::ifstream infile("test_graph.json");
        std::stringstream buffer;
        buffer << infile.rdbuf();
        infile.close();
        
        Graph loaded = Graph::fromJSON(buffer.str());
        std::cout << "  ✓ 加载成功" << std::endl;
        std::cout << "  ✓ 节点数: " << loaded.nodes.size() << std::endl;
        std::cout << "  ✓ 边数: " << loaded.edges.size() << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== 基础测试通过 ===" << std::endl;
        std::cout << std::endl;
        std::cout << "[下一步]" << std::endl;
        std::cout << "需要实现：" << std::endl;
        std::cout << "1. YModel2图构建器 - 从配置构建完整的transformer图" << std::endl;
        std::cout << "2. 算子执行逻辑 - Linear, RMSNorm, Attention, FFN等" << std::endl;
        std::cout << "3. 权重加载 - 从.yt文件加载模型参数" << std::endl;
        std::cout << "4. 执行测试 - 与ymodel2-s-2的数值输出进行精确比对" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
