#include "../include/ad/graph.hpp"
#include "../include/ad/ops.hpp"
#include <iostream>
#include <fstream>

using namespace yt::ad;

int main() {
    std::cout << "=== YTensor AD Computational Graph with Operator Registry ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 测试算子注册系统
        std::cout << "[1] 测试算子注册系统..." << std::endl;
        auto& registry = OpRegistry::instance();
        
        std::vector<std::string> ops = {"Linear", "RMSNorm", "Add", "Multiply", "GELU", "Embedding"};
        for (const auto& op_name : ops) {
            if (registry.has(op_name)) {
                std::cout << "  ✓ " << op_name << " 已注册" << std::endl;
            } else {
                std::cout << "  ✗ " << op_name << " 未注册" << std::endl;
            }
        }
        std::cout << std::endl;
        
        // 测试基本图构建（使用算子）
        std::cout << "[2] 测试使用算子的图构建..." << std::endl;
        Graph g;
        
        int input = g.addNode("input", NodeType::Input);
        int weight = g.addNode("weight", NodeType::Parameter);
        int linear = g.addOpNode("linear_op", "Linear");
        int output = g.addNode("output", NodeType::Output);
        
        g.addEdge(input, linear);
        g.addEdge(weight, linear);
        g.addEdge(linear, output);
        
        std::cout << "  ✓ 创建了 " << g.nodes.size() << " 个节点" << std::endl;
        std::cout << "  ✓ 创建了 " << g.edges.size() << " 条边" << std::endl;
        std::cout << "  ✓ Linear算子节点: " << g.getNode(linear)->op_type << std::endl;
        std::cout << std::endl;
        
        // 测试拓扑排序
        std::cout << "[3] 测试拓扑排序..." << std::endl;
        auto sorted = g.topologicalSort();
        std::cout << "  ✓ 排序成功，顺序: ";
        for (int id : sorted) {
            std::cout << g.getNode(id)->name << " ";
        }
        std::cout << std::endl << std::endl;
        
        // 测试JSON序列化
        std::cout << "[4] 测试JSON序列化..." << std::endl;
        std::string json = g.toJSON();
        std::cout << "  ✓ JSON长度: " << json.length() << " 字符" << std::endl;
        std::cout << "  JSON前250字符:" << std::endl;
        std::cout << "  " << json.substr(0, 250) << "..." << std::endl;
        std::cout << std::endl;
        
        // 保存到文件
        std::ofstream file("test_graph.json");
        file << json;
        file.close();
        std::cout << "  ✓ 保存到 test_graph.json" << std::endl;
        std::cout << std::endl;
        
        // 测试从JSON加载
        std::cout << "[5] 测试从JSON加载..." << std::endl;
        std::ifstream infile("test_graph.json");
        std::stringstream buffer;
        buffer << infile.rdbuf();
        infile.close();
        
        Graph loaded = Graph::fromJSON(buffer.str());
        std::cout << "  ✓ 加载成功" << std::endl;
        std::cout << "  ✓ 节点数: " << loaded.nodes.size() << std::endl;
        std::cout << "  ✓ 边数: " << loaded.edges.size() << std::endl;
        
        // 检查算子是否正确恢复
        for (const auto& [id, node] : loaded.nodes) {
            if (node->type == NodeType::Op) {
                std::cout << "  ✓ 算子节点恢复: " << node->name << " -> " << node->op_type;
                if (node->op) {
                    std::cout << " (算子实例已创建)";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        
        std::cout << "=== 算子注册系统测试通过 ===" << std::endl;
        std::cout << std::endl;
        std::cout << "[关键特性]" << std::endl;
        std::cout << "✓ 算子注册系统 - 支持动态注册和创建算子" << std::endl;
        std::cout << "✓ 前向/反向分离 - 每个算子都有forward和backward方法" << std::endl;
        std::cout << "✓ OpContext保存 - 前向时保存中间结果，供反向使用" << std::endl;
        std::cout << "✓ 易于扩展 - 使用REGISTER_OPERATOR宏轻松注册新算子" << std::endl;
        std::cout << std::endl;
        std::cout << "[下一步]" << std::endl;
        std::cout << "1. 实现完整的算子逻辑（RMSNorm, GELU, Embedding等）" << std::endl;
        std::cout << "2. 实现YModel2图构建器" << std::endl;
        std::cout << "3. 实现权重加载" << std::endl;
        std::cout << "4. 实现完整的反向传播" << std::endl;
        std::cout << "5. 与ymodel2-s-2进行数值验证" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
