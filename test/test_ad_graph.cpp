#include <iostream>
#include <memory>
#include <iomanip>
#include "../ytensor.hpp"
#include "../include/ad.hpp"

using namespace yt;
using namespace yt::ad;

// 测试用：构建一个简单的前馈神经网络计算图
// 模拟 ymodel2 的一个简化版本结构
class TestFFNGraph {
public:
    ComputationGraph graph;
    
    // 构建一个简单的 FFN 层计算图: y = gelu(x @ W1) @ W2
    // 现在 W1 和 W2 是参数节点，不是边
    void build() {
        std::cout << "构建计算图..." << std::endl;
        
        // 注册算子
        registerOperators();
        
        // 创建节点
        auto input_node = graph.createNode("input", "input", NodeType::Input);
        auto w1_node = graph.createNode("w1", "parameter", NodeType::Parameter);
        auto w2_node = graph.createNode("w2", "parameter", NodeType::Parameter);
        auto matmul1_node = graph.createNode("matmul1", "matmul", NodeType::Operator);
        auto gelu_node = graph.createNode("gelu", "gelu", NodeType::Operator);
        auto matmul2_node = graph.createNode("matmul2", "matmul", NodeType::Operator);
        auto output_node = graph.createNode("output", "output", NodeType::Operator);
        
        // 设置节点名称
        input_node->setName("输入节点");
        w1_node->setName("权重参数W1");
        w2_node->setName("权重参数W2");
        matmul1_node->setName("第一层矩阵乘法");
        gelu_node->setName("GELU激活");
        matmul2_node->setName("第二层矩阵乘法");
        output_node->setName("输出节点");
        
        // 创建边连接：现在 w1 和 w2 是节点，需要通过边连接到算子
        auto x_edge = graph.createEdge("x", nullptr, input_node);              // 外部输入 -> input节点
        auto x_to_mm1 = graph.createEdge("x_to_mm1", input_node, matmul1_node); // input -> matmul1
        auto w1_to_mm1 = graph.createEdge("w1_to_mm1", w1_node, matmul1_node); // w1参数 -> matmul1
        auto h1_edge = graph.createEdge("h1", matmul1_node, gelu_node);        // matmul1 -> gelu
        auto h2_edge = graph.createEdge("h2", gelu_node, matmul2_node);        // gelu -> matmul2
        auto w2_to_mm2 = graph.createEdge("w2_to_mm2", w2_node, matmul2_node); // w2参数 -> matmul2
        auto y_edge = graph.createEdge("y", matmul2_node, output_node);        // matmul2 -> output
        auto out_edge = graph.createEdge("out", output_node, nullptr);         // output -> 外部输出
        
        std::cout << "计算图构建完成！" << std::endl;
        std::cout << "  节点数: " << graph.nodeCount() << std::endl;
        std::cout << "  边数: " << graph.edgeCount() << std::endl;
    }
    
    void registerOperators() {
        // 输出算子：直接传递输出
        graph.registerOperator("output", [](const GraphNode& node,
                                            const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                            std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (!inputs.empty() && !outputs.empty()) {
                outputs[0]->setTensor(inputs[0]->getTensor());
            }
        });
        
        // 矩阵乘法算子 (支持3D输入)
        graph.registerOperator("matmul", [](const GraphNode& node,
                                            const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                            std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (inputs.size() < 2 || outputs.empty()) {
                throw std::runtime_error("matmul requires 2 inputs and 1 output");
            }
            auto x = inputs[0]->getTensor();
            const auto& w = inputs[1]->getTensor();
            
            // 如果是3D张量，需要reshape成2D，执行matmul，再reshape回3D
            if (x.ndim() == 3) {
                int b = x.shape(0), s = x.shape(1), h = x.shape(2);
                x = x.view(b * s, h);  // [b, s, h] -> [b*s, h]
                YTensorBase result = x.matmul(w);  // [b*s, h] @ [h, h'] -> [b*s, h']
                int h_out = result.shape(1);
                result = result.view(b, s, h_out);  // [b*s, h'] -> [b, s, h']
                outputs[0]->setTensor(result);
            } else {
                // 2D矩阵乘法
                YTensorBase result = x.matmul(w);
                outputs[0]->setTensor(result);
            }
        });
        
        // GELU激活函数算子
        graph.registerOperator("gelu", [](const GraphNode& node,
                                          const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                          std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (inputs.empty() || outputs.empty()) {
                throw std::runtime_error("gelu requires 1 input and 1 output");
            }
            const auto& x = inputs[0]->getTensor();
            
            // GELU近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            YTensorBase result = x;  // 简化版本，实际应该计算GELU
            for (size_t i = 0; i < result.size(); i++) {
                float val = result.atData<float>(i);
                // 简化的GELU近似
                float gelu_val = val * 0.5f * (1.0f + std::tanh(0.797885f * (val + 0.044715f * val * val * val)));
                result.atData<float>(i) = gelu_val;
            }
            outputs[0]->setTensor(result);
        });
    }
    
    // 打印计算图结构
    void printGraph() {
        std::cout << "\n========== 计算图结构 ==========" << std::endl;
        std::cout << "节点总数: " << graph.nodeCount() << std::endl;
        std::cout << "边总数: " << graph.edgeCount() << std::endl;
        
        std::cout << "\n节点列表:" << std::endl;
        for (const auto& [id, node] : graph.getNodes()) {
            std::string nodeTypeStr;
            switch(node->getNodeType()) {
                case NodeType::Operator: nodeTypeStr = "算子"; break;
                case NodeType::Parameter: nodeTypeStr = "参数"; break;
                case NodeType::Input: nodeTypeStr = "输入"; break;
                case NodeType::Constant: nodeTypeStr = "常量"; break;
            }
            std::cout << "  [" << id << "] " << node->getName() 
                      << " (类型: " << nodeTypeStr << ", op: " << node->getOpType() << ")" << std::endl;
            std::cout << "    输入边: ";
            for (const auto& edge : node->getInputEdges()) {
                std::cout << edge->getId() << " ";
            }
            std::cout << std::endl;
            std::cout << "    输出边: ";
            for (const auto& edge : node->getOutputEdges()) {
                std::cout << edge->getId() << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n边列表:" << std::endl;
        for (const auto& [id, edge] : graph.getEdges()) {
            std::cout << "  [" << id << "] ";
            if (edge->getFromNode()) {
                std::cout << edge->getFromNode()->getId();
            } else {
                std::cout << "<EXTERNAL_INPUT>";
            }
            std::cout << " -> ";
            if (edge->getToNode()) {
                std::cout << edge->getToNode()->getId();
            } else {
                std::cout << "<EXTERNAL_OUTPUT>";
            }
            std::cout << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
    }
    
    // 执行前向传播
    YTensorBase forward(const YTensorBase& x, const YTensorBase& w1, const YTensorBase& w2) {
        std::cout << "执行前向传播..." << std::endl;
        
        // 设置参数节点的数据
        auto w1_node = graph.getNode("w1");
        auto w2_node = graph.getNode("w2");
        w1_node->setData(w1);
        w2_node->setData(w2);
        
        // 准备输入（只有外部输入x）
        std::unordered_map<std::string, YTensorBase> inputs = {
            {"x", x}
        };
        
        // 执行计算图
        auto outputs = graph.execute(inputs, {"out"});
        
        std::cout << "前向传播完成！" << std::endl;
        return outputs["out"];
    }
    
    // 序列化为JSON
    std::string toJson() {
        return GraphSerializer::toJson(graph);
    }
};

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  YTensor 计算图测试程序" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        // 创建测试图
        TestFFNGraph test_graph;
        test_graph.build();
        
        // 打印计算图结构
        test_graph.printGraph();
        
        // 准备测试数据
        std::cout << "准备测试数据..." << std::endl;
        int batch = 2, seq_len = 3, hidden = 4, intermediate = 6;
        
        YTensorBase x({batch, seq_len, hidden}, "float32");
        YTensorBase w1({hidden, intermediate}, "float32");
        YTensorBase w2({intermediate, hidden}, "float32");
        
        // 初始化数据
        for (size_t i = 0; i < x.size(); i++) {
            x.atData<float>(i) = 0.1f * (i % 10);
        }
        for (size_t i = 0; i < w1.size(); i++) {
            w1.atData<float>(i) = 0.01f * (i % 20 - 10);
        }
        for (size_t i = 0; i < w2.size(); i++) {
            w2.atData<float>(i) = 0.01f * (i % 20 - 10);
        }
        
        std::cout << "  输入形状: [" << x.shape(0) << ", " << x.shape(1) << ", " << x.shape(2) << "]" << std::endl;
        std::cout << "  W1形状: [" << w1.shape(0) << ", " << w1.shape(1) << "]" << std::endl;
        std::cout << "  W2形状: [" << w2.shape(0) << ", " << w2.shape(1) << "]" << std::endl;
        
        // 执行前向传播
        auto output = test_graph.forward(x, w1, w2);
        
        std::cout << "\n前向传播结果:" << std::endl;
        std::cout << "  输出形状: [" << output.shape(0) << ", " << output.shape(1) << ", " << output.shape(2) << "]" << std::endl;
        std::cout << "  输出数据（前10个元素）: ";
        for (int i = 0; i < std::min(10, (int)output.size()); i++) {
            std::cout << std::fixed << std::setprecision(4) << output.atData<float>(i) << " ";
        }
        std::cout << std::endl;
        
        // 序列化为JSON
        std::cout << "\n序列化为JSON..." << std::endl;
        std::string json = test_graph.toJson();
        std::cout << "JSON输出:" << std::endl;
        std::cout << json << std::endl;
        
        // 保存到文件
        std::string filename = "/tmp/test_graph.json";
        if (GraphSerializer::toJsonFile(test_graph.graph, filename)) {
            std::cout << "\n计算图已保存到: " << filename << std::endl;
        }
        
        // 验证拓扑排序
        std::cout << "\n拓扑排序结果:" << std::endl;
        auto topo_order = test_graph.graph.topologicalSort();
        for (size_t i = 0; i < topo_order.size(); i++) {
            auto node = test_graph.graph.getNode(topo_order[i]);
            std::cout << "  " << (i+1) << ". " << node->getName() 
                      << " (" << node->getId() << ")" << std::endl;
        }
        
        std::cout << "\n✓ 所有测试通过！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ 错误: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  测试完成" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
