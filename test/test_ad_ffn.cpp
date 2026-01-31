#include <iostream>
#include <memory>
#include <iomanip>
#include <cmath>
#include <random>
#include "../ytensor.hpp"
#include "../include/ad.hpp"

using namespace yt;
using namespace yt::ad;

// 简化的FFN块：与ymodel2的FFN类似
// FFN: x -> linear(up) -> gelu -> linear(down) -> output

struct FFNConfig {
    int hidden_size = 64;
    int intermediate_size = 128;
};

class SimplifiedFFNGraph {
public:
    ComputationGraph graph;
    FFNConfig config;
    
    void build() {
        std::cout << "\n=== 构建简化FFN计算图 ===" << std::endl;
        
        // 注册算子
        registerOperators();
        
        // 创建节点
        // 输入节点
        auto input_node = graph.createNode("input", "input", NodeType::Input);
        input_node->setName("输入X");
        
        // 参数节点
        auto up_weight = graph.createNode("up_weight", "parameter", NodeType::Parameter);
        up_weight->setName("上投影权重");
        
        auto down_weight = graph.createNode("down_weight", "parameter", NodeType::Parameter);
        down_weight->setName("下投影权重");
        
        // 算子节点
        auto linear1 = graph.createNode("linear1", "linear", NodeType::Operator);
        linear1->setName("第一层线性变换");
        
        auto gelu = graph.createNode("gelu", "gelu", NodeType::Operator);
        gelu->setName("GELU激活");
        
        auto linear2 = graph.createNode("linear2", "linear", NodeType::Operator);
        linear2->setName("第二层线性变换");
        
        auto output_node = graph.createNode("output", "output", NodeType::Operator);
        output_node->setName("输出");
        
        // 创建边连接
        graph.createEdge("x_input", nullptr, input_node);
        graph.createEdge("x_to_l1", input_node, linear1);
        graph.createEdge("up_to_l1", up_weight, linear1);
        graph.createEdge("h1", linear1, gelu);
        graph.createEdge("h2", gelu, linear2);
        graph.createEdge("down_to_l2", down_weight, linear2);
        graph.createEdge("y", linear2, output_node);
        graph.createEdge("output", output_node, nullptr);
        
        std::cout << "图构建完成: " << graph.nodeCount() << " 节点, " 
                  << graph.edgeCount() << " 边" << std::endl;
    }
    
    void registerOperators() {
        // 输出算子
        graph.registerOperator("output", [](const GraphNode&,
                                            const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                            std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (!inputs.empty() && !outputs.empty()) {
                outputs[0]->setTensor(inputs[0]->getTensor());
            }
        });
        
        // 线性层算子 (matmul)
        graph.registerOperator("linear", [](const GraphNode&,
                                            const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                            std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (inputs.size() < 2 || outputs.empty()) {
                throw std::runtime_error("linear requires 2 inputs");
            }
            
            auto x = inputs[0]->getTensor();
            const auto& w = inputs[1]->getTensor();
            
            // 支持3D输入 [batch, seq, hidden]
            if (x.ndim() == 3) {
                int b = x.shape(0), s = x.shape(1), h = x.shape(2);
                x = x.view(b * s, h);
                YTensorBase result = x.matmul(w);
                int h_out = result.shape(1);
                result = result.view(b, s, h_out);
                outputs[0]->setTensor(result);
            } else {
                outputs[0]->setTensor(x.matmul(w));
            }
        });
        
        // GELU激活
        graph.registerOperator("gelu", [](const GraphNode&,
                                          const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                                          std::vector<std::shared_ptr<GraphEdge>>& outputs) {
            if (inputs.empty() || outputs.empty()) {
                throw std::runtime_error("gelu requires 1 input");
            }
            
            const auto& x = inputs[0]->getTensor();
            YTensorBase result = x;
            
            // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_over_pi = 0.7978845608f;
            for (size_t i = 0; i < result.size(); i++) {
                float val = result.atData<float>(i);
                float x3 = val * val * val;
                float inner = sqrt_2_over_pi * (val + 0.044715f * x3);
                float tanh_val = std::tanh(inner);
                result.atData<float>(i) = 0.5f * val * (1.0f + tanh_val);
            }
            
            outputs[0]->setTensor(result);
        });
    }
    
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
                      << " (类型: " << nodeTypeStr << ")" << std::endl;
        }
        
        std::cout << "\n拓扑排序:" << std::endl;
        auto topo = graph.topologicalSort();
        for (size_t i = 0; i < topo.size(); i++) {
            auto node = graph.getNode(topo[i]);
            std::cout << "  " << (i+1) << ". " << node->getName() << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
    }
    
    YTensorBase forward(const YTensorBase& x, const YTensorBase& up_w, const YTensorBase& down_w) {
        // 设置参数
        graph.getNode("up_weight")->setData(up_w);
        graph.getNode("down_weight")->setData(down_w);
        
        // 执行
        std::unordered_map<std::string, YTensorBase> inputs = {{"x_input", x}};
        auto outputs = graph.execute(inputs, {"output"});
        return outputs["output"];
    }
    
    std::string toJson() {
        return GraphSerializer::toJson(graph);
    }
};

// 直接实现的FFN（用于对比）
YTensorBase directFFN(const YTensorBase& x, const YTensorBase& up_w, const YTensorBase& down_w) {
    // x @ up_w
    auto x_copy = x;
    if (x_copy.ndim() == 3) {
        int b = x_copy.shape(0), s = x_copy.shape(1), h = x_copy.shape(2);
        x_copy = x_copy.view(b * s, h);
        auto h1 = x_copy.matmul(up_w);
        int h_out = h1.shape(1);
        h1 = h1.view(b, s, h_out);
        
        // GELU
        const float sqrt_2_over_pi = 0.7978845608f;
        for (size_t i = 0; i < h1.size(); i++) {
            float val = h1.atData<float>(i);
            float x3 = val * val * val;
            float inner = sqrt_2_over_pi * (val + 0.044715f * x3);
            float tanh_val = std::tanh(inner);
            h1.atData<float>(i) = 0.5f * val * (1.0f + tanh_val);
        }
        
        // h1 @ down_w
        h1 = h1.view(b * s, h1.shape(2));
        auto output = h1.matmul(down_w);
        output = output.view(b, s, output.shape(1));
        return output;
    }
    return x;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  YTensor 计算图完整测试" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // 创建简化的FFN图
        SimplifiedFFNGraph ffn_graph;
        ffn_graph.build();
        ffn_graph.printGraph();
        
        // 准备测试数据
        std::cout << "\n=== 准备测试数据 ===" << std::endl;
        int batch = 2, seq_len = 3, hidden = 64, intermediate = 128;
        
        YTensorBase x({batch, seq_len, hidden}, "float32");
        YTensorBase up_weight({hidden, intermediate}, "float32");
        YTensorBase down_weight({intermediate, hidden}, "float32");
        
        // 使用固定种子初始化
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (size_t i = 0; i < x.size(); i++) {
            x.atData<float>(i) = dist(gen);
        }
        for (size_t i = 0; i < up_weight.size(); i++) {
            up_weight.atData<float>(i) = dist(gen);
        }
        for (size_t i = 0; i < down_weight.size(); i++) {
            down_weight.atData<float>(i) = dist(gen);
        }
        
        std::cout << "输入形状: [" << batch << ", " << seq_len << ", " << hidden << "]" << std::endl;
        std::cout << "上投影权重: [" << hidden << ", " << intermediate << "]" << std::endl;
        std::cout << "下投影权重: [" << intermediate << ", " << hidden << "]" << std::endl;
        
        // 通过计算图执行
        std::cout << "\n=== 计算图前向传播 ===" << std::endl;
        auto graph_output = ffn_graph.forward(x, up_weight, down_weight);
        
        std::cout << "输出形状: [" << graph_output.shape(0) << ", " 
                  << graph_output.shape(1) << ", " << graph_output.shape(2) << "]" << std::endl;
        std::cout << "输出数据（前5个元素）: ";
        for (int i = 0; i < 5 && i < (int)graph_output.size(); i++) {
            std::cout << std::fixed << std::setprecision(6) << graph_output.atData<float>(i) << " ";
        }
        std::cout << std::endl;
        
        // 直接实现对比
        std::cout << "\n=== 直接实现对比 ===" << std::endl;
        auto direct_output = directFFN(x, up_weight, down_weight);
        
        std::cout << "输出形状: [" << direct_output.shape(0) << ", " 
                  << direct_output.shape(1) << ", " << direct_output.shape(2) << "]" << std::endl;
        std::cout << "输出数据（前5个元素）: ";
        for (int i = 0; i < 5 && i < (int)direct_output.size(); i++) {
            std::cout << std::fixed << std::setprecision(6) << direct_output.atData<float>(i) << " ";
        }
        std::cout << std::endl;
        
        // 验证结果
        std::cout << "\n=== 验证结果一致性 ===" << std::endl;
        float max_diff = 0.0f;
        for (size_t i = 0; i < graph_output.size(); i++) {
            float diff = std::abs(graph_output.atData<float>(i) - direct_output.atData<float>(i));
            max_diff = std::max(max_diff, diff);
        }
        
        std::cout << "最大差异: " << std::scientific << max_diff << std::endl;
        if (max_diff < 1e-5f) {
            std::cout << "✓ 结果一致！（误差 < 1e-5）" << std::endl;
        } else {
            std::cout << "✗ 结果不一致！" << std::endl;
        }
        
        // JSON序列化
        std::cout << "\n=== JSON序列化 ===" << std::endl;
        std::string json = ffn_graph.toJson();
        std::cout << "JSON长度: " << json.length() << " 字符" << std::endl;
        std::cout << "保存到 /tmp/ffn_graph.json" << std::endl;
        GraphSerializer::toJsonFile(ffn_graph.graph, "/tmp/ffn_graph.json");
        
        // 显示JSON片段
        std::cout << "\nJSON片段（前300字符）:" << std::endl;
        std::cout << json.substr(0, 300) << "..." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ 错误: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  所有测试完成" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
