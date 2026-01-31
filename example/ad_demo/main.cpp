#include <iostream>
#include <memory>
#include "../../ytensor.hpp"
#include "../../include/ad.hpp"

using namespace yt;
using namespace yt::ad;

// 示例：实现简单的算子
void addOperator(const GraphNode& node,
                 const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                 std::vector<std::shared_ptr<GraphEdge>>& outputs) {
    if (inputs.size() < 2) {
        throw std::runtime_error("Add operator requires at least 2 inputs");
    }
    if (outputs.empty()) {
        throw std::runtime_error("Add operator requires at least 1 output");
    }
    
    const auto& tensor1 = inputs[0]->getTensor();
    const auto& tensor2 = inputs[1]->getTensor();
    
    // 简单相加操作
    YTensorBase result = tensor1 + tensor2;
    outputs[0]->setTensor(result);
}

void mulOperator(const GraphNode& node,
                 const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                 std::vector<std::shared_ptr<GraphEdge>>& outputs) {
    if (inputs.size() < 2) {
        throw std::runtime_error("Mul operator requires at least 2 inputs");
    }
    if (outputs.empty()) {
        throw std::runtime_error("Mul operator requires at least 1 output");
    }
    
    const auto& tensor1 = inputs[0]->getTensor();
    const auto& tensor2 = inputs[1]->getTensor();
    
    // 简单相乘操作
    YTensorBase result = tensor1 * tensor2;
    outputs[0]->setTensor(result);
}

int main() {
    std::cout << "=== YTensor Computational Graph Demo ===" << std::endl;
    
    try {
        // 创建计算图
        ComputationGraph graph;
        
        // 注册算子
        graph.registerOperator("add", addOperator);
        graph.registerOperator("mul", mulOperator);
        
        // 创建节点
        auto node1 = graph.createNode("add1", "add");
        node1->setName("加法节点1");
        
        auto node2 = graph.createNode("mul1", "mul");
        node2->setName("乘法节点1");
        
        auto node3 = graph.createNode("add2", "add");
        node3->setName("加法节点2");
        
        // 创建输入边
        auto input1 = graph.createEdge("input1", nullptr, node1);
        auto input2 = graph.createEdge("input2", nullptr, node1);
        auto input3 = graph.createEdge("input3", nullptr, node2);
        
        // 创建中间边
        auto edge1 = graph.createEdge("edge1", node1, node2);
        auto edge2 = graph.createEdge("edge2", node2, node3);
        
        // 创建输出边
        auto input4 = graph.createEdge("input4", nullptr, node3);
        auto output1 = graph.createEdge("output1", node3, nullptr);
        
        std::cout << "\n图结构创建完成！" << std::endl;
        std::cout << "节点数量: " << graph.nodeCount() << std::endl;
        std::cout << "边数量: " << graph.edgeCount() << std::endl;
        
        // 序列化为JSON
        std::cout << "\n序列化计算图为JSON..." << std::endl;
        std::string jsonStr = GraphSerializer::toJson(graph);
        std::cout << "JSON输出:\n" << jsonStr << std::endl;
        
        // 保存到文件
        bool saved = GraphSerializer::toJsonFile(graph, "/tmp/graph.json");
        if (saved) {
            std::cout << "\n计算图已保存到 /tmp/graph.json" << std::endl;
        }
        
        // 准备输入数据
        std::cout << "\n准备输入数据..." << std::endl;
        YTensorBase t1({2, 3}, "float32");
        YTensorBase t2({2, 3}, "float32");
        YTensorBase t3({2, 3}, "float32");
        YTensorBase t4({2, 3}, "float32");
        
        // 填充数据
        for (int i = 0; i < 6; i++) {
            t1.atData<float>(i) = 1.0f;
            t2.atData<float>(i) = 2.0f;
            t3.atData<float>(i) = 3.0f;
            t4.atData<float>(i) = 4.0f;
        }
        
        std::cout << "t1 = 1.0 (全部元素)" << std::endl;
        std::cout << "t2 = 2.0 (全部元素)" << std::endl;
        std::cout << "t3 = 3.0 (全部元素)" << std::endl;
        std::cout << "t4 = 4.0 (全部元素)" << std::endl;
        
        // 执行计算图
        // 计算过程: output1 = (t1 + t2) * t3 + t4 = 3 * 3 + 4 = 13
        std::cout << "\n执行计算图..." << std::endl;
        std::cout << "计算表达式: output = (input1 + input2) * input3 + input4" << std::endl;
        std::cout << "             = (1.0 + 2.0) * 3.0 + 4.0 = 13.0" << std::endl;
        
        std::unordered_map<std::string, YTensorBase> inputs = {
            {"input1", t1},
            {"input2", t2},
            {"input3", t3},
            {"input4", t4}
        };
        
        auto outputs = graph.execute(inputs, {"output1"});
        
        // 输出结果
        std::cout << "\n执行完成！" << std::endl;
        const auto& result = outputs["output1"];
        std::cout << "输出张量形状: [" << result.shape(0) << ", " << result.shape(1) << "]" << std::endl;
        std::cout << "输出值 (前6个元素): ";
        for (int i = 0; i < 6; i++) {
            std::cout << result.atData<float>(i) << " ";
        }
        std::cout << std::endl;
        
        // 验证结果
        bool correct = true;
        for (int i = 0; i < 6; i++) {
            if (std::abs(result.atData<float>(i) - 13.0f) > 1e-5) {
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "\n✓ 计算结果正确！" << std::endl;
        } else {
            std::cout << "\n✗ 计算结果不正确！" << std::endl;
        }
        
        // 测试拓扑排序
        std::cout << "\n拓扑排序结果:" << std::endl;
        auto topoOrder = graph.topologicalSort();
        for (const auto& nodeId : topoOrder) {
            auto node = graph.getNode(nodeId);
            std::cout << "  - " << nodeId << " (" << node->getName() << ")" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Demo完成 ===" << std::endl;
    return 0;
}
