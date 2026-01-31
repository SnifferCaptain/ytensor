#include "graph_executor.hpp"
#include <iostream>

namespace yt {
namespace ad {

void GraphExecutor::execute() {
    // 获取拓扑排序后的节点顺序
    auto sortedNodes = graph_.topologicalSort();
    
    // 按拓扑顺序执行每个节点
    for (int nodeId : sortedNodes) {
        auto node = graph_.getNode(nodeId);
        auto output = executeNode(node);
        
        // 保存节点输出
        if (output) {
            nodeOutputs_[nodeId] = output;
            
            // 如果是输出节点，保存到outputs_
            if (node->type() == NodeType::Output) {
                outputs_[node->name()] = output;
            }
        }
    }
}

std::shared_ptr<YTensorBase> GraphExecutor::executeNode(std::shared_ptr<Node> node) {
    NodeType type = node->type();
    
    // 处理输入节点
    if (type == NodeType::Input) {
        auto it = inputs_.find(node->name());
        if (it == inputs_.end()) {
            throw std::runtime_error("Input not provided: " + node->name());
        }
        return it->second;
    }
    
    // 处理参数节点
    if (type == NodeType::Parameter || type == NodeType::Constant) {
        return node->data();
    }
    
    // 获取输入节点的输出
    std::vector<std::shared_ptr<YTensorBase>> inputTensors;
    for (int inputId : node->inputs()) {
        auto it = nodeOutputs_.find(inputId);
        if (it == nodeOutputs_.end()) {
            throw std::runtime_error("Input node not executed: " + std::to_string(inputId));
        }
        inputTensors.push_back(it->second);
    }
    
    // 根据节点类型执行操作
    switch (type) {
        case NodeType::Add: {
            if (inputTensors.size() != 2) {
                throw std::runtime_error("Add requires 2 inputs");
            }
            // 简单实现：假设形状兼容
            auto result = std::make_shared<YTensorBase>(*inputTensors[0]);
            // TODO: 实现加法操作
            return result;
        }
        
        case NodeType::Output: {
            // 输出节点直接传递输入
            if (inputTensors.size() != 1) {
                throw std::runtime_error("Output requires 1 input");
            }
            return inputTensors[0];
        }
        
        default:
            // 其他节点类型暂时不实现，返回空指针
            std::cout << "Warning: Node type " << nodeTypeToString(type) 
                     << " not implemented yet" << std::endl;
            if (!inputTensors.empty()) {
                return inputTensors[0];  // 暂时直接返回第一个输入
            }
            return nullptr;
    }
}

} // namespace ad
} // namespace yt
