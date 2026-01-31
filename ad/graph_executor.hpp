#pragma once
/***************
* @file: graph_executor.hpp
* @brief: 计算图执行器
***************/
#include <unordered_map>
#include <memory>
#include <functional>
#include "computation_graph.hpp"
#include "../ytensor.hpp"

namespace yt {
namespace ad {

// 计算图执行器
class GraphExecutor {
public:
    GraphExecutor(const ComputationGraph& graph) : graph_(graph) {}
    
    // 设置输入数据
    void setInput(const std::string& name, const YTensorBase& data) {
        inputs_[name] = std::make_shared<YTensorBase>(data);
    }
    
    // 执行图
    void execute();
    
    // 获取输出数据
    std::shared_ptr<YTensorBase> getOutput(const std::string& name) const {
        auto it = outputs_.find(name);
        if (it == outputs_.end()) {
            throw std::runtime_error("Output not found: " + name);
        }
        return it->second;
    }
    
    // 获取中间结果
    std::shared_ptr<YTensorBase> getNodeOutput(int nodeId) const {
        auto it = nodeOutputs_.find(nodeId);
        if (it == nodeOutputs_.end()) {
            return nullptr;
        }
        return it->second;
    }
    
private:
    const ComputationGraph& graph_;
    std::unordered_map<std::string, std::shared_ptr<YTensorBase>> inputs_;
    std::unordered_map<std::string, std::shared_ptr<YTensorBase>> outputs_;
    std::unordered_map<int, std::shared_ptr<YTensorBase>> nodeOutputs_;
    
    // 执行单个节点
    std::shared_ptr<YTensorBase> executeNode(std::shared_ptr<Node> node);
};

} // namespace ad
} // namespace yt
