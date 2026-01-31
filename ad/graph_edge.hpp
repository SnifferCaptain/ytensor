#pragma once
/***************
* @file: graph_edge.hpp
* @brief: 计算图边类，表示数据流
***************/
#include <string>
#include <vector>
#include <memory>
#include "../ytensor.hpp"

namespace yt {
namespace ad {

// 计算图边，表示张量数据流
class Edge {
public:
    // 构造函数
    Edge(int from, int to, const std::string& name = "")
        : from_(from), to_(to), name_(name), id_(nextId_++) {}
    
    // 获取源节点ID
    int from() const { return from_; }
    
    // 获取目标节点ID
    int to() const { return to_; }
    
    // 获取边ID
    int id() const { return id_; }
    
    // 获取边名称
    const std::string& name() const { return name_; }
    
    // 设置边的数据（张量）
    void setData(const YTensorBase& data) {
        data_ = std::make_shared<YTensorBase>(data);
    }
    
    // 获取边的数据
    std::shared_ptr<YTensorBase> data() const { return data_; }
    
    // 设置形状信息（用于序列化时记录）
    void setShape(const std::vector<int>& shape) { shape_ = shape; }
    
    // 获取形状信息
    const std::vector<int>& shape() const { return shape_; }
    
    // 设置数据类型
    void setDtype(const std::string& dtype) { dtype_ = dtype; }
    
    // 获取数据类型
    const std::string& dtype() const { return dtype_; }
    
private:
    int from_;                              // 源节点ID
    int to_;                                // 目标节点ID
    std::string name_;                      // 边名称
    int id_;                                // 边ID
    std::shared_ptr<YTensorBase> data_;     // 边上的数据（张量）
    std::vector<int> shape_;                // 张量形状
    std::string dtype_;                     // 数据类型
    
    static int nextId_;                     // 全局边ID计数器
};

} // namespace ad
} // namespace yt
