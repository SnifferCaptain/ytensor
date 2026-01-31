#pragma once
/***************
* @file: graph_serialization.hpp
* @brief: 计算图序列化，支持JSON格式的导入导出
* @description: 提供计算图与JSON之间的序列化和反序列化功能
***************/

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

#include "graph_runtime.hpp"

namespace yt {
namespace ad {

/// @brief 简单的JSON构建器类（用于序列化）
class JsonBuilder {
public:
    /// @brief 开始一个对象
    void beginObject();
    
    /// @brief 结束一个对象
    void endObject();
    
    /// @brief 开始一个数组
    void beginArray();
    
    /// @brief 结束一个数组
    void endArray();
    
    /// @brief 添加键值对
    void addKey(const std::string& key);
    
    /// @brief 添加字符串值
    void addString(const std::string& value);
    
    /// @brief 添加数字值
    void addNumber(double value);
    
    /// @brief 添加整数值
    void addInt(int value);
    
    /// @brief 添加布尔值
    void addBool(bool value);
    
    /// @brief 添加null值
    void addNull();
    
    /// @brief 获取生成的JSON字符串
    std::string toString() const { return buffer_; }
    
    /// @brief 清空缓冲区
    void clear() { buffer_.clear(); needComma_ = false; }

private:
    /// @brief 添加逗号（如果需要）
    void addCommaIfNeeded();
    
    std::string buffer_;
    bool needComma_ = false;
};

/// @brief 计算图序列化类
class GraphSerializer {
public:
    /// @brief 将计算图序列化为JSON字符串
    /// @param graph 计算图
    /// @return JSON字符串
    static std::string toJson(const ComputationGraph& graph);

    /// @brief 将计算图序列化到JSON文件
    /// @param graph 计算图
    /// @param filename 文件名
    /// @return 如果成功返回true
    static bool toJsonFile(const ComputationGraph& graph, const std::string& filename);

    /// @brief 从JSON字符串反序列化计算图
    /// @param json JSON字符串
    /// @param graph 输出的计算图
    /// @return 如果成功返回true
    static bool fromJson(const std::string& json, ComputationGraph& graph);

    /// @brief 从JSON文件反序列化计算图
    /// @param filename 文件名
    /// @param graph 输出的计算图
    /// @return 如果成功返回true
    static bool fromJsonFile(const std::string& filename, ComputationGraph& graph);

private:
    /// @brief 序列化单个节点
    /// @param node 节点指针
    /// @param builder JSON构建器
    static void serializeNode(const std::shared_ptr<GraphNode>& node, JsonBuilder& builder);

    /// @brief 序列化单个边
    /// @param edge 边指针
    /// @param builder JSON构建器
    static void serializeEdge(const std::shared_ptr<GraphEdge>& edge, JsonBuilder& builder);

    /// @brief 序列化节点参数
    /// @param parameters 参数映射
    /// @param builder JSON构建器
    static void serializeParameters(
        const std::unordered_map<std::string, std::any>& parameters,
        JsonBuilder& builder
    );
};

} // namespace ad
} // namespace yt
