/***************
* @file: graph_serialization.inl
* @brief: 计算图序列化的实现
***************/

#include <iomanip>
#include <typeinfo>

namespace yt {
namespace ad {

// JsonBuilder 实现
inline void JsonBuilder::beginObject() {
    addCommaIfNeeded();
    buffer_ += "{";
    needComma_ = false;
}

inline void JsonBuilder::endObject() {
    buffer_ += "}";
    needComma_ = true;
}

inline void JsonBuilder::beginArray() {
    addCommaIfNeeded();
    buffer_ += "[";
    needComma_ = false;
}

inline void JsonBuilder::endArray() {
    buffer_ += "]";
    needComma_ = true;
}

inline void JsonBuilder::addKey(const std::string& key) {
    addCommaIfNeeded();
    buffer_ += "\"" + key + "\":";
    needComma_ = false;
}

inline void JsonBuilder::addString(const std::string& value) {
    addCommaIfNeeded();
    buffer_ += "\"";
    // 转义特殊字符
    for (char c : value) {
        switch (c) {
            case '"': buffer_ += "\\\""; break;
            case '\\': buffer_ += "\\\\"; break;
            case '\n': buffer_ += "\\n"; break;
            case '\r': buffer_ += "\\r"; break;
            case '\t': buffer_ += "\\t"; break;
            default: buffer_ += c;
        }
    }
    buffer_ += "\"";
    needComma_ = true;
}

inline void JsonBuilder::addNumber(double value) {
    addCommaIfNeeded();
    std::ostringstream oss;
    oss << std::setprecision(15) << value;
    buffer_ += oss.str();
    needComma_ = true;
}

inline void JsonBuilder::addInt(int value) {
    addCommaIfNeeded();
    buffer_ += std::to_string(value);
    needComma_ = true;
}

inline void JsonBuilder::addBool(bool value) {
    addCommaIfNeeded();
    buffer_ += value ? "true" : "false";
    needComma_ = true;
}

inline void JsonBuilder::addNull() {
    addCommaIfNeeded();
    buffer_ += "null";
    needComma_ = true;
}

inline void JsonBuilder::addCommaIfNeeded() {
    if (needComma_) {
        buffer_ += ",";
    }
}

// GraphSerializer 实现
inline std::string GraphSerializer::toJson(const ComputationGraph& graph) {
    JsonBuilder builder;
    
    builder.beginObject();
    
    // 序列化节点
    builder.addKey("nodes");
    builder.beginArray();
    for (const auto& [nodeId, node] : graph.getNodes()) {
        serializeNode(node, builder);
    }
    builder.endArray();
    
    // 序列化边
    builder.addKey("edges");
    builder.beginArray();
    for (const auto& [edgeId, edge] : graph.getEdges()) {
        serializeEdge(edge, builder);
    }
    builder.endArray();
    
    builder.endObject();
    
    return builder.toString();
}

inline bool GraphSerializer::toJsonFile(const ComputationGraph& graph, const std::string& filename) {
    std::string json = toJson(graph);
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    file << json;
    file.close();
    return true;
}

inline bool GraphSerializer::fromJson(const std::string& json, ComputationGraph& graph) {
    // 简化的JSON解析实现
    // 注意：这是一个简化版本，仅用于基本功能演示
    // 实际应用中建议使用专业的JSON库如nlohmann/json
    
    graph.clear();
    
    // 这里提供一个基本的解析框架
    // 完整实现需要一个完整的JSON解析器
    // 暂时返回false表示需要使用外部JSON库
    
    return false;
}

inline bool GraphSerializer::fromJsonFile(const std::string& filename, ComputationGraph& graph) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return fromJson(buffer.str(), graph);
}

inline void GraphSerializer::serializeNode(const std::shared_ptr<GraphNode>& node, JsonBuilder& builder) {
    builder.beginObject();
    
    builder.addKey("id");
    builder.addString(node->getId());
    
    builder.addKey("opType");
    builder.addString(node->getOpType());
    
    builder.addKey("name");
    builder.addString(node->getName());
    
    // 序列化输入边ID列表
    builder.addKey("inputEdges");
    builder.beginArray();
    for (const auto& edge : node->getInputEdges()) {
        builder.addString(edge->getId());
    }
    builder.endArray();
    
    // 序列化输出边ID列表
    builder.addKey("outputEdges");
    builder.beginArray();
    for (const auto& edge : node->getOutputEdges()) {
        builder.addString(edge->getId());
    }
    builder.endArray();
    
    // 序列化参数
    builder.addKey("parameters");
    serializeParameters(node->getParameters(), builder);
    
    builder.endObject();
}

inline void GraphSerializer::serializeEdge(const std::shared_ptr<GraphEdge>& edge, JsonBuilder& builder) {
    builder.beginObject();
    
    builder.addKey("id");
    builder.addString(edge->getId());
    
    builder.addKey("name");
    builder.addString(edge->getName());
    
    builder.addKey("fromNode");
    if (edge->getFromNode()) {
        builder.addString(edge->getFromNode()->getId());
    } else {
        builder.addNull();
    }
    
    builder.addKey("toNode");
    if (edge->getToNode()) {
        builder.addString(edge->getToNode()->getId());
    } else {
        builder.addNull();
    }
    
    builder.endObject();
}

inline void GraphSerializer::serializeParameters(
    const std::unordered_map<std::string, std::any>& parameters,
    JsonBuilder& builder) {
    
    builder.beginObject();
    
    for (const auto& [key, value] : parameters) {
        builder.addKey(key);
        
        // 尝试转换常见类型
        // 注意：std::any的类型检查在运行时进行
        try {
            if (value.type() == typeid(int)) {
                builder.addInt(std::any_cast<int>(value));
            } else if (value.type() == typeid(double)) {
                builder.addNumber(std::any_cast<double>(value));
            } else if (value.type() == typeid(float)) {
                builder.addNumber(static_cast<double>(std::any_cast<float>(value)));
            } else if (value.type() == typeid(bool)) {
                builder.addBool(std::any_cast<bool>(value));
            } else if (value.type() == typeid(std::string)) {
                builder.addString(std::any_cast<std::string>(value));
            } else if (value.type() == typeid(const char*)) {
                builder.addString(std::any_cast<const char*>(value));
            } else {
                // 未知类型，序列化为null
                builder.addNull();
            }
        } catch (...) {
            // 转换失败，序列化为null
            builder.addNull();
        }
    }
    
    builder.endObject();
}

} // namespace ad
} // namespace yt
