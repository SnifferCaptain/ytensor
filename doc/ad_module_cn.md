# 自动微分 (Automatic Differentiation) 模块

## 概述

本模块实现了基于计算图的运行时系统，为YTensor提供自动微分的基础设施。计算图采用节点-边的设计，支持动态构建和执行。

## 设计理念

- **节点 (Node)**: 表示计算图中的算子操作（如加法、乘法、矩阵乘等）
- **边 (Edge)**: 表示有向数据流，连接节点并携带YTensorBase张量数据
- **运行时 (Runtime)**: 管理计算图的构建、执行和序列化

## 特性

✅ **动态计算图**: 无需编译时确定，使用YTensorBase基类实现运行时灵活性  
✅ **拓扑排序**: 自动确定节点执行顺序  
✅ **算子注册**: 灵活的算子执行器注册机制  
✅ **JSON序列化**: 支持计算图的导入导出  
✅ **节点参数**: 支持算子的可配置参数  

## 文件结构

```
include/ad/
├── graph_node.hpp          # 节点定义
├── graph_edge.hpp          # 边定义
├── graph_runtime.hpp       # 运行时引擎
└── graph_serialization.hpp # JSON序列化

src/ad/
├── graph_node.inl          # 节点实现
├── graph_edge.inl          # 边实现
├── graph_runtime.inl       # 运行时实现
└── graph_serialization.inl # 序列化实现

include/ad.hpp              # 模块总入口
```

## 快速开始

### 1. 包含头文件

```cpp
#include "ytensor.hpp"
#include "include/ad.hpp"

using namespace yt;
using namespace yt::ad;
```

### 2. 创建计算图

```cpp
ComputationGraph graph;

// 创建节点
auto node1 = graph.createNode("add1", "add");
auto node2 = graph.createNode("mul1", "mul");

// 创建边并连接节点
auto input1 = graph.createEdge("input1", nullptr, node1);
auto input2 = graph.createEdge("input2", nullptr, node1);
auto edge1 = graph.createEdge("edge1", node1, node2);
auto output = graph.createEdge("output", node2, nullptr);
```

### 3. 注册算子

```cpp
void addOperator(const GraphNode& node,
                 const std::vector<std::shared_ptr<GraphEdge>>& inputs,
                 std::vector<std::shared_ptr<GraphEdge>>& outputs) {
    const auto& t1 = inputs[0]->getTensor();
    const auto& t2 = inputs[1]->getTensor();
    YTensorBase result = t1 + t2;
    outputs[0]->setTensor(result);
}

graph.registerOperator("add", addOperator);
```

### 4. 执行计算图

```cpp
// 准备输入
YTensorBase t1({2, 3}, "float32");
YTensorBase t2({2, 3}, "float32");
// ... 填充数据 ...

std::unordered_map<std::string, YTensorBase> inputs = {
    {"input1", t1},
    {"input2", t2}
};

// 执行
auto outputs = graph.execute(inputs, {"output"});
const auto& result = outputs["output"];
```

### 5. 序列化

```cpp
// 序列化为JSON字符串
std::string json = GraphSerializer::toJson(graph);

// 保存到文件
GraphSerializer::toJsonFile(graph, "graph.json");

// 从文件加载（需要完整的JSON解析器支持）
ComputationGraph newGraph;
GraphSerializer::fromJsonFile("graph.json", newGraph);
```

## API 文档

### ComputationGraph 类

#### 主要方法

- `createNode(id, opType)` - 创建新节点
- `createEdge(id, from, to)` - 创建新边
- `connect(fromId, toId, edgeId)` - 连接两个节点
- `registerOperator(opType, executor)` - 注册算子执行器
- `execute(inputs, outputs)` - 执行计算图
- `topologicalSort()` - 获取拓扑排序结果
- `clear()` - 清空计算图

### GraphNode 类

#### 主要方法

- `getId()` / `getOpType()` / `getName()` - 获取节点信息
- `setParameter(key, value)` - 设置节点参数
- `getParameter(key)` - 获取节点参数
- `getInputEdges()` / `getOutputEdges()` - 获取连接的边

### GraphEdge 类

#### 主要方法

- `getId()` / `getName()` - 获取边信息
- `setTensor(tensor)` / `getTensor()` - 设置/获取张量数据
- `getFromNode()` / `getToNode()` - 获取连接的节点
- `hasTensor()` - 检查是否有张量数据

### GraphSerializer 类

#### 静态方法

- `toJson(graph)` - 序列化为JSON字符串
- `toJsonFile(graph, filename)` - 序列化到文件
- `fromJson(json, graph)` - 从JSON反序列化
- `fromJsonFile(filename, graph)` - 从文件反序列化

## 示例程序

完整的示例程序位于 `example/ad_demo/main.cpp`，演示了：

1. 创建包含3个节点的计算图
2. 注册加法和乘法算子
3. 执行计算: `(input1 + input2) * input3 + input4`
4. 序列化为JSON格式
5. 验证计算结果

运行示例：

```bash
cd example/ad_demo
mkdir build && cd build
cmake ..
make
./ad_demo
```

## JSON格式说明

计算图序列化为JSON格式，包含nodes和edges两个主要部分：

```json
{
  "nodes": [
    {
      "id": "node1",
      "opType": "add",
      "name": "加法节点",
      "inputEdges": ["edge1", "edge2"],
      "outputEdges": ["edge3"],
      "parameters": {}
    }
  ],
  "edges": [
    {
      "id": "edge1",
      "name": "输入边1",
      "fromNode": null,
      "toNode": "node1"
    }
  ]
}
```

## 注意事项

1. **节点ID和边ID必须唯一**：在同一个计算图中，节点ID和边ID不能重复
2. **算子必须先注册**：执行前需要注册所有用到的算子类型
3. **输入边必须有数据**：执行时所有输入边必须设置张量数据
4. **无环要求**：计算图必须是有向无环图(DAG)
5. **JSON反序列化**：当前fromJson实现为简化版本，完整功能建议使用nlohmann/json库

## 后续规划

- [ ] 实现反向传播和梯度计算
- [ ] 支持更多内置算子
- [ ] 优化内存管理
- [ ] 支持子图和条件执行
- [ ] 完整的JSON反序列化实现
- [ ] 计算图可视化工具
- [ ] 性能分析和优化

## 贡献

欢迎提交Issue和Pull Request！

## 许可

遵循YTensor项目的许可协议。
