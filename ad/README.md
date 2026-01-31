# YTensor Automatic Differentiation (AD) Module

## 概述

本模块实现了基于计算图的自动微分功能，支持模型的运行时构建和序列化。

## 特性

- **计算图构建**: 使用节点(Node)和边(Edge)表示计算图
  - 节点: 代表算子(operators)，如线性层、归一化、激活函数等
  - 边: 代表数据流，即张量的传递

- **JSON序列化**: 计算图可以序列化为JSON格式，便于存储和传输
  - 支持保存到文件和从文件加载
  - 完整保存节点属性和拓扑结构

- **灵活性**: 避免了编译时确定的限制，支持运行时动态构建

## 文件结构

```
ad/
├── ad.hpp                      # 主头文件
├── graph_node.hpp/cpp          # 节点类定义
├── graph_edge.hpp/cpp          # 边类定义
├── computation_graph.hpp/cpp   # 计算图类
├── graph_executor.hpp/cpp      # 图执行器
├── ymodel2_graph_builder.hpp/cpp # YModel2图构建器
├── test_graph.cpp              # 测试程序
├── CMakeLists.txt              # 构建配置
└── README.md                   # 本文件
```

## 使用示例

### 1. 构建计算图

```cpp
#include "ad/ad.hpp"
#include "ad/ymodel2_graph_builder.hpp"

// 构建YModel2-s-2的计算图
auto graph = yt::ad::YModel2GraphBuilder::buildYModel2S2Graph();
```

### 2. 序列化为JSON

```cpp
// 保存到文件
graph.saveToFile("model_graph.json");

// 或者获取JSON字符串
std::string json = graph.toJSON();
```

### 3. 从JSON加载

```cpp
// 从文件加载
auto loadedGraph = yt::ad::ComputationGraph::loadFromFile("model_graph.json");
```

### 4. 遍历计算图

```cpp
// 获取所有节点
for (const auto& [id, node] : graph.nodes()) {
    std::cout << "Node: " << node->name() 
              << " Type: " << yt::ad::nodeTypeToString(node->type()) 
              << std::endl;
}

// 拓扑排序
auto sortedNodes = graph.topologicalSort();
```

## 节点类型

支持的节点类型包括:

- **Input**: 输入节点
- **Parameter**: 参数节点(可学习的权重)
- **Constant**: 常量节点
- **Linear**: 线性变换
- **Embedding**: 词嵌入
- **RMSNorm**: RMS归一化
- **GELU**: GELU激活函数
- **Attention**: 注意力机制
- **FFN**: 前馈神经网络
- **Add**: 加法操作
- **Multiply**: 乘法操作
- 以及其他张量操作(Slice, Reshape, Permute等)

## YModel2-s-2 示例

YModel2-s-2是一个小型语言模型,其配置为:
- 层数: 4
- 隐藏维度: 512
- 注意力头数: 8
- 头维度: 64
- FFN中间层大小: 1024
- 词表大小: 6400

构建的计算图包含62个节点和69条边,完整描述了模型的前向传播过程。

## 编译和测试

```bash
cd ad
mkdir build && cd build
cmake ..
make -j8
./test_graph
```

测试程序会:
1. 构建YModel2-s-2的计算图
2. 序列化为JSON
3. 保存到文件
4. 从文件加载
5. 验证序列化/反序列化的正确性
6. 执行拓扑排序

## JSON格式示例

```json
{
  "nodes": [
    {
      "id": 0,
      "name": "input_ids",
      "type": "Input",
      "attributes": {
        "shape": "[batch, seq_len]",
        "dtype": "int32"
      },
      "inputs": [],
      "outputs": [2]
    },
    {
      "id": 1,
      "name": "model.embed_tokens.weight",
      "type": "Parameter",
      "attributes": {
        "shape": "[6400, 512]",
        "dtype": "float32"
      },
      "inputs": [],
      "outputs": [2]
    }
  ],
  "edges": [
    {
      "id": 0,
      "from": 0,
      "to": 2,
      "name": "",
      "shape": [],
      "dtype": ""
    }
  ]
}
```

## 未来扩展

- [ ] 实现完整的图执行器
- [ ] 支持反向传播和梯度计算
- [ ] 优化图优化(如算子融合)
- [ ] 支持更多的算子类型
- [ ] 参数的实际数据存储和加载
