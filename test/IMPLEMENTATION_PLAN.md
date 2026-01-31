# 计算图实现计划

## 目标
实现能够加载JSON并执行的计算图，输出必须与example/ymodel2-s-2完全一致（数值精确匹配）

## 当前状态
✅ 已完成基础框架：
- `include/ad/graph.hpp` - 图结构定义
- `src/ad/graph.cpp` - 基础实现（序列化、拓扑排序）
- `test/test_ad_graph.cpp` - 基础测试通过

## 需要实现的内容（大量工作）

### 1. 算子执行逻辑
需要在`src/ad/graph.cpp`的`execute()`函数中实现所有算子：

- [ ] **Embedding** - 词嵌入查找
- [ ] **Linear** - 线性变换（矩阵乘法 + 可选bias）
- [ ] **RMSNorm** - RMS归一化
- [ ] **Attention** (PEGA2) - 完整的注意力机制
  - QKV投影
  - RoPE位置编码
  - 注意力计算
  - 输出投影
- [ ] **FFN** - 前馈神经网络（含GELU激活）
- [ ] **Add** - 残差连接
- [ ] 其他辅助操作（slice, reshape, permute等）

### 2. 权重加载
- [ ] 从`.yt`文件加载模型参数
- [ ] 将参数关联到图中的Parameter节点
- [ ] 确保参数形状正确

### 3. YModel2图构建器
- [ ] 创建`include/ad/ymodel2_builder.hpp`
- [ ] 实现`src/ad/ymodel2_builder.cpp`
- [ ] 从YConfig2配置构建完整的transformer图结构
- [ ] 支持scale=-2配置（4层，512隐藏维度）

### 4. KV Cache支持
- [ ] 在图执行中支持KV缓存
- [ ] 处理自回归生成

### 5. 精确数值匹配
- [ ] 相同的随机种子初始化
- [ ] 相同的数值精度
- [ ] 与ymodel2-s-2逐个token比对输出

## 工作量评估
这是一个**完整的推理引擎实现**，预计需要：
- 核心算子实现：500-1000行代码
- 图构建器：200-300行代码
- 权重加载：100-200行代码  
- 测试与调试：大量时间

## 测试要求
```cpp
// 伪代码
Graph g = Graph::fromJSON("ymodel2_s2.json");
loadWeights(g, "model/y2_sft_s-2.yt");

auto output = g.execute(inputs);
// output 必须与 example/ymodel2-s-2 的输出完全一致
```

## 现状
基础框架已就绪，但实际执行功能需要大量算子实现工作。
