# YTensor

> 现代 C++20 轻量级多维张量库 —— 默认 header-only，支持 single-header 打包与 `YT_USE_LIB` 加速后端

## 特性亮点

- 🧩 **Header-only**：直接包含 `ytensor.hpp` 或 `ytensor_single.hpp` 即用，零第三方依赖
- 📐 **任意维度**：支持 shape、切片、转置、reshape、permute 等常用操作
- ⚡ **多功能**：支持并行、广播、基础数学、常用深度学习函数、多轴归一化，以及 SDPA / Flash Attention
- 🛠️ **易扩展**：源码清晰，便于二次开发
- 🗂️ **I/O 支持**：可保存/加载 YTensor 支持的文件
- 🔢 **广泛的类型支持**：支持常用的数据类型，如 float16，bfloat16等。

---

## 运行示例

### 🚀 简单的语言模型推理
YTensor的功能并非纸上谈兵！此项目已经在example内完成了对基于transformer架构的语言模型提供了在cpu上的推理示例。运行流程如下：
1. 前往huggingface下载模型权重文件。模型网址为[https://huggingface.co/collections/SnifferCaptain/ymodel2](https://huggingface.co/collections/SnifferCaptain/ymodel2)。可以选择的模型有：  

| 模型名称 | 参数量 | 说明 |
| --- | --- | --- |
| ymodel2-s-2 | 11M | 基于YModel2架构的最小语言模型，具备最基础的问答能力，适合入门和测试。 |
| ymodel2-s0 | 99M | 中等规模的YModel2架构语言模型,在数学、代码能力、自然知识方面都具备更好的性能。|

2. 点击页面中的Files and versions，查看仓库中的文件。下载`tokenlizer.json`, `tokenlizer_config.json`，以及`y2_sft_s-2.yt`（如果是ymodel2-s0的话，下载`y2_sft_s-0.yt`）。将下载的三个文件放到example/ymodel2-s-2/model目录下。  

3. 打开main.cpp，根据下载的模型，设置模型的大小配置：
```cpp
// 初始化模型配置
ymodel2::YConfig2 config;
config.scale_lvl(-2);       // <-- 如果下载的模型为ymodel2-s-2的话，使用-2；如果是ymodel2-s0的话，使用0。
```

4. 编译运行：
首先来到目录example/ymodel2-s-2下，执行以下命令：
```bash
mkdir build
cd build
cmake ..
make -j8
./ymodel2-s-2
```
5. 运行结果如下（使用ymodel2-s-2）：
```text
===SnifferCaptain Chat===

loading tokenizer...
tokenizer loaded successfully.
loading model weights...
  File contains 31 tensors
  lm_head.weight found in file (shared with embed_tokens)
  Successfully loaded 31/31 tensors
model loaded successfully.
using backend: naive
===============recipe================
  send your message with [Enter]
  'exit' or 'quit' to exit
  'clear' to clear chat history
======================================

You: 你好，SnifferCaptain！你可以成为我的得力助手吗？
SnifferCaptain: 作为一名程序员，我可以给你提供一些建议，帮助你成为一名成功的用户。以下是一些建议和技巧，可以帮助你开始你的需求：

1. **明确目标**：首先，明确你的社交圈。了解你希望通过的主题或故事来构建你的社交圈，以建立起您的信任和友谊。

2. **设计故事**：根据你的需求设计一些主题，如故事、科学、历史故事、故事背景等等。不同的视角可以帮助你更好地理解故事的主题和思想。

3. **个性化体验**：考虑你的兴趣爱好、兴趣爱好或是职业兴趣。这可以包括兴趣爱好、运动爱好、文化体验等。同时，提供多样化的视角，帮助你更深入地了解你所在群体的兴趣和经历。

4. **持续学习**：技术和知识是学习新技能的关键。持续学习新知识和技能可以帮助你保持竞争力。同时，关注用户反馈，了解行业动态。

5. **适应性和灵活性**：学习新的技能可能会带来新的创作机会。尝试新的艺术形式，或者与其他艺术家交流，这可以提供灵感。

6. **建立人脉**：尝试与您有相似兴趣的人建立联系，可以减少误解和欺骗。这可以提供情感支持和鼓励，帮助你扩大你的影响力。

7. **持续学习**：技术的发展是一个持续的过程。尝试新的技术、新方法或方法，保持好奇心。持续学习新技能、新技能，你会逐渐提升自己的音乐水平。

8. **自我激励**：学习新技术和知识是成功的关键因素。参加社区活动，无论是通过在线课程、教学技巧，还是编程社区。这些活动能够帮助你保持竞争力，从而激发你的创造力。

9. **持续学习和成长**：编程是一个持续的过程，需要时间和努力。保持好奇心和积极的心态，不断学习和成长。

10. **保持耐心**：学习并尝试新的技能，这可能会让你更加坚强。保持积极的心态对前提的深好保持信心，而不是失败。

11. **保持耐心和毅力**：学习新技术需要时间。不要让任何人知道你希望获得某些东西。保持耐心和毅力，相信自己可以克服困难。

每个人的体验都是独一无二的，每个人的价值和意义都会影响你的成功和成就。重要的是保持开放和积极的态度。
[Info] encoding length: 43, decoding length: 576, encoding speed: 519.088 tokens/s, decoding speed: 193.069 tokens/s
       context length: 619/8192 tokens
```

## YTensor 基础用法

```cpp
// 1. 初始化张量
yt::YTensor<float, 2> a(3, 4);                  // 操作将会预先分配形状为[3, 4]的张量，元素使用默认初始化
auto b = yt::YTensor<float, 3>::ones(2, 3, 4);  // 全1初始化，形状为[2, 3, 4]

// 2. 访问和修改元素
float v1 = a.at(1, 2);  // 推荐：快速访问位于(1, 2)的元素
float v2 = a[1][2];     // 也可以使用下标访问位于(1, 2)的元素
b.at(0, 1, 2) = 42.f;   // 可以赋值位于(0, 1, 2)的元素

// 3. 获取形状与大小
std::vector<int> shape = b.shape(); // 获取形状
size_t sz = b.size();               // 元素总数

// 4. 视图与变换
auto b2 = b.view(6, -1);    // 将[2, 3, 4]张量转换成[6, 4]张量，其中-1表示根据元素数量自动计算的轴长。
b2 = b2.permute(1, 0);      // 交换维度，此时b2的形状为[4, 6]
b2 = b2.contiguous();       // 将b2转换为连续张量，也等价于b2.contiguous_()原地操作。

// 5. 打印 shape
std::cout << "shape: ";
for (int d : b2.shape()){
    std::cout << d << " ";
}
std::cout << std::endl;
```

> 上述代码展示了 YTensor 的常见初始化、元素访问、形状获取与视图操作。更多用法见下方详细分块与 example/。

---

## 三种使用形态（接口完全等价）

YTensor 现在保持三种并存形态，外部 API 与用法保持一致：

- **默认 header-only**（标准形态）：不定义任何库后端宏，直接包含 `ytensor.hpp` 或 `ytensor_single.hpp`。
- **single-header**（发布打包形态）：使用 `single-header/ytensor_single.hpp`，行为与默认 header-only 一致。
- **`YT_USE_LIB`**（编译加速后端）：定义 `YT_USE_LIB=1` 并链接 `libytensor`，仅改变实现归属以减少重复编译。

> `YT_USE_LIB` 的唯一额外要求是 **链接库**；API、返回类型、支持类型集合与自定义类型能力保持一致。

---

## 🧩 Header-only 零依赖
只需下载 `single-header/ytensor_single.hpp`，放到你的工程目录：

```cpp
#include "ytensor_single.hpp"   // 仅需包含此文件即可

int main() {
    auto a = yt::YTensor<float, 2>::randn(3, 4);    // [3, 4] 正态分布随机张量
    std::cout << a << std::endl;                    // 打印张量详细信息
    return 0;
}
```
> 只需一个头文件，零依赖，可以快速在任意 C++20 项目中使用。
> 
> **注意：** YTensor 使用了大量 C++20 特性，请确保你的编译器支持 C++20。

### `YT_USE_LIB` 加速后端示例

如果你希望减少用户侧重复编译，可使用预编译库后端：

```cpp
#define YT_USE_LIB 1
#include "ytensor.hpp"   // 或 include <lib/include/ytensor.hpp>
```

```bash
# 示例：链接 libytensor（Linux）
g++ -std=c++20 -O2 -fopenmp main.cpp \
  -I/path/to/ytensor \
  -I/path/to/ytensor/lib/include \
  -L/path/to/ytensor/lib/bin \
  -Wl,-rpath,/path/to/ytensor/lib/bin \
  -lytensor -lz -o main
```

---

## 📐 任意维度与常用操作
支持 shape、切片、转置、reshape、permute 等常用张量操作。

```cpp
yt::YTensor<float, 3> t(4, 5, 3);   // 构造一个4x5x3的张量
t.fill(1.0f);                       // 将所有元素设为 1.0

// 链式单语句写法
auto sliced = t
    .permute(1, 2, 0)    // 调换轴的排列顺序：[4, 5, 3] -> [5, 3, 4]
    .contiguous()        // 保证内存连续性，是view使用的必要条件。
    .view(15, -1)        // 将前两个轴融合 [3, 4, 5] -> [15, 4]
    .slice(1, 1, 3, 1);  // 在轴 1 上切取索引1、2，[15, 4] -> [15, 2]

// 查看形状
const auto& s = sliced.shape();
std::cout << "shape: [";
for (size_t i = 0; i < s.size(); ++i) {
    std::cout << s[i];
    if (i + 1 < s.size()) std::cout << ", ";
}
std::cout << "]\n";
```
---

## ⚡ 多功能：支持多种计算

```cpp
// ReLU 激活（直接使用库函数）
auto x = yt::YTensor<float, 2>::randn(3, 4);    // 随机初始化x: [3, 4]
auto y = yt::YTensor<float, 2>::randn(1, 4);    // 随机初始化y: [1, 4]
auto reluOutput = yt::function::relu(x);        // 逐元素 relu。reluOutput: [3, 4]

// 支持高自由度的元素级广播计算。siluOutput: [3, 4]
auto siluOutput = x.broadcastInplace([](float& v) {
    float s = 1.0f / (1.0f + std::exp(-v)); // sigmoid
    v = v * s;
});

// 支持元素级别的多元自定义计算。out: [3, 4]
auto out = yt::kernel::broadcast([](
    const float& t1,    // 来自张量x的元素
    const float& t2,    // 来自张量y的元素
    const float& t3,    // 来自张量reluOutput的元素
    const float& t4,    // 来自张量siluOutput的元素
    const float& s5,    // 来自常量
){
    return t1 + t2 + t3 + t4 + s5;
}, x, y, reluOutput, siluOutput, 0.5f); // 输入需要与函数的参数一一对齐

// 也支持一些符号广播运算
out += y - 0.1f;

std::cout << "ReLU output:\n" << reluOutput << std::endl;
std::cout << "SiLU output:\n" << siluOutput << std::endl;
std::cout << "Custom output:\n" << out << std::endl;
```

支持常用操作，对自定义操作有极高的自由度，示例展示了库原语在构建自定义算子时的灵活性与高扩展性。

---
## 🦾 多类型支持
支持多种数据类型的张量，包括标准库类型float、std::string等，也支持自定义类型，能否使用取决于是否对类型进行对应运算符符的重载。
```cpp
yt::YTensor<std::string, 2> strTensor(3, 4);    // 创建一个3x4的std::string类型张量
strTensor.fill("hello");                        // 初始化为"hello"
strTensor += " world";                          // 广播加法（字符串拼接）
std::cout << strTensor << std::endl;            // 打印张量
std::cout << strTensor[0][0] << std::endl;      // 访问元素

// 对于自定义类型，提供类型注册机制
struct MyType {
    int value = 0;
    MyType operator+(const MyType& other) const{
        return MyType{value - other.value};// 只要有自定义运算符即可支持相应运算
    }
    // ...
};

// 注册类型，并提供字符串的转换函数。需要提供类型名称与类型转换函数（可选，影响打印输出）
yt::types::registerType<MyType>("MyType", [](const void* data) {
    const MyType* p = reinterpret_cast<const MyType*>(data);// 转为MyType指针
    return std::to_string(p->value + 1);// 直接使用value+1为打印内容
});

yt::YTensor<MyType, 2> myTensor(2, 3);
static int i = 0;
myTensor.foreach([&](auto& x){
    x.value = i++;
});
myTensor += MyType{5};
myTensor[0][0].value = 114513;
std::cout << myTensor << std::endl;
/*输出示例：
[YTensor]:<MyType>
[itemSize]: 6
[byteSize]: 24
[shape]: [2, 3]
[data]:
[
  [114514 -3 -2]
  [-1 0 1]
]
*/
```

---

## 🗂️ I/O 支持
可保存/加载自定义二进制格式，支持压缩（需 zlib）。适合高效序列化与跨平台数据交换。

```cpp
yt::io::verbose = true;     // 打印详细信息（默认关闭）
yt::io::compressMethod = "";// 不压缩（默认不压缩，zlib对浮点数的压缩效果并不好）

auto t0 = yt::YTensor<float, 2>::randn(3, 4);
auto t1 = yt::YTensor<float, 3>::randn(5, 6, 7);
yt::io::YTensorIO io;               // 创建文件IO对象
io.open("./test.yt", yt::io::Write);// 打开文件，写模式
io.save(t0, "name0");               // 保存张量到文件
io.save(t1, "name1");               // 支持多张量保存
io.close();                         // 关闭文件，写入磁盘

// 加载文件
io.open("./test.yt", yt::io::Read); // 打开文件，读模式
yt::YTensor<float, 2> read0;
yt::YTensor<float, 3> read1;
yt::YTensorBase base0;
io.load(read0, "name0");    // 加载张量，注意数据类型（<float, 2>）需要匹配
io.load(read1, "name1");    // 数据类型<float, 3>
io.load(base0, "name0");    // 也可以加载到YTensorBase内
io.close();

// 支持非POD类型（如 std::string）I/O；自定义类型可通过 registerType 提供序列化/反序列化后进行保存与加载。
```
同时，还可以使用example/convert目录下的转换函数。实现部分的数据格式转换（如numpy等）。
> 适合模型权重、数据集等便捷存储。

---
## 📦 可选预编译库

定义 `YT_USE_LIB=1` 并链接库，可减少大型项目的重复编译开销，API 与 header-only 保持一致。

首先构建库：
```bash
mkdir build && cd build
cmake .. && make -j8
```

库文件输出到 `lib/bin/`。使用时定义宏并链接：
```cpp
#define YT_USE_LIB 1
#include "ytensor.hpp"
```

```bash
# Linux 链接示例
g++ -std=c++20 -O2 -fopenmp main.cpp \
  -I/path/to/ytensor -I/path/to/ytensor/lib/include \
  -L/path/to/ytensor/lib/bin -Wl,-rpath,/path/to/ytensor/lib/bin \
  -lytensor -lz -o main
```

> **提示**：库模式下统一使用 `YTensor<T>` 而非 `YTensorBase`，以获得对自定义类型的完整支持。

---

## 文件结构

```tree
./
├─ example/                                         | 示例代码
│  ├─ convert/                                      | 数据格式转换脚本
│  │   ├─ __init__.py                               |
│  │   ├─ numpy2yt.py                               | numpy格式转换为ytensor格式[待完善]
│  │   ├─ safetensors2yt.py                         | safetensors格式转换为ytensor格式[待完善]
│  │   └─ ytfile.py                                 | ytensor文件类[待完善]
│  ├─ qwen3/                                        | Qwen3推理示例
│  │   ├─ CMakeLists.txt                            | 构建配置文件
│  │   ├─ include/                                  |
│  │   │   ├─ json.hpp                              | nlohmann json解析库
│  │   │   ├─ qwen3.hpp                             | Qwen3 模型接口与推理封装声明
│  │   │   └─ tokenlizer.hpp                        | 分词器
│  │   ├─ main.cpp                                  | 程序入口
│  │   ├─ model/                                    | 模型权重与分词器文件存放目录
│  │   │   ├─ config.json                           | Qwen3 模型配置文件
│  │   │   ├─ tokenizer.json                        | 分词器词表文件
│  │   │   └─ tokenizer_config.json                 | 分词器配置文件
│  │   └─ src/                                      | 源码实现目录
│  │       ├─ qwen3.cpp                             | Qwen3 模型推理实现
│  │       └─ tokenlizer.cpp                        | 分词器实现
│  └─ ymodel2-s-2/                                  | 推理ymodel2语言模型示例
│      ├─ CMakeLists.txt                            |
│      ├─ json.hpp                                  | nlohmann json解析库
│      ├─ main.cpp                                  | 主程序入口
│      ├─ model/                                    | 模型权重、分词器文件存储
│      │   ├─ tokenizer.json                        | 词表文件，需要从huggingface下载
│      │   ├─ tokenizer_config.json                 | 分词器配置文件，需要从huggingface下载
│      │   └─ y2_sft_s-2.yt                         | 模型权重文件，需要从huggingface下载
│      ├─ tokenlizer.cpp                            | 分词器实现
│      ├─ tokenlizer.hpp                            | 分词器头文件
│      ├─ ymodel2.cpp                               | 模型实现✨
│      └─ ymodel2.hpp                               | 模型头文件✨
├─ include/                                         | 头文件目录
│  ├─ 3rd/                                          | 第三方依赖
│  │   └─ backward.hpp                              | google堆栈追踪库，可以移除，方便调试用
│  ├─ function/                                     | 函数式子模块
│  │   ├─ activation.hpp                            | 激活相关函数声明
│  │   ├─ loss.hpp                                  | 损失函数声明
│  │   ├─ normalization.hpp                         | 归一化相关函数声明
│  │   └─ ops.hpp                                   | 通用算子与融合算子声明
│  ├─ kernel/                                       | 内核实现
│  │   ├─ avx2/                                     | AVX2 内核
│  │   │   ├─ flash_attention.hpp                   | Flash Attention 内核接口
│  │   │   ├─ gemm_utils.hpp                        | AVX2 GEMM 辅助工具
│  │   │   ├─ hdot.hpp                              | 半精度点积内核
│  │   │   ├─ hgemm.hpp                             | 半精度 GEMM
│  │   │   ├─ hgemv.hpp                             | 半精度 GEMV
│  │   │   ├─ hger.hpp                              | 半精度 GER
│  │   │   ├─ sdot.hpp                              | 单精度点积内核
│  │   │   ├─ sgemm.hpp                             | 单精度 GEMM
│  │   │   ├─ sgemv.hpp                             | 单精度 GEMV
│  │   │   └─ sger.hpp                              | 单精度 GER
│  │   ├─ broadcast.hpp                             | 广播运算
│  │   ├─ memory_utils.hpp                          | 内存分配
│  │   ├─ parallel_for.hpp                          | 并行循环
│  │   └─ type_dispatch.hpp                         | 类型分发辅助
│  ├─ types/                                        | 类型相关
│  │   ├─ bfloat16.hpp                              | bfloat16类型支持
│  │   └─ float_spec.hpp                            | 多种浮点数类型支持
│  ├─ ytensor_base.hpp                              | YTensor基类
│  ├─ ytensor_base_math.hpp                         | YTensor基类数学操作
│  ├─ ytensor_concepts.hpp                          | 类型检查等概念
│  ├─ ytensor_core.hpp                              | YTensor核心类
│  ├─ ytensor_function.hpp                          | 函数式编程
│  ├─ ytensor_extern_templates.hpp                  | 预实例化模板声明
│  ├─ ytensor_infos.hpp                             | 全局设置信息
│  ├─ ytensor_io.hpp                                | 文件存储系统
│  ├─ ytensor_math.hpp                              | YTensor数学操作
│  └─ ytensor_types.hpp                             | 类型相关
├─ lib/                                             | `YT_USE_LIB` 预编译后端
│  ├─ bin/                                          | 库产物目录（如 libytensor.so）
│  ├─ CMakeLists.txt                                | 构建配置
│  └─ src/                                          |
│      └─ ytensor_library.cpp                       | 库实现入口
├─ single-header/                                   | 单头文件版本
│  ├─ ytensor_single.hpp                            | 单头文件版本的YTensor，包含所有功能
│  └─ packer.py                                     | 单头文件打包脚本
├─ src/                                             | 源文件实现目录
│  ├─ function/                                     | 函数式实现
│  │   ├─ activation.inl                            | 激活相关函数实现
│  │   ├─ loss.inl                                  | 损失函数实现
│  │   ├─ normalization.inl                         | 归一化相关函数实现
│  │   └─ ops.inl                                   | 通用算子与融合算子实现
│  ├─ kernel/                                       | 底层内核实现
│  │   ├─ avx2/
│  │   │   ├─ flash_attention.inl                   | AVX2 Flash Attention 内核
│  │   │   ├─ gemm_utils.inl                        | AVX2 GEMM 辅助实现
│  │   │   ├─ hdot.inl                              | 半精度点积内核
│  │   │   ├─ hgemm.inl                             | 半精度 GEMM 内核
│  │   │   ├─ hgemv.inl                             | 半精度 GEMV 内核
│  │   │   ├─ hger.inl                              | 半精度 GER 内核
│  │   │   ├─ sdot.inl                              | 单精度点积内核
│  │   │   ├─ sgemm.inl                             | AVX2 GEMM 内核
│  │   │   ├─ sgemv.inl                             | AVX2 GEMV 内核
│  │   │   └─ sger.inl                              | 单精度 GER 内核
│  │   ├─ broadcast.inl                             | 广播内核实现
│  │   ├─ memory_utils.inl                          | 内存工具实现
│  │   ├─ parallel_for.inl                          | 并行循环实现
│  │   └─ type_dispatch.inl                         | 类型分发实现
│  ├─ ytensor_base.inl                              | YTensor基类
│  ├─ ytensor_base_math.inl                         | YTensor基类数学操作
│  ├─ ytensor_base_templates.inl                    | YTensor基类模板实例化
│  ├─ ytensor_core.inl                              | YTensor实现
│  ├─ ytensor_function.inl                          | YTensor函数式编程
│  ├─ ytensor_io.inl                                | YTensor文件存储系统
│  ├─ ytensor_io_templates.inl                      | YTensor文件存储模板实例化
│  └─ ytensor_math.inl                              | YTensor数学操作
└─ ytensor.hpp                                      | 主头文件，包含所有必要的头文件
```
> YTensor 版本：0.13  
**注意： 当前版本仍在快速迭代中，部分不常用或底层API 可能会有较大变动，请密切关注更新日志。**

---

## 最新更新

- 整理代码结构。
- 新增多个常用函数。
- 修复order接口，现在只有逐元素的操作会保留order的导数/积分接口。
- 新增SDPA后端FLASH_AVX2，采用flash attention算法加速注意力计算。

---
如需更多示例、API 细节或贡献建议，欢迎查阅 example/ 目录或提交 issue！
