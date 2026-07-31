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
auto out = yt::strided::broadcast([](
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
支持标准库类型和自定义类型。Typed `YTensor<T, dim>` 数学能力取决于 `T` 是否提供对应运算符；runtime `YTensorBase` 数学能力需要注册相应 `YDTypeKernels`，tensor-scalar转换还可能需要精确cast kernel。
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
yt::type::registerType<MyType>("MyType", [](const void* data) {
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

## 文件结构

```tree
./
├─ doc/                          | 使用指南与 API 文档
│  ├─ en/                        | 英文文档
│  │  ├─ api/                    | 英文 API 文档
│  │  └─ installation/           | 英文安装与构建说明
│  └─ zh/                        | 中文文档
│     ├─ api/                    | 中文 API 文档
│     └─ installation/           | 中文安装与构建说明
├─ example/                      | 数据转换与模型推理示例
│  ├─ convert/                   | YTensor 与其他数据格式的转换工具
│  ├─ qwen3/                     | Qwen3 CPU 推理示例
│  │  ├─ include/                | Qwen3 示例声明与第三方头文件
│  │  ├─ model/                  | Qwen3 配置、权重与分词器资源目录
│  │  └─ src/                    | Qwen3 示例实现
│  └─ ymodel2-s-2/               | YModel2 CPU 推理示例
│     └─ model/                  | YModel2 权重与分词器资源目录
├─ include/                      | YTensor 公共声明与模板接口
│  ├─ 3rd/                       | 项目内使用的第三方头文件
│  ├─ blas/                      | YBLAS 接口与 frame 声明
│  │  └─ kernels/                | YBLAS 物理微内核声明
│  │     └─ avx2/                | AVX2/FMA 微内核声明
│  ├─ function/                  | 神经网络与函数式算子接口
│  ├─ strided/                   | Strided layout 算法接口
│  ├─ type/                      | dtype 与数值类型支持
│  └─ utils/                     | 通用工具接口
├─ src/                          | YTensor 实现
│  ├─ blas/                      | YBLAS frame 与算子实现
│  │  └─ kernels/                | YBLAS 物理微内核实现
│  │     └─ avx2/                | AVX2/FMA 微内核实现
│  ├─ function/                  | 神经网络与函数式算子实现
│  ├─ strided/                   | Strided layout 算法实现
│  ├─ type/                      | dtype 分发与类型实现
│  └─ utils/                     | 通用工具实现
├─ lib/                          | YTensor 库构建目录
│  └─ src/                       | 库实现入口
└─ single-header/                | 单头文件版本与打包工具
```
> YTensor 版本：0.16  
**注意： 当前版本仍在快速迭代中，部分不常用或底层API 可能会有较大变动，请密切关注更新日志。**

---

## 最新更新

- 重新设计 BLAS 后端架构，在原有 AVX2 实现之上建立独立的 YBLAS 层，让上层调用不再依赖具体指令集。
- 优化 Qwen3 的 CPU 推理性能，encode 速度恢复正常。
- 新增卷积算子。

---
如需更多示例、API 细节或贡献建议，欢迎查阅 example/ 目录或提交 issue！
