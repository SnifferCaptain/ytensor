# YTensor

> 现代 C++20 轻量级多维张量库 —— header-only，极简集成，科研/竞赛/工程皆宜

## 特性亮点

- 🧩 **Header-only**：只需 `ytensor_single.hpp`，零第三方依赖，直接 `#include` 即用
- 📐 **任意维度**：支持 shape、切片、转置、reshape、permute 等常用操作
- ⚡ **多功能**：支持并行、广播、基础线性代数、常用的深度学习函数（matmul/softmax/attention）
- 🛠️ **易扩展**：源码清晰，便于二次开发
- 🗂️ **I/O 支持**：可保存/加载 YTensor 支持的文件

---

## YTensor 基础用法

```cpp
// 1. 初始化张量
YTensor<float, 2> a(3, 4); // 操作将会预先分配形状为[3, 4]的张量，元素使用默认初始化
auto b = YTensor<float, 3>::ones(2, 3, 4); // 全1初始化

// 2. 访问和修改元素
float v1 = a.at(1, 2);  // 推荐：快速访问位于(1, 2)的元素
float v2 = a[1][2];     // 也可以使用下标访问位于(1, 2)的元素
b.at(0, 1, 2) = 42.f;   // 可以赋值位于(0, 1, 2)的元素

// 3. 获取形状与大小
std::vector<int> shape = b.shape(); // 获取形状
size_t sz = b.size();               // 元素总数

// 4. 视图与变换
auto b2 = b.view(6, -1);    // 将[2, 3, 4]张量转换成[6, 4]张量，其中-1表示根据元素数量自动计算的轴长
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

## 🧩 Header-only 零依赖
只需下载 `single-header/ytensor_single.hpp`，放到你的工程目录：

```cpp
#include "ytensor_single.hpp"

int main() {
    YTensor<float, 2> a = YTensor<float, 2>::randn(3, 4);   // [3, 4] 正态分布随机张量
    std::cout << a << std::endl;                            // 打印张量详细信息
    return 0;
}
```
> 只需一个头文件，零依赖，可以快速在任意 C++20 项目中使用。
> 
> **注意：** YTensor 使用了大量 C++20 特性，请确保你的编译器支持 C++20。

---

## 📐 任意维度与常用操作
支持 shape、切片、转置、reshape、permute 等常用张量操作。

```cpp
// 基础示例：构造并填充一个 3x4x5 张量
YTensor<float, 3> t(3, 4, 5);
t.fill(1.0f); // 将所有元素设为 1.0

// 链式单语句写法（每个函数调用换行，避免显式中间变量）
auto sliced = t
    .permute(1, 2, 0)    // 新维度顺序：原来的 (1,2,0)
    .contiguous()        // 保证连续性（必要时会拷贝）
    .view(15, -1)        // reshape：将 [3,4,5] -> [15,4]
    .slice(1, 1, 3, 1);  // 在轴 1 上切取索引2、3，[15, 4] -> [15, 2]

// 5) 打印 shape（更可读的格式）
const auto& s = t_view.shape();
std::cout << "shape: [";
for (size_t i = 0; i < s.size(); ++i) {
    std::cout << s[i];
    if (i + 1 < s.size()) std::cout << ", ";
}
std::cout << "]\n";
```
---

## ⚡ 多功能：ReLU 与 SiLU 示例

```cpp
// ReLU 激活（直接使用库函数）
YTensor<float, 2> x = YTensor<float, 2>::randn(3, 4); // 随机初始化
YTensor<float, 2> reluOutput = x.relu(); // 逐元素 relu

// 该方法在每个元素上直接计算 sigmoid(a) 并把 a 替换为 a * sigmoid(a)
auto siluOutput = a.binaryOpTransform(0.0f, [](float& v, const float&) {
    float s = 1.0f / (1.0f + std::exp(-v)); // sigmoid
    v = v * s;
    return v;
});

std::cout << "ReLU output:\n" << reluOutput << std::endl;
std::cout << "SiLU output:\n" << siluOutput << std::endl;
```

支持常用激活与归一化操作，示例展示了库原语在构建自定义算子时的灵活性与高扩展性。

---

## 🗂️ I/O 支持
可保存/加载自定义二进制格式，支持压缩（需 zlib）。适合高效序列化与跨平台数据交换。

```cpp
yt::io::verbose = true;     // 打印详细信息（默认关闭）
yt::io::compressMethod = "";// 不压缩（默认不压缩）

YTensor<float, 2> t0 = YTensor<float, 2>::randn(3, 4);
YTensor<float, 3> t1 = YTensor<float, 3>::randn(5, 6, 7);
yt::io::YTensorIO io;               // 创建文件IO对象
io.open("./test.yt", yt::io::Write);// 打开文件，写模式
io.save(t0, "name0");               // 保存张量到文件
io.save(t1, "name1");               // 支持多张量保存
io.close();                         // 关闭文件

// 加载文件
io.open("./test.yt", yt::io::Read); // 打开文件，读模式
YTensor<float, 2> read0;
YTensor<float, 3> read1;
io.load(read0, "name0");    // 加载张量，注意数据类型（<float, 2>）需要匹配
io.load(read1, "name1");    // 数据类型<float, 3>
io.close();
```
> 适合模型权重、数据集等高效存储。

---

## 进阶用例

### 单头注意力前向传播（基础算子实现）

```cpp
int batch = 4;  // 参数设置
int seq = 512;
int dim = 512;

YTensor<float, 3> x   = YTensor<float, 3>::randn(batch, seq, dim); // 输入
YTensor<float, 2> w_q = YTensor<float, 2>::randn(dim, dim);        // Q权重
YTensor<float, 2> w_k = YTensor<float, 2>::randn(dim, dim);        // K权重
YTensor<float, 2> w_v = YTensor<float, 2>::randn(dim, dim);        // V权重
YTensor<float, 2> w_o = YTensor<float, 2>::randn(dim, dim);        // O权重

// Q, K, V 线性变换（无偏置）
YTensor<float, 3> q = yt::function::matmul(x, w_q);
YTensor<float, 3> k = yt::function::matmul(x, w_k);
YTensor<float, 3> v = yt::function::matmul(x, w_v);

// Attention: QK^T -> softmax -> V
YTensor<float, 3> score = yt::function::matmul(q, k.transpose());   // QK^T
float scale = 1.0f / std::sqrt(static_cast<float>(q.shape(-1)));    // 缩放因子
score.binaryOpTransformInplace(scale, [](float& a, const float& b) {
    a *= b;
});
score = yt::function::softmax(score, -1);                   // 对最后一个维度进行 softmax
YTensor<float, 3> attn = yt::function::matmul(score, v);    // score @ V
YTensor<float, 3> out  = yt::function::matmul(attn, w_o);   // 输出
```
> 该示例展示了用基础算子手写 attention 前向传播，权重全部用 randn 初始化，便于理解和自定义。

---

## 文件结构

```markdown
./
├─ include/
│  ├─ ytensor_base.hpp
│  ├─ ytensor_concepts.hpp
│  ├─ ytensor_function.hpp
│  ├─ ytensor_infos.hpp
│  ├─ ytensor_io.hpp
│  ├─ ytensor_math.hpp
│  ├─ ytensor_types.hpp
│  ├─ 3rd/
│  └─ types/
│      └─ bfloat16.hpp
├─ single-header/
│  ├─ ytensor_single.hpp
│  └─ packer.py
├─ src/
│  ├─ ytensor.inl
│  ├─ ytensor_base.inl
│  ├─ ytensor_function.inl
│  ├─ ytensor_io.inl
│  └─ ytensor_math.inl
├─ example/
│  ├─ numpy2yt/
│  │   └─ converter.py
│  └─ train_cifar10/
│      ├─ CMakeLists.txt
│      └─ train_cifar.cpp
└─ ytensor.hpp
```
---
如需更多示例、API 细节或贡献建议，欢迎查阅 example/ 目录或提交 issue！

