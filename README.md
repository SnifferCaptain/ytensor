# YTensor

> A modern lightweight C++20 multidimensional tensor library - header-only by default, with optional single-header packaging and a `YT_USE_LIB` accelerated backend

## Highlights

- 🧩 **Header-only**: just include `ytensor.hpp` or `ytensor_single.hpp`, with zero third-party runtime dependencies
- 📐 **Arbitrary dimensions**: supports common tensor operations such as shape, slicing, transpose, reshape, and permute
- ⚡ **Multi-purpose**: supports parallelism, broadcasting, basic math, common deep learning ops, multi-axis normalization, and SDPA/Flash Attention
- 🛠️ **Easy to extend**: clear source layout, convenient for secondary development
- 🗂️ **I/O support**: save/load files supported by YTensor
- 🔢 **Broad type support**: supports common data types such as float16 and bfloat16

---

## Run Example

### 🚀 Simple language model inference
YTensor is not just theory. This project already provides a CPU inference example for a Transformer-based language model in `example/`. The workflow is:

1. Go to Hugging Face and download model weights from [https://huggingface.co/collections/SnifferCaptain/ymodel2](https://huggingface.co/collections/SnifferCaptain/ymodel2). Available models:

| Model Name | Params | Description |
| --- | --- | --- |
| ymodel2-s-2 | 11M | The smallest language model based on the YModel2 architecture, with basic Q&A capability. Good for getting started and testing. |
| ymodel2-s0 | 99M | A medium-scale YModel2 model with better performance in math, coding, and general knowledge. |

2. Click **Files and versions** on the model page. Download `tokenlizer.json`, `tokenlizer_config.json`, and `y2_sft_s-2.yt` (if using `ymodel2-s0`, download `y2_sft_s-0.yt` instead). Put these three files into `example/ymodel2-s-2/model/`.

3. Open `main.cpp` and set model scale according to the downloaded model:

```cpp
// Initialize model config
ymodel2::YConfig2 config;
config.scale_lvl(-2);       // <-- Use -2 for ymodel2-s-2; use 0 for ymodel2-s0.
```

4. Build and run:
First go to `example/ymodel2-s-2`, then run:

```bash
mkdir build
cd build
cmake ..
make -j8
./ymodel2-s-2
```

5. Example output (`ymodel2-s-2`):

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

## Basic YTensor Usage

```cpp
// 1. Initialize tensors
yt::YTensor<float, 2> a(3, 4);                  // Pre-allocates a tensor with shape [3, 4], default-initialized
auto b = yt::YTensor<float, 3>::ones(2, 3, 4);  // All ones, shape [2, 3, 4]

// 2. Access and modify elements
float v1 = a.at(1, 2);  // Recommended: fast access to element at (1, 2)
float v2 = a[1][2];     // Bracket indexing is also supported
b.at(0, 1, 2) = 42.f;   // Assign element at (0, 1, 2)

// 3. Get shape and size
std::vector<int> shape = b.shape(); // Shape
size_t sz = b.size();               // Total number of elements

// 4. View and transform
auto b2 = b.view(6, -1);    // Convert [2, 3, 4] into [6, 4], where -1 is inferred from element count
b2 = b2.permute(1, 0);      // Swap dimensions, b2 shape becomes [4, 6]
b2 = b2.contiguous();       // Convert b2 to a contiguous tensor (equivalent to in-place b2.contiguous_())

// 5. Print shape
std::cout << "shape: ";
for (int d : b2.shape()){
    std::cout << d << " ";
}
std::cout << std::endl;
```

> The snippet above shows common initialization, element access, shape querying, and view operations in YTensor. See detailed sections below and `example/` for more.

---

## Three Usage Forms (API-Equivalent)

YTensor currently provides three coexisting forms with identical external API and usage:

- **Default header-only** (standard form): define no backend macro; include `ytensor.hpp` or `ytensor_single.hpp` directly.
- **single-header** (distribution form): use `single-header/ytensor_single.hpp`, behavior is the same as default header-only.
- **`YT_USE_LIB`** (compile-acceleration backend): define `YT_USE_LIB=1` and link `libytensor`; only implementation ownership changes to reduce repeated compilation.

> The only extra requirement for `YT_USE_LIB` is **linking the library**. API, return types, supported type set, and custom type capability remain the same.

---

## 🧩 Zero-Dependency Header-Only
Just download `single-header/ytensor_single.hpp` and place it in your project:

```cpp
#include "ytensor_single.hpp"   // This single include is enough

int main() {
    auto a = yt::YTensor<float, 2>::randn(3, 4);    // [3, 4] random tensor from normal distribution
    std::cout << a << std::endl;                    // Print detailed tensor info
    return 0;
}
```

> One header, zero dependencies, quick to integrate in any C++20 project.
>
> **Note:** YTensor uses many C++20 features. Make sure your compiler supports C++20.

### `YT_USE_LIB` Accelerated Backend Example

If you want to reduce repeated user-side compilation, you can use the precompiled library backend:

```cpp
#define YT_USE_LIB 1
#include "ytensor.hpp"   // or include <lib/include/ytensor.hpp>
```

```bash
# Example: link libytensor (Linux)
g++ -std=c++20 -O2 -fopenmp main.cpp \
  -I/path/to/ytensor \
  -I/path/to/ytensor/lib/include \
  -L/path/to/ytensor/lib/bin \
  -Wl,-rpath,/path/to/ytensor/lib/bin \
  -lytensor -lz -o main
```

---

## 📐 Arbitrary Dimensions and Common Ops
Supports common tensor operations such as shape, slicing, transpose, reshape, and permute.

```cpp
yt::YTensor<float, 3> t(4, 5, 3);   // Construct a 4x5x3 tensor
t.fill(1.0f);                       // Set all elements to 1.0

// Chained one-liner style
auto sliced = t
    .permute(1, 2, 0)    // Reorder axes: [4, 5, 3] -> [5, 3, 4]
    .contiguous()        // Ensure contiguous memory; required before view
    .view(15, -1)        // Merge first two axes: [3, 4, 5] -> [15, 4]
    .slice(1, 1, 3, 1);  // Slice axis 1 at indices 1,2: [15, 4] -> [15, 2]

// Print shape
const auto& s = sliced.shape();
std::cout << "shape: [";
for (size_t i = 0; i < s.size(); ++i) {
    std::cout << s[i];
    if (i + 1 < s.size()) std::cout << ", ";
}
std::cout << "]\n";
```

---

## ⚡ Multi-Purpose Computation Support

```cpp
// ReLU activation (library function)
auto x = yt::YTensor<float, 2>::randn(3, 4);    // Randomly initialized x: [3, 4]
auto y = yt::YTensor<float, 2>::randn(1, 4);    // Randomly initialized y: [1, 4]
auto reluOutput = yt::function::relu(x);        // Element-wise relu. reluOutput: [3, 4]

// Highly flexible element-wise broadcast compute. siluOutput: [3, 4]
auto siluOutput = x.broadcastInplace([](float& v) {
    float s = 1.0f / (1.0f + std::exp(-v)); // sigmoid
    v = v * s;
});

// Element-wise custom multi-input compute. out: [3, 4]
auto out = yt::strided::broadcast([](
    const float& t1,    // Element from tensor x
    const float& t2,    // Element from tensor y
    const float& t3,    // Element from tensor reluOutput
    const float& t4,    // Element from tensor siluOutput
    const float& s5,    // Scalar constant
){
    return t1 + t2 + t3 + t4 + s5;
}, x, y, reluOutput, siluOutput, 0.5f); // Inputs must match function parameters in order

// Also supports symbolic broadcast operations
out += y - 0.1f;

std::cout << "ReLU output:\n" << reluOutput << std::endl;
std::cout << "SiLU output:\n" << siluOutput << std::endl;
std::cout << "Custom output:\n" << out << std::endl;
```

Common operations are supported out of the box, while custom operation composition remains highly flexible. The example demonstrates strong extensibility based on library primitives.

---

## 🦾 Multi-Type Support
Supports tensors with many data types, including standard library and custom types. Typed `YTensor<T, dim>` math is available when `T` provides the required operators. Runtime `YTensorBase` math instead requires a registered `YDTypeKernels` capability; tensor-scalar conversion may additionally require an exact cast kernel.

```cpp
yt::YTensor<std::string, 2> strTensor(3, 4);    // Create a 3x4 std::string tensor
strTensor.fill("hello");                        // Initialize with "hello"
strTensor += " world";                          // Broadcast add (string concatenation)
std::cout << strTensor << std::endl;            // Print tensor
std::cout << strTensor[0][0] << std::endl;      // Access element

// For custom types, a type registration mechanism is provided
struct MyType {
    int value = 0;
    MyType operator+(const MyType& other) const{
        return MyType{value - other.value}; // As long as operators are defined, corresponding ops are supported
    }
    // ...
};

// Register type and provide a string conversion function.
// Type name is required; conversion function is optional (affects print output).
yt::type::registerType<MyType>("MyType", [](const void* data) {
    const MyType* p = reinterpret_cast<const MyType*>(data); // Cast to MyType pointer
    return std::to_string(p->value + 1); // Print value+1 directly
});

yt::YTensor<MyType, 2> myTensor(2, 3);
static int i = 0;
myTensor.foreach([&](auto& x){
    x.value = i++;
});
myTensor += MyType{5};
myTensor[0][0].value = 114513;
std::cout << myTensor << std::endl;
/* Example output:
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

## 🗂️ I/O Support
Supports save/load with a custom binary format, with optional compression (requires zlib). Good for efficient serialization and cross-platform data exchange.

```cpp
yt::io::verbose = true;      // Print detailed logs (off by default)
yt::io::compressMethod = ""; // No compression (default is no compression; zlib is usually not very effective for floats)

auto t0 = yt::YTensor<float, 2>::randn(3, 4);
auto t1 = yt::YTensor<float, 3>::randn(5, 6, 7);
yt::io::YTensorIO io;               // Create file I/O object
io.open("./test.yt", yt::io::Write);// Open file in write mode
io.save(t0, "name0");               // Save tensor to file
io.save(t1, "name1");               // Multi-tensor save is supported
io.close();                         // Close file and flush to disk

// Load file
io.open("./test.yt", yt::io::Read); // Open file in read mode
yt::YTensor<float, 2> read0;
yt::YTensor<float, 3> read1;
yt::YTensorBase base0;
io.load(read0, "name0");    // Load tensor; note type must match (<float, 2>)
io.load(read1, "name1");    // Type: <float, 3>
io.load(base0, "name0");    // Can also load into YTensorBase
io.close();

// I/O for non-POD types (e.g., std::string) is also supported.
// For custom types, use registerType to provide serialization/deserialization before save/load.
```

You can also use conversion scripts under `example/convert/` for partial format conversion (such as numpy, etc.).

> Suitable for convenient storage of model weights, datasets, and related assets.

---

## File Structure

```tree
./
├─ doc/                          | User guides and API reference
│  ├─ en/                        | English documentation
│  │  ├─ api/                    | English API reference
│  │  └─ installation/           | English installation and build guides
│  └─ zh/                        | Chinese documentation
│     ├─ api/                    | Chinese API reference
│     └─ installation/           | Chinese installation and build guides
├─ example/                      | Data conversion and model inference examples
│  ├─ convert/                   | Conversion tools for YTensor and other formats
│  ├─ qwen3/                     | Qwen3 CPU inference example
│  │  ├─ include/                | Qwen3 declarations and third-party headers
│  │  ├─ model/                  | Qwen3 configuration, weights, and tokenizer resources
│  │  └─ src/                    | Qwen3 implementation
│  └─ ymodel2-s-2/               | YModel2 CPU inference example
│     └─ model/                  | YModel2 weights and tokenizer resources
├─ include/                      | Public declarations and template interfaces
│  ├─ 3rd/                       | Third-party headers used by the project
│  ├─ blas/                      | YBLAS interfaces and frame declarations
│  │  └─ kernels/                | YBLAS physical microkernel declarations
│  │     └─ avx2/                | AVX2/FMA microkernel declarations
│  ├─ function/                  | Neural-network and functional operator interfaces
│  ├─ strided/                   | Strided-layout algorithm interfaces
│  ├─ type/                      | Dtype and numeric-type support
│  └─ utils/                     | Common utility interfaces
├─ src/                          | YTensor implementations
│  ├─ blas/                      | YBLAS frames and operators
│  │  └─ kernels/                | YBLAS physical microkernel implementations
│  │     └─ avx2/                | AVX2/FMA microkernel implementations
│  ├─ function/                  | Neural-network and functional operator implementations
│  ├─ strided/                   | Strided-layout algorithm implementations
│  ├─ type/                      | Dtype dispatch and type implementations
│  └─ utils/                     | Common utility implementations
├─ lib/                          | YTensor library build directory
│  └─ src/                       | Library implementation entry
└─ single-header/                | Single-header distribution and packing tools
```

> YTensor version: 0.16
>
> **Note:** The current version is still evolving rapidly. Some uncommon or low-level APIs may change significantly. Please keep an eye on release notes.

---

## Latest Update

- Redesigned the BLAS backend architecture by introducing an independent YBLAS layer over the existing AVX2 implementation, removing the upper layers' dependency on a specific instruction set.
- Improved Qwen3 CPU inference performance and restored expected encode performance.
- Added convolution operators.

---
For more examples, API details, or contribution suggestions, check the `example/` directory or open an issue.
