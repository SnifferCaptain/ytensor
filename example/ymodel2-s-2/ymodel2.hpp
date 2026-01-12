#pragma once
// #define YT_USE_EIGEN 1   // 让库自动查看否支持Eigen
// #define YT_USE_AVX2 1    // 让库自动查看否支持AVX2加速
#include "../../ytensor.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <functional>

namespace ymodel2 {

using namespace yt;

class KVCache {
public:
    yt::YTensor<float, 4> buffer;   // [b, 2, l, hd] or [b, 2, hd, l]
    int cur_len = 0;                // 当前存储的token数量（最大为max_len）
    int max_len = 0;                // buffer最大容量
    bool transpose = false;
    int write_pos = 0;              // 下一个写入位置（循环buffer）
    int total_written = 0;          // 总共写入的token数（用于计算RoPE全局位置）
    
    void init(int batch, int max_length, int head_dim, bool transpose = true);
    void reset() { cur_len = 0; write_pos = 0; total_written = 0; }

    // 添加 new_kv [b, 2, l, hd] 到缓存
    void append(const yt::YTensor<float, 4>& new_kv);

    // 获取当前KV缓存（已经保证了内存排布最优）
    yt::YTensor<float, 4> get() const;
    bool empty() const { return cur_len == 0; }
    bool full() const { return cur_len >= max_len; }
    int remaining() const { return max_len - cur_len; }

    // 获取当前全局位置（用于RoPE）
    int get_global_position() const { return total_written; }

    // 获取kv缓存对应的自回归掩码[query_len, kv_len]
    yt::YTensor<float, 2> get_mask(int query_len) const;
};

struct YConfig2 {
    int hidden_size = 512;
    int num_layers = 4;
    int vocab_size = 6400;
    float rms_norm_eps = 1e-8f;
    float rope_theta = 5e4f;
    int intermediate_size = 1024;
    int num_heads = 8;
    int head_dim = 64;
    int max_position_embeddings = 8192;
    
    void scale_lvl(int lvl);
};

struct RoPECache {
    yt::YTensor<float, 2> cos, sin;
    void precompute(int dim, int max_len, float theta);
};

class RMSNorm {
public:
    yt::YTensor<float, 1> weight;
    float eps = 1e-8f;
    yt::YTensor<float, 3> forward(const yt::YTensor<float, 3>& x) const;
    void forward_(yt::YTensor<float, 3>& x) const;
};

class FFN {
public:
    yt::YTensor<float, 2> up, down;
    int intermediate_size;
    yt::YTensor<float, 3> forward(const yt::YTensor<float, 3>& x) const;
};

class PEGA2 {
public:
    yt::YTensor<float, 2> qkv_0, qkv_1, o;
    int hidden_size, num_heads, head_dim;
    float rsqrt_dim;
    std::vector<int> qkv_list;
    
    // 传入KVCache指针，在内部更新cache，返回输出tensor
    yt::YTensor<float, 3> forward(
        const yt::YTensor<float, 3>& x,
        const yt::YTensor<float, 2>& cos,
        const yt::YTensor<float, 2>& sin,
        KVCache* kv_cache = nullptr
    ) const;
};

class YBlock2 {
public:
    PEGA2 attn;
    FFN ffn;
    RMSNorm norm1, norm2;
    
    yt::YTensor<float, 3> forward(
        const yt::YTensor<float, 3>& x,
        const yt::YTensor<float, 2>& cos,
        const yt::YTensor<float, 2>& sin,
        KVCache* kv_cache = nullptr
    ) const;
};

class YModel2 {
public:
    YConfig2 config;
    yt::YTensor<float, 2> embed;
    std::vector<YBlock2> layers;
    RMSNorm norm;
    RoPECache rope;
    
    void init(const YConfig2& cfg);
    bool load(const std::string& path);
    // kv_caches: 每层对应一个KVCache指针，在内部更新
    yt::YTensor<float, 3> forward(
        const yt::YTensor<int, 2>& ids,
        std::vector<KVCache>* kv_caches = nullptr
    );
};

class YForCausalLM2 {
public:
    YConfig2 config;
    YModel2 model;
    std::vector<KVCache> kv_caches;  // 每层一个KVCache
    
    void init(const YConfig2& cfg);

    // 从文件加载模型参数
    bool load(const std::string& path);

    // 返回logits，内部更新kv_caches
    yt::YTensor<float, 2> forward(const yt::YTensor<int, 2>& ids);

    // generate: 智能生成
    std::vector<int> generate(const std::vector<int>& new_ids, int max_tokens, int eos = 2,
                              std::function<void(int)> on_token = nullptr);
    void reset_kv_cache();          // 重置KV cache（清空对话历史）
    int get_kv_cache_len() const;   // 获取当前KV缓存的token长度
    int get_max_context_len() const;// 获取最大上下文长度
};

namespace ops {
    template<int dim> yt::YTensor<float, dim> linear(const yt::YTensor<float, dim>& x, const yt::YTensor<float, 2>& weight);
    
    template<int dim> yt::YTensor<float, dim>& gelu_(yt::YTensor<float, dim>& x);

    void rope(yt::YTensor<float, 4>& q, yt::YTensor<float, 4>& k, const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin);

    yt::YTensor<float, 5> repeat_kv(const yt::YTensor<float, 4>& x, int n);  // 零拷贝版本，返回5D张量
}// namespace ops

}// namespace ymodel2
