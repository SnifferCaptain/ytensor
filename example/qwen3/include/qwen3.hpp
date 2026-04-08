#pragma once
// #define YT_USE_EIGEN 1   // 让库自动查看否支持Eigen
// #define YT_USE_AVX2 0    // 让库自动查看否支持AVX2加速
#include "../../../ytensor.hpp"
// #include "ytensor_single.hpp"

#include <vector>
#include <string>
#include <functional>

namespace qwen3 {

using namespace yt;

class KVCache {
public:
	yt::YTensor<float, 4> k_buffer; // [b, kv_h, l, hd]
	yt::YTensor<float, 4> v_buffer; // [b, kv_h, l, hd]
	int cur_len = 0;                // 当前存储的token数量（最大为max_len）
	int max_len = 0;                // buffer最大容量
	int write_pos = 0;              // 下一个写入位置（循环buffer）
	int total_written = 0;          // 总共写入的token数（用于计算RoPE全局位置）
	int num_kv_heads = 0;
	int head_dim = 0;

	void init(int batch, int max_length, int kv_heads, int head_dim);
	void reset() { cur_len = 0; write_pos = 0; total_written = 0; }

	// 添加 new_k/new_v [b, kv_h, l, hd] 到缓存
	void append(const yt::YTensor<float, 4>& new_k, const yt::YTensor<float, 4>& new_v);

	// 获取当前KV缓存
	yt::YTensor<float, 4> get_k() const;
	yt::YTensor<float, 4> get_v() const;
	bool empty() const { return cur_len == 0; }
	bool full() const { return cur_len >= max_len; }
	int remaining() const { return max_len - cur_len; }

	// 获取当前全局位置（用于RoPE）
	int get_global_position() const { return total_written; }

	// 获取kv缓存对应的自回归掩码[query_len, kv_len]
	yt::YTensor<bool, 2> get_mask(int query_len) const;
};

struct Qwen3Config {
	int hidden_size = 1024;
	int num_hidden_layers = 28;
	int vocab_size = 151936;
	float rms_norm_eps = 1e-6f;
	float rope_theta = 1e6f;
	int intermediate_size = 3072;
	int num_attention_heads = 16;
	int num_key_value_heads = 8;
	int head_dim = 128;
	int max_position_embeddings = 4096;
	std::string model_dir = "model";

	void init(const std::string& model_dir);
};

struct RoPECache {
	yt::YTensor<float, 2> cos, sin;
	void precompute(int dim, int max_len, float theta);
};

class RMSNorm {
public:
	yt::YTensor<float, 1> weight;
	float eps = 1e-6f;
	yt::YTensor<float, 3> forward(const yt::YTensor<float, 3>& x) const;
	void forward_(yt::YTensor<float, 3>& x) const;
};

class HeadRMSNorm {
public:
	yt::YTensor<float, 1> weight;
	float eps = 1e-6f;
	yt::YTensor<float, 4> forward(const yt::YTensor<float, 4>& x) const;
	void forward_(yt::YTensor<float, 4>& x) const;
};

class Qwen3MLP {
public:
	yt::YTensor<float, 2> gate_up, down;
	int intermediate_size = 0;
	yt::YTensor<float, 3> forward(const yt::YTensor<float, 3>& x) const;
};

class Qwen3Attention {
public:
	yt::YTensor<float, 2> qkv_proj, o_proj;
	HeadRMSNorm q_norm, k_norm;
	int hidden_size = 0;
	int num_heads = 0;
	int num_kv_heads = 0;
	int head_dim = 0;
	int num_groups = 0;
	float rsqrt_dim = 1.0f;

	yt::YTensor<float, 3> forward(
		const yt::YTensor<float, 3>& x,
		const yt::YTensor<float, 2>& cos,
		const yt::YTensor<float, 2>& sin,
		KVCache* kv_cache = nullptr
	) const;

	void prefill_kv_only(
		const yt::YTensor<float, 3>& x,
		const yt::YTensor<float, 2>& cos,
		const yt::YTensor<float, 2>& sin,
		KVCache* kv_cache
	) const;
};

class Qwen3DecoderLayer {
public:
	Qwen3Attention attn;
	Qwen3MLP mlp;
	RMSNorm norm1, norm2;

	yt::YTensor<float, 3> forward(
		const yt::YTensor<float, 3>& x,
		const yt::YTensor<float, 2>& cos,
		const yt::YTensor<float, 2>& sin,
		KVCache* kv_cache = nullptr
	) const;
};

class Qwen3Model {
public:
	Qwen3Config config;
	yt::YTensor<float, 2> embed;
	yt::YTensor<float, 2> lm_head;  // 若存在则存储 lm_head 权重
	std::vector<Qwen3DecoderLayer> layers;
	RMSNorm norm;
	RoPECache rope;

	void init(const Qwen3Config& cfg);
	bool load(const std::string& path);
	// kv_caches: 每层对应一个KVCache指针，在内部更新
	yt::YTensor<float, 3> forward(
		const yt::YTensor<int, 2>& ids,
		std::vector<KVCache>* kv_caches = nullptr,
		bool chat_only = false
	);
};

class Qwen3ForCausalLM {
public:
	Qwen3Config config;
	Qwen3Model model;
	std::vector<KVCache> kv_caches;  // 每层一个KVCache
	yt::YTensor<float, 2> lm_head;
	bool use_lm_head = false;

	void init(const Qwen3Config& cfg);

	// 从文件加载模型参数
	bool load(const std::string& path);

	// 返回logits，内部更新kv_caches
	yt::YTensor<float, 2> forward(const yt::YTensor<int, 2>& ids, bool chat_only = false);

	// generate: 智能生成
	std::vector<int> generate(const std::vector<int>& new_ids, int max_tokens, int eos = 2,
							  std::function<void(int)> on_token = nullptr,
							  bool chat_only = true);
	void reset_kv_cache();          // 重置KV cache（清空对话历史）
	int get_kv_cache_len() const;   // 获取当前KV缓存的token长度
	int get_max_context_len() const;// 获取最大上下文长度
};

namespace ops {
	void rope(yt::YTensor<float, 4>& q, yt::YTensor<float, 4>& k,
			  const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin);

	void rope_k(yt::YTensor<float, 4>& k,
				const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin);

	yt::YTensor<float, 5> repeat_kv(const yt::YTensor<float, 4>& x, int n);  // 零拷贝版本，返回5D张量
}// namespace ops

}// namespace qwen3
