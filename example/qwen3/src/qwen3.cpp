#include "../include/qwen3.hpp"
#include "../include/json.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <algorithm>

using nlohmann::json;

namespace qwen3 {

template<int dim>
yt::YTensor<float, dim> toFloat(const yt::YTensorBase& base, bool transpose = false) {
	if (base.dtype() == "float32") {
		yt::YTensor<float, dim> op(base);
		if (transpose) {
			op = op.transpose().contiguous().transpose();
		}
		return op;
	}
	yt::YTensor<float, dim> op;
	op.reserve(base.shape());
	float* dst = op.data();
	size_t n = base.size();
	if (base.dtype() == "bfloat16") {
		const yt::bfloat16* src = base.data<yt::bfloat16>();
		for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
	} else if (base.dtype() == "float64") {
		const double* src = base.data<double>();
		for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
	} else if (base.dtype() == "int32") {
		const int32_t* src = base.data<int32_t>();
		for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
	} else {
		std::cerr << "Warning: unsupported dtype " << base.dtype() << ", zeros returned\n";
	}
	if (transpose) {
		op = op.transpose().contiguous().transpose();
	}
	return op;
}

static yt::YTensor<bool, 2> make_causal_mask(int query_len, int kv_len) {
	yt::YTensor<bool, 2> mask(query_len, kv_len);
	int start = kv_len - query_len;
	for (int qi = 0; qi < query_len; ++qi) {
		for (int ki = 0; ki < kv_len; ++ki) {
			mask.at(qi, ki) = (ki <= start + qi);
		}
	}
	return mask;
}

void KVCache::init(int batch, int max_length, int kv_heads, int head_dim) {
	max_len = max_length;
	cur_len = 0;
	write_pos = 0;
	total_written = 0;
	num_kv_heads = kv_heads;
	this->head_dim = head_dim;
	k_buffer.reserve({batch, kv_heads, max_length, head_dim});
	v_buffer.reserve({batch, kv_heads, max_length, head_dim});
	k_buffer.fill(0.0f);
	v_buffer.fill(0.0f);
}

void KVCache::append(const yt::YTensor<float, 4>& new_k, const yt::YTensor<float, 4>& new_v) {
	// new_k/new_v: [b, kv_h, l, hd]
	int new_len = new_k.shape(2);
	for (int t = 0; t < new_len; ++t) {
		int pos = write_pos;
		auto src_k = new_k.slice(2, t, t + 1);   // [b, kv_h, 1, hd]
		auto src_v = new_v.slice(2, t, t + 1);
		auto dst_k = k_buffer.slice(2, pos, pos + 1);
		auto dst_v = v_buffer.slice(2, pos, pos + 1);
		dst_k.copy_(src_k);
		dst_v.copy_(src_v);

		write_pos = (write_pos + 1) % max_len;
		total_written++;
		if (cur_len < max_len) {
			cur_len++;
		}
	}
}

yt::YTensor<float, 4> KVCache::get_k() const {
	if (full()) {
		return k_buffer;
	}
	return yt::YTensor<float, 4>(k_buffer.slice(2, 0, cur_len));
}

yt::YTensor<float, 4> KVCache::get_v() const {
	if (full()) {
		return v_buffer;
	}
	return yt::YTensor<float, 4>(v_buffer.slice(2, 0, cur_len));
}

yt::YTensor<bool, 2> KVCache::get_mask(int query_len) const {
	int kv_len = cur_len;
	yt::YTensor<bool, 2> mask(query_len, kv_len);

	if (!full()) {
		int start = kv_len - query_len;
		for (int qi = 0; qi < query_len; ++qi) {
			for (int ki = 0; ki < kv_len; ++ki) {
				mask.at(qi, ki) = (ki <= start + qi);
			}
		}
	} else {
		int start = kv_len - query_len;
		for (int qi = 0; qi < query_len; ++qi) {
			int q_logical = start + qi;
			for (int ki = 0; ki < kv_len; ++ki) {
				int k_logical = (ki - write_pos + max_len) % max_len;
				mask.at(qi, ki) = (k_logical <= q_logical);
			}
		}
	}
	return mask;
}

void Qwen3Config::init(const std::string& model_dir) {
	this->model_dir = model_dir;
	std::ifstream ifs(model_dir + "/config.json");
	if (!ifs) {
		// 允许在 build 目录运行时自动回退到上级目录的 model
		std::string fallback = "../model";
		if (model_dir == "./model" || model_dir == "model") {
			fallback = "../model";
		} else if (model_dir.rfind("./", 0) == 0) {
			fallback = "../" + model_dir.substr(2);
		}
		ifs.open(fallback + "/config.json");
		if (ifs) {
			this->model_dir = fallback;
		} else {
			std::cerr << "Warning: failed to open config.json at " << model_dir << "\n";
			return;
		}
	}
	json j;
	ifs >> j;
	hidden_size = j.value("hidden_size", hidden_size);
	num_hidden_layers = j.value("num_hidden_layers", num_hidden_layers);
	vocab_size = j.value("vocab_size", vocab_size);
	rms_norm_eps = j.value("rms_norm_eps", rms_norm_eps);
	rope_theta = j.value("rope_theta", rope_theta);
	intermediate_size = j.value("intermediate_size", intermediate_size);
	num_attention_heads = j.value("num_attention_heads", num_attention_heads);
	num_key_value_heads = j.value("num_key_value_heads", num_key_value_heads);
	head_dim = j.value("head_dim", head_dim);
	max_position_embeddings = j.value("max_position_embeddings", max_position_embeddings);
}

void RoPECache::precompute(int dim, int max_len, float theta) {
	int half = dim / 2;
	cos.reserve(max_len, dim);
	sin.reserve(max_len, dim);

	#pragma omp parallel for collapse(2) proc_bind(close)
	for (int t = 0; t < max_len; ++t) {
		for (int i = 0; i < half; ++i) {
			float freq = 1.0f / std::pow(theta, 2.0f * i / dim);
			float angle = t * freq;
			cos.at(t, i) = cos.at(t, i + half) = std::cos(angle);
			sin.at(t, i) = sin.at(t, i + half) = std::sin(angle);
		}
	}
}

yt::YTensor<float, 3> RMSNorm::forward(const yt::YTensor<float, 3>& x) const {
	return yt::function::rmsNorm(x, weight, -1, eps);
}

void RMSNorm::forward_(yt::YTensor<float, 3>& x) const {
	yt::function::rmsNorm_(x, weight, -1, eps);
}

yt::YTensor<float, 4> HeadRMSNorm::forward(const yt::YTensor<float, 4>& x) const {
	return yt::function::rmsNorm(x, weight, -1, eps);
}

void HeadRMSNorm::forward_(yt::YTensor<float, 4>& x) const {
	yt::function::rmsNorm_(x, weight, -1, eps);
}

yt::YTensor<float, 3> Qwen3MLP::forward(const yt::YTensor<float, 3>& x) const {
	auto h = yt::function::linear(x, gate_up);
	auto gate_proj = h.slice(-1, 0, intermediate_size);
	auto up_proj = h.slice(-1, intermediate_size, 2 * intermediate_size);
	yt::function::swish_(gate_proj);
	gate_proj *= up_proj;
	return yt::function::linear(gate_proj, down);
}

void Qwen3Attention::prefill_kv_only(
	const yt::YTensor<float, 3>& x,
	const yt::YTensor<float, 2>& cos,
	const yt::YTensor<float, 2>& sin,
	KVCache* kv_cache
) const {
	if (!kv_cache) return;

	int b = x.shape(0), l = x.shape(1);
	int kv_h = num_kv_heads, hd = head_dim;

	auto qkv = yt::function::linear(x, qkv_proj);
	int q_size = num_heads * hd;
	int kv_size = kv_h * hd;
	auto k4 = qkv.slice(-1, q_size, q_size + kv_size).reshape(b, l, kv_h, hd).permute(0, 2, 1, 3);
	auto v4 = qkv.slice(-1, q_size + kv_size, q_size + kv_size + kv_size).reshape(b, l, kv_h, hd).permute(0, 2, 1, 3);

	k_norm.forward_(k4);
	ops::rope_k(k4, cos, sin);
	kv_cache->append(k4, v4);
}

yt::YTensor<float, 3> Qwen3Attention::forward(
	const yt::YTensor<float, 3>& x,
	const yt::YTensor<float, 2>& cos,
	const yt::YTensor<float, 2>& sin,
	KVCache* kv_cache
) const {
	int b = x.shape(0), l = x.shape(1);
	int h = num_heads, kv_h = num_kv_heads, hd = head_dim;
	int groups = num_groups;

	auto qkv = yt::function::linear(x, qkv_proj);
	int q_size = h * hd;
	int kv_size = kv_h * hd;
	auto q4 = qkv.slice(-1, 0, q_size).reshape(b, l, h, hd).permute(0, 2, 1, 3);
	auto k4 = qkv.slice(-1, q_size, q_size + kv_size).reshape(b, l, kv_h, hd).permute(0, 2, 1, 3);
	auto v4 = qkv.slice(-1, q_size + kv_size, q_size + kv_size + kv_size).reshape(b, l, kv_h, hd).permute(0, 2, 1, 3);

	q_norm.forward_(q4);
	k_norm.forward_(k4);
	ops::rope(q4, k4, cos, sin);

	if (kv_cache) {
		kv_cache->append(k4, v4);
	}

	auto k_full = (kv_cache && !kv_cache->empty()) ? kv_cache->get_k() : k4;
	auto v_full = (kv_cache && !kv_cache->empty()) ? kv_cache->get_v() : v4;

	auto k_5d = ops::repeat_kv(k_full, groups);
	auto v_5d = ops::repeat_kv(v_full, groups);
	auto q_5d = q4.reshape(b, kv_h, groups, l, hd);
	yt::YTensor<bool, 2> causal_mask = kv_cache ? kv_cache->get_mask(l) : make_causal_mask(l, k_full.shape(2));
	auto attn_5d = yt::function::scaledDotProductAttention(
		q_5d,
		k_5d,
		v_5d,
		rsqrt_dim,
		&causal_mask,
		nullptr,
		yt::function::sdpaBackend::FLASH_AVX2
	);

	auto attn4 = attn_5d.reshape(b, h, l, hd);
	auto attn3 = attn4.permute(0, 2, 1, 3).reshape(b, l, h * hd);
	return yt::function::linear(attn3, o_proj);
}

yt::YTensor<float, 3> Qwen3DecoderLayer::forward(
	const yt::YTensor<float, 3>& x,
	const yt::YTensor<float, 2>& cos,
	const yt::YTensor<float, 2>& sin,
	KVCache* kv_cache
) const {
	auto h1 = norm1.forward(x);
	auto x2 = attn.forward(h1, cos, sin, kv_cache);
	x2 += x;
	auto h2 = norm2.forward(x2);
	x2 += mlp.forward(h2);
	return x2;
}

void Qwen3Model::init(const Qwen3Config& cfg) {
	config = cfg;
	embed.reserve(cfg.vocab_size, cfg.hidden_size);
	layers.resize(cfg.num_hidden_layers);
	for (auto& layer : layers) {
		layer.attn.hidden_size = cfg.hidden_size;
		layer.attn.num_heads = cfg.num_attention_heads;
		layer.attn.num_kv_heads = cfg.num_key_value_heads;
		layer.attn.head_dim = cfg.head_dim;
		layer.attn.num_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
		layer.attn.rsqrt_dim = 1.0f / std::sqrt((float)cfg.head_dim);
		layer.mlp.intermediate_size = cfg.intermediate_size;
		layer.norm1.eps = layer.norm2.eps = cfg.rms_norm_eps;
		layer.attn.q_norm.eps = layer.attn.k_norm.eps = cfg.rms_norm_eps;
	}
	norm.eps = cfg.rms_norm_eps;
	rope.precompute(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta);
}

bool Qwen3Model::load(const std::string& path) {
	yt::io::YTensorIO io;
	if (!io.open(path, yt::io::Read)) return false;

	auto tensor_names = io.getTensorNames();
	std::cout << "  File contains " << tensor_names.size() << " tensors" << std::endl;

	int loaded_count = 0;
	yt::YTensorBase base;
	bool useTranspose = false;
	if (yt::infos::defaultMatmulBackend == yt::infos::MatmulBackend::Eigen) {
		useTranspose = true;
	}

	if (io.load(base, "model.embed_tokens.weight")) { embed = toFloat<2>(base, useTranspose); loaded_count++; }

	for (int i = 0; i < config.num_hidden_layers; ++i) {
		std::string p = "model.layers." + std::to_string(i) + ".";
		yt::YTensor<float, 2> q_w, k_w, v_w;
		if (io.load(base, p + "self_attn.q_proj.weight")) { q_w = toFloat<2>(base, useTranspose); loaded_count++; }
		if (io.load(base, p + "self_attn.k_proj.weight")) { k_w = toFloat<2>(base, useTranspose); loaded_count++; }
		if (io.load(base, p + "self_attn.v_proj.weight")) { v_w = toFloat<2>(base, useTranspose); loaded_count++; }
		if (q_w.size() > 0 && k_w.size() > 0 && v_w.size() > 0) {
			auto qkv = yt::YTensor<float, 2>(q_w.concat(k_w.concat(v_w, 0), 0));
			// concat() 会强制把输入变为 contiguous(行主序) 再拷贝，
			// 这会破坏 toFloat(..., useTranspose=true) 为 Eigen 准备的“列主序”布局。
			// 为了保持 Eigen 下 weight.transpose() 的连续性，这里需要对合并后的权重重新做一次布局转换。
			if (useTranspose) {
				qkv = qkv.transpose().contiguous().transpose();
			}
			layers[i].attn.qkv_proj = qkv;
		}
		if (io.load(base, p + "self_attn.o_proj.weight")) { layers[i].attn.o_proj = toFloat<2>(base, useTranspose); loaded_count++; }
		if (io.load(base, p + "self_attn.q_norm.weight")) { layers[i].attn.q_norm.weight = toFloat<1>(base); loaded_count++; }
		if (io.load(base, p + "self_attn.k_norm.weight")) { layers[i].attn.k_norm.weight = toFloat<1>(base); loaded_count++; }
		yt::YTensor<float, 2> gate_w, up_w;
		if (io.load(base, p + "mlp.gate_proj.weight")) { gate_w = toFloat<2>(base, useTranspose); loaded_count++; }
		if (io.load(base, p + "mlp.up_proj.weight")) { up_w = toFloat<2>(base, useTranspose); loaded_count++; }
		if (io.load(base, p + "mlp.down_proj.weight")) { layers[i].mlp.down = toFloat<2>(base, useTranspose); loaded_count++; }
		if (gate_w.size() > 0 && up_w.size() > 0) {
			auto gate_up = yt::YTensor<float, 2>(gate_w.concat(up_w, 0));
			if (useTranspose) {
				gate_up = gate_up.transpose().contiguous().transpose();
			}
			layers[i].mlp.gate_up = gate_up;
		}
		if (io.load(base, p + "input_layernorm.weight")) { layers[i].norm1.weight = toFloat<1>(base); loaded_count++; }
		if (io.load(base, p + "post_attention_layernorm.weight")) { layers[i].norm2.weight = toFloat<1>(base); loaded_count++; }
	}
	if (io.load(base, "model.norm.weight")) { norm.weight = toFloat<1>(base); loaded_count++; }

	if (io.load(base, "lm_head.weight")) {
		lm_head = toFloat<2>(base, useTranspose);
		loaded_count++;
	}

	std::cout << "  Successfully loaded " << loaded_count << "/" << tensor_names.size() << " tensors" << std::endl;
	io.close();
	return true;
}

yt::YTensor<float, 3> Qwen3Model::forward(const yt::YTensor<int, 2>& ids, std::vector<KVCache>* kv_caches, bool chat_only) {
	int b = ids.shape(0), l = ids.shape(1);
	int start = (kv_caches && !kv_caches->empty() && !(*kv_caches)[0].empty())
				? (*kv_caches)[0].get_global_position() : 0;

	if (start + l > config.max_position_embeddings) {
		start = (kv_caches && !kv_caches->empty()) ? (*kv_caches)[0].cur_len : 0;
		if (start + l > config.max_position_embeddings) {
			start = 0;
		}
	}

	int hidden = config.hidden_size;
	yt::YTensor<float, 3> x(b, l, hidden);
	#pragma omp parallel for collapse(2) proc_bind(close)
	for (int i = 0; i < b; ++i) {
		for (int j = 0; j < l; ++j) {
			auto tokenEmbed = embed.slice(0, ids.at(i, j), ids.at(i, j) + 1).contiguous();
			std::memcpy(&x.at(i, j, 0), tokenEmbed.data(), hidden * sizeof(float));
		}
	}

	auto cos_slice = rope.cos.slice(0, start, start + l);
	auto sin_slice = rope.sin.slice(0, start, start + l);

	for (int i = 0; i < config.num_hidden_layers; ++i) {
		KVCache* cache = (kv_caches && i < (int)kv_caches->size()) ? &(*kv_caches)[i] : nullptr;
		if (chat_only && cache && l > 1 && i == config.num_hidden_layers - 1) {
			auto h1 = layers[i].norm1.forward(x);
			auto h1_prefix = yt::YTensor<float, 3>(h1.slice(1, 0, l - 1));
			auto h1_last = yt::YTensor<float, 3>(h1.slice(1, l - 1, l));
			auto cos_prefix = yt::YTensor<float, 2>(cos_slice.slice(0, 0, l - 1));
			auto sin_prefix = yt::YTensor<float, 2>(sin_slice.slice(0, 0, l - 1));
			auto cos_last = yt::YTensor<float, 2>(cos_slice.slice(0, l - 1, l));
			auto sin_last = yt::YTensor<float, 2>(sin_slice.slice(0, l - 1, l));

			layers[i].attn.prefill_kv_only(h1_prefix, cos_prefix, sin_prefix, cache);

			auto x_last = yt::YTensor<float, 3>(x.slice(1, l - 1, l));
			auto x2_last = layers[i].attn.forward(h1_last, cos_last, sin_last, cache);
			x2_last += x_last;
			auto h2_last = layers[i].norm2.forward(x2_last);
			x2_last += layers[i].mlp.forward(h2_last);
			return norm.forward(x2_last);
		}
		x = layers[i].forward(x, cos_slice, sin_slice, cache);
	}
	return norm.forward(x);
}

void Qwen3ForCausalLM::init(const Qwen3Config& cfg) {
	config = cfg;
	model.init(cfg);

	kv_caches.resize(cfg.num_hidden_layers);
	for (auto& cache : kv_caches) {
		cache.init(1, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim);
	}
}

bool Qwen3ForCausalLM::load(const std::string& path) {
	bool ok = model.load(path);
	use_lm_head = model.lm_head.size() > 0;
	if (use_lm_head) {
		lm_head = model.lm_head;
	}
	return ok;
}

void Qwen3ForCausalLM::reset_kv_cache() {
	for (auto& cache : kv_caches) {
		cache.reset();
	}
}

int Qwen3ForCausalLM::get_kv_cache_len() const {
	if (kv_caches.empty()) return 0;
	return kv_caches[0].cur_len;
}

int Qwen3ForCausalLM::get_max_context_len() const {
	return config.max_position_embeddings;
}

yt::YTensor<float, 2> Qwen3ForCausalLM::forward(const yt::YTensor<int, 2>& ids, bool chat_only) {
	auto h = model.forward(ids, &kv_caches, chat_only);
	int b = h.shape(0), l = h.shape(1);
	auto last = h.slice(1, l - 1, l).contiguous().view(b, config.hidden_size);
	if (use_lm_head) {
		return last.matmul(lm_head.transpose());
	}
	return last.matmul(model.embed.transpose());
}

std::vector<int> Qwen3ForCausalLM::generate(const std::vector<int>& new_ids, int max_tokens, int eos,
											std::function<void(int)> on_token,
											bool chat_only) {
	std::vector<int> out;
	static std::mt19937 rng(42);

	auto sample_token = [&](const yt::YTensor<float, 2>& logits) -> int {
		auto probs = yt::function::softmax(logits, -1);
		constexpr int k = 20;
		int vocab_size = probs.shape(1);
		int top_k = std::min(k, vocab_size);
		std::vector<std::pair<float, int>> prob_idx(vocab_size);
		for (int i = 0; i < vocab_size; ++i) {
			prob_idx[i] = {probs.at(0, i), i};
		}
		std::partial_sort(prob_idx.begin(), prob_idx.begin() + top_k, prob_idx.end(),
			[](const auto& a, const auto& b) { return a.first > b.first; });
		float sum = 0;
		for (int i = 0; i < top_k; ++i) sum += prob_idx[i].first;
		std::uniform_real_distribution<float> dist(0.0f, sum);
		float r = dist(rng);
		float cumsum = 0;
		for (int i = 0; i < top_k; ++i) {
			cumsum += prob_idx[i].first;
			if (r <= cumsum) return prob_idx[i].second;
		}
		return prob_idx[0].second;
	};

	int new_len = static_cast<int>(new_ids.size());
	int max_ctx = get_max_context_len();
	if (new_len > max_ctx) {
		std::cerr << "[Warning] Input length (" << new_len << ") exceeds max context length ("
				  << max_ctx << "), only last " << max_ctx << " tokens will be used\n";
	}

	yt::YTensor<int, 2> input(1, new_len);
	std::copy(new_ids.begin(), new_ids.end(), input.data());

	auto logits = forward(input, chat_only);
	int next = sample_token(logits);
	if (next == eos) {
		return out;
	}
	out.push_back(next);
	if (on_token) on_token(next);

	for (int step = 1; step < max_tokens; ++step) {
		yt::YTensor<int, 2> nid(1, 1);
		nid.at(0, 0) = next;
		auto sl = forward(nid, false);
		next = sample_token(sl);
		if (next == eos) {
			break;
		}
		out.push_back(next);
		if (on_token) on_token(next);
	}
	return out;
}

namespace ops {

void rope(yt::YTensor<float, 4>& q, yt::YTensor<float, 4>& k,
		  const yt::YTensor<float, 2>& cos_cache, const yt::YTensor<float, 2>& sin_cache) {
	int b = q.shape(0), h = q.shape(1), l = q.shape(2), hd = q.shape(3);
	int half = hd / 2;

	#pragma omp parallel for collapse(3) proc_bind(close)
	for (int bi = 0; bi < b; ++bi) {
		for (int hi = 0; hi < h; ++hi) {
			for (int li = 0; li < l; ++li) {
				for (int di = 0; di < half; ++di) {
					float c = cos_cache.at(li, di);
					float s = sin_cache.at(li, di);
					float qr = q.at(bi, hi, li, di);
					float qi = q.at(bi, hi, li, di + half);
					q.at(bi, hi, li, di) = qr * c - qi * s;
					q.at(bi, hi, li, di + half) = qi * c + qr * s;
				}
			}
		}
	}

	int kh = k.shape(1);
	#pragma omp parallel for collapse(3) proc_bind(close)
	for (int bi = 0; bi < b; ++bi) {
		for (int hi = 0; hi < kh; ++hi) {
			for (int li = 0; li < l; ++li) {
				for (int di = 0; di < half; ++di) {
					float c = cos_cache.at(li, di);
					float s = sin_cache.at(li, di);
					float kr = k.at(bi, hi, li, di);
					float ki = k.at(bi, hi, li, di + half);
					k.at(bi, hi, li, di) = kr * c - ki * s;
					k.at(bi, hi, li, di + half) = ki * c + kr * s;
				}
			}
		}
	}
}

void rope_k(yt::YTensor<float, 4>& k,
			const yt::YTensor<float, 2>& cos_cache, const yt::YTensor<float, 2>& sin_cache) {
	int b = k.shape(0), kh = k.shape(1), l = k.shape(2), hd = k.shape(3);
	int half = hd / 2;

	#pragma omp parallel for collapse(3) proc_bind(close)
	for (int bi = 0; bi < b; ++bi) {
		for (int hi = 0; hi < kh; ++hi) {
			for (int li = 0; li < l; ++li) {
				for (int di = 0; di < half; ++di) {
					float c = cos_cache.at(li, di);
					float s = sin_cache.at(li, di);
					float kr = k.at(bi, hi, li, di);
					float ki = k.at(bi, hi, li, di + half);
					k.at(bi, hi, li, di) = kr * c - ki * s;
					k.at(bi, hi, li, di + half) = ki * c + kr * s;
				}
			}
		}
	}
}

yt::YTensor<float, 5> repeat_kv(const yt::YTensor<float, 4>& x, int n) {
	if (n == 1) {
		return x.unsqueeze(2);
	}
	auto x5d = x.unsqueeze(2);
	return yt::YTensor<float, 5>(x5d.repeat(1, 1, n, 1, 1));
}

}// namespace ops

}// namespace qwen3
