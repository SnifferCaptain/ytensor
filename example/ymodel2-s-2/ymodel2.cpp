#include "ymodel2.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <queue>
#include <random>
#include <string>

namespace ymodel2 {

static yt::YTensor<bool, 2> make_causal_mask(int query_len, int kv_len) {
    yt::YTensor<bool, 2> mask(query_len, kv_len);
    int start = kv_len - query_len;
    for (int qi = 0; qi < query_len; ++qi)
        for (int ki = 0; ki < kv_len; ++ki)
            mask.at(qi, ki) = ki <= start + qi;
    return mask;
}

template<int dim>
yt::YTensor<float, dim> toFloat(const yt::YTensorBase& base, bool transpose = false) {
    if (base.dtype() == "float32") {
        yt::YTensor<float, dim> op(base);
        // 转置优化：对Eigen后端来说，列主序的排序方式是更加高效的。
        // 如果是naive的朴素矩阵乘法，则需要是行主序。如果是优化后的avx2后端，decode阶段也是行主序更快。
        if (transpose){
            op = op.transpose().contiguous().transpose();   // 保证转置后内存连续
        }
        return op;
    }
    yt::YTensor<float, dim> op;
    op.reserve(base.shape());   // 提前分配float型内存
    float* dst = op.data();     // 获取目标数据指针
    size_t n = base.size();     // 元素数量
    
    if (base.dtype() == "bfloat16") {
        const yt::bfloat16* src = base.data<yt::bfloat16>();
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        // 相当于下面的语法
        // op.copy_(src);
    } else if (base.dtype() == "float64") {
        const double* src = base.data<double>();
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    } else if (base.dtype() == "int32") {
        const int32_t* src = base.data<int32_t>();
        for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
    } else {
        std::cerr << "Warning: unsupported dtype " << base.dtype() << ", zeros returned\n";
    }
    // 转置优化
    if (transpose){
        op = op.transpose().contiguous().transpose();
    }
    return op;
}

void KVCache::init(int batch, int max_length, int head_dim, bool transpose) {
    max_len = max_length;
    cur_len = 0;
    write_pos = 0;
    total_written = 0;
    this->transpose = transpose;
    if(transpose){
        // 如果是转置存储，则嵌入轴应当放前面，缓存长度放后面
        buffer.reserve({batch, 2, head_dim, max_length}); // [b, 2, hd, l]
    }else{
        buffer.reserve({batch, 2, max_length, head_dim}); // [b, 2, l, hd]
    }
}

void KVCache::append(const yt::YTensor<float, 4>& new_kv) {
    // new_kv: [b, 2, l, hd]
    int new_len = new_kv.shape(2);
    
    // 循环写入。其实是支持直接slice+copy的写法，但是这里需要支持KV缓存循环写入的逻辑，以支持超过最大长度的情况
    // 但是由于模型没有经过特殊训练，像attention sink等问题其实没法解决，超过最大长度的时候还是会说胡话。
    for(int t = 0; t < new_len; t++){
        int pos = write_pos;  // 当前写入位置
        if(transpose){
            // buffer [b, 2, hd, l]
            // new_kv[:, :, t, :] -> buffer[:, :, :, pos]
            auto src = new_kv.slice(2, t, t+1).squeeze(2);          // [b, 2, hd]
            auto dst = buffer.slice(-1, pos, pos+1).squeeze(-1);    // [b, 2, hd]
            dst.copy_(src);                                         // 使用copy_进行数据拷贝，无需考虑源数据内存连续
        }else{
            // buffer [b, 2, l, hd]
            // new_kv[:, :, t, :] -> buffer[:, :, pos, :]
            auto src = new_kv.slice(2, t, t+1);  // [b, 2, 1, hd]
            auto dst = buffer.slice(-2, pos, pos+1);  // [b, 2, 1, hd]
            dst.copy_(src);
        }
        
        // 更新写入位置
        write_pos = (write_pos + 1) % max_len;
        total_written++;
        
        // 更新当前长度（最大为max_len）
        if(cur_len < max_len){
            cur_len++;
        }
    }
}

yt::YTensor<float, 4> KVCache::get() const {
    // 直接返回buffer中当前有效的数据（可能乱序，但mask会处理）    
    if (transpose){
        // buffer [b, 2, hd, l] -> 返回 [b, 2, l, hd]（需要transpose）
        if(full()){
            return yt::YTensor<float, 4>(buffer.transpose());
        }else{
            // 使用slice截取有效长度部分再转置。全过程零拷贝。
            return yt::YTensor<float, 4>(buffer.slice(-1, 0, cur_len).transpose());
        }
    }else{
        // buffer [b, 2, l, hd]
        if(full()){
            return buffer;
        }else{
            return yt::YTensor<float, 4>(buffer.slice(-2, 0, cur_len));
        }
    }
}

yt::YTensor<bool, 2> KVCache::get_mask(int query_len) const {
    // 由于支持了循环写入拓展对话，causal mask的底层排布也与实际kv缓存不同了，因此需要根据kv cache的情况来生成causal mask
    int kv_len = cur_len;  // KV缓存的当前长度
    yt::YTensor<bool, 2> mask(query_len, kv_len);
    
    if(!full()){
        // 没满的时候，标准causal mask: query[i]只能看到kv[0:start+i]
        int start = kv_len - query_len;
        for(int qi = 0; qi < query_len; qi++){
            for(int ki = 0; ki < kv_len; ki++){
                // causal: 只能看到 ki <= start + qi
                mask.at(qi, ki) = (ki <= start + qi);
            }
        }
    }else{
        int start = kv_len - query_len;  // query中第一个token对应的逻辑KV位置
        
        for(int qi = 0; qi < query_len; qi++){
            int q_logical = start + qi;  // query的逻辑位置
            for(int ki = 0; ki < kv_len; ki++){
                // 计算物理位置ki对应的逻辑位置
                int k_logical = (ki - write_pos + max_len) % max_len;
                // causal: 只能看到 k_logical <= q_logical
                mask.at(qi, ki) = (k_logical <= q_logical);
            }
        }
    }
    
    return mask;
}

void YConfig2::scale_lvl(int lvl) {
    // 按照训练设置的模型规模，创建模型
    if (lvl == 0) { num_layers=16; hidden_size=768; num_heads=16; head_dim=128; intermediate_size=2048; }
    else if (lvl == -1) { num_layers=8; hidden_size=512; num_heads=8; head_dim=64; intermediate_size=1536; }
    else if (lvl == -2) { num_layers=4; hidden_size=512; num_heads=8; head_dim=64; intermediate_size=1024; }
}

void RoPECache::precompute(int dim, int max_len, float theta) {
    // 预先计算RoPE的cos和sin矩阵
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

yt::YTensor<float, 3> FFN::forward(const yt::YTensor<float, 3>& x) const {
    // 为了加速计算，将up与gate两个线性变换合并为一个
    auto h = yt::function::linear(x, up);
    auto gate = h.slice(-1, intermediate_size, 2 * intermediate_size);
    auto up_proj = h.slice(-1, 0, intermediate_size);
    yt::function::gelu_(gate);
    up_proj *= gate;
    return yt::function::linear(up_proj, down);
}

yt::YTensor<float, 3> PEGA2::forward(
    const yt::YTensor<float, 3>& x, const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin,
    KVCache* kv_cache) const 
{
    // x: [b, l, hidden_size]
    int b = x.shape(0), l = x.shape(1);
    int h = num_heads, hd = head_dim, hh = h / 2;
    
    // pega架构中，qkv是有一个z的lora低秩分解的，且多个线性变换合并为一个，加速计算。
    auto qkv = yt::function::linear(yt::function::linear(x, qkv_0), qkv_1);

    // 将qkv计算结果拆分为qpe, q, kpe, kv四部分，分别表示 带位置嵌入q，不带位置嵌入q，带位置嵌入k，不带位置嵌入的共享kv
    auto qkv_heads = qkv.reshape(b, l, 2 * hh + 2, hd);
    auto qpe = qkv_heads.slice(2, 0, hh).permute(0, 2, 1, 3);
    auto q = qkv_heads.slice(2, hh, 2 * hh).permute(0, 2, 1, 3);
    auto kpe = qkv_heads.slice(2, 2 * hh, 2 * hh + 1).permute(0, 2, 1, 3);
    auto kv = qkv_heads.slice(2, 2 * hh + 1, 2 * hh + 2).permute(0, 2, 1, 3);

    // 对位嵌入部分进行RoPE嵌入
    ops::rope(qpe, kpe, cos, sin);
    
    // 使用concat将带位置嵌入和不带位置嵌入的部分合并，融合完成前向计算
    auto q_full = yt::YTensor<float, 4>(qpe.concat(q, 1));
    auto kv_full = yt::YTensor<float, 4>(kpe.concat(kv, 1));
    
    // 使用KVCache，先追加新的kv，然后获取完整的kv序列
    bool cache_was_empty = kv_cache && kv_cache->empty();
    if (kv_cache) {
        kv_cache->append(kv_full);
    }
    
    yt::YTensor<float, 4> kv_out;
    if (cache_was_empty && l <= kv_cache->max_len) {
        kv_out = kv_full;
    } else if (kv_cache && !kv_cache->empty()) {
        // 从cache获取完整的kv（包含刚追加的）
        kv_out = kv_cache->get();
    } else {
        kv_out = kv_full;
    }
    
    // 使用 GQA：将 k 从 [b, 2, l_all, hd] 扩展到 [b, 2, hh, l_all, hd]
    // 使用5维张量增加自由度，这样就可以做到零拷贝，相当于repeat interleave。
    auto k_5d = ops::repeat_kv(kv_out, hh);                     // [b, 2, hh, l_all, hd]

    // v是与k的无位置编码部分共享的，直接slice避免拷贝。
    auto v_slice = yt::YTensor<float, 4>(kv_out.slice(1, 1, 2));// [b, 1, l_all, hd]
    auto v_repeated = v_slice.repeat(1, 2, 1, 1);               // [b, 2, l_all, hd] 零拷贝
    auto v_5d = ops::repeat_kv(v_repeated, hh);                 // [b, 2, hh, l_all, hd] 零拷贝
    
    // [b, h, l, hd] -> [b, 2, hh, l, hd]对齐
    yt::YTensor<float, 5> q_5d = q_full.reshape(b, 2, hh, l, hd);
    
    yt::YTensor<float, 5> attn_5d;
    if (l == 1) {
        attn_5d = yt::function::scaledDotProductAttention(
            q_5d, k_5d, v_5d, rsqrt_dim,
            static_cast<yt::YTensor<bool, 2>*>(nullptr), nullptr,
            yt::function::sdpaBackend::FLASH
        );
    } else {
        auto causal_mask = kv_cache ? kv_cache->get_mask(l) : make_causal_mask(l, kv_out.shape(2));
        attn_5d = yt::function::scaledDotProductAttention(
            q_5d, k_5d, v_5d, rsqrt_dim, &causal_mask, nullptr,
            yt::function::sdpaBackend::FLASH
        );
    }

    // 直接在 5D 上进行 gate 操作，避免拷贝
    // 第二个轴上的第一个是带位置嵌入的注意力输出，第二个是不带位置嵌入的注意力输出
    // attn_5d: [b, 2, hh, l, hd]
    // ope = attn_5d[:, 0:1, :, :, :] -> [b, 1, hh, l, hd]
    // onope = attn_5d[:, 1:2, :, :, :] -> [b, 1, hh, l, hd]
    auto ope_5d = yt::YTensor<float, 5>(attn_5d.slice(1, 0, 1));    // [b, 1, hh, l, hd]
    auto onope_5d = yt::YTensor<float, 5>(attn_5d.slice(1, 1, 2));  // [b, 1, hh, l, hd]
    yt::function::gelu_(onope_5d);                                  // ymodel2是对无位置嵌入部分门控
    ope_5d *= onope_5d;

    // 恢复形状
    auto gated = ope_5d.squeeze(1).permute(0, 2, 1, 3).reshape(b, l, hh * hd);
    auto op = yt::function::linear(gated, o);
    return op;
}

yt::YTensor<float, 3> YBlock2::forward(
    const yt::YTensor<float, 3>& x, const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin,
    KVCache* kv_cache) const 
{
    auto h1 = norm1.forward(x);
    auto x2 = attn.forward(h1, cos, sin, kv_cache);
    x2 += x; // 残差链接
    auto h2 = norm2.forward(x2);
    x2 += ffn.forward(h2); // 残差链接
    return x2;
}

void YModel2::init(const YConfig2& cfg) {
    config = cfg;
    embed.reserve(cfg.vocab_size, cfg.hidden_size);
    layers.resize(cfg.num_layers);
    for (auto& layer : layers) {
        layer.attn.hidden_size = cfg.hidden_size;
        layer.attn.num_heads = cfg.num_heads;
        layer.attn.head_dim = cfg.head_dim;
        layer.attn.rsqrt_dim = 1.0f / std::sqrt((float)cfg.head_dim);
        layer.attn.qkv_list = {cfg.num_heads/2 * cfg.head_dim, cfg.num_heads/2 * cfg.head_dim, cfg.head_dim, cfg.head_dim};
        layer.ffn.intermediate_size = cfg.intermediate_size;
        layer.norm1.eps = layer.norm2.eps = cfg.rms_norm_eps;
    }
    norm.eps = cfg.rms_norm_eps;
    rope.precompute(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta);
}

bool YModel2::load(const std::string& path) {
    yt::io::YTensorIO io;
    if (!io.open(path, yt::io::Read)) return false;
    
    // 获取文件中的张量列表
    auto tensor_names = io.getTensorNames();
    std::cout << "  File contains " << tensor_names.size() << " tensors" << std::endl;
    
    int loaded_count = 0;
    yt::YTensorBase base;
    bool useTranspose = false;
    if(yt::info::defaultMatmulBackend == yt::info::MatmulBackend::Eigen){
        // 使用Eigen的时候，权重矩阵主序相同会更加高效。
        useTranspose = true;
    }
    
    if (io.load(base, "model.embed_tokens.weight")) { embed = toFloat<2>(base, false); loaded_count++; }
    
    for (int i = 0; i < config.num_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        if (io.load(base, p + "attn.qkv.0.weight")) { layers[i].attn.qkv_0 = toFloat<2>(base, useTranspose); loaded_count++; }
        if (io.load(base, p + "attn.qkv.1.weight")) { layers[i].attn.qkv_1 = toFloat<2>(base, useTranspose); loaded_count++; }
        if (io.load(base, p + "attn.o.weight")) { layers[i].attn.o = toFloat<2>(base, useTranspose); loaded_count++; }
        if (io.load(base, p + "ffn.up.weight")) { layers[i].ffn.up = toFloat<2>(base, useTranspose); loaded_count++; }
        if (io.load(base, p + "ffn.down.weight")) { layers[i].ffn.down = toFloat<2>(base, useTranspose); loaded_count++; }
        if (io.load(base, p + "norm1.weight")) { layers[i].norm1.weight = toFloat<1>(base); loaded_count++; }
        if (io.load(base, p + "norm2.weight")) { layers[i].norm2.weight = toFloat<1>(base); loaded_count++; }
    }
    if (io.load(base, "model.norm.weight")) { norm.weight = toFloat<1>(base); loaded_count++; }
    
    if (std::find(tensor_names.begin(), tensor_names.end(), "lm_head.weight") != tensor_names.end()) {
        loaded_count++;
        std::cout << "  lm_head.weight found in file (shared with embed_tokens)" << std::endl;
    }

    std::cout << "  Successfully loaded " << loaded_count << "/" << tensor_names.size() << " tensors" << std::endl;
    io.close();
    return true;
}

yt::YTensor<float, 3> YModel2::forward(const yt::YTensor<int, 2>& ids, std::vector<KVCache>* kv_caches) {
    int b = ids.shape(0), l = ids.shape(1);
    // 从第一层的KVCache获取当前全局位置（包含position_offset）
    int start = (kv_caches && !kv_caches->empty() && !(*kv_caches)[0].empty()) 
                ? (*kv_caches)[0].get_global_position() : 0;
    
    int required_positions = start + l;
    if (required_positions > rope.cos.shape(0)) {
        int expanded_positions = std::max(required_positions, rope.cos.shape(0) * 2);
        rope.precompute(config.head_dim, expanded_positions, config.rope_theta);
    }
    
    int hidden = config.hidden_size;
    
    yt::YTensor<float, 3> x(b, l, hidden);
    #pragma omp parallel for collapse(2) proc_bind(close)
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < l; ++j) {
            // 手动词嵌入
            // embed:[vocab_size, hidden]
            int token = ids.at(i, j);
            std::memcpy(&x.at(i, j, 0), embed.data() + token * embed.stride_(0), hidden * sizeof(float));
        }
    }
    
    auto cos_slice = rope.cos.slice(0, start, start + l);
    auto sin_slice = rope.sin.slice(0, start, start + l);
    
    for (int i = 0; i < config.num_layers; ++i) {
        KVCache* cache = (kv_caches && i < (int)kv_caches->size()) ? &(*kv_caches)[i] : nullptr;
        x = layers[i].forward(x, cos_slice, sin_slice, cache);
    }
    return norm.forward(x);
}

void YForCausalLM2::init(const YConfig2& cfg) { 
    config = cfg; 
    model.init(cfg); 
    
    // 初始化每层的KVCache
    kv_caches.resize(cfg.num_layers);
    for (auto& cache : kv_caches) {
        cache.init(1, cfg.max_position_embeddings, cfg.head_dim);
    }
}

bool YForCausalLM2::load(const std::string& path) {
    return model.load(path);
}

void YForCausalLM2::reset_kv_cache() {
    for (auto& cache : kv_caches) {
        cache.reset();
    }
}

int YForCausalLM2::get_kv_cache_len() const {
    if (kv_caches.empty()) return 0;
    return kv_caches[0].cur_len;
}

yt::YTensor<float, 2> YForCausalLM2::forward(const yt::YTensor<int, 2>& ids) {
    auto h = model.forward(ids, &kv_caches);
    int b = h.shape(0), l = h.shape(1);
    
    auto last = h.slice(1, l - 1, l).contiguous().view(b, config.hidden_size);
    auto logits = last.matmul(model.embed.transpose());
    return logits;
}

int YForCausalLM2::get_max_context_len() const {
    return config.max_position_embeddings;
}

std::vector<int> YForCausalLM2::generate(const std::vector<int>& new_ids, int max_tokens, int eos,
                                         std::function<void(int)> on_token) {
    std::vector<int> out;  // 只返回生成的新token，不含输入
    
    // 随机数生成器用于采样
    static std::mt19937 rng(42);
    
    auto sample_token = [&](const yt::YTensor<float, 2>& logits) -> int {
        constexpr int k = 20;
        int vocab_size = logits.shape(1);
        int top_k = std::min(k, vocab_size);
        std::priority_queue<std::pair<float, int>,
            std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> heap;
        for (int i = 0; i < vocab_size; ++i) {
            std::pair<float, int> candidate{logits.at(0, i), i};
            if (static_cast<int>(heap.size()) < top_k) heap.push(candidate);
            else if (candidate.first > heap.top().first) {
                heap.pop();
                heap.push(candidate);
            }
        }
        std::vector<std::pair<float, int>> prob_idx(top_k);
        for (int i = top_k - 1; i >= 0; --i) {
            prob_idx[i] = heap.top();
            heap.pop();
        }
        float max_logit = prob_idx[0].first;
        float sum = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            prob_idx[i].first = std::exp(prob_idx[i].first - max_logit);
            sum += prob_idx[i].first;
        }
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
    if (new_len == 0 || max_tokens <= 0) return out;

    yt::YTensor<float, 2> logits;
    int consumed = 0;
    while (consumed < new_len) {
        int remaining = get_max_context_len() - get_kv_cache_len();
        int chunk = remaining > 0 ? std::min(remaining, new_len - consumed) : 1;
        yt::YTensor<int, 2> input(1, chunk);
        std::copy(new_ids.begin() + consumed, new_ids.begin() + consumed + chunk, input.data());
        logits = forward(input);
        consumed += chunk;
    }
    int next = sample_token(logits);
    
    if (next == eos) {
        return out;
    }
    
    out.push_back(next);
    if (on_token) on_token(next);
    
    bool final_token_cached = false;
    for (int step = 1; step < max_tokens; ++step) {
        yt::YTensor<int, 2> nid(1, 1);
        nid.at(0, 0) = next;

        auto sl = forward(nid);
        final_token_cached = true;
        next = sample_token(sl);

        if (next == eos) {
            break;
        }

        out.push_back(next);
        final_token_cached = false;
        if (on_token) on_token(next);
    }
    if (!out.empty() && !final_token_cached) {
        yt::YTensor<int, 2> final_input(1, 1);
        final_input.at(0, 0) = out.back();
        (void)forward(final_input);
    }
    return out;
}

namespace ops {

void rope(yt::YTensor<float, 4>& q, yt::YTensor<float, 4>& k, const yt::YTensor<float, 2>& cos_cache, const yt::YTensor<float, 2>& sin_cache) {
    // 使用at()来支持非contiguous
    int b = q.shape(0), h = q.shape(1), l = q.shape(2), hd = q.shape(3);
    int half = hd / 2;
    
    // 使用串行处理最内层循环，避免数据竞争
    // 因为di和di+half需要同时读写，不能并行处理最内层
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

// 零拷贝版本的 repeat_kv，返回 5D 张量 [b, h, n, l, ch]
yt::YTensor<float, 5> repeat_kv(const yt::YTensor<float, 4>& x, int n) {
    if (n == 1) {
        return x.unsqueeze(2);  // [b, h, l, ch] -> [b, h, 1, l, ch]
    }
    // [b, h, l, ch] -> [b, h, 1, l, ch] -> repeat -> [b, h, n, l, ch]
    auto x5d = x.unsqueeze(2);  // 零拷贝
    return yt::YTensor<float, 5>(x5d.repeat(1, 1, n, 1, 1));  // 零拷贝
}

}// namespace ops
}// namespace ymodel2
