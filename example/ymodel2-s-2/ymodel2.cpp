#include <string>
#include "ymodel2.hpp"
#include <iostream>
#include <cstring>
#include <random>
#include <algorithm>

namespace ymodel2 {

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
    buffer.fill(0.0f);  // 其实这是不必要的
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

yt::YTensor<float, 2> KVCache::get_mask(int query_len) const {
    // 由于支持了循环写入拓展对话，causal mask的底层排布也与实际kv缓存不同了，因此需要根据kv cache的情况来生成causal mask
    int kv_len = cur_len;  // KV缓存的当前长度
    yt::YTensor<float, 2> mask(query_len, kv_len);
    
    if(!full()){
        // 没满的时候，标准causal mask: query[i]只能看到kv[0:start+i]
        int start = kv_len - query_len;
        for(int qi = 0; qi < query_len; qi++){
            for(int ki = 0; ki < kv_len; ki++){
                // causal: 只能看到 ki <= start + qi
                mask.at(qi, ki) = (ki <= start + qi) ? 0.0f : -1e9f;
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
                mask.at(qi, ki) = (k_logical <= q_logical) ? 0.0f : -1e9f;
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
    auto out = x.clone();
    RMSNorm::forward_(out);
    return out;
}

void RMSNorm::forward_(yt::YTensor<float, 3>& x) const {
    // rms norm其实相当于l2 norm + scale + weight的融合
    auto x_sq = x * x;                  // 计算平方
    auto sum_sq = x_sq.mean(-1);        // 沿平方的最后一个维度求平均值[b, l, 1]

    x.broadcastInplace([this](float& a, const float& b, const float& c) {
        // 逐元素执行 输出 = 输入 / (根号(平方的均值)) * 权重，eps为防止除零的小数
        a = a / (sqrtf(b + eps)) * c;
    }, sum_sq, weight.view(1, 1, x.shape(-1)));
}

yt::YTensor<float, 3> FFN::forward(const yt::YTensor<float, 3>& x) const {
    // 为了加速计算，将up与gate两个线性变换合并为一个
    auto h = ops::linear(x, up);
    auto gate = h.slice(-1, intermediate_size, 2 * intermediate_size);
    auto up_proj = h.slice(-1, 0, intermediate_size);
    ops::gelu_(gate);
    up_proj *= gate;
    return ops::linear(up_proj, down);
}

yt::YTensor<float, 3> PEGA2::forward(
    const yt::YTensor<float, 3>& x, const yt::YTensor<float, 2>& cos, const yt::YTensor<float, 2>& sin,
    KVCache* kv_cache) const 
{
    // x: [b, l, hidden_size]
    int b = x.shape(0), l = x.shape(1);
    int h = num_heads, hd = head_dim, hh = h / 2;
    
    // pega架构中，qkv是有一个z的lora低秩分解的，且多个线性变换合并为一个，加速计算。
    auto qkv = ops::linear(ops::linear(x, qkv_0), qkv_1);

    // 将qkv计算结果拆分为qpe, q, kpe, kv四部分，分别表示 带位置嵌入q，不带位置嵌入q，带位置嵌入k，不带位置嵌入的共享kv
    int qpe_size = hh * hd, q_size = hh * hd, kpe_size = hd, kv_size = hd;
    auto qpe_flat = qkv.slice(-1, 0, qpe_size);
    auto q_flat = qkv.slice(-1, qpe_size, qpe_size + q_size);
    auto kpe_flat = qkv.slice(-1, qpe_size + q_size, qpe_size + q_size + kpe_size);
    auto kv_flat = qkv.slice(-1, qpe_size + q_size + kpe_size, qpe_size + q_size + kpe_size + kv_size);

    auto qpe = qpe_flat.reshape(b, l, hh, hd).permute(0, 2, 1, 3);  // 拆分多头，使用GQA
    auto q = q_flat.reshape(b, l, hh, hd).permute(0, 2, 1, 3);
    auto kpe = kpe_flat.reshape(b, l, 1, hd).permute(0, 2, 1, 3);
    auto kv = kv_flat.reshape(b, l, 1, hd).permute(0, 2, 1, 3);

    // 对位嵌入部分进行RoPE嵌入
    ops::rope(qpe, kpe, cos, sin);
    
    // 使用concat将带位置嵌入和不带位置嵌入的部分合并，融合完成前向计算
    auto q_full = yt::YTensor<float, 4>(qpe.concat(q, 1));
    auto kv_full = yt::YTensor<float, 4>(kpe.concat(kv, 1));
    
    // 使用KVCache，先追加新的kv，然后获取完整的kv序列
    if (kv_cache) {
        kv_cache->append(kv_full);
    }
    
    yt::YTensor<float, 4> kv_out;
    if (kv_cache && !kv_cache->empty()) {
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
    
    // 获取causal mask
    yt::YTensor<float, 2> causal_mask = kv_cache->get_mask(l);
    
    // 使用 scaledDotProductAttention完成标准注意力计算。
    // q_5d, k_5d, v_5d: [b, 2, hh, l, hd]
    auto attn_5d = yt::function::scaledDotProductAttention(
        q_5d, k_5d, v_5d, 
        rsqrt_dim,  // scale
        &causal_mask
    );

    // 直接在 5D 上进行 gate 操作，避免拷贝
    // 第二个轴上的第一个是带位置嵌入的注意力输出，第二个是不带位置嵌入的注意力输出
    // attn_5d: [b, 2, hh, l, hd]
    // ope = attn_5d[:, 0:1, :, :, :] -> [b, 1, hh, l, hd]
    // onope = attn_5d[:, 1:2, :, :, :] -> [b, 1, hh, l, hd]
    auto ope_5d = yt::YTensor<float, 5>(attn_5d.slice(1, 0, 1));    // [b, 1, hh, l, hd]
    auto onope_5d = yt::YTensor<float, 5>(attn_5d.slice(1, 1, 2));  // [b, 1, hh, l, hd]
    ops::gelu_(onope_5d);                                           // ymodel2是对无位置嵌入部分门控
    ope_5d *= onope_5d;

    // 恢复形状
    auto gated = ope_5d.squeeze(1).permute(0, 2, 1, 3).reshape(b, l, hh * hd);
    auto op = ops::linear(gated, o);
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
    if(yt::infos::defaultMatmulBackend == yt::infos::MatmulBackend::Eigen){
        // 使用Eigen的时候，权重矩阵主序相同会更加高效。
        useTranspose = true;
    }
    
    if (io.load(base, "model.embed_tokens.weight")) { embed = toFloat<2>(base, useTranspose); loaded_count++; }
    
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
    
    // 尝试加载 lm_head（如果存在）
    if (io.load(base, "lm_head.weight")) {
        // lm_head 与 embed_tokens 共享权重，这里只是验证文件中有这个张量
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
    
    // 安全检查：如果全局位置超出预计算范围，回退到使用相对位置
    if (start + l > config.max_position_embeddings) {
        // 回退到相对位置（从kv cache当前长度开始）
        start = (kv_caches && !kv_caches->empty()) ? (*kv_caches)[0].cur_len : 0;
        // 如果仍然超出，则从0开始（极端情况，通常不应该发生）
        if (start + l > config.max_position_embeddings) {
            start = 0;
        }
    }
    
    int hidden = config.hidden_size;
    
    yt::YTensor<float, 3> x(b, l, hidden);
    #pragma omp simd collapse(2)
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < l; ++j) {
            // 手动词嵌入
            // embed:[vocab_size, hidden]
            auto tokenEmbed = embed.slice(0, ids.at(i, j), ids.at(i, j) + 1).contiguous();
            std::memcpy(&x.at(i, j, 0), tokenEmbed.data(), hidden * sizeof(float));
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
        auto probs = yt::function::softmax(logits, -1);
        constexpr int k = 20;
        int vocab_size = probs.shape(1);
        std::vector<std::pair<float, int>> prob_idx(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            prob_idx[i] = {probs.at(0, i), i};
        }
        std::partial_sort(prob_idx.begin(), prob_idx.begin() + k, prob_idx.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        float sum = 0;
        for (int i = 0; i < k; ++i) sum += prob_idx[i].first;
        std::uniform_real_distribution<float> dist(0.0f, sum);
        float r = dist(rng);
        float cumsum = 0;
        for (int i = 0; i < k; ++i) {
            cumsum += prob_idx[i].first;
            if (r <= cumsum) return prob_idx[i].second;
        }
        return prob_idx[0].second;
    };
    
    // 循环KV缓存会自动覆盖旧数据，无需手动截断
    int new_len = static_cast<int>(new_ids.size());
    int max_ctx = get_max_context_len();
    
    if (new_len > max_ctx) {
        std::cerr << "[Warning] Input length (" << new_len << ") exceeds max context length (" 
                  << max_ctx << "), only last " << max_ctx << " tokens will be used\n";
    }
    
    // prefill新增的tokens（循环KV cache会自动处理溢出）
    yt::YTensor<int, 2> input(1, new_len);
    std::copy(new_ids.begin(), new_ids.end(), input.data());
    
    auto logits = forward(input);  // KVCache在forward内部自动更新
    int next = sample_token(logits);
    
    if (next == eos) {
        return out;
    }
    
    out.push_back(next);
    if (on_token) on_token(next);
    
    // 自回归生成（循环KV cache会自动处理溢出）
    for (int step = 1; step < max_tokens; ++step) {
        yt::YTensor<int, 2> nid(1, 1);
        nid.at(0, 0) = next;
        
        auto sl = forward(nid);
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

template<int dim>
yt::YTensor<float, dim> linear(const yt::YTensor<float, dim>& x, const yt::YTensor<float, 2>& weight) {
    // 权重遵循pytorch的权重形状，即[输出, 输入]，因此需要转置
    return x.matmul(weight.transpose());
}

template yt::YTensor<float, 3> linear(const yt::YTensor<float, 3>&, const yt::YTensor<float, 2>&);

// 原地版本的gelu（tanh近似），也可以使用std::erf去计算
template<int dim>
yt::YTensor<float, dim>& gelu_(yt::YTensor<float, dim>& x) {
    constexpr float sqrt_2_pi = 0.7978845608028654f;
    return x.broadcastInplace([sqrt_2_pi](float& v) {
        v = 0.5f * v * (1.0f + std::tanh(sqrt_2_pi * (v + 0.044715f * v * v * v)));
    });
}

template yt::YTensor<float, 3>& gelu_(yt::YTensor<float, 3>&);
template yt::YTensor<float, 4>& gelu_(yt::YTensor<float, 4>&);
template yt::YTensor<float, 5>& gelu_(yt::YTensor<float, 5>&);

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
