#include "ymodel2.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace {

void fillWeights(ymodel2::YForCausalLM2& model, unsigned seed) {
    yt::YTensor<float, 1>::seed(seed);
    const auto& cfg = model.config;
    model.model.embed = yt::YTensor<float, 2>::randn(cfg.vocab_size, cfg.hidden_size);
    model.model.norm.weight = yt::YTensor<float, 1>::randn(cfg.hidden_size);
    for (auto& layer : model.model.layers) {
        layer.norm1.weight = yt::YTensor<float, 1>::randn(cfg.hidden_size);
        layer.norm2.weight = yt::YTensor<float, 1>::randn(cfg.hidden_size);
        layer.attn.qkv_0 = yt::YTensor<float, 2>::randn(cfg.hidden_size, cfg.hidden_size);
        layer.attn.qkv_1 = yt::YTensor<float, 2>::randn(cfg.num_heads * cfg.head_dim + 2 * cfg.head_dim, cfg.hidden_size);
        layer.attn.o = yt::YTensor<float, 2>::randn(cfg.hidden_size, (cfg.num_heads / 2) * cfg.head_dim);
        layer.ffn.up = yt::YTensor<float, 2>::randn(2 * cfg.intermediate_size, cfg.hidden_size);
        layer.ffn.down = yt::YTensor<float, 2>::randn(cfg.hidden_size, cfg.intermediate_size);
    }
}

void moveToDevice(ymodel2::YForCausalLM2& model, const std::string& device) {
    model.model.embed.to_(device);
    model.model.norm.weight.to_(device);
    model.model.rope.cos.to_(device);
    model.model.rope.sin.to_(device);
    for (auto& layer : model.model.layers) {
        layer.norm1.weight.to_(device);
        layer.norm2.weight.to_(device);
        layer.attn.qkv_0.to_(device);
        layer.attn.qkv_1.to_(device);
        layer.attn.o.to_(device);
        layer.ffn.up.to_(device);
        layer.ffn.down.to_(device);
    }
    for (auto& cache : model.kv_caches) {
        cache.buffer.to_(device);
    }
}

void expectClose(const yt::YTensor<float, 2>& a, const yt::YTensor<float, 2>& b, float eps = 1e-3f) {
    assert(a.shape() == b.shape());
    for (int i = 0; i < a.shape(0); ++i) {
        for (int j = 0; j < a.shape(1); ++j) {
            const float da = a.at(i, j);
            const float db = b.at(i, j);
            if (std::fabs(da - db) > eps) {
                std::cerr << "mismatch at (" << i << "," << j << "): " << da << " vs " << db << std::endl;
                std::abort();
            }
        }
    }
}

} // namespace

int main() {
    ymodel2::YConfig2 cfg;
    cfg.scale_lvl(-2);

    ymodel2::YForCausalLM2 cpu;
    ymodel2::YForCausalLM2 gpu;
    cpu.init(cfg);
    gpu.init(cfg);
    fillWeights(cpu, 7);
    fillWeights(gpu, 7);
    moveToDevice(gpu, "kompute");

    yt::YTensor<int, 2> ids(1, 8);
    for (int i = 0; i < 8; ++i) ids.at(0, i) = i % cfg.vocab_size;

    auto cpuLogits = cpu.forward(ids);
    auto gpuLogits = gpu.forward(ids).to("cpu");
    expectClose(cpuLogits, gpuLogits);

    std::cout << "ymodel2 parity ok" << std::endl;
    return 0;
}
