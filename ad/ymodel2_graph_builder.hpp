#pragma once
/***************
* @file: ymodel2_graph_builder.hpp
* @brief: YModel2计算图构建器
***************/
#include "computation_graph.hpp"
#include <string>

namespace yt {
namespace ad {

// YModel2计算图构建器
class YModel2GraphBuilder {
public:
    // 构建YModel2-s-2的计算图（scale=-2）
    static ComputationGraph buildYModel2S2Graph();
    
    // 从配置构建YModel2计算图
    static ComputationGraph buildYModel2Graph(
        int num_layers = 4,
        int hidden_size = 512,
        int num_heads = 8,
        int head_dim = 64,
        int intermediate_size = 1024,
        int vocab_size = 6400,
        int max_position_embeddings = 8192
    );
};

} // namespace ad
} // namespace yt
