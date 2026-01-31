#include "ymodel2_graph_builder.hpp"
#include <sstream>

namespace yt {
namespace ad {

ComputationGraph YModel2GraphBuilder::buildYModel2S2Graph() {
    // YModel2-s-2 使用 scale=-2 的配置
    return buildYModel2Graph(4, 512, 8, 64, 1024, 6400, 8192);
}

ComputationGraph YModel2GraphBuilder::buildYModel2Graph(
    int num_layers,
    int hidden_size,
    int num_heads,
    int head_dim,
    int intermediate_size,
    int vocab_size,
    int max_position_embeddings
) {
    ComputationGraph graph;
    
    // 添加输入节点
    int inputIds = graph.addNode("input_ids", NodeType::Input);
    auto inputNode = graph.getNode(inputIds);
    inputNode->setAttribute("shape", "[batch, seq_len]");
    inputNode->setAttribute("dtype", "int32");
    
    // 添加嵌入层参数
    int embedParam = graph.addNode("model.embed_tokens.weight", NodeType::Parameter);
    auto embedNode = graph.getNode(embedParam);
    embedNode->setAttribute("shape", "[" + std::to_string(vocab_size) + ", " + std::to_string(hidden_size) + "]");
    embedNode->setAttribute("dtype", "float32");
    
    // 添加嵌入操作
    int embedOp = graph.addNode("embedding", NodeType::Embedding);
    graph.addEdge(inputIds, embedOp);
    graph.addEdge(embedParam, embedOp);
    
    int prevOutput = embedOp;
    
    // 构建每一层
    for (int layer = 0; layer < num_layers; ++layer) {
        std::string layerPrefix = "model.layers." + std::to_string(layer) + ".";
        
        // ===== Attention Block =====
        
        // Norm1
        int norm1Weight = graph.addNode(layerPrefix + "norm1.weight", NodeType::Parameter);
        auto norm1WeightNode = graph.getNode(norm1Weight);
        norm1WeightNode->setAttribute("shape", "[" + std::to_string(hidden_size) + "]");
        
        int norm1Op = graph.addNode(layerPrefix + "norm1", NodeType::RMSNorm);
        auto norm1Node = graph.getNode(norm1Op);
        norm1Node->setAttribute("eps", "1e-8");
        graph.addEdge(prevOutput, norm1Op);
        graph.addEdge(norm1Weight, norm1Op);
        
        // Attention QKV projection (split into qkv_0 and qkv_1 for LoRA)
        int qkv0Weight = graph.addNode(layerPrefix + "attn.qkv.0.weight", NodeType::Parameter);
        auto qkv0Node = graph.getNode(qkv0Weight);
        // LoRA低秩分解: 使用hidden_size/4作为中间维度进行降维
        int qkv_hidden = hidden_size / 4;
        qkv0Node->setAttribute("shape", "[" + std::to_string(qkv_hidden) + ", " + std::to_string(hidden_size) + "]");
        
        int qkv1Weight = graph.addNode(layerPrefix + "attn.qkv.1.weight", NodeType::Parameter);
        auto qkv1Node = graph.getNode(qkv1Weight);
        int qkv_out_size = num_heads * head_dim + head_dim + head_dim;  // qpe + q + kpe + kv
        qkv1Node->setAttribute("shape", "[" + std::to_string(qkv_out_size) + ", " + std::to_string(qkv_hidden) + "]");
        
        // Attention operation
        int attnOp = graph.addNode(layerPrefix + "attn", NodeType::Attention);
        auto attnNode = graph.getNode(attnOp);
        attnNode->setAttribute("num_heads", std::to_string(num_heads));
        attnNode->setAttribute("head_dim", std::to_string(head_dim));
        attnNode->setAttribute("hidden_size", std::to_string(hidden_size));
        graph.addEdge(norm1Op, attnOp);
        graph.addEdge(qkv0Weight, attnOp);
        graph.addEdge(qkv1Weight, attnOp);
        
        // Attention output projection
        int oWeight = graph.addNode(layerPrefix + "attn.o.weight", NodeType::Parameter);
        auto oNode = graph.getNode(oWeight);
        int half_heads = num_heads / 2;
        int attn_out_size = half_heads * head_dim;
        oNode->setAttribute("shape", "[" + std::to_string(hidden_size) + ", " + std::to_string(attn_out_size) + "]");
        
        int oOp = graph.addNode(layerPrefix + "attn.o", NodeType::Linear);
        graph.addEdge(attnOp, oOp);
        graph.addEdge(oWeight, oOp);
        
        // Residual connection 1
        int res1 = graph.addNode(layerPrefix + "residual1", NodeType::Add);
        graph.addEdge(prevOutput, res1);
        graph.addEdge(oOp, res1);
        
        // ===== FFN Block =====
        
        // Norm2
        int norm2Weight = graph.addNode(layerPrefix + "norm2.weight", NodeType::Parameter);
        auto norm2WeightNode = graph.getNode(norm2Weight);
        norm2WeightNode->setAttribute("shape", "[" + std::to_string(hidden_size) + "]");
        
        int norm2Op = graph.addNode(layerPrefix + "norm2", NodeType::RMSNorm);
        auto norm2Node = graph.getNode(norm2Op);
        norm2Node->setAttribute("eps", "1e-8");
        graph.addEdge(res1, norm2Op);
        graph.addEdge(norm2Weight, norm2Op);
        
        // FFN
        int upWeight = graph.addNode(layerPrefix + "ffn.up.weight", NodeType::Parameter);
        auto upNode = graph.getNode(upWeight);
        // YModel2使用gate和up合并的权重，因此维度是intermediate_size * 2
        upNode->setAttribute("shape", "[" + std::to_string(intermediate_size * 2) + ", " + std::to_string(hidden_size) + "]");
        
        int downWeight = graph.addNode(layerPrefix + "ffn.down.weight", NodeType::Parameter);
        auto downNode = graph.getNode(downWeight);
        downNode->setAttribute("shape", "[" + std::to_string(hidden_size) + ", " + std::to_string(intermediate_size) + "]");
        
        int ffnOp = graph.addNode(layerPrefix + "ffn", NodeType::FFN);
        auto ffnNode = graph.getNode(ffnOp);
        ffnNode->setAttribute("intermediate_size", std::to_string(intermediate_size));
        graph.addEdge(norm2Op, ffnOp);
        graph.addEdge(upWeight, ffnOp);
        graph.addEdge(downWeight, ffnOp);
        
        // Residual connection 2
        int res2 = graph.addNode(layerPrefix + "residual2", NodeType::Add);
        graph.addEdge(res1, res2);
        graph.addEdge(ffnOp, res2);
        
        prevOutput = res2;
    }
    
    // 最后的归一化层
    int normWeight = graph.addNode("model.norm.weight", NodeType::Parameter);
    auto normWeightNode = graph.getNode(normWeight);
    normWeightNode->setAttribute("shape", "[" + std::to_string(hidden_size) + "]");
    
    int normOp = graph.addNode("model.norm", NodeType::RMSNorm);
    auto normNode = graph.getNode(normOp);
    normNode->setAttribute("eps", "1e-8");
    graph.addEdge(prevOutput, normOp);
    graph.addEdge(normWeight, normOp);
    
    // 输出节点
    int output = graph.addNode("output", NodeType::Output);
    graph.addEdge(normOp, output);
    
    return graph;
}

} // namespace ad
} // namespace yt
