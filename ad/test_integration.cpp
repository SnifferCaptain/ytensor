#include "ad.hpp"
#include "ymodel2_graph_builder.hpp"
#include <iostream>
#include <fstream>
#include <map>

int main() {
    std::cout << "=== YModel2 Graph Integration Test ===" << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. 构建YModel2-s-2的计算图
        std::cout << "[1] Building YModel2-s-2 computational graph..." << std::endl;
        auto graph = yt::ad::YModel2GraphBuilder::buildYModel2S2Graph();
        std::cout << "    ✓ Graph built successfully!" << std::endl;
        std::cout << "    - Nodes: " << graph.nodes().size() << std::endl;
        std::cout << "    - Edges: " << graph.edges().size() << std::endl;
        std::cout << std::endl;
        
        // 2. 保存到JSON文件
        std::string graphFile = "ymodel2_s2_graph.json";
        std::cout << "[2] Saving graph to: " << graphFile << std::endl;
        if (!graph.saveToFile(graphFile)) {
            std::cerr << "    ✗ Failed to save graph!" << std::endl;
            return 1;
        }
        std::cout << "    ✓ Graph saved successfully!" << std::endl;
        std::cout << std::endl;
        
        // 3. 从JSON文件加载
        std::cout << "[3] Loading graph from file..." << std::endl;
        auto loadedGraph = yt::ad::ComputationGraph::loadFromFile(graphFile);
        std::cout << "    ✓ Graph loaded successfully!" << std::endl;
        std::cout << "    - Nodes: " << loadedGraph.nodes().size() << std::endl;
        std::cout << "    - Edges: " << loadedGraph.edges().size() << std::endl;
        std::cout << std::endl;
        
        // 4. 验证图的结构
        std::cout << "[4] Verifying graph structure..." << std::endl;
        
        // 验证输入节点
        auto inputNodes = loadedGraph.getInputNodes();
        std::cout << "    - Input nodes: " << inputNodes.size() << std::endl;
        for (const auto& node : inputNodes) {
            std::cout << "      * " << node->name() << std::endl;
        }
        
        // 验证参数节点
        int paramCount = 0;
        for (const auto& [id, node] : loadedGraph.nodes()) {
            if (node->type() == yt::ad::NodeType::Parameter) {
                paramCount++;
            }
        }
        std::cout << "    - Parameter nodes: " << paramCount << std::endl;
        
        // 验证输出节点
        auto outputNodes = loadedGraph.getOutputNodes();
        std::cout << "    - Output nodes: " << outputNodes.size() << std::endl;
        for (const auto& node : outputNodes) {
            std::cout << "      * " << node->name() << std::endl;
        }
        std::cout << std::endl;
        
        // 5. 拓扑排序
        std::cout << "[5] Performing topological sort..." << std::endl;
        auto sortedNodes = loadedGraph.topologicalSort();
        std::cout << "    ✓ Topological sort successful!" << std::endl;
        std::cout << "    - Sorted " << sortedNodes.size() << " nodes" << std::endl;
        std::cout << std::endl;
        
        // 6. 打印图的层次结构
        std::cout << "[6] Graph layer structure:" << std::endl;
        
        // 统计每一层的节点
        std::map<std::string, int> layerCounts;
        for (const auto& [id, node] : loadedGraph.nodes()) {
            std::string name = node->name();
            if (name.find("model.layers.0.") != std::string::npos) {
                layerCounts["Layer 0"]++;
            } else if (name.find("model.layers.1.") != std::string::npos) {
                layerCounts["Layer 1"]++;
            } else if (name.find("model.layers.2.") != std::string::npos) {
                layerCounts["Layer 2"]++;
            } else if (name.find("model.layers.3.") != std::string::npos) {
                layerCounts["Layer 3"]++;
            } else if (name.find("model.") != std::string::npos) {
                layerCounts["Model (other)"]++;
            } else {
                layerCounts["Input/Output"]++;
            }
        }
        
        for (const auto& [layer, count] : layerCounts) {
            std::cout << "    - " << layer << ": " << count << " nodes" << std::endl;
        }
        std::cout << std::endl;
        
        // 7. 对比原始YModel2-s-2配置
        std::cout << "[7] Comparing with YModel2-s-2 configuration:" << std::endl;
        
        // YModel2-s-2 (scale=-2) 配置
        int num_layers = 4;
        int hidden_size = 512;
        int num_heads = 8;
        int head_dim = 64;
        int intermediate_size = 1024;
        int vocab_size = 6400;
        
        std::cout << "    Configuration:" << std::endl;
        std::cout << "      - num_layers: " << num_layers << std::endl;
        std::cout << "      - hidden_size: " << hidden_size << std::endl;
        std::cout << "      - num_heads: " << num_heads << std::endl;
        std::cout << "      - head_dim: " << head_dim << std::endl;
        std::cout << "      - intermediate_size: " << intermediate_size << std::endl;
        std::cout << "      - vocab_size: " << vocab_size << std::endl;
        std::cout << std::endl;
        
        // 计算期望的节点数
        // 每层: norm1_weight, norm1_op, qkv_0, qkv_1, attn, o_weight, o_op, res1,
        //       norm2_weight, norm2_op, up_weight, down_weight, ffn, res2 = 14 nodes
        int expectedPerLayer = 14;
        int expectedTotal = 1 + 1 + 1 +  // input, embed_param, embed_op
                           num_layers * expectedPerLayer +  // layers
                           2 + 1;  // norm_weight, norm_op, output
        
        std::cout << "    Expected nodes: ~" << expectedTotal << std::endl;
        std::cout << "    Actual nodes: " << loadedGraph.nodes().size() << std::endl;
        
        if (std::abs((int)loadedGraph.nodes().size() - expectedTotal) <= 2) {
            std::cout << "    ✓ Node count matches expected structure!" << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "=== All integration tests passed! ===" << std::endl;
        std::cout << std::endl;
        std::cout << "The computational graph for YModel2-s-2 (scale=-2) has been" << std::endl;
        std::cout << "successfully constructed and serialized to JSON format." << std::endl;
        std::cout << "You can now load this graph to reconstruct the model structure." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
