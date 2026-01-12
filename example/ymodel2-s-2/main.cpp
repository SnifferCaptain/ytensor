#include "ymodel2.hpp"
#include "tokenlizer.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include "../../include/3rd/backward.hpp"

backward::SignalHandling sh;

int main() {
    std::cout << "===SnifferCaptain Chat===" << std::endl;
    std::cout << std::endl;
    
    // 加载tokenizer
    MiniTokenizer tokenizer;
    MiniTokenizer::Config tok_cfg;
    tok_cfg.model_dir = "../model";
    std::string err;
    std::cout << "loading tokenizer..." << std::endl;
    if (!tokenizer.load(tok_cfg, &err)) {
        std::cerr << "loading tokenizer failed: " << err << std::endl;
        return 1;
    }
    std::cout << "tokenizer loaded successfully." << std::endl;
    
    // 初始化模型配置
    ymodel2::YConfig2 config;
    config.scale_lvl(-2);
    
    // 初始化并加载模型
    ymodel2::YForCausalLM2 model;
    model.init(config);
    
    std::cout << "loading model weights..." << std::endl;
    std::string weight_path = "../model/y2_sft_s-2.yt";
    if (!model.load(weight_path)) {
        std::cerr << "loading model weights failed" << std::endl;
        return 1;
    }
    std::cout << "model loaded successfully." << std::endl;
    std::cout << "using backend: ";
    if(yt::infos::defaultMatmulBackend == yt::infos::MatmulBackend::Naive){
        std::cout << "Naive";
    }else if(yt::infos::defaultMatmulBackend == yt::infos::MatmulBackend::Eigen){
        std::cout << "Eigen";
    }else if(yt::infos::defaultMatmulBackend == yt::infos::MatmulBackend::AVX2){
        std::cout << "AVX2";
    }
    std::cout << std::endl;

    std::cout << "===============recipe================" << std::endl;
    std::cout << "  send your message with [Enter]" << std::endl;
    std::cout << "  'exit' or 'quit' to exit" << std::endl;
    std::cout << "  'clear' to clear chat history" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;
    
    int eos_id = tokenizer.id_eos().value_or(2);
    
    // 存储对话历史
    std::vector<std::pair<std::string, std::string>> chat_history;
    // 记录上一次对话结束后的完整token序列（用于计算增量）
    std::vector<int64_t> prev_tokens;
    
    while (true) {
        std::cout << "You: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        
        // 去除首尾空白
        size_t start = user_input.find_first_not_of(" \t\n\r");
        size_t end = user_input.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) {
            continue;  // 空输入，继续等待
        }
        user_input = user_input.substr(start, end - start + 1);
        
        // 检查退出命令
        if (user_input == "exit" || user_input == "quit") {
            std::cout << std::endl;
            std::cout << "goodbye!" << std::endl;
            break;
        }
        
        // 检查清空历史命令
        if (user_input == "clear") {
            chat_history.clear();
            prev_tokens.clear();
            model.reset_kv_cache();
            std::cout << "chat history cleared." << std::endl;
            std::cout << std::endl;
            continue;
        }
        
        // 添加用户消息到历史
        chat_history.push_back({"user", user_input});
        
        // 使用chat template构建完整prompt
        std::string chat_prompt = tokenizer.apply_chat_template(chat_history, true);
        auto input_ids = tokenizer.encode(chat_prompt);
        
        // 生成回复
        std::cout << "SnifferCaptain: " << std::flush;
        std::string response;
        
        auto et0 = std::chrono::system_clock::now();
        auto dt0 = std::chrono::system_clock::now();
        auto et1 = std::chrono::system_clock::now();
        
        int encode_len = 0;
        int decode_len = 0;
        
        // 判断是否可以使用增量生成
        bool use_incremental = !prev_tokens.empty() && prev_tokens.size() < input_ids.size();
        
        // 验证前缀是否匹配（确保KV缓存有效）
        if (use_incremental) {
            for (size_t i = 0; i < prev_tokens.size(); ++i) {
                if (prev_tokens[i] != input_ids[i]) {
                    use_incremental = false;
                    break;
                }
            }
        }
        
        std::vector<int> new_ids;
        
        if (use_incremental) {
            // 增量生成：只处理新增的token
            for (size_t i = prev_tokens.size(); i < input_ids.size(); ++i) {
                new_ids.push_back(static_cast<int>(input_ids[i]));
            }
        } else {
            // 首次对话或前缀不匹配：重置KV cache，完整处理
            model.reset_kv_cache();
            prev_tokens.clear();
            new_ids = std::vector<int>(input_ids.begin(), input_ids.end());
        }
        
        encode_len = new_ids.size();
        
        // 统一使用generate函数（自动处理KV缓存复用和上下文溢出）
        model.generate(new_ids, 8192, eos_id, [&](int token_id) {
            std::string token_str = tokenizer.decode({token_id});
            response += token_str;
            std::cout << token_str << std::flush;
            if(!decode_len++){
                et1 = std::chrono::system_clock::now();
                dt0 = std::chrono::system_clock::now();
            }
        });
        
        auto dt1 = std::chrono::system_clock::now();
        float etps = static_cast<float>(encode_len) / std::chrono::duration<float>(et1 - et0).count();
        float dtps = static_cast<float>(decode_len) / std::chrono::duration<float>(dt1 - dt0).count();
        std::cout << std::endl;
        std::cout << "[Info] encoding length: " << encode_len << ", decoding length: " << decode_len
                  << ", encoding speed: " << etps << " tokens/s"
                  << ", decoding speed: " << dtps << " tokens/s"
                  << std::endl
                  << "       context length: " << model.get_kv_cache_len() << "/" << model.get_max_context_len() << " tokens"
                  << std::endl;
        std::cout << std::endl;
        
        // 添加助手回复到历史
        chat_history.push_back({"assistant", response});
        
        // 更新prev_tokens：当前完整对话（包含assistant回复）的token序列
        // 这样下次用户输入时，可以只处理新增的user消息部分
        std::string full_chat_with_response = tokenizer.apply_chat_template(chat_history, false);
        prev_tokens = tokenizer.encode(full_chat_with_response);
    }
    
    return 0;
}
