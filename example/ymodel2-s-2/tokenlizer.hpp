#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <utility>
#include <functional>

class MiniTokenizer {
public:
	struct Config {
		std::string model_dir = "model"; // 需包含 tokenizer.json & tokenizer_config.json
	};
	struct PairHash {
		size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
			std::hash<std::string> h; return h(p.first) * 1315423911u + h(p.second);
		}
	};

	bool load(const Config& cfg, std::string* err = nullptr);

	std::vector<int64_t> encode(const std::string& text) const;
	std::string decode(const std::vector<int64_t>& ids) const;

	// Chat 模板（string->string）：
	// messages: { {"system"|"user"|"assistant", content}, ... }
	// add_generation_prompt=true: 若最后一条为 user，会附加 "<|im_start|>assistant\n" 以便后续生成。
	std::string apply_chat_template(
		const std::vector<std::pair<std::string, std::string>>& messages,
		bool add_generation_prompt = true
	) const;

	// 仅用户单条输入的便捷接口（自动加默认 system）
	std::string apply_chat_template_from_user(
		const std::string& user_prompt,
		bool add_generation_prompt = true
	) const;

	size_t vocab_size() const { return vocab_size_; }

	std::optional<int64_t> id_bos() const { return id_bos_; }
	std::optional<int64_t> id_eos() const { return id_eos_; }
	std::optional<int64_t> id_pad() const { return id_pad_; }
	std::optional<int64_t> id_unk() const { return id_unk_; }

private:
	// BPE 相关结构
	size_t vocab_size_ = 0;
	std::unordered_map<std::string, int64_t> vocab_;
	std::vector<std::string> inv_vocab_;
	std::unordered_map<std::pair<std::string,std::string>, int, PairHash> bpe_ranks_{}; // pair->rank

	// 配置
	bool add_prefix_space_ = false;
	bool trim_offsets_ = true;

	// ByteLevel 映射表（bytes 0..255 <-> unicode 可见字符）
	std::vector<std::string> byte_to_unicode_; // size 256, each a UTF-8 string of mapped char
	std::unordered_map<std::string, unsigned int> unicode_to_byte_; // inverse

	std::string default_system_message_ = "You are a helpful assistant";

	// 特殊 token
	std::optional<int64_t> id_bos_;
	std::optional<int64_t> id_eos_;
	std::optional<int64_t> id_pad_;
	std::optional<int64_t> id_unk_;
	
	// 特殊令牌集合，用于在预分词前保护
	std::unordered_set<std::string> special_tokens_;
};

