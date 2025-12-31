#include "tokenlizer.hpp"
#include "json.hpp"

#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cctype>
#include <algorithm>
#include <climits>
#include <regex>

using nlohmann::json;

namespace {
// UTF-8 encode a single codepoint
static std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// UTF-8 decode next codepoint from a string at index i
static uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i++];
    if (c < 0x80) {
        return c;
    }
    if ((c >> 5) == 0x6) {
        uint32_t cp = (c & 0x1F) << 6;
        cp |= ((unsigned char)s[i++] & 0x3F);
        return cp;
    }
    if ((c >> 4) == 0xE) {
        uint32_t cp = (c & 0x0F) << 12;
        cp |= ((unsigned char)s[i++] & 0x3F) << 6;
        cp |= ((unsigned char)s[i++] & 0x3F);
        return cp;
    }
    // 4 bytes
    uint32_t cp = (c & 0x07) << 18;
    cp |= ((unsigned char)s[i++] & 0x3F) << 12;
    cp |= ((unsigned char)s[i++] & 0x3F) << 6;
    cp |= ((unsigned char)s[i++] & 0x3F);
    return cp;
}
}

static bool load_json(const std::string& path, json& j, std::string* err) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { if (err) *err = "Failed to open: " + path; return false; }
    try { ifs >> j; return true; } catch (const std::exception& e) { if (err) *err = std::string("JSON parse error for ") + path + ": " + e.what(); return false; }
}

// byte-level pre-tokenization: convert bytes to unicode space-preserving form
static void build_bytes_unicode_maps(std::vector<std::string>& b2u, std::unordered_map<std::string, unsigned int>& u2b) {
    // GPT-2 style mapping
    std::vector<int> bs;
    for (int i=33;i<=126;++i) bs.push_back(i);
    for (int i=161;i<=172;++i) bs.push_back(i);
    for (int i=174;i<=255;++i) bs.push_back(i);
    std::vector<int> cs = bs;
    int n=0;
    for (int b=0;b<256;++b) {
        if (std::find(bs.begin(), bs.end(), b)==bs.end()) { bs.push_back(b); cs.push_back(256+n); ++n; }
    }
    b2u.resize(256);
    for (size_t i=0;i<bs.size();++i) {
        unsigned int b = (unsigned int)bs[i];
        unsigned int c = (unsigned int)cs[i];
        std::string u = utf8_encode(c);
        b2u[b] = u;
        u2b[u] = b;
    }
}

bool MiniTokenizer::load(const Config& cfg, std::string* err) {
    vocab_.clear(); inv_vocab_.clear(); vocab_size_ = 0;
    bpe_ranks_.clear();
    id_bos_.reset(); id_eos_.reset(); id_pad_.reset(); id_unk_.reset();

    json jtok, jcfg;
    if (!load_json(cfg.model_dir + "/tokenizer.json", jtok, err)) return false;
    load_json(cfg.model_dir + "/tokenizer_config.json", jcfg, err);

    // config flags
    if (jtok.contains("pre_tokenizer") && jtok["pre_tokenizer"].contains("add_prefix_space"))
        add_prefix_space_ = jtok["pre_tokenizer"]["add_prefix_space"].get<bool>();
    if (jtok.contains("pre_tokenizer") && jtok["pre_tokenizer"].contains("trim_offsets"))
        trim_offsets_ = jtok["pre_tokenizer"]["trim_offsets"].get<bool>();

    // build byte-unicode maps
    build_bytes_unicode_maps(byte_to_unicode_, unicode_to_byte_);

    // vocab
    if (jtok.contains("model") && jtok["model"].contains("vocab")) {
        const auto& v = jtok["model"]["vocab"];
        if (v.is_object()) {
            for (auto it = v.begin(); it != v.end(); ++it) {
                vocab_[it.key()] = it.value().get<int64_t>();
            }
        }
    }
    if (jtok.contains("added_tokens") && jtok["added_tokens"].is_array()) {
        for (const auto& t : jtok["added_tokens"]) {
            if (t.contains("content") && t.contains("id")) {
                vocab_[t["content"].get<std::string>()] = t["id"].get<int64_t>();
            }
        }
    }

    int64_t max_id = -1; for (auto& kv : vocab_) max_id = std::max(max_id, kv.second);
    if (max_id >= 0) { inv_vocab_.assign(size_t(max_id + 1), ""); for (auto& kv : vocab_) if (kv.second >= 0 && size_t(kv.second) < inv_vocab_.size()) inv_vocab_[size_t(kv.second)] = kv.first; vocab_size_ = inv_vocab_.size(); }

    // merges -> ranks
    if (jtok.contains("model") && jtok["model"].contains("merges")) {
        const auto& merges = jtok["model"]["merges"];
        if (merges.is_array()) {
            int rank = 0;
            bpe_ranks_.reserve(merges.size());
            for (const auto& m : merges) {
                std::string a;
                std::string b;

                // Some tokenizer.json store merges as "tokenA tokenB" strings,
                // others as ["tokenA", "tokenB"] arrays. Support both.
                if (m.is_string()) {
                    const std::string s = m.get<std::string>();
                    const auto pos = s.find(' ');
                    if (pos == std::string::npos) {
                        continue;
                    }
                    a = s.substr(0, pos);
                    b = s.substr(pos + 1);
                } else if (m.is_array() && m.size() == 2 && m[0].is_string() && m[1].is_string()) {
                    a = m[0].get<std::string>();
                    b = m[1].get<std::string>();
                } else {
                    // Unknown format; skip
                    continue;
                }

                bpe_ranks_.emplace(std::make_pair(a, b), rank++);
            }
        }
    }

    // Parse special tokens from tokenizer.json
    special_tokens_.clear();
    if (jtok.contains("added_tokens")) {
        const auto& added_tokens = jtok["added_tokens"];
        if (added_tokens.is_array()) {
            for (const auto& token : added_tokens) {
                if (token.contains("content") && token["content"].is_string()) {
                    std::string content = token["content"].get<std::string>();
                    auto it = vocab_.find(content);
                    if (it != vocab_.end()) {
                        special_tokens_.insert(content);
                    }
                }
            }
        }
    }

    auto find_id = [&](const char* key) -> std::optional<int64_t> {
        if (jcfg.contains(key)) {
            std::string tok = jcfg[key].get<std::string>();
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) return it->second;
        }
        return std::nullopt;
    };
    id_bos_ = find_id("bos_token");
    id_eos_ = find_id("eos_token");
    id_pad_ = find_id("pad_token");
    id_unk_ = find_id("unk_token");

    return true;
}

// BPE apply for a single pre-token
static std::vector<std::string> bpe_tokenize_units(const std::vector<std::string>& initial_units,
    const std::unordered_map<std::pair<std::string,std::string>, int, MiniTokenizer::PairHash>& ranks) {
    std::vector<std::string> word = initial_units;
    
    // Handle empty input
    if (word.empty()) return word;
    
    // If only one unit, no merging needed
    if (word.size() == 1) return word;

    auto get_pair = [&](const std::vector<std::string>& w) {
        std::pair<std::string,std::string> best;
        int best_rank = INT_MAX; 
        bool has = false;
        
        for (size_t i = 0; i + 1 < w.size(); ++i) {
            auto it = ranks.find({w[i], w[i+1]});
            if (it != ranks.end() && it->second < best_rank) { 
                best_rank = it->second; 
                best = it->first; 
                has = true; 
            }
        }
        return std::make_pair(has, best);
    };

    while (true) {
        auto pr = get_pair(word);
        if (!pr.first) break;  // No more pairs to merge
        
        const auto& pair = pr.second;
        std::vector<std::string> new_word; 
        new_word.reserve(word.size());
        
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == pair.first && word[i+1] == pair.second) {
                new_word.push_back(word[i] + word[i+1]); 
                i += 2;
            } else { 
                new_word.push_back(word[i]); 
                ++i; 
            }
        }
        word.swap(new_word);
        
        // Early termination if we can't merge further
        if (word.size() == 1) break;
    }
    return word;
}

// ByteLevel pre-tokenization that works purely on bytes
// This avoids all UTF-8 issues by treating input as byte sequence
// Simple pre-tokenization that handles common patterns
// This avoids complex regex to prevent UTF-8 issues
static std::vector<std::string> simple_pre_tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    if (text.empty()) return tokens;
    
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        
        // Handle contractions first
        if (i + 1 < text.size() && text[i] == '\'') {
            // Check for common contractions
            std::string rest = text.substr(i + 1);
            if ((rest.size() >= 2 && (rest.substr(0, 2) == "s " || rest.substr(0, 2) == "t " || 
                rest.substr(0, 2) == "m " || rest.substr(0, 2) == "d ")) ||
                rest == "s" || rest == "t" || rest == "m" || rest == "d") {
                tokens.push_back(text.substr(i, 2));
                i += 2;
                continue;
            } else if ((rest.size() >= 3 && (rest.substr(0, 3) == "re " || rest.substr(0, 3) == "ve " || 
                       rest.substr(0, 3) == "ll ")) || rest == "re" || rest == "ve" || rest == "ll") {
                tokens.push_back(text.substr(i, 3));
                i += 3;
                continue;
            }
        }
        
        // Handle sequences of letters (with optional leading space)
        if (i < text.size() && text[i] == ' ') {
            // Space followed by letters
            size_t j = i + 1;
            while (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            if (j > i + 1) {
                tokens.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }
        
        // Handle sequences of letters (without leading space)
        if (i < text.size() && std::isalpha(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && std::isalpha(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        
        // Handle sequences of digits (with optional leading space)
        if (i < text.size() && text[i] == ' ') {
            size_t j = i + 1;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            if (j > i + 1) {
                tokens.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }
        
        // Handle sequences of digits (without leading space)
        if (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        
        // Handle sequences of non-alphanumeric characters (with optional leading space)
        if (i < text.size() && text[i] == ' ') {
            size_t j = i + 1;
            while (j < text.size() && !std::isspace(static_cast<unsigned char>(text[j])) && 
                   !std::isalnum(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            if (j > i + 1) {
                tokens.push_back(text.substr(i, j - i));
                i = j;
                continue;
            }
        }
        
        // Handle sequences of non-alphanumeric characters (without leading space)
        if (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i])) && 
            !std::isalnum(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && !std::isspace(static_cast<unsigned char>(text[j])) && 
                   !std::isalnum(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        
        // Handle whitespace
        if (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j]))) {
                ++j;
            }
            tokens.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }
        
        // Fallback: single character
        tokens.push_back(text.substr(i, 1));
        ++i;
    }
    
    return tokens;
}

std::vector<int64_t> MiniTokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};
    
    std::vector<int64_t> ids;
    std::string input = text;
    
    // Add prefix space if required
    if (add_prefix_space_ && !input.empty() && input.front() != ' ') {
        input = " " + input;
    }

    // Step 1: Split text by special tokens, keeping the special tokens
    std::vector<std::pair<std::string, bool>> segments; // (text, is_special_token)
    
    size_t pos = 0;
    while (pos < input.size()) {
        // Find the earliest special token
        size_t best_start = std::string::npos;
        size_t best_end = 0;
        std::string best_special;
        
        for (const auto& special : special_tokens_) {
            size_t found = input.find(special, pos);
            if (found != std::string::npos && (best_start == std::string::npos || found < best_start)) {
                best_start = found;
                best_end = found + special.size();
                best_special = special;
            }
        }
        
        if (best_start != std::string::npos) {
            // Add text before special token (if any)
            if (best_start > pos) {
                segments.emplace_back(input.substr(pos, best_start - pos), false);
            }
            // Add special token
            segments.emplace_back(best_special, true);
            pos = best_end;
        } else {
            // No more special tokens, add remaining text
            segments.emplace_back(input.substr(pos), false);
            break;
        }
    }
    
    // Step 2: Process each segment
    for (const auto& segment : segments) {
        const std::string& text_part = segment.first;
        bool is_special = segment.second;
        
        if (is_special) {
            // Special token: look up directly in vocabulary
            auto it = vocab_.find(text_part);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else if (id_unk_) {
                ids.push_back(*id_unk_);
            }
        } else {
            // Regular text: apply ByteLevel pre-tokenization + BPE
            if (text_part.empty()) continue;
            
            // Simple pre-tokenization to avoid UTF-8 regex issues
            auto pre_tokens = simple_pre_tokenize(text_part);
            
            // Process each pre-token with ByteLevel encoding + BPE
            for (const auto& pre_token : pre_tokens) {
                if (pre_token.empty()) continue;
                
                // Convert each byte to its ByteLevel unicode representation
                std::vector<std::string> units;
                units.reserve(pre_token.size());
                
                for (unsigned char byte : pre_token) {
                    units.push_back(byte_to_unicode_[byte]);
                }
                
                // Apply BPE merging
                auto bpe_tokens = bpe_tokenize_units(units, bpe_ranks_);
                
                // Convert to vocabulary IDs
                for (const auto& token : bpe_tokens) {
                    auto it = vocab_.find(token);
                    if (it != vocab_.end()) {
                        ids.push_back(it->second);
                    } else if (id_unk_) {
                        ids.push_back(*id_unk_);
                    }
                }
            }
        }
    }
    
    return ids;
}

std::string MiniTokenizer::decode(const std::vector<int64_t>& ids) const {
    // Concatenate token strings then map unicode units back to bytes
    std::string inter;
    for (auto id: ids) if (id>=0 && (size_t)id<inv_vocab_.size()) inter += inv_vocab_[(size_t)id];
    std::string out; out.reserve(inter.size());
    size_t idx=0; while (idx < inter.size()) {
        size_t start = idx;
        uint32_t cp = utf8_next(inter, idx);
        std::string u = inter.substr(start, idx-start);
        auto it = unicode_to_byte_.find(u);
        if (it != unicode_to_byte_.end()) out.push_back(static_cast<char>(it->second));
        else out += u; // unknown direct append
    }
    return out;
}

// Chat template: align with tokenizer_config.json chat_template when possible.
// 参考你提供的模板：
// "{%- if messages[0]['role'] == 'system' -%}\n"
// "<|im_start|>system\n{{ system_message }}<|im_end|>\n" 等
std::string MiniTokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) const {
    std::string out;
    // 开头 system：若首条不是 system，注入默认 system
    if (!messages.empty() && messages.front().first == "system") {
        out += "<|im_start|>system\n";
        out += messages.front().second;
        out += "<|im_end|>\n";
    } else {
        out += "<|im_start|>system\n";
        out += default_system_message_;
        out += "<|im_end|>\n";
    }

    // 遍历消息
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& role = messages[i].first;
        const auto& content = messages[i].second;
        if (role == "user") {
            out += "<|im_start|>user\n";
            out += content;
            out += "<|im_end|>\n";
            if (add_generation_prompt && i == messages.size() - 1) {
                out += "<|im_start|>assistant\n"; // 生成从 assistant 段落开始
            }
        } else if (role == "assistant") {
            out += content;
            out += "<|im_end|>\n";
        } else if (role == "system" && i != 0) {
            // 额外 system 段落
            out += "<|im_start|>system\n";
            out += content;
            out += "<|im_end|>\n";
        }
    }
    return out;
}

std::string MiniTokenizer::apply_chat_template_from_user(
    const std::string& user_prompt,
    bool add_generation_prompt) const {
    std::vector<std::pair<std::string,std::string>> msgs;
    msgs.emplace_back("user", user_prompt);
    return apply_chat_template(msgs, add_generation_prompt);
}
