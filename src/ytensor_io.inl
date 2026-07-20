/***************
 * file: ytensor_io.inl
 * purpose: YTensor二进制格式、压缩和事务式文件提交实现。
 ***************/

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <mutex>
#include "../include/ytensor_io.hpp"
#include <zlib.h>

namespace yt::io {

// 文件布局（整数均为当前平台native endian）：
// [8-byte magic][uint8 version][uint32 metadata bytes][metadata]
// [tensor records...][uint64 record offsets...][uint32 tensor count]
// 每个record依次保存name、dtype、typeSize、tensorType、shape、compression、payload size和payload。

// 在目标文件同目录生成事务临时路径，以保证最终rename位于同一filesystem。
// 注意：时间戳和进程内sequence只降低碰撞概率，不提供跨进程全局唯一性。
inline std::filesystem::path makeCommitTemporaryPath(const std::filesystem::path& target) {
    static std::atomic<uint64_t> sequence{0};
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path temporary = target;
    temporary += ".tmp." + std::to_string(stamp) + "." +
                 std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    return temporary;
}

// 将已flush/close的完整临时文件提交为target，并在平台不支持replace时执行backup/restore。
// 注意：static mutex串行化进程内提交；返回false时会尽力恢复旧target，但无法报告restore自身失败。
inline bool commitPreparedFile(
    const std::filesystem::path& temporary, const std::filesystem::path& target
) {
    static std::mutex commitMutex;
    std::lock_guard<std::mutex> lock(commitMutex);

    std::error_code ec;
    std::filesystem::rename(temporary, target, ec);
    if (!ec) return true;

    ec.clear();
    if (!std::filesystem::is_regular_file(target, ec) || ec) return false;

    std::filesystem::path backup = makeCommitTemporaryPath(target);
    backup += ".backup";
    // 某些平台不能直接replace现有文件：先备份旧commit，提升temporary失败时再恢复。
    std::filesystem::rename(target, backup, ec);
    if (ec) return false;

    ec.clear();
    std::filesystem::rename(temporary, target, ec);
    if (!ec) {
        std::error_code cleanupError;
        std::filesystem::remove(backup, cleanupError);
        return true;
    }

    std::error_code restoreError;
    std::filesystem::rename(backup, target, restoreError);
    return false;
}

// ==================== primitive encoding and compression ====================

YT_IMPL_INLINE std::string checkCompressMethod(const std::string& method) {
    if (method == "zlib" ||
        method.empty()
    ) {
        return method;
    } else {
        if (yt::io::verbose) {
            std::cerr << "Warning: Unknown compression method '" << method << "'. Falling back to no compression." << std::endl;
        }
        return ""; // 回退到不压缩
    }
}

YT_IMPL_INLINE std::vector<char> string2data(const std::string& str) {
    if (str.length() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("string2data: string exceeds uint32 format limit");
    }
    uint32_t length = static_cast<uint32_t>(str.length());
    std::vector<char> op(sizeof(uint32_t) + length);
    std::memcpy(op.data(), &length, sizeof(uint32_t));
    if (!str.empty()) {
        std::memcpy(op.data() + sizeof(uint32_t), str.c_str(), length);
    }
    return op;
}

// 读取uint32长度前缀字符串；seek=true消费输入，false在成功后恢复原位置。
// 注意：长度字段是文件信任边界，读取前先验证剩余字节和streamsize范围。
YT_IMPL_INLINE std::string data2string(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint32_t length;
    if (!file.read(reinterpret_cast<char*>(&length), sizeof(uint32_t))) {
        throw std::runtime_error("Failed to read string length");
    }
    std::streampos dataPos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos endPos = file.tellg();
    if (dataPos < 0 || endPos < dataPos || static_cast<uint64_t>(endPos - dataPos) < length) {
        file.seekg(originalPos);
        throw std::runtime_error("String length exceeds remaining file data");
    }
    file.seekg(dataPos);
    std::string op;
    if (length > 0) {
        if (static_cast<uint64_t>(length) >
            static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            file.seekg(originalPos);
            throw std::runtime_error("String length exceeds streamsize range");
        }
        op.resize(length);
        if (!file.read(op.data(), static_cast<std::streamsize>(length))) {
            throw std::runtime_error("Failed to read string data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }    
    return op;
}

// 读取uint64字节数前缀数组；返回raw bytes而不是解释后的元素数组。
// 注意：seek参数与data2string一致，false用于窥探且恢复原位置。
YT_IMPL_INLINE std::vector<char> data2array(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint64_t count;
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t))) {
        throw std::runtime_error("Failed to read array count");
    }
    std::streampos dataPos = file.tellg();
    file.seekg(0, std::ios::end);
    std::streampos endPos = file.tellg();
    if (dataPos < 0 || endPos < dataPos || static_cast<uint64_t>(endPos - dataPos) < count ||
        count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        file.seekg(originalPos);
        throw std::runtime_error("Array length exceeds remaining file data");
    }
    file.seekg(dataPos);
    std::vector<char> op;
    if (count > 0) {
        if (count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            file.seekg(originalPos);
            throw std::runtime_error("Array length exceeds streamsize range");
        }
        op.resize(count);
        if (!file.read(op.data(), static_cast<std::streamsize>(count))) {
            throw std::runtime_error("Failed to read array data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }
    return op;
}

// 解压内存payload；decompressedSize仅是受限的首块分配hint，不作为可信结果长度。
// 返回：解压失败和合法空payload都返回空vector，调用方需结合声明大小区分。
YT_IMPL_INLINE std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize, const std::string& method) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
        if (input.size() > static_cast<size_t>(std::numeric_limits<uInt>::max())) {
            return {};  // vector入口不能把超过zlib单次avail_in范围的长度静默截断
        }
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        if (inflateInit2(&stream, 15 + 32) != Z_OK) {
            return {}; // 初始化失败
        }
        stream.avail_in = input.size();
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        std::vector<std::vector<char>> chunks;
        const size_t chunk_size = 65536; // 64KB chunks
        // 第一个chunk：如果已知解压大小，使用该大小；否则使用默认chunk大小
        // 声明长度只作为hint，不能让损坏metadata在inflate前触发任意大分配。
        size_t first_chunk_size = decompressedSize > 0 ? std::min(decompressedSize, chunk_size) : chunk_size;
        chunks.emplace_back(first_chunk_size);
        std::vector<char>& current_chunk = chunks.back();
        stream.avail_out = first_chunk_size;
        stream.next_out = reinterpret_cast<Bytef*>(current_chunk.data());
        int result = inflate(&stream, Z_NO_FLUSH);
        if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
            inflateEnd(&stream);
            if (yt::io::verbose) {
                std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
            }
            return {};
        }
        size_t chunk_used = first_chunk_size - stream.avail_out;
        current_chunk.resize(chunk_used);
        // 如果还有剩余数据需要解压，继续处理
        while (result != Z_STREAM_END && stream.avail_out == 0) {
            chunks.emplace_back(chunk_size);
            std::vector<char>& next_chunk = chunks.back();

            stream.avail_out = chunk_size;
            stream.next_out = reinterpret_cast<Bytef*>(next_chunk.data());

            result = inflate(&stream, Z_NO_FLUSH);

            if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
                inflateEnd(&stream);
                if (yt::io::verbose) {
                    std::cerr << "Warning: Failed to decompress data. Returning empty vector." << std::endl;
                }
                return {};
            }

            chunk_used = chunk_size - stream.avail_out;
            next_chunk.resize(chunk_used);
        }
        inflateEnd(&stream);
        if (result != Z_STREAM_END) {
            return {};
        }
        // 合并所有chunks
        size_t total_size = 0;
        for (const auto& chunk : chunks) {
            total_size += chunk.size();
        }
        std::vector<char> decompressed;
        decompressed.reserve(total_size);
        for (auto& chunk : chunks) {
            decompressed.insert(decompressed.end(), 
                               std::make_move_iterator(chunk.begin()), 
                               std::make_move_iterator(chunk.end()));
        }
        return decompressed;
    } else {
        // 缺省值/fallback，不压缩
        return input;
    }
}

// 从当前stream位置分块读取并解压payload，避免把不可信声明长度直接作为大分配。
// 注意：seek=false时成功或失败均尽力恢复原位置；zlib必须到达Z_STREAM_END才算完整。
YT_IMPL_INLINE std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize, bool seek, const std::string& method) {
    if (compressedSize == 0) {
        return {};
    }

    std::streampos originalPos = file.tellg();
    std::vector<char> output;
    const size_t io_chunk = 65536;

    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        
        if (inflateInit2(&stream, 15 + 32) != Z_OK) {
            return {};
        }

        std::vector<char> input_buffer(io_chunk);
        size_t remaining = compressedSize;

        // 按流逐块增长；decompressedSize可能来自不可信metadata，也可能低估map变长payload。
        output.resize(decompressedSize > 0 ? std::min(decompressedSize, io_chunk) : io_chunk);

        size_t out_pos = 0;

        stream.avail_out = static_cast<uInt>(output.size() - out_pos);
        stream.next_out = reinterpret_cast<Bytef*>(output.data() + out_pos);

        int result = Z_OK;
        while (result != Z_STREAM_END) {
            if (stream.avail_in == 0 && remaining > 0) {
                size_t to_read = std::min(remaining, io_chunk);
                if (!file.read(input_buffer.data(), to_read)) {
                    inflateEnd(&stream);
                    if (!seek) file.seekg(originalPos);
                    if (yt::io::verbose) {
                        std::cerr << "Error: Failed to read compressed data from file" << std::endl;
                    }
                    return {};
                }
                stream.next_in = reinterpret_cast<Bytef*>(input_buffer.data());
                stream.avail_in = static_cast<uInt>(to_read);
                remaining -= to_read;
            }

            if (stream.avail_out == 0) {
                size_t add = io_chunk;
                if (decompressedSize > 0) {
                    if (decompressedSize > out_pos) {
                        size_t need = decompressedSize - out_pos;
                        add = std::min(add, need);
                    } else {
                        add = io_chunk;
                    }
                }
                size_t old_size = output.size();
                output.resize(old_size + add);
                stream.next_out = reinterpret_cast<Bytef*>(output.data() + out_pos);
                stream.avail_out = static_cast<uInt>(output.size() - out_pos);
            }

            result = inflate(&stream, Z_NO_FLUSH);
            if (result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR) {
                inflateEnd(&stream);
                if (!seek) file.seekg(originalPos);
                if (yt::io::verbose) {
                    std::cerr << "Error: Failed to decompress data from file" << std::endl;
                }
                return {};
            }

            size_t wrote = (output.size() - out_pos) - stream.avail_out;
            out_pos += wrote;

            if (result == Z_STREAM_END) break;

            if (stream.avail_in == 0 && remaining == 0 && stream.avail_out > 0 && result == Z_BUF_ERROR) {
                break;
            }
        }

        const bool streamComplete = result == Z_STREAM_END;
        inflateEnd(&stream);
        if (!streamComplete) {
            if (!seek) file.seekg(originalPos);
            if (yt::io::verbose) {
                std::cerr << "Error: Compressed stream ended before Z_STREAM_END" << std::endl;
            }
            return {};
        }
        output.resize(out_pos);

    } else {
        output.resize(compressedSize);
        size_t position = 0;
        while (position < compressedSize) {
            size_t chunk = std::min(compressedSize - position, io_chunk);
            if (!file.read(output.data() + position, static_cast<std::streamsize>(chunk))) {
                file.seekg(originalPos);
                if (yt::io::verbose) {
                    std::cerr << "Error: Failed to read compressed data from file" << std::endl;
                }
                return {};
            }
            position += chunk;
        }
    }
    
    if (!seek) {
        file.seekg(originalPos);
    }
    return output;
}

YT_IMPL_INLINE YTensorIO::~YTensorIO() noexcept {
    // 析构函数无法报告transaction commit失败；持久化调用方必须显式检查close()。
    try {
        close();
    } catch (...) {
    }
}

// ==================== file transaction lifecycle ====================

// 打开read或write transaction；开始新会话前先提交/关闭当前会话。
// Append先解析现有index以便close时完整重写；损坏的非空文件绝不被当作空文件覆盖。
YT_IMPL_INLINE bool YTensorIO::open(const std::string& fileName, int fileMode) {
    try {
    if (!close()) return false;
    _fileName = fileName;
    _fileMode = fileMode;
    _writeFailed = false;
    if (fileMode == yt::io::Write || fileMode == yt::io::Append) {
        // 写入模式：先尝试读取现有文件的张量信息
        std::ifstream checkFile(fileName, std::ios::binary);
        bool fileExists = checkFile.good();
        checkFile.close();
        
        if (fileExists) {
            // 如果文件存在，先读取现有的张量信息和数据到内存
            _file.open(fileName, std::ios::binary | std::ios::in);
            if (_file.is_open() && readHeader() && readIndex()) {
                _file.close();
            } else {
                _file.close();
                _tensorInfos.clear();
                // Append不能把已有但不可解析的文件当作空文件，否则close会静默截断原内容。
                if (fileMode == yt::io::Append) {
                    std::error_code ec;
                    auto existingSize = std::filesystem::file_size(fileName, ec);
                    if (ec || existingSize > 0) {
                        _fileMode = yt::io::Closed;
                        return false;
                    }
                }
            }
        } else {
            // 文件在完整临时文件提交前保持不存在。
            _tensorInfos.clear();
            return true;
        }
        
        // 在关闭文件之前，都以只读模式打开
        _file.open(fileName, std::ios::binary | std::ios::in);
        if (!_file.is_open()) {
            _fileMode = yt::io::Closed;
            return false;
        }
        return true;
    } else {
        // 读取模式
        _file.open(fileName, std::ios::binary | std::ios::in);
        if (!_file.is_open()) {
            if (verbose) {
                std::cerr << "Error: Failed to open file for reading: " << fileName << std::endl;
            }
            return false;
        }
        // 读取并验证文件
        if (!readHeader()) {
            if (verbose) {
                std::cerr << "Error: Failed to read file header" << std::endl;
            }
            close();
            return false;
        }
        if (!readIndex()) {
            if (verbose) {
                std::cerr << "Error: Failed to read file index" << std::endl;
            }
            close();
            return false;
        }
        return true;
    }
    } catch (...) {
        if (_fileMode == yt::io::Write || _fileMode == yt::io::Append) _writeFailed = true;
        close();
        return false;
    }
}

// write transaction的唯一commit点：组装完整临时文件、flush后原子提升，再清空对象状态。
// 返回：staging、重写或rename任一步失败均返回false；析构调用无法替代显式检查。
YT_IMPL_INLINE bool YTensorIO::close() {
    bool closeSucceeded = true;
    try {
    if (_file.is_open() || _fileMode == yt::io::Write || _fileMode == yt::io::Append) {
        if (_fileMode == yt::io::Write || _fileMode == yt::io::Append) {
            bool rewriteReady = !_writeFailed;
            // payload有三种状态：staged内存数据、磁盘dataOffset引用、已staged的合法空payload。
            // close先把最终record集合全部变成staged状态，随后才开始写临时文件。
            if (rewriteReady && _fileMode == yt::io::Append) {
                // 如果是附加模式，先加载所有的张量数据进内存。无需解压。
                for (size_t i = 0; i < _tensorInfos.size(); ++i) {
                    auto& info = _tensorInfos[i];
                    if(info.payloadStaged) {
                        // 已经在内存中，无需重复加载
                        info.compressedSize = info.compressedData.size();
                        continue;
                    }
                    _file.seekg(info.dataOffset, std::ios::beg);

                    // 读取压缩数据到内存
                    info.compressedData.resize(info.compressedSize);
                    if (info.compressedSize > 0) {
                        if (info.compressedSize >
                                static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
                            !_file.read(
                                info.compressedData.data(), static_cast<std::streamsize>(info.compressedSize)
                            )) {
                            if (verbose) {
                                std::cerr << "Error: Failed to read compressed data for tensor '" 
                                          << info.name << "'" << std::endl;
                            }
                            info.compressedData.clear();
                            rewriteReady = false;
                            closeSucceeded = false;
                            break;
                        }
                    }
                }
            } else if (rewriteReady) {
                // 如果是写入模式，去除所有不在内存中的张量
                std::erase_if(_tensorInfos, [](const TensorInfo& info) {
                    return !info.payloadStaged;
                });
            }

            _file.close();
            if (rewriteReady) {
                // target在临时文件完整写入并flush前保持不变。
                std::filesystem::path targetPath(_fileName);
                std::filesystem::path temporaryPath = makeCommitTemporaryPath(targetPath);
                std::error_code ec;
                std::filesystem::remove(temporaryPath, ec);
                bool writeOk = true;
                try {
                    _file.open(temporaryPath, std::ios::binary | std::ios::out | std::ios::trunc);
                    writeOk = _file.is_open() && writeHeader();
                    std::vector<uint64_t> offsets;
                    offsets.reserve(_tensorInfos.size());
                    for (const auto& info : _tensorInfos) {
                        if (!writeOk) break;
                        std::streampos position = _file.tellp();
                        if (position < 0) {
                            writeOk = false;
                            break;
                        }
                        offsets.push_back(static_cast<uint64_t>(position));
                        auto nameData = string2data(info.name);
                        auto typeNameData = string2data(info.typeName);
                        auto tensorTypeData = string2data(info.tensorType);
                        auto shapeData = array2data(info.shape);
                        auto compressMethodData = string2data(info.compressMethod);
                        const size_t maxWrite =
                            static_cast<size_t>(std::numeric_limits<std::streamsize>::max());
                        if (nameData.size() > maxWrite || typeNameData.size() > maxWrite ||
                            tensorTypeData.size() > maxWrite || shapeData.size() > maxWrite ||
                            compressMethodData.size() > maxWrite) {
                            writeOk = false;
                            break;
                        }
                        writeOk = static_cast<bool>(
                            _file.write(nameData.data(), static_cast<std::streamsize>(nameData.size())) &&
                            _file.write(typeNameData.data(), static_cast<std::streamsize>(typeNameData.size())) &&
                            _file.write(reinterpret_cast<const char*>(&info.typeSize), sizeof(info.typeSize)) &&
                            _file.write(tensorTypeData.data(), static_cast<std::streamsize>(tensorTypeData.size())) &&
                            _file.write(shapeData.data(), static_cast<std::streamsize>(shapeData.size())) &&
                            _file.write(compressMethodData.data(), static_cast<std::streamsize>(compressMethodData.size())) &&
                            _file.write(reinterpret_cast<const char*>(&info.compressedSize), sizeof(info.compressedSize))
                        );
                        if (writeOk && !info.compressedData.empty()) {
                            if (info.compressedData.size() >
                                static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
                                writeOk = false;
                            } else {
                                writeOk = static_cast<bool>(_file.write(
                                    info.compressedData.data(),
                                    static_cast<std::streamsize>(info.compressedData.size())
                                ));
                            }
                        }
                    }
                    if (writeOk) writeOk = writeIndex(offsets);
                    if (writeOk) {
                        _file.flush();
                        writeOk = static_cast<bool>(_file);
                    }
                } catch (...) {
                    writeOk = false;
                }
                if (_file.is_open()) _file.close();
                if (writeOk) {
                    writeOk = commitPreparedFile(temporaryPath, targetPath);
                }
                if (!writeOk) {
                    std::filesystem::remove(temporaryPath, ec);
                    if (verbose) std::cerr << "Error: Failed to commit YTensor file transaction" << std::endl;
                }
                closeSucceeded = writeOk;
            }
        }
        if (_file.is_open()) _file.close();
    }
    } catch (...) {
        closeSucceeded = false;
    }
    try {
        if (_file.is_open()) _file.close();
    } catch (...) {
        closeSucceeded = false;
    }
    if (_writeFailed) closeSucceeded = false;
    _fileMode = yt::io::Closed;
    _writeFailed = false;
    _isHeaderRead = false;
    _version = 0;
    _tensorInfos.clear();
    _metadata.clear();
    _fileName.clear();
    return closeSucceeded;
}

YT_IMPL_INLINE std::vector<std::string> YTensorIO::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& tensorInfo : _tensorInfos) {
        names.push_back(tensorInfo.name);
    }
    return names;
}

YT_IMPL_INLINE TensorInfo YTensorIO::getTensorInfo(const std::string& name) const {
    if (name.empty()) {
        // 缺省取第一个张量
        if (_tensorInfos.empty()) {
            throw std::runtime_error("No tensors available");
        }
        return _tensorInfos[0];
    }
    
    // 按名称查找
    for (const auto& tensorInfo : _tensorInfos) {
        if (tensorInfo.name == name) {
            return tensorInfo;
        }
    }
    throw std::runtime_error("Tensor not found: " + name);
}

// ==================== header and footer index ====================

// 仅检查当前stream开头的8-byte magic，不执行version、record或payload完整性校验。
YT_IMPL_INLINE bool YTensorIO::validateFile() {
    try {
    if (!_file.is_open() || !_fileMode) {
        return false;
    }
    _file.seekg(0, std::ios::beg);
    // Check magic
    std::string magic(8, '\0');
    if (!_file.read(magic.data(), magic.size())) {
        return false;
    }
    return magic == yt::io::YTENSOR_FILE_MAGIC;
    } catch (...) {
        return false;
    }
}

// 读取magic、version和长度前缀metadata；成功后stream位于首个tensor record。
YT_IMPL_INLINE bool YTensorIO::readHeader() {
    if (!_file.is_open()) {
        return false;
    }
    _file.seekg(0, std::ios::beg);
    // Read magic
    std::string magic(8, '\0');
    if (!_file.read(magic.data(), magic.size()) || magic != yt::io::YTENSOR_FILE_MAGIC) {
        return false;
    }
    // Read version
    if (!_file.read(reinterpret_cast<char*>(&_version), sizeof(uint8_t))) {
        return false;
    }
    if (_version > yt::io::YTENSOR_FILE_VERSION) {
        // 文件版本过高时输出警告，但不阻止读取（向下兼容）
        if (verbose) {
            std::cerr << "Warning: YTensor file version is newer than supported. "
                      << "Current version " << static_cast<int>(yt::io::YTENSOR_FILE_VERSION) 
                      << ", file version " << static_cast<int>(_version) 
                      << ". Reading may fail or produce unexpected results." << std::endl;
        }
    }
    // 读取 metadata（紧接着版本号）；长度字段损坏时保持open()的bool失败合同。
    try {
        _metadata = data2string(_file);
    } catch (...) {
        return false;
    }
    _isHeaderRead = true;
    return true;
}

// 按固定顺序写入magic、当前version和metadata，为record区建立起点。
YT_IMPL_INLINE bool YTensorIO::writeHeader() {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        return false;
    }
    _file.seekp(0, std::ios::beg);
    // magic
    if (!_file.write(yt::io::YTENSOR_FILE_MAGIC.data(), yt::io::YTENSOR_FILE_MAGIC.size())) {
        return false;
    }
    // version
    if (!_file.write(reinterpret_cast<const char*>(&yt::io::YTENSOR_FILE_VERSION), sizeof(uint8_t))) {
        return false;
    }
    // metadata
    auto metadataData = string2data(_metadata);
    if (!_file.write(metadataData.data(), metadataData.size())) {
        return false;
    }
    return true;
}

// 从footer `[uint64 offsets...][uint32 count]`读取并验证所有tensor record边界。
// 所有offset、shape乘积、typeSize和payload span都在保存到_tensorInfos前完成校验。
YT_IMPL_INLINE bool YTensorIO::readIndex() {
    if (!_file.is_open()) {
        return false;
    }
    _tensorInfos.clear();
    std::streampos recordsBegin = _file.tellg();
    _file.seekg(0, std::ios::end);
    std::streampos fileEnd = _file.tellg();
    if (recordsBegin < 0 || fileEnd < recordsBegin || fileEnd < static_cast<std::streamoff>(sizeof(uint32_t))) {
        return false;
    }
    uint64_t fileSize = static_cast<uint64_t>(fileEnd);
    _file.seekg(-static_cast<int>(sizeof(uint32_t)), std::ios::end);
    uint32_t tensorCount;
    if (!_file.read(reinterpret_cast<char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    // 读取索引
    uint64_t offsetsBytes = static_cast<uint64_t>(tensorCount) * sizeof(uint64_t);
    uint64_t indexSize64 = sizeof(uint32_t) + offsetsBytes;
    if (indexSize64 > fileSize || indexSize64 > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    const std::streamoff indexSize = static_cast<std::streamoff>(indexSize64);
    uint64_t indexStart = fileSize - indexSize64;
    if (indexStart < static_cast<uint64_t>(recordsBegin)) return false;
    _file.seekg(-indexSize, std::ios::end);

    std::vector<uint64_t> offsets(tensorCount);
    if (!_file.read(reinterpret_cast<char*>(offsets.data()), static_cast<std::streamsize>(offsetsBytes))) {
        return false;
    }
    
    // Read tensor information from each offset
    for (size_t i = 0; i < tensorCount; ++i) {
        if (offsets[i] < static_cast<uint64_t>(recordsBegin) || offsets[i] >= indexStart) return false;
        _file.seekg(offsets[i], std::ios::beg);
        TensorInfo info;
        try {
            info.name = data2string(_file);
            info.typeName = data2string(_file);
        } catch (...) {
            return false;
        }
        if (!_file.read(reinterpret_cast<char*>(&info.typeSize), sizeof(int32_t))) {
            return false;
        }
        std::vector<char> shapeData;
        try {
            info.tensorType = data2string(_file);
            shapeData = data2array(_file);
        } catch (...) {
            return false;
        }
        if (shapeData.size() % sizeof(int32_t) != 0) return false;
        size_t shapeCount = shapeData.size() / sizeof(int32_t);
        info.shape.resize(shapeCount);
        if (!shapeData.empty()) std::memcpy(info.shape.data(), shapeData.data(), shapeData.size());
        try {
            info.compressMethod = data2string(_file);
        } catch (...) {
            return false;
        }
        uint64_t compressedSize;
        if (!_file.read(reinterpret_cast<char*>(&compressedSize), sizeof(uint64_t))) {
            return false;
        }
        info.compressedSize = compressedSize;
        if (info.compressedSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
        // data offset是指向压缩数据的偏移量
        std::streampos dataOffset = _file.tellg();
        if (dataOffset < 0) return false;
        info.dataOffset = static_cast<uint64_t>(dataOffset);
        if (info.dataOffset > indexStart || info.compressedSize > indexStart - info.dataOffset) return false;
        if (info.typeSize <= 0) return false;
        info.uncompressedSize = static_cast<uint64_t>(info.typeSize);
        for (auto dim : info.shape) {
            if (dim < 0 || (dim != 0 && info.uncompressedSize > std::numeric_limits<uint64_t>::max() / dim)) {
                return false;
            }
            info.uncompressedSize *= static_cast<uint64_t>(dim);
        }
        info.compressedData.clear(); // 初始时不加载数据
        _tensorInfos.push_back(info);
    }
    return true;
}

// 将record offset表和uint32记录数写到文件尾部。
YT_IMPL_INLINE bool YTensorIO::writeIndex(std::vector<uint64_t> offsets) {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        return false;
    }
    if (offsets.size() > std::numeric_limits<uint32_t>::max() ||
        offsets.size() > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(uint64_t)) {
        return false;
    }
    size_t offsetsBytes = offsets.size() * sizeof(uint64_t);
    if (!_file.write(reinterpret_cast<const char*>(offsets.data()), static_cast<std::streamsize>(offsetsBytes))) {
        return false;
    }
    uint32_t tensorCount = static_cast<uint32_t>(offsets.size());
    if (!_file.write(reinterpret_cast<const char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    return true;
}

// ============ Save 后端 ============

// 将POD类型张量编码为未压缩的dense payload
// 参数 tensor：要保存的张量；函数内部会按逻辑顺序连续化
// 返回：未压缩payload，压缩由save统一处理
// 注意：bool使用canonical 0/1字节值；其他POD保存当前平台native object representation。
YT_IMPL_INLINE std::vector<char> YTensorIO::encDense(const yt::YTensorBase& tensor) {
    if (tensor.elementSize() != 0 && tensor.size() > std::numeric_limits<size_t>::max() / tensor.elementSize()) {
        throw std::overflow_error("encDense: payload size overflow");
    }
    size_t dataSize = tensor.size() * tensor.elementSize();
    std::vector<char> rawData(dataSize);
    auto contiguous = tensor.contiguous();
    if (tensor.dtype() == "bool") {
        const bool* values = contiguous.data<bool>();
        for (size_t i = 0; i < tensor.size(); ++i) {
            rawData[i * sizeof(bool)] = values[i] ? 1 : 0;
        }
    } else if (dataSize > 0) {
        std::memcpy(rawData.data(), contiguous.rawData(), dataSize);
    }
    return rawData;
}

// 将非POD类型张量编码为未压缩的map payload
// 参数 tensor：要保存的连续逻辑快照
// 返回：未压缩payload，压缩由save统一处理
// 格式为elemCount个uint64起始offset（无terminal offset），随后拼接变长序列化数据。
YT_IMPL_INLINE std::vector<char> YTensorIO::encMap(const yt::YTensorBase& tensor) {
    auto typeInfoOpt = yt::type::getTypeInfo(tensor.dtype());
    if (!typeInfoOpt || !typeInfoOpt->get().serialize) {
        throw std::runtime_error(
            "Non-POD type '" + tensor.dtype() + "' has no serialize function registered"
        );
    }
    
    const auto& typeInfo = typeInfoOpt->get();
    auto serialize = typeInfo.serialize;
    
    size_t elemCount = tensor.size();
    size_t elemSize = tensor.elementSize();
    const char* srcData = tensor.rawData();
    
    // 序列化所有元素，同时计算偏移量
    std::vector<uint64_t> offsets(elemCount);
    std::vector<char> serializedData;
    
    for (size_t i = 0; i < elemCount; ++i) {
        offsets[i] = static_cast<uint64_t>(serializedData.size());
        auto serialized = serialize(srcData + i * elemSize);
        serializedData.insert(serializedData.end(), serialized.begin(), serialized.end());
    }
    
    // 拼接：先是偏移量数组，再是序列化数据
    if (elemCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
        throw std::overflow_error("encMap: offset table size overflow");
    }
    size_t offsetsBytes = elemCount * sizeof(uint64_t);
    if (serializedData.size() > std::numeric_limits<size_t>::max() - offsetsBytes) {
        throw std::overflow_error("encMap: payload size overflow");
    }
    std::vector<char> rawData(offsetsBytes + serializedData.size());
    std::memcpy(rawData.data(), offsets.data(), offsetsBytes);
    std::memcpy(rawData.data() + offsetsBytes, serializedData.data(), serializedData.size());
    
    return rawData;
}

// 将磁盘shape安全收窄到runtime int metadata，并验证元素数量乘积不溢出。
inline bool decodeTensorShape(const TensorInfo& info, std::vector<int>& shape, size_t& elemCount) {
    shape.clear();
    shape.reserve(info.shape.size());
    elemCount = 1;
    for (uint64_t extent : info.shape) {
        if (extent > static_cast<uint64_t>(std::numeric_limits<int>::max())) return false;
        if (extent != 0 && elemCount > std::numeric_limits<size_t>::max() / extent) return false;
        elemCount *= static_cast<size_t>(extent);
        shape.push_back(static_cast<int>(extent));
    }
    return true;
}

// 加载POD类型张量（dense格式）
// 参数 tensor：输出张量
// 参数 info：张量信息
// 参数 rawData：解压后的原始数据
// 返回：是否成功
// 注意：严格匹配payload字节数并验证canonical bool；成功前不修改输出tensor。
YT_IMPL_INLINE bool YTensorIO::loadDense(yt::YTensorBase& tensor, const TensorInfo& info, const std::vector<char>& rawData) {
    try {
    std::vector<int> shape;
    size_t elemCount = 0;
    if (!decodeTensorShape(info, shape, elemCount)) return false;
    int32_t typeSize = yt::type::getTypeSize(info.typeName);
    if (typeSize <= 0 || (elemCount != 0 && elemCount > std::numeric_limits<size_t>::max() / typeSize)) {
        return false;
    }
    size_t expectedBytes = elemCount * static_cast<size_t>(typeSize);
    if (rawData.size() != expectedBytes) return false;
    yt::YTensorBase loaded(shape, info.typeName);
    if (info.typeName == "bool") {
        if (typeSize != static_cast<int32_t>(sizeof(bool))) return false;
        for (char byte : rawData) {
            if (static_cast<unsigned char>(byte) > 1) return false;
        }
        bool* values = loaded.data<bool>();
        for (size_t i = 0; i < elemCount; ++i) {
            values[i] = rawData[i * sizeof(bool)] != 0;
        }
    } else if (expectedBytes > 0) {
        std::memcpy(loaded.rawData(), rawData.data(), expectedBytes);
    }
    tensor = std::move(loaded);
    return true;
    } catch (...) {
        return false;
    }
}

// 加载非POD类型张量（map格式）
// 参数 tensor：输出张量
// 参数 info：张量信息
// 参数 rawData：解压后的原始数据
// 返回：是否成功
// 注意：先验证offset单调性和边界，再在局部tensor的已构造对象上deserialize；成功后一次性提交。
YT_IMPL_INLINE bool YTensorIO::loadMap(yt::YTensorBase& tensor, const TensorInfo& info, const std::vector<char>& rawData) {
    try {
    auto typeInfoOpt = yt::type::getTypeInfo(info.typeName);
    if (!typeInfoOpt) {
        if (verbose) {
            std::cerr << "Error: Type '" << info.typeName << "' is not registered" << std::endl;
        }
        return false;
    }
    
    const auto& typeInfo = typeInfoOpt->get();
    auto deserialize = typeInfo.deserialize;
    
    if (!deserialize) {
        if (verbose) {
            std::cerr << "Error: Non-POD type '" << info.typeName 
                      << "' has no deserialize function registered. Cannot load from file." << std::endl;
        }
        return false;
    }
    
    // 计算元素个数
    std::vector<int> shape;
    size_t elemCount = 0;
    if (!decodeTensorShape(info, shape, elemCount)) return false;
    
    // 解析偏移量数组
    if (elemCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) return false;
    size_t offsetsBytes = elemCount * sizeof(uint64_t);
    if (rawData.size() < offsetsBytes) {
        if (verbose) {
            std::cerr << "Error: Map data corrupted (offset array too small)" << std::endl;
        }
        return false;
    }
    
    const char* serializedData = rawData.data() + offsetsBytes;
    size_t serializedDataSize = rawData.size() - offsetsBytes;

    std::vector<uint64_t> offsets(elemCount);
    for (size_t i = 0; i < elemCount; ++i) {
        std::memcpy(&offsets[i], rawData.data() + i * sizeof(uint64_t), sizeof(uint64_t));
        // 偏移量必须是单调非降的且在serializedData范围内
        if (offsets[i] > serializedDataSize || (i > 0 && offsets[i] < offsets[i - 1])) return false;
    }
    if (!offsets.empty() && offsets[0] != 0) return false;
    
    // 创建张量（会调用defaultConstruct）
    yt::YTensorBase loaded(shape, info.typeName);
    
    // 反序列化每个元素
    char* dstData = loaded.rawData();
    size_t elemSize = loaded.elementSize();
    
    for (size_t i = 0; i < elemCount; ++i) {
        size_t elemStart = offsets[i];
        size_t elemEnd = (i + 1 < elemCount) ? offsets[i + 1] : serializedDataSize;
        if (elemEnd < elemStart || elemEnd > serializedDataSize) return false;
        size_t elemLen = elemEnd - elemStart;

        try {
            deserialize(dstData + i * elemSize, serializedData + elemStart, elemLen);
        } catch (...) {
            // deserialize回调抛异常：视为数据损坏，不部分反序列化
            return false;
        }
    }
    tensor = std::move(loaded);
    return true;
    } catch (...) {
        return false;
    }
}

// ==================== tensor staging and decoding ====================

// 将tensor的独立连续快照编码并暂存到当前write transaction。
// 注意：true只表示staging成功；持久化必须由调用方显式检查close()。
YT_IMPL_INLINE bool YTensorIO::save(const yt::YTensorBase& tensor, const std::string& name) {
    if (_fileMode != yt::io::Write && _fileMode != yt::io::Append) {
        if (verbose) {
            std::cerr << "Error: File not open for writing" << std::endl;
        }
        return false;
    }

    try {
    auto fail = [this]() {
        _writeFailed = true;
        return false;
    };
    // 如果 name 为空，自动命名为 tensorInfos 的 size
    std::string tensorName = name.empty() ? std::to_string(_tensorInfos.size()) : name;
    
    // 检查是否存在同名张量，如果存在则覆盖
    int existingIndex = -1;
    for (size_t i = 0; i < _tensorInfos.size(); ++i) {
        if (_tensorInfos[i].name == tensorName) {
            existingIndex = static_cast<int>(i);
            if (verbose) {
                std::cerr << "Warning: Tensor '" << tensorName << "' already exists, will be overwritten" << std::endl;
            }
            break;
        }
    }
    
    // clone同时解除外部alias并按逻辑顺序连续化，save返回后源tensor修改不会污染待提交payload。
    yt::YTensorBase contiguousTensor = tensor.clone();

    // 创建张量信息
    TensorInfo info;
    info.name = tensorName;
    info.typeName = tensor.dtype();
    info.typeSize = static_cast<int32_t>(tensor.elementSize());
    
    // 获取形状
    auto shape = contiguousTensor.shape();
    info.shape.resize(shape.size());
    std::transform(shape.begin(), shape.end(), info.shape.begin(), [](int s) {
        return static_cast<int32_t>(s);
    });

    // 准备并压缩张量数据
    info.compressMethod = checkCompressMethod(yt::io::compressMethod);
    
    // 检查是否为非POD类型，选择对应的保存后端
    auto typeInfoOpt = yt::type::getTypeInfo(tensor.dtype());
    bool isPOD = !typeInfoOpt || typeInfoOpt->get().isPOD;
    
    std::vector<char> rawData;
    if (isPOD) {
        info.tensorType = "dense";
        rawData = encDense(contiguousTensor);
    } else {
        info.tensorType = "map";
        try {
            rawData = encMap(contiguousTensor);
        } catch (const std::exception& error) {
            // encMap内部serialize回调抛异常：捕获后返回false而非传播
            if (verbose) std::cerr << "Error: " << error.what() << std::endl;
            return fail();
        }
    }
    
    // map payload包含offset表和变长序列化数据，不能用元素数量乘sizeof(T)代替真实长度。
    info.uncompressedSize = static_cast<uint64_t>(rawData.size());
    std::vector<char> compressedData = compressData(rawData);
    if (compressedData.empty() && !rawData.empty()) {
        if (verbose) {
            std::cerr << "Error: Failed to compress tensor data" << std::endl;
        }
        return fail();
    }
    
    info.compressedSize = static_cast<uint64_t>(compressedData.size());
    info.dataOffset = 0;
    info.payloadStaged = true;
    
    // dataOffset 将在 close 函数中重新计算
    
    // 先复制record表再swap，分配或覆盖失败时保留此前成功staging的事务状态。
    auto stagedInfos = _tensorInfos;
    if (existingIndex >= 0) {
        // 覆盖现有张量
        stagedInfos[existingIndex] = info;
        stagedInfos[existingIndex].compressedData = std::move(compressedData);
    } else {
        // 添加新张量
        info.compressedData = std::move(compressedData);
        stagedInfos.push_back(std::move(info));
    }
    _tensorInfos.swap(stagedInfos);
    return true;
    } catch (const std::exception& error) {
        _writeFailed = true;
        if (verbose) std::cerr << "Error: " << error.what() << std::endl;
        return false;
    } catch (...) {
        _writeFailed = true;
        if (verbose) std::cerr << "Error: Unknown tensor save failure" << std::endl;
        return false;
    }
}

// 从staged内存或磁盘payload加载tensor；所有解码后端都在成功前保持输出不变。
YT_IMPL_INLINE bool YTensorIO::load(yt::YTensorBase& tensor, const std::string& name) {
    try {
    if (!_file.is_open() || !_fileMode) {
        if (verbose) {
            std::cerr << "Error: File not open for reading" << std::endl;
        }
        return false;
    }
    
    // 获取张量信息
    TensorInfo info;
    try {
        info = getTensorInfo(name);
    } catch (const std::runtime_error& e) {
        if (verbose) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        return false;
    }
    
    std::vector<char> rawData;
    
    // 根据数据位置选择读取方式
    if (!info.compressedData.empty()) {
        // 数据在内存中，直接解压内存中的压缩数据
        rawData = decompressData(info.compressedData, info.uncompressedSize, info.compressMethod);
    } else {
        // 数据在磁盘中，从文件读取并解压
        _file.seekg(info.dataOffset, std::ios::beg);
        rawData = decompressData(_file, info.compressedSize, info.uncompressedSize, true, info.compressMethod);
    }
    
    if (rawData.empty() && info.uncompressedSize > 0) {
        if (verbose) {
            std::cerr << "Error: File corrupted or incomplete" << std::endl;
        }
        return false; // Decompression failed
    }
    
    // 根据tensorType选择加载后端（dense/map），二者均为事务性：
    // 在局部loaded对象上操作，成功后再move到输出形参
    if (info.tensorType == "map") {
        return loadMap(tensor, info, rawData);
    } else if (info.tensorType == "dense") {
        return loadDense(tensor, info, rawData);
    }
    return false;
    } catch (...) {
        return false;
    }
}

// ==================== one-shot convenience API ====================
YT_IMPL_INLINE bool saveTensorBase(const std::string& fileName, const yt::YTensorBase& tensor, const std::string& name) {
    try {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Write)) {
        return false;
    }
    if (!io.save(tensor, name)) {
        return false;
    }
    return io.close();
    } catch (...) {
        return false;
    }
}

YT_IMPL_INLINE bool loadTensorBase(const std::string& fileName, yt::YTensorBase& tensor, const std::string& name) {
    try {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Read)) {
        return false;
    }
    return io.load(tensor, name);
    } catch (...) {
        return false;
    }
}

}  // namespace yt::io
