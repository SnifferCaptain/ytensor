#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include "../include/ytensor_io.hpp"
#include <zlib.h>

namespace yt::io {

inline std::string checkCompressMethod(const std::string& method) {
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

inline std::vector<char> string2data(const std::string& str) {
    uint32_t length = static_cast<uint32_t>(str.length());
    std::vector<char> op(sizeof(uint32_t) + length);
    std::memcpy(op.data(), &length, sizeof(uint32_t));
    if (!str.empty()) {
        std::memcpy(op.data() + sizeof(uint32_t), str.c_str(), length);
    }
    return op;
}

inline std::string data2string(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint32_t length;
    if (!file.read(reinterpret_cast<char*>(&length), sizeof(uint32_t))) {
        throw std::runtime_error("Failed to read string length");
    }
    std::string op;
    if (length > 0) {
        op.resize(length);
        if (!file.read(op.data(), length)) {
            throw std::runtime_error("Failed to read string data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }    
    return op;
}

template<typename T>
std::vector<char> array2data(const std::vector<T>& data) {
    uint64_t count = static_cast<uint64_t>(data.size() * sizeof(T));
    std::vector<char> op(sizeof(uint64_t) + count);
    std::memcpy(op.data(), &count, sizeof(uint64_t));
    if (!data.empty()) {
        std::memcpy(op.data() + sizeof(uint64_t), data.data(), count);
    }
    return op;
}

inline std::vector<char> data2array(std::fstream& file, bool seek) {
    std::streampos originalPos = file.tellg();
    uint64_t count;
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t))) {
        throw std::runtime_error("Failed to read array count");
    }
    std::vector<char> op;
    if (count > 0) {
        op.resize(count);
        if (!file.read(op.data(), count)) {
            throw std::runtime_error("Failed to read array data");
        }
    }
    if (!seek) {
        file.seekg(originalPos);
    }
    return op;
}

template<typename T>
std::vector<char> compressData(const std::vector<T>& input) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(yt::io::compressMethod);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        // 初始化压缩流zlib
        if (deflateInit2(&stream, yt::io::compressLevel, Z_DEFLATED, 15, 8, Z_FILTERED) != Z_OK) {
            return {}; // 初始化失败
        }
        stream.avail_in = input.size() * sizeof(T);
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        // 预估压缩后大小，通常比原数据稍大一些作为缓冲
        size_t estimated_size = deflateBound(&stream, stream.avail_in);
        std::vector<char> compressed(estimated_size);
        stream.avail_out = compressed.size();
        stream.next_out = reinterpret_cast<Bytef*>(compressed.data());
        int result = deflate(&stream, Z_FINISH);
        deflateEnd(&stream);

        if (result != Z_STREAM_END) {
            return {}; // 压缩失败
        }
        compressed.resize(stream.total_out);
        return compressed;
    } else {
        // 缺省值/fallback，不压缩
        std::vector<char> op(input.size() * sizeof(T));
        std::memcpy(op.data(), input.data(), input.size() * sizeof(T));
        return op;
    }
}

inline std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize, const std::string& method) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
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
        size_t first_chunk_size = (decompressedSize > 0) ? decompressedSize : chunk_size;
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

inline std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize, bool seek, const std::string& method) {
    if (compressedSize == 0) {
        return {};
    }

    std::streampos originalPos = file.tellg();
    std::vector<char> output;

    std::string cpm = checkCompressMethod(method);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        
        if (inflateInit2(&stream, 15 + 32) != Z_OK) {
            return {};
        }

        const size_t io_chunk = 65536; // 64KB 输入缓冲
        std::vector<char> input_buffer(io_chunk);
        size_t remaining = compressedSize;

        if (decompressedSize > 0) {
            output.resize(decompressedSize);
        } else {
            output.resize(io_chunk);
        }

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

        inflateEnd(&stream);
        output.resize(out_pos);

    } else {
        output.resize(compressedSize);
        if (!file.read(output.data(), compressedSize)) {
            file.seekg(originalPos);
            if (yt::io::verbose) {
                std::cerr << "Error: Failed to read compressed data from file" << std::endl;
            }
            return {};
        }
    }
    
    if (!seek) {
        file.seekg(originalPos);
    }
    return output;
}

inline YTensorIO::~YTensorIO() {
    close();
}

inline bool YTensorIO::open(const std::string& fileName, int fileMode) {
    close();
    _fileName = fileName;
    _fileMode = fileMode;
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
                // 读取失败，当作新文件处理
                _file.close();
                _tensorInfos.clear();
            }
        } else {
            // 文件不存在，创建新的文件
            _file.open(fileName, std::ios::binary | std::ios::out | std::ios::trunc);
            _file.close();
        }
        
        // 在关闭文件之前，都以只读模式打开
        _file.open(fileName, std::ios::binary | std::ios::in);
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
}

inline void YTensorIO::close() {
    if (_file.is_open()) {
        if (_fileMode == yt::io::Write || _fileMode == yt::io::Append) {
            if (_fileMode == yt::io::Append) {
                // 如果是附加模式，先加载所有的张量数据进内存。无需解压。
                for (size_t i = 0; i < _tensorInfos.size(); ++i) {
                    auto& info = _tensorInfos[i];
                    if(!info.compressedData.empty()) {
                        // 已经在内存中，无需重复加载
                        info.compressedSize = info.compressedData.size();
                        continue;
                    }
                    _file.seekg(info.dataOffset, std::ios::beg);

                    // 读取压缩数据到内存
                    info.compressedData.resize(info.compressedSize);
                    if (info.compressedSize > 0) {
                        if (!_file.read(info.compressedData.data(), info.compressedSize)){
                            if (verbose) {
                                std::cerr << "Error: Failed to read compressed data for tensor '" 
                                          << info.name << "'" << std::endl;
                            }
                            info.compressedData.clear();
                        }
                    }
                }
            } else {
                // 如果是写入模式，去除所有不在内存中的张量
                std::erase_if(_tensorInfos, [](const TensorInfo& info) {
                    return info.compressedData.empty();
                });
            }

            // 重新打开文件进行写入
            std::string tempFileName = _fileName;
            _file.close();
            _file.open(tempFileName, std::ios::binary | std::ios::out | std::ios::trunc);
            if (_file.is_open()) {
                // 写入头部（包含 metadata）
                if (!writeHeader()) {
                    if (verbose) {
                        std::cerr << "Error: Failed to write header during close" << std::endl;
                    }
                }
                
                // 写入所有张量数据
                std::vector<uint64_t> offsets;
                for (size_t i = 0; i < _tensorInfos.size(); ++i) {
                    const auto& info = _tensorInfos[i];

                    // 记录数据偏移量（张量结构数据的开始位置）
                    uint64_t dataOffset = _file.tellp();
                    offsets.push_back(dataOffset);
                    // 写入张量元数据
                    auto nameData = string2data(info.name);
                    auto typeNameData = string2data(info.typeName);
                    int32_t typeSize = info.typeSize;
                    auto tensorTypeData = string2data(info.tensorType);
                    auto shapeData = array2data(info.shape);
                    auto compressMethodData = string2data(info.compressMethod);

                    if (!_file.write(nameData.data(), nameData.size()) ||
                        !_file.write(typeNameData.data(), typeNameData.size()) ||
                        !_file.write(reinterpret_cast<const char*>(&typeSize), sizeof(int32_t)) ||
                        !_file.write(tensorTypeData.data(), tensorTypeData.size()) ||
                        !_file.write(shapeData.data(), shapeData.size()) ||
                        !_file.write(compressMethodData.data(), compressMethodData.size())) {
                        if (verbose) {
                            std::cerr << "Error: Failed to write tensor metadata for '" << info.name << "'" << std::endl;
                        }
                        break;
                    }
                    // 写入压缩数据大小
                    if (!_file.write(reinterpret_cast<const char*>(&info.compressedSize), sizeof(uint64_t))) {
                        if (verbose) {
                            std::cerr << "Error: Failed to write compressed size for '" << info.name << "'" << std::endl;
                        }
                        break;
                    }
                    if (!info.compressedData.empty()) {
                        if (!_file.write(info.compressedData.data(), info.compressedData.size())) {
                            if (verbose) {
                                std::cerr << "Error: Failed to write tensor data for '" << info.name << "'" << std::endl;
                            }
                            break;
                        }
                    }
                }
                // 写入索引到文件末尾
                if (!writeIndex(offsets)) {
                    if (verbose) {
                        std::cerr << "Error: Failed to write index during close" << std::endl;
                    }
                }
            }
        }
        _file.close();
    }
    _fileMode = yt::io::Closed;
    _isHeaderRead = false;
    _version = 0;
    _tensorInfos.clear();
    _metadata.clear();
    _fileName.clear();
}

inline std::vector<std::string> YTensorIO::getTensorNames() const {
    std::vector<std::string> names;
    for (const auto& tensorInfo : _tensorInfos) {
        names.push_back(tensorInfo.name);
    }
    return names;
}

inline TensorInfo YTensorIO::getTensorInfo(const std::string& name) const {
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

inline bool YTensorIO::validateFile() {
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
}

inline bool YTensorIO::readHeader() {
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
    // 读取 metadata（紧接着版本号）
    _metadata = data2string(_file);
    _isHeaderRead = true;
    return true;
}

inline bool YTensorIO::writeHeader() {
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

inline bool YTensorIO::readIndex() {
    if (!_file.is_open()) {
        return false;
    }
    _tensorInfos.clear();
    _file.seekg(-static_cast<int>(sizeof(uint32_t)), std::ios::end); 
    uint32_t tensorCount;
    if (!_file.read(reinterpret_cast<char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    // 读取索引
    const std::streamoff indexSize = 
        static_cast<std::streamoff>(sizeof(uint32_t)) + 
        static_cast<std::streamoff>(tensorCount) * sizeof(uint64_t);
    _file.seekg(-indexSize, std::ios::end);

    std::vector<uint64_t> offsets(tensorCount);
    if (!_file.read(reinterpret_cast<char*>(offsets.data()), tensorCount * sizeof(uint64_t))) {
        return false;
    }
    
    // Read tensor information from each offset
    for (size_t i = 0; i < tensorCount; ++i) {
        _file.seekg(offsets[i], std::ios::beg);
        TensorInfo info;
        info.name = data2string(_file);
        info.typeName = data2string(_file);
        if (!_file.read(reinterpret_cast<char*>(&info.typeSize), sizeof(int32_t))) {
            return false;
        }
        info.tensorType = data2string(_file);
        auto shapeData = data2array(_file);
        const int32_t* shapePtr = reinterpret_cast<const int32_t*>(shapeData.data());
        size_t shapeCount = shapeData.size() / sizeof(int32_t);
        info.shape.assign(shapePtr, shapePtr + shapeCount);
        info.compressMethod = data2string(_file);
        uint64_t compressedSize;
        if (!_file.read(reinterpret_cast<char*>(&compressedSize), sizeof(uint64_t))) {
            return false;
        }
        info.compressedSize = compressedSize;
        // data offset是指向压缩数据的偏移量
        info.dataOffset = _file.tellg();
        info.uncompressedSize = info.typeSize;
        for (auto dim : info.shape) {
            info.uncompressedSize *= dim;
        }
        info.compressedData.clear(); // 初始时不加载数据
        _tensorInfos.push_back(info);
    }
    return true;
}

inline bool YTensorIO::writeIndex(std::vector<uint64_t> offsets) {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        return false;
    }
    if (!_file.write(reinterpret_cast<const char*>(offsets.data()), 
                     offsets.size() * sizeof(uint64_t))) {
        return false;
    }
    uint32_t tensorCount = static_cast<uint32_t>(offsets.size());
    if (!_file.write(reinterpret_cast<const char*>(&tensorCount), sizeof(uint32_t))) {
        return false;
    }
    return true;
}

inline bool YTensorIO::save(const yt::YTensorBase& tensor, const std::string& name) {
    if (!_file.is_open() || _fileMode == yt::io::Read) {
        if (verbose) {
            std::cerr << "Error: File not open for writing" << std::endl;
        }
        return false;
    }
    
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
    
    // 需要保证是连续的，否则数据排布不正确
    yt::YTensorBase contiguousTensor = tensor.clone();
    
    // 创建张量信息
    TensorInfo info;
    info.name = tensorName;
    info.typeName = tensor.dtype();
    info.typeSize = static_cast<int32_t>(tensor.elementSize());
    info.tensorType = "dense";
    
    // 获取形状
    auto shape = contiguousTensor.shape();
    info.shape.resize(shape.size());
    std::transform(shape.begin(), shape.end(), info.shape.begin(), [](int s) {
        return static_cast<int32_t>(s);
    });

    // 准备并压缩张量数据
    info.compressMethod = checkCompressMethod(yt::io::compressMethod);
    
    // 获取原始数据
    size_t dataSize = contiguousTensor.size() * contiguousTensor.elementSize();
    std::vector<char> rawData(dataSize);
    std::memcpy(rawData.data(), contiguousTensor.data(), dataSize);
    
    // 压缩数据
    auto compressedData = compressData(rawData);
    if (compressedData.empty() && dataSize > 0) {
        if (verbose) {
            std::cerr << "Error: Failed to compress tensor data" << std::endl;
        }
        return false;
    }
    
    info.compressedSize = static_cast<uint64_t>(compressedData.size());  
    info.uncompressedSize = dataSize;
    
    // dataOffset 将在 close 函数中重新计算
    
    if (existingIndex >= 0) {
        // 覆盖现有张量
        _tensorInfos[existingIndex] = info;
        _tensorInfos[existingIndex].compressedData = std::move(compressedData);
    } else {
        // 添加新张量
        info.compressedData = std::move(compressedData);
        _tensorInfos.push_back(info);
    }
    return true;
}

// 模板方法实现：保存 YTensor<T, dim>
template<typename T, int dim>
bool YTensorIO::save(const yt::YTensor<T, dim>& tensor, const std::string& name) {
    // YTensor 继承自 YTensorBase，直接调用基类版本
    return save(static_cast<const yt::YTensorBase&>(tensor), name);
}

inline bool YTensorIO::load(yt::YTensorBase& tensor, const std::string& name) {
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

    std::vector<int> shape(info.shape.begin(), info.shape.end());
    
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
    
    // 检查tensorSize是否与解压的数据大小匹配
    if (info.uncompressedSize != rawData.size()) {
        if (verbose) {
            std::cerr << "Error: YTensorBase data size mismatch. Expected " << info.uncompressedSize 
                      << ", got " << rawData.size() <<
                       ". Please check your file." << std::endl;
        }
        return false; // Decompressed data size mismatch
    }
    
    // 使用文件中的 dtype 创建张量
    tensor = yt::YTensorBase(shape, info.typeName);
    
    // Copy decompressed data to tensor
    if (!rawData.empty()) {
        std::memcpy(tensor.data(), rawData.data(), rawData.size());
    }
    return true;
}

// 模板方法实现：加载 YTensor<T, dim>
template<typename T, int dim>
bool YTensorIO::load(yt::YTensor<T, dim>& tensor, const std::string& name) {
    // 先加载为 YTensorBase
    yt::YTensorBase base;
    if (!load(base, name)) {
        return false;
    }
    // 转换为 YTensor<T, dim>
    tensor = yt::YTensor<T, dim>(base);
    return true;
}

// 便利函数实现
inline bool saveTensorBase(const std::string& fileName, const yt::YTensorBase& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Write)) {
        return false;
    }
    if (!io.save(tensor, name)) {
        return false;
    }
    io.close();
    return true;
}

inline bool loadTensorBase(const std::string& fileName, yt::YTensorBase& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Read)) {
        return false;
    }
    return io.load(tensor, name);
}

// 便利函数模板实现
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const yt::YTensor<T, dim>& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Write)) {
        return false;
    }
    if (!io.save(tensor, name)) {
        return false;
    }
    io.close();
    return true;
}

template<typename T, int dim>
bool loadTensor(const std::string& fileName, yt::YTensor<T, dim>& tensor, const std::string& name) {
    YTensorIO io;
    if (!io.open(fileName, yt::io::Read)) {
        return false;
    }
    return io.load(tensor, name);
}

} // namespace yt::io
