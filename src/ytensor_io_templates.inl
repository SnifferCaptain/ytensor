#pragma once
/***************
 * file: ytensor_io_templates.inl
 * purpose: YTensor IO模板编码、压缩和typed便利接口实现。
 ***************/

namespace yt::io {

// 将trivially-copyable数组编码为`uint64 byteCount + native bytes`。
// 注意：前缀是字节数而非元素数；格式沿用当前平台endianness/object representation。
template<typename T>
std::vector<char> array2data(const std::vector<T>& data) {
    if (data.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::overflow_error("array2data: byte size overflow");
    }
    size_t byteCount = data.size() * sizeof(T);
    if (byteCount > std::numeric_limits<uint64_t>::max() ||
        byteCount > std::numeric_limits<size_t>::max() - sizeof(uint64_t)) {
        throw std::overflow_error("array2data: encoded size overflow");
    }
    uint64_t count = static_cast<uint64_t>(byteCount);
    std::vector<char> op(sizeof(uint64_t) + byteCount);
    std::memcpy(op.data(), &count, sizeof(uint64_t));
    if (!data.empty()) {
        std::memcpy(op.data() + sizeof(uint64_t), data.data(), count);
    }
    return op;
}

// 按全局compressMethod/compressLevel压缩连续数组字节。
// 返回：失败和合法空输入均返回空vector，调用方需结合原始长度判断。
template<typename T>
std::vector<char> compressData(const std::vector<T>& input) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(yt::io::compressMethod);
    if (input.size() > std::numeric_limits<size_t>::max() / sizeof(T)) return {};
    size_t byteCount = input.size() * sizeof(T);
    if (cpm == "zlib") {
        if (byteCount > static_cast<size_t>(std::numeric_limits<uInt>::max())) return {};
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        if (deflateInit2(&stream, yt::io::compressLevel, Z_DEFLATED, 15, 8, Z_FILTERED) != Z_OK) {
            return {};
        }
        stream.avail_in = static_cast<uInt>(byteCount);
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        size_t estimated_size = deflateBound(&stream, stream.avail_in);
        if (estimated_size > static_cast<size_t>(std::numeric_limits<uInt>::max())) {
            deflateEnd(&stream);
            return {};
        }
        std::vector<char> compressed(estimated_size);
        stream.avail_out = static_cast<uInt>(compressed.size());
        stream.next_out = reinterpret_cast<Bytef*>(compressed.data());
        int result = deflate(&stream, Z_FINISH);
        deflateEnd(&stream);

        if (result != Z_STREAM_END) {
            return {};
        }
        compressed.resize(stream.total_out);
        return compressed;
    }

    std::vector<char> op(byteCount);
    std::memcpy(op.data(), input.data(), byteCount);
    return op;
}

// typed save仅转交runtime staging；异常会poison当前write transaction。
template<typename T, int dim>
bool YTensorIO::save(const yt::YTensor<T, dim>& tensor, const std::string& name) {
    try {
        return save(static_cast<const yt::YTensorBase&>(tensor), name);
    } catch (...) {
        _writeFailed = true;
        return false;
    }
}

// 先解码到局部YTensorBase，再校验rank/dtype并赋给typed目标，失败时保持目标不变。
template<typename T, int dim>
bool YTensorIO::load(yt::YTensor<T, dim>& tensor, const std::string& name) {
    try {
    yt::YTensorBase base;
    if (!load(base, name)) {
        return false;
    }
    tensor = yt::YTensor<T, dim>(base);
    return true;
    } catch (...) {
        return false;
    }
}

// 一次性写文件并显式检查close commit结果。
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const yt::YTensor<T, dim>& tensor, const std::string& name) {
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

// 一次性typed加载；open、decode或typed转换失败时目标保持不变。
template<typename T, int dim>
bool loadTensor(const std::string& fileName, yt::YTensor<T, dim>& tensor, const std::string& name) {
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
