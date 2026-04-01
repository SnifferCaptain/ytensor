#pragma once

namespace yt::io {

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

template<typename T>
std::vector<char> compressData(const std::vector<T>& input) {
    if (input.empty()) {
        return {};
    }
    std::string cpm = checkCompressMethod(yt::io::compressMethod);
    if (cpm == "zlib") {
        z_stream stream;
        std::memset(&stream, 0, sizeof(z_stream));
        if (deflateInit2(&stream, yt::io::compressLevel, Z_DEFLATED, 15, 8, Z_FILTERED) != Z_OK) {
            return {};
        }
        stream.avail_in = input.size() * sizeof(T);
        stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
        size_t estimated_size = deflateBound(&stream, stream.avail_in);
        std::vector<char> compressed(estimated_size);
        stream.avail_out = compressed.size();
        stream.next_out = reinterpret_cast<Bytef*>(compressed.data());
        int result = deflate(&stream, Z_FINISH);
        deflateEnd(&stream);

        if (result != Z_STREAM_END) {
            return {};
        }
        compressed.resize(stream.total_out);
        return compressed;
    }

    std::vector<char> op(input.size() * sizeof(T));
    std::memcpy(op.data(), input.data(), input.size() * sizeof(T));
    return op;
}

template<typename T, int dim>
bool YTensorIO::save(const yt::YTensor<T, dim>& tensor, const std::string& name) {
    return save(static_cast<const yt::YTensorBase&>(tensor), name);
}

template<typename T, int dim>
bool YTensorIO::load(yt::YTensor<T, dim>& tensor, const std::string& name) {
    yt::YTensorBase base;
    if (!load(base, name)) {
        return false;
    }
    tensor = yt::YTensor<T, dim>(base);
    return true;
}

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

