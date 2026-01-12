#pragma once
/***************
* @file: ytensor_io.hpp
* @brief: YTensor/YTensorBase 类的文件输入/输出功能。
***************/

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <zlib.h>

#include "ytensor_infos.hpp"
#include "ytensor_base.hpp"

namespace yt::io {

/// @brief 文件头标识
using yt::infos::YTENSOR_FILE_MAGIC;

/// @brief 文件版本
using yt::infos::YTENSOR_FILE_VERSION;

/// @brief 压缩级别，在zlib压缩方式中使用
static int8_t compressLevel = Z_DEFAULT_COMPRESSION;

/// @brief 是否输出警告信息
static bool verbose = false;

static constexpr int Closed = 0;
static constexpr int Read = 1;
static constexpr int Write = 2;
static constexpr int Append = 3;

/// @brief 压缩方法
/// @enum "": 不压缩
/// @enum "zlib": 使用zlib压缩，compressLevel控制压缩级别
/// @enum "zlibfloat": 优化后的浮点数专用的压缩方法
static std::string compressMethod = "";

/// @brief 检查压缩方法
/// @param method 压缩方法
/// @return 返回检查后的方法，如果方法无效，回退为空字符串，表示不压缩
std::string checkCompressMethod(const std::string& method);

/// @brief 将字符串转换为写入文件的字节流
std::vector<char> string2data(const std::string& str);

/// @brief 将文件中的字节流转换为字符串
std::string data2string(std::fstream& file, bool seek = true);

/// @brief 将数组转换为写入文件的字节流
template<typename T>
std::vector<char> array2data(const std::vector<T>& data);

/// @brief 从文件中读取数组数据
std::vector<char> data2array(std::fstream& file, bool seek = true);

/// @brief 压缩数据
template<typename T>
std::vector<char> compressData(const std::vector<T>& input);

/// @brief 解压数据
std::vector<char> decompressData(const std::vector<char>& input, size_t decompressedSize = 0, const std::string& method = "");

/// @brief 从文件中解压数据
std::vector<char> decompressData(std::fstream& file, size_t compressedSize, size_t decompressedSize = 0, bool seek = true, const std::string& method = "");

/// @brief 张量信息结构体
struct TensorInfo {
    std::string name;                   // 张量名称
    std::string typeName;               // 元素类型名称
    int32_t typeSize;                   // 元素类型的字节大小
    std::string tensorType;             // dense, sparse等
    std::vector<int> shape;             // 张量形状
    uint64_t dataOffset;                // 在文件中的偏移
    uint64_t compressedSize;            // 压缩后的数据大小
    std::string compressMethod;         // 压缩算法
    uint64_t uncompressedSize;          // 原始数据大小（通过shape计算得到，单位是字节）
    std::vector<char> compressedData;   // 为空表示数据在磁盘，使用dataOffset读取；不为空表示数据在内存中，且原dataOffset失效。
};

/// @brief yt::YTensor/YTensorBase 专用的 IO 类
class YTensorIO {
public:
    YTensorIO() = default;
    ~YTensorIO();

    /// @brief 打开文件
    /// @param fileName 文件名
    /// @param fileMode 文件模式：Read, Write, Append
    /// @return 如果文件打开成功，返回true；否则返回false。
    bool open(const std::string& fileName, int fileMode = yt::io::Read);

    /// @brief 关闭文件
    void close();

    /// @brief 获取文件中所有张量的名称
    /// @return 张量名称列表
    std::vector<std::string> getTensorNames() const;

    /// @brief 获取张量信息
    /// @param name 张量名称，缺省表示获取第一个张量
    /// @return 张量信息，如果不存在抛出异常
    TensorInfo getTensorInfo(const std::string& name = "") const;

    /// @brief 保存张量数据
    /// @param tensor 需要保存的张量
    /// @param name 张量名称
    /// @return 如果保存成功，返回true；否则返回false。
    bool save(const yt::YTensorBase& tensor, const std::string& name);

    /// @brief 保存张量数据 (模板版本)
    /// @tparam T 张量元素类型
    /// @tparam dim 张量维度
    /// @param tensor 需要保存的张量
    /// @param name 张量名称
    /// @return 如果保存成功，返回true；否则返回false。
    template<typename T, int dim>
    bool save(const yt::YTensor<T, dim>& tensor, const std::string& name);

    /// @brief 加载张量数据
    /// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
    /// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
    /// @return 如果加载成功，返回true；否则返回false。
    bool load(yt::YTensorBase& tensor, const std::string& name = "");

    /// @brief 加载张量数据 (模板版本)
    /// @tparam T 张量元素类型
    /// @tparam dim 张量维度
    /// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
    /// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
    /// @return 如果加载成功，返回true；否则返回false。
    template<typename T, int dim>
    bool load(yt::YTensor<T, dim>& tensor, const std::string& name = "");

    /// @brief 检查文件格式是否正确
    /// @return 如果文件格式正确，返回true；否则返回false。
    bool validateFile();

protected:
    /// @brief 读取文件头信息
    /// @return 如果成功，返回true；否则返回false。
    bool readHeader();

    /// @brief 写入文件头信息
    /// @return 如果成功，返回true；否则返回false。
    bool writeHeader();

    /// @brief 读取索引信息
    /// @return 如果成功，返回true；否则返回false。
    bool readIndex();

    /// @brief 写入索引信息
    /// @param tensorStructureOffsets 张量结构体的文件偏移量
    /// @return 如果成功，返回true；否则返回false。
    bool writeIndex(std::vector<uint64_t> tensorStructureOffsets);

protected:
    std::fstream _file;                     // 文件流
    std::string _fileName;                  // 文件名
    int _fileMode = yt::io::Closed;         // 当前文件模式
    bool _isHeaderRead = false;             // 是否已读取文件头
    uint8_t _version = 0;                   // 版本号
    std::string _metadata;                  // JSON格式的元数据
    std::vector<TensorInfo> _tensorInfos;   // 保存张量信息和顺序
};

/// @brief 便利函数用于快速保存 YTensorBase
/// @param fileName 文件名
/// @param tensor 需要保存的张量
/// @param name 张量名称
/// @return 如果保存成功，返回true；否则返回false。
bool saveTensorBase(const std::string& fileName, const yt::YTensorBase& tensor, const std::string& name = "");

/// @brief 便利函数用于快速加载 YTensorBase
/// @param fileName 文件名
/// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
/// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
/// @return 如果加载成功，返回true；否则返回false。
bool loadTensorBase(const std::string& fileName, yt::YTensorBase& tensor, const std::string& name = "");

/// @brief 便利函数用于快速保存 yt::YTensor (模板版本)
/// @tparam T 张量元素类型
/// @tparam dim 张量维度
/// @param fileName 文件名
/// @param tensor 需要保存的张量
/// @param name 张量名称
/// @return 如果保存成功，返回true；否则返回false。
template<typename T, int dim>
bool saveTensor(const std::string& fileName, const yt::YTensor<T, dim>& tensor, const std::string& name = "");

/// @brief 便利函数用于快速加载 yt::YTensor (模板版本)
/// @tparam T 张量元素类型
/// @tparam dim 张量维度
/// @param fileName 文件名
/// @param tensor 需要加载的张量，会进行创建操作，原有的数据、引用均会失效。
/// @param name 张量名称，需要与文件中的张量名称一致，缺省表示读取第一个张量。
/// @return 如果加载成功，返回true；否则返回false。
template<typename T, int dim>
bool loadTensor(const std::string& fileName, yt::YTensor<T, dim>& tensor, const std::string& name = "");

}; // namespace yt::io

