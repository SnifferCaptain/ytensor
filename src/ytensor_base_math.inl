/***************
* @file: ytensor_base_math.inl
* @brief: YTensorBase 数学运算的实现
* @author: SnifferCaptain
* @date: 2025-12-1
* @version 1.0
* @email: 3586554865@qq.com
***************/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include "../include/ytensor_infos.hpp"
#include "../include/ytensor_types.hpp"
#include "../include/kernel/parallel_for.hpp"
#include "../include/kernel/math_utils.hpp"
#include "../include/kernel/memory_utils.hpp"
#include "../include/kernel/broadcast.hpp"
#include "../include/kernel/type_dispatch.hpp"
#if YT_USE_AVX2
#include "../include/kernel/gemm.hpp"
#endif

namespace yt{

// ======================== broadcastInplace 实现 ========================

template<typename Func, typename... Args>
YTensorBase& YTensorBase::broadcastInplace(Func&& func, Args&&... tensors) {
    using namespace yt::traits;
    
    // 从func的第一个参数推断DType（去掉引用和const）
    using DType = std::remove_cvref_t<first_arg_of_t<Func>>;
    
    // 收集所有张量的shape（包括this）
    std::vector<std::vector<int>> shapes;
    shapes.push_back(this->shape());
    
    auto collectShape = [&shapes](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            shapes.push_back(arg.shape());
        }
    };
    (collectShape(tensors), ...);
    
    // 计算广播shape
    auto broadcastShape = yt::kernel::computeBroadcastShape(shapes);
    int thisDim = this->ndim();
    
    // 验证this的shape与广播shape兼容
    auto thisShapeVec = this->shape();
    if (static_cast<int>(broadcastShape.size()) != thisDim) {
        throw std::runtime_error("broadcastInplace: result dimension mismatch");
    }
    for (int i = 0; i < thisDim; ++i) {
        if (thisShapeVec[i] != broadcastShape[i]) {
            throw std::runtime_error("broadcastInplace: this tensor shape must match broadcast shape");
        }
    }
    
    int totalSize = this->size();
    bool allContiguous = this->isContiguous();
    bool allEqualShape = true;
    
    auto checkContiguousAndShape = [&](auto&& arg) {
        if constexpr (is_ytensor_v<decltype(arg)>) {
            if (!arg.isContiguous()) {
                allContiguous = false;
            }
            auto argShape = arg.shape();
            if (argShape.size() != shapes[0].size()) {
                allEqualShape = false;
            } else {
                for (size_t i = 0; i < argShape.size(); ++i) {
                    if (argShape[i] != shapes[0][i]) {
                        allEqualShape = false;
                        break;
                    }
                }
            }
        }
    };
    (checkContiguousAndShape(tensors), ...);
    
    // 不再使用YT_DISPATCH_BY_DTYPE，因为DType已经从func的参数类型推断出来了
    if (allContiguous && allEqualShape) {
        // 快速路径：所有张量连续且shape相同
        DType* thisDataPtr = this->data<DType>();
        
        // 收集所有张量的数据指针
        std::vector<const DType*> dataPtrs;
        [[maybe_unused]] auto collectPtrs = [&](auto&& arg) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                if constexpr (is_ytensor_template_v<decltype(arg)>) {
                    dataPtrs.push_back(reinterpret_cast<const DType*>(arg.data()));
                } else {
                    dataPtrs.push_back(arg.template data<DType>());
                }
            }
        };
        (collectPtrs(tensors), ...);
        
        // 创建fastpath的getValue lambda
        [[maybe_unused]] auto getValueFast = [&](auto&& arg, int index, int& tensorIdx) -> decltype(auto) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                ++tensorIdx;
                return dataPtrs[tensorIdx - 1][index];
            } else {
                return static_cast<DType>(std::forward<decltype(arg)>(arg));
            }
        };
        
        yt::kernel::parallelFor(0, totalSize, [&](int index) {
            [[maybe_unused]] int tensorIdx = 0;
            func(thisDataPtr[index], getValueFast(tensors, index, tensorIdx)...);
        });
    } else {
        // 慢速路径：需要计算广播索引
        DType* thisDataPtr = this->data<DType>();
        auto logicStride = this->stride();
        auto thisStride = this->stride_();
        
        // 编译时计算张量参数中有多少个是张量类型
        constexpr size_t numTensors = ((is_ytensor_v<std::decay_t<Args>> ? 1 : 0) + ... + 0);
        
        // 收集每个张量的广播stride和数据指针（使用std::array）
        std::array<std::vector<int>, numTensors> broadcastStrides;
        std::array<const DType*, numTensors> dataPtrs;
        
        [[maybe_unused]] size_t tensorIdx = 0;
        [[maybe_unused]] auto collectBroadcastInfo = [&](auto&& arg) {
            if constexpr (is_ytensor_v<decltype(arg)>) {
                broadcastStrides[tensorIdx] = yt::kernel::getBroadcastStride(arg.shape(), arg.stride_(), broadcastShape);
                if constexpr (is_ytensor_template_v<decltype(arg)>) {
                    dataPtrs[tensorIdx] = reinterpret_cast<const DType*>(arg.data());
                } else {
                    dataPtrs[tensorIdx] = arg.template data<DType>();
                }
                ++tensorIdx;
            }
        };
        (collectBroadcastInfo(tensors), ...);
        
        // 预分配strides指针数组用于computeBroadcastIndices
        std::array<const int*, numTensors + 1> stridesArray;
        stridesArray[0] = thisStride.data();
        for (size_t i = 0; i < numTensors; ++i) {
            stridesArray[i + 1] = broadcastStrides[i].data();
        }
        
        yt::kernel::parallelFor(0, totalSize, [&](int index) {
            // 使用编译期模板版本计算索引，返回std::array，无堆分配
            auto indices = yt::kernel::computeBroadcastIndices<numTensors + 1>(
                index, logicStride, broadcastShape, stridesArray, thisDim);
            int thisIdx = indices[0];
            
            // 获取值并调用func
            [[maybe_unused]] size_t tIdx = 0;
            [[maybe_unused]] auto getValue = [&](auto&& arg) -> decltype(auto) {
                if constexpr (is_ytensor_v<decltype(arg)>) {
                    size_t idx = tIdx++;
                    return dataPtrs[idx][indices[idx + 1]];
                } else {
                    return static_cast<DType>(std::forward<decltype(arg)>(arg));
                }
            };
            
            func(thisDataPtr[thisIdx + _offset], getValue(tensors)...);
        });
    }
    
    return *this;
}


// ======================== 算术运算符实现 ========================

// 统一运算符宏 - 同时生成 Tensor op Tensor 和 Tensor op Scalar 的4个版本
// TypeListT: 类型列表（如 yt::types::AllNumericTypes）
#define YT_IMPL_BINARY_OP(OP, OP_NAME, TypeListT)                                          \
/* Tensor op Tensor - 非原地版本 */                                                         \
inline YTensorBase YTensorBase::operator OP(const YTensorBase& other) const {              \
    auto opShape = yt::kernel::computeBroadcastShape({this->shape(), other.shape()});      \
    YTensorBase result(opShape, _dtype);                                                   \
    result.copy_(*this);                                                                   \
    yt::kernel::dispatchOrThrow<TypeListT>(_dtype,                                         \
        [&]<typename DType>() {                                                            \
            result.broadcastInplace([](DType& a, const DType& b) { a = a OP b; }, other);  \
        }, OP_NAME);                                                                       \
    return result;                                                                         \
}                                                                                          \
/* Tensor op Tensor - 原地版本 */                                                            \
inline YTensorBase& YTensorBase::operator OP##=(const YTensorBase& other) {                \
    yt::kernel::dispatchOrThrow<TypeListT>(_dtype,                                         \
        [&]<typename DType>() {                                                            \
            this->broadcastInplace([](DType& a, const DType& b) { a = a OP b; }, other);   \
        }, OP_NAME);                                                                       \
    return *this;                                                                          \
}                                                                                          \
/* Tensor op Scalar - 非原地版本 */                                                          \
template<typename T>                                                                       \
YTensorBase YTensorBase::operator OP(const T& scalar) const {                              \
    YTensorBase result(_shape, yt::types::getTypeName<T>());                               \
    result.copy_(*this);                                                                   \
    result.broadcastInplace([](T& a, const T& b) { a = a OP b; }, scalar);                 \
    return result;                                                                         \
}                                                                                          \
/* Tensor op Scalar - 原地版本 */                                                            \
template<typename T>                                                                       \
YTensorBase& YTensorBase::operator OP##=(const T& scalar) {                                \
    this->broadcastInplace([](T& a, const T& b) { a = a OP b; }, scalar);                  \
    return *this;                                                                          \
}

// 实例化所有运算符 - 数值类型
YT_IMPL_BINARY_OP(+, "+", yt::types::AllNumericTypes)
YT_IMPL_BINARY_OP(-, "-", yt::types::AllNumericTypes)
YT_IMPL_BINARY_OP(*, "*", yt::types::AllNumericTypes)
YT_IMPL_BINARY_OP(/, "/", yt::types::AllNumericTypes)

// 实例化所有运算符 - 仅整数类型
YT_IMPL_BINARY_OP(%, "%", yt::types::IntegerTypes)
YT_IMPL_BINARY_OP(&, "&", yt::types::IntegerTypes)
YT_IMPL_BINARY_OP(|, "|", yt::types::IntegerTypes)
YT_IMPL_BINARY_OP(^, "^", yt::types::IntegerTypes)
YT_IMPL_BINARY_OP(<<, "<<", yt::types::IntegerTypes)
YT_IMPL_BINARY_OP(>>, ">>", yt::types::IntegerTypes)

// 清理宏
#undef YT_IMPL_BINARY_OP

// ======================== 比较运算符实现 ========================
// 比较运算符返回 dtype="bool" 的 YTensorBase

#define YT_IMPL_CMP_OP(OP, OP_NAME)                                                        \
/* Tensor op Tensor - 返回 bool 张量 */                                                      \
inline YTensorBase YTensorBase::operator OP(const YTensorBase& other) const {              \
    auto opShape = yt::kernel::computeBroadcastShape({this->shape(), other.shape()});      \
    YTensorBase result(opShape, "bool");                                                   \
    yt::kernel::dispatchOrThrow<yt::types::AllNumericTypes>(_dtype,                       \
        [&]<typename DType>() {                                                            \
            bool* resultData = result.data<bool>();                                        \
            size_t totalSize = result.size();                                              \
            const DType* thisData = this->data<DType>();                                   \
            const DType* otherData = other.data<DType>();                                  \
            if (this->isContiguous() && other.isContiguous() && this->shapeMatch(other.shape())) { \
                yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {       \
                    resultData[i] = thisData[i] OP otherData[i];                           \
                });                                                                        \
            } else {                                                                       \
                auto thisStride = this->stride_();                                         \
                auto otherStride = other.stride_();                                        \
                auto thisBroadcastStride = yt::kernel::getBroadcastStride(this->shape(), thisStride, opShape);  \
                auto otherBroadcastStride = yt::kernel::getBroadcastStride(other.shape(), otherStride, opShape);\
                auto logicStride = result.stride();                                        \
                int dim = static_cast<int>(opShape.size());                                \
                yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {       \
                    int remaining = i;                                                     \
                    int thisIdx = 0, otherIdx = 0;                                         \
                    for (int d = dim - 1; d >= 0; --d) {                                   \
                        int coord = remaining / logicStride[d];                            \
                        remaining %= logicStride[d];                                       \
                        thisIdx += coord * thisBroadcastStride[d];                         \
                        otherIdx += coord * otherBroadcastStride[d];                       \
                    }                                                                      \
                    resultData[i] = thisData[thisIdx + _offset] OP otherData[otherIdx + other._offset]; \
                });                                                                        \
            }                                                                              \
        }, OP_NAME);                                                                       \
    return result;                                                                         \
}                                                                                          \
/* Tensor op Scalar */                                                                      \
template<typename T>                                                                       \
YTensorBase YTensorBase::operator OP(const T& scalar) const {                              \
    YTensorBase result(_shape, "bool");                                                    \
    bool* resultData = result.data<bool>();                                                \
    size_t totalSize = result.size();                                                      \
    if (this->isContiguous()) {                                                            \
        const T* thisData = this->data<T>();                                               \
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {               \
            resultData[i] = thisData[i] OP scalar;                                         \
        });                                                                                \
    } else {                                                                               \
        const T* thisData = this->data<T>();                                               \
        auto logicStride = this->stride();                                                 \
        auto physStride = this->stride_();                                                 \
        int dim = this->ndim();                                                            \
        yt::kernel::parallelFor(0, static_cast<int>(totalSize), [&](int i) {               \
            int remaining = i;                                                             \
            int physIdx = 0;                                                               \
            for (int d = dim - 1; d >= 0; --d) {                                           \
                int coord = remaining / logicStride[d];                                    \
                remaining %= logicStride[d];                                               \
                physIdx += coord * physStride[d];                                          \
            }                                                                              \
            resultData[i] = thisData[physIdx + _offset] OP scalar;                         \
        });                                                                                \
    }                                                                                      \
    return result;                                                                         \
}

YT_IMPL_CMP_OP(<, "<")
YT_IMPL_CMP_OP(<=, "<=")
YT_IMPL_CMP_OP(>, ">")
YT_IMPL_CMP_OP(>=, ">=")
YT_IMPL_CMP_OP(==, "==")
YT_IMPL_CMP_OP(!=, "!=")

#undef YT_IMPL_CMP_OP

// ======================== sum ========================

inline YTensorBase YTensorBase::sum(int axis) const {
    int dim = ndim();
    if (dim == 0) {
        throw std::runtime_error("[YTensorBase::sum] Cannot sum a 0-dim tensor");
    }
    axis = (axis % dim + dim) % dim;
    
    // 计算输出形状（对应轴设为1）
    auto newShape = this->shape();
    int axisSize = newShape[axis];
    newShape[axis] = 1;
    
    YTensorBase op(newShape, _dtype);
    size_t outSize = op.size();
    
    yt::kernel::dispatchOrThrow<yt::types::AllNumericTypes>(_dtype,
        [&]<typename DType>() {
            DType* opData = op.data<DType>();
            yt::kernel::parallelFor(0, static_cast<int>(outSize), [&](int i) {
                auto coord = op.toCoord(i);
                DType sum{};
                for (int j = 0; j < axisSize; j++) {
                    auto subCoord = coord;
                    subCoord[axis] = j;
                    int physIdx = 0;
                    for (int k = 0; k < dim; ++k) {
                        physIdx += subCoord[k] * _stride[k];
                    }
                    sum += *(reinterpret_cast<const DType*>(_data.get() + (_offset + physIdx) * _element_size));
                }
                opData[i] = sum;
            }, static_cast<double>(axisSize));
        }, "sum");
    
    return op;
}

inline YTensorBase YTensorBase::sum(const std::vector<int>& axes) const {
    // 简单实现：逐个轴求和
    YTensorBase result = *this;
    // 排序axes，从大到小，避免轴索引变化问题
    std::vector<int> sortedAxes = axes;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    for (int ax : sortedAxes) {
        result = result.sum(ax);
    }
    return result;
}

// ======================== max ========================

inline std::pair<YTensorBase, YTensorBase> YTensorBase::max(int axis) const {
    int dim = ndim();
    if (dim == 0) {
        throw std::runtime_error("[YTensorBase::max] Cannot max a 0-dim tensor");
    }
    
    // 处理负索引
    axis = (axis % dim + dim) % dim;
    
    // 计算输出形状（对应轴设为1）
    auto newShape = this->shape();
    int axisSize = newShape[axis];
    newShape[axis] = 1;
    
    YTensorBase values(newShape, _dtype);
    YTensorBase indices(newShape, "int32");
    size_t outSize = values.size();
    
    yt::kernel::dispatchOrThrow<yt::types::AllNumericTypes>(_dtype,
        [&]<typename DType>() {
            DType* valData = values.data<DType>();
            int32_t* idxData = indices.data<int32_t>();
            yt::kernel::parallelFor(0, static_cast<int>(outSize), [&](int i) {
                auto coord = values.toCoord(i);
                DType maxVal = std::numeric_limits<DType>::lowest();
                int32_t maxIdx = 0;
                for (int j = 0; j < axisSize; j++) {
                    auto subCoord = coord;
                    subCoord[axis] = j;
                    int physIdx = 0;
                    for (int k = 0; k < dim; ++k) {
                        physIdx += subCoord[k] * _stride[k];
                    }
                    DType val = *(reinterpret_cast<const DType*>(_data.get() + (_offset + physIdx) * _element_size));
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                valData[i] = maxVal;
                idxData[i] = maxIdx;
            }, static_cast<double>(axisSize));
        }, "max");
    
    return {values, indices};
}

inline std::pair<YTensorBase, YTensorBase> YTensorBase::max(const std::vector<int>& axes) const {
    // 简单实现：逐个轴求最大值
    YTensorBase values = *this;
    YTensorBase indices;
    std::vector<int> sortedAxes = axes;
    std::sort(sortedAxes.begin(), sortedAxes.end(), std::greater<int>());
    for (int ax : sortedAxes) {
        auto [v, idx] = values.max(ax);
        values = v;
        indices = idx;  // 只保留最后一次的索引
    }
    return {values, indices};
}

// ======================== matView ========================

// 注意：YTensorBase的matView实现与YTensor不同
// YTensorBase是类型擦除的，我们返回一个特殊的YTensorBase，其"标量"是YTensorBase子视图
// 
// 为什么这里需要手动处理非POD（YTensorBase）？
// 1. ytensor_base.inl 中的非POD支持是针对"用户通过registerType注册的自定义类型"
// 2. 但 YTensorBase 本身不能被注册（循环依赖：YTensorBase 在 ytensor_types.hpp 之前定义）
// 3. matView 返回的张量的元素类型就是 YTensorBase 本身，这是一个特殊情况
// 4. 因此必须手动使用 placement new 和自定义删除器来正确管理 YTensorBase 元素的生命周期
//
// dtype 命名规范：
// matView 返回的 YTensorBase 的 dtype 为 "YTensorBase<inner_dtype>"
// 其中 inner_dtype 是原始张量的 dtype

inline YTensorBase YTensorBase::matView() const {
    int dim = ndim();
    if (dim < 1) {
        throw std::runtime_error("[YTensorBase::matView] Tensor must have at least 1 dimension");
    }
    
    // 构建规范化的dtype
    std::string innerDtype = yt::types::makeYTensorBaseDtype(_dtype);  // "YTensorBase<float32>"
    
    // 对于1D张量，视为1xN矩阵
    if (dim == 1) {
        // 创建一个2D视图 [1, shape[0]]
        YTensorBase mat2d;
        mat2d._shape = {1, _shape[0]};
        mat2d._stride = {0, _stride[0]};  // 第0维stride=0因为只有1行
        mat2d._offset = _offset;
        mat2d._data = _data;
        mat2d._element_size = _element_size;
        mat2d._dtype = _dtype;
        
        // 返回一个shape=[1]的YTensorBase，元素是YTensorBase类型
        // 由于YTensorBase不在类型注册表中，我们用特殊方式处理
        YTensorBase result;
        result._shape = {1};
        result._stride = {1};
        result._offset = 0;
        result._element_size = sizeof(YTensorBase);
        result._dtype = innerDtype;  // "YTensorBase<float32>"
        
        // 分配存储并放置构造YTensorBase元素
        result._data = std::shared_ptr<char[]>(
            new char[sizeof(YTensorBase)],
            [](char* p) {
                // 析构YTensorBase对象
                reinterpret_cast<YTensorBase*>(p)->~YTensorBase();
                delete[] p;
            }
        );
        new (result._data.get()) YTensorBase(mat2d);
        return result;
    }
    
    // 对于2D张量，返回shape=[1]的matView
    if (dim == 2) {
        YTensorBase result;
        result._shape = {1};
        result._stride = {1};
        result._offset = 0;
        result._element_size = sizeof(YTensorBase);
        result._dtype = innerDtype;  // "YTensorBase<float32>"
        
        result._data = std::shared_ptr<char[]>(
            new char[sizeof(YTensorBase)],
            [](char* p) {
                reinterpret_cast<YTensorBase*>(p)->~YTensorBase();
                delete[] p;
            }
        );
        // 直接复制当前张量作为2D视图
        new (result._data.get()) YTensorBase(*this);
        return result;
    }
    
    // 对于高维张量，前dim-2维是batch维度
    std::vector<int> batchShape(_shape.begin(), _shape.end() - 2);
    int matRows = _shape[dim - 2];
    int matCols = _shape[dim - 1];
    int matRowStride = _stride[dim - 2];
    int matColStride = _stride[dim - 1];
    
    // 计算batch数量
    size_t batchSize = 1;
    for (int s : batchShape) batchSize *= s;
    
    // 创建结果tensor
    YTensorBase result;
    result._shape = batchShape;
    result._element_size = sizeof(YTensorBase);
    result._dtype = innerDtype;  // "YTensorBase<float32>"
    
    // 计算stride（按连续存储）
    result._stride.resize(batchShape.size());
    if (!batchShape.empty()) {
        result._stride.back() = 1;
        for (int i = static_cast<int>(batchShape.size()) - 2; i >= 0; --i) {
            result._stride[i] = result._stride[i + 1] * batchShape[i + 1];
        }
    }
    result._offset = 0;
    
    // 使用封装函数分配存储
    result._data = yt::kernel::makeSharedPlacementArray<YTensorBase>(batchSize);
    
    // 为每个batch创建2D视图
    YTensorBase* dataPtr = reinterpret_cast<YTensorBase*>(result._data.get());
    
    // 计算原始tensor的batch stride（前dim-2维）
    std::vector<int> batchStride(_stride.begin(), _stride.end() - 2);
    
    yt::kernel::parallelFor(0, static_cast<int>(batchSize), [&](int batchIdx) {
        // 计算坐标
        std::vector<int> coord(batchShape.size());
        int remaining = batchIdx;
        for (int i = static_cast<int>(batchShape.size()) - 1; i >= 0; --i) {
            coord[i] = remaining % batchShape[i];
            remaining /= batchShape[i];
        }
        
        // 计算偏移
        int batchOffset = 0;
        for (size_t i = 0; i < batchShape.size(); ++i) {
            batchOffset += coord[i] * batchStride[i];
        }
        
        // 创建2D视图
        YTensorBase mat2d;
        mat2d._shape = {matRows, matCols};
        mat2d._stride = {matRowStride, matColStride};
        mat2d._offset = _offset + batchOffset;
        mat2d._data = _data;  // 共享数据
        mat2d._element_size = _element_size;
        mat2d._dtype = _dtype;
        
        // placement new
        new (&dataPtr[batchIdx]) YTensorBase(std::move(mat2d));
    });
    
    return result;
}

// ======================== matmul ========================

// 模板化的naive matmul实现（零后端）- 使用broadcastInplace简化
template<typename DType>
inline YTensorBase matmul_naive_impl(const YTensorBase& self, const YTensorBase& other) {
    // 获取matView
    auto thisMatView = self.matView();
    auto otherMatView = other.matView();
    
    // 获取矩阵维度
    int ah = (self.ndim() >= 2) ? self.shape(self.ndim() - 2) : 1;
    int aw = self.shape(self.ndim() - 1);
    int bw = other.shape(other.ndim() - 1);
    
    // 计算输出形状
    std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
    int opBatchDim = std::max(std::max(0, self.ndim() - 2), std::max(0, other.ndim() - 2));
    std::vector<int> opShape;
    for (int i = static_cast<int>(opBatchShape.size()) - opBatchDim; i < static_cast<int>(opBatchShape.size()); ++i) {
        opShape.push_back(opBatchShape[i]);
    }
    opShape.push_back(ah);
    opShape.push_back(bw);
    
    // 创建输出tensor
    YTensorBase op(opShape, self.dtype());
    auto opMatView = op.matView();
    
    // 使用broadcastInplace处理广播和并行化
    opMatView.broadcastInplace([ah, aw, bw](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
        for (int i = 0; i < ah; ++i) {
            for (int j = 0; j < bw; ++j) {
                DType sum{};
                for (int k = 0; k < aw; ++k) {
                    sum += A.template at<DType>({i, k}) * B.template at<DType>({k, j});
                }
                C.template at<DType>({i, j}) = sum;
            }
        }
    }, thisMatView, otherMatView);
    
    return op;
}

inline YTensorBase YTensorBase::matmul(const YTensorBase& other, 
                                       yt::infos::MatmulBackend backend) const {
    // 验证维度
    if (ndim() < 1 || other.ndim() < 1) {
        throw std::runtime_error("[YTensorBase::matmul] Both tensors must have at least 1 dimension");
    }
    
    // 检查类型匹配
    if (_dtype != other._dtype) {
        throw std::runtime_error("[YTensorBase::matmul] dtype mismatch: " + _dtype + " vs " + other._dtype);
    }
    
    // 获取矩阵维度
    [[maybe_unused]] int thisRows = (ndim() >= 2) ? _shape[ndim() - 2] : 1;
    int thisCols = _shape[ndim() - 1];
    int otherRows = (other.ndim() >= 2) ? other._shape[other.ndim() - 2] : 1;
    [[maybe_unused]] int otherCols = other._shape[other.ndim() - 1];
    
    if (thisCols != otherRows) {
        throw std::runtime_error("[YTensorBase::matmul] Inner dimensions mismatch: " + 
            std::to_string(thisCols) + " vs " + std::to_string(otherRows));
    }
    
    // 检查是否是需要转换的扩展浮点类型
    bool needsCastToFloat32 = (_dtype == "bfloat16" || _dtype == "float16" || 
                               _dtype == "float8_e5m2" || _dtype == "float8_e4m3" || 
                               _dtype == "float8_e8m0" || _dtype == "float8_ue8m0");
    
    if (needsCastToFloat32) {
        // 转换为 float32 执行 matmul，再转回原类型
        YTensorBase thisF32 = this->cast("float32");
        YTensorBase otherF32 = other.cast("float32");
        YTensorBase resultF32 = thisF32.matmul(otherF32, backend);
        return resultF32.cast(_dtype);
    }
    
    // 根据backend选择实现
    switch (backend) {
        case yt::infos::MatmulBackend::Naive:
            return matmul_naive_backend(other);
            
        case yt::infos::MatmulBackend::AVX2:
#if YT_USE_AVX2
            if (_dtype == "float32") {
                return matmul_avx2_backend(other);
            }
#endif
            // AVX2不可用或非float32时fallthrough到Eigen
            [[fallthrough]];
            
        case yt::infos::MatmulBackend::Eigen:
#if YT_USE_EIGEN
            return matmul_eigen_backend(other);
#else
            // Eigen不可用时fallthrough到Naive
            [[fallthrough]];
#endif
            
        default:
            return matmul_naive_backend(other);
    }
}

inline YTensorBase YTensorBase::matmul_naive_backend(const YTensorBase& other) const {
    YTensorBase result(_shape, _dtype);
    yt::kernel::dispatchOrThrow<yt::types::AllNumericTypes>(_dtype,
        [&]<typename DType>() {
            if constexpr (yt::concepts::HAVE_MUL<DType>) {
                result = matmul_naive_impl<DType>(*this, other);
            } else {
                throw std::runtime_error("[YTensorBase::matmul_naive_backend] Type does not support multiplication");
            }
        }, "matmul_naive_backend");
    return result;
}


inline void YTensorBase::throwOperatorNotSupport(const std::string& typeName, const std::string& opName) {
    throw std::runtime_error("[YTensorBase] Operator " + opName + " not support for type " + typeName);
}

inline void YTensorBase::throwShapeNotMatch(const std::string& opName, const std::vector<int>& otherShape) const {
    std::string thisShapeStr = "[";
    for (size_t i = 0; i < _shape.size(); ++i) {
        thisShapeStr += std::to_string(_shape[i]);
        if (i + 1 < _shape.size()) thisShapeStr += ", ";
    }
    thisShapeStr += "]";
    
    std::string otherShapeStr = "[";
    for (size_t i = 0; i < otherShape.size(); ++i) {
        otherShapeStr += std::to_string(otherShape[i]);
        if (i + 1 < otherShape.size()) otherShapeStr += ", ";
    }
    otherShapeStr += "]";
    
    throw std::runtime_error("[YTensorBase::" + opName + "] Shape mismatch: " + thisShapeStr + " vs " + otherShapeStr);
}

// ======================== Eigen Support ========================
#if YT_USE_EIGEN

// Eigen类型别名
template<typename T>
using EigenStridedMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 
                                   0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

template<typename T>
using EigenConstStridedMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 
                                        0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

/// @brief 从YTensorBase创建Eigen Map
template<typename T>
inline auto toEigenMap(YTensorBase& mat) {
    return EigenStridedMap<T>(mat.template data<T>(), mat.shape(0), mat.shape(1), 
                              Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(mat.stride_(0), mat.stride_(1)));
}

template<typename T>
inline auto toEigenConstMap(const YTensorBase& mat) {
    return EigenConstStridedMap<T>(mat.template data<T>(), mat.shape(0), mat.shape(1), 
                                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(mat.stride_(0), mat.stride_(1)));
}

/// @brief Eigen后端主接口 - 简化版，借鉴YTensor设计
inline YTensorBase YTensorBase::matmul_eigen_backend(const YTensorBase& other) const {
    int selfDim = ndim(), otherDim = other.ndim();
    int ah = (selfDim >= 2) ? shape(selfDim - 2) : 1;
    int aw = shape(selfDim - 1), bw = other.shape(otherDim - 1);
    
    YTensorBase result;
    yt::kernel::dispatchOrThrow<yt::types::EigenNativeTypes>(_dtype,
        [&]<typename DType>() {
            // ==================== Fastpath检测 ====================
            // fastpath: [..., m, k] @ [k, n]，左张量除最后一维外连续
            if (selfDim > 2) {
                bool rightIs2D = (otherDim <= 2);
                if (!rightIs2D) {
                    rightIs2D = true;
                    for (int i = 0; i < otherDim - 2; ++i) {
                        if (other.shape(i) != 1) { rightIs2D = false; break; }
                    }
                }
                if (rightIs2D) {
                    int contiguousStart = isContiguousFrom(0, -1);
                    if (contiguousStart < selfDim - 1) {
                        // 使用fastpath
                        int outerSize = 1, innerRows = 1;
                        for (int i = 0; i < contiguousStart; ++i) outerSize *= shape(i);
                        for (int i = contiguousStart; i < selfDim - 1; ++i) innerRows *= shape(i);
                        
                        std::vector<int> opShape;
                        for (int i = 0; i < selfDim - 1; ++i) opShape.push_back(shape(i));
                        opShape.push_back(bw);
                        YTensorBase op(opShape, _dtype);
                        
                        // 右矩阵2D view
                        YTensorBase right2D;
                        right2D._shape = {aw, bw}; 
                        right2D._stride = {other.stride_(otherDim - 2), other.stride_(otherDim - 1)};
                        right2D._offset = other._offset; right2D._data = other._data;
                        right2D._element_size = sizeof(DType); right2D._dtype = _dtype;
                        
                        int innerStride = (contiguousStart == 0) ? aw : stride_(contiguousStart);
                        int opInnerStride = (contiguousStart == 0) ? bw : op.stride_(contiguousStart);
                        auto eigenB = toEigenConstMap<DType>(right2D);
                        
                        for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                            int leftOffset = 0, opOffset = 0;
                            if (contiguousStart > 0) {
                                int idx = outerIdx;
                                for (int i = contiguousStart - 1; i >= 0; --i) {
                                    int coord = idx % shape(i); idx /= shape(i);
                                    leftOffset += coord * stride_(i);
                                    opOffset += coord * op.stride_(i);
                                }
                            }
                            // 直接创建Eigen Map
                            EigenConstStridedMap<DType> eigenA(this->data<DType>() + leftOffset, innerRows, aw,
                                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(innerStride, stride_(selfDim - 1)));
                            EigenStridedMap<DType> eigenC(op.data<DType>() + opOffset, innerRows, bw,
                                Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(opInnerStride, 1));
                            eigenC.noalias() = eigenA * eigenB;
                        }
                        result = op;
                        return;
                    }
                }
            }
            
            // ==================== 普通路径（使用broadcastInplace简化） ====================
            auto thisMatView = matView();
            auto otherMatView = other.matView();
            
            // 计算输出形状
            std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
            int opBatchDim = std::max(std::max(0, selfDim - 2), std::max(0, otherDim - 2));
            std::vector<int> opShape;
            for (int i = static_cast<int>(opBatchShape.size()) - opBatchDim; i < static_cast<int>(opBatchShape.size()); ++i) {
                opShape.push_back(opBatchShape[i]);
            }
            opShape.push_back(ah); opShape.push_back(bw);
            
            YTensorBase op(opShape, _dtype);
            auto opMatView = op.matView();
            
            // 使用broadcastInplace处理广播和并行化，捕获DType进行Eigen操作
            opMatView.broadcastInplace([](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
                toEigenMap<DType>(C).noalias() = toEigenConstMap<DType>(A) * toEigenConstMap<DType>(B);
            }, thisMatView, otherMatView);
            
            result = op;
        }, "matmul_eigen_backend");
    return result;
}

/// @brief 通用Eigen单元操作 - 对每个2D矩阵应用func
template<typename Func>
inline YTensorBase YTensorBase::applyEigenOp(Func&& func, const std::string& opName) const {
    if (ndim() < 2) throw std::runtime_error("[YTensorBase::" + opName + "] requires at least 2 dimensions");
    
    // 扩展浮点类型转换为float32处理
    if (_dtype == "bfloat16" || _dtype == "float16" || _dtype.find("float8") != std::string::npos) {
        return this->cast("float32").applyEigenOp(std::forward<Func>(func), opName).cast(_dtype);
    }
    
    YTensorBase result;
    yt::kernel::dispatchOrThrow<yt::types::EigenNativeTypes>(_dtype, [&]<typename T>() {
        auto thisMatView = matView();
        YTensorBase* mats = thisMatView.template data<YTensorBase>();
        
        // 推断输出形状
        auto result0 = func(toEigenConstMap<T>(mats[0]));
        int outRows = static_cast<int>(result0.rows()), outCols = static_cast<int>(result0.cols());
        
        std::vector<int> outShape(_shape.begin(), _shape.end() - 2);
        outShape.push_back(outRows); outShape.push_back(outCols);
        
        YTensorBase op(outShape, _dtype);
        auto opMatView = op.matView();
        YTensorBase* opMats = opMatView.template data<YTensorBase>();
        
        yt::kernel::parallelFor(0, static_cast<int>(thisMatView.size()), [&](int i) {
            toEigenMap<T>(opMats[i]) = func(toEigenConstMap<T>(mats[i]));
        }, static_cast<double>(shape(-2) * shape(-1)));
        
        result = op;
    }, opName);
    return result;
}

/// @brief 通用Eigen二元操作 - 支持广播（使用broadcastInplace简化）
template<typename Func>
inline YTensorBase YTensorBase::applyEigenBinaryOp(const YTensorBase& other, Func&& func, const std::string& opName) const {
    if (ndim() < 2 || other.ndim() < 2) throw std::runtime_error("[YTensorBase::" + opName + "] requires at least 2 dimensions");
    if (_dtype != other._dtype) throw std::runtime_error("[YTensorBase::" + opName + "] dtype mismatch");
    
    // 扩展浮点类型转换为float32处理
    if (_dtype == "bfloat16" || _dtype == "float16" || _dtype.find("float8") != std::string::npos) {
        return this->cast("float32").applyEigenBinaryOp(other.cast("float32"), std::forward<Func>(func), opName).cast(_dtype);
    }
    
    YTensorBase result;
    yt::kernel::dispatchOrThrow<yt::types::EigenNativeTypes>(_dtype, [&]<typename T>() {
        auto thisMatView = matView(), otherMatView = other.matView();
        YTensorBase *thisMats = thisMatView.template data<YTensorBase>();
        YTensorBase *otherMats = otherMatView.template data<YTensorBase>();
        
        std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        
        // 推断输出形状
        auto result0 = func(toEigenConstMap<T>(thisMats[0]), toEigenConstMap<T>(otherMats[0]));
        int outRows = static_cast<int>(result0.rows()), outCols = static_cast<int>(result0.cols());
        
        int opBatchDim = std::max(std::max(0, ndim() - 2), std::max(0, other.ndim() - 2));
        std::vector<int> opShape;
        for (int i = static_cast<int>(opBatchShape.size()) - opBatchDim; i < static_cast<int>(opBatchShape.size()); ++i) {
            opShape.push_back(opBatchShape[i]);
        }
        opShape.push_back(outRows); opShape.push_back(outCols);
        
        YTensorBase op(opShape, _dtype);
        auto opMatView = op.matView();
        
        // 使用broadcastInplace简化广播和循环
        opMatView.broadcastInplace([&func](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
            toEigenMap<T>(C) = func(toEigenConstMap<T>(A), toEigenConstMap<T>(B));
        }, thisMatView, otherMatView);
        
        result = op;
    }, opName);
    return result;
}

#endif // YT_USE_EIGEN

#if YT_USE_AVX2
inline YTensorBase YTensorBase::matmul_avx2_backend(const YTensorBase& other) const {
    if (_dtype != "float32") {
#if YT_USE_EIGEN
        return matmul_eigen_backend(other);
#else
        YTensorBase result(_shape, _dtype);
        yt::kernel::dispatchOrThrow<yt::types::AllNumericTypes>(_dtype,
            [&]<typename DType>() {
                result = matmul_naive_impl<DType>(*this, other);
            }, "matmul_avx2_backend");
        return result;
#endif
    }

    int selfDim = ndim();
    int otherDim = other.ndim();
    int aw = shape(selfDim - 1);
    int bw = other.shape(otherDim - 1);
    int ah = (selfDim >= 2) ? shape(selfDim - 2) : 1;

    // ==================== Fastpath检测 ====================
    if (selfDim > 2) {
        bool rightIs2D = (otherDim <= 2);
        if (!rightIs2D) {
            rightIs2D = true;
            for (int i = 0; i < otherDim - 2; ++i) {
                if (other.shape(i) != 1) { rightIs2D = false; break; }
            }
        }
        if (rightIs2D) {
            int contiguousStart = isContiguousFrom(0, -1);
            if (contiguousStart < selfDim - 1) {
                // 可以使用fastpath
                int outerSize = 1, innerRows = 1;
                for (int i = 0; i < contiguousStart; ++i) outerSize *= shape(i);
                for (int i = contiguousStart; i < selfDim - 1; ++i) innerRows *= shape(i);
                
                // 准备输出
                std::vector<int> opShape;
                for (int i = 0; i < selfDim - 1; ++i) opShape.push_back(shape(i));
                opShape.push_back(bw);
                YTensorBase op(opShape, "float32");
                
                // 右矩阵2D view
                YTensorBase right2D;
                right2D._shape = {aw, bw}; right2D._stride = {other.stride_(otherDim - 2), other.stride_(otherDim - 1)};
                right2D._offset = other._offset; right2D._data = other._data;
                right2D._element_size = sizeof(float); right2D._dtype = "float32";
                
                int innerStride = (contiguousStart == 0) ? aw : stride_(contiguousStart);
                int opInnerStride = (contiguousStart == 0) ? bw : op.stride_(contiguousStart);
                
                for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                    int leftOffset = 0, opOffset = 0;
                    if (contiguousStart > 0) {
                        int idx = outerIdx;
                        for (int i = contiguousStart - 1; i >= 0; --i) {
                            int coord = idx % shape(i); idx /= shape(i);
                            leftOffset += coord * stride_(i);
                            opOffset += coord * op.stride_(i);
                        }
                    }
                    YTensorBase leftFlat, opFlat;
                    leftFlat._shape = {innerRows, aw}; leftFlat._stride = {innerStride, stride_(selfDim - 1)};
                    leftFlat._offset = _offset + leftOffset; leftFlat._data = _data;
                    leftFlat._element_size = sizeof(float); leftFlat._dtype = "float32";
                    opFlat._shape = {innerRows, bw}; opFlat._stride = {opInnerStride, 1};
                    opFlat._offset = opOffset; opFlat._data = op._data;
                    opFlat._element_size = sizeof(float); opFlat._dtype = "float32";
                    
                    yt::kernel::gemm::matmul(
                        leftFlat.data<float>(), right2D.data<float>(), opFlat.data<float>(),
                        innerRows, bw, aw,
                        static_cast<int64_t>(leftFlat.stride_(0)), static_cast<int64_t>(leftFlat.stride_(1)),
                        static_cast<int64_t>(right2D.stride_(0)), static_cast<int64_t>(right2D.stride_(1)),
                        static_cast<int64_t>(opFlat.stride_(0)), static_cast<int64_t>(opFlat.stride_(1)));
                }
                return op;
            }
        }
    }

    // ==================== 普通路径（使用broadcastInplace简化） ====================
    auto thisMatView = matView();
    auto otherMatView = other.matView();
    std::vector<int> opBatchShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
    int opBatchDim = std::max(std::max(0, selfDim - 2), std::max(0, otherDim - 2));
    
    std::vector<int> opShape;
    int skipDims = static_cast<int>(opBatchShape.size()) - opBatchDim;
    for (int i = skipDims; i < static_cast<int>(opBatchShape.size()); ++i) opShape.push_back(opBatchShape[i]);
    opShape.push_back(ah);
    opShape.push_back(bw);
    
    YTensorBase op(opShape, "float32");
    auto opMatView = op.matView();
    
    // 使用broadcastInplace处理广播和并行化
    opMatView.broadcastInplace([](YTensorBase& C, const YTensorBase& A, const YTensorBase& B) {
        yt::kernel::gemm::matmul(
            A.data<float>(), B.data<float>(), C.data<float>(),
            A.shape(0), B.shape(1), A.shape(1),
            static_cast<int64_t>(A.stride_(0)), static_cast<int64_t>(A.stride_(1)),
            static_cast<int64_t>(B.stride_(0)), static_cast<int64_t>(B.stride_(1)),
            static_cast<int64_t>(C.stride_(0)), static_cast<int64_t>(C.stride_(1))
        );
    }, thisMatView, otherMatView);
    
    return op;
}
#endif

} // namespace yt