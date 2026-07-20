#pragma once
/***************
 * file: strided/view.inl
 * purpose: strided layout 的 view 类职责实现。
 ***************/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "../../include/utils/memory_utils.hpp"

namespace yt::strided {

// ==================== metadata validation and privileged access ====================

// 为shape构造row-major连续stride，并验证所有逻辑stride可由int metadata表示。
inline std::vector<int> contiguousStrideForShape(
    const std::vector<int>& shape, const std::string& context
) {
    std::vector<int> stride(shape.size());
    size_t nextStride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] < 0) throw std::invalid_argument(context + ": negative shape extent");
        stride[i] = yt::utils::checkedIntSize(nextStride, context + " stride");
        if (shape[i] != 0 &&
            nextStride > static_cast<size_t>(std::numeric_limits<int>::max()) /
                             static_cast<size_t>(shape[i])) {
            throw std::overflow_error(context + ": logical size exceeds int indexing range");
        }
        nextStride *= static_cast<size_t>(shape[i]);
    }
    return stride;
}

// 在发布view前验证rank、逻辑索引范围和完整可达物理span。
// 负stride轴分别贡献min/max端点，因此reverse view也按实际可达storage范围校验。
inline void validateViewMetadata(
    const std::vector<int>& shape, const std::vector<int>& stride, int offset, const YMemory& memory,
    size_t elementSize, const std::string& context
) {
    if (shape.size() != stride.size()) throw std::invalid_argument(context + ": shape/stride rank mismatch");
    if (elementSize == 0) throw std::invalid_argument(context + ": element size must be positive");
    bool empty = false;
    size_t logicalSize = 1;
    for (int extent : shape) {
        if (extent < 0) throw std::invalid_argument(context + ": negative shape extent");
        empty = empty || extent == 0;
        if (extent != 0 && logicalSize > static_cast<size_t>(std::numeric_limits<int>::max()) / extent) {
            throw std::overflow_error(context + ": logical size exceeds int indexing range");
        }
        logicalSize *= static_cast<size_t>(extent);
    }
    // 实际值不需要保存；这里只验证后续int-indexed逻辑坐标转换仍可表示。
    int64_t logicalStride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i > 0; --i) {
        logicalStride *= shape[i];
        if (logicalStride > std::numeric_limits<int>::max()) {
            throw std::overflow_error(context + ": logical stride exceeds int range");
        }
    }
    if (empty) return;  // empty view不会解引用storage，保留历史slice offset语义。
    if (!memory) throw std::out_of_range(context + ": non-empty view has no storage");

    // 每个轴只需选择delta的负端或正端，即可得到整个矩形view的物理包围区间。
    int64_t minIndex = offset;
    int64_t maxIndex = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
        int64_t delta = static_cast<int64_t>(shape[i] - 1) * stride[i];
        if (delta < 0) {
            if (minIndex < std::numeric_limits<int64_t>::min() - delta) {
                throw std::overflow_error(context + ": physical span overflow");
            }
            minIndex += delta;
        } else {
            if (maxIndex > std::numeric_limits<int64_t>::max() - delta) {
                throw std::overflow_error(context + ": physical span overflow");
            }
            maxIndex += delta;
        }
    }
    size_t storageElements = memory.nbytes() / elementSize;
    if (minIndex < 0 || maxIndex < minIndex || static_cast<uint64_t>(maxIndex) >= storageElements) {
        throw std::out_of_range(context + ": view physical span exceeds storage");
    }
}

// 集中管理YTensorBase的Strided metadata/storage特权访问，避免普通算法直接拼装半成品view。
struct BaseViewAccess {
    static const std::vector<int>& shapeOf(const YTensorBase& tensor) { return tensor.stridedShape(); }

    static const std::vector<int>& strideOf(const YTensorBase& tensor) { return tensor.stridedStride(); }

    static std::vector<int>& shapeOf(YTensorBase& tensor) { return tensor.stridedShape(); }

    static std::vector<int>& strideOf(YTensorBase& tensor) { return tensor.stridedStride(); }

    static int offsetOf(const YTensorBase& tensor) { return tensor.stridedOffset(); }

    static int& offsetOf(YTensorBase& tensor) { return tensor.stridedOffset(); }

    static const YMemory& memoryOf(const YTensorBase& tensor) { return tensor._memory; }

    static const std::string& dtypeOf(const YTensorBase& tensor) { return tensor._dtype; }

    static size_t elementSizeOf(const YTensorBase& tensor) { return tensor._element_size; }

    // 将shape/stride/offset/storage/dtype作为一个一致tuple校验并提交。
    static void setView(
        YTensorBase& tensor, const std::vector<int>& shape, const std::vector<int>& stride, int offset,
        const YMemory& memory, size_t elementSize, const std::string& dtype
    ) {
        validateViewMetadata(shape, stride, offset, memory, elementSize, "strided view");
        // 先构造完整replacement tuple，再提交layout/storage/dtype，保持view替换的强异常保证。
        YLayout replacementLayout;
        auto& replacementMeta = replacementLayout.as<YLayoutType::Strided>();
        replacementMeta.shape = shape;
        replacementMeta.stride = stride;
        replacementMeta.offset = offset;
        YMemory replacementMemory = memory;
        std::string replacementDtype = dtype;

        tensor._layout = std::move(replacementLayout);
        tensor._memory = std::move(replacementMemory);
        tensor._element_size = elementSize;
        tensor._dtype = std::move(replacementDtype);
    }

    // 创建metadata独立但共享values storage和dtype的同源view。
    static YTensorBase makeSiblingView(
        const YTensorBase& tensor, const std::vector<int>& shape, const std::vector<int>& stride, int offset
    ) {
        YTensorBase out;
        setView(out, shape, stride, offset, tensor._memory, tensor._element_size, tensor._dtype);
        return out;
    }
};

// ==================== shape, contiguity and overlap queries ====================

YT_IMPL_INLINE std::vector<int> shape(const YTensorBase& tensor) { return BaseViewAccess::shapeOf(tensor); }

YT_IMPL_INLINE int ndim(const YTensorBase& tensor) {
    if (BaseViewAccess::shapeOf(tensor).size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("strided::ndim: rank exceeds int range");
    }
    return static_cast<int>(BaseViewAccess::shapeOf(tensor).size());
}

YT_IMPL_INLINE int shape(const YTensorBase& tensor, int dim) {
    int rank = yt::strided::ndim(tensor);
    if (rank == 0) throw std::out_of_range("strided::shape: cannot access a 0-dim tensor");
    return BaseViewAccess::shapeOf(tensor)[(dim % rank + rank) % rank];
}

YT_IMPL_INLINE std::vector<int> logicalStride(const YTensorBase& tensor) {
    const auto& tensorShape = BaseViewAccess::shapeOf(tensor);
    std::vector<int> stride(tensorShape.size(), 1);
    for (int i = static_cast<int>(tensorShape.size()) - 2; i >= 0; --i) {
        int64_t next = static_cast<int64_t>(stride[i + 1]) * tensorShape[i + 1];
        if (next > std::numeric_limits<int>::max()) {
            throw std::overflow_error("strided::logicalStride: stride exceeds int range");
        }
        stride[i] = static_cast<int>(next);
    }
    return stride;
}

YT_IMPL_INLINE std::vector<int> physicalStride(const YTensorBase& tensor) {
    return BaseViewAccess::strideOf(tensor);
}

YT_IMPL_INLINE int logicalStride(const YTensorBase& tensor, int dim) {
    int rank = yt::strided::ndim(tensor);
    if (rank == 0) throw std::out_of_range("strided::logicalStride: cannot access a 0-dim tensor");
    return yt::strided::logicalStride(tensor)[(dim % rank + rank) % rank];
}

YT_IMPL_INLINE int physicalStride(const YTensorBase& tensor, int dim) {
    int rank = yt::strided::ndim(tensor);
    if (rank == 0) throw std::out_of_range("strided::physicalStride: cannot access a 0-dim tensor");
    return BaseViewAccess::strideOf(tensor)[(dim % rank + rank) % rank];
}

YT_IMPL_INLINE size_t size(const YTensorBase& tensor) {
    size_t total = 1;
    for (int extent : BaseViewAccess::shapeOf(tensor)) {
        if (extent < 0) throw std::invalid_argument("strided::size: negative shape extent");
        if (extent != 0 && total > std::numeric_limits<size_t>::max() / static_cast<size_t>(extent)) {
            throw std::overflow_error("strided::size: element count overflow");
        }
        total *= static_cast<size_t>(extent);
    }
    return total;
}

YT_IMPL_INLINE bool isContiguous(const YTensorBase& tensor, int fromDim, int toDim) {
    int rank = yt::strided::ndim(tensor);
    if (rank == 0) return true;
    if (yt::strided::size(tensor) == 0) return true;
    if (BaseViewAccess::memoryOf(tensor) == nullptr) return false;
    // 区间为[fromDim, toDim)；默认toDim=-1映射到rank，singleton轴忽略stride差异。
    fromDim = (fromDim % rank + rank) % rank;
    toDim = toDim < 0 ? rank + toDim + 1 : (toDim % rank + rank) % rank;
    if (fromDim >= toDim) return true;
    auto expected = yt::strided::logicalStride(tensor);
    for (int i = fromDim; i < toDim; ++i) {
        if (expected[i] != BaseViewAccess::strideOf(tensor)[i] && BaseViewAccess::shapeOf(tensor)[i] > 1)
            return false;
    }
    return true;
}

YT_IMPL_INLINE int isContiguousFrom(const YTensorBase& tensor, int fromDim, int toDim) {
    int rank = yt::strided::ndim(tensor);
    if (rank == 0) return 0;
    if (yt::strided::size(tensor) == 0) return 0;
    if (BaseViewAccess::memoryOf(tensor) == nullptr) return rank;
    // 从尾部反查第一个断点，返回其后一维作为可整块访问的连续suffix起点。
    fromDim = (fromDim % rank + rank) % rank;
    toDim = toDim < 0 ? rank + toDim + 1 : (toDim % rank + rank) % rank;
    if (fromDim >= toDim) return fromDim;
    auto expected = yt::strided::logicalStride(tensor);
    for (int i = toDim - 1; i >= fromDim; --i) {
        if (expected[i] != BaseViewAccess::strideOf(tensor)[i] && BaseViewAccess::shapeOf(tensor)[i] > 1)
            return i + 1;
    }
    return fromDim;
}

YT_IMPL_INLINE bool isDisjoint(const YTensorBase& tensor) {
    if (BaseViewAccess::memoryOf(tensor) == nullptr) return false;
    int rank = yt::strided::ndim(tensor);
    if (rank <= 1) return !(rank == 1 && shape(tensor, 0) > 1 && physicalStride(tensor, 0) == 0);
    // 按绝对stride从小到大扩张已占span；下一轴stride小于span即存在地址重叠。
    // overflow时保守返回false，调用方会退回串行写入而不是冒险并发。
    std::vector<std::pair<uint64_t, int>> dimensions;
    for (int i = 0; i < rank; ++i) {
        if (shape(tensor, i) > 1) {
            const int64_t stride = physicalStride(tensor, i);
            const uint64_t magnitude = static_cast<uint64_t>(stride < 0 ? -stride : stride);
            dimensions.emplace_back(magnitude, i);
        }
    }
    std::sort(dimensions.begin(), dimensions.end());
    uint64_t span = 1;
    for (const auto& [stride, dim] : dimensions) {
        if (stride < span) return false;
        const uint64_t extentDelta = static_cast<uint64_t>(shape(tensor, dim) - 1);
        if (stride != 0 && extentDelta > (std::numeric_limits<uint64_t>::max() - span) / stride) {
            return false;
        }
        span += extentDelta * stride;
    }
    return true;
}

YT_IMPL_INLINE bool physicalSpansOverlap(const YTensorBase& left, const YTensorBase& right) {
    if (!left.isStrided() || !right.isStrided()) {
        throw std::runtime_error("strided::physicalSpansOverlap: layout not implemented");
    }
    if (size(left) == 0 || size(right) == 0 || left.rawData() == nullptr || right.rawData() == nullptr) {
        return false;
    }

    // 这是保守的byte bounding-span测试，不枚举稀疏stride实际触及的每个地址。
    // false positive只会触发额外快照；不能产生漏判导致alias写入错误。
    auto byteSpan = [](const YTensorBase& tensor) {
        int64_t minOffset = 0;
        int64_t maxOffset = 0;
        const auto tensorShape = shape(tensor);
        const auto tensorStride = physicalStride(tensor);
        for (size_t i = 0; i < tensorShape.size(); ++i) {
            const int64_t delta = static_cast<int64_t>(tensorShape[i] - 1) * tensorStride[i];
            if (delta < 0) {
                minOffset += delta;
            } else {
                maxOffset += delta;
            }
        }

        const auto elementSize = static_cast<int64_t>(tensor.elementSize());
        const char* base = tensor.rawData();
        return std::pair<const char*, const char*>{
            base + minOffset * elementSize, base + maxOffset * elementSize + elementSize
        };
    };

    const auto leftSpan = byteSpan(left);
    const auto rightSpan = byteSpan(right);
    // std::less为不同allocation的指针提供可用全序，避免直接关系比较的限制。
    const std::less<const char*> before;
    return before(leftSpan.first, rightSpan.second) && before(rightSpan.first, leftSpan.second);
}

// typed facade的特权view入口，复用runtime tuple校验并保留编译期T/dim。
template <typename T, int dim>
struct ViewTensorAccess {
    static YMemory& memoryOf(YTensor<T, dim>& tensor) { return tensor._memory; }

    static const YMemory& memoryOf(const YTensor<T, dim>& tensor) { return tensor._memory; }

    static int offsetOf(const YTensor<T, dim>& tensor) { return tensor.stridedOffset(); }

    static void setView(
        YTensor<T, dim>& tensor, const std::vector<int>& shape, const std::vector<int>& stride, int offset,
        const YMemory& memory, const std::string& dtype = yt::type::getTypeName<T>()
    ) {
        validateViewMetadata(shape, stride, offset, memory, sizeof(T), "typed strided view");
        BaseViewAccess::setView(
            static_cast<YTensorBase&>(tensor), shape, stride, offset, memory, sizeof(T), dtype
        );
    }
};

// 将batch逻辑坐标映射为包含base offset的绝对storage元素偏移。
template <typename T, int dim>
YT_IMPL_INLINE int typedBatchStorageOffset(
    const YTensor<T, dim>& tensor, const std::vector<int>& coordinate, const std::string& context
) {
    if (coordinate.size() > static_cast<size_t>(dim)) {
        throw std::invalid_argument(context + ": coordinate rank exceeds tensor rank");
    }
    const auto tensorShape = tensor.shape();
    const auto tensorStride = tensor.stride_();
    int64_t offset = ViewTensorAccess<T, dim>::offsetOf(tensor);
    for (size_t i = 0; i < coordinate.size(); ++i) {
        if (coordinate[i] < 0 || coordinate[i] >= tensorShape[i]) {
            throw std::out_of_range(context + ": coordinate out of range");
        }
        offset += static_cast<int64_t>(coordinate[i]) * tensorStride[i];
    }
    if (offset < std::numeric_limits<int>::min() || offset > std::numeric_limits<int>::max()) {
        throw std::overflow_error(context + ": storage offset exceeds int range");
    }
    return static_cast<int>(offset);
}

// ==================== metadata-only view transforms ====================

YT_IMPL_INLINE YTensorBase
slice(const YTensorBase& tensor, int atDim, int start, int end, int step, bool autoFix) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::slice: layout not implemented");
    }
    int d = tensor.ndim();
    if (d == 0) {
        throw std::out_of_range("[yt::strided::slice] Cannot slice a 0-dim tensor.");
    }

    const auto& shape = BaseViewAccess::shapeOf(tensor);
    const auto& stride = BaseViewAccess::strideOf(tensor);
    if (step == 0) {
        throw std::invalid_argument("Step cannot be 0 in slice operation.");
    }
    atDim = (atDim % d + d) % d;
    // 空维度没有可归一化的循环索引，直接保留空view，避免对extent执行取模。
    if (shape[atDim] == 0) {
        return BaseViewAccess::makeSiblingView(tensor, shape, stride, BaseViewAccess::offsetOf(tensor));
    }
    if (step == std::numeric_limits<int>::min()) {
        throw std::overflow_error("slice: step magnitude is too large");
    }
    const int64_t extent = shape[atDim];
    auto normalize = [extent](int64_t value) {
        return static_cast<int>((value % extent + extent) % extent);
    };
    start = normalize(start);
    int last = normalize(static_cast<int64_t>(end) - 1);
    // 保留项目历史循环索引语义：autoFix交换反向区间后收缩端点，而不是Python slice规则。
    if (autoFix && last < start) {
        std::swap(start, last);
        last--;
        start++;
    }

    std::vector<int> newShape = shape;
    std::vector<int> newStride = stride;
    int64_t newOffset = BaseViewAccess::offsetOf(tensor);
    // 正负step共享同一inclusive-last长度公式，起点则按方向选择start或last。
    int64_t slicedExtent =
        (static_cast<int64_t>(last) - start) / std::abs(static_cast<int64_t>(step)) + 1;
    newShape[atDim] = static_cast<int>(std::max<int64_t>(0, slicedExtent));
    // metadata仍使用int；先在宽类型中计算，确认可表示后再写回。
    int64_t steppedStride = static_cast<int64_t>(step) * stride[atDim];
    if (steppedStride < std::numeric_limits<int>::min() || steppedStride > std::numeric_limits<int>::max()) {
        throw std::overflow_error("slice: stride overflow");
    }
    newStride[atDim] = static_cast<int>(steppedStride);
    if (step > 0) {
        newOffset += static_cast<int64_t>(start) * stride[atDim];
    } else if (step < 0) {
        newOffset += static_cast<int64_t>(last) * stride[atDim];
    }
    if (newOffset < std::numeric_limits<int>::min() || newOffset > std::numeric_limits<int>::max()) {
        throw std::overflow_error("slice: offset overflow");
    }
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, static_cast<int>(newOffset));
}

YT_IMPL_INLINE YTensorBase& slice_(
    YTensorBase& tensor, int atDim, int start, int end, int step, bool autoFix
) {
    tensor = yt::strided::slice(tensor, atDim, start, end, step, autoFix);
    return tensor;
}

YT_IMPL_INLINE YTensorBase permute(const YTensorBase& tensor, const std::vector<int>& newOrder) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::permute: layout not implemented");
    }
    if (newOrder.size() != static_cast<size_t>(tensor.ndim())) {
        throw std::invalid_argument("permute: order size must match ndim");
    }

    int d = tensor.ndim();
    std::vector<int> newShape(d), newStride(d);
    std::vector<bool> seen(d, false);
    const auto& shape = BaseViewAccess::shapeOf(tensor);
    const auto& stride = BaseViewAccess::strideOf(tensor);
    // 每个axis先循环规范化再判重，因此超出rank的正负值按历史规则wrap。
    for (int i = 0; i < d; ++i) {
        int rotate = (newOrder[i] % d + d) % d;
        if (seen[rotate]) {
            throw std::invalid_argument("permute: order must contain each dimension exactly once");
        }
        seen[rotate] = true;
        newShape[i] = shape[rotate];
        newStride[i] = stride[rotate];
    }
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase& permute_(YTensorBase& tensor, const std::vector<int>& newOrder) {
    tensor = yt::strided::permute(tensor, newOrder);
    return tensor;
}

YT_IMPL_INLINE YTensorBase transpose(const YTensorBase& tensor, int dim0, int dim1) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::transpose: layout not implemented");
    }
    int d = tensor.ndim();
    if (d == 0) {
        throw std::out_of_range("[yt::strided::transpose] Cannot transpose a 0-dim tensor.");
    }
    dim0 = (dim0 % d + d) % d;
    dim1 = (dim1 % d + d) % d;
    // 同轴transpose仍返回metadata独立、storage共享的handle。
    if (dim0 == dim1) {
        return tensor;
    }

    std::vector<int> newShape = BaseViewAccess::shapeOf(tensor);
    std::vector<int> newStride = BaseViewAccess::strideOf(tensor);
    std::swap(newShape[dim0], newShape[dim1]);
    std::swap(newStride[dim0], newStride[dim1]);
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase view(const YTensorBase& tensor, const std::vector<int>& newShape) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::view: layout not implemented");
    }
    if (!tensor.isContiguous()) {
        throw std::runtime_error("\"view\" requires contiguous tensor.");
    }
    const std::vector<int> shape = tensor.autoShape(newShape);
    std::vector<int> stride = contiguousStrideForShape(shape, "strided::view");
    return BaseViewAccess::makeSiblingView(
        tensor, shape, stride, BaseViewAccess::offsetOf(tensor)
    );
}

YT_IMPL_INLINE YTensorBase reshape(const YTensorBase& tensor, const std::vector<int>& newShape) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::reshape: layout not implemented");
    }
    return yt::strided::view(yt::strided::contiguous(tensor), newShape);
}

YT_IMPL_INLINE YTensorBase unsqueeze(const YTensorBase& tensor, int dim) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::unsqueeze: layout not implemented");
    }
    int d = tensor.ndim();
    dim = ((dim % (d + 1)) + (d + 1)) % (d + 1);
    std::vector<int> newShape = BaseViewAccess::shapeOf(tensor);
    std::vector<int> newStride = BaseViewAccess::strideOf(tensor);
    newShape.insert(newShape.begin() + dim, 1);
    // 插入中间轴时用原stride*extent保持连续解释；末尾singleton的stride取1。
    int64_t newStrideWide =
        (dim < d) ? static_cast<int64_t>(newStride[dim]) * newShape[dim + 1] : 1;
    if (newStrideWide < std::numeric_limits<int>::min() || newStrideWide > std::numeric_limits<int>::max()) {
        throw std::overflow_error("strided::unsqueeze: stride overflow");
    }
    int newStrideValue = static_cast<int>(newStrideWide);
    newStride.insert(newStride.begin() + dim, newStrideValue);
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase& unsqueeze_(YTensorBase& tensor, int dim) {
    tensor = yt::strided::unsqueeze(tensor, dim);
    return tensor;
}

YT_IMPL_INLINE YTensorBase squeeze(const YTensorBase& tensor, int dim) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::squeeze: layout not implemented");
    }
    std::vector<int> newShape;
    std::vector<int> newStride;
    if (dim >= 0) {
        int d = tensor.ndim();
        if (d == 0) throw std::out_of_range("squeeze: cannot select an axis from a scalar");
        dim = (dim % d + d) % d;
        if (BaseViewAccess::shapeOf(tensor)[dim] != 1) {
            throw std::runtime_error("squeeze: can only squeeze dimensions of size 1");
        }
        newShape = BaseViewAccess::shapeOf(tensor);
        newStride = BaseViewAccess::strideOf(tensor);
        newShape.erase(newShape.begin() + dim);
        newStride.erase(newStride.begin() + dim);
    } else {
        // 任意负dim表示删除全部singleton轴；全部删除时保留历史rank-1 shape [1]。
        for (int i = 0; i < tensor.ndim(); ++i) {
            if (BaseViewAccess::shapeOf(tensor)[i] != 1) {
                newShape.push_back(BaseViewAccess::shapeOf(tensor)[i]);
                newStride.push_back(BaseViewAccess::strideOf(tensor)[i]);
            }
        }
        if (newShape.empty()) {
            newShape.push_back(1);
            newStride.push_back(1);
        }
    }
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase& squeeze_(YTensorBase& tensor, int dim) {
    tensor = yt::strided::squeeze(tensor, dim);
    return tensor;
}

YT_IMPL_INLINE YTensorBase repeat(const YTensorBase& tensor, const std::vector<int>& times) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::repeat: layout not implemented");
    }
    if (times.size() != static_cast<size_t>(tensor.ndim())) {
        throw std::invalid_argument("repeat: times size must match ndim");
    }
    std::vector<int> newShape = BaseViewAccess::shapeOf(tensor);
    std::vector<int> newStride = BaseViewAccess::strideOf(tensor);
    // repeat只扩展singleton轴且不物化数据；zero stride使所有重复坐标alias同一元素。
    // times<=1保持历史no-op语义，包括0和负值。
    for (int i = 0; i < tensor.ndim(); ++i) {
        if (times[i] <= 1) {
            continue;
        }
        if (BaseViewAccess::shapeOf(tensor)[i] != 1) {
            throw std::runtime_error("Can only repeat on dimensions of size 1.");
        }
        newShape[i] = times[i];
        newStride[i] = 0;
    }
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase& repeat_(YTensorBase& tensor, const std::vector<int>& times) {
    tensor = yt::strided::repeat(tensor, times);
    return tensor;
}

YT_IMPL_INLINE YTensorBase
unfold(const YTensorBase& tensor, int atDim, int kernel, int stride, int dilation) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::unfold: layout not implemented");
    }
    if (kernel <= 0 || stride <= 0 || dilation <= 0) {
        throw std::invalid_argument("Invalid kernel/stride/dilation");
    }
    int d = tensor.ndim();
    if (d == 0) {
        throw std::out_of_range("[yt::strided::unfold] Cannot unfold a 0-dim tensor.");
    }
    atDim = (atDim % d + d) % d;
    // kernel和dilation来自运行时，必须在合法性比较前避免int乘法溢出。
    int64_t effectiveKernel = static_cast<int64_t>(kernel - 1) * dilation + 1;
    if (effectiveKernel > std::numeric_limits<int>::max()) {
        throw std::overflow_error("unfold: effective kernel overflow");
    }
    if (BaseViewAccess::shapeOf(tensor)[atDim] < effectiveKernel) {
        throw std::invalid_argument("Dimension size is too small for unfold.");
    }
    int nums = static_cast<int>((BaseViewAccess::shapeOf(tensor)[atDim] - effectiveKernel) / stride + 1);
    std::vector<int> newShape = BaseViewAccess::shapeOf(tensor);
    std::vector<int> newStride = BaseViewAccess::strideOf(tensor);
    newShape[atDim] = nums;
    newShape.insert(newShape.begin() + atDim + 1, kernel);
    int64_t windowStride = static_cast<int64_t>(BaseViewAccess::strideOf(tensor)[atDim]) * stride;
    int64_t kernelStride = static_cast<int64_t>(BaseViewAccess::strideOf(tensor)[atDim]) * dilation;
    if (windowStride < std::numeric_limits<int>::min() || windowStride > std::numeric_limits<int>::max() ||
        kernelStride < std::numeric_limits<int>::min() || kernelStride > std::numeric_limits<int>::max()) {
        throw std::overflow_error("unfold: stride overflow");
    }
    newStride[atDim] = static_cast<int>(windowStride);
    newStride.insert(newStride.begin() + atDim + 1, static_cast<int>(kernelStride));
    return BaseViewAccess::makeSiblingView(tensor, newShape, newStride, BaseViewAccess::offsetOf(tensor));
}

YT_IMPL_INLINE YTensorBase& unfold_(YTensorBase& tensor, int atDim, int kernel, int stride, int dilation) {
    tensor = yt::strided::unfold(tensor, atDim, kernel, stride, dilation);
    return tensor;
}

YT_IMPL_INLINE YTensorBase mostContinuousView(const YTensorBase& tensor) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::mostContinuousView: layout not implemented");
    }
    if (BaseViewAccess::memoryOf(tensor) == nullptr) {
        YTensorBase out;
        BaseViewAccess::setView(
            out, tensor.shape(), contiguousStrideForShape(tensor.shape(), "strided::mostContinuousView"), 0,
            YMemory(), BaseViewAccess::elementSizeOf(tensor), BaseViewAccess::dtypeOf(tensor)
        );
        return out;
    }

    // 大绝对stride轴放在外层，尽量形成可整块处理的连续尾部。
    std::vector<std::pair<int, int>> mapper(tensor.ndim());
    for (int i = 0; i < tensor.ndim(); ++i) {
        mapper[i] = {BaseViewAccess::strideOf(tensor)[i], i};
    }
    std::sort(mapper.begin(), mapper.end(), [](const auto& a, const auto& b) {
        return std::abs(static_cast<int64_t>(a.first)) > std::abs(static_cast<int64_t>(b.first));
    });
    std::vector<int> perm(tensor.ndim());
    for (int i = 0; i < tensor.ndim(); ++i) {
        perm[i] = mapper[i].second;
    }
    YTensorBase out = yt::strided::permute(tensor, perm);
    std::vector<int> normalizedStride = BaseViewAccess::strideOf(out);
    int64_t normalizedOffset = BaseViewAccess::offsetOf(out);
    // 反向轴通过移动view原点并翻转stride符号归一化，不复制values storage。
    for (int i = 0; i < out.ndim(); ++i) {
        if (normalizedStride[i] < 0) {
            int64_t positiveStride = -static_cast<int64_t>(normalizedStride[i]);
            if (positiveStride > std::numeric_limits<int>::max()) {
                if (BaseViewAccess::shapeOf(out)[i] == 1) {
                    positiveStride = 1;
                } else {
                    throw std::overflow_error("mostContinuousView: stride magnitude overflow");
                }
            }
            normalizedStride[i] = static_cast<int>(positiveStride);
            normalizedOffset -=
                static_cast<int64_t>(BaseViewAccess::shapeOf(out)[i] - 1) * normalizedStride[i];
        }
    }
    if (normalizedOffset < std::numeric_limits<int>::min() ||
        normalizedOffset > std::numeric_limits<int>::max()) {
        throw std::overflow_error("mostContinuousView: offset overflow");
    }
    return BaseViewAccess::makeSiblingView(
        out, BaseViewAccess::shapeOf(out), normalizedStride, static_cast<int>(normalizedOffset)
    );
}

YT_IMPL_INLINE size_t logicalIndex(const YTensorBase& tensor, const std::vector<int>& position) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::logicalIndex: layout not implemented");
    }
    if (position.size() != static_cast<size_t>(tensor.ndim())) {
        throw std::invalid_argument("logicalIndex: position dimensions do not match ndim");
    }
    const auto logicalStride = tensor.stride();
    size_t index = 0;
    for (int i = 0; i < tensor.ndim(); ++i) {
        if (position[i] < 0 || position[i] >= tensor.shape(i)) {
            throw std::out_of_range("logicalIndex: coordinate out of range");
        }
        size_t term = static_cast<size_t>(position[i]) * static_cast<size_t>(logicalStride[i]);
        if (index > std::numeric_limits<size_t>::max() - term) {
            throw std::overflow_error("logicalIndex: index overflow");
        }
        index += term;
    }
    return index;
}

YT_IMPL_INLINE int relativeOffset(const YTensorBase& tensor, const std::vector<int>& position) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::relativeOffset: layout not implemented");
    }
    // rank-0 scalar的唯一合法逻辑坐标为空，其相对view起点偏移为0。
    if (position.empty()) return 0;
    if (position.size() != static_cast<size_t>(tensor.ndim())) {
        throw std::invalid_argument("relativeOffset: position dimensions do not match ndim");
    }
    int64_t index = 0;
    const auto stride = tensor.stride_();
    for (int i = 0; i < tensor.ndim(); ++i) {
        if (position[i] < 0 || position[i] >= tensor.shape(i)) {
            throw std::out_of_range("relativeOffset: coordinate out of range");
        }
        int64_t term = static_cast<int64_t>(position[i]) * stride[i];
        if ((term > 0 && index > std::numeric_limits<int64_t>::max() - term) ||
            (term < 0 && index < std::numeric_limits<int64_t>::min() - term)) {
            throw std::overflow_error("relativeOffset: index overflow");
        }
        index += term;
    }
    if (index < std::numeric_limits<int>::min() || index > std::numeric_limits<int>::max()) {
        throw std::overflow_error("relativeOffset: index exceeds int metadata range");
    }
    return static_cast<int>(index);
}

YT_IMPL_INLINE std::ptrdiff_t relativeOffset(const YTensorBase& tensor, size_t logicalIndex) {
    return relativeOffset(tensor, coordinate(tensor, logicalIndex));
}

YT_IMPL_INLINE int storageOffset(const YTensorBase& tensor, const std::vector<int>& position) {
    int64_t index = static_cast<int64_t>(BaseViewAccess::offsetOf(tensor)) + relativeOffset(tensor, position);
    if (index < std::numeric_limits<int>::min() || index > std::numeric_limits<int>::max()) {
        throw std::overflow_error("storageOffset: index exceeds int metadata range");
    }
    return static_cast<int>(index);
}

YT_IMPL_INLINE size_t physicalIndex(const YTensorBase& tensor, const std::vector<int>& position) {
    // 返回值相对当前data()指针，因此无法表示位于view origin之前的负相对偏移。
    int index = relativeOffset(tensor, position);
    if (index < 0) {
        throw std::out_of_range("physicalIndex: negative offset cannot be represented by toIndex_");
    }
    return static_cast<size_t>(index);
}

YT_IMPL_INLINE size_t physicalIndex(const YTensorBase& tensor, size_t logicalIndex) {
    return physicalIndex(tensor, coordinate(tensor, logicalIndex));
}

YT_IMPL_INLINE size_t storageIndex(const YTensorBase& tensor, size_t logicalIndex) {
    auto position = coordinate(tensor, logicalIndex);
    int index = storageOffset(tensor, position);
    size_t elementSize = BaseViewAccess::elementSizeOf(tensor);
    size_t storageElements = elementSize == 0 ? 0 : BaseViewAccess::memoryOf(tensor).nbytes() / elementSize;
    if (index < 0 || static_cast<size_t>(index) >= storageElements) {
        throw std::out_of_range("storageIndex: physical element index is outside storage");
    }
    return static_cast<size_t>(index);
}

YT_IMPL_INLINE std::vector<int> coordinate(const YTensorBase& tensor, size_t index) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::coordinate: layout not implemented");
    }
    if (index >= size(tensor)) {
        throw std::out_of_range("coordinate: logical index out of range");
    }
    std::vector<int> position(tensor.ndim());
    for (int i = tensor.ndim() - 1; i >= 0; --i) {
        position[i] = index % tensor.shape(i);
        index /= tensor.shape(i);
    }
    return position;
}

YT_IMPL_INLINE std::vector<int> autoShape(const YTensorBase& tensor, const std::vector<int>& requested) {
    std::vector<int> result(requested.size());
    int inferred = -1;
    for (size_t i = 0; i < requested.size(); ++i) {
        if (requested[i] < 0) {
            // 多个负维度是项目兼容规则：前面的负维恢复原shape，仅最后一个执行size推导。
            if (inferred != -1) {
                if (inferred >= ndim(tensor)) {
                    throw std::invalid_argument("auto shape cannot infer out of range");
                }
                result[inferred] = BaseViewAccess::shapeOf(tensor)[inferred];
            }
            inferred = static_cast<int>(i);
        } else {
            result[i] = requested[i];
        }
    }
    size_t knownSize = 1;
    for (int i = 0; i < static_cast<int>(result.size()); ++i) {
        if (i == inferred) continue;
        if (result[i] < 0) throw std::invalid_argument("auto shape produced a negative extent");
        if (result[i] != 0 && knownSize > std::numeric_limits<size_t>::max() / result[i]) {
            throw std::overflow_error("auto shape size overflow");
        }
        knownSize *= static_cast<size_t>(result[i]);
    }
    if (inferred != -1) {
        if (knownSize == 0 || size(tensor) % knownSize != 0) {
            throw std::invalid_argument("auto shape cannot infer shape");
        }
        size_t inferredSize = size(tensor) / knownSize;
        if (inferredSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("auto shape inferred extent overflow");
        }
        result[inferred] = static_cast<int>(inferredSize);
    } else if (knownSize != size(tensor)) {
        throw std::invalid_argument("auto shape must preserve the tensor element count");
    }
    return result;
}

YT_IMPL_INLINE std::vector<YTensorBase> split(
    const YTensorBase& tensor, const std::vector<int>& splitSizes, int axis
) {
    int rank = ndim(tensor);
    if (rank == 0) throw std::out_of_range("strided::split: cannot split a scalar");
    axis = (axis % rank + rank) % rank;
    int64_t total = 0;
    for (int part : splitSizes) {
        if (part <= 0) throw std::invalid_argument("strided::split: split size must be positive");
        total += part;
        if (total > shape(tensor, axis)) {
            throw std::invalid_argument("strided::split: sizes do not match axis");
        }
    }
    if (total != shape(tensor, axis)) throw std::invalid_argument("strided::split: sizes do not match axis");
    std::vector<YTensorBase> result;
    result.reserve(splitSizes.size());
    int offset = 0;
    for (int part : splitSizes) {
        result.push_back(yt::strided::slice(tensor, axis, offset, offset + part, 1, true));
        offset += part;
    }
    return result;
}

YT_IMPL_INLINE std::vector<YTensorBase> split(const YTensorBase& tensor, int parts, int axis) {
    int rank = ndim(tensor);
    if (rank == 0) throw std::out_of_range("strided::split: cannot split a scalar");
    axis = (axis % rank + rank) % rank;
    if (parts <= 0) throw std::invalid_argument("strided::split: parts must be positive");
    int axisSize = shape(tensor, axis);
    if (axisSize % parts != 0) throw std::invalid_argument("strided::split: axis is not divisible");
    return yt::strided::split(tensor, std::vector<int>(parts, axisSize / parts), axis);
}

// ==================== matrix wrapper views ====================

// 将最后两个轴包装成二维YTensorBase对象，batch轴成为外层wrapper tensor。
YT_IMPL_INLINE YTensorBase matView(const YTensorBase& tensor) {
    int dim = tensor.ndim();
    if (dim < 1) {
        throw std::runtime_error("[yt::strided::matView] Tensor must have at least 1 dimension");
    }

    // 外层storage拥有wrapper对象；wrapper仅共享输入values storage，因此必须禁止raw-byte clone。
    std::string innerDtype = yt::type::makeYTensorBaseDtype(tensor._dtype);

    // rank-1按历史matmul合同提升为单个1xN行矩阵；rank-2包装为长度1的batch。
    if (dim == 1) {
        YTensorBase mat2d;
        BaseViewAccess::setView(
            mat2d, {1, tensor.stridedShape()[0]}, {0, tensor.stridedStride()[0]},
            tensor.stridedOffset(), tensor._memory, tensor._element_size, tensor._dtype
        );

        YTensorBase result;
        auto resultMemory = yt::utils::makeSharedPlacementArray<YTensorBase>(
            1, [&](YTensorBase* dest, size_t) { new (dest) YTensorBase(mat2d); }
        );
        BaseViewAccess::setView(
            result, {1}, {1}, 0,
            YMemory(resultMemory, sizeof(YTensorBase), "cpu", false), sizeof(YTensorBase), innerDtype
        );
        return result;
    }

    if (dim == 2) {
        YTensorBase result;
        auto resultMemory = yt::utils::makeSharedPlacementArray<YTensorBase>(
            1, [&](YTensorBase* dest, size_t) { new (dest) YTensorBase(tensor); }
        );
        BaseViewAccess::setView(
            result, {1}, {1}, 0,
            YMemory(resultMemory, sizeof(YTensorBase), "cpu", false), sizeof(YTensorBase), innerDtype
        );
        return result;
    }

    std::vector<int> batchShape(tensor.stridedShape().begin(), tensor.stridedShape().end() - 2);
    int matRows = tensor.stridedShape()[dim - 2];
    int matCols = tensor.stridedShape()[dim - 1];
    int matRowStride = tensor.stridedStride()[dim - 2];
    int matColStride = tensor.stridedStride()[dim - 1];

    size_t batchSize = 1;
    for (int size : batchShape) {
        if (size != 0 && batchSize > static_cast<size_t>(std::numeric_limits<int>::max()) / size) {
            throw std::overflow_error("strided::matView: batch size exceeds int indexing range");
        }
        batchSize *= size;
    }

    std::vector<int> resultStride(batchShape.size());
    if (!batchShape.empty()) {
        resultStride.back() = 1;
        for (int i = static_cast<int>(batchShape.size()) - 2; i >= 0; --i) {
            int64_t next = static_cast<int64_t>(resultStride[i + 1]) * batchShape[i + 1];
            if (next > std::numeric_limits<int>::max()) {
                throw std::overflow_error("strided::matView: batch stride exceeds int range");
            }
            resultStride[i] = static_cast<int>(next);
        }
    }
    std::vector<int> batchStride(tensor.stridedStride().begin(), tensor.stridedStride().end() - 2);
    // wrapper对象在外层连续存放；每个wrapper共享原values storage并保留原matrix stride。
    auto resultMemory = yt::utils::makeSharedPlacementArray<YTensorBase>(batchSize, [&](YTensorBase* dest, size_t index) {
        // 将row-major batch linear index解码为原tensor前缀坐标，再计算matrix绝对offset。
        int batchIdx = static_cast<int>(index);
        std::vector<int> coord(batchShape.size());
        int remaining = batchIdx;
        for (int i = static_cast<int>(batchShape.size()) - 1; i >= 0; --i) {
            coord[i] = remaining % batchShape[i];
            remaining /= batchShape[i];
        }

        int64_t batchOffset = 0;
        for (size_t i = 0; i < batchShape.size(); ++i) {
            batchOffset += static_cast<int64_t>(coord[i]) * batchStride[i];
        }
        int64_t absoluteOffset = static_cast<int64_t>(tensor.stridedOffset()) + batchOffset;
        if (absoluteOffset < std::numeric_limits<int>::min() || absoluteOffset > std::numeric_limits<int>::max()) {
            throw std::overflow_error("strided::matView: matrix offset exceeds int range");
        }

        YTensorBase mat2d;
        BaseViewAccess::setView(
            mat2d, {matRows, matCols}, {matRowStride, matColStride}, static_cast<int>(absoluteOffset),
            tensor._memory, tensor._element_size, tensor._dtype
        );
        new (dest) YTensorBase(std::move(mat2d));
    });
    YTensorBase result;
    BaseViewAccess::setView(
        result, batchShape, resultStride, 0,
        YMemory(resultMemory, batchSize * sizeof(YTensorBase), "cpu", false), sizeof(YTensorBase), innerDtype
    );

    return result;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<YTensor<T, 2>, std::max(1, dim - 2)> matView(const YTensor<T, dim>& tensor) {
    static_assert(dim >= 1, "matView only support dim >= 1");
    // MatType是真实C++ wrapper对象并借用输入storage，外层storage负责逐对象析构而非字节释放。
    using MatType = YTensor<T, 2>;
    std::string scalarDtype = yt::type::getTypeName<T>();
    std::string matDtype = yt::type::makeYTensorDtype(scalarDtype, 2);
    auto tensorShape = tensor.shape();
    auto tensorStride = tensor.stride_();

    // 三个分支分别处理1-D提升、2-D单wrapper和N-D batch wrapper，均不复制scalar values。
    if constexpr (dim == 1) {
        MatType mat;
        ViewTensorAccess<T, 2>::setView(
            mat, {1, tensorShape[0]}, {0, tensorStride[0]}, ViewTensorAccess<T, dim>::offsetOf(tensor),
            ViewTensorAccess<T, dim>::memoryOf(tensor)
        );

        YTensor<MatType, 1> out;
        auto memory = yt::utils::makeSharedPlacement<MatType>(mat);
        ViewTensorAccess<MatType, 1>::setView(
            out, {1}, {0}, 0, YMemory(memory, sizeof(MatType), "cpu", false),
            yt::type::makeYTensorDtype(matDtype, 1)
        );
        return out;
    } else if constexpr (dim == 2) {
        YTensor<MatType, 1> out;
        MatType tensorCopy = tensor;
        auto memory = yt::utils::makeSharedPlacement<MatType>(tensorCopy);
        ViewTensorAccess<MatType, 1>::setView(
            out, {1}, {0}, 0, YMemory(memory, sizeof(MatType), "cpu", false),
            yt::type::makeYTensorDtype(matDtype, 1)
        );
        return out;
    } else {
        auto newShape = std::vector<int>(tensorShape.begin(), tensorShape.end() - 2);
        size_t batchSize = 1;
        for (int size : newShape) {
            if (size != 0 && batchSize > static_cast<size_t>(std::numeric_limits<int>::max()) / size) {
                throw std::overflow_error("strided::matView: batch size exceeds int indexing range");
            }
            batchSize *= size;
        }
        YTensor<MatType, std::max(1, dim - 2)> out;
        std::vector<int> outStride(newShape.size(), 0);
        if (!newShape.empty()) {
            outStride.back() = 1;
            for (int i = static_cast<int>(newShape.size()) - 2; i >= 0; --i) {
                int64_t next = static_cast<int64_t>(outStride[i + 1]) * newShape[i + 1];
                if (next > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("strided::matView: batch stride exceeds int range");
                }
                outStride[i] = static_cast<int>(next);
            }
        }
        auto outMemory = yt::utils::makeSharedPlacementArray<MatType>(batchSize, [&](MatType* dest, size_t index) {
            std::vector<int> coord(newShape.size());
            size_t remaining = index;
            for (int i = static_cast<int>(newShape.size()) - 1; i >= 0; --i) {
                coord[i] = static_cast<int>(remaining % static_cast<size_t>(newShape[i]));
                remaining /= static_cast<size_t>(newShape[i]);
            }
            MatType mat;
            ViewTensorAccess<T, 2>::setView(
                mat, {tensorShape[dim - 2], tensorShape[dim - 1]},
                {tensorStride[dim - 2], tensorStride[dim - 1]},
                typedBatchStorageOffset(tensor, coord, "strided::matView"),
                ViewTensorAccess<T, dim>::memoryOf(tensor)
            );
            new (dest) MatType(std::move(mat));
        });
        ViewTensorAccess<MatType, std::max(1, dim - 2)>::setView(
            out, newShape, outStride, 0, YMemory(outMemory, batchSize * sizeof(MatType), "cpu", false),
            yt::type::makeYTensorDtype(matDtype, std::max(1, dim - 2))
        );
        return out;
    }
}

#if YT_USE_EIGEN
template <typename T, int dim>
YT_IMPL_INLINE auto matViewEigen(const YTensor<T, dim>& tensor) requires(dim > 2) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::matViewEigen: layout not implemented");
    }
    using EigenMatrixMap = typename YTensor<T, dim>::EigenMatrixMap;
    const std::vector<int> tensorShape = tensor.shape();
    std::vector<int> batchShape(tensorShape.begin(), tensorShape.end() - 2);
    YTensor<EigenMatrixMap, std::max(1, dim - 2)> out;
    std::vector<int> batchStride(batchShape.size(), 1);
    for (int i = static_cast<int>(batchShape.size()) - 2; i >= 0; --i) {
        int64_t next = static_cast<int64_t>(batchStride[i + 1]) * batchShape[i + 1];
        if (next > std::numeric_limits<int>::max()) {
            throw std::overflow_error("strided::matViewEigen: batch stride exceeds int range");
        }
        batchStride[i] = static_cast<int>(next);
    }
    size_t batchSize = 1;
    for (int size : batchShape) {
        if (size != 0 && batchSize > static_cast<size_t>(std::numeric_limits<int>::max()) / size) {
            throw std::overflow_error("strided::matViewEigen: batch size exceeds int indexing range");
        }
        batchSize *= size;
    }
    const auto tensorStride = tensor.stride_();
    const T* tensorData = tensor.data_();
    // Eigen::Map只保存raw pointer；map对象由结果持有，但输入storage生命周期仍由调用方保证。
    // 外层tensor只拥有Eigen::Map对象；Map中的scalar pointer仍借用输入tensor storage。
    auto mapMemory = yt::utils::makeSharedPlacementArray<EigenMatrixMap>(
        batchSize, [&](EigenMatrixMap* dest, size_t index) {
            std::vector<int> coord(batchShape.size());
            size_t remaining = index;
            for (int i = static_cast<int>(batchShape.size()) - 1; i >= 0; --i) {
                coord[i] = static_cast<int>(remaining % static_cast<size_t>(batchShape[i]));
                remaining /= static_cast<size_t>(batchShape[i]);
            }
            Eigen::Stride<-1, -1> matrixStride(tensorStride[dim - 2], tensorStride[dim - 1]);
            T* matrixData = tensorData == nullptr
                                ? nullptr
                                : const_cast<T*>(tensorData) +
                                      typedBatchStorageOffset(tensor, coord, "strided::matViewEigen");
            new (dest) EigenMatrixMap(matrixData, tensor.shape(dim - 2), tensor.shape(dim - 1), matrixStride);
        }
    );
    YMemory memory(mapMemory, batchSize * sizeof(EigenMatrixMap), "cpu", false);
    ViewTensorAccess<EigenMatrixMap, std::max(1, dim - 2)>::setView(
        out, batchShape, batchStride, 0, memory, yt::type::getTypeName<EigenMatrixMap>()
    );

    return out;
}

template <typename T, int dim>
YT_IMPL_INLINE auto matViewEigen(const YTensor<T, dim>& tensor) requires(dim <= 2) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::matViewEigen: layout not implemented");
    }
    using EigenMatrixMap = typename YTensor<T, dim>::EigenMatrixMap;
    static_assert(dim >= 1, "matViewEigen only supports dim >= 1");
    const auto stride = tensor.stride_();
    T* data = const_cast<T*>(tensor.data());
    if constexpr (dim == 1) {
        // Eigen helper沿用既有Nx1列向量方向；普通matView的1xN提升用于matmul facade。
        return EigenMatrixMap(data, tensor.shape(0), 1, Eigen::Stride<-1, -1>(0, stride[0]));
    } else {
        return EigenMatrixMap(
            data, tensor.shape(0), tensor.shape(1), Eigen::Stride<-1, -1>(stride[0], stride[1])
        );
    }
}
#endif

}  // namespace yt::strided
