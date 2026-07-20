#pragma once
/***************
 * file: strided/copy.inl
 * purpose: strided layout 的 copy/clone/contiguous 职责实现。
 ***************/

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace yt::strided {

// ==================== dtype conversion helpers ====================

using NumericReader = long double (*)(const void*);
using NumericWriter = void (*)(void*, long double);

// 管理临时非POD对象数组；`constructed`始终表示已成功构造的前缀长度。
// 注意：任一callback抛出时，析构函数只回收此前构造成功的对象。
struct TemporaryObjectBuffer {
    std::unique_ptr<char[]> data;
    std::function<void(void*)> destructor;
    size_t elementSize = 0;
    size_t constructed = 0;

    ~TemporaryObjectBuffer() {
        // RAII：异常中途退出时自动析构已拷贝构造的临时非POD对象，防止资源泄漏
        if (!destructor) return;
        for (size_t i = 0; i < constructed; ++i) {
            destructor(data.get() + i * elementSize);
        }
    }
};

template <typename T>
long double readNumeric(const void* source) {
    return static_cast<long double>(*static_cast<const T*>(source));
}

template <typename T>
void writeNumeric(void* destination, long double value) {
    *static_cast<T*>(destination) = static_cast<T>(value);
}

// 返回内置数值dtype到long double的读取入口；custom dtype返回nullptr。
inline NumericReader getNumericReader(const std::string& dtype) {
    if (dtype == "float32") return &readNumeric<float>;
    if (dtype == "float64") return &readNumeric<double>;
    if (dtype == "int8") return &readNumeric<int8_t>;
    if (dtype == "int16") return &readNumeric<int16_t>;
    if (dtype == "int32") return &readNumeric<int32_t>;
    if (dtype == "int64") return &readNumeric<int64_t>;
    if (dtype == "uint8") return &readNumeric<uint8_t>;
    if (dtype == "uint16") return &readNumeric<uint16_t>;
    if (dtype == "uint32") return &readNumeric<uint32_t>;
    if (dtype == "uint64") return &readNumeric<uint64_t>;
    if (dtype == "bool") return &readNumeric<bool>;
    if (dtype == "bfloat16") return &readNumeric<yt::bfloat16>;
    if (dtype == "float16") return &readNumeric<yt::float16>;
    if (dtype == "float8_e5m2") return &readNumeric<yt::float8_e5m2>;
    if (dtype == "float8_e4m3") return &readNumeric<yt::float8_e4m3>;
    if (dtype == "float8_e8m0") return &readNumeric<yt::float8_e8m0>;
    if (dtype == "float8_ue8m0") return &readNumeric<yt::float8_ue8m0>;
    return nullptr;
}

// 返回long double到内置数值dtype的写入入口；写入前必须完成目标范围校验。
inline NumericWriter getNumericWriter(const std::string& dtype) {
    if (dtype == "float32") return &writeNumeric<float>;
    if (dtype == "float64") return &writeNumeric<double>;
    if (dtype == "int8") return &writeNumeric<int8_t>;
    if (dtype == "int16") return &writeNumeric<int16_t>;
    if (dtype == "int32") return &writeNumeric<int32_t>;
    if (dtype == "int64") return &writeNumeric<int64_t>;
    if (dtype == "uint8") return &writeNumeric<uint8_t>;
    if (dtype == "uint16") return &writeNumeric<uint16_t>;
    if (dtype == "uint32") return &writeNumeric<uint32_t>;
    if (dtype == "uint64") return &writeNumeric<uint64_t>;
    if (dtype == "bool") return &writeNumeric<bool>;
    if (dtype == "bfloat16") return &writeNumeric<yt::bfloat16>;
    if (dtype == "float16") return &writeNumeric<yt::float16>;
    if (dtype == "float8_e5m2") return &writeNumeric<yt::float8_e5m2>;
    if (dtype == "float8_e4m3") return &writeNumeric<yt::float8_e4m3>;
    if (dtype == "float8_e8m0") return &writeNumeric<yt::float8_e8m0>;
    if (dtype == "float8_ue8m0") return &writeNumeric<yt::float8_ue8m0>;
    return nullptr;
}

inline bool isIntegralNumericDtype(const std::string& dtype) {
    return dtype == "int8" || dtype == "int16" || dtype == "int32" || dtype == "int64" ||
           dtype == "uint8" || dtype == "uint16" || dtype == "uint32" || dtype == "uint64" ||
           dtype == "bool";
}

using IntegerCastValidator = void (*)(const void*);

// 同一整数dtype pair的预校验与无抛出写入操作，供两阶段commit使用。
struct IntegerCastOps {
    yt::type::YCastKernel cast = nullptr;
    IntegerCastValidator validate = nullptr;

    explicit operator bool() const { return cast != nullptr && validate != nullptr; }
};

// 按signedness和位宽判断整数值能否无越界地写入目标dtype。
// 注意：bool目标只接受0/1；signed到unsigned先拒绝负数，再比较无符号范围。
template <typename Src, typename Dst>
inline bool integerValueFits(Src value) {
    static_assert(std::is_integral_v<Src> && std::is_integral_v<Dst>);
    if constexpr (std::is_same_v<Dst, bool>) {
        return value == 0 || value == 1;
    } else if constexpr (std::is_same_v<Src, bool>) {
        return true;
    } else if constexpr (std::is_signed_v<Src> == std::is_signed_v<Dst>) {
        if constexpr (sizeof(Dst) >= sizeof(Src)) {
            return true;
        } else {
            return value >= static_cast<Src>(std::numeric_limits<Dst>::lowest()) &&
                   value <= static_cast<Src>(std::numeric_limits<Dst>::max());
        }
    } else if constexpr (std::is_signed_v<Src>) {
        if (value < 0) return false;
        using UnsignedSrc = std::make_unsigned_t<Src>;
        if constexpr (sizeof(Dst) >= sizeof(UnsignedSrc)) {
            return true;
        } else {
            return static_cast<UnsignedSrc>(value) <=
                   static_cast<UnsignedSrc>(std::numeric_limits<Dst>::max());
        }
    } else {
        if constexpr (sizeof(Dst) > sizeof(Src)) {
            return true;
        } else {
            return value <= static_cast<Src>(std::numeric_limits<Dst>::max());
        }
    }
}

template <typename Src, typename Dst>
inline void validateIntegerCast(const void* source) {
    if (!integerValueFits<Src, Dst>(*static_cast<const Src*>(source))) {
        throw std::out_of_range("copy_: numeric value is outside destination integer range");
    }
}

template <typename Src, typename Dst>
inline void castInteger(void* destination, const void* source) {
    *static_cast<Dst*>(destination) = static_cast<Dst>(*static_cast<const Src*>(source));
}

template <typename Src>
inline IntegerCastOps getIntegerCastDestination(const std::string& dtype) {
    if (dtype == "int8") return {&castInteger<Src, int8_t>, &validateIntegerCast<Src, int8_t>};
    if (dtype == "int16") return {&castInteger<Src, int16_t>, &validateIntegerCast<Src, int16_t>};
    if (dtype == "int32") return {&castInteger<Src, int32_t>, &validateIntegerCast<Src, int32_t>};
    if (dtype == "int64") return {&castInteger<Src, int64_t>, &validateIntegerCast<Src, int64_t>};
    if (dtype == "uint8") return {&castInteger<Src, uint8_t>, &validateIntegerCast<Src, uint8_t>};
    if (dtype == "uint16") return {&castInteger<Src, uint16_t>, &validateIntegerCast<Src, uint16_t>};
    if (dtype == "uint32") return {&castInteger<Src, uint32_t>, &validateIntegerCast<Src, uint32_t>};
    if (dtype == "uint64") return {&castInteger<Src, uint64_t>, &validateIntegerCast<Src, uint64_t>};
    if (dtype == "bool") return {&castInteger<Src, bool>, &validateIntegerCast<Src, bool>};
    return {};
}

inline IntegerCastOps getBuiltinIntegerCast(
    const std::string& sourceDtype, const std::string& destinationDtype
) {
    if (sourceDtype == "int8") return getIntegerCastDestination<int8_t>(destinationDtype);
    if (sourceDtype == "int16") return getIntegerCastDestination<int16_t>(destinationDtype);
    if (sourceDtype == "int32") return getIntegerCastDestination<int32_t>(destinationDtype);
    if (sourceDtype == "int64") return getIntegerCastDestination<int64_t>(destinationDtype);
    if (sourceDtype == "uint8") return getIntegerCastDestination<uint8_t>(destinationDtype);
    if (sourceDtype == "uint16") return getIntegerCastDestination<uint16_t>(destinationDtype);
    if (sourceDtype == "uint32") return getIntegerCastDestination<uint32_t>(destinationDtype);
    if (sourceDtype == "uint64") return getIntegerCastDestination<uint64_t>(destinationDtype);
    if (sourceDtype == "bool") return getIntegerCastDestination<bool>(destinationDtype);
    return {};
}

// 验证浮点中间值能否进入整数目标；合法小数仍按static_cast规则截断。
// 注意：当long double无法精确表示64位最大值时，使用排他的2^N上界避免边界误判。
inline void validateNumericWrite(const std::string& dtype, long double value) {
    if (!isIntegralNumericDtype(dtype)) return;
    if (!std::isfinite(value)) {
        throw std::domain_error("copy_: non-finite value cannot be cast to integer dtype");
    }
    long double lowest = 0;
    long double upperBound = 0;
    bool upperBoundExclusive = false;
    if (dtype == "int8") {
        lowest = std::numeric_limits<int8_t>::lowest();
        upperBound = std::numeric_limits<int8_t>::max();
    } else if (dtype == "int16") {
        lowest = std::numeric_limits<int16_t>::lowest();
        upperBound = std::numeric_limits<int16_t>::max();
    } else if (dtype == "int32") {
        lowest = std::numeric_limits<int32_t>::lowest();
        upperBound = std::numeric_limits<int32_t>::max();
    } else if (dtype == "int64") {
        lowest = static_cast<long double>(std::numeric_limits<int64_t>::lowest());
        if constexpr (std::numeric_limits<long double>::digits >= 63) {
            upperBound = static_cast<long double>(std::numeric_limits<int64_t>::max());
        } else {
            upperBound = std::ldexp(1.0L, 63);
            upperBoundExclusive = true;
        }
    } else if (dtype == "uint8") {
        upperBound = std::numeric_limits<uint8_t>::max();
    } else if (dtype == "uint16") {
        upperBound = std::numeric_limits<uint16_t>::max();
    } else if (dtype == "uint32") {
        upperBound = std::numeric_limits<uint32_t>::max();
    } else if (dtype == "uint64") {
        if constexpr (std::numeric_limits<long double>::digits >= 64) {
            upperBound = static_cast<long double>(std::numeric_limits<uint64_t>::max());
        } else {
            upperBound = std::ldexp(1.0L, 64);
            upperBoundExclusive = true;
        }
    } else {
        upperBound = 1;
    }
    const bool aboveUpperBound = upperBoundExclusive ? value >= upperBound : value > upperBound;
    if (value < lowest || aboveUpperBound) {
        throw std::out_of_range("copy_: numeric value is outside destination integer range");
    }
}

// 遍历目标逻辑位置；重叠view串行写入，避免多个逻辑元素并发修改同一对象。
template <typename Func>
inline void forEachCopyIndex(const YTensorBase& dst, int total, Func&& func) {
    if (dst.isDisjoint()) {
        yt::utils::parallelFor(0, total, std::forward<Func>(func));
    } else {
        for (int index = 0; index < total; ++index) func(index);
    }
}

// 将row-major逻辑下标映射为storage绝对元素下标，支持负stride和非零offset。
struct LinearStorageIndexer {
    const std::vector<int>& shape;
    const std::vector<int>& stride;
    int offset;
    size_t storageElements;

    size_t operator()(size_t logicalIndex) const {
        // metadata使用int，但多维stride累加必须在int64_t中完成并最终验证storage边界。
        int64_t physicalIndex = offset;
        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim) {
            const size_t extent = static_cast<size_t>(shape[dim]);
            const int64_t delta = static_cast<int64_t>(logicalIndex % extent) * stride[dim];
            logicalIndex /= extent;
            if ((delta > 0 && physicalIndex > std::numeric_limits<int64_t>::max() - delta) ||
                (delta < 0 && physicalIndex < std::numeric_limits<int64_t>::min() - delta)) {
                throw std::overflow_error("strided copy: physical index overflow");
            }
            physicalIndex += delta;
        }
        if (physicalIndex < 0 || static_cast<uint64_t>(physicalIndex) >= storageElements) {
            throw std::out_of_range("strided copy: physical index is outside storage");
        }
        return static_cast<size_t>(physicalIndex);
    }
};

// ==================== clone/copy ownership ====================

// 将任意Strided逻辑内容物化为独立的连续storage。
// 先建立连续metadata，再按POD、注册非POD或nested wrapper生命周期分别构造数据。
YT_IMPL_INLINE YTensorBase clone(const YTensorBase& tensor) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::clone: layout not implemented");
    }

    const auto shape = tensor.stridedShape();
    int ndim = static_cast<int>(shape.size());
    std::vector<int> stride(ndim);
    if (ndim > 0) {
        stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            int64_t next = static_cast<int64_t>(stride[i + 1]) * shape[i + 1];
            if (next > std::numeric_limits<int>::max()) {
                throw std::overflow_error("clone: logical stride exceeds int range");
            }
            stride[i] = static_cast<int>(next);
        }
    }

    size_t total = 1;
    for (int v : shape) {
        if (v < 0) throw std::invalid_argument("clone: negative shape extent");
        if (v != 0 && total > static_cast<size_t>(std::numeric_limits<int>::max()) / v) {
            throw std::overflow_error("clone: logical size exceeds int indexing range");
        }
        total *= static_cast<size_t>(v);
    }
    YTensorBase out;
    out.stridedShape() = shape;
    out.stridedStride() = std::move(stride);
    out.stridedOffset() = 0;
    out._dtype = tensor._dtype;
    out._element_size = tensor._element_size;
    // 空tensor保留shape/dtype，但不分配storage，也不构造伪元素。
    if (total == 0) return out;

    auto typeInfoOpt = yt::type::getTypeInfo(tensor._dtype);
    size_t elemSize = tensor._element_size;
    if (elemSize == 0 || total > std::numeric_limits<size_t>::max() / elemSize) {
        throw std::overflow_error("clone: storage byte size overflow");
    }
    const size_t byteSize = total * elemSize;
    const LinearStorageIndexer sourceIndex{
        tensor.stridedShape(), tensor.stridedStride(), tensor.stridedOffset(),
        tensor._memory ? tensor._memory.nbytes() / elemSize : 0
    };
    const auto outerDtype = yt::type::parseDtypeInner(tensor._dtype).first;
    // runtime nested wrapper是真实C++对象，必须逐元素placement-copy并由object-backed storage析构。
    if (outerDtype == "YTensorBase") {
        if (elemSize != sizeof(YTensorBase)) {
            throw std::invalid_argument("clone: YTensorBase dtype has an invalid element size");
        }
        auto objects = yt::utils::makeSharedPlacementArray<YTensorBase>(
            total, [&](YTensorBase* destination, size_t index) {
                new (destination) YTensorBase(
                    *reinterpret_cast<const YTensorBase*>(
                        tensor._memory.get() + sourceIndex(index) * elemSize
                    )
                );
            }
        );
        out._memory = YMemory(objects, byteSize, "cpu", false);
        return out;
    }
    if (outerDtype == "YTensor") {
        throw std::runtime_error("clone: typed nested tensor requires the typed facade");
    }

    if (typeInfoOpt && !typeInfoOpt->get().isPOD) {
        const auto& typeInfo = typeInfoOpt->get();
        size_t numElems = total;
        auto destructor = typeInfo.destructor;
        auto copyConstruct = typeInfo.copyConstruct;
        auto defaultConstruct = typeInfo.defaultConstruct;
        if (tensor._memory && !copyConstruct) {
            throw std::runtime_error("clone: non-POD type has no copyConstruct registered");
        }
        char* rawPtr = new char[byteSize];
        size_t constructed = 0;
        try {
            if (copyConstruct && tensor._memory) {
                // 有源数据可用：用拷贝构造逐元素复制（含非POD深拷贝语义）
                for (; constructed < numElems; ++constructed) {
                    size_t srcIndex = sourceIndex(constructed);
                    copyConstruct(
                        rawPtr + constructed * elemSize, tensor._memory.get() + srcIndex * elemSize
                    );
                }
            } else if (defaultConstruct) {
                // storage缺失时仍需建立每个目标对象的合法默认生命周期。
                for (; constructed < numElems; ++constructed) {
                    defaultConstruct(rawPtr + constructed * elemSize);
                }
            } else {
                throw std::runtime_error("clone: non-POD type has no constructor registered");
            }
        } catch (...) {
            // 自定义callback（copyConstruct/defaultConstruct）抛异常：回滚已构造元素
            if (destructor) {
                for (size_t i = 0; i < constructed; ++i) {
                    destructor(rawPtr + i * elemSize);
                }
            }
            delete[] rawPtr;
            throw;
        }

        out._memory = YMemory(
            std::shared_ptr<char[]>(
                rawPtr,
                [destructor, elemSize, numElems](char* ptr) {
                    if (destructor) {
                        for (size_t i = 0; i < numElems; ++i) {
                            destructor(ptr + i * elemSize);
                        }
                    }
                    delete[] ptr;
                }
            ),
            byteSize, "cpu", false
        );
    } else {
        out._memory = YMemory(std::shared_ptr<char[]>(new char[byteSize]), byteSize);
        if (tensor._memory) {
            if (tensor.isContiguous()) {
                // 连续POD：整块memcpy，最快路径
                std::memcpy(
                    out._memory.get(), tensor._memory.get() + tensor.stridedOffset() * elemSize,
                    byteSize
                );
            } else {
                // 非连续POD：逐元素按physicalIndex拷贝
                char* dstPtr = out._memory.get();
                const char* srcBase = tensor._memory.get();
                for (size_t dst = 0; dst < total; ++dst) {
                    size_t srcIndex = sourceIndex(dst);
                    std::memcpy(dstPtr + dst * elemSize, srcBase + srcIndex * elemSize, elemSize);
                }
            }
        }
    }

    return out;
}

// 在保留dst metadata/storage identity的前提下复制逻辑元素，必要时执行dtype cast。
// alias输入先快照；可能抛出的范围检查和非POD构造在写入dst前完成。
YT_IMPL_INLINE YTensorBase& copy_(YTensorBase& dst, const YTensorBase& src) {
    if (!dst.isStrided() || !src.isStrided()) {
        throw std::runtime_error("strided::copy_: layout not implemented");
    }
    if (!dst.shapeMatch(src.shape())) {
        throw std::runtime_error("copy_: source and destination shapes must match");
    }

    size_t dstElemSize = dst._element_size;
    size_t srcElemSize = src._element_size;
    int total = yt::utils::checkedIntSize(dst.size(), "strided::copy_");
    if (total == 0) return dst;
    if (dstElemSize == 0 || srcElemSize == 0) {
        throw std::invalid_argument("copy_: source and destination element sizes must be positive");
    }
    bool mayOverlap = yt::strided::physicalSpansOverlap(dst, src);
    bool sameType = (dst._dtype == src._dtype);
    const auto dstOuterDtype = yt::type::parseDtypeInner(dst._dtype).first;
    const auto srcOuterDtype = yt::type::parseDtypeInner(src._dtype).first;
    const bool dstIsNestedTensor = dstOuterDtype == "YTensor" || dstOuterDtype == "YTensorBase";
    const bool srcIsNestedTensor = srcOuterDtype == "YTensor" || srcOuterDtype == "YTensorBase";
    if (dstIsNestedTensor || srcIsNestedTensor) {
        throw std::runtime_error("copy_: nested tensor elements are not supported by the runtime facade");
    }
    auto dstTypeInfo = yt::type::getTypeInfo(dst._dtype);
    auto srcTypeInfo = yt::type::getTypeInfo(src._dtype);
    bool dstIsPOD = !dstTypeInfo || dstTypeInfo->get().isPOD;
    bool srcIsPOD = !srcTypeInfo || srcTypeInfo->get().isPOD;
    const LinearStorageIndexer dstIndex{
        dst.stridedShape(), dst.stridedStride(), dst.stridedOffset(), dst._memory.nbytes() / dstElemSize
    };
    const LinearStorageIndexer srcIndex{
        src.stridedShape(), src.stridedStride(), src.stridedOffset(), src._memory.nbytes() / srcElemSize
    };

    // 连续POD同类型且无overlap：整块memcpy，最快路径
    if (sameType && dst.isContiguous() && src.isContiguous() && !mayOverlap && dstIsPOD) {
        std::memcpy(
            dst._memory.get() + dst.stridedOffset() * dstElemSize,
            src._memory.get() + src.stridedOffset() * srcElemSize, static_cast<size_t>(total) * dstElemSize
        );
        return dst;
    }

    if (sameType && dstIsPOD && dst.isDisjoint() && !mayOverlap) {
        const int contiguousFrom = std::max(dst.isContiguousFrom(), src.isContiguousFrom());
        size_t blockElements = 1;
        for (int dim = contiguousFrom; dim < dst.ndim(); ++dim) {
            blockElements *= static_cast<size_t>(dst.shape(dim));
        }
        if (blockElements > 1) {
            // contiguousFrom之后双方都连续，每个outer logical index可安全合并成一个memcpy块。
            const int blockCount = total / static_cast<int>(blockElements);
            char* dstBase = dst._memory.get();
            const char* srcBase = src._memory.get();
            // 小拷贝避免创建OpenMP任务；1 MiB仅是并行化粒度阈值，不影响语义。
            const bool useParallel = static_cast<size_t>(total) * dstElemSize >= (1U << 20);
            yt::utils::parallelFor(0, blockCount, [&](int block) {
                const size_t logicalIndex = static_cast<size_t>(block) * blockElements;
                std::memcpy(
                    dstBase + dstIndex(logicalIndex) * dstElemSize,
                    srcBase + srcIndex(logicalIndex) * srcElemSize, blockElements * dstElemSize
                );
            }, useParallel);
            return dst;
        }
    }

    TemporaryObjectBuffer tempBuffer;
    const char* srcBasePtr = src._memory.get();
    bool needTemp = mayOverlap;

    if (needTemp) {
        // bounding-span检测是保守的；即使是假阳性，快照也只影响性能而不改变结果。
        // 源和目标内存重叠时，中转可保护尚未读取的数据不被提前覆盖。
        tempBuffer.data = std::make_unique<char[]>(static_cast<size_t>(total) * srcElemSize);
        tempBuffer.elementSize = srcElemSize;
        if (!srcIsPOD) tempBuffer.destructor = srcTypeInfo->get().destructor;
        if (src.isContiguous()) {
            if (srcIsPOD) {
                std::memcpy(
                    tempBuffer.data.get(), srcBasePtr + src.stridedOffset() * srcElemSize,
                    static_cast<size_t>(total) * srcElemSize
                );
            } else {
                auto copyConstruct = srcTypeInfo->get().copyConstruct;
                if (!copyConstruct) {
                    throw std::runtime_error("copy_: non-POD type has no copyConstruct registered");
                }
                for (int i = 0; i < total; ++i) {
                    copyConstruct(
                        tempBuffer.data.get() + i * srcElemSize,
                        srcBasePtr + (src.stridedOffset() + i) * srcElemSize
                    );
                    ++tempBuffer.constructed;
                }
            }
        } else {
            if (srcIsPOD) {
                yt::utils::parallelFor(0, total, [&](int index) {
                    size_t sourceIndex = srcIndex(static_cast<size_t>(index));
                    std::memcpy(
                        tempBuffer.data.get() + index * srcElemSize,
                        srcBasePtr + sourceIndex * srcElemSize,
                        srcElemSize
                    );
                });
            } else {
                auto copyConstruct = srcTypeInfo->get().copyConstruct;
                if (!copyConstruct) {
                    throw std::runtime_error("copy_: non-POD type has no copyConstruct registered");
                }
                for (int index = 0; index < total; ++index) {
                    size_t sourceIndex = srcIndex(static_cast<size_t>(index));
                    copyConstruct(
                        tempBuffer.data.get() + index * srcElemSize,
                        srcBasePtr + sourceIndex * srcElemSize
                    );
                    ++tempBuffer.constructed;
                }
            }
        }
        srcBasePtr = tempBuffer.data.get();
    }

    char* dstBasePtr = dst._memory.get();
    auto calcDstIndex = [&](int index) -> size_t {
        return dstIndex(static_cast<size_t>(index));
    };

    auto calcSrcIndex = [&](int index) -> size_t {
        if (needTemp) {
            return static_cast<size_t>(index);
        }
        return srcIndex(static_cast<size_t>(index));
    };

    if (sameType) {
        if (dstIsPOD) {
            forEachCopyIndex(dst, total, [&](int index) {
                size_t dstIndex = calcDstIndex(index);
                size_t srcIndex = calcSrcIndex(index);
                std::memcpy(
                    dstBasePtr + dstIndex * dstElemSize, srcBasePtr + srcIndex * srcElemSize, dstElemSize
                );
            });
        } else {
            auto copyConstruct = dstTypeInfo->get().copyConstruct;
            auto swapObjects = dstTypeInfo->get().swap;
            if (!copyConstruct || !swapObjects) {
                throw std::runtime_error(
                    "copy_: non-POD type requires copyConstruct and noexcept swap for transactional copy"
                );
            }
            TemporaryObjectBuffer replacement;
            replacement.data = std::make_unique<char[]>(static_cast<size_t>(total) * dstElemSize);
            replacement.elementSize = dstElemSize;
            replacement.destructor = dstTypeInfo->get().destructor;
            for (int index = 0; index < total; ++index) {
                size_t srcIndex = calcSrcIndex(index);
                copyConstruct(
                    replacement.data.get() + static_cast<size_t>(index) * dstElemSize,
                    srcBasePtr + srcIndex * srcElemSize
                );
                ++replacement.constructed;
            }
            // 所有可能抛异常的构造完成后，再用noexcept swap一次性提交到原view。
            for (int index = 0; index < total; ++index) {
                size_t dstIndex = calcDstIndex(index);
                swapObjects(
                    dstBasePtr + dstIndex * dstElemSize,
                    replacement.data.get() + static_cast<size_t>(index) * dstElemSize
                );
            }
        }
    } else {
        // 跨类型拷贝：优先使用已注册的cast kernel（精确类型转换）
        auto castKernel = yt::type::getCastKernel(src._dtype, dst._dtype);
        if (castKernel) {
            if (dstIsPOD) {
                // POD custom cast callback若抛出可能留下部分写入；callback应自行提供无抛出写入合同。
                for (int index = 0; index < total; ++index) {
                    size_t dstIndex = calcDstIndex(index);
                    size_t srcIndex = calcSrcIndex(index);
                    castKernel(dstBasePtr + dstIndex * dstElemSize, srcBasePtr + srcIndex * srcElemSize);
                }
            } else {
                auto copyConstruct = dstTypeInfo->get().copyConstruct;
                auto swapObjects = dstTypeInfo->get().swap;
                if (!copyConstruct || !swapObjects) {
                    throw std::runtime_error(
                        "copy_: non-POD cast destination requires copyConstruct and noexcept swap"
                    );
                }
                TemporaryObjectBuffer replacement;
                replacement.data = std::make_unique<char[]>(static_cast<size_t>(total) * dstElemSize);
                replacement.elementSize = dstElemSize;
                replacement.destructor = dstTypeInfo->get().destructor;
                for (int index = 0; index < total; ++index) {
                    size_t dstIndex = calcDstIndex(index);
                    copyConstruct(
                        replacement.data.get() + static_cast<size_t>(index) * dstElemSize,
                        dstBasePtr + dstIndex * dstElemSize
                    );
                    ++replacement.constructed;
                }
                // cast先写临时已构造对象；任一callback抛异常时目标保持原样。
                for (int index = 0; index < total; ++index) {
                    size_t srcIndex = calcSrcIndex(index);
                    castKernel(
                        replacement.data.get() + static_cast<size_t>(index) * dstElemSize,
                        srcBasePtr + srcIndex * srcElemSize
                    );
                }
                for (int index = 0; index < total; ++index) {
                    size_t dstIndex = calcDstIndex(index);
                    swapObjects(
                        dstBasePtr + dstIndex * dstElemSize,
                        replacement.data.get() + static_cast<size_t>(index) * dstElemSize
                    );
                }
            }
        } else if (auto integerCast = getBuiltinIntegerCast(src._dtype, dst._dtype); integerCast) {
            // 在任何写入前完成全量range验证，失败时保持destination完全不变。
            for (int index = 0; index < total; ++index) {
                size_t srcIndex = calcSrcIndex(index);
                integerCast.validate(srcBasePtr + srcIndex * srcElemSize);
            }
            forEachCopyIndex(dst, total, [&](int index) {
                size_t dstIndex = calcDstIndex(index);
                size_t srcIndex = calcSrcIndex(index);
                integerCast.cast(
                    dstBasePtr + dstIndex * dstElemSize, srcBasePtr + srcIndex * srcElemSize
                );
            });
        } else {
            // 无pairwise kernel时，仅对已知builtin numeric dtype使用long double中间值转换。
            // 这不是按element-size猜测类型，custom/非POD pair必须显式注册cast kernel。
            if (!srcIsPOD || !dstIsPOD) {
                throw std::runtime_error(
                    "copy_: no cast kernel registered for dtype pair " + src._dtype + " -> " + dst._dtype
                );
            }
            auto reader = getNumericReader(src._dtype);
            auto writer = getNumericWriter(dst._dtype);
            if (!reader || !writer) {
                throw std::runtime_error(
                    "copy_: no builtin cast kernel for dtype pair " + src._dtype + " -> " + dst._dtype
                );
            }
            // domain/range验证与commit分离，避免并行写入中途失败留下部分结果。
            for (int index = 0; index < total; ++index) {
                size_t srcIndex = calcSrcIndex(index);
                validateNumericWrite(dst._dtype, reader(srcBasePtr + srcIndex * srcElemSize));
            }
            forEachCopyIndex(dst, total, [&](int index) {
                size_t dstIndex = calcDstIndex(index);
                size_t srcIndex = calcSrcIndex(index);
                writer(
                    dstBasePtr + dstIndex * dstElemSize,
                    reader(srcBasePtr + srcIndex * srcElemSize)
                );
            });
        }
    }

    return dst;
}

// 已连续时返回共享storage的浅拷贝，否则返回独立连续clone。
YT_IMPL_INLINE YTensorBase contiguous(const YTensorBase& tensor) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::contiguous: layout not implemented");
    }
    if (tensor._memory == nullptr) {
        return tensor;
    }
    if (tensor.isContiguous()) {
        return tensor;
    }
    return yt::strided::clone(tensor);
}

// 仅在必要时用连续clone替换当前handle；已有view引用不会自动跟随新storage。
YT_IMPL_INLINE YTensorBase& contiguous_(YTensorBase& tensor) {
    if (!tensor.isStrided()) {
        throw std::runtime_error("strided::contiguous_: layout not implemented");
    }
    if (tensor._memory == nullptr || tensor.isContiguous()) {
        return tensor;
    }
    YTensorBase cloned = yt::strided::clone(tensor);
    tensor = cloned;
    return tensor;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim>& copy_(YTensor<T, dim>& dst, const YTensorBase& src) {
    yt::strided::copy_(static_cast<YTensorBase&>(dst), src);
    return dst;
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> clone(const YTensor<T, dim>& tensor) {
    if constexpr (yt::utils::is_ytensor_template_v<T>) {
        // nested typed元素必须复制构造真实wrapper对象，runtime raw-byte路径无法恢复其静态类型。
        const auto shape = tensor.shape();
        const auto stride = contiguousStrideForShape(shape, "strided::clone nested tensor");
        const size_t total = tensor.size();
        auto objects = yt::utils::makeSharedPlacementArray<T>(total, [&](T* destination, size_t index) {
            new (destination) T(tensor.atData(static_cast<int>(index)));
        });
        YTensor<T, dim> out;
        ViewTensorAccess<T, dim>::setView(
            out, shape, stride, 0, YMemory(objects, total * sizeof(T), "cpu", false), tensor.dtype()
        );
        return out;
    }
    // typed facade复用runtime owner，避免这里再次实现negative stride和non-POD生命周期。
    return YTensor<T, dim>(yt::strided::clone(static_cast<const YTensorBase&>(tensor)));
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim> contiguous(const YTensor<T, dim>& tensor) {
    if (tensor.data_() == nullptr) {
        return tensor;
    }
    if (tensor.isContiguous()) {
        return tensor;
    }
    return yt::strided::clone(tensor);
}

template <typename T, int dim>
YT_IMPL_INLINE YTensor<T, dim>& contiguous_(YTensor<T, dim>& tensor) {
    if (!tensor.isContiguous()) {
        tensor = yt::strided::contiguous(tensor);
    }
    return tensor;
}

// 分配独立结果并逐slice复用copy_，从而统一处理strided、cast和非POD生命周期。
YT_IMPL_INLINE YTensorBase concat(const std::vector<YTensorBase>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("[strided::concat] Empty tensor list");
    }
    for (const auto& t : tensors) {
        if (!t.isStrided()) {
            throw std::runtime_error("strided::concat: layout not implemented");
        }
    }
    // concat始终返回独立结果；单输入也不能退化为共享storage handle。
    if (tensors.size() == 1) {
        return yt::strided::clone(tensors[0]);
    }

    const auto& first = tensors[0];
    int d = first.ndim();
    if (d == 0) {
        throw std::invalid_argument("[strided::concat] Cannot concatenate scalar tensors");
    }
    axis = (axis % d + d) % d;

    std::vector<int> resultShape = first.shape();
    int64_t totalAxisSize = resultShape[axis];

    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        if (t.ndim() != d) {
            throw std::invalid_argument("[strided::concat] Dimension mismatch");
        }
        if (t.dtype() != first.dtype()) {
            throw std::invalid_argument("[strided::concat] dtype mismatch");
        }
        for (int dim = 0; dim < d; ++dim) {
            if (dim != axis && t.shape(dim) != resultShape[dim]) {
                throw std::invalid_argument("[strided::concat] Shape mismatch on non-concat axis");
            }
        }
        totalAxisSize += t.shape(axis);
        if (totalAxisSize > std::numeric_limits<int>::max()) {
            throw std::overflow_error("[strided::concat] Axis size overflow");
        }
    }
    resultShape[axis] = static_cast<int>(totalAxisSize);

    YTensorBase result(resultShape, first.dtype());

    int offset = 0;
    for (const auto& t : tensors) {
        int axisSize = t.shape(axis);
        if (axisSize == 0) continue;
        YTensorBase dstSlice = yt::strided::slice(result, axis, offset, offset + axisSize, 1, true);
        yt::strided::copy_(dstSlice, t);
        offset += axisSize;
    }

    return result;
}

}  // namespace yt::strided
