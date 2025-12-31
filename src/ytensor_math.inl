#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <omp.h>
#include <string>
#include <typeinfo>
#include "../include/ytensor_infos.hpp"
#include "../include/kernel/math_utils.hpp"
#include "../include/kernel/memory_utils.hpp"
#include "../include/kernel/broadcast.hpp"

template <typename T, int dim> 
template<int dim1, typename Func>
YTensor<T, std::max(dim, dim1)> YTensor<T, dim>::binaryOpBroadcast(const YTensor<T, dim1> &other, Func&& func,
    std::string opName, YTensor<T, std::max(dim, dim1)>* result, double flop) const {
    // 1、填充this与other到相同的维度
    auto thisShape = this->shape();
    auto otherShape = other.shape();
    auto thisStride = this->stride_();
    auto otherStride = other.stride_();
    constexpr int opdim = std::max(dim, dim1);
    int thisLack = opdim - dim;
    int otherLack = opdim - dim1;
    bool equalShape = true;
    if(thisLack > 0){
        thisShape.insert(thisShape.begin(), thisLack, 1);
        thisStride.insert(thisStride.begin(), thisLack, 0);
        equalShape = false;
    }
    if(otherLack > 0){
        otherShape.insert(otherShape.begin(), otherLack, 1);
        otherStride.insert(otherStride.begin(), otherLack, 0);
        equalShape = false;
    }
    // 2、检查维度是否匹配
    std::vector<int> opShape(opdim);
    for (int i = 0; i < opdim; ++i) {
        if(thisShape[i] != otherShape[i]) {
            if (thisShape[i] == 1) {
                opShape[i] = otherShape[i];
                thisStride[i] = 0;
            } else if (otherShape[i] == 1) {
                opShape[i] = thisShape[i];
                otherStride[i] = 0;
            } else {
                throwShapeNotMatch(opName, otherShape);
            }
            equalShape = false;
        } else {
            opShape[i] = thisShape[i];
        }
    }
    YTensor<T, opdim> op;
    if(result != nullptr) {
        if(!result->shapeMatch(opShape)){
            result->reserve(opShape);
        }
        op.shallowCopyFrom(*result);
    } else {
        op.reserve(opShape);
    }
    
    if constexpr (std::is_invocable_v<Func, const T&, const T&, T&>) {
        if(equalShape && this->isContiguous() && other.isContiguous()) {
            // fast path
            int max = op.size();
            const T* thisPtr = this->data_() + this->_offset;
            const T* otherPtr = other.data_() + other._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                func(thisPtr[index], otherPtr[index], op.atData(index));
            }, flop);
            return op;
        }
        // 3、定义并行计算的哈希函数
        auto logicStride = op.stride();
        auto kernel = [&thisStride, &otherStride, this, &other, &op, func, &logicStride](int index) {
            int thisIndex = 0, otherIndex = 0;
            #pragma omp simd reduction(+:thisIndex, otherIndex)
            for (int i = 0; i < op.shapeSize(); ++i) {
                int posi = (index / logicStride[i]) % op._shape[i];
                thisIndex += posi * thisStride[i];
                otherIndex += posi * otherStride[i];
            }
            return func(this->atData_(thisIndex), other.atData_(otherIndex), op.atData_(index));
        };
    
        // 4、并行计算
        int max = op.size();
        yt::kernel::parallelFor(0, max, kernel, flop);
    }
    else{
        if(equalShape && this->isContiguous() && other.isContiguous()) {
            // fast path
            int max = op.size();
            T* opPtr = op.data_();
            const T* thisPtr = this->data_() + this->_offset;
            const T* otherPtr = other.data_() + other._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                opPtr[index] = func(thisPtr[index], otherPtr[index]);
            }, flop);
            return op;
        }
        // 3、定义并行计算的哈希函数
        auto logicStride = op.stride();
        auto kernel = [&thisStride, &otherStride, this, &other, &op, func, &logicStride](int index) {
            int thisIndex = 0, otherIndex = 0;
            #pragma omp simd reduction(+:thisIndex, otherIndex)
            for (int i = 0; i < op.shapeSize(); ++i) {
                int posi = (index / logicStride[i]) % op._shape[i];
                thisIndex += posi * thisStride[i];
                otherIndex += posi * otherStride[i];
            }
            return func(this->atData_(thisIndex), other.atData_(otherIndex));
        };
    
        // 4、并行计算
        int max = op.size();
        yt::kernel::parallelFor(0, max, [&](int index) {
            op.atData_(index) = kernel(index);
        }, flop);
    }
    return op;
}

template <typename T, int dim>
template <int dim1, typename Func>
YTensor<T, dim> &YTensor<T, dim>::binaryOpBroadcastInplace(const YTensor<T, dim1> &other, Func &&func, std::string opName, double flop){
    // 1、填充this与other到相同的维度
    auto thisShape = this->shape();
    auto otherShape = other.shape();
    auto otherStride = other.stride_();
    constexpr int thisLack = std::max(dim, dim1) - dim;
    constexpr int otherLack = std::max(dim, dim1) - dim1;
    bool equalShape = true;
    if(thisLack > 0){
        throwShapeNotMatch(opName, thisShape);
        equalShape = false;
    }
    if(otherLack > 0){
        otherShape.insert(otherShape.begin(), otherLack, 1);
        otherStride.insert(otherStride.begin(), otherLack, 0);
        equalShape = false;
    }

    // 2、检查维度是否匹配
    for (int i = 0; i < dim; ++i) {
        if(thisShape[i] != otherShape[i]) {
            if (otherShape[i] == 1) {
                otherStride[i] = 0;
            } else {
                throwShapeNotMatch(opName, otherShape);
            }
            equalShape = false;
        }
    }

    if(equalShape && this->isContiguous() && other.isContiguous()) {
        // fast path
        int max = this->size();
        T* thisPtr = this->data_() + this->_offset;
        const T* otherPtr = other.data_() + other._offset;
        yt::kernel::parallelFor(0, max, [&](int index) {
            func(thisPtr[index], otherPtr[index]);
        }, flop);
        return *this;
    }

    // 3、定义并行计算的哈希函数
    auto logicStride = this->stride();
    auto kernel = [&otherStride, &logicStride, this, &other, func](int index) -> void {
        int thisIndex = 0, otherIndex = 0;
        #pragma omp simd reduction(+:thisIndex, otherIndex)
        for (int i = 0; i < dim; i++) {
            int posi = (index / logicStride[i]) % _shape[i];
            thisIndex += posi * _stride[i];
            otherIndex += posi * otherStride[i];
        }
        func(this->atData_(thisIndex), other.atData_(otherIndex));
        return;
    };

    // 3、并行计算
    int max = this->size();
    yt::kernel::parallelFor(0, max, kernel, flop);
    return *this;
}

template <typename T, int dim>
template <typename Func, typename... Args>
YTensor<T, dim>& YTensor<T, dim>::broadcastInplace(Func&& func, Args&&... tensors) {
    return yt::kernel::broadcastInplace(*this, std::forward<Func>(func), std::forward<Args>(tensors)...);
}

template<typename T, int dim> template<typename Func>
YTensor<T, dim> YTensor<T, dim>::binaryOpTransform(const T& other, Func&& func,  YTensor<T, dim>* result, double flop) const{
    YTensor<T, dim> op;
    if(result != nullptr){
        if(!result->shapeMatch(this->shape())){
            result->reserve(this->shape());
        }
        op = *result;
    } else {
        op.reserve(this->shape());
    }

    // 连续性优化检测
    auto mcView = this->mostContinuousView();

    int thisSize = this->size();
    if constexpr (std::is_invocable_v<Func, const T&, const T&, T&>){
        if (mcView.isContiguous()) {
            // fast path
            int max = mcView.size();
            T* thisPtr = mcView.data_() + mcView._offset;
            T* opPtr = op.data_() + op._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                func(thisPtr[index], other, opPtr[index]);
            }, flop);
            return op;
        }
        // 创建核函数
        auto logicStride = this->stride();
        auto kernel = [this, &other, func, &op, &logicStride](int index){
            int thisIndex = 0;
            #pragma omp simd reduction(+:thisIndex)
            for (int i = 0; i < dim; i++) {
                int posi = (index / logicStride[i]) % _shape[i];
                thisIndex += posi * _stride[i];
            }
            return func(this->atData_(thisIndex), other, op.atData_(index));
        };
    
        // 并行计算
        yt::kernel::parallelFor(0, thisSize, kernel, flop);
    }else{
        if(mcView.isContiguous()) {
            // fast path
            int max = mcView.size();
            T* thisPtr = mcView.data_() + mcView._offset;
            T* opPtr = op.data_() + op._offset;
            yt::kernel::parallelFor(0, max, [&](int index) {
                opPtr[index] = func(thisPtr[index], other);
            }, flop);
            return op;
        }
        // 创建核函数
        auto logicStride = this->stride();
        auto kernel = [this, &other, func, &logicStride](int index){
            int thisIndex = 0;
            #pragma omp simd reduction(+:thisIndex)
            for (int i = 0; i < dim; i++) {
                int posi = (index / logicStride[i]) % _shape[i];
                thisIndex += posi * _stride[i];
            }
            return func(this->atData_(thisIndex), other);
        };

        // 并行计算
        yt::kernel::parallelFor(0, thisSize, [&](int i) {
            op.atData_(i) = kernel(i);
        }, flop);
    }
    return op;
}

template<typename T, int dim> template<typename Func>
YTensor<T, dim>& YTensor<T, dim>::binaryOpTransformInplace(const T& other, Func&& func, double flop){
    auto wrappedFunc = [func](T& a, const T& b) {
        using ResultType = std::invoke_result_t<Func, T &, const T &>;
        if constexpr (std::is_void_v<ResultType>) {
            func(a, b);
        } else {
            a = func(a, b);
        }
    };
    // 连续性优化检测
    YTensor<T, dim> mcView = this->mostContinuousView();

    if(mcView.isContiguous()) {
        // fast path
        int max = mcView.size();
        T* thisPtr = mcView.data_() + mcView._offset;
        yt::kernel::parallelFor(0, max, [&](int index) {
            wrappedFunc(thisPtr[index], other);
        }, flop);
        return *this;
    }
    int thisSize = this->size();
    // 对于非连续情况，使用遍历法
    auto logicStride = stride();
    auto kernel = [this, &other, &logicStride, wrappedFunc](int index) -> void{
        int thisIndex = 0;
        #pragma omp simd reduction(+:thisIndex)
        for (int i = 0; i < dim; i++) {
            int posi = (index / logicStride[i]) % _shape[i];
            thisIndex += posi * _stride[i];
        }
        wrappedFunc(this->atData_(thisIndex), other);
        return;
    };
    yt::kernel::parallelFor(0, thisSize, kernel, flop);
    return *this;
}

// 运算符生成规则
#define YT_YTENSOR_OPERATOR(OP, ENABLE_IF_T)                                                          \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    YTensor<T, std::max(dim, dim1)>                                                                   \
        YTensor<T, dim>::operator OP(const YTensor<T, dim1>& other) const {                           \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return binaryOpBroadcast(                                                                 \
                other, [](const T& a, const T& b) {                                                   \
                    return a OP b;                                                                    \
                },                                                                                    \
                #OP);                                                                                 \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    YTensor<T, std::max(dim, dim1)>& YTensor<T, dim>::operator OP##=(const YTensor<T, dim1>& other) { \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return binaryOpBroadcastInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a OP## = b;                                                                \
                },                                                                                    \
                "+=");                                                                                \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return binaryOpBroadcastInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a = a OP b;                                                                \
                },                                                                                    \
                "+=");                                                                                \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                        \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    YTensor<T, dim> YTensor<T, dim>::operator OP(const T& other) const {                              \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return binaryOpTransform(                                                                 \
                other, [](const T& a, const T& b) {                                                   \
                    return a OP b;                                                                    \
                },                                                                                    \
                nullptr);                                                                             \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    YTensor<T, dim>& YTensor<T, dim>::operator OP##=(const T& other) {                                \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return binaryOpTransformInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a OP## = b;                                                                \
                });                                                                                   \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return binaryOpTransformInplace(                                                          \
                other, [](T& a, const T& b) {                                                         \
                    return a = a OP b;                                                                \
                });                                                                                   \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                        \
        }                                                                                             \
    }
                                                                                                      

YT_YTENSOR_OPERATOR(+, yt::concepts::HAVE_ADD)
YT_YTENSOR_OPERATOR(-, yt::concepts::HAVE_SUB)
YT_YTENSOR_OPERATOR(*, yt::concepts::HAVE_MUL)
YT_YTENSOR_OPERATOR(/, yt::concepts::HAVE_DIV)
// %有特殊处理
YT_YTENSOR_OPERATOR(&, yt::concepts::HAVE_AND)
YT_YTENSOR_OPERATOR(|, yt::concepts::HAVE_MOD)
YT_YTENSOR_OPERATOR(^, yt::concepts::HAVE_XOR)

#undef YT_YTENSOR_OPERATOR

template <typename T, int dim> template<int dim1>
YTensor<T, std::max(dim, dim1)> YTensor<T, dim>::operator%(const YTensor<T, dim1>& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return binaryOpBroadcast(other, [](const T& a, const T& b) {
            return a % b;
        }, "%");
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpBroadcast(other, [](const T& a, const T& b) {
            return std::fmod(a, b);
        }, "%");
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, std::max(dim, dim1)>& YTensor<T, dim>::operator%=(const YTensor<T, dim1>& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a %= b;
        }, "%=");
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a = a % b;
        }, "%=");
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpBroadcastInplace(other, [](T& a, const T& b) {
            return a = fmod(a, b);
        }, "%=");
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::operator%(const T& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return binaryOpTransform(other, [](const T& a, const T& b) {
            return a % b;
        });
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpTransform(other, [](const T& a, const T& b) {
            return std::fmod(a, b);
        });
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim>
YTensor<T, dim>& YTensor<T, dim>::operator%=(const T& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a %= b;
        });
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a = a % b;
        });
    }
    else if constexpr (std::is_floating_point_v<T>){
        return binaryOpTransformInplace(other, [](T& a, const T& b) {
            return a = fmod(a, b);
        });
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
YTensor<YTensor<T, 2>, std::max(1, dim - 2)> YTensor<T, dim>::matView() const {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    using MatType = YTensor<T, 2>;
    
    if constexpr (dim == 1){
        MatType mat;
        mat._shape = std::vector<int>({1, this->_shape[0]});
        mat._stride = std::vector<int>({0, this->_stride[0]});
        mat._offset = this->_offset;
        mat._element_size = sizeof(T);
        mat._dtype = safeGetTypeName<T>();
        mat._data = this->_data;
        
        YTensor<MatType, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";
        
        // 使用封装函数分配内存
        op._data = yt::kernel::makeSharedPlacement<MatType>(mat);
        return op;
    }else if constexpr (dim == 2){
        YTensor<MatType, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";
        
        // 使用封装函数分配内存
        MatType thisCopy = *this;  // 创建当前张量的副本
        op._data = yt::kernel::makeSharedPlacement<MatType>(thisCopy);
        return op;
    }else{
        auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
        YTensor<MatType, std::max(1, dim - 2)> op;
        op._shape = newShape;
        op._stride = op.stride();
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";
        int batchSize = op.size();
        
        // 使用封装函数分配数组内存
        op._data = yt::kernel::makeSharedPlacementArray<MatType>(batchSize);
        MatType* dataptr = reinterpret_cast<MatType*>(op._data.get());

        // 使用 placement new 构造每个 MatType
        for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
            auto coord = op.toCoord(batchIdx);
            MatType mat;
            mat._shape = {this->_shape[dim-2], this->_shape[dim-1]};
            mat._stride = {this->_stride[dim-2], this->_stride[dim-1]};
            mat._offset = this->offset_(coord);
            mat._element_size = sizeof(T);
            mat._dtype = safeGetTypeName<T>();
            mat._data = this->_data;
            new (&dataptr[batchIdx]) MatType(mat);
        }

        return op;
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul(const YTensor<T, dim1>& other)const{
    static_assert(yt::concepts::HAVE_ADD<T> && yt::concepts::HAVE_MUL<T>, "Type must have add and mul in matmul");
    static_assert(dim >= 1 && dim1 >= 1, "matmul only support dim >= 1");
    int lw = this->shape(-1);
    int rw = other.shape(-2);
    if(lw != rw){
        throwShapeNotMatch("matmul", other.shape());
    }
    // 如果是数字，就用 eigen
    if constexpr (std::is_arithmetic_v<T> && YT_USE_EIGEN) {
        return matmul_eigen_backend(other);
    }else{
        return matmul_zero_backend(other);
    }
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::sum(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    size_t max = op.size();
    if (max * _shape[axis] > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                sum += this->at(subCoord);
            }
            op.atData_(i) = sum;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (int j = 0; j < _shape[axis]; j++) {
                auto subCoord = coord;
                subCoord[axis] = j;
                sum += this->at(subCoord);
            }
            op.atData_(i) = sum;
        }
    }
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::sum(std::vector<int> axis) const requires (dim > 1) {
    for (auto& ax : axis) {
        ax = (ax % dim + dim) % dim;
    }
    auto newShape = this->shape();
    int targetSize = 1;
    for (auto& ax : axis) {
        newShape[ax] = 1;
        targetSize *= shape(ax);
    }
    // 偏移量列表，使用里程表法构建
    std::vector<int> offsets(targetSize);
    std::vector<int> records(axis.size(), 0);
    int offset0 = 0;
    for (int i = 0; i < targetSize; i++) {
        offsets[i] = offset0;
        // 更新里程计
        for (int j = axis.size() - 1; j >= 0; j--) {
            if (records[j] < _shape[axis[j]] - 1) {
                records[j]++;
                break;
            } else {
                records[j] = 0;
            }
        }
        // 更新offset
        offset0 = 0;
        for (int j = axis.size() - 1; j >= 0; j--) {
            offset0 += records[j] * _stride[axis[j]];
        }
    }
    // 现在，offsets已经构建完成

    YTensor<T, dim> op(newShape);
    size_t max = op.size();
    if (max > yt::infos::minParOps){
        #pragma omp parallel for simd  proc_bind(close)
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < offsets.size(); j++) {
                sum += this->atData_(base + offsets[j]);
            }
            op.atData_(i) = sum;
        }
    }else{
        #pragma omp simd
        for (size_t i = 0; i < max; i++) {
            auto coord = op.toCoord(i);
            auto base = this->offset(coord);
            T sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < offsets.size(); j++) {
                sum += this->atData_(base + offsets[j]);
            }
            op.atData_(i) = sum;
        }
    }
    return op;
}

template <typename T, int dim>
T YTensor<T, dim>::sum(int) const requires (dim == 1) {
    T sum = 0;
    int max = this->size();
    if (max * 1. > yt::infos::minParOps){
        #pragma omp parallel for simd reduction(+:sum)  proc_bind(close)
        for (int i = 0; i < max; i++) {
            sum += this->at(i);
        }
    }else{
        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < max; i++) {
            sum += this->at(i);
        }
    }
    return sum;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::mean(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    int axisLen = newShape[axis];
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    int max = static_cast<int>(op.size());
        
    yt::kernel::parallelFor(0, max, [&](int i){
        // 使用Welford算法进行均值计算
        auto coord = op.toCoord(i);
        T mean = 0;
        for (int j = 0; j < axisLen; j++) {
            auto subCoord = coord;
            subCoord[axis] = j;
            T x = this->at(subCoord);
            mean += (x - mean) / static_cast<T>(j + 1);
        }
        op.atData_(i) = mean;
    }, static_cast<double>(axisLen));
    
    return op;
}

template <typename T, int dim>
YTensor<T, dim> YTensor<T, dim>::mean(std::vector<int> axes) const requires (dim > 1) {
    // 规范化轴并计算每个轴的长度
    std::vector<int> normalizedAxes;
    int totalN = 1;
    for (int ax : axes) {
        ax = (ax % dim + dim) % dim;
        normalizedAxes.push_back(ax);
        totalN *= this->shape(ax);
    }
    
    // 依次对每个轴使用mean(int axis)
    YTensor<T, dim> result = *this;
    for (int ax : normalizedAxes) {
        result = result.mean(ax);
    }
    
    return result;
}

template <typename T, int dim>
T YTensor<T, dim>::mean(int) const requires (dim == 1) {
    int n = this->size();
    if (n == 0) return static_cast<T>(0);
    
    // 使用Welford算法进行均值计算，提高数值稳定性
    T mean_val = 0;
    for (int i = 0; i < n; i++) {
        T x = this->at(i);
        mean_val += (x - mean_val) / static_cast<T>(i + 1);
    }
    
    return mean_val;
}

template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> YTensor<T, dim>::max(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    YTensor<T, dim> op(newShape);
    YTensor<int, dim> opi(newShape);
    int max = static_cast<int>(op.size());
    int axisSize = _shape[axis];
    
    yt::kernel::parallelFor(0, max, [&](int i) {
        auto coord = op.toCoord(i);
        T maxer = this->at(coord);
        int maxerIndex = 0;
        for (int j = 0; j < axisSize; j++) {
            auto subCoord = coord;
            subCoord[axis] = j;
            const T& value = this->at(subCoord);
            if (value > maxer) {
                maxer = value;
                maxerIndex = j;
            }
        }
        op.atData_(i) = maxer;
        opi.atData_(i) = maxerIndex;
    }, static_cast<double>(axisSize));
    
    return std::make_pair(op, opi);
}

template <typename T, int dim>
std::pair<YTensor<T, dim>, YTensor<int, dim>> YTensor<T, dim>::max(std::vector<int> axis) const requires (dim > 1) {
    for (auto& ax : axis) {
        ax = (ax % dim + dim) % dim;
    }
    auto newShape = this->shape();
    int targetSize = 1;
    for (auto& ax : axis) {
        newShape[ax] = 1;
        targetSize *= shape(ax);
    }
    // 偏移量列表，使用里程表法构建
    std::vector<int> offsets(targetSize);
    std::vector<int> records(axis.size(), 0);
    int offset0 = 0;
    for (int i = 0; i < targetSize; i++) {
        offsets[i] = offset0;
        // 更新里程计
        for (int j = axis.size() - 1; j >= 0; j--) {
            if (records[j] < _shape[axis[j]] - 1) {
                records[j]++;
                break;
            } else {
                records[j] = 0;
            }
        }
        // 更新offset
        offset0 = 0;
        for (int j = axis.size() - 1; j >= 0; j--) {
            offset0 += records[j] * _stride[axis[j]];
        }
    }
    // 现在，offsets已经构建完成

    YTensor<T, dim> op(newShape);
    YTensor<int, dim> opi(newShape);
    int maxSize = static_cast<int>(op.size());
    int offsetsSize = static_cast<int>(offsets.size());
    
    yt::kernel::parallelFor(0, maxSize, [&](int i) {
        auto coord = op.toCoord(i);
        auto base = this->offset(coord);
        T maxer = this->at(coord);
        int maxerIndex = 0;
        for (int j = 0; j < offsetsSize; j++) {
            const T& value = this->atData_(base + offsets[j]);
            if (value > maxer) {
                maxer = value;
                maxerIndex = j;
            }
        }
        op.atData_(i) = maxer;
        opi.atData_(i) = maxerIndex;
    }, static_cast<double>(offsetsSize));
    
    return std::make_pair(op, opi);
}

template<typename T, int dim>
std::pair<T, int> YTensor<T, dim>::max(int)const requires (dim == 1) { 
    T maxer = this->at(0);
    int maxerIndex = 0;
    int max = this->size();
    for (int i = 0; i < max; i++) {
        const T& value = this->at(i);
        if (value > maxer) {
            maxer = value;
            maxerIndex = i;
        }
    }
    return std::make_pair(maxer, maxerIndex);
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul_zero_backend(const YTensor<T, dim1>& other) const{
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    int aw = this->shape(-1);
    // int bh = other.shape(-2);
    int bw = other.shape(-1);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        // 如果是二维矩阵，直接返回
        opShape = {ah, bw};
    } else {
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        opShape.push_back(ah); opShape.push_back(bw);
    }
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    auto mulop = thisMatView.binaryOpBroadcast(otherMatView, [&ah, &aw, &bw](const YTensor<T, 2>& a, const YTensor<T, 2>& b, YTensor<T, 2>& o) {
        #pragma omp parallel for simd collapse(2) proc_bind(close)
        for (int y=0; y<ah; ++y) {
            for (int x=0; x<bw; ++x) {
                T sum = 0;
                #pragma omp simd reduction(+:sum)
                for (int k=0; k<aw; ++k) {
                    sum = sum + a.at(y, k) * b.at(k, x);
                }
                o.at(y, x) = sum;
            }
        }
        return;
    }, "matmul_zero_backend", &opMatView, yt::infos::flopMatmul(ah, aw, bw));
    return op;
}

///////////// Eigen support ////////////////

#if YT_USE_EIGEN
template <typename T, int dim>
YTensor<typename YTensor<T, dim>::EigenMatrixMap, std::max(1, dim - 2)> YTensor<T, dim>::matViewEigen() const requires (dim > 2) {
    auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
    YTensor<EigenMatrixMap, std::max(1, dim - 2)> op;
    op._shape = newShape;
    op._stride = op.stride();
    op._offset = 0;
    op._element_size = sizeof(EigenMatrixMap);
    op._dtype = "eigen_map";
    int batchSize = op.size();
    
    // 使用封装函数分配数组内存
    op._data = yt::kernel::makeSharedPlacementArray<EigenMatrixMap>(batchSize);
    EigenMatrixMap* opData = reinterpret_cast<EigenMatrixMap*>(op._data.get());

    // 使用 placement new 构造每个 EigenMatrixMap
    const T* thisData = this->data_();
    for(int batchIdx = 0; batchIdx < batchSize; batchIdx++){
        auto coord = op.toCoord(batchIdx);
        Eigen::Stride<-1, -1> mstride(this->_stride[dim - 2], this->_stride[dim - 1]);
        T* matDataPtr = const_cast<T*>(thisData) + this->offset_(coord);  // 使用 offset_ 考虑张量自身的 _offset
        new (&opData[batchIdx]) EigenMatrixMap(matDataPtr, this->_shape[dim - 2], this->_shape[dim - 1], mstride);
    }
    return op;
}

template <typename T, int dim> typename
YTensor<T, dim>::EigenMatrixMap YTensor<T, dim>::matViewEigen() const requires (dim <= 2) {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    if constexpr (dim == 1){
        Eigen::Stride<-1, -1> mstride(0, this->_stride[0]);
        T* dataptr = const_cast<T*>(this->data_()) + this->_offset;
        EigenMatrixMap op(dataptr, this->_shape[0], 1, mstride);
        return op;
    }else{
        Eigen::Stride<-1, -1> mstride(this->_stride[0], this->_stride[1]);
        T* dataptr = const_cast<T*>(this->data_()) + this->_offset;
        EigenMatrixMap op(dataptr, this->_shape[0], this->_shape[1], mstride);
        return op;
    }
}

template <typename T, int dim> template<int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> YTensor<T, dim>::matmul_eigen_backend(const YTensor<T, dim1>& other) const{    
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    int aw = this->shape(-1);
    // int bh = other.shape(-2);
    int bw = other.shape(-1);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        // 如果是二维矩阵，直接返回
        opShape = {ah, bw};
    } else {
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        opShape.push_back(ah); opShape.push_back(bw);
    }
    YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    auto mulop = thisMatView.binaryOpBroadcast(otherMatView, [](const YTensor<T, 2>& a, const YTensor<T, 2>& b, YTensor<T, 2>& o) {
        auto mapA = a.matViewEigen();
        auto mapB = b.matViewEigen();
        auto mapO = o.matViewEigen();
        mapO.noalias() = mapA * mapB;
        return;
    }, "matmul_eigen_backend", &opMatView, yt::infos::flopMatmul(ah, aw, bw));
    return op;
}
#endif //YT_USE_EIGEN