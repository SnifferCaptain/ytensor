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
#if YT_USE_AVX2
#include "../include/kernel/gemm.hpp"
#endif

template <typename T, int dim>
template <typename Func, typename... Args>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::broadcastInplace(Func&& func, Args&&... tensors) {
    return yt::kernel::broadcastInplace(*this, std::forward<Func>(func), std::forward<Args>(tensors)...);
}

// 运算符生成规则
#define YT_YTENSOR_OPERATOR(OP, ENABLE_IF_T)                                                          \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    auto yt::YTensor<T, dim>::operator OP(const yt::YTensor<T, dim1>& other) const {                  \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return yt::kernel::broadcast([](const T& a, const T& b) {                                 \
                return a OP b;                                                                        \
            },*this, other);                                                                          \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    yt::YTensor<T, std::max(dim, dim1)>& yt::YTensor<T, dim>::operator OP##=(const yt::YTensor<T, dim1>& other) { \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return this->broadcastInplace([](T& a, const T& b) {                                      \
                return a OP## = b;                                                                    \
            }, other);                                                                                \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return this->broadcastInplace([](T& a, const T& b) {                                      \
                return a = a OP b;                                                                    \
            }, other);                                                                                \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), std::string(#OP) + "=");                        \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    auto yt::YTensor<T, dim>::operator OP(const T& other) const {                                     \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return yt::kernel::broadcast([](const T& a, const T& b) {                                 \
                return a OP b;                                                                        \
            }, *this, other);                                                                         \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator OP##=(const T& other) {                        \
        if constexpr (ENABLE_IF_T##_INPLACE<T>) {                                                     \
            return broadcastInplace([](T& a, const T& b) {                                            \
                return a OP## = b;                                                                    \
            }, other);                                                                                \
        } else if constexpr (ENABLE_IF_T<T>) {                                                        \
            return broadcastInplace([](T& a, const T& b) {                                            \
                return a = a OP b;                                                                    \
            }, other);                                                                                \
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
YT_YTENSOR_OPERATOR(|, yt::concepts::HAVE_OR)
YT_YTENSOR_OPERATOR(^, yt::concepts::HAVE_XOR)
YT_YTENSOR_OPERATOR(<<, yt::concepts::HAVE_LSHIFT)
YT_YTENSOR_OPERATOR(>>, yt::concepts::HAVE_RSHIFT)

#undef YT_YTENSOR_OPERATOR

// 比较运算符生成规则，返回YTensor<bool, dim>
#define YT_YTENSOR_CMP_OPERATOR(OP, ENABLE_IF_T)                                                      \
    template <typename T, int dim>                                                                    \
    template <int dim1>                                                                               \
    auto yt::YTensor<T, dim>::operator OP(const yt::YTensor<T, dim1>& other) const {                  \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return yt::kernel::broadcast([](const T& a, const T& b) {                                 \
                return a OP b;                                                                        \
            }, *this, other);                                                                         \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }                                                                                                 \
                                                                                                      \
    template <typename T, int dim>                                                                    \
    auto yt::YTensor<T, dim>::operator OP(const T& other) const {                                     \
        if constexpr (ENABLE_IF_T<T>) {                                                               \
            return yt::kernel::broadcast([](const T& a, const T& b) {                                 \
                return a OP b;                                                                        \
            }, *this, other);                                                                         \
        } else {                                                                                      \
            throwOperatorNotSupport(typeid(T).name(), #OP);                                           \
        }                                                                                             \
    }

YT_YTENSOR_CMP_OPERATOR(<, yt::concepts::HAVE_LT)
YT_YTENSOR_CMP_OPERATOR(<=, yt::concepts::HAVE_LE)
YT_YTENSOR_CMP_OPERATOR(>, yt::concepts::HAVE_GT)
YT_YTENSOR_CMP_OPERATOR(>=, yt::concepts::HAVE_GE)
YT_YTENSOR_CMP_OPERATOR(==, yt::concepts::HAVE_EQ)
YT_YTENSOR_CMP_OPERATOR(!=, yt::concepts::HAVE_NEQ)

#undef YT_YTENSOR_CMP_OPERATOR

template <typename T, int dim> template<int dim1>
auto yt::YTensor<T, dim>::operator%(const yt::YTensor<T, dim1>& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return yt::kernel::broadcast([](const T& a, const T& b) {
            return a % b;
        }, *this, other);
    }
    else if constexpr (std::is_floating_point_v<T>){
        return yt::kernel::broadcast([](const T& a, const T& b) {
            return std::fmod(a, b);
        }, *this, other);
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim> template<int dim1>
yt::YTensor<T, std::max(dim, dim1)>& yt::YTensor<T, dim>::operator%=(const yt::YTensor<T, dim1>& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return broadcastInplace([](T& a, const T& b) {
            a %= b;
        }, other);
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return broadcastInplace([](T& a, const T& b) {
            a = a % b;
        }, other);
    }
    else if constexpr (std::is_floating_point_v<T>){
        return broadcastInplace([](T& a, const T& b) {
            a = fmod(a, b);
        }, other);
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
auto yt::YTensor<T, dim>::operator%(const T& other) const {
    if constexpr (yt::concepts::HAVE_MOD<T>){
        return yt::kernel::broadcast([](const T& a, const T& b) {
            return a % b;
        }, *this, other);
    }
    else if constexpr (std::is_floating_point_v<T>){
        return yt::kernel::broadcast([](const T& a, const T& b) {
            return std::fmod(a, b);
        }, *this, other);
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%");
    }
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::YTensor<T, dim>::operator%=(const T& other){
    if constexpr (yt::concepts::HAVE_MOD_INPLACE<T>){
        return broadcastInplace([](T& a, const T& b) {
            a %= b;
        }, other);
    }
    else if constexpr (yt::concepts::HAVE_MOD<T>) {
        return broadcastInplace([](T& a, const T& b) {
            a = a % b;
        }, other);
    }
    else if constexpr (std::is_floating_point_v<T>){
        return broadcastInplace([](T& a, const T& b) {
            a = fmod(a, b);
        }, other);
    }
    else {
        std::string typeName = typeid(T).name();
        throwOperatorNotSupport(typeName, "%=");
    }
}

template <typename T, int dim>
yt::YTensor<yt::YTensor<T, 2>, std::max(1, dim - 2)> yt::YTensor<T, dim>::matView() const {
    // 将最后两个维度视作矩阵的视图，维度不足就填充1。
    static_assert(dim >= 1, "matView only support dim >= 1");
    using MatType = yt::YTensor<T, 2>;
    
    if constexpr (dim == 1){
        MatType mat;
        mat._shape = std::vector<int>({1, this->_shape[0]});
        mat._stride = std::vector<int>({0, this->_stride[0]});
        mat._offset = this->_offset;
        mat._element_size = sizeof(T);
        mat._dtype = yt::types::getTypeName<T>();
        mat._data = this->_data;
        
        yt::YTensor<MatType, 1> op;
        op._shape = std::vector<int>({1});
        op._stride = std::vector<int>({0});
        op._offset = 0;
        op._element_size = sizeof(MatType);
        op._dtype = "tensor_view";
        
        // 使用封装函数分配内存
        op._data = yt::kernel::makeSharedPlacement<MatType>(mat);
        return op;
    }else if constexpr (dim == 2){
        yt::YTensor<MatType, 1> op;
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
        yt::YTensor<MatType, std::max(1, dim - 2)> op;
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
            mat._dtype = yt::types::getTypeName<T>();
            mat._data = this->_data;
            new (&dataptr[batchIdx]) MatType(mat);
        }

        return op;
    }
}

template <typename T, int dim> template<int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::matmul(
    const yt::YTensor<T, dim1>& other, 
    yt::infos::MatmulBackend backend) const {
    static_assert(yt::concepts::HAVE_ADD<T> && yt::concepts::HAVE_MUL<T>, "Type must have add and mul in matmul");
    static_assert(dim >= 1 && dim1 >= 1, "matmul only support dim >= 1");
    if(this->shape(-1) != other.shape(-2)){
        throwShapeNotMatch("matmul", other.shape());
    }
    
    // 根据后端分派，各后端自己处理fastpath优化
    if constexpr (std::is_arithmetic_v<T>) {
        switch (backend) {
#if YT_USE_AVX2
            case yt::infos::MatmulBackend::AVX2:
                if constexpr (std::is_same_v<T, float>) return matmul_avx2_backend(other);
                [[fallthrough]];
#endif
#if YT_USE_EIGEN
            case yt::infos::MatmulBackend::Eigen: return matmul_eigen_backend(other);
#endif
            default: return matmul_naive_backend(other);
        }
    } else {
        return matmul_naive_backend(other);
    }
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::YTensor<T, dim>::sum(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    yt::YTensor<T, dim> op(newShape);
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
yt::YTensor<T, dim> yt::YTensor<T, dim>::sum(std::vector<int> axis) const requires (dim > 1) {
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

    yt::YTensor<T, dim> op(newShape);
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
T yt::YTensor<T, dim>::sum(int) const requires (dim == 1) {
    T sum = 0;
    int max = this->size();
    if (max * 1. > yt::infos::minParOps){
        #pragma omp parallel for reduction(+:sum) proc_bind(close)
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
yt::YTensor<T, dim> yt::YTensor<T, dim>::mean(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    int axisLen = newShape[axis];
    newShape[axis] = 1;
    yt::YTensor<T, dim> op(newShape);
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
yt::YTensor<T, dim> yt::YTensor<T, dim>::mean(std::vector<int> axes) const requires (dim > 1) {
    // 规范化轴并计算每个轴的长度
    std::vector<int> normalizedAxes;
    int totalN = 1;
    for (int ax : axes) {
        ax = (ax % dim + dim) % dim;
        normalizedAxes.push_back(ax);
        totalN *= this->shape(ax);
    }
    
    // 依次对每个轴使用mean(int axis)
    yt::YTensor<T, dim> result = *this;
    for (int ax : normalizedAxes) {
        result = result.mean(ax);
    }
    
    return result;
}

template <typename T, int dim>
T yt::YTensor<T, dim>::mean(int) const requires (dim == 1) {
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
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> yt::YTensor<T, dim>::max(int axis) const requires (dim > 1) {
    axis = (axis % dim + dim) % dim;
    auto newShape = this->shape();
    newShape[axis] = 1;
    yt::YTensor<T, dim> op(newShape);
    yt::YTensor<int, dim> opi(newShape);
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
std::pair<yt::YTensor<T, dim>, yt::YTensor<int, dim>> yt::YTensor<T, dim>::max(std::vector<int> axis) const requires (dim > 1) {
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

    yt::YTensor<T, dim> op(newShape);
    yt::YTensor<int, dim> opi(newShape);
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
std::pair<T, int> yt::YTensor<T, dim>::max(int)const requires (dim == 1) { 
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
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::matmul_naive_backend(const yt::YTensor<T, dim1>& other) const{
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
    yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    opMatView.broadcastInplace([&ah, &aw, &bw](yt::YTensor<T, 2>& o, const yt::YTensor<T, 2>& a, const yt::YTensor<T, 2>& b) {
        #pragma omp parallel for collapse(2) proc_bind(close)
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
    },thisMatView, otherMatView);
    return op;
}

///////////// Eigen support ////////////////

#if YT_USE_EIGEN
template <typename T, int dim>
yt::YTensor<typename yt::YTensor<T, dim>::EigenMatrixMap, std::max(1, dim - 2)> yt::YTensor<T, dim>::matViewEigen() const requires (dim > 2) {
    auto newShape = std::vector<int>(this->_shape.begin(), this->_shape.end() - 2);
    yt::YTensor<EigenMatrixMap, std::max(1, dim - 2)> op;
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
yt::YTensor<T, dim>::EigenMatrixMap yt::YTensor<T, dim>::matViewEigen() const requires (dim <= 2) {
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
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::matmul_eigen_backend(const yt::YTensor<T, dim1>& other) const{    
    int aw = this->shape(-1);
    int bw = other.shape(-1);
    
    // ==================== Fastpath检测 ====================
    if constexpr (dim > 2) {
        bool rightIs2D = (dim1 <= 2);
        if (!rightIs2D) {
            rightIs2D = true;
            for (int i = 0; i < dim1 - 2; ++i) {
                if (other.shape(i) != 1) { rightIs2D = false; break; }
            }
        }
        if (rightIs2D) {
            int contiguousStart = this->isContiguousFrom(0, -1);
            if (contiguousStart < dim - 1) {
                // 可以使用fastpath
                int outerSize = 1, innerRows = 1;
                for (int i = 0; i < contiguousStart; ++i) outerSize *= this->shape(i);
                for (int i = contiguousStart; i < dim - 1; ++i) innerRows *= this->shape(i);
                
                // 准备输出
                std::vector<int> opShape;
                for (int i = 0; i < dim - 1; ++i) opShape.push_back(this->shape(i));
                opShape.push_back(bw);
                yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
                
                // 右矩阵2D view
                yt::YTensor<T, 2> right2D;
                right2D._shape = {aw, bw}; right2D._stride = {other.stride_(-2), other.stride_(-1)};
                right2D._offset = other._offset; right2D._data = other._data;
                right2D._element_size = sizeof(T); right2D._dtype = yt::types::getTypeName<T>();
                
                int innerStride = (contiguousStart == 0) ? aw : this->stride_(contiguousStart);
                int opInnerStride = (contiguousStart == 0) ? bw : op.stride_(contiguousStart);
                
                for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                    int leftOffset = 0, opOffset = 0;
                    if (contiguousStart > 0) {
                        int idx = outerIdx;
                        for (int i = contiguousStart - 1; i >= 0; --i) {
                            int coord = idx % this->shape(i); idx /= this->shape(i);
                            leftOffset += coord * this->stride_(i);
                            opOffset += coord * op.stride_(i);
                        }
                    }
                    yt::YTensor<T, 2> leftFlat, opFlat;
                    leftFlat._shape = {innerRows, aw}; leftFlat._stride = {innerStride, this->stride_(-1)};
                    leftFlat._offset = this->_offset + leftOffset; leftFlat._data = this->_data;
                    leftFlat._element_size = sizeof(T); leftFlat._dtype = yt::types::getTypeName<T>();
                    opFlat._shape = {innerRows, bw}; opFlat._stride = {opInnerStride, 1};
                    opFlat._offset = opOffset; opFlat._data = op._data;
                    opFlat._element_size = sizeof(T); opFlat._dtype = yt::types::getTypeName<T>();
                    
                    auto mapA = leftFlat.matViewEigen();
                    auto mapB = right2D.matViewEigen();
                    auto mapC = opFlat.matViewEigen();
                    mapC.noalias() = mapA * mapB;
                }
                return op;
            }
        }
    }
    
    // ==================== 普通路径 ====================
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        opShape = {ah, bw};
    } else {
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        opShape.push_back(ah); opShape.push_back(bw);
    }
    yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();
    opMatView.broadcastInplace([](yt::YTensor<T, 2>& o, const yt::YTensor<T, 2>& a, const yt::YTensor<T, 2>& b) {
        auto mapA = a.matViewEigen();
        auto mapB = b.matViewEigen();
        auto mapO = o.matViewEigen();
        mapO.noalias() = mapA * mapB;
        return;
    }, thisMatView, otherMatView);
    return op;
}
#endif //YT_USE_EIGEN

/////////////// AVX2 GEMM 后端 ///////////////
#if YT_USE_AVX2

template <typename T, int dim> template<int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> yt::YTensor<T, dim>::matmul_avx2_backend(const yt::YTensor<T, dim1>& other) const requires std::is_same_v<T, float> {    
    int aw = this->shape(-1);
    int bw = other.shape(-1);
    
    // ==================== Fastpath检测 ====================
    if constexpr (dim > 2) {
        bool rightIs2D = (dim1 <= 2);
        if (!rightIs2D) {
            rightIs2D = true;
            for (int i = 0; i < dim1 - 2; ++i) {
                if (other.shape(i) != 1) { rightIs2D = false; break; }
            }
        }
        if (rightIs2D) {
            int contiguousStart = this->isContiguousFrom(0, -1);
            if (contiguousStart < dim - 1) {
                // 可以使用fastpath
                int outerSize = 1, innerRows = 1;
                for (int i = 0; i < contiguousStart; ++i) outerSize *= this->shape(i);
                for (int i = contiguousStart; i < dim - 1; ++i) innerRows *= this->shape(i);
                
                // 准备输出
                std::vector<int> opShape;
                for (int i = 0; i < dim - 1; ++i) opShape.push_back(this->shape(i));
                opShape.push_back(bw);
                yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
                
                // 右矩阵2D view
                yt::YTensor<T, 2> right2D;
                right2D._shape = {aw, bw}; right2D._stride = {other.stride_(-2), other.stride_(-1)};
                right2D._offset = other._offset; right2D._data = other._data;
                right2D._element_size = sizeof(T); right2D._dtype = yt::types::getTypeName<T>();
                
                int innerStride = (contiguousStart == 0) ? aw : this->stride_(contiguousStart);
                int opInnerStride = (contiguousStart == 0) ? bw : op.stride_(contiguousStart);
                
                for (int outerIdx = 0; outerIdx < outerSize; ++outerIdx) {
                    int leftOffset = 0, opOffset = 0;
                    if (contiguousStart > 0) {
                        int idx = outerIdx;
                        for (int i = contiguousStart - 1; i >= 0; --i) {
                            int coord = idx % this->shape(i); idx /= this->shape(i);
                            leftOffset += coord * this->stride_(i);
                            opOffset += coord * op.stride_(i);
                        }
                    }
                    yt::YTensor<T, 2> leftFlat, opFlat;
                    leftFlat._shape = {innerRows, aw}; leftFlat._stride = {innerStride, this->stride_(-1)};
                    leftFlat._offset = this->_offset + leftOffset; leftFlat._data = this->_data;
                    leftFlat._element_size = sizeof(T); leftFlat._dtype = yt::types::getTypeName<T>();
                    opFlat._shape = {innerRows, bw}; opFlat._stride = {opInnerStride, 1};
                    opFlat._offset = opOffset; opFlat._data = op._data;
                    opFlat._element_size = sizeof(T); opFlat._dtype = yt::types::getTypeName<T>();
                    
                    yt::kernel::gemm::matmul(
                        leftFlat.data(), right2D.data(), opFlat.data(),
                        innerRows, bw, aw,
                        static_cast<int64_t>(leftFlat.stride_(0)), static_cast<int64_t>(leftFlat.stride_(1)),
                        static_cast<int64_t>(right2D.stride_(0)), static_cast<int64_t>(right2D.stride_(1)),
                        static_cast<int64_t>(opFlat.stride_(0)), static_cast<int64_t>(opFlat.stride_(1)));
                }
                return op;
            }
        }
    }
    
    // ==================== 普通路径 ====================
    auto thisMatView = this->matView();
    auto otherMatView = other.matView();
    int ah = this->shape(-2);
    std::vector<int> opShape;
    if constexpr(yt::concepts::CONSTEXPR_MAX({dim, dim1, 2}) == 2){
        opShape = {ah, bw};
    } else {
        opShape = yt::kernel::computeBroadcastShape({thisMatView.shape(), otherMatView.shape()});
        opShape.push_back(ah); opShape.push_back(bw);
    }
    yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim, dim1, 2})> op(opShape);
    auto opMatView = op.matView();

    opMatView.broadcastInplace([](yt::YTensor<T, 2>& o, const yt::YTensor<T, 2>& a, const yt::YTensor<T, 2>& b) {
        int m = a.shape(0);
        int k = a.shape(1);
        int n = b.shape(1);
        auto aStride = a.stride_();
        auto bStride = b.stride_();
        auto oStride = o.stride_();
        yt::kernel::gemm::matmul(
            a.data(), b.data(), o.data(),
            m, n, k,
            static_cast<int64_t>(aStride[0]), static_cast<int64_t>(aStride[1]),
            static_cast<int64_t>(bStride[0]), static_cast<int64_t>(bStride[1]),
            static_cast<int64_t>(oStride[0]), static_cast<int64_t>(oStride[1])
        );
    }, thisMatView, otherMatView);
    return op;
}
#endif // YT_USE_AVX2