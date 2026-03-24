#include "../include/ytensor_concepts.hpp"

namespace yt::function::detail {

template <typename T>
inline T zero() {
    return static_cast<T>(0);
}

template <typename T>
inline T one() {
    return static_cast<T>(1);
}

template <typename T>
inline T expValue(const T& value) {
    using std::exp;
    return exp(value);
}

template <typename T>
inline T log1pValue(const T& value) {
    using std::log1p;
    return log1p(value);
}

template <typename T>
inline T tanhValue(const T& value) {
    using std::tanh;
    return tanh(value);
}

template <typename T>
inline T absValue(const T& value) {
    using std::abs;
    return abs(value);
}

template <typename T>
inline T stableSigmoid(const T& value) {
    const T zero_v = zero<T>();
    const T one_v = one<T>();
    if (value >= zero_v) {
        T z = expValue(-value);
        return one_v / (one_v + z);
    }
    T z = expValue(value);
    return z / (one_v + z);
}

template <typename T>
inline T stableSoftplus(const T& value) {
    const T zero_v = zero<T>();
    const T positive = value > zero_v ? value : zero_v;
    return positive + log1pValue(expValue(-absValue(value)));
}

}  // namespace yt::function::detail

template <typename T, int dim0, int dim1>
yt::YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> yt::function::matmul(const yt::YTensor<T, dim0>& a, const yt::YTensor<T, dim1>& b) {
    return a.matmul(b);
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::relu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    yt::YTensor<T, dim> op;
    if(order == 0){
        op = yt::kernel::broadcast([](const T& a) {
            return std::max(a, static_cast<T>(0));
        }, x);
    }
    else if(order == 1){
        op = yt::kernel::broadcast([](const T& a) {
            return static_cast<T>(a > 0);
        }, x);
    }
    else if(order > 1){
        op = yt::YTensor<T, dim>::zeros(x.shape());
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        op = yt::kernel::broadcast([&pow, &fact](const T& a) {
            if(a > 0){
                return std::pow(a, pow) / static_cast<T>(fact);
            }
            else{
                return static_cast<T>(0);
            }
        }, x);
    }
    return op;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::relu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            a = std::max(a, static_cast<T>(0));
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            a = static_cast<T>(a > 0);
        });
    }
    else if(order > 1){
        x.broadcastInplace([](T& a) {
            a = static_cast<T>(0); // 这里可以根据需要修改
        });
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        x.broadcastInplace([&pow, &fact](T& a) {
            if(a > 0){
                a = std::pow(a, pow) / static_cast<T>(fact);
            }
            else{
                a = static_cast<T>(0);
            }
        });
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::exp(const yt::YTensor<T, dim>& x, int) {
    return x.unaryOpTransform(0, [](const T& a, const T&){
        return std::exp(a);
    });
}

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::sigmoid(const yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            return yt::function::detail::stableSigmoid(a);
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            return sig * (yt::function::detail::one<T>() - sig);
        }, x);
    }
    else if(order == -1){
        return yt::function::softplus(x, 0);
    }
    else{
        throwNotSupport("yt::function::sigmoid", "order != 0, 1, -1");
    }
    return x;
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::softmax(const yt::YTensor<T, dim>& x, int axis) {
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;
    
    auto shape = x.shape();
    yt::YTensor<T, dim> output(shape);
    
    // 快速路径：连续张量 - 使用优化实现（Flash Attention 风格）
    if (x.isContiguous() && output.isContiguous()) {
        int64_t dim_size = shape[axis];
        
        // 计算外层大小和内层大小
        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        
        int64_t inner_size = 1;
        for (int i = axis + 1; i < dim; ++i) {
            inner_size *= shape[i];
        }
        
        int64_t dim_stride = inner_size;
        int64_t outer_stride = dim_size * dim_stride;
        
        const T* input_data_base = x.data();
        T* output_data_base = output.data();
        
        // 针对连续数据的优化循环
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                const T* input_data = input_data_base + outer_idx * outer_stride + inner_idx;
                T* output_data = output_data_base + outer_idx * outer_stride + inner_idx;
                
                // 步骤 1: 获取最大值（数值稳定性）
                T max_val = input_data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, input_data[d * dim_stride]);
                }
                
                // 步骤 2: 计算 exp(x - max) 并累加和（融合操作）
                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(input_data[d * dim_stride] - max_val);
                    output_data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }
                
                // 步骤 3: 用和归一化
                for (int64_t d = 0; d < dim_size; ++d) {
                    output_data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        // 通用路径：支持非连续张量
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }
        
        int64_t total_iterations = 1;
        for (int s : iter_shape) {
            total_iterations *= s;
        }
        
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            // 将平铺索引转换为多维索引（不包括 axis 维度）
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = iter_shape.size() - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }
            
            // 插入 axis 维度，转换为完整索引
            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0); // 占位符
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }
            
            // 融合传递：max、exp 和 sum 在一次循环中完成
            T max_val = std::numeric_limits<T>::lowest();
            
            // 第一遍：找最大值
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }
            
            // 第二遍：计算 exp(x - max) 并求和
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                output.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }
            
            // 第三遍：归一化
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) /= sum_exp;
            }
        }
    }
    
    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::softmax_(yt::YTensor<T, dim>& x, int axis) {
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;
    
    auto shape = x.shape();
    
    // 快速路径：连续张量 - 使用优化实现（Flash Attention 风格）
    if (x.isContiguous()) {
        int64_t dim_size = shape[axis];
        
        // 计算外层大小和内层大小
        int64_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape[i];
        }
        
        int64_t inner_size = 1;
        for (int i = axis + 1; i < dim; ++i) {
            inner_size *= shape[i];
        }
        
        int64_t dim_stride = inner_size;
        int64_t outer_stride = dim_size * dim_stride;
        
        T* data_base = x.data();
        
        // 针对连续数据的优化循环
        #pragma omp parallel for collapse(2) if(outer_size * inner_size > 1024)
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
                T* data = data_base + outer_idx * outer_stride + inner_idx;
                
                // 步骤 1: 获取最大值（数值稳定性）
                T max_val = data[0];
                for (int64_t d = 1; d < dim_size; ++d) {
                    max_val = std::max(max_val, data[d * dim_stride]);
                }
                
                // 步骤 2: 计算 exp(x - max) 并累加和（融合操作，原地更新）
                T sum_exp = static_cast<T>(0);
                for (int64_t d = 0; d < dim_size; ++d) {
                    T exp_val = std::exp(data[d * dim_stride] - max_val);
                    data[d * dim_stride] = exp_val;
                    sum_exp += exp_val;
                }
                
                // 步骤 3: 用和归一化
                for (int64_t d = 0; d < dim_size; ++d) {
                    data[d * dim_stride] /= sum_exp;
                }
            }
        }
    } else {
        // 通用路径：支持非连续张量
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }
        
        int64_t total_iterations = 1;
        for (int s : iter_shape) {
            total_iterations *= s;
        }
        
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            // 将平铺索引转换为多维索引（不包括 axis 维度）
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = iter_shape.size() - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }
            
            // 插入 axis 维度，转换为完整索引
            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0); // 占位符
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }
            
            // 融合传递：max、exp 和 sum 在一次循环中完成
            T max_val = std::numeric_limits<T>::lowest();
            
            // 第一遍：找最大值
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }
            
            // 第二遍：计算 exp(x - max) 并求和（原地更新）
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                x.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }
            
            // 第三遍：归一化
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) /= sum_exp;
            }
        }
    }
    
    return x;
}

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::scaledDotProductAttention(
    yt::YTensor<T, dim>& query,// [..., n0, c0]
    yt::YTensor<T, dim>& key,// [..., n1, c0]
    yt::YTensor<T, dim>& value,// [..., n1, c1]
    T scale,
    yt::YTensor<bool, 2>* mask,
    yt::YTensor<T, 2>* bias,
    sdpaBackend backend
) {
    if(static_cast<T>(0.0) == scale){
        // auto
        scale = static_cast<T>(1.0) / std::sqrt(static_cast<T>(query.shape(-1)));
    }
    if(backend == sdpaBackend::MATH){
        // auto t0 = std::chrono::high_resolution_clock::now();
        // auto score = yt::function::matmul(query, key.transpose());// [..., n0, n1]
        // auto t1 =  std::chrono::high_resolution_clock::now();
        // score.binaryOpTransformInplace(scale, [](T& a, const T& b) {
        //     a *= b; // scale
        // });
        // auto t2 =  std::chrono::high_resolution_clock::now();
        // if(mask != nullptr){
        //     if(mask->shape(0) != score.shape(-2) || mask->shape(1) != score.shape(-1)){
        //         throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
        //     }
        //     score += *mask;
        // }
        // auto t3 =  std::chrono::high_resolution_clock::now();
        // yt::function::softmax_(score, -1);// [..., n0, n1] inplace
        // auto t4 =  std::chrono::high_resolution_clock::now();
        // auto op = yt::function::matmul(score, value);// [..., n0, c1]
        // auto t5 =  std::chrono::high_resolution_clock::now();
        // double dt0 = std::chrono::duration<double>(t1 - t0).count() * 1e6;
        // double dt1 = std::chrono::duration<double>(t2 - t1).count() * 1e6;
        // double dt2 = std::chrono::duration<double>(t3 - t2).count() * 1e6;
        // double dt3 = std::chrono::duration<double>(t4 - t3).count() * 1e6;
        // double dt4 = std::chrono::duration<double>(t5 - t4).count() * 1e6;
        // std::cout << "QK: " << dt0 << "us" << std::endl;
        // std::cout << "scale: " << dt1 << "us" << std::endl;
        // std::cout << "mask: " << dt2 << "us" << std::endl;
        // std::cout << "softmax: " << dt3 << "us" << std::endl;
        // std::cout << "V: " << dt4 << "us" << std::endl;
        // return op;

        auto keyT = key.transpose();
        yt::YTensor<T, dim> score;

        if(mask != nullptr){
            if(mask->shape(0) != query.shape(-2) || mask->shape(1) != key.shape(-2)){
                throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
            }
            score = query.masked_matmul(
                keyT,
                [mask](int row, int col) {
                    return mask->at(row, col);
                },
                static_cast<T>(-1e9)
            );
        }
        else{
            score = yt::function::matmul(query, keyT);// [..., n0, n1]
        }

        score.broadcastInplace([](T& a, const T& b) {
            a *= b; // scale
        }, scale);
        if(bias != nullptr){
            if(bias->shape(0) != score.shape(-2) || bias->shape(1) != score.shape(-1)){
                throw std::invalid_argument("Bias shape must match the last two dimensions of the score tensor.");
            }
            score += *bias;
        }
        yt::function::softmax_(score, -1);// [..., n0, n1] inplace
        auto op = yt::function::matmul(score, value);// [..., n0, c1]
        return op;
    }
    else{
        throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
        return yt::YTensor<T, dim>();
    }
}


// ========== exp_ (inplace) ==========

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::exp_(yt::YTensor<T, dim>& x, int order) {
    (void)order;
    // exp 的任意阶导数/不计常数的原函数仍然是 exp
    x.broadcastInplace([](T& a) {
        using std::exp;
        a = exp(a);
    });
    return x;
}

// ========== sigmoid_ (inplace) ==========

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::sigmoid_(yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        x.broadcastInplace([](T& a) {
            a = yt::function::detail::stableSigmoid(a);
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            a = sig * (yt::function::detail::one<T>() - sig);
        });
    }
    else if(order == -1){
        return yt::function::softplus_(x, 0);
    }
    else{
        throwNotSupport("yt::function::sigmoid_", "order != 0, 1, -1");
    }
    return x;
}

// ========== leakyRelu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::leakyRelu(const yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::leakyRelu()");
    if(order == 0){
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? a : alpha * a;
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? static_cast<T>(1) : alpha;
        }, x);
    }
    else{
        throwNotSupport("yt::function::leakyRelu", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::leakyRelu_(yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::leakyRelu_()");
    if(order == 0){
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? a : alpha * a;
        });
    }
    else if(order == 1){
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? static_cast<T>(1) : alpha;
        });
    }
    else{
        throwNotSupport("yt::function::leakyRelu_", "order != 0, 1");
    }
    return x;
}

// ========== elu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::elu(const yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::elu()");
    if(order == 0){
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? a : alpha * (std::exp(a) - static_cast<T>(1));
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([alpha](const T& a) {
            return a > static_cast<T>(0) ? static_cast<T>(1) : alpha * std::exp(a);
        }, x);
    }
    else{
        throwNotSupport("yt::function::elu", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::elu_(yt::YTensor<T, dim>& x, T alpha, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::elu_()");
    if(order == 0){
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? a : alpha * (std::exp(a) - static_cast<T>(1));
        });
    }
    else if(order == 1){
        x.broadcastInplace([alpha](T& a) {
            a = a > static_cast<T>(0) ? static_cast<T>(1) : alpha * std::exp(a);
        });
    }
    else{
        throwNotSupport("yt::function::elu_", "order != 0, 1");
    }
    return x;
}

// ========== selu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::selu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::selu()");
    constexpr T lambda = static_cast<T>(1.0507009873554804934193349852946);
    constexpr T alpha  = static_cast<T>(1.6732632423543772848170429916717);
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            return a > static_cast<T>(0) ? l * a : l * al * (std::exp(a) - static_cast<T>(1));
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            return a > static_cast<T>(0) ? l : l * al * std::exp(a);
        }, x);
    }
    else{
        throwNotSupport("yt::function::selu", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::selu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::selu_()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            a = a > static_cast<T>(0) ? l * a : l * al * (std::exp(a) - static_cast<T>(1));
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            constexpr T l = static_cast<T>(1.0507009873554804934193349852946);
            constexpr T al = static_cast<T>(1.6732632423543772848170429916717);
            a = a > static_cast<T>(0) ? l : l * al * std::exp(a);
        });
    }
    else{
        throwNotSupport("yt::function::selu_", "order != 0, 1");
    }
    return x;
}

// ========== gelu ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::gelu(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::gelu()");
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            return static_cast<T>(0.5) * a * (static_cast<T>(1) + std::erf(a / sqrt2));
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            constexpr T inv_sqrt2pi = static_cast<T>(0.3989422804014326779399460599343);
            T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(a / sqrt2));
            T pdf = inv_sqrt2pi * std::exp(static_cast<T>(-0.5) * a * a);
            return cdf + a * pdf;
        }, x);
    }
    else{
        throwNotSupport("yt::function::gelu", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::gelu_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::gelu_()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            a = static_cast<T>(0.5) * a * (static_cast<T>(1) + std::erf(a / sqrt2));
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            constexpr T sqrt2 = static_cast<T>(1.4142135623730950488016887242097);
            constexpr T inv_sqrt2pi = static_cast<T>(0.3989422804014326779399460599343);
            T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + std::erf(a / sqrt2));
            T pdf = inv_sqrt2pi * std::exp(static_cast<T>(-0.5) * a * a);
            a = cdf + a * pdf;
        });
    }
    else{
        throwNotSupport("yt::function::gelu_", "order != 0, 1");
    }
    return x;
}

// ========== tanh ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::tanh(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::tanh()");
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            return std::tanh(a);
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            T t = std::tanh(a);
            return static_cast<T>(1) - t * t;
        }, x);
    }
    else{
        throwNotSupport("yt::function::tanh", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::tanh_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::tanh_()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            a = std::tanh(a);
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            T t = std::tanh(a);
            a = static_cast<T>(1) - t * t;
        });
    }
    else{
        throwNotSupport("yt::function::tanh_", "order != 0, 1");
    }
    return x;
}

// ========== swish (SiLU) ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::swish(const yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            return a * sig;
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            return sig * (yt::function::detail::one<T>() + a * (yt::function::detail::one<T>() - sig));
        }, x);
    }
    else{
        throwNotSupport("yt::function::swish", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::swish_(yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        x.broadcastInplace([](T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            a = a * sig;
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            T sig = yt::function::detail::stableSigmoid(a);
            a = sig * (yt::function::detail::one<T>() + a * (yt::function::detail::one<T>() - sig));
        });
    }
    else{
        throwNotSupport("yt::function::swish_", "order != 0, 1");
    }
    return x;
}

// ========== softplus ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::softplus(const yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            return yt::function::detail::stableSoftplus(a);
        }, x);
    }
    else if(order == 1){
        return yt::function::sigmoid(x, 0);
    }
    else{
        throwNotSupport("yt::function::softplus", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::softplus_(yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        x.broadcastInplace([](T& a) {
            a = yt::function::detail::stableSoftplus(a);
        });
    }
    else if(order == 1){
        return yt::function::sigmoid_(x, 0);
    }
    else{
        throwNotSupport("yt::function::softplus_", "order != 0, 1");
    }
    return x;
}

// ========== mish ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::mish(const yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            T sp = yt::function::detail::stableSoftplus(a);
            return a * yt::function::detail::tanhValue(sp);
        }, x);
    }
    else if(order == 1){
        // mish'(x) = tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
        return yt::kernel::broadcast([](const T& a) {
            T sp = yt::function::detail::stableSoftplus(a);
            T tsp = yt::function::detail::tanhValue(sp);
            T sig = yt::function::detail::stableSigmoid(a);
            return tsp + a * (yt::function::detail::one<T>() - tsp * tsp) * sig;
        }, x);
    }
    else{
        throwNotSupport("yt::function::mish", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::mish_(yt::YTensor<T, dim>& x, int order) {
    if(order == 0){
        x.broadcastInplace([](T& a) {
            T sp = yt::function::detail::stableSoftplus(a);
            a = a * yt::function::detail::tanhValue(sp);
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            T sp = yt::function::detail::stableSoftplus(a);
            T tsp = yt::function::detail::tanhValue(sp);
            T sig = yt::function::detail::stableSigmoid(a);
            a = tsp + a * (yt::function::detail::one<T>() - tsp * tsp) * sig;
        });
    }
    else{
        throwNotSupport("yt::function::mish_", "order != 0, 1");
    }
    return x;
}

// ========== hardSigmoid ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::hardSigmoid(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSigmoid()");
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            T v = a / static_cast<T>(6) + static_cast<T>(0.5);
            return std::max(static_cast<T>(0), std::min(static_cast<T>(1), v));
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            return (a > static_cast<T>(-3) && a < static_cast<T>(3))
                ? static_cast<T>(1) / static_cast<T>(6)
                : static_cast<T>(0);
        }, x);
    }
    else{
        throwNotSupport("yt::function::hardSigmoid", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::hardSigmoid_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSigmoid_()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            T v = a / static_cast<T>(6) + static_cast<T>(0.5);
            a = std::max(static_cast<T>(0), std::min(static_cast<T>(1), v));
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            a = (a > static_cast<T>(-3) && a < static_cast<T>(3))
                ? static_cast<T>(1) / static_cast<T>(6)
                : static_cast<T>(0);
        });
    }
    else{
        throwNotSupport("yt::function::hardSigmoid_", "order != 0, 1");
    }
    return x;
}

// ========== hardSwish ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::hardSwish(const yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSwish()");
    if(order == 0){
        return yt::kernel::broadcast([](const T& a) {
            if(a <= static_cast<T>(-3)) return static_cast<T>(0);
            if(a >= static_cast<T>(3)) return a;
            return a * (a + static_cast<T>(3)) / static_cast<T>(6);
        }, x);
    }
    else if(order == 1){
        return yt::kernel::broadcast([](const T& a) {
            if(a <= static_cast<T>(-3)) return static_cast<T>(0);
            if(a >= static_cast<T>(3)) return static_cast<T>(1);
            return (static_cast<T>(2) * a + static_cast<T>(3)) / static_cast<T>(6);
        }, x);
    }
    else{
        throwNotSupport("yt::function::hardSwish", "order != 0, 1");
    }
    return x;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::hardSwish_(yt::YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::hardSwish_()");
    if(order == 0){
        x.broadcastInplace([](T& a) {
            if(a <= static_cast<T>(-3)){ a = static_cast<T>(0); return; }
            if(a >= static_cast<T>(3)) return;
            a = a * (a + static_cast<T>(3)) / static_cast<T>(6);
        });
    }
    else if(order == 1){
        x.broadcastInplace([](T& a) {
            if(a <= static_cast<T>(-3)){ a = static_cast<T>(0); return; }
            if(a >= static_cast<T>(3)){ a = static_cast<T>(1); return; }
            a = (static_cast<T>(2) * a + static_cast<T>(3)) / static_cast<T>(6);
        });
    }
    else{
        throwNotSupport("yt::function::hardSwish_", "order != 0, 1");
    }
    return x;
}

// ========== logsumexp ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logsumexp(const yt::YTensor<T, dim>& x, int axis) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logsumexp()");
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();
    // 输出shape：axis维度为1
    auto out_shape = shape;
    out_shape[axis] = 1;
    yt::YTensor<T, dim> output(out_shape);

    // 构建迭代shape（去掉axis维度）
    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) {
        total_iterations *= s;
    }

    #pragma omp parallel for if(total_iterations > 1024)
    for (int64_t idx = 0; idx < total_iterations; ++idx) {
        // 将平铺索引转换为多维索引（不包括 axis 维度）
        std::vector<int> iter_indices(iter_shape.size());
        int64_t temp_idx = idx;
        for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
            iter_indices[i] = temp_idx % iter_shape[i];
            temp_idx /= iter_shape[i];
        }

        // 构建完整索引
        std::vector<int> full_indices;
        int iter_pos = 0;
        for (int i = 0; i < dim; ++i) {
            if (i == axis) {
                full_indices.push_back(0);
            } else {
                full_indices.push_back(iter_indices[iter_pos++]);
            }
        }

        // 找最大值（数值稳定性）
        T max_val = std::numeric_limits<T>::lowest();
        for (int d = 0; d < shape[axis]; ++d) {
            full_indices[axis] = d;
            max_val = std::max(max_val, x.at(full_indices));
        }

        // 计算 sum(exp(x - max))
        T sum_exp = static_cast<T>(0);
        for (int d = 0; d < shape[axis]; ++d) {
            full_indices[axis] = d;
            sum_exp += std::exp(x.at(full_indices) - max_val);
        }

        // 输出 = log(sum_exp) + max_val
        full_indices[axis] = 0;
        output.at(full_indices) = std::log(sum_exp) + max_val;
    }

    return output;
}

// ========== logSoftmax ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::logSoftmax(const yt::YTensor<T, dim>& x, int axis, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logSoftmax()");
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();
    yt::YTensor<T, dim> output(shape);

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) {
        total_iterations *= s;
    }

    if(order == 0){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 找最大值
            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            // 计算 sum(exp(x - max))
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                sum_exp += std::exp(x.at(full_indices) - max_val);
            }
            T lse = std::log(sum_exp) + max_val;

            // logSoftmax = x - logsumexp
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) = x.at(full_indices) - lse;
            }
        }
    }
    else if(order == 1){
        // logSoftmax的导数对角线: 1 - softmax(x)
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 先计算softmax
            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(x.at(full_indices) - max_val);
                output.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            // 导数 = 1 - softmax
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) = static_cast<T>(1) - output.at(full_indices) / sum_exp;
            }
        }
    }
    else{
        throwNotSupport("yt::function::logSoftmax", "order != 0, 1");
    }

    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::logSoftmax_(yt::YTensor<T, dim>& x, int axis, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::logSoftmax_()");
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) {
        total_iterations *= s;
    }

    if(order == 0){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                sum_exp += std::exp(x.at(full_indices) - max_val);
            }
            T lse = std::log(sum_exp) + max_val;

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) -= lse;
            }
        }
    }
    else if(order == 1){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, x.at(full_indices));
            }

            // 需要临时存储softmax值
            std::vector<T> sm(shape[axis]);
            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                sm[d] = std::exp(x.at(full_indices) - max_val);
                sum_exp += sm[d];
            }

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) = static_cast<T>(1) - sm[d] / sum_exp;
            }
        }
    }
    else{
        throwNotSupport("yt::function::logSoftmax_", "order != 0, 1");
    }

    return x;
}

// ========== layerNorm ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::layerNorm(const yt::YTensor<T, dim>& x, int axis, T eps, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::layerNorm()");
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();
    yt::YTensor<T, dim> output(shape);

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) {
        total_iterations *= s;
    }

    int64_t n = shape[axis];

    if(order == 0){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 计算均值
            T mean_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                mean_val += x.at(full_indices);
            }
            mean_val /= static_cast<T>(n);

            // 计算方差
            T var_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                T diff = x.at(full_indices) - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<T>(n);

            T inv_std = static_cast<T>(1) / std::sqrt(var_val + eps);

            // 归一化
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) = (x.at(full_indices) - mean_val) * inv_std;
            }
        }
    }
    else if(order == 1){
        // layerNorm导数（简化版，对角线元素）：
        // d/dx_i [(x_i - mu) / sigma] ≈ (1/sigma) * (1 - 1/N) 当忽略mu和sigma对x_i的依赖时
        // 完整版：(1/sigma) * (1 - 1/N - (x_i - mu)^2 / (N * sigma^2))
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            T mean_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                mean_val += x.at(full_indices);
            }
            mean_val /= static_cast<T>(n);

            T var_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                T diff = x.at(full_indices) - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<T>(n);

            T inv_std = static_cast<T>(1) / std::sqrt(var_val + eps);
            T inv_n = static_cast<T>(1) / static_cast<T>(n);

            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                T x_hat = (x.at(full_indices) - mean_val) * inv_std;
                output.at(full_indices) = inv_std * (static_cast<T>(1) - inv_n - x_hat * x_hat * inv_n);
            }
        }
    }
    else{
        throwNotSupport("yt::function::layerNorm", "order != 0, 1");
    }

    return output;
}

template<typename T, int dim>
yt::YTensor<T, dim>& yt::function::layerNorm_(yt::YTensor<T, dim>& x, int axis, T eps, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::layerNorm_()");
    axis = (axis % dim + dim) % dim;

    auto shape = x.shape();

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) {
        total_iterations *= s;
    }

    int64_t n = shape[axis];

    if(order == 0){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            T mean_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                mean_val += x.at(full_indices);
            }
            mean_val /= static_cast<T>(n);

            T var_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                T diff = x.at(full_indices) - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<T>(n);

            T inv_std = static_cast<T>(1) / std::sqrt(var_val + eps);

            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) = (x.at(full_indices) - mean_val) * inv_std;
            }
        }
    }
    else if(order == 1){
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            T mean_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                mean_val += x.at(full_indices);
            }
            mean_val /= static_cast<T>(n);

            T var_val = static_cast<T>(0);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                T diff = x.at(full_indices) - mean_val;
                var_val += diff * diff;
            }
            var_val /= static_cast<T>(n);

            T inv_std = static_cast<T>(1) / std::sqrt(var_val + eps);
            T inv_n = static_cast<T>(1) / static_cast<T>(n);

            // 需要临时存储归一化值
            std::vector<T> x_hat(n);
            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                x_hat[d] = (x.at(full_indices) - mean_val) * inv_std;
            }

            for (int d = 0; d < n; ++d) {
                full_indices[axis] = d;
                x.at(full_indices) = inv_std * (static_cast<T>(1) - inv_n - x_hat[d] * x_hat[d] * inv_n);
            }
        }
    }
    else{
        throwNotSupport("yt::function::layerNorm_", "order != 0, 1");
    }

    return x;
}

// ========== maxPool1d ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::maxPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::maxPool1d()");
    axis = (axis % dim + dim) % dim;
    if(stride < 0) stride = kernelSize;

    auto shape = x.shape();
    int input_size = shape[axis];
    int output_size = (input_size - kernelSize) / stride + 1;
    if(output_size <= 0){
        throw std::invalid_argument("yt::function::maxPool1d: kernelSize too large for input dimension");
    }

    if(order == 0){
        // 输出shape: axis维度变为output_size
        auto out_shape = shape;
        out_shape[axis] = output_size;
        yt::YTensor<T, dim> output(out_shape);

        // 构建迭代shape
        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) total_iterations *= s;

        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> in_indices, out_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    in_indices.push_back(0);
                    out_indices.push_back(0);
                } else {
                    in_indices.push_back(iter_indices[iter_pos]);
                    out_indices.push_back(iter_indices[iter_pos]);
                    iter_pos++;
                }
            }

            for (int o = 0; o < output_size; ++o) {
                int start = o * stride;
                in_indices[axis] = start;
                T max_val = x.at(in_indices);
                for (int k = 1; k < kernelSize; ++k) {
                    in_indices[axis] = start + k;
                    max_val = std::max(max_val, x.at(in_indices));
                }
                out_indices[axis] = o;
                output.at(out_indices) = max_val;
            }
        }

        return output;
    }
    else if(order == 1){
        // 返回与输入同shape的梯度掩码：max位置为1，其他为0
        yt::YTensor<T, dim> grad = yt::YTensor<T, dim>::zeros(shape);

        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) total_iterations *= s;

        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            for (int o = 0; o < output_size; ++o) {
                int start = o * stride;
                full_indices[axis] = start;
                T max_val = x.at(full_indices);
                int max_pos = start;
                for (int k = 1; k < kernelSize; ++k) {
                    full_indices[axis] = start + k;
                    T v = x.at(full_indices);
                    if(v > max_val){
                        max_val = v;
                        max_pos = start + k;
                    }
                }
                full_indices[axis] = max_pos;
                grad.at(full_indices) += static_cast<T>(1);
            }
        }

        return grad;
    }
    else{
        throwNotSupport("yt::function::maxPool1d", "order != 0, 1");
    }
    return x;
}

// ========== avgPool1d ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::avgPool1d(const yt::YTensor<T, dim>& x, int kernelSize, int stride, int axis, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in yt::function::avgPool1d()");
    axis = (axis % dim + dim) % dim;
    if(stride < 0) stride = kernelSize;

    auto shape = x.shape();
    int input_size = shape[axis];
    int output_size = (input_size - kernelSize) / stride + 1;
    if(output_size <= 0){
        throw std::invalid_argument("yt::function::avgPool1d: kernelSize too large for input dimension");
    }

    if(order == 0){
        auto out_shape = shape;
        out_shape[axis] = output_size;
        yt::YTensor<T, dim> output(out_shape);

        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) total_iterations *= s;

        T inv_k = static_cast<T>(1) / static_cast<T>(kernelSize);

        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> in_indices, out_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    in_indices.push_back(0);
                    out_indices.push_back(0);
                } else {
                    in_indices.push_back(iter_indices[iter_pos]);
                    out_indices.push_back(iter_indices[iter_pos]);
                    iter_pos++;
                }
            }

            for (int o = 0; o < output_size; ++o) {
                int start = o * stride;
                T sum_val = static_cast<T>(0);
                for (int k = 0; k < kernelSize; ++k) {
                    in_indices[axis] = start + k;
                    sum_val += x.at(in_indices);
                }
                out_indices[axis] = o;
                output.at(out_indices) = sum_val * inv_k;
            }
        }

        return output;
    }
    else if(order == 1){
        // 返回与输入同shape的梯度：每个元素被包含在多少个窗口中 / kernelSize
        yt::YTensor<T, dim> grad = yt::YTensor<T, dim>::zeros(shape);

        T inv_k = static_cast<T>(1) / static_cast<T>(kernelSize);

        std::vector<int> iter_shape;
        for (int i = 0; i < dim; ++i) {
            if (i != axis) {
                iter_shape.push_back(shape[i]);
            }
        }

        int64_t total_iterations = 1;
        for (int s : iter_shape) total_iterations *= s;

        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            for (int o = 0; o < output_size; ++o) {
                int start = o * stride;
                for (int k = 0; k < kernelSize; ++k) {
                    full_indices[axis] = start + k;
                    grad.at(full_indices) += inv_k;
                }
            }
        }

        return grad;
    }
    else{
        throwNotSupport("yt::function::avgPool1d", "order != 0, 1");
    }
    return x;
}

// ========== mseLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::mseLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::mseLoss()");
    if(order == 0){
        return yt::kernel::broadcast([](const T& a, const T& b) {
            T diff = a - b;
            return diff * diff;
        }, input, target);
    }
    else if(order == 1){
        // d/d(input) [(input - target)^2] = 2 * (input - target)
        return yt::kernel::broadcast([](const T& a, const T& b) {
            return static_cast<T>(2) * (a - b);
        }, input, target);
    }
    else{
        throwNotSupport("yt::function::mseLoss", "order != 0, 1");
    }
    return input;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::mseLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::mseLoss_()");
    if(order == 0){
        input.broadcastInplace([](T& a, const T& b) {
            T diff = a - b;
            a = diff * diff;
        }, target);
    }
    else if(order == 1){
        input.broadcastInplace([](T& a, const T& b) {
            a = static_cast<T>(2) * (a - b);
        }, target);
    }
    else{
        throwNotSupport("yt::function::mseLoss_", "order != 0, 1");
    }
    return input;
}

// ========== crossEntropyLoss ==========

template<typename T, int dim>
yt::YTensor<T, dim> yt::function::crossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int axis, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::crossEntropyLoss()");
    axis = (axis % dim + dim) % dim;

    auto shape = input.shape();
    yt::YTensor<T, dim> output(shape);

    std::vector<int> iter_shape;
    for (int i = 0; i < dim; ++i) {
        if (i != axis) {
            iter_shape.push_back(shape[i]);
        }
    }

    int64_t total_iterations = 1;
    for (int s : iter_shape) total_iterations *= s;

    if(order == 0){
        // 交叉熵: -sum(target * log(softmax(input)))，按元素返回 -target * logSoftmax(input)
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 计算logSoftmax
            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, input.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                sum_exp += std::exp(input.at(full_indices) - max_val);
            }
            T lse = std::log(sum_exp) + max_val;

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T log_sm = input.at(full_indices) - lse;
                output.at(full_indices) = -target.at(full_indices) * log_sm;
            }
        }
    }
    else if(order == 1){
        // 梯度: softmax(input) - target
        #pragma omp parallel for if(total_iterations > 1024)
        for (int64_t idx = 0; idx < total_iterations; ++idx) {
            std::vector<int> iter_indices(iter_shape.size());
            int64_t temp_idx = idx;
            for (int i = static_cast<int>(iter_shape.size()) - 1; i >= 0; --i) {
                iter_indices[i] = temp_idx % iter_shape[i];
                temp_idx /= iter_shape[i];
            }

            std::vector<int> full_indices;
            int iter_pos = 0;
            for (int i = 0; i < dim; ++i) {
                if (i == axis) {
                    full_indices.push_back(0);
                } else {
                    full_indices.push_back(iter_indices[iter_pos++]);
                }
            }

            // 计算softmax
            T max_val = std::numeric_limits<T>::lowest();
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                max_val = std::max(max_val, input.at(full_indices));
            }

            T sum_exp = static_cast<T>(0);
            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                T exp_val = std::exp(input.at(full_indices) - max_val);
                output.at(full_indices) = exp_val;
                sum_exp += exp_val;
            }

            for (int d = 0; d < shape[axis]; ++d) {
                full_indices[axis] = d;
                output.at(full_indices) = output.at(full_indices) / sum_exp - target.at(full_indices);
            }
        }
    }
    else{
        throwNotSupport("yt::function::crossEntropyLoss", "order != 0, 1");
    }

    return output;
}

// ========== binaryCrossEntropyLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::binaryCrossEntropyLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::binaryCrossEntropyLoss()");
    constexpr T eps = static_cast<T>(1e-7);
    if(order == 0){
        return yt::kernel::broadcast([eps](const T& a, const T& b) {
            T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
            return -(b * std::log(clamped) + (static_cast<T>(1) - b) * std::log(static_cast<T>(1) - clamped));
        }, input, target);
    }
    else if(order == 1){
        // d/d(input) = -target/input + (1-target)/(1-input)
        return yt::kernel::broadcast([eps](const T& a, const T& b) {
            T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
            return -b / clamped + (static_cast<T>(1) - b) / (static_cast<T>(1) - clamped);
        }, input, target);
    }
    else{
        throwNotSupport("yt::function::binaryCrossEntropyLoss", "order != 0, 1");
    }
    return input;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::binaryCrossEntropyLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::binaryCrossEntropyLoss_()");
    constexpr T eps = static_cast<T>(1e-7);
    if(order == 0){
        input.broadcastInplace([eps](T& a, const T& b) {
            T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
            a = -(b * std::log(clamped) + (static_cast<T>(1) - b) * std::log(static_cast<T>(1) - clamped));
        }, target);
    }
    else if(order == 1){
        input.broadcastInplace([eps](T& a, const T& b) {
            T clamped = std::max(eps, std::min(static_cast<T>(1) - eps, a));
            a = -b / clamped + (static_cast<T>(1) - b) / (static_cast<T>(1) - clamped);
        }, target);
    }
    else{
        throwNotSupport("yt::function::binaryCrossEntropyLoss_", "order != 0, 1");
    }
    return input;
}

// ========== huberLoss ==========

template <typename T, int dim>
yt::YTensor<T, dim> yt::function::huberLoss(const yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::huberLoss()");
    if(order == 0){
        return yt::kernel::broadcast([delta](const T& a, const T& b) {
            T diff = a - b;
            T abs_diff = std::abs(diff);
            if(abs_diff <= delta){
                return static_cast<T>(0.5) * diff * diff;
            }
            else{
                return delta * (abs_diff - static_cast<T>(0.5) * delta);
            }
        }, input, target);
    }
    else if(order == 1){
        // d/d(input): diff if |diff| <= delta, else delta * sign(diff)
        return yt::kernel::broadcast([delta](const T& a, const T& b) {
            T diff = a - b;
            T abs_diff = std::abs(diff);
            if(abs_diff <= delta){
                return diff;
            }
            else{
                return diff > static_cast<T>(0) ? delta : -delta;
            }
        }, input, target);
    }
    else{
        throwNotSupport("yt::function::huberLoss", "order != 0, 1");
    }
    return input;
}

template <typename T, int dim>
yt::YTensor<T, dim>& yt::function::huberLoss_(yt::YTensor<T, dim>& input, const yt::YTensor<T, dim>& target, T delta, int order) {
    static_assert(std::is_floating_point_v<T>, "T must be floating point type in yt::function::huberLoss_()");
    if(order == 0){
        input.broadcastInplace([delta](T& a, const T& b) {
            T diff = a - b;
            T abs_diff = std::abs(diff);
            if(abs_diff <= delta){
                a = static_cast<T>(0.5) * diff * diff;
            }
            else{
                a = delta * (abs_diff - static_cast<T>(0.5) * delta);
            }
        }, target);
    }
    else if(order == 1){
        input.broadcastInplace([delta](T& a, const T& b) {
            T diff = a - b;
            T abs_diff = std::abs(diff);
            if(abs_diff <= delta){
                a = diff;
            }
            else{
                a = diff > static_cast<T>(0) ? delta : -delta;
            }
        }, target);
    }
    else{
        throwNotSupport("yt::function::huberLoss_", "order != 0, 1");
    }
    return input;
}

void yt::function::throwNotSupport(const std::string& funcName, const std::string& caseDiscription) {
    throw std::invalid_argument("Function " + funcName + " is not supported for case: " + caseDiscription);
}
