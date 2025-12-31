
template <typename T,int dim>
class YTensor;

template <typename T, int dim0, int dim1>
YTensor<T, yt::concepts::CONSTEXPR_MAX({dim0, dim1, 2})> yt::function::matmul(const YTensor<T, dim0>& a, const YTensor<T, dim1>& b) {
    return a.matmul(b);
}

template <typename T, int dim>
YTensor<T, dim> yt::function::relu(const YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    YTensor<T, dim> op;
    if(order == 0){
        op = x.binaryOpTransform(0, [](const T& a, const T&) {
            return std::max(a, static_cast<T>(0));
        });
    }
    else if(order == 1){
        op = x.binaryOpTransform(0, [](const T& a, const T&) {
            return static_cast<T>(a > 0);
        });
    }
    else if(order > 1){
        // op = YTensor<T, dim>::zeros(x.shape());
        throwNotSupport("yt::function::relu", "order > 1");
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        op = x.binaryOpTransform(0, [&pow, &fact](const T& a, const T&) {
            if(a > 0){
                return std::pow(a, pow) / static_cast<T>(fact);
            }
            else{
                return static_cast<T>(0);
            }
        });
    }
}

template <typename T, int dim>
YTensor<T, dim>& yt::function::relu_(YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::relu()");
    if(order == 0){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = std::max(a, static_cast<T>(0));
        });
    }
    else if(order == 1){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = static_cast<T>(a > 0);
        });
    }
    else if(order > 1){
        x.binaryOpTransformInplace(0, [](T& a, const T&) {
            a = static_cast<T>(0); // 这里可以根据需要修改
        });
    }
    else{
        int pow = -order + 1;
        int fact = 1;
        for (int i = 2; i < -order + 2; i++){
            fact *= i;
        }
        x.binaryOpTransformInplace(0, [&pow, &fact](T& a, const T&) {
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
YTensor<T, dim> yt::function::exp(const YTensor<T, dim>& x, int) {
    return x.unaryOpTransform(0, [](const T& a, const T&){
        return std::exp(a);
    });
}

template <typename T, int dim>
YTensor<T, dim> yt::function::sigmoid(const YTensor<T, dim>& x, int order) {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic type in YTensorFunction::sigmoid()");
    if(order == 0){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-a));
        });
    }
    else if(order == 1){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            T sig = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-a));
            return sig * (static_cast<T>(1) - sig);
        });
    }
    else if(order == -1){
        return x.binaryOpTransform(static_cast<T>(0), [](const T& a, const T&) {
            return std::log(static_cast<T>(1) + std::exp(a));
        });
    }
    else{
        throwNotSupport("yt::function::sigmoid", "order != 0, 1, -1");
    }
}

template<typename T, int dim>
YTensor<T, dim> yt::function::softmax(const YTensor<T, dim>& x, int axis) {
    // 标准化 axis 索引
    axis = (axis % dim + dim) % dim;
    
    auto shape = x.shape();
    YTensor<T, dim> output(shape);
    
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
YTensor<T, dim>& yt::function::softmax_(YTensor<T, dim>& x, int axis) {
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
YTensor<T, dim> yt::function::scaledDotProductAttention(
    YTensor<T, dim>& query,// [..., n0, c0]
    YTensor<T, dim>& key,// [..., n1, c0]
    YTensor<T, dim>& value,// [..., n1, c1]
    T scale,
    YTensor<T, 2>* mask,
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

        auto score = yt::function::matmul(query, key.transpose());// [..., n0, n1]
        score.binaryOpTransformInplace(scale, [](T& a, const T& b) {
            a *= b; // scale
        });
        if(mask != nullptr){
            if(mask->shape(0) != score.shape(-2) || mask->shape(1) != score.shape(-1)){
                throw std::invalid_argument("Mask shape must match the last two dimensions of the score tensor.");
            }
            score += *mask;
        }
        yt::function::softmax_(score, -1);// [..., n0, n1] inplace
        auto op = yt::function::matmul(score, value);// [..., n0, c1]
        return op;
    }
    else{
        throwNotSupport("yt::function::scaledDotProductAttention", "other backends");
        return YTensor<T, dim>();
    }
}


void yt::function::throwNotSupport(const std::string& funcName, const std::string& caseDiscription) {
    throw std::invalid_argument("Function " + funcName + " is not supported for case: " + caseDiscription);
}
