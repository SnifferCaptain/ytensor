#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <type_traits>
#include <functional>
#include <cmath>
#include "../../ytensor.hpp"

namespace nn {

// 基础模块类
template<typename InputType, typename OutputType>
class Module {
public:
    using Input = InputType;
    using Output = OutputType;
    
    virtual ~Module() = default;
    
    virtual OutputType forward(const InputType& input) = 0;
    virtual InputType backward(const OutputType& grad_output) = 0;
    virtual std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() = 0;
    virtual std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() = 0;
    virtual void train(bool mode = true) { is_training_ = mode; }
    virtual bool is_training() const { return is_training_; }
    virtual std::string name() const { return "Module"; }
    
protected:
    bool is_training_ = true;
};

// 改进的初始化方法 - 支持任意维度的张量
class Initializer {
public:
    // He正态初始化 - 支持任意维度的张量
    template<typename T, int dim>
    static void he_normal(YTensor<T, dim>& tensor) {
        auto shape = tensor.shape();
        
        // 计算fan_in和fan_out
        // 对于权重矩阵，通常第一个维度是fan_out，其余维度的乘积是fan_in
        int fan_out = shape[0];
        int fan_in = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            fan_in *= shape[i];
        }
        
        float stddev = std::sqrt(2.0f / fan_in);
        
        // 创建随机张量并应用标准差
        auto random_tensor = YTensor<float, dim>::randn(shape);
        tensor = random_tensor * stddev;
    }
    
    // Xavier/Glorot正态初始化
    template<typename T, int dim>
    static void xavier_normal(YTensor<T, dim>& tensor) {
        auto shape = tensor.shape();
        
        int fan_out = shape[0];
        int fan_in = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            fan_in *= shape[i];
        }
        
        float stddev = std::sqrt(2.0f / (fan_in + fan_out));
        
        auto random_tensor = YTensor<float, dim>::randn(shape);
        tensor = random_tensor * stddev;
    }
    
    // 零初始化 - 支持任意维度
    template<typename T, int dim>
    static void zeros(YTensor<T, dim>& tensor) {
        auto shape = tensor.shape();
        tensor = YTensor<float, dim>::zeros(shape);
    }
    
    // 常量初始化 - 支持任意维度
    template<typename T, int dim>
    static void constant(YTensor<T, dim>& tensor, float value) {
        auto shape = tensor.shape();
        tensor = YTensor<float, dim>::ones(shape) * value;
    }
};

// Flatten层 - 支持任意维度输入到2D输出
template<typename T, int dim>
class Flatten : public Module<YTensor<T, dim>, YTensor<T, 2>> {
public:
    using InputType = YTensor<T, dim>;
    using OutputType = YTensor<T, 2>;
    
private:
    std::vector<int> _input_shape;
    
public:
    OutputType forward(const InputType& input) override {
        _input_shape = input.shape();
        int batch_size = _input_shape[0];
        int flattened_size = 1;
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            flattened_size *= _input_shape[i];
        }
        return input.view(batch_size, flattened_size);
    }
    
    InputType backward(const OutputType& grad_output) override {
        return grad_output.view(_input_shape);
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() override {
        return {};
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() override {
        return {};
    }
    
    std::string name() const override { return "Flatten"; }
};

// 改进的全连接层 - 支持任意维度输入
template<typename T, int dim>
class Linear : public Module<YTensor<T, dim>, YTensor<T, dim>> {
public:
    using InputType = YTensor<T, dim>;
    using OutputType = YTensor<T, dim>;

private:
    int _in_features, _out_features;
    YTensor<T, 2> _weight, _weight_grad;
    YTensor<T, 1> _bias, _bias_grad;
    InputType input_cache_;
    
public:
    Linear(int in_features, int out_features)
        : _in_features(in_features), _out_features(out_features),
          _weight(out_features, in_features),
          _bias(out_features),
          _weight_grad(out_features, in_features),
          _bias_grad(out_features) {

        Initializer::he_normal(_weight);
        Initializer::zeros(_bias);
        Initializer::zeros(_weight_grad);
        Initializer::zeros(_bias_grad);
    }
    
    OutputType forward(const InputType& input) override {
        input_cache_ = input.clone();
        
        // 矩阵乘法 + 偏置
        OutputType output = yt::function::matmul(input, _weight.transpose());
        output += _bias;

        return output;
    }
    
    InputType backward(const OutputType& grad_output) override {
        auto flat_input = input_cache_;
        
        // 计算梯度
        _weight_grad = yt::function::matmul(grad_output.transpose(), flat_input);
        // _bias_grad = grad_output.sum(0);// *********** 错误 ***********

        auto flat_grad_input = yt::function::matmul(grad_output, _weight);

        // 恢复原始形状
        return flat_grad_input;
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() override {
        return {std::ref(_weight), std::ref(_bias.view(1, _bias.shape(0)))};
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() override {
        return {std::ref(_weight_grad), std::ref(_bias_grad.view(1, _bias_grad.shape(0)))};
    }
    
    std::string name() const override { return "Linear"; }
};

// 改进的ReLU激活层 - 支持任意维度
template<typename T, int dim>
class ReLU : public Module<YTensor<T, dim>, YTensor<T, dim>> {
public:
    using InputType = YTensor<T, dim>;
    using OutputType = YTensor<T, dim>;

private:
    InputType _input_cache;
    
public:
    OutputType forward(const InputType& input) override {
        _input_cache = yt::function::relu(input, 1);
        OutputType output = input.clone();
        yt::function::relu_(output);
        return output;
    }
    
    InputType backward(const OutputType& grad_output) override {
        return grad_output * _input_cache;
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() override {
        return {};
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() override {
        return {};
    }
    
    std::string name() const override { return "ReLU"; }
};

// 改进的Softmax层 - 支持任意维度
template<typename T, int dim>
class Softmax : public Module<YTensor<T, dim>, YTensor<T, dim>> {
public:
    using InputType = YTensor<T, dim>;
    using OutputType = YTensor<T, dim>;

private:
    OutputType _cache;
    int _axis;

public:
    Softmax(int axis = 1) : _axis(axis) {}
    
    OutputType forward(const InputType& input) override {
        _cache = yt::function::softmax(input, _axis);
        return _cache;
    }
    
    InputType backward(const OutputType& grad_output) override {
        auto j = grad_output * (grad_output * _cache).sum(_axis);
        return j;
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() override {
        return {};
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() override {
        return {};
    }
    
    std::string name() const override { return "Softmax"; }
};

// 模板化的Sequential容器 - 支持任意维度的输入输出
template<typename T, int dim>
class Sequential : public Module<YTensor<T, dim>, YTensor<T, dim>> {
public:
    using InputType = YTensor<T, dim>;
    using OutputType = YTensor<T, dim>;
    
private:
    std::vector<std::shared_ptr<Module<InputType, OutputType>>> modules_;
    
public:
    Sequential() = default;
    
    // 添加模块
    void add(std::shared_ptr<Module<InputType, OutputType>> module) {
        modules_.push_back(module);
    }
    
    OutputType forward(const InputType& input) override {
        if (modules_.empty()) {
            throw std::runtime_error("Sequential container is empty");
        }
        
        auto current_output = input;
        for (auto& module : modules_) {
            current_output = module->forward(current_output);
        }
        return current_output;
    }
    
    InputType backward(const OutputType& grad_output) override {
        if (modules_.empty()) {
            throw std::runtime_error("Sequential container is empty");
        }
        
        auto current_grad = grad_output;
        // 反向遍历模块进行反向传播
        for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
        return current_grad;
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters() override {
        std::vector<std::reference_wrapper<YTensor<float, 2>>> params;
        for (auto& module : modules_) {
            auto module_params = module->parameters();
            params.insert(params.end(), module_params.begin(), module_params.end());
        }
        return params;
    }
    
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients() override {
        std::vector<std::reference_wrapper<YTensor<float, 2>>> grads;
        for (auto& module : modules_) {
            auto module_grads = module->gradients();
            grads.insert(grads.end(), module_grads.begin(), module_grads.end());
        }
        return grads;
    }
    
    void train(bool mode = true) override {
        for (auto& module : modules_) {
            module->train(mode);
        }
        this->is_training_ = mode;
    }
    
    std::string name() const override { return "Sequential"; }
    
    size_t size() const { return modules_.size(); }
    bool empty() const { return modules_.empty(); }
};

// 交叉熵损失函数
class CrossEntropyLoss {
private:
    YTensor<float, 2> input_cache_;
    YTensor<int64_t, 1> target_cache_;
    YTensor<float, 1> class_weights_;
    int batch_size_, num_classes_;
    bool use_weights_;
    
public:
    CrossEntropyLoss(const YTensor<float, 1>& weights = YTensor<float, 1>()) 
        : class_weights_(weights), use_weights_(weights.size() > 0) {}
    
    float forward(const YTensor<float, 2>& input, const YTensor<int64_t, 1>& target) {
        batch_size_ = input.shape(0);
        num_classes_ = input.shape(1);
        input_cache_ = input.clone();
        target_cache_ = target.clone();
        
        YTensor<float, 2> softmax_output = yt::function::softmax(input, 1);
        
        float total_loss = 0.0f;
        for (int i = 0; i < batch_size_; ++i) {
            int true_class = target[i];
            float prob = softmax_output[i][true_class];
            float loss_val = -std::log(prob + 1e-8f);
            
            if (use_weights_) {
                loss_val *= class_weights_[true_class];
            }
            
            total_loss += loss_val;
        }
        
        return total_loss / batch_size_;
    }
    
    YTensor<float, 2> backward() {
        YTensor<float, 2> grad_input({batch_size_, num_classes_});
        YTensor<float, 2> softmax_output = yt::function::softmax(input_cache_, 1);
        
        for (int i = 0; i < batch_size_; ++i) {
            int true_class = target_cache_[i];
            for (int j = 0; j < num_classes_; ++j) {
                grad_input[i][j] = softmax_output[i][j] - (j == true_class ? 1.0f : 0.0f);
                
                if (use_weights_) {
                    grad_input[i][j] *= class_weights_[true_class];
                }
            }
        }
        
        float batch_size_float = static_cast<float>(batch_size_);
        grad_input.binaryOpTransformInplace(batch_size_float, [](float& item, float divisor) {
            item /= divisor;
        });
        
        return grad_input;
    }
};

// SGD优化器
class SGD {
private:
    std::vector<std::reference_wrapper<YTensor<float, 2>>> parameters_;
    std::vector<std::reference_wrapper<YTensor<float, 2>>> gradients_;
    float learning_rate_;
    float momentum_;
    std::vector<YTensor<float, 2>> velocity_;
    
public:
    SGD(std::vector<std::reference_wrapper<YTensor<float, 2>>> params,
        std::vector<std::reference_wrapper<YTensor<float, 2>>> grads,
        float lr, float momentum = 0.0f)
        : parameters_(std::move(params)), gradients_(std::move(grads)),
          learning_rate_(lr), momentum_(momentum) {
        
        for (const auto& param : parameters_) {
            velocity_.emplace_back(param.get().shape());
            Initializer::zeros(velocity_.back());
        }
    }
    
    void step() {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i].get();
            auto& grad = gradients_[i].get();
            auto& vel = velocity_[i];
            
            if (momentum_ > 0) {
                vel.binaryOpTransformInplace(momentum_, [](float& v, float mom) {
                    v *= mom;
                });
                
                YTensor<float, 2> grad_scaled = grad;
                grad_scaled.binaryOpTransformInplace(learning_rate_, [](float& g, float lr) {
                    g *= lr;
                });
                
                vel = vel - grad_scaled;
                param = param + vel;
            } else {
                YTensor<float, 2> grad_scaled = grad;
                grad_scaled.binaryOpTransformInplace(learning_rate_, [](float& g, float lr) {
                    g *= lr;
                });
                
                param = param - grad_scaled;
            }
        }
    }
    
    void zero_grad() {
        for (auto& grad_ref : gradients_) {
            auto& grad = grad_ref.get();
            Initializer::zeros(grad);
        }
    }
};

} // namespace nn

// 改进的CIFAR10模型 - 支持任意维度的模块
class CIFAR10Model {
private:
    // 使用模板化的模块
    std::shared_ptr<nn::Flatten<float, 4>> flatten_;
    std::shared_ptr<nn::Sequential<float, 2>> sequential_;
    
    nn::CrossEntropyLoss criterion_;
    std::unique_ptr<nn::SGD> optimizer_;
    
public:
    CIFAR10Model(float learning_rate = 0.01f, float momentum = 0.9f,
                 const YTensor<float, 1>& class_weights = YTensor<float, 1>()) 
        : criterion_(class_weights) {
        
        // 创建Flatten层
        flatten_ = std::make_shared<nn::Flatten<float, 4>>();
        
        // 创建Sequential容器并添加层
        sequential_ = std::make_shared<nn::Sequential<float, 2>>();
        sequential_->add(std::make_shared<nn::Linear<float, 2>>(3072, 512));
        sequential_->add(std::make_shared<nn::ReLU<float, 2>>());
        sequential_->add(std::make_shared<nn::Linear<float, 2>>(512, 256));
        sequential_->add(std::make_shared<nn::ReLU<float, 2>>());
        sequential_->add(std::make_shared<nn::Linear<float, 2>>(256, 10));

        // 收集所有参数
        std::vector<std::reference_wrapper<YTensor<float, 2>>> params;
        std::vector<std::reference_wrapper<YTensor<float, 2>>> grads;
        
        // 添加Sequential的参数
        auto seq_params = sequential_->parameters();
        auto seq_grads = sequential_->gradients();
        params.insert(params.end(), seq_params.begin(), seq_params.end());
        grads.insert(grads.end(), seq_grads.begin(), seq_grads.end());
        
        // 创建优化器
        optimizer_ = std::make_unique<nn::SGD>(params, grads, learning_rate, momentum);
    }
    
    YTensor<float, 2> forward(const YTensor<float, 4>& input) {
        // 先展平，然后通过Sequential
        auto flattened = flatten_->forward(input);
        return sequential_->forward(flattened);
    }
    
    float train_step(const YTensor<float, 4>& input, const YTensor<int64_t, 1>& target) {
        auto output = forward(input);
        float loss = criterion_.forward(output, target);
        auto grad_output = criterion_.backward();
        
        // 反向传播：先通过Sequential，然后通过Flatten
        auto seq_grad = sequential_->backward(grad_output);
        flatten_->backward(seq_grad);
        
        optimizer_->step();
        optimizer_->zero_grad();
        return loss;
    }
    
    YTensor<int64_t, 1> predict(const YTensor<float, 4>& input) {
        auto output = forward(input);
        YTensor<int64_t, 1> predictions({output.shape(0)});
        
        for (int i = 0; i < output.shape(0); ++i) {
            int max_idx = 0;
            float max_val = output[i][0];
            for (int j = 1; j < output.shape(1); ++j) {
                if (output[i][j] > max_val) {
                    max_val = output[i][j];
                    max_idx = j;
                }
            }
            predictions[i] = max_idx;
        }
        
        return predictions;
    }
    
    float evaluate(const YTensor<float, 4>& input, const YTensor<int64_t, 1>& target) {
        auto predictions = predict(input);
        int correct = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == target[i]) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / predictions.size();
    }
    
    std::string info() const {
        return "CIFAR10Model with Flatten + " + std::to_string(sequential_->size()) + " layers";
    }
};

// 示例：使用不同维度的模型
class MultiInputModel {
private:
    // 处理3D输入的模型（例如时序数据）
    std::shared_ptr<nn::Flatten<float, 3>> flatten_3d_;
    std::shared_ptr<nn::Sequential<float, 2>> sequential_3d_;
    
    // 处理4D输入的模型（例如图像数据）
    std::shared_ptr<nn::Flatten<float, 4>> flatten_4d_;
    std::shared_ptr<nn::Sequential<float, 2>> sequential_4d_;
    
    nn::CrossEntropyLoss criterion_;
    
public:
    MultiInputModel() {
        // 3D模型（时序数据）
        flatten_3d_ = std::make_shared<nn::Flatten<float, 3>>();
        sequential_3d_ = std::make_shared<nn::Sequential<float, 2>>();
        sequential_3d_->add(std::make_shared<nn::Linear<float, 2>>(100, 64));
        sequential_3d_->add(std::make_shared<nn::ReLU<float, 2>>());
        sequential_3d_->add(std::make_shared<nn::Linear<float, 2>>(64, 10));

        // 4D模型（图像数据）
        flatten_4d_ = std::make_shared<nn::Flatten<float, 4>>();
        sequential_4d_ = std::make_shared<nn::Sequential<float, 2>>();
        sequential_4d_->add(std::make_shared<nn::Linear<float, 2>>(3072, 512));
        sequential_4d_->add(std::make_shared<nn::ReLU<float, 2>>());
        sequential_4d_->add(std::make_shared<nn::Linear<float, 2>>(512, 10));
    }
    
    YTensor<float, 2> forward_3d(const YTensor<float, 3>& input) {
        auto flattened = flatten_3d_->forward(input);
        return sequential_3d_->forward(flattened);
    }
    
    YTensor<float, 2> forward_4d(const YTensor<float, 4>& input) {
        auto flattened = flatten_4d_->forward(input);
        return sequential_4d_->forward(flattened);
    }
};