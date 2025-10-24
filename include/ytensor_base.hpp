#pragma once
/***************
* @file: ytensor_base.hpp
* @brief: YTensorBase基类的功能实现，YTensorBase类不包含模板参数，提供运行时接口。
***************/
#include <memory>

#include "ytensor_concepts.hpp"
#include "ytensor_infos.hpp"

class YTensorBase { 
public:
    YTensorBase(){
        // 未实现
        throw std::runtime_error("[YTensorBase] class not implemented.");
    }
    virtual ~YTensorBase() = default;
protected:
    std::shared_ptr<char[]> _data;          // 存储数据
    int _offset = 0;                        // 数据偏移 
    std::vector<int> _shape;                // 形状
    std::vector<int> _stride;               // 步长
    size_t _element_size = 0;               // 元素大小
    std::string _dtype;                     // 用于序列化/反序列化友好名称
};

#include "../src/ytensor_base.inl"