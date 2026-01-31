#pragma once
/***************
* @file: ad.hpp
* @brief: 自动微分模块总入口
* @description: 包含计算图的所有头文件和实现
***************/

// 头文件
#include "ad/graph_node.hpp"
#include "ad/graph_edge.hpp"
#include "ad/graph_runtime.hpp"
#include "ad/graph_serialization.hpp"

// 实现文件
#include "../src/ad/graph_node.inl"
#include "../src/ad/graph_edge.inl"
#include "../src/ad/graph_runtime.inl"
#include "../src/ad/graph_serialization.inl"
