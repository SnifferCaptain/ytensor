#pragma once
/***************
 * @file parallel_for.hpp
 * @brief 并行for循环声明
 ***************/

#include "../ytensor_infos.hpp"

namespace yt::kernel {

/// @brief 并行for循环，由外部显式决定是否启用并行
/// @param from 起始索引（包含）
/// @param to 结束索引（不包含）
/// @param func 可调用对象，接受一个int参数，表示当前索引
/// @param enableParallel 是否启用并行。建议由外部根据任务粒度显式传入 true/false。
template<typename Func>
void parallelFor(int from, int to, Func&& func, bool enableParallel = true);

} // namespace yt::kernel

#include "../../src/kernel/parallel_for.inl"
