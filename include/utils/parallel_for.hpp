#pragma once
/***************
 * @file parallel_for.hpp
 * @brief 并行for循环声明
 ***************/

#include "../ytensor_infos.hpp"
#include <limits>
#include <stdexcept>
#include <string>

namespace yt::utils {

/// @brief 将 size_t 工作量安全转换为当前 int-indexed kernel 的工作量。
/// @param size 待转换的元素数量
/// @param context 发生溢出时写入异常消息的调用上下文
/// @return 可由 int 表示的元素数量
/// @throws std::overflow_error size 超出 int 范围时抛出
inline int checkedIntSize(size_t size, const std::string& context) {
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(context + ": element count exceeds int indexing range");
    }
    return static_cast<int>(size);
}

/// @brief 并行for循环，由外部显式决定是否启用并行
/// @param from 起始索引（包含）
/// @param to 结束索引（不包含）
/// @param func 可调用对象，接受一个int参数，表示当前索引
/// @param enableParallel 是否启用并行。建议由外部根据任务粒度显式传入 true/false。
template <typename Func>
void parallelFor(int from, int to, Func&& func, bool enableParallel = true);

}  // namespace yt::utils

#include "../../src/utils/parallel_for.inl"
