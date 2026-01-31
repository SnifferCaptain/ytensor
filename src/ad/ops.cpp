#include "../../include/ad/ops.hpp"

namespace yt {
namespace ad {
namespace ops {

// 注册所有基础算子
REGISTER_OPERATOR("Linear", LinearOp);
REGISTER_OPERATOR("RMSNorm", RMSNormOp);
REGISTER_OPERATOR("Add", AddOp);
REGISTER_OPERATOR("Multiply", MultiplyOp);
REGISTER_OPERATOR("GELU", GELUOp);
REGISTER_OPERATOR("Embedding", EmbeddingOp);

} // namespace ops
} // namespace ad
} // namespace yt
