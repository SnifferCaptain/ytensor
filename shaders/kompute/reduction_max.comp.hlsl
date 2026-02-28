// High-performance parallel max reduction with argmax using shared memory
// tree reduction. Inspired by ggml argmax.comp: each workgroup collaboratively
// finds the maximum value and its index for one output element.
//
// BLOCK_SIZE threads per workgroup. Each thread strides across the axis,
// then a tree reduction in shared memory finds the global max and its index.
// Dispatch: (outputSize, 1, 1) workgroups.

#define BLOCK_SIZE 256
#define FLT_MIN_VAL (-3.402823466e+38f)

cbuffer ReduceShape : register(b0)
{
    uint axisLength;
    uint outputSize;
};

StructuredBuffer<float> inputData : register(t0);
RWStructuredBuffer<float> outputMax : register(u1);
RWStructuredBuffer<uint> outputArgmax : register(u2);

groupshared float sharedMax[BLOCK_SIZE];
groupshared uint  sharedArg[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint outIdx = gid.x;
    if (outIdx >= outputSize) return;

    uint col = gtid.x;
    uint base = outIdx * axisLength;

    // Phase 1: each thread finds local max across strided elements.
    float localMax = FLT_MIN_VAL;
    uint  localArg = 0;

    for (uint i = col; i < axisLength; i += BLOCK_SIZE) {
        float val = inputData[base + i];
        if (val > localMax) {
            localMax = val;
            localArg = i;
        }
    }

    sharedMax[col] = localMax;
    sharedArg[col] = localArg;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: parallel tree reduction in shared memory (ggml pattern).
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (col < s && col + s < axisLength) {
            if (sharedMax[col] < sharedMax[col + s]) {
                sharedMax[col] = sharedMax[col + s];
                sharedArg[col] = sharedArg[col + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (col == 0) {
        outputMax[outIdx] = sharedMax[0];
        outputArgmax[outIdx] = sharedArg[0];
    }
}
