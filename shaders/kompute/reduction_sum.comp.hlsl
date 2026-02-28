// High-performance parallel sum reduction with shared memory tree reduction.
// Inspired by ggml sum_rows.comp: each workgroup collaboratively reduces
// one output element using shared-memory parallel tree reduction.
//
// BLOCK_SIZE threads per workgroup. Each thread strides across the axis
// to accumulate a partial sum, then a tree reduction in shared memory
// produces the final result.
// Dispatch: (outputSize, 1, 1) workgroups.

#define BLOCK_SIZE 256

cbuffer ReduceShape : register(b0)
{
    uint axisLength;
    uint outputSize;
};

StructuredBuffer<float> inputData : register(t0);
RWStructuredBuffer<float> outputData : register(u1);

groupshared float sharedSum[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint outIdx = gid.x;
    if (outIdx >= outputSize) return;

    uint col = gtid.x;
    uint base = outIdx * axisLength;

    // Phase 1: each thread accumulates strided elements into a partial sum.
    float partialSum = 0.0f;
    for (uint i = col; i < axisLength; i += BLOCK_SIZE) {
        partialSum += inputData[base + i];
    }

    sharedSum[col] = partialSum;
    GroupMemoryBarrierWithGroupSync();

    // Phase 2: parallel tree reduction in shared memory (ggml pattern).
    [unroll] for (uint s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (col < s) {
            sharedSum[col] += sharedSum[col + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (col == 0) {
        outputData[outIdx] = sharedSum[0];
    }
}
