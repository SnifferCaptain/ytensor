// High-performance matrix view / reshape index copy.
// 256 threads per workgroup, 4 elements per thread.
// Dispatch: ((totalSize + 1023) / 1024, 1, 1).

StructuredBuffer<uint> identityIndex : register(t0);
RWStructuredBuffer<uint> outputIndex : register(u1);
cbuffer MatviewShape : register(b0)
{
    uint totalSize;
};

#define NUM_THREADS 256
#define NUM_ITER 4

[numthreads(NUM_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputIndex[idx] = identityIndex[idx];
    }
}
