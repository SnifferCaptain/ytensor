// High-performance element-wise comparison operations.
// 256 threads per workgroup, 4 elements per thread (ggml pattern: num_iter).
// Effective elements per workgroup: 256 × 4 = 1024.
// Dispatch: ((totalSize + 1023) / 1024, 1, 1).

StructuredBuffer<float> inputA : register(t0);
StructuredBuffer<float> inputB : register(t1);
RWStructuredBuffer<uint> outputC : register(u2);

cbuffer BinarySize : register(b0) { uint totalSize; };

#define NUM_THREADS 256
#define NUM_ITER 4

[numthreads(NUM_THREADS, 1, 1)]
// `main` is kept as a default compile entrypoint for bulk HLSL->SPIR-V build steps.
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] < inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void lt_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] < inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void le_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] <= inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void gt_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] > inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void ge_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] >= inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void eq_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] == inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void ne_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] != inputB[idx];
    }
}
