// High-performance element-wise integer operations.
// 256 threads per workgroup, 4 elements per thread (ggml pattern: num_iter).
// Effective elements per workgroup: 256 × 4 = 1024.
// Dispatch: ((totalSize + 1023) / 1024, 1, 1).

StructuredBuffer<int> inputA : register(t0);
StructuredBuffer<int> inputB : register(t1);
RWStructuredBuffer<int> outputC : register(u2);

cbuffer BinarySize : register(b0) { uint totalSize; };

#define NUM_THREADS 256
#define NUM_ITER 4

[numthreads(NUM_THREADS, 1, 1)]
// `main` is kept as a default compile entrypoint for bulk HLSL->SPIR-V build steps.
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] + inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void add_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] + inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void sub_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] - inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void mul_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] * inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
// Deterministic guard for zero divisor in shader path.
void div_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = (inputB[idx] == 0) ? 0 : (inputA[idx] / inputB[idx]);
    }
}
[numthreads(NUM_THREADS, 1, 1)]
// Deterministic guard for zero divisor in shader path.
void mod_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = (inputB[idx] == 0) ? 0 : (inputA[idx] % inputB[idx]);
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void and_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] & inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void or_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] | inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
void xor_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] ^ inputB[idx];
    }
}
[numthreads(NUM_THREADS, 1, 1)]
// Shift amount masking follows 32-bit integer lane semantics.
void shl_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] << (inputB[idx] & 31);
    }
}
[numthreads(NUM_THREADS, 1, 1)]
// Shift amount masking follows 32-bit integer lane semantics.
void shr_main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint base = gid.x * (NUM_THREADS * NUM_ITER) + gtid.x;
    [unroll] for (uint i = 0; i < NUM_ITER; ++i) {
        uint idx = base + i * NUM_THREADS;
        if (idx < totalSize) outputC[idx] = inputA[idx] >> (inputB[idx] & 31);
    }
}
