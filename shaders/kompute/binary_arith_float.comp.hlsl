StructuredBuffer<float> inputA : register(t0);
StructuredBuffer<float> inputB : register(t1);
RWStructuredBuffer<float> outputC : register(u2);

cbuffer BinarySize : register(b0) { uint totalSize; };

// `main` is kept for compatibility with default HLSL->SPIR-V compilation (-e main).
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { if (tid.x >= totalSize) return; outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fadd_main(uint3 tid : SV_DispatchThreadID) { if (tid.x >= totalSize) return; outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fsub_main(uint3 tid : SV_DispatchThreadID) { if (tid.x >= totalSize) return; outputC[tid.x] = inputA[tid.x] - inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fmul_main(uint3 tid : SV_DispatchThreadID) { if (tid.x >= totalSize) return; outputC[tid.x] = inputA[tid.x] * inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fdiv_main(uint3 tid : SV_DispatchThreadID) { if (tid.x >= totalSize) return; outputC[tid.x] = (inputB[tid.x] == 0.0f) ? 0.0f : (inputA[tid.x] / inputB[tid.x]); }
