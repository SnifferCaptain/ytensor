StructuredBuffer<float> inputA : register(t0);
StructuredBuffer<float> inputB : register(t1);
RWStructuredBuffer<uint> outputC : register(u2);

[numthreads(64, 1, 1)]
// `main` is kept as a default compile entrypoint for bulk HLSL->SPIR-V build steps.
void main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] < inputB[tid.x]; }
[numthreads(64, 1, 1)]
void lt_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] < inputB[tid.x]; }
[numthreads(64, 1, 1)]
void le_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] <= inputB[tid.x]; }
[numthreads(64, 1, 1)]
void gt_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] > inputB[tid.x]; }
[numthreads(64, 1, 1)]
void ge_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] >= inputB[tid.x]; }
[numthreads(64, 1, 1)]
void eq_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] == inputB[tid.x]; }
[numthreads(64, 1, 1)]
void ne_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] != inputB[tid.x]; }
