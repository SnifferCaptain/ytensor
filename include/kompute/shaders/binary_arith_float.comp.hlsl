StructuredBuffer<float> inputA : register(t0);
StructuredBuffer<float> inputB : register(t1);
RWStructuredBuffer<float> outputC : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fadd_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fsub_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] - inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fmul_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] * inputB[tid.x]; }
[numthreads(64, 1, 1)]
void fdiv_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = (inputB[tid.x] == 0.0f) ? 0.0f : (inputA[tid.x] / inputB[tid.x]); }
