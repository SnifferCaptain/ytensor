StructuredBuffer<int> inputA : register(t0);
StructuredBuffer<int> inputB : register(t1);
RWStructuredBuffer<int> outputC : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void add_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] + inputB[tid.x]; }
[numthreads(64, 1, 1)]
void sub_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] - inputB[tid.x]; }
[numthreads(64, 1, 1)]
void mul_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] * inputB[tid.x]; }
[numthreads(64, 1, 1)]
void div_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] / max(inputB[tid.x], 1); }
[numthreads(64, 1, 1)]
void mod_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] % max(inputB[tid.x], 1); }
[numthreads(64, 1, 1)]
void and_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] & inputB[tid.x]; }
[numthreads(64, 1, 1)]
void or_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] | inputB[tid.x]; }
[numthreads(64, 1, 1)]
void xor_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] ^ inputB[tid.x]; }
[numthreads(64, 1, 1)]
void shl_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] << (inputB[tid.x] & 31); }
[numthreads(64, 1, 1)]
void shr_main(uint3 tid : SV_DispatchThreadID) { outputC[tid.x] = inputA[tid.x] >> (inputB[tid.x] & 31); }
