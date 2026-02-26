StructuredBuffer<float> input0 : register(t0);
StructuredBuffer<float> input1 : register(t1);
RWStructuredBuffer<float> output0 : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    output0[tid.x] = input0[tid.x] + input1[tid.x];
}
