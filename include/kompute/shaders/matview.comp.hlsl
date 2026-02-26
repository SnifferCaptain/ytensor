StructuredBuffer<uint> identityIndex : register(t0);
RWStructuredBuffer<uint> outputIndex : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    outputIndex[tid.x] = identityIndex[tid.x];
}
