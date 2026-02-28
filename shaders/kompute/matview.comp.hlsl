StructuredBuffer<uint> identityIndex : register(t0);
RWStructuredBuffer<uint> outputIndex : register(u1);
cbuffer MatviewShape : register(b0)
{
    uint totalSize;
};

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= totalSize) return;
    outputIndex[tid.x] = identityIndex[tid.x];
}
