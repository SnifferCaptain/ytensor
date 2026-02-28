cbuffer ReduceShape : register(b0)
{
    uint axisLength;
    uint outputSize;
};

StructuredBuffer<float> inputData : register(t0);
RWStructuredBuffer<float> outputMax : register(u1);
RWStructuredBuffer<uint> outputArgmax : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint outIdx = tid.x;
    if (outIdx >= outputSize) return;
    uint base = outIdx * axisLength;
    float best = inputData[base];
    uint arg = 0;
    for (uint i = 1; i < axisLength; ++i) {
        float v = inputData[base + i];
        if (v > best) {
            best = v;
            arg = i;
        }
    }
    outputMax[outIdx] = best;
    outputArgmax[outIdx] = arg;
}
