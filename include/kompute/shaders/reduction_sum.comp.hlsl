cbuffer ReduceShape : register(b0)
{
    uint axisLength;
    uint outputSize;
};

StructuredBuffer<float> inputData : register(t0);
RWStructuredBuffer<float> outputData : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint outIdx = tid.x;
    if (outIdx >= outputSize) return;
    float sum = 0.0f;
    uint base = outIdx * axisLength;
    for (uint i = 0; i < axisLength; ++i) {
        sum += inputData[base + i];
    }
    outputData[outIdx] = sum;
}
