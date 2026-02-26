#define VK_BINDING(x) [[vk::binding(x, 0)]]

VK_BINDING(0) RWStructuredBuffer<float> inputA : register(u0);
VK_BINDING(1) RWStructuredBuffer<float> inputB : register(u1);
VK_BINDING(2) RWStructuredBuffer<float> outputC : register(u2);

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    outputC[idx] = inputA[idx] + inputB[idx];
}
