cbuffer MatmulShape : register(b0)
{
    uint M;
    uint N;
    uint K;
};

StructuredBuffer<float> matrixA : register(t0);
StructuredBuffer<float> matrixB : register(t1);
RWStructuredBuffer<float> matrixC : register(u2);

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint row = tid.y;
    uint col = tid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += matrixA[row * K + k] * matrixB[k * N + col];
    }
    matrixC[row * N + col] = acc;
}
