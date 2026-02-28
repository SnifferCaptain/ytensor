// Tiled matrix multiplication with shared memory.
// Inspired by ggml mul_mm.comp: tile-based blocking with bank-conflict-free
// shared memory, loop unrolling, and per-thread multi-element accumulation.
//
// Tile sizes: BM=BN=32, BK=16.
// Each thread computes a TM×TN = 4×4 sub-tile of output.
// Workgroup: [numthreads(8, 8, 1)] = 64 threads.
// Each workgroup covers BM×BN = 32×32 output elements.
// Dispatch: ((N+31)/32, (M+31)/32, 1).

#define BM 32
#define BN 32
#define BK 16
#define TM 4
#define TN 4
// Pad shared memory stride by 1 to avoid bank conflicts (ggml pattern).
#define SHMEM_PAD 1
#define SHMEM_STRIDE_A (BK + SHMEM_PAD)
#define SHMEM_STRIDE_B (BK + SHMEM_PAD)

cbuffer MatmulShape : register(b0)
{
    uint M;
    uint N;
    uint K;
};

StructuredBuffer<float> matrixA : register(t0);
StructuredBuffer<float> matrixB : register(t1);
RWStructuredBuffer<float> matrixC : register(u2);

groupshared float tileA[BM][SHMEM_STRIDE_A];
groupshared float tileB[BN][SHMEM_STRIDE_B];

[numthreads(8, 8, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID)
{
    const uint localRow = ltid.y; // 0..7
    const uint localCol = ltid.x; // 0..7
    const uint localIdx = localRow * 8 + localCol; // 0..63

    // Global output tile origin.
    const uint blockRow = gid.y * BM;
    const uint blockCol = gid.x * BN;

    // Per-thread accumulators (TM×TN = 4×4 = 16 values).
    float acc[TM][TN];
    [unroll] for (uint i = 0; i < TM; ++i)
        [unroll] for (uint j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // Loop over K dimension in steps of BK.
    for (uint bk = 0; bk < K; bk += BK) {

        // --- Load tileA (BM×BK = 32×16 = 512 floats, 64 threads → 8 loads each) ---
        [unroll] for (uint t = 0; t < 8; ++t) {
            uint idx = localIdx + t * 64;
            uint tr = idx / BK;
            uint tc = idx % BK;
            uint globalR = blockRow + tr;
            uint globalC = bk + tc;
            tileA[tr][tc] = (globalR < M && globalC < K) ? matrixA[globalR * K + globalC] : 0.0f;
        }

        // --- Load tileB (BN×BK = 32×16 = 512 floats, 64 threads → 8 loads each) ---
        // tileB stored as [col][k] for coalesced access during compute.
        [unroll] for (uint t = 0; t < 8; ++t) {
            uint idx = localIdx + t * 64;
            uint tr = idx / BK; // column index
            uint tc = idx % BK; // k index
            uint globalR = bk + tc;
            uint globalC = blockCol + tr;
            tileB[tr][tc] = (globalR < K && globalC < N) ? matrixB[globalR * N + globalC] : 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        // --- Compute: each thread accumulates TM×TN sub-tile ---
        [unroll] for (uint kk = 0; kk < BK; ++kk) {
            // Cache values from shared memory into registers.
            float regA[TM];
            float regB[TN];

            [unroll] for (uint i = 0; i < TM; ++i)
                regA[i] = tileA[localRow * TM + i][kk];

            [unroll] for (uint j = 0; j < TN; ++j)
                regB[j] = tileB[localCol * TN + j][kk];

            [unroll] for (uint i = 0; i < TM; ++i)
                [unroll] for (uint j = 0; j < TN; ++j)
                    acc[i][j] = mad(regA[i], regB[j], acc[i][j]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // --- Store results ---
    [unroll] for (uint i = 0; i < TM; ++i) {
        uint globalR = blockRow + localRow * TM + i;
        if (globalR >= M) continue;
        [unroll] for (uint j = 0; j < TN; ++j) {
            uint globalC = blockCol + localCol * TN + j;
            if (globalC < N) {
                matrixC[globalR * N + globalC] = acc[i][j];
            }
        }
    }
}
