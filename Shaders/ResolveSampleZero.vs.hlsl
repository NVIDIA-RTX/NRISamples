// © 2026 NVIDIA Corporation

#include "NRI.hlsl"

struct RootConstants {
    float depth;
    uint value;
};

NRI_ROOT_CONSTANTS(RootConstants, g_RootConstants, 0, 0);

float4 main(uint vertexId : SV_VertexID) : SV_Position {
    const float2 positions[] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};

    return float4(positions[vertexId], g_RootConstants.depth, 1.0);
}
