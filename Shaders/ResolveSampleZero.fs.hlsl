// © 2026 NVIDIA Corporation

#include "NRI.hlsl"

struct RootConstants {
    float depth;
    uint value;
};

NRI_ROOT_CONSTANTS(RootConstants, g_RootConstants, 0, 0);

uint4 main() : SV_Target {
    return uint4(g_RootConstants.value, g_RootConstants.value + 1, g_RootConstants.value + 2, g_RootConstants.value + 3);
}
