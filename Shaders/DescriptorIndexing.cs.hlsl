// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

#if (NRI_SHADER_MODEL >= 66)

struct Constants
{
    float4 unused;
    float4 color;
};

struct RootConstants
{
    float4 color;
    uint2 viewportSize;
    uint2 viewportGridSize;
};

NRI_ROOT_CONSTANTS(RootConstants, g_RootConstants, 0, 1);
NRI_RESOURCE(SamplerState, g_RootSampler0, s, 0, 1);
NRI_RESOURCE(SamplerState, g_RootSampler1, s, 1, 1);

[numthreads(16, 16, 1)]
void main(int2 pixelPos : SV_DispatchThreadId)
{
    RWTexture2D<float4> dest = ResourceDescriptorHeap[0];
    ConstantBuffer<Constants> constants = ResourceDescriptorHeap[1];
    Texture2D<float4> src0 = ResourceDescriptorHeap[2];
    Texture2D<float4> src1 = ResourceDescriptorHeap[3];
    SamplerState sampler0 = SamplerDescriptorHeap[0];
    SamplerState sampler1 = SamplerDescriptorHeap[1];

    uint2 viewportPos = pixelPos / g_RootConstants.viewportSize;
    uint2 localPos = pixelPos % g_RootConstants.viewportSize;
    uint viewportIndex = viewportPos.y * g_RootConstants.viewportGridSize.x + viewportPos.x;
    float2 uv = (localPos + 0.5) / (g_RootConstants.viewportSize * 16.0) + 0.33;

    float4 result = constants.color;
    if (all(viewportPos < g_RootConstants.viewportGridSize))
    {
        switch (viewportIndex)
        {
            case 0: break;
            case 1: result = src0[localPos]; break;
            case 2: result = src1[localPos]; break;
            case 3: result = g_RootConstants.color; break;
            case 4: result = src0.SampleLevel(sampler0, uv, 0); break;
            case 5: result = src0.SampleLevel(sampler1, uv, 0); break;
            case 6: result = src1.SampleLevel(g_RootSampler0, uv, 0); break;
            case 7: result = src1.SampleLevel(g_RootSampler1, uv, 0); break;
        }
    }

    dest[pixelPos] = result;
}

#else

[numthreads(16, 16, 1)]
void main(int2 pixelPos : SV_DispatchThreadId)
{
}

#endif
