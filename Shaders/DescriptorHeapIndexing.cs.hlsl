// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

#if (NRI_SHADER_MODEL >= 66)

struct Constants
{
    float4 unused;
    float4 color;
};

[numthreads(16, 16, 1)]
void main(int2 pixelPos : SV_DispatchThreadId)
{
    // All access is uniform because of "pow-of-2" viewport dims
    RWTexture2D<float4> dest = ResourceDescriptorHeap[0];
    ConstantBuffer<Constants> constants = ResourceDescriptorHeap[1];
    Texture2D<float4> src0 = ResourceDescriptorHeap[2];
    Texture2D<float4> src1 = ResourceDescriptorHeap[3];

    SamplerState sampler0 = SamplerDescriptorHeap[0];
    SamplerState sampler1 = SamplerDescriptorHeap[1];

    float4 result = constants.color;
    if (pixelPos.y < 512)
    {
        if (pixelPos.x < 512)
            result = src0[pixelPos];
        else if (pixelPos.x < 1024)
            result = src1[pixelPos - int2(512, 0)];
    }
    else if(pixelPos.y < 1024)
    {
        if (pixelPos.x < 512)
            result = src0.SampleLevel(sampler0, (pixelPos - int2(0, 512)) / 4096.0 + 0.33, 0);
        else if (pixelPos.x < 1024)
            result = src0.SampleLevel(sampler1, (pixelPos - int2(512, 512)) / 4096.0 + 0.33, 0);
    }

    dest[pixelPos] = result;
}

#else

[numthreads(16, 16, 1)]
void main(int2 pixelPos : SV_DispatchThreadId)
{
}

#endif
