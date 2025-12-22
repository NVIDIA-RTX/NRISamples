// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

// Root
NRI_RESOURCE( SamplerState, g_Sampler, s, 0, 1 );
NRI_RESOURCE( cbuffer, CommonConstants, b, 1, 1 )
{
    float3 L;
};

// Descriptors
NRI_RESOURCE( Texture2D, g_NormalTexture, t, 0, 0 );

struct outputVS
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

float4 main( in outputVS input ) : SV_Target1
{
    float4 output;
    output.xyz = g_NormalTexture.Sample( g_Sampler, input.texCoord ).xyz;
    output.w = 1.0;

    return output;
}
