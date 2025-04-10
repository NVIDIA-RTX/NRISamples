// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

NRI_RESOURCE( cbuffer, CommonConstants, b, 0, 0 )
{
    float3 color;
    float scale;
};

struct PushConstants
{
    float transparency;
};

NRI_ROOT_CONSTANTS( PushConstants, g_PushConstants, 1, 0 );

NRI_RESOURCE( Texture2D, g_DiffuseTexture, t, 0, 1 );
NRI_RESOURCE( SamplerState, g_Sampler, s, 0, 1 );

struct outputVS
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

float4 main( in outputVS input ) : SV_Target
{
    float4 output;
    output.xyz = g_DiffuseTexture.Sample( g_Sampler, input.texCoord ).xyz * color;
    output.w = g_PushConstants.transparency;

    return output;
}
