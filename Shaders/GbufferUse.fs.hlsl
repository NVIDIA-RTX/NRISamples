// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

// Root
NRI_RESOURCE( cbuffer, CommonConstants, b, 1, 1 )
{
    float3 L;
};

// Resources
NRI_INPUT_ATTACHMENT( g_Normals, 1, 1, 0 );

struct outputVS
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

float4 main( in outputVS input ) : SV_Target0
{
    uint2 pixelPos = uint2( input.position.xy );

    float3 N = NRI_INPUT_ATTACHMENT_LOAD( g_Normals, pixelPos ).xyz;
    N = normalize( N * 2.0 - 1.0 );

    float4 output;
    output.xyz = saturate( dot( N, normalize( L ) ) );
    output.w = 1.0;

    return output;
}
