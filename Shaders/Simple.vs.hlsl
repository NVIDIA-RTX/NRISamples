// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

struct InputVS
{
    float3 position : POSITION;
    float2 texCoords : TEXCOORD0;
};

struct OutputVS
{
    float4 position : SV_Position;
    float2 texCoords : TEXCOORD0;
};

NRI_RESOURCE( cbuffer, Constants, b, 0, 0 )
{
    float4x4 transform;
};

OutputVS main( in InputVS input )
{
    OutputVS output;
    output.position = mul( transform, float4( input.position, 1 ) );
    output.texCoords = input.texCoords;
    return output;
}
