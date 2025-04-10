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

NRI_RESOURCE( cbuffer, GeometryConstants, b, 0, 0 )
{
    float4x4 transform;
};

NRI_RESOURCE( cbuffer, GlobalConstants, b, 1, 0 )
{
    float4 globalConstants;
};

NRI_RESOURCE( cbuffer, ViewConstants, b, 2, 0 )
{
    float4x4 projView;
    float4 viewConstants;
};

NRI_RESOURCE( cbuffer, MaterialConstants, b, 3, 0 )
{
    float4 materialConstants;
};

OutputVS main( in InputVS input )
{
    const float4 constants = globalConstants + viewConstants + materialConstants;

    OutputVS output;
    output.position = mul( projView, mul( transform, float4( input.position, 1 ) + constants ) );
    output.texCoords = input.texCoords;

    return output;
}
