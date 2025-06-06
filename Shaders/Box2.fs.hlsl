// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

NRI_RESOURCE( cbuffer, GlobalConstants, b, 1, 0 )
{
    float4 globalConstants;
};

NRI_RESOURCE( cbuffer, ViewConstants, b, 2, 0 )
{
    float4 viewConstants;
};

NRI_RESOURCE( cbuffer, MaterialConstants, b, 3, 0 )
{
    float4 materialConstants;
};

NRI_RESOURCE( SamplerState, sampler0, s, 0, 1 );
NRI_RESOURCE( Texture2D, texture0, t, 0, 0 );
NRI_RESOURCE( Texture2D, texture1, t, 1, 0 );
NRI_RESOURCE( Texture2D, texture2, t, 2, 0 );

struct OutputVS
{
    float4 position : SV_Position;
    float2 texCoords : TEXCOORD0;
};

float4 main( in OutputVS input ) : SV_Target
{
    const float4 constants = globalConstants + viewConstants + materialConstants;
    const float4 sample0 = texture0.Sample( sampler0, input.texCoords );
    const float4 sample1 = texture1.Sample( sampler0, input.texCoords );
    const float4 sample2 = texture2.Sample( sampler0, input.texCoords );

    return sample0 + constants + sample1 * 0.001 + sample2 * 0.0028;
}
