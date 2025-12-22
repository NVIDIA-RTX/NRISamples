// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

struct outputVS
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

outputVS main
(
    uint vertexID : SV_VertexID
)
{
    outputVS output;

    output.position.zw = float2( 0.0, 1.0 );

    if( vertexID == 0 )
        output.position.xy = float2( -1, -1 );
    else if( vertexID == 1 )
        output.position.xy = float2( 1, -1 );
    else if( vertexID == 2 )
        output.position.xy = float2( -1, 1 );
    else
        output.position.xy = float2( 1, 1 );

    output.texCoord = NRI_CLIP_TO_UV( output.position.xy );

    return output;
}
