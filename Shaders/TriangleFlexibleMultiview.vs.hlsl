// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

NRI_RESOURCE( cbuffer, CommonConstants, b, 0, 0 )
{
    float3 color;
    float scale;
};

struct outputVS
{
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
#ifdef NRI_DXIL
    uint viewportIndex : SV_ViewportArrayIndex;
#endif
};

outputVS main
(
    float2 inPos : POSITION0,
    float2 inTexCoord : TEXCOORD0
#ifdef NRI_DXIL
    , uint viewID : SV_ViewID
#endif
)
{
    outputVS output;

    output.position.xy = inPos * scale;
    output.position.zw = float2( 0.0, 1.0 );
    output.texCoord = inTexCoord;
#ifdef NRI_DXIL
    output.viewportIndex = viewID;
#endif

    return output;
}
