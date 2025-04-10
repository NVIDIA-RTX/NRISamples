// © 2021 NVIDIA Corporation

#include "NRI.hlsl"
#include "ForwardResources.hlsli"

float4 main( in Attributes input ) : SV_Target
{
    PS_INPUT;
    float4 output = Shade( float4( albedo, diffuse.w ), Rf0, roughness, emissive, N, L, V, Clight, FAKE_AMBIENT );
    if( output.w < 0.5 )
        discard;

    output.xyz = Color::HdrToLinear( output.xyz * exposure );
    return output;
}
