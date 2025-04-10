// © 2021 NVIDIA Corporation

#define DONT_DECLARE_RESOURCES

#include "NRI.hlsl"

#ifndef NRI_DXBC

#include "ForwardResources.hlsli"
#include "SceneViewerBindlessStructs.h"

NRI_RESOURCE(SamplerState, AnisotropicSampler, s, 0, 0 );
NRI_RESOURCE(StructuredBuffer<MaterialData>, Materials, t, 0, 0);
NRI_RESOURCE(StructuredBuffer<MeshData>, Meshes, t, 1, 0);
NRI_RESOURCE(StructuredBuffer<InstanceData>, Instances, t, 2, 0);
NRI_RESOURCE(Texture2D, Textures[], t, 0, 1);

struct BindlessAttributes
{
    float4 Position : SV_Position;
    float4 Normal : TEXCOORD0; //.w = TexCoord.x
    float4 View : TEXCOORD1; //.w = TexCoord.y
    float4 Tangent : TEXCOORD2;
    nointerpolation uint DrawParameters : ATTRIBUTES;
};

[earlydepthstencil]
float4 main( in BindlessAttributes input ) : SV_Target
{
    uint instanceIndex = input.DrawParameters;
    uint materialIndex = Instances[instanceIndex].materialIndex;

    uint baseColorTexIndex = Materials[materialIndex].baseColorTexIndex;
    uint roughnessMetalnessTexIndex = Materials[materialIndex].roughnessMetalnessTexIndex;
    uint normalTexIndex = Materials[materialIndex].normalTexIndex;
    uint emissiveTexIndex = Materials[materialIndex].emissiveTexIndex;

    Texture2D DiffuseMap = Textures[baseColorTexIndex];
    Texture2D SpecularMap = Textures[roughnessMetalnessTexIndex];
    Texture2D NormalMap = Textures[normalTexIndex];
    Texture2D EmissiveMap = Textures[emissiveTexIndex];

    float2 uv = float2( input.Normal.w, input.View.w );
    float3 V = normalize( input.View.xyz );
    float3 Nvertex = input.Normal.xyz;
    Nvertex = normalize( Nvertex );
    float4 T = input.Tangent;
    T.xyz = normalize( T.xyz );

    float4 diffuse = DiffuseMap.Sample( AnisotropicSampler, uv );
    float3 materialProps = SpecularMap.Sample( AnisotropicSampler, uv ).xyz;
    float3 emissive = EmissiveMap.Sample( AnisotropicSampler, uv ).xyz;
    float2 packedNormal = NormalMap.Sample( AnisotropicSampler, uv ).xy;

    float3 N = Geometry::TransformLocalNormal( packedNormal, T, Nvertex );
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0( diffuse.xyz, materialProps.z, albedo, Rf0 );
    float roughness = materialProps.y;
    const float3 sunDirection = normalize( float3( -0.8, -0.8, 1.0 ) );
    float3 L = ImportanceSampling::CorrectDirectionToInfiniteSource( N, sunDirection, V, tan( SUN_ANGULAR_SIZE ) );
    const float3 Clight = 80000.0;
    const float exposure = 0.00025;

    float4 output = Shade( float4( albedo, diffuse.w ), Rf0, roughness, emissive, N, L, V, Clight, FAKE_AMBIENT );
    output.xyz = Color::HdrToLinear( output.xyz * exposure );

    return output;
}

#else

    [earlydepthstencil]
    float4 main() : SV_Target
    {
        return 0;
    }

#endif
