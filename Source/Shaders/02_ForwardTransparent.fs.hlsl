/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "BindingBridge.hlsli"
#include "02_Resources.hlsli"

float4 main( in Attributes input, bool isFrontFace : SV_IsFrontFace ) : SV_Target
{
    PS_INPUT;
    N = isFrontFace ? N : -N;

    float4 output = Shade( float4( albedo, diffuse.w ), Rf0, roughness, emissive, N, L, V, Clight, FAKE_AMBIENT | GLASS_HACK );

    output.xyz = STL::Color::HdrToLinear( output.xyz * exposure );
    return output;
}
