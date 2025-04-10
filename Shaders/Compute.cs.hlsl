// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

NRI_FORMAT("r16f") NRI_RESOURCE(RWBuffer<float>, buffer, u, 0, 0);

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    v ^= v >> 16u;

    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;

    return v;
}

[numthreads(256, 1, 1)]
void main(uint threadID : SV_DispatchThreadId)
{
    float value = buffer[threadID];
    uint3 result = uint3(asuint(value), asuint(value) >> 12, asuint(value) * asuint(value));

    [loop]
    for (uint i = 0; i < 128; i++)
        result = pcg3d(result);

    buffer[threadID] = result.x * result.y * result.z;
}