// © 2021 NVIDIA Corporation

#include "NRI.hlsl"

NRI_RESOURCE(RWBuffer<uint>, g_Nv12Buffer, u, 0, 0);
NRI_FORMAT("rgba8") NRI_RESOURCE(RWTexture2D<float4>, g_SourcePreview, u, 1, 0);
NRI_FORMAT("rgba8") NRI_RESOURCE(RWTexture2D<float4>, g_DecodePreview, u, 2, 0);

static const uint OP_GENERATE_PATTERN = 0u;
static const uint OP_NV12_TO_PREVIEW = 1u;

struct PatternRootConstants {
    uint width;
    uint height;
    uint yOffsetBytes;
    uint yRowPitchBytes;
    uint uvRowPitchBytes;
    uint uvOffsetBytes;
    uint operation;
    float time;
    uint padding;
    uint padding1;
};

NRI_ROOT_CONSTANTS(PatternRootConstants, g_Pattern, 0, 0);

float3 MakePatternColor(float2 normalizedPixelPos, float time) {
    const float fx = normalizedPixelPos.x;
    const float fy = normalizedPixelPos.y;
    const float cx = fx - 0.5f;
    const float cy = fy - 0.5f;
    const float radius = sqrt(cx * cx + cy * cy);
    const float angle = atan2(cy, cx);

    const float sweep = sin(angle * 3.0f + time * 1.7f) * 0.5f + 0.5f;
    const float rings = sin((radius * 16.0f - time * 1.25f) * 6.283185307179586f) * 0.5f + 0.5f;
    const float diagonal = sin((fx * 5.0f + fy * 3.0f + time * 0.45f) * 6.283185307179586f) * 0.5f + 0.5f;
    const float grid = (((uint)(fx * 16.0f) ^ (uint)(fy * 10.0f)) & 1) ? 0.08f : 0.0f;

    const float r = min(1.0f, 0.10f + 0.78f * sweep + 0.18f * diagonal + grid);
    const float g = min(1.0f, 0.14f + 0.72f * rings + 0.20f * fy + grid);
    const float b = min(1.0f, 0.18f + 0.52f * diagonal + 0.34f * (1.0f - radius) + grid);

    return float3(r, g, b);
}

uint ClampToByte(float v) {
    return (v <= 0.0f) ? 0u : (v >= 255.0f ? 255u : uint(v));
}

uint RGBToY(uint3 rgb) {
    return ClampToByte(16.0f + 0.257f * float(rgb.x) + 0.504f * float(rgb.y) + 0.098f * float(rgb.z));
}

uint RGBToU(uint3 rgb) {
    return ClampToByte(128.0f - 0.148f * float(rgb.x) - 0.291f * float(rgb.y) + 0.439f * float(rgb.z));
}

uint RGBToV(uint3 rgb) {
    return ClampToByte(128.0f + 0.439f * float(rgb.x) - 0.368f * float(rgb.y) - 0.071f * float(rgb.z));
}

float3 YuvToRgb(uint y, uint u, uint v) {
    const float yy = float(y);
    const float uu = float(u) - 128.0f;
    const float vv = float(v) - 128.0f;

    const float r = clamp((298.082f * (yy - 16.0f) + 408.583f * vv + 128.0f) / 256.0f, 0.0f, 255.0f);
    const float g = clamp((298.082f * (yy - 16.0f) - 100.291f * uu - 208.120f * vv + 128.0f) / 256.0f, 0.0f, 255.0f);
    const float b = clamp((298.082f * (yy - 16.0f) + 516.412f * uu + 128.0f) / 256.0f, 0.0f, 255.0f);

    return float3(r / 255.0f, g / 255.0f, b / 255.0f);
}

float4 LoadPatternColor(uint px, uint py) {
    return float4(MakePatternColor(float2(px, py) / float2(g_Pattern.width - 1u, g_Pattern.height - 1u), g_Pattern.time), 1.0f);
}

void StorePreview(uint2 pixel, float4 color) {
    if (g_Pattern.operation == OP_NV12_TO_PREVIEW)
        g_DecodePreview[pixel] = color;
    else
        g_SourcePreview[pixel] = color;
}

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint blockX = dispatchThreadID.x * 4u;
    const uint y = dispatchThreadID.y;

    if (blockX + 3u >= g_Pattern.width || y >= g_Pattern.height)
        return;

    const float4 c0 = LoadPatternColor(blockX + 0u, y);
    const float4 c1 = LoadPatternColor(blockX + 1u, y);
    const float4 c2 = LoadPatternColor(blockX + 2u, y);
    const float4 c3 = LoadPatternColor(blockX + 3u, y);

    if (g_Pattern.operation == OP_GENERATE_PATTERN) {
        StorePreview(uint2(blockX + 0u, y), c0);
        StorePreview(uint2(blockX + 1u, y), c1);
        StorePreview(uint2(blockX + 2u, y), c2);
        StorePreview(uint2(blockX + 3u, y), c3);

        const uint3 rgb0 = uint3(c0.rgb * 255.0f);
        const uint3 rgb1 = uint3(c1.rgb * 255.0f);
        const uint3 rgb2 = uint3(c2.rgb * 255.0f);
        const uint3 rgb3 = uint3(c3.rgb * 255.0f);

        const uint y0 = RGBToY(rgb0);
        const uint y1 = RGBToY(rgb1);
        const uint y2 = RGBToY(rgb2);
        const uint y3 = RGBToY(rgb3);

        const uint packedY = y0 | (y1 << 8u) | (y2 << 16u) | (y3 << 24u);
        const uint yWordIndex = (g_Pattern.yOffsetBytes + y * g_Pattern.yRowPitchBytes + blockX) / 4u;
        g_Nv12Buffer[yWordIndex] = packedY;

        if ((y & 1u) != 0u)
            return;
        if (y + 1u >= g_Pattern.height)
            return;

        const float4 c4 = LoadPatternColor(blockX + 0u, y + 1u);
        const float4 c5 = LoadPatternColor(blockX + 1u, y + 1u);
        const float4 c6 = LoadPatternColor(blockX + 2u, y + 1u);
        const float4 c7 = LoadPatternColor(blockX + 3u, y + 1u);

        const uint3 rgb4 = uint3(c4.rgb * 255.0f);
        const uint3 rgb5 = uint3(c5.rgb * 255.0f);
        const uint3 rgb6 = uint3(c6.rgb * 255.0f);
        const uint3 rgb7 = uint3(c7.rgb * 255.0f);

        const uint u0 = (RGBToU(rgb0) + RGBToU(rgb1) + RGBToU(rgb4) + RGBToU(rgb5) + 2u) >> 2u;
        const uint v0 = (RGBToV(rgb0) + RGBToV(rgb1) + RGBToV(rgb4) + RGBToV(rgb5) + 2u) >> 2u;
        const uint u1 = (RGBToU(rgb2) + RGBToU(rgb3) + RGBToU(rgb6) + RGBToU(rgb7) + 2u) >> 2u;
        const uint v1 = (RGBToV(rgb2) + RGBToV(rgb3) + RGBToV(rgb6) + RGBToV(rgb7) + 2u) >> 2u;

        const uint packedUV = u0 | (v0 << 8u) | (u1 << 16u) | (v1 << 24u);
        const uint uvWordIndex = (g_Pattern.uvOffsetBytes + ((y >> 1u) * g_Pattern.uvRowPitchBytes + blockX)) / 4u;
        g_Nv12Buffer[uvWordIndex] = packedUV;
        return;
    }

    if (g_Pattern.operation == OP_NV12_TO_PREVIEW) {
        const uint uvBase = (g_Pattern.uvOffsetBytes + (y >> 1u) * g_Pattern.uvRowPitchBytes + blockX) + ((y & 1u) * 0u);
        for (uint i = 0; i < 4; i++) {
            const uint px = blockX + i;
            const uint yIndex = y * g_Pattern.yRowPitchBytes + px + g_Pattern.yOffsetBytes;
            const uint yWord = g_Nv12Buffer[yIndex / 4u];
            const uint yValue = (yWord >> ((yIndex & 3u) * 8u)) & 255u;

            const uint uvOffset = uvBase + (i & 2u);
            const uint uvWord = g_Nv12Buffer[(uvOffset) / 4u];
            const uint uvShift = (uvOffset & 3u) * 8u;
            const uint uValue = (uvWord >> uvShift) & 255u;
            const uint vValue = (uvWord >> (uvShift + 8u)) & 255u;

            StorePreview(uint2(px, y), float4(YuvToRgb(yValue, uValue, vValue), 1.0f));
        }
    }
}
