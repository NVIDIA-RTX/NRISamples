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
    float motionSpeed;
    float detailStrength;
    float diffStrength;
    uint showDifference;
    uint frameIndex;
};

NRI_ROOT_CONSTANTS(PatternRootConstants, g_Pattern, 0, 0);

float Stripe(float v, float frequency, float width) {
    const float phase = frac(v * frequency);
    return smoothstep(width, 0.0f, min(phase, 1.0f - phase));
}

float RectMask(float2 p, float2 minP, float2 maxP, float feather) {
    const float2 lo = smoothstep(minP, minP + feather, p);
    const float2 hi = 1.0f - smoothstep(maxP - feather, maxP, p);
    return lo.x * lo.y * hi.x * hi.y;
}

float3 ColorWheel(float phase) {
    const float a = phase * 6.283185307179586f;
    return saturate(0.5f + 0.5f * cos(a + float3(0.0f, 4.1887902047863905f, 2.0943951023931953f)));
}

float3 MakePatternColor(float2 normalizedPixelPos, float time, uint frameIndex) {
    const float fx = normalizedPixelPos.x;
    const float fy = normalizedPixelPos.y;
    const float cx = fx - 0.5f;
    const float cy = fy - 0.5f;
    const float radius = sqrt(cx * cx + cy * cy);
    const float angle = atan2(cy, cx);
    const float t = time * g_Pattern.motionSpeed;
    const float detail = g_Pattern.detailStrength;
    const float detail01 = saturate(detail);

    // Quiet base image: broad gradients and mild chroma rotation reveal banding
    // and color drift without hiding problems under noise.
    const float lumaRamp = fx;
    const float verticalRamp = fy;
    const float radialFalloff = saturate(1.0f - radius * 1.45f);
    float3 color = lerp(float3(lumaRamp, lumaRamp, lumaRamp), float3(0.12f + 0.78f * lumaRamp, 0.10f + 0.72f * verticalRamp, 0.18f + 0.62f * radialFalloff), 0.42f);

    // Top-left: clean grayscale and chroma ramps for banding/chroma subsampling.
    const float rampPanel = RectMask(normalizedPixelPos, float2(0.035f, 0.055f), float2(0.325f, 0.390f), 0.006f);
    const float graySteps = floor(fx * 32.0f) / 31.0f;
    const float smoothGray = saturate((fx - 0.035f) / 0.290f);
    const float panelSplit = step(0.225f, fy);
    const float3 rampColor = lerp(float3(smoothGray, smoothGray, smoothGray), ColorWheel(smoothGray + 0.08f * sin(t)), panelSplit);
    color = lerp(color, rampColor, rampPanel);
    color = lerp(color, float3(graySteps, graySteps, graySteps), rampPanel * (1.0f - panelSplit) * 0.28f * detail01);

    // Center: moving slanted hard edge plus soft rings. Ringing and block edges
    // are easy to see here, but the region remains spatially coherent.
    const float edgePanel = RectMask(normalizedPixelPos, float2(0.365f, 0.070f), float2(0.665f, 0.610f), 0.006f);
    const float edge = smoothstep(-0.004f, 0.004f, (fx - 0.515f) + (fy - 0.33f) * 0.82f + sin(t * 0.7f) * 0.085f);
    const float rings = sin((radius * 20.0f - t * 0.75f) * 6.283185307179586f) * 0.5f + 0.5f;
    const float3 edgeColor = lerp(float3(0.04f, 0.06f, 0.08f), float3(0.92f, 0.90f, 0.84f), edge);
    color = lerp(color, lerp(edgeColor, float3(rings, rings * 0.65f, 1.0f - rings), 0.22f), edgePanel);

    // Right: bounded detail patterns with no frame-to-frame flicker. This exposes
    // luma blur, chroma bleed and block behavior without high-contrast temporal noise.
    const float detailPanel = RectMask(normalizedPixelPos, float2(0.700f, 0.070f), float2(0.965f, 0.610f), 0.006f);
    const float2 detailUv = saturate((normalizedPixelPos - float2(0.700f, 0.070f)) / float2(0.265f, 0.540f));
    const float barFreq = lerp(10.0f, 28.0f, detail01);
    const float lumaBars = 0.5f + 0.38f * sin((detailUv.x * barFreq + 0.12f * sin(t * 0.35f)) * 6.283185307179586f);
    const float chromaWave = 0.5f + 0.35f * sin((detailUv.y * 7.0f - t * 0.035f) * 6.283185307179586f);
    const float diagonalWedge = smoothstep(0.012f, 0.0f, abs(frac((detailUv.x + detailUv.y) * 8.0f) - 0.5f));
    const float3 softBars = lerp(float3(lumaBars, lumaBars, lumaBars), float3(0.78f, 0.24f + 0.48f * chromaWave, 0.28f + 0.42f * (1.0f - chromaWave)), 0.42f);
    const float3 wedgeColor = lerp(softBars, float3(0.18f, 0.22f, 0.26f), diagonalWedge * 0.28f * detail01);
    color = lerp(color, wedgeColor, detailPanel);

    // Bottom: moving color wheel and measuring grid. Temporal glitches, repeated
    // frames and chroma phase issues are obvious but still structured.
    const float motionPanel = RectMask(normalizedPixelPos, float2(0.035f, 0.665f), float2(0.965f, 0.935f), 0.006f);
    const float wheel = frac(fx * 1.75f + t * 0.10f);
    const float markerX = frac(t * 0.16f);
    const float marker = smoothstep(0.018f, 0.0f, abs(fx - markerX)) * smoothstep(0.0f, 0.04f, fy - 0.665f) * smoothstep(0.0f, 0.04f, 0.935f - fy);
    const float frameBlink = ((frameIndex / 8u) & 1u) ? 1.0f : 0.0f;
    const float grid = max(Stripe(fx, 16.0f, 0.018f), Stripe(fy, 9.0f, 0.020f));
    float3 motionColor = ColorWheel(wheel);
    motionColor = lerp(motionColor, float3(0.0f, 0.0f, 0.0f), grid * 0.35f * detail01);
    motionColor = lerp(motionColor, lerp(float3(1.0f, 1.0f, 1.0f), float3(0.0f, 0.0f, 0.0f), frameBlink), marker);
    color = lerp(color, motionColor, motionPanel);

    return saturate(color);
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
    return float4(MakePatternColor(float2(px, py) / float2(g_Pattern.width - 1u, g_Pattern.height - 1u), g_Pattern.time, g_Pattern.frameIndex), 1.0f);
}

uint3 LoadPatternRgb8(uint px, uint py) {
    return uint3(LoadPatternColor(px, py).rgb * 255.0f);
}

float3 LoadPatternNv12Rgb(uint px, uint py) {
    const uint chromaX = px & ~1u;
    const uint chromaY = py & ~1u;

    const uint3 rgb = LoadPatternRgb8(px, py);
    const uint3 chromaRgb0 = LoadPatternRgb8(chromaX + 0u, chromaY + 0u);
    const uint3 chromaRgb1 = LoadPatternRgb8(chromaX + 1u, chromaY + 0u);
    const uint3 chromaRgb2 = LoadPatternRgb8(chromaX + 0u, chromaY + 1u);
    const uint3 chromaRgb3 = LoadPatternRgb8(chromaX + 1u, chromaY + 1u);

    const uint yValue = RGBToY(rgb);
    const uint uValue = (RGBToU(chromaRgb0) + RGBToU(chromaRgb1) + RGBToU(chromaRgb2) + RGBToU(chromaRgb3) + 2u) >> 2u;
    const uint vValue = (RGBToV(chromaRgb0) + RGBToV(chromaRgb1) + RGBToV(chromaRgb2) + RGBToV(chromaRgb3) + 2u) >> 2u;

    return YuvToRgb(yValue, uValue, vValue);
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

        const uint3 rgb0 = LoadPatternRgb8(blockX + 0u, y);
        const uint3 rgb1 = LoadPatternRgb8(blockX + 1u, y);
        const uint3 rgb2 = LoadPatternRgb8(blockX + 2u, y);
        const uint3 rgb3 = LoadPatternRgb8(blockX + 3u, y);

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

        const uint3 rgb4 = LoadPatternRgb8(blockX + 0u, y + 1u);
        const uint3 rgb5 = LoadPatternRgb8(blockX + 1u, y + 1u);
        const uint3 rgb6 = LoadPatternRgb8(blockX + 2u, y + 1u);
        const uint3 rgb7 = LoadPatternRgb8(blockX + 3u, y + 1u);

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

            const float3 decodedRgb = YuvToRgb(yValue, uValue, vValue);
            if (g_Pattern.showDifference != 0u) {
                const float3 sourceRgb = LoadPatternNv12Rgb(px, y);
                const float3 diffRgb = saturate(abs(sourceRgb - decodedRgb) * g_Pattern.diffStrength);
                StorePreview(uint2(px, y), float4(diffRgb, 1.0f));
            } else {
                StorePreview(uint2(px, y), float4(decodedRgb, 1.0f));
            }
        }
    }
}
