// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool CreateView(test::Context& context, const nri::TextureViewDesc& desc) {
    nri::Descriptor* descriptor = nullptr;
    TEST_CHECK(context.core.CreateTextureView(desc, descriptor));
    context.Track(descriptor);

    return true;
}

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    const nri::Format format = nri::Format::RGBA8_UNORM;
    const nri::FormatSupportBits formatSupport = context.core.GetFormatSupport(*context.device, format);
    if (!(formatSupport & nri::FormatSupportBits::TEXTURE)) {
        printf("SKIP  RGBA8_UNORM textures are unsupported\n");

        return true;
    }

    nri::TextureUsageBits usage = nri::TextureUsageBits::SHADER_RESOURCE;
    if (formatSupport & nri::FormatSupportBits::STORAGE_TEXTURE)
        usage |= nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;

    // WebGPU 1D textures don't support arrays or mipmaps.
    const bool isWGPU = context.deviceDesc->graphicsAPI == nri::GraphicsAPI::WGPU;

    nri::TextureDesc texture1DDesc = {};
    texture1DDesc.type = nri::TextureType::TEXTURE_1D;
    texture1DDesc.usage = usage;
    texture1DDesc.format = format;
    texture1DDesc.width = 64;
    texture1DDesc.mipNum = isWGPU ? 1 : 4;
    texture1DDesc.layerNum = isWGPU ? 1 : 4;

    nri::TextureDesc texture2DDesc = {};
    texture2DDesc.type = nri::TextureType::TEXTURE_2D;
    texture2DDesc.usage = usage;
    texture2DDesc.format = format;
    texture2DDesc.width = 32;
    texture2DDesc.height = 32;
    texture2DDesc.mipNum = 3;
    texture2DDesc.layerNum = 12;

    nri::TextureDesc texture3DDesc = {};
    texture3DDesc.type = nri::TextureType::TEXTURE_3D;
    texture3DDesc.usage = usage;
    texture3DDesc.format = format;
    texture3DDesc.width = 16;
    texture3DDesc.height = 8;
    texture3DDesc.depth = 4;
    texture3DDesc.mipNum = 3;

    nri::Texture* texture1D = nullptr;
    nri::Texture* texture2D = nullptr;
    nri::Texture* texture3D = nullptr;
    TEST_CHECK(context.CreateTexture(texture1DDesc, nri::MemoryLocation::DEVICE, texture1D));
    TEST_CHECK(context.CreateTexture(texture2DDesc, nri::MemoryLocation::DEVICE, texture2D));
    TEST_CHECK(context.CreateTexture(texture3DDesc, nri::MemoryLocation::DEVICE, texture3D));

    if (context.deviceDesc->features.getMemoryDesc2) {
        nri::MemoryDesc memoryDesc = {};
        context.core.GetTextureMemoryDesc2(*context.device, texture3DDesc, nri::MemoryLocation::DEVICE, memoryDesc);
        if (!memoryDesc.size || !memoryDesc.alignment) {
            printf("FAIL  GetTextureMemoryDesc2 returned an empty description\n");

            return false;
        }
    }

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = texture1D;
    viewDesc.type = texture1DDesc.layerNum > 1 ? nri::TextureView::TEXTURE_ARRAY : nri::TextureView::TEXTURE;
    viewDesc.format = format;
    viewDesc.mipNum = nri::REMAINING;
    viewDesc.layerNum = nri::REMAINING;
    viewDesc.sliceNum = nri::REMAINING;
    TEST_CHECK(CreateView(context, viewDesc));

    if (usage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE) {
        viewDesc.type = texture1DDesc.layerNum > 1 ? nri::TextureView::STORAGE_TEXTURE_ARRAY : nri::TextureView::STORAGE_TEXTURE;
        viewDesc.mipNum = 1;
        TEST_CHECK(CreateView(context, viewDesc));
    }

    viewDesc = {};
    viewDesc.texture = texture2D;
    viewDesc.type = nri::TextureView::TEXTURE_CUBE;
    viewDesc.format = format;
    viewDesc.mipNum = nri::REMAINING;
    viewDesc.layerNum = 6;
    viewDesc.sliceNum = nri::REMAINING;
    TEST_CHECK(CreateView(context, viewDesc));

    viewDesc.type = nri::TextureView::TEXTURE_CUBE_ARRAY;
    viewDesc.layerNum = 12;
    TEST_CHECK(CreateView(context, viewDesc));

    viewDesc.type = nri::TextureView::TEXTURE_ARRAY;
    viewDesc.layerOffset = 3;
    viewDesc.layerNum = 5;
    if (context.deviceDesc->features.componentSwizzle)
        viewDesc.components = {nri::ComponentSwizzle::B, nri::ComponentSwizzle::G, nri::ComponentSwizzle::R, nri::ComponentSwizzle::ONE};
    TEST_CHECK(CreateView(context, viewDesc));

    if (usage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE) {
        viewDesc.type = nri::TextureView::STORAGE_TEXTURE_ARRAY;
        viewDesc.mipNum = 1;
        viewDesc.components = {};
        TEST_CHECK(CreateView(context, viewDesc));
    }

    viewDesc = {};
    viewDesc.texture = texture3D;
    viewDesc.type = nri::TextureView::TEXTURE;
    viewDesc.format = format;
    viewDesc.mipNum = 1;
    viewDesc.layerNum = nri::REMAINING;
    viewDesc.sliceOffset = 1;
    viewDesc.sliceNum = 2;
    TEST_CHECK(CreateView(context, viewDesc));

    if (usage & nri::TextureUsageBits::SHADER_RESOURCE_STORAGE) {
        viewDesc.type = nri::TextureView::STORAGE_TEXTURE;
        viewDesc.mipNum = 1;
        TEST_CHECK(CreateView(context, viewDesc));
    }

    nri::SamplerDesc samplerDesc = {};
    samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::NEAREST, nri::Filter::LINEAR, nri::FilterOp::AVERAGE};
    samplerDesc.addressModes = {nri::AddressMode::MIRRORED_REPEAT, nri::AddressMode::CLAMP_TO_EDGE, nri::AddressMode::REPEAT};
    samplerDesc.mipMax = 16.0f;
    samplerDesc.anisotropy = 1;

    nri::Descriptor* sampler = nullptr;
    TEST_CHECK(context.core.CreateSampler(*context.device, samplerDesc, sampler));
    context.Track(sampler);

    if (context.deviceDesc->features.filterOpMinMax) {
        samplerDesc.filters.op = nri::FilterOp::MIN;
        TEST_CHECK(context.core.CreateSampler(*context.device, samplerDesc, sampler));
        context.Track(sampler);
    }

    return test::Report("texture types and views", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
