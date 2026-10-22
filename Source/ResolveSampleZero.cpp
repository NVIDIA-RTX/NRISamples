// © 2026 NVIDIA Corporation

// "ResolveOp::SAMPLE_ZERO" resolves with different values in every sample (integer color, depth and stencil).
// Sample 0 is neither the minimum, nor the maximum, nor the average, so only a correct "sample 0" resolve passes

#include "TestShared.h"

namespace {

constexpr uint32_t SIZE = 8;
constexpr uint32_t ROW_PITCH = 256;
constexpr uint32_t PLANE_SIZE = ROW_PITCH * SIZE;
constexpr nri::Format COLOR_FORMAT = nri::Format::RGBA8_UINT;

struct RootConstants {
    float depth;
    uint32_t value;
};

// Sample masks and per-sample values: sample 0 = 0.5 / 5, sample 1 = 0.2 / 2, samples 2-3 = 0.8 / 9
constexpr uint32_t SAMPLE_MASKS[] = {0x1, 0x2, 0xC};
constexpr RootConstants SAMPLE_VALUES[] = {{0.5f, 5}, {0.2f, 2}, {0.8f, 9}};

struct Resources {
    nri::PipelineLayout* pipelineLayout = nullptr;
    nri::Texture* colorMs = nullptr;
    nri::Texture* depthMs = nullptr;
    nri::Descriptor* colorMsView = nullptr;
    nri::Descriptor* depthMsView = nullptr;
    nri::Pipeline* pipelines[3] = {};
};

bool HasSupport(test::Context& context, nri::Format format, nri::FormatSupportBits bits) {
    return (uint32_t(context.core.GetFormatSupport(*context.device, format)) & uint32_t(bits)) == uint32_t(bits);
}

bool CreateTexture(test::Context& context, nri::Format format, uint32_t sampleNum, bool isDepth, nri::Texture*& texture, nri::Descriptor*& view) {
    nri::TextureDesc desc = {};
    desc.type = nri::TextureType::TEXTURE_2D;
    desc.usage = isDepth ? nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT : nri::TextureUsageBits::COLOR_ATTACHMENT;
    desc.format = format;
    desc.width = SIZE;
    desc.height = SIZE;
    desc.sampleNum = (nri::Sample_t)sampleNum;
    TEST_CHECK(context.CreateTexture(desc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = texture;
    viewDesc.type = isDepth ? nri::TextureView::DEPTH_STENCIL_ATTACHMENT : nri::TextureView::COLOR_ATTACHMENT;
    viewDesc.format = format;
    viewDesc.mipNum = viewDesc.layerNum = viewDesc.sliceNum = 1;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, view));
    context.Track(view);

    return true;
}

void Barrier(test::Context& context, nri::CommandBuffer& commandBuffer, nri::Texture* texture, nri::AccessLayoutStage& state, nri::AccessLayoutStage after) {
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = texture;
    barrier.mipNum = barrier.layerNum = 1;
    barrier.before = state;
    barrier.after = after;

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &barrier;
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(commandBuffer, barrierDesc);

    state = after;
}

constexpr nri::AccessLayoutStage COLOR_STATE = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
constexpr nri::AccessLayoutStage DEPTH_STATE = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT};
constexpr nri::AccessLayoutStage COLOR_RESOLVE_ATTACHMENT_SRC_STATE = {nri::AccessBits::COLOR_ATTACHMENT | nri::AccessBits::RESOLVE_ATTACHMENT_SOURCE, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT | nri::StageBits::RESOLVE_ATTACHMENT};
constexpr nri::AccessLayoutStage DEPTH_RESOLVE_ATTACHMENT_SRC_STATE = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT | nri::AccessBits::RESOLVE_ATTACHMENT_SOURCE, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT | nri::StageBits::RESOLVE_ATTACHMENT};
constexpr nri::AccessLayoutStage COLOR_RESOLVE_ATTACHMENT_DST_STATE = {nri::AccessBits::RESOLVE_ATTACHMENT_DESTINATION, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::RESOLVE_ATTACHMENT};
constexpr nri::AccessLayoutStage DEPTH_RESOLVE_ATTACHMENT_DST_STATE = {nri::AccessBits::RESOLVE_ATTACHMENT_DESTINATION, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::RESOLVE_ATTACHMENT};
constexpr nri::AccessLayoutStage COPY_STATE = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
constexpr nri::AccessLayoutStage RESOLVE_SRC_STATE = {nri::AccessBits::RESOLVE_SOURCE, nri::Layout::RESOLVE_SOURCE, nri::StageBits::RESOLVE};
constexpr nri::AccessLayoutStage RESOLVE_DST_STATE = {nri::AccessBits::RESOLVE_DESTINATION, nri::Layout::RESOLVE_DESTINATION, nri::StageBits::RESOLVE};

// Draws the per-sample values into "colorMs" and "depthMs" in the given rendering
void Draw(test::Context& context, nri::CommandBuffer& commandBuffer, const Resources& resources) {
    const nri::Viewport viewport = {0.0f, 0.0f, (float)SIZE, (float)SIZE, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE};
    context.core.CmdSetViewports(commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(commandBuffer, &scissor, 1);
    context.core.CmdSetPipelineLayout(commandBuffer, nri::BindPoint::GRAPHICS, *resources.pipelineLayout);

    for (uint32_t i = 0; i < 3; i++) {
        context.core.CmdSetPipeline(commandBuffer, *resources.pipelines[i]);
        context.core.CmdSetStencilReference(commandBuffer, (uint8_t)SAMPLE_VALUES[i].value, (uint8_t)SAMPLE_VALUES[i].value);
        nri::SetRootConstantsDesc rootConstants = {0, &SAMPLE_VALUES[i], sizeof(RootConstants)};
        context.core.CmdSetRootConstants(commandBuffer, rootConstants);
        context.core.CmdDraw(commandBuffer, {3, 1, 0, 0});
    }
}

bool VerifyColor(const uint8_t* data, const char* name, nri::Format depthFormat) {
    bool passed = true;
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            const uint8_t* p = data + y * ROW_PITCH + x * 4;
            passed &= p[0] == 5 && p[1] == 6 && p[2] == 7 && p[3] == 8;
        }
    }

    std::string fullName = std::string(name) + " (" + nriGetFormatProps(depthFormat)->name + ")";

    return test::Report(fullName.c_str(), passed);
}

// D24: 24-bit UNORM in the low bits of a 32-bit texel, D32: float
float ReadDepth(const uint8_t* data, uint32_t x, uint32_t y, nri::Format depthFormat) {
    uint32_t texel;
    memcpy(&texel, data + y * ROW_PITCH + x * sizeof(uint32_t), sizeof(texel));
    if (depthFormat == nri::Format::D24_UNORM_S8_UINT)
        return (texel & 0xFFFFFF) / 16777215.0f;

    float depth;
    memcpy(&depth, &texel, sizeof(depth));

    return depth;
}

bool VerifyDepthStencil(const uint8_t* data, float depth, uint8_t stencil, const char* name, nri::Format depthFormat) {
    bool passed = true;
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            passed &= std::abs(ReadDepth(data, x, y, depthFormat) - depth) < 0.0001f;
            passed &= data[PLANE_SIZE + y * ROW_PITCH + x] == stencil;
        }
    }

    if (!passed) {
        for (uint32_t i = 0; i < SIZE * SIZE; i++) {
            const uint32_t x = i % SIZE;
            const uint32_t y = i / SIZE;
            const float d = ReadDepth(data, x, y, depthFormat);
            const uint8_t s = data[PLANE_SIZE + y * ROW_PITCH + x];
            if (std::abs(d - depth) >= 0.0001f || s != stencil) {
                printf("      (%u, %u): depth=%f stencil=%u (expected %f / %u)\n", x, y, d, s, depth, stencil);
                break;
            }
        }
    }

    std::string fullName = std::string(name) + " (" + nriGetFormatProps(depthFormat)->name + ")";

    return test::Report(fullName.c_str(), passed);
}

// Depth and stencil plane upload + readback round trip ("bufferRowLength" of single-plane copies)
bool TestPlaneRoundTrip(test::Context& context, nri::Queue& queue, nri::Format depthFormat) {
    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    if (graphicsAPI == nri::GraphicsAPI::D3D11 || graphicsAPI == nri::GraphicsAPI::WGPU) {
        printf("SKIP  Depth/stencil plane upload round trip is unsupported on %s\n", nri::nriGetGraphicsAPIString(graphicsAPI));

        return true;
    }

    if (!HasSupport(context, depthFormat, nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT))
        return true;

    context.skipped = false;

    nri::Texture* texture = nullptr;
    nri::Descriptor* view = nullptr;
    TEST_CHECK(CreateTexture(context, depthFormat, 1, true, texture, view));

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = PLANE_SIZE * 2;
    nri::Buffer* upload = nullptr;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, upload));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readback));

    // Unique values per pixel: depth (D24: 24-bit UNORM, D32: float) and stencil
    uint8_t* src = (uint8_t*)context.core.MapBuffer(*upload, 0, nri::WHOLE_SIZE);
    TEST_CHECK(src != nullptr);
    memset(src, 0, PLANE_SIZE * 2);
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            const float depth = (y * SIZE + x + 1) / 128.0f;
            if (depthFormat == nri::Format::D24_UNORM_S8_UINT)
                ((uint32_t*)(src + y * ROW_PITCH))[x] = uint32_t(depth * 16777215.0f + 0.5f);
            else
                ((float*)(src + y * ROW_PITCH))[x] = depth;
            src[PLANE_SIZE + y * ROW_PITCH + x] = uint8_t(y * SIZE + x + 1);
        }
    }
    context.core.UnmapBuffer(*upload);

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::AccessLayoutStage state = {};
    Barrier(context, *commandBuffer, texture, state, {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY});
    nri::TextureRegionDesc region = {0, 0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE, 1, 0, 0, nri::PlaneBits::DEPTH};
    context.core.CmdUploadBufferToTexture(*commandBuffer, *texture, region, *upload, {0, ROW_PITCH, PLANE_SIZE});
    region.planes = nri::PlaneBits::STENCIL;
    context.core.CmdUploadBufferToTexture(*commandBuffer, *texture, region, *upload, {PLANE_SIZE, ROW_PITCH, PLANE_SIZE});

    Barrier(context, *commandBuffer, texture, state, COPY_STATE);
    region.planes = nri::PlaneBits::DEPTH;
    context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {0, ROW_PITCH, PLANE_SIZE}, *texture, region);
    region.planes = nri::PlaneBits::STENCIL;
    context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE, ROW_PITCH, PLANE_SIZE}, *texture, region);
    TEST_CHECK(context.SubmitAndWait(queue, *commandBuffer));

    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);

    bool passed = true;
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            passed &= std::abs(ReadDepth(data, x, y, depthFormat) - (y * SIZE + x + 1) / 128.0f) < 0.0001f;
            passed &= data[PLANE_SIZE + y * ROW_PITCH + x] == uint8_t(y * SIZE + x + 1);
        }
    }
    context.core.UnmapBuffer(*readback);

    std::string name = std::string("depth and stencil plane upload/readback round trip (") + nriGetFormatProps(depthFormat)->name + ")";

    return test::Report(name.c_str(), passed);
}

// Exercise attachment resolve scopes independently of SAMPLE_ZERO support: arrays, a destination mip,
// global barriers between resolves, and read-only/mixed depth-stencil sources (with shader-read usage).
bool TestAttachmentResolve(test::Context& context, nri::Queue& queue, nri::Format format, bool depthStencil) {
    const nri::ResolveOps& ops = context.deviceDesc->resolve.attachment;
    nri::ResolveOpBits supported = ops.color;
    if (depthStencil)
        supported = nri::ResolveOpBits(ops.depth & ops.stencil);
    nri::ResolveOp resolveOp;
    if (supported & nri::ResolveOpBits::MIN)
        resolveOp = nri::ResolveOp::MIN;
    else if (supported & nri::ResolveOpBits::SAMPLE_ZERO)
        resolveOp = nri::ResolveOp::SAMPLE_ZERO;
    else if (supported & nri::ResolveOpBits::AVERAGE)
        resolveOp = nri::ResolveOp::AVERAGE;
    else {
        printf("SKIP  Attachment resolve scope test (%s): no supported MIN, SAMPLE_ZERO or AVERAGE resolve\n", nriGetFormatProps(format)->name);
        return true;
    }

    nri::FormatSupportBits required = (depthStencil ? nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT : nri::FormatSupportBits::COLOR_ATTACHMENT)
        | nri::FormatSupportBits::MULTISAMPLE_4X | nri::FormatSupportBits::MULTISAMPLE_RESOLVE;
    if (!HasSupport(context, format, required)) {
        printf("SKIP  Attachment resolve scope test (%s): format does not support the required attachment, MSAA or resolve usage\n", nriGetFormatProps(format)->name);
        return true;
    }

    context.skipped = false;

    // WGPU has no multisampled arrays; D3D11 currently resolves a single attachment layer.
    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    const nri::Dim_t layerNum = (graphicsAPI == nri::GraphicsAPI::D3D11 || graphicsAPI == nri::GraphicsAPI::WGPU) ? 1 : 2;
    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = depthStencil ? (nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT | nri::TextureUsageBits::SHADER_RESOURCE)
        : (nri::TextureUsageBits::COLOR_ATTACHMENT | nri::TextureUsageBits::INPUT_ATTACHMENT);
    textureDesc.format = format;
    textureDesc.width = textureDesc.height = SIZE;
    textureDesc.layerNum = layerNum;
    textureDesc.sampleNum = 4;
    nri::Texture* source = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, source));

    textureDesc.width = textureDesc.height = SIZE * 2;
    textureDesc.layerNum = 3;
    textureDesc.mipNum = 2;
    textureDesc.sampleNum = 1;
    nri::Texture* destination = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, destination));

    const nri::Layout layouts[] = {nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::Layout::DEPTH_STENCIL_READONLY,
        nri::Layout::DEPTH_READONLY_STENCIL_ATTACHMENT, nri::Layout::DEPTH_ATTACHMENT_STENCIL_READONLY};
    nri::Descriptor* sourceViews[4] = {};
    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = source;
    viewDesc.type = depthStencil ? nri::TextureView::DEPTH_STENCIL_ATTACHMENT : nri::TextureView::COLOR_ATTACHMENT;
    viewDesc.format = format;
    viewDesc.mipNum = viewDesc.sliceNum = 1;
    viewDesc.layerNum = layerNum;
    const uint32_t passNum = depthStencil ? 4 : 2;
    for (uint32_t pass = 0; pass < passNum; pass++) {
        // Test both explicit and implicit attachment layouts on input-attachment textures.
        viewDesc.layout = depthStencil ? layouts[pass] : (pass ? nri::Layout::UNDEFINED : nri::Layout::COLOR_ATTACHMENT);
        TEST_CHECK(context.core.CreateTextureView(viewDesc, sourceViews[pass]));
        context.Track(sourceViews[pass]);
    }

    viewDesc.texture = destination;
    viewDesc.layout = depthStencil ? nri::Layout::DEPTH_STENCIL_ATTACHMENT : nri::Layout::COLOR_ATTACHMENT;
    viewDesc.mipOffset = viewDesc.layerOffset = 1;
    nri::Descriptor* destinationView = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, destinationView));
    context.Track(destinationView);

    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = PLANE_SIZE * 4;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::TextureBarrierDesc barriers[2] = {};
    barriers[0].texture = source;
    barriers[0].mipNum = 1;
    barriers[0].layerNum = layerNum;
    barriers[1].texture = destination;
    barriers[1].mipOffset = barriers[1].layerOffset = 1;
    barriers[1].mipNum = 1;
    barriers[1].layerNum = layerNum;
    barriers[1].after = depthStencil ? DEPTH_RESOLVE_ATTACHMENT_DST_STATE : COLOR_RESOLVE_ATTACHMENT_DST_STATE;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = barriers;
    for (uint32_t pass = 0; pass < passNum; pass++) {
        barriers[0].after = depthStencil ? DEPTH_RESOLVE_ATTACHMENT_SRC_STATE : COLOR_RESOLVE_ATTACHMENT_SRC_STATE;
        if (!depthStencil && pass)
            barriers[0].after.stages = nri::StageBits::GRAPHICS | nri::StageBits::RESOLVE_ATTACHMENT;
        if (depthStencil) {
            barriers[0].after.layout = layouts[pass];
            if (pass == 1)
                barriers[0].after.access = nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_READ | nri::AccessBits::RESOLVE_ATTACHMENT_SOURCE;
            if (pass) {
                barriers[0].after.access |= nri::AccessBits::SHADER_RESOURCE;
                barriers[0].after.stages |= nri::StageBits::FRAGMENT_SHADER;
            }
        }
        barrierDesc.textureNum = pass ? 1 : 2;
        context.core.CmdBarrier(*commandBuffer, barrierDesc);
        nri::AttachmentDesc attachment = {};
        attachment.descriptor = sourceViews[pass];
        attachment.resolveDst = destinationView;
        attachment.resolveOp = resolveOp;
        attachment.loadOp = pass ? nri::LoadOp::LOAD : nri::LoadOp::CLEAR;
        attachment.storeOp = nri::StoreOp::STORE;
        if (depthStencil)
            attachment.clearValue.depthStencil = {0.5f, 5};
        else
            attachment.clearValue.color.f = {5.0f / 255.0f, 6.0f / 255.0f, 7.0f / 255.0f, 8.0f / 255.0f};

        nri::RenderingDesc rendering = {};
        if (depthStencil) {
            rendering.depth = rendering.stencil = attachment;
            if (pass == 1 || pass == 2)
                rendering.depth.storeOp = nri::StoreOp::NONE;
            if (pass == 1 || pass == 3)
                rendering.stencil.storeOp = nri::StoreOp::NONE;
        } else {
            rendering.colors = &attachment;
            rendering.colorNum = 1;
        }
        context.core.CmdBeginRendering(*commandBuffer, rendering);
        context.core.CmdEndRendering(*commandBuffer);
        barriers[0].before = barriers[0].after;

        nri::GlobalBarrierDesc global = {};
        global.before = global.after = {nri::AccessBits::RESOLVE_ATTACHMENT_DESTINATION, nri::StageBits::RESOLVE_ATTACHMENT};
        nri::BarrierDesc globalDesc = {};
        globalDesc.globals = &global;
        globalDesc.globalNum = 1;
        context.core.CmdBarrier(*commandBuffer, globalDesc);
    }

    barriers[1].before = barriers[1].after;
    barriers[1].after = COPY_STATE;
    barrierDesc.textures = &barriers[1];
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    uint32_t planeNum = depthStencil ? 2 : 1;
    for (uint32_t layer = 0; layer < layerNum; layer++) {
        for (uint32_t plane = 0; plane < planeNum; plane++) {
            nri::TextureRegionDesc region = {0, 0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE, 1, 1, (nri::Dim_t)(layer + 1),
                depthStencil ? (plane ? nri::PlaneBits::STENCIL : nri::PlaneBits::DEPTH) : nri::PlaneBits::COLOR};
            context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE * (layer * planeNum + plane), ROW_PITCH, PLANE_SIZE}, *destination, region);
        }
    }
    TEST_CHECK(context.SubmitAndWait(queue, *commandBuffer));
    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);
    bool passed = true;
    for (uint32_t layer = 0; layer < layerNum; layer++) {
        const uint8_t* layerData = data + PLANE_SIZE * layer * planeNum;
        if (depthStencil)
            passed &= VerifyDepthStencil(layerData, 0.5f, 5, "attachment resolve scopes: array layer and destination mip, read-only/mixed sources, global barrier", format);
        else
            passed &= VerifyColor(layerData, "attachment resolve scopes: destination mip/layer, global barrier", format);
    }
    context.core.UnmapBuffer(*readback);

    return passed;
}

bool Run(test::Context& context, const test::Settings& settings, nri::Format depthFormat) {
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    context.skipped = true; // cleared when a supported subtest runs
    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));
    TEST_CHECK(TestPlaneRoundTrip(context, *queue, depthFormat));
    TEST_CHECK(TestAttachmentResolve(context, *queue, nri::Format::RGBA8_UNORM, false));
    TEST_CHECK(TestAttachmentResolve(context, *queue, depthFormat, true));

    const nri::ResolveOps& attachmentOps = context.deviceDesc->resolve.attachment;
    const nri::ResolveOps& commandOps = context.deviceDesc->resolve.command;
    const nri::ResolveOpBits sampleZero = nri::ResolveOpBits::SAMPLE_ZERO;
    if (!(attachmentOps.colorInteger & sampleZero) || !(attachmentOps.depth & sampleZero) || !(attachmentOps.stencil & sampleZero) || !(commandOps.colorInteger & sampleZero)) {
        printf("SKIP  'SAMPLE_ZERO' is not supported for integer color, depth and stencil attachments and integer color 'CmdResolveTexture'\n");

        return true;
    }

    const bool isColorSupported = HasSupport(context, COLOR_FORMAT, nri::FormatSupportBits::COLOR_ATTACHMENT | nri::FormatSupportBits::MULTISAMPLE_4X | nri::FormatSupportBits::MULTISAMPLE_RESOLVE);
    const bool isDepthSupported = HasSupport(context, depthFormat, nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT | nri::FormatSupportBits::MULTISAMPLE_4X | nri::FormatSupportBits::MULTISAMPLE_RESOLVE);
    if (!isColorSupported || !isDepthSupported) {
        printf("SKIP  RGBA8_UINT (%u) or %s (%u) can't be resolved\n", isColorSupported, nriGetFormatProps(depthFormat)->name, isDepthSupported);

        return true;
    }

    context.skipped = false;

    // Pipelines
    nri::RootConstantDesc rootConstant = {0, sizeof(RootConstants), nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER};
    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.rootConstantNum = 1;
    pipelineLayoutDesc.rootConstants = &rootConstant;
    pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    const nri::ShaderDesc shaders[] = {
        utils::LoadShader(graphicsAPI, "ResolveSampleZero.vs", context.shaderStorage),
        utils::LoadShader(graphicsAPI, "ResolveSampleZero.fs", context.shaderStorage),
    };

    nri::ColorAttachmentDesc colorDesc = {};
    colorDesc.format = COLOR_FORMAT;
    colorDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::GraphicsPipelineDesc pipelineDesc = {};
    pipelineDesc.pipelineLayout = pipelineLayout;
    pipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    pipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    pipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    nri::MultisampleDesc multisample = {};
    multisample.sampleNum = 4;
    pipelineDesc.multisample = &multisample;
    pipelineDesc.outputMerger.colors = &colorDesc;
    pipelineDesc.outputMerger.colorNum = 1;
    pipelineDesc.outputMerger.depthStencilFormat = depthFormat;
    pipelineDesc.outputMerger.depth.compareOp = nri::CompareOp::ALWAYS;
    pipelineDesc.outputMerger.depth.write = true;
    pipelineDesc.outputMerger.stencil.front = {nri::CompareOp::ALWAYS, nri::StencilOp::REPLACE, nri::StencilOp::REPLACE, nri::StencilOp::REPLACE, 0xFF, 0xFF};
    pipelineDesc.outputMerger.stencil.back = pipelineDesc.outputMerger.stencil.front;
    pipelineDesc.shaders = shaders;
    pipelineDesc.shaderNum = 2;

    Resources resources = {};
    resources.pipelineLayout = pipelineLayout;
    for (uint32_t i = 0; i < 3; i++) {
        multisample.sampleMask = SAMPLE_MASKS[i];
        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, resources.pipelines[i]));
        context.Track(resources.pipelines[i]);
    }

    // Multisampled sources and destinations for attachment, command and independent depth/stencil resolves
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 4, false, resources.colorMs, resources.colorMsView));
    TEST_CHECK(CreateTexture(context, depthFormat, 4, true, resources.depthMs, resources.depthMsView));

    nri::Texture* colorAttachmentResolve = nullptr;
    nri::Texture* colorCommandResolve = nullptr;
    nri::Texture* depthAttachmentResolve = nullptr;
    nri::Texture* depthCommandResolve = nullptr;
    nri::Texture* depthIndependentResolve = nullptr;
    nri::Descriptor* colorAttachmentResolveView = nullptr;
    nri::Descriptor* colorCommandResolveView = nullptr;
    nri::Descriptor* depthAttachmentResolveView = nullptr;
    nri::Descriptor* depthCommandResolveView = nullptr;
    nri::Descriptor* depthIndependentResolveView = nullptr;
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 1, false, colorAttachmentResolve, colorAttachmentResolveView));
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 1, false, colorCommandResolve, colorCommandResolveView));
    TEST_CHECK(CreateTexture(context, depthFormat, 1, true, depthAttachmentResolve, depthAttachmentResolveView));
    TEST_CHECK(CreateTexture(context, depthFormat, 1, true, depthCommandResolve, depthCommandResolveView));
    TEST_CHECK(CreateTexture(context, depthFormat, 1, true, depthIndependentResolve, depthIndependentResolveView));

    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = PLANE_SIZE * 2 * 5;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::AccessLayoutStage colorMsState = {};
    nri::AccessLayoutStage depthMsState = {};
    nri::AccessLayoutStage colorAttachmentResolveState = {};
    nri::AccessLayoutStage colorCommandResolveState = {};
    nri::AccessLayoutStage depthAttachmentResolveState = {};
    nri::AccessLayoutStage depthCommandResolveState = {};
    nri::AccessLayoutStage depthIndependentResolveState = {};
    Barrier(context, *commandBuffer, resources.colorMs, colorMsState, COLOR_RESOLVE_ATTACHMENT_SRC_STATE);
    Barrier(context, *commandBuffer, resources.depthMs, depthMsState, DEPTH_RESOLVE_ATTACHMENT_SRC_STATE);
    Barrier(context, *commandBuffer, colorAttachmentResolve, colorAttachmentResolveState, COLOR_RESOLVE_ATTACHMENT_DST_STATE);
    Barrier(context, *commandBuffer, depthAttachmentResolve, depthAttachmentResolveState, DEPTH_RESOLVE_ATTACHMENT_DST_STATE);
    Barrier(context, *commandBuffer, depthIndependentResolve, depthIndependentResolveState, DEPTH_RESOLVE_ATTACHMENT_DST_STATE);

    nri::AttachmentDesc color = {};
    color.descriptor = resources.colorMsView;
    color.loadOp = nri::LoadOp::CLEAR;
    color.storeOp = nri::StoreOp::STORE;

    nri::AttachmentDesc depth = {};
    depth.descriptor = resources.depthMsView;
    depth.loadOp = nri::LoadOp::CLEAR;
    depth.storeOp = nri::StoreOp::STORE;

    // The stencil plane of the combined "depth" attachment is resolved with "depth.resolveOp"
    {
        color.resolveDst = colorAttachmentResolveView;
        color.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        nri::RenderingDesc rendering = {};
        rendering.colors = &color;
        rendering.colorNum = 1;
        rendering.depth = depth;
        rendering.depth.resolveDst = depthAttachmentResolveView;
        rendering.depth.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        context.core.CmdBeginRendering(*commandBuffer, rendering);
        Draw(context, *commandBuffer, resources);
        context.core.CmdEndRendering(*commandBuffer);
        color.resolveDst = nullptr;
    }

    // Depth "MIN" and explicit stencil "SAMPLE_ZERO" (separate "stencil" attachment, same resolve destination)
    const bool isIndependentResolve = (attachmentOps.depth & nri::ResolveOpBits::MIN) && context.deviceDesc->resolve.independentDepthStencil;
    if (isIndependentResolve) {
        Barrier(context, *commandBuffer, resources.colorMs, colorMsState, COLOR_STATE);
        Barrier(context, *commandBuffer, resources.depthMs, depthMsState, DEPTH_RESOLVE_ATTACHMENT_SRC_STATE);

        nri::RenderingDesc rendering = {};
        rendering.colors = &color;
        rendering.colorNum = 1;
        rendering.depth = depth;
        rendering.depth.resolveDst = depthIndependentResolveView;
        rendering.depth.resolveOp = nri::ResolveOp::MIN;
        rendering.stencil = depth;
        rendering.stencil.resolveDst = depthIndependentResolveView;
        rendering.stencil.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        context.core.CmdBeginRendering(*commandBuffer, rendering);
        Draw(context, *commandBuffer, resources);
        context.core.CmdEndRendering(*commandBuffer);
    }

    // "CmdResolveTexture" (depth-stencil only if supported by "resolve.command")
    const bool isDepthCommandResolve = (commandOps.depth & sampleZero) && (commandOps.stencil & sampleZero);
    Barrier(context, *commandBuffer, resources.colorMs, colorMsState, RESOLVE_SRC_STATE);
    Barrier(context, *commandBuffer, colorCommandResolve, colorCommandResolveState, RESOLVE_DST_STATE);
    context.core.CmdResolveTexture(*commandBuffer, *colorCommandResolve, nullptr, *resources.colorMs, nullptr, nri::ResolveOp::SAMPLE_ZERO);
    if (isDepthCommandResolve) {
        Barrier(context, *commandBuffer, resources.depthMs, depthMsState, RESOLVE_SRC_STATE);
        Barrier(context, *commandBuffer, depthCommandResolve, depthCommandResolveState, RESOLVE_DST_STATE);
        context.core.CmdResolveTexture(*commandBuffer, *depthCommandResolve, nullptr, *resources.depthMs, nullptr, nri::ResolveOp::SAMPLE_ZERO);
    }

    // Readback
    nri::Texture* colors[] = {colorAttachmentResolve, colorCommandResolve};
    nri::AccessLayoutStage* colorStates[] = {&colorAttachmentResolveState, &colorCommandResolveState};
    for (uint32_t i = 0; i < 2; i++) {
        Barrier(context, *commandBuffer, colors[i], *colorStates[i], COPY_STATE);
        nri::TextureRegionDesc region = {0, 0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE, 1, 0, 0, nri::PlaneBits::ALL};
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE * 2 * i, ROW_PITCH, PLANE_SIZE}, *colors[i], region);
    }

    nri::Texture* depths[] = {depthAttachmentResolve, depthCommandResolve, depthIndependentResolve};
    nri::AccessLayoutStage* depthStates[] = {&depthAttachmentResolveState, &depthCommandResolveState, &depthIndependentResolveState};
    for (uint32_t i = 0; i < 3; i++) {
        if ((i == 1 && !isDepthCommandResolve) || (i == 2 && !isIndependentResolve))
            continue;

        Barrier(context, *commandBuffer, depths[i], *depthStates[i], COPY_STATE);
        nri::TextureRegionDesc region = {0, 0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE, 1, 0, 0, nri::PlaneBits::DEPTH};
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE * 2 * (2 + i), ROW_PITCH, PLANE_SIZE}, *depths[i], region);
        region.planes = nri::PlaneBits::STENCIL;
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE * 2 * (2 + i) + PLANE_SIZE, ROW_PITCH, PLANE_SIZE}, *depths[i], region);
    }

    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    const uint8_t* data = (const uint8_t*)context.core.MapBuffer(*readback, 0, nri::WHOLE_SIZE);
    TEST_CHECK(data != nullptr);

    bool passed = true;
    passed &= VerifyColor(data, "SAMPLE_ZERO: integer color attachment resolve", depthFormat);
    passed &= VerifyColor(data + PLANE_SIZE * 2, "SAMPLE_ZERO: integer color CmdResolveTexture", depthFormat);
    passed &= VerifyDepthStencil(data + PLANE_SIZE * 4, 0.5f, 5, "SAMPLE_ZERO: depth-stencil attachment resolve (stencil through 'depth')", depthFormat);
    if (isDepthCommandResolve)
        passed &= VerifyDepthStencil(data + PLANE_SIZE * 6, 0.5f, 5, "SAMPLE_ZERO: depth-stencil CmdResolveTexture", depthFormat);
    if (isIndependentResolve)
        passed &= VerifyDepthStencil(data + PLANE_SIZE * 8, 0.2f, 5, "depth MIN + explicit stencil SAMPLE_ZERO attachment resolve", depthFormat);

    context.core.UnmapBuffer(*readback);

    return passed;
}

} // namespace

int main(int argc, char** argv) {
    const test::Settings settings = test::ParseSettings(argc, argv);

    bool passed = true;
    bool skipped = true;
    for (nri::Format format : {nri::Format::D32_SFLOAT_S8_UINT, nri::Format::D24_UNORM_S8_UINT}) {
        test::Context context;
        passed &= Run(context, settings, format);
        skipped &= context.skipped;
    }

    return passed ? (skipped ? NRIF_SKIP : NRIF_OK) : NRIF_FAIL;
}
