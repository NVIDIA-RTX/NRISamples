// © 2026 NVIDIA Corporation

// "ResolveOp::SAMPLE_ZERO" resolves with different values in every sample (integer color, depth and stencil).
// Sample 0 is neither the minimum, nor the maximum, nor the average, so only a correct "sample 0" resolve passes

#include "TestShared.h"

namespace {

constexpr uint32_t SIZE = 8;
constexpr uint32_t ROW_PITCH = 256;
constexpr uint32_t PLANE_SIZE = ROW_PITCH * SIZE;
constexpr nri::Format COLOR_FORMAT = nri::Format::RGBA8_UINT;
nri::Format DEPTH_FORMAT = nri::Format::D32_SFLOAT_S8_UINT; // tested formats: see "main"

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

const nri::AccessLayoutStage COLOR_STATE = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
const nri::AccessLayoutStage DEPTH_STATE = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT};
const nri::AccessLayoutStage COPY_STATE = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
const nri::AccessLayoutStage RESOLVE_SRC_STATE = {nri::AccessBits::RESOLVE_SOURCE, nri::Layout::RESOLVE_SOURCE, nri::StageBits::RESOLVE};
const nri::AccessLayoutStage RESOLVE_DST_STATE = {nri::AccessBits::RESOLVE_DESTINATION, nri::Layout::RESOLVE_DESTINATION, nri::StageBits::RESOLVE};

// Draws the per-sample values into "colorMs" and "depthMs" in the given rendering
void Draw(test::Context& context, nri::CommandBuffer& commandBuffer, const Resources& r) {
    const nri::Viewport viewport = {0.0f, 0.0f, (float)SIZE, (float)SIZE, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE};
    context.core.CmdSetViewports(commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(commandBuffer, &scissor, 1);
    context.core.CmdSetPipelineLayout(commandBuffer, nri::BindPoint::GRAPHICS, *r.pipelineLayout);

    for (uint32_t i = 0; i < 3; i++) {
        context.core.CmdSetPipeline(commandBuffer, *r.pipelines[i]);
        context.core.CmdSetStencilReference(commandBuffer, (uint8_t)SAMPLE_VALUES[i].value, (uint8_t)SAMPLE_VALUES[i].value);
        nri::SetRootConstantsDesc rootConstants = {0, &SAMPLE_VALUES[i], sizeof(RootConstants)};
        context.core.CmdSetRootConstants(commandBuffer, rootConstants);
        context.core.CmdDraw(commandBuffer, {3, 1, 0, 0});
    }
}

bool VerifyColor(const uint8_t* data, const char* name) {
    bool passed = true;
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            const uint8_t* p = data + y * ROW_PITCH + x * 4;
            passed &= p[0] == 5 && p[1] == 6 && p[2] == 7 && p[3] == 8;
        }
    }

    std::string fullName = std::string(name) + " (" + nriGetFormatProps(DEPTH_FORMAT)->name + ")";

    return test::Report(fullName.c_str(), passed);
}

// D24: 24-bit UNORM in the low bits of a 32-bit texel, D32: float
float ReadDepth(const uint8_t* data, uint32_t x, uint32_t y) {
    const uint32_t texel = ((const uint32_t*)(data + y * ROW_PITCH))[x];
    if (DEPTH_FORMAT == nri::Format::D24_UNORM_S8_UINT)
        return (texel & 0xFFFFFF) / 16777215.0f;

    return ((const float*)(data + y * ROW_PITCH))[x];
}

bool VerifyDepthStencil(const uint8_t* data, float depth, uint8_t stencil, const char* name) {
    bool passed = true;
    for (uint32_t y = 0; y < SIZE; y++) {
        for (uint32_t x = 0; x < SIZE; x++) {
            passed &= std::abs(ReadDepth(data, x, y) - depth) < 0.0001f;
            passed &= data[PLANE_SIZE + y * ROW_PITCH + x] == stencil;
        }
    }

    if (!passed) {
        for (uint32_t i = 0; i < SIZE * SIZE; i++) {
            const uint32_t x = i % SIZE, y = i / SIZE;
            const float d = ReadDepth(data, x, y);
            const uint8_t s = data[PLANE_SIZE + y * ROW_PITCH + x];
            if (std::abs(d - depth) >= 0.0001f || s != stencil) {
                printf("      (%u, %u): depth=%f stencil=%u (expected %f / %u)\n", x, y, d, s, depth, stencil);
                break;
            }
        }
    }

    std::string fullName = std::string(name) + " (" + nriGetFormatProps(DEPTH_FORMAT)->name + ")";

    return test::Report(fullName.c_str(), passed);
}

// Depth and stencil plane upload + readback round trip ("bufferRowLength" of single-plane copies)
bool TestPlaneRoundTrip(test::Context& context, nri::Queue& queue) {
    if (!HasSupport(context, DEPTH_FORMAT, nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT))
        return true;

    nri::Texture* texture = nullptr;
    nri::Descriptor* view = nullptr;
    TEST_CHECK(CreateTexture(context, DEPTH_FORMAT, 1, true, texture, view));

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
            if (DEPTH_FORMAT == nri::Format::D24_UNORM_S8_UINT)
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
            passed &= std::abs(ReadDepth(data, x, y) - (y * SIZE + x + 1) / 128.0f) < 0.0001f;
            passed &= data[PLANE_SIZE + y * ROW_PITCH + x] == uint8_t(y * SIZE + x + 1);
        }
    }
    context.core.UnmapBuffer(*readback);

    std::string name = std::string("depth and stencil plane upload/readback round trip (") + nriGetFormatProps(DEPTH_FORMAT)->name + ")";

    return test::Report(name.c_str(), passed);
}

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    {
        nri::Queue* copyQueue = nullptr;
        TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, copyQueue));
        TEST_CHECK(TestPlaneRoundTrip(context, *copyQueue));
    }

    const nri::ResolveOps& attachmentOps = context.deviceDesc->resolve.attachment;
    const nri::ResolveOps& commandOps = context.deviceDesc->resolve.command;
    const nri::ResolveOpBits sampleZero = nri::ResolveOpBits::SAMPLE_ZERO;
    if (!(attachmentOps.colorInteger & sampleZero) || !(attachmentOps.depth & sampleZero) || !(attachmentOps.stencil & sampleZero) || !(commandOps.colorInteger & sampleZero)) {
        printf("SKIP  'SAMPLE_ZERO' is not supported for integer color, depth and stencil attachments and integer color 'CmdResolveTexture'\n");

        return true;
    }

    const bool isColorSupported = HasSupport(context, COLOR_FORMAT, nri::FormatSupportBits::COLOR_ATTACHMENT | nri::FormatSupportBits::MULTISAMPLE_4X | nri::FormatSupportBits::MULTISAMPLE_RESOLVE);
    const bool isDepthSupported = HasSupport(context, DEPTH_FORMAT, nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT | nri::FormatSupportBits::MULTISAMPLE_4X | nri::FormatSupportBits::MULTISAMPLE_RESOLVE);
    if (!isColorSupported || !isDepthSupported) {
        printf("SKIP  RGBA8_UINT (%u) or %s (%u) can't be resolved\n", isColorSupported, nriGetFormatProps(DEPTH_FORMAT)->name, isDepthSupported);

        return true;
    }

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

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
    pipelineDesc.outputMerger.depthStencilFormat = DEPTH_FORMAT;
    pipelineDesc.outputMerger.depth.compareOp = nri::CompareOp::ALWAYS;
    pipelineDesc.outputMerger.depth.write = true;
    pipelineDesc.outputMerger.stencil.front = {nri::CompareOp::ALWAYS, nri::StencilOp::REPLACE, nri::StencilOp::REPLACE, nri::StencilOp::REPLACE, 0xFF, 0xFF};
    pipelineDesc.outputMerger.stencil.back = pipelineDesc.outputMerger.stencil.front;
    pipelineDesc.shaders = shaders;
    pipelineDesc.shaderNum = 2;

    Resources r = {};
    r.pipelineLayout = pipelineLayout;
    for (uint32_t i = 0; i < 3; i++) {
        multisample.sampleMask = SAMPLE_MASKS[i];
        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, pipelineDesc, r.pipelines[i]));
        context.Track(r.pipelines[i]);
    }

    // Textures: multisampled sources and resolve destinations (A - attachments, B - "CmdResolveTexture", C - depth "MIN" + explicit stencil)
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 4, false, r.colorMs, r.colorMsView));
    TEST_CHECK(CreateTexture(context, DEPTH_FORMAT, 4, true, r.depthMs, r.depthMsView));

    nri::Texture* colorA = nullptr;
    nri::Texture* colorB = nullptr;
    nri::Texture* depthA = nullptr;
    nri::Texture* depthB = nullptr;
    nri::Texture* depthC = nullptr;
    nri::Descriptor* colorAView = nullptr;
    nri::Descriptor* colorBView = nullptr;
    nri::Descriptor* depthAView = nullptr;
    nri::Descriptor* depthBView = nullptr;
    nri::Descriptor* depthCView = nullptr;
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 1, false, colorA, colorAView));
    TEST_CHECK(CreateTexture(context, COLOR_FORMAT, 1, false, colorB, colorBView));
    TEST_CHECK(CreateTexture(context, DEPTH_FORMAT, 1, true, depthA, depthAView));
    TEST_CHECK(CreateTexture(context, DEPTH_FORMAT, 1, true, depthB, depthBView));
    TEST_CHECK(CreateTexture(context, DEPTH_FORMAT, 1, true, depthC, depthCView));

    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = PLANE_SIZE * 2 * 5;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, allocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::AccessLayoutStage colorMsState = {}, depthMsState = {}, colorAState = {}, colorBState = {}, depthAState = {}, depthBState = {}, depthCState = {};
    Barrier(context, *commandBuffer, r.colorMs, colorMsState, COLOR_STATE);
    Barrier(context, *commandBuffer, r.depthMs, depthMsState, DEPTH_STATE);
    Barrier(context, *commandBuffer, colorA, colorAState, COLOR_STATE);
    Barrier(context, *commandBuffer, depthA, depthAState, DEPTH_STATE);
    Barrier(context, *commandBuffer, depthC, depthCState, DEPTH_STATE);

    nri::AttachmentDesc color = {};
    color.descriptor = r.colorMsView;
    color.loadOp = nri::LoadOp::CLEAR;
    color.storeOp = nri::StoreOp::STORE;

    nri::AttachmentDesc depth = {};
    depth.descriptor = r.depthMsView;
    depth.loadOp = nri::LoadOp::CLEAR;
    depth.storeOp = nri::StoreOp::STORE;

    // A: attachment resolves, the stencil plane of the combined "depth" attachment is resolved with "depth.resolveOp"
    {
        color.resolveDst = colorAView;
        color.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        nri::RenderingDesc rendering = {};
        rendering.colors = &color;
        rendering.colorNum = 1;
        rendering.depth = depth;
        rendering.depth.resolveDst = depthAView;
        rendering.depth.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        context.core.CmdBeginRendering(*commandBuffer, rendering);
        Draw(context, *commandBuffer, r);
        context.core.CmdEndRendering(*commandBuffer);
        color.resolveDst = nullptr;
    }

    // C: depth "MIN" and explicit stencil "SAMPLE_ZERO" (separate "stencil" attachment, same resolve destination)
    const bool isIndependentResolve = (attachmentOps.depth & nri::ResolveOpBits::MIN) && context.deviceDesc->resolve.independentDepthStencil;
    if (isIndependentResolve) {
        nri::RenderingDesc rendering = {};
        rendering.colors = &color;
        rendering.colorNum = 1;
        rendering.depth = depth;
        rendering.depth.resolveDst = depthCView;
        rendering.depth.resolveOp = nri::ResolveOp::MIN;
        rendering.stencil = depth;
        rendering.stencil.resolveDst = depthCView;
        rendering.stencil.resolveOp = nri::ResolveOp::SAMPLE_ZERO;

        context.core.CmdBeginRendering(*commandBuffer, rendering);
        Draw(context, *commandBuffer, r);
        context.core.CmdEndRendering(*commandBuffer);
    }

    // B: "CmdResolveTexture" (depth-stencil only if supported by "resolve.command")
    const bool isDepthCommandResolve = (commandOps.depth & sampleZero) && (commandOps.stencil & sampleZero);
    Barrier(context, *commandBuffer, r.colorMs, colorMsState, RESOLVE_SRC_STATE);
    Barrier(context, *commandBuffer, colorB, colorBState, RESOLVE_DST_STATE);
    context.core.CmdResolveTexture(*commandBuffer, *colorB, nullptr, *r.colorMs, nullptr, nri::ResolveOp::SAMPLE_ZERO);
    if (isDepthCommandResolve) {
        Barrier(context, *commandBuffer, r.depthMs, depthMsState, RESOLVE_SRC_STATE);
        Barrier(context, *commandBuffer, depthB, depthBState, RESOLVE_DST_STATE);
        context.core.CmdResolveTexture(*commandBuffer, *depthB, nullptr, *r.depthMs, nullptr, nri::ResolveOp::SAMPLE_ZERO);
    }

    // Readback
    nri::Texture* colors[] = {colorA, colorB};
    nri::AccessLayoutStage* colorStates[] = {&colorAState, &colorBState};
    for (uint32_t i = 0; i < 2; i++) {
        Barrier(context, *commandBuffer, colors[i], *colorStates[i], COPY_STATE);
        nri::TextureRegionDesc region = {0, 0, 0, (nri::Dim_t)SIZE, (nri::Dim_t)SIZE, 1, 0, 0, nri::PlaneBits::ALL};
        context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, {PLANE_SIZE * 2 * i, ROW_PITCH, PLANE_SIZE}, *colors[i], region);
    }

    nri::Texture* depths[] = {depthA, depthB, depthC};
    nri::AccessLayoutStage* depthStates[] = {&depthAState, &depthBState, &depthCState};
    for (uint32_t i = 0; i < 3; i++) {
        if (i == 1 && !isDepthCommandResolve)
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
    passed &= VerifyColor(data, "SAMPLE_ZERO: integer color attachment resolve");
    passed &= VerifyColor(data + PLANE_SIZE * 2, "SAMPLE_ZERO: integer color CmdResolveTexture");
    passed &= VerifyDepthStencil(data + PLANE_SIZE * 4, 0.5f, 5, "SAMPLE_ZERO: depth-stencil attachment resolve (stencil through 'depth')");
    if (isDepthCommandResolve)
        passed &= VerifyDepthStencil(data + PLANE_SIZE * 6, 0.5f, 5, "SAMPLE_ZERO: depth-stencil CmdResolveTexture");
    if (isIndependentResolve)
        passed &= VerifyDepthStencil(data + PLANE_SIZE * 8, 0.2f, 5, "depth MIN + explicit stencil SAMPLE_ZERO attachment resolve");

    context.core.UnmapBuffer(*readback);

    return passed;
}

} // namespace

int main(int argc, char** argv) {
    const test::Settings settings = test::ParseSettings(argc, argv);

    bool passed = true;
    for (nri::Format format : {nri::Format::D32_SFLOAT_S8_UINT, nri::Format::D24_UNORM_S8_UINT}) {
        DEPTH_FORMAT = format;
        passed &= Run(settings);
    }

    return passed ? 0 : 1;
}
