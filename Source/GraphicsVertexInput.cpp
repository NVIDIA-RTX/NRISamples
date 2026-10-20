// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool CreatePipeline(test::Context& context, const nri::GraphicsPipelineDesc& desc) {
    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, desc, pipeline));
    context.Track(pipeline);

    return true;
}

bool CreateCachedPipeline(test::Context& context, nri::GraphicsPipelineDesc desc, nri::Pipeline*& pipeline) {
    struct Cache {
        const nri::CoreInterface& core;
        nri::PipelineCache* object = nullptr;

        ~Cache() {
            core.DestroyPipelineCache(object);
        }
    } cache{context.core};

    if (context.deviceDesc->features.pipelineCache) {
        const nri::PipelineCacheDesc empty = {};
        TEST_CHECK(context.core.CreatePipelineCache(*context.device, empty, cache.object));
        desc.cache = cache.object;

        if (context.deviceDesc->features.pipelineCacheControl) {
            desc.flags = nri::GraphicsPipelineBits::FAIL_ON_CACHE_MISS;
            TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, desc, pipeline) != nri::Result::SUCCESS && pipeline == nullptr);
            desc.flags = nri::GraphicsPipelineBits::NONE;
        }
    }

    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, desc, pipeline));
    context.Track(pipeline);

    if (cache.object) {
        uint64_t size = 0;
        TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, nullptr, size));
        std::vector<uint8_t> bytes(size);
        TEST_CHECK(context.core.GetPipelineCacheData(*cache.object, bytes.data(), size));
        context.core.DestroyPipelineCache(cache.object);
        cache.object = nullptr;
        const nri::PipelineCacheDesc saved = {bytes.data(), size};
        TEST_CHECK(context.core.CreatePipelineCache(*context.device, saved, cache.object));
        desc.cache = cache.object;

        if (context.deviceDesc->features.pipelineCacheControl)
            desc.flags = nri::GraphicsPipelineBits::FAIL_ON_CACHE_MISS;

        TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, desc, pipeline));
        context.Track(pipeline);
    }

    return true;
}

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    if (context.deviceDesc->features.geometryShader)
        pipelineLayoutDesc.shaderStages |= nri::StageBits::GEOMETRY_SHADER;
    if (context.deviceDesc->features.tessellationShader)
        pipelineLayoutDesc.shaderStages |= nri::StageBits::TESSELLATION_SHADERS;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    const nri::ShaderDesc shaders[] = {
        utils::LoadShader(graphicsAPI, "GraphicsVertexInput.vs", context.shaderStorage),
        utils::LoadShader(graphicsAPI, "GraphicsPipelineStates.fs", context.shaderStorage),
    };

    const nri::Format depthStencilFormat = nri::GetSupportedDepthFormat(context.core, *context.device, 24, true);
    if (depthStencilFormat == nri::Format::UNKNOWN) {
        printf("SKIP  No depth-stencil format is supported\n");

        return true;
    }

    nri::ColorAttachmentDesc colorDesc = {};
    colorDesc.format = nri::Format::RGBA8_UNORM;
    colorDesc.colorWriteMask = nri::ColorWriteBits::RGBA;
    colorDesc.blendEnabled = true;
    colorDesc.colorBlend = {nri::BlendFactor::CONSTANT_COLOR, nri::BlendFactor::ONE_MINUS_CONSTANT_COLOR, nri::BlendOp::ADD};
    colorDesc.alphaBlend = {nri::BlendFactor::ONE, nri::BlendFactor::ZERO, nri::BlendOp::ADD};

    nri::OutputMergerDesc outputMerger = {};
    outputMerger.colors = &colorDesc;
    outputMerger.colorNum = 1;
    outputMerger.depthStencilFormat = depthStencilFormat;
    outputMerger.depth.compareOp = nri::CompareOp::LESS_EQUAL;
    outputMerger.depth.write = true;
    outputMerger.depth.boundsTest = context.deviceDesc->features.depthBoundsTest;
    outputMerger.stencil.front = {nri::CompareOp::ALWAYS, nri::StencilOp::KEEP, nri::StencilOp::REPLACE, nri::StencilOp::KEEP, 0xFF, 0xFF};
    outputMerger.stencil.back = outputMerger.stencil.front;

    const nri::VertexStreamDesc stream = {0, nri::VertexStreamStepRate::PER_VERTEX, 16};
    nri::VertexAttributeDesc attribute = {};
    attribute.format = nri::Format::RG32_SFLOAT;
    attribute.offset = 4;
    attribute.d3d = {"POSITION", 0};
    const nri::VertexInputDesc vertexInput = {&attribute, 1, &stream, 1};
    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = pipelineLayout;
    graphicsPipelineDesc.vertexInput = &vertexInput;
    graphicsPipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    graphicsPipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    graphicsPipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    graphicsPipelineDesc.rasterization.depthBias = {1.0f, 0.0f, 1.0f};
    graphicsPipelineDesc.outputMerger = outputMerger;
    graphicsPipelineDesc.shaders = shaders;
    graphicsPipelineDesc.shaderNum = 2;

    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, pipeline));
    context.Track(pipeline);

    if (context.deviceDesc->features.logicOp) {
        nri::ColorAttachmentDesc logicColorDesc = colorDesc;
        logicColorDesc.blendEnabled = false;
        nri::GraphicsPipelineDesc logicPipelineDesc = graphicsPipelineDesc;
        logicPipelineDesc.outputMerger.colors = &logicColorDesc;
        logicPipelineDesc.outputMerger.logicOp = nri::LogicOp::XOR;
        TEST_CHECK(CreatePipeline(context, logicPipelineDesc));
    }

    if (context.deviceDesc->tiers.conservativeRaster) {
        nri::GraphicsPipelineDesc conservativePipelineDesc = graphicsPipelineDesc;
        conservativePipelineDesc.rasterization.conservativeRaster = true;
        TEST_CHECK(CreatePipeline(context, conservativePipelineDesc));
    }

    if (context.deviceDesc->features.lineSmoothing) {
        nri::GraphicsPipelineDesc linePipelineDesc = graphicsPipelineDesc;
        linePipelineDesc.inputAssembly.topology = nri::Topology::LINE_LIST;
        linePipelineDesc.rasterization.lineSmoothing = true;
        TEST_CHECK(CreatePipeline(context, linePipelineDesc));
    }

    nri::Pipeline* geometryPipeline = nullptr;
    nri::Pipeline* tessellationPipeline = nullptr;
    nri::Pipeline* combinedPipeline = nullptr;
    if (context.deviceDesc->features.geometryShader) {
        const nri::ShaderDesc geometryShaders[] = {
            shaders[0],
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStates.gs", context.shaderStorage),
            shaders[1],
        };
        nri::GraphicsPipelineDesc geometryPipelineDesc = graphicsPipelineDesc;
        geometryPipelineDesc.shaders = geometryShaders;
        geometryPipelineDesc.shaderNum = 3;
        TEST_CHECK(CreateCachedPipeline(context, geometryPipelineDesc, geometryPipeline));
        geometryPipelineDesc.shaderNum = 2;
        TEST_CHECK(CreatePipeline(context, geometryPipelineDesc));
    }

    if (context.deviceDesc->features.tessellationShader) {
        const nri::ShaderDesc tessellationShaders[] = {
            utils::LoadShader(graphicsAPI, "GraphicsVertexInputTess.vs", context.shaderStorage),
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStatesTess.tcs", context.shaderStorage),
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStatesTess.tes", context.shaderStorage),
            shaders[1],
        };
        nri::GraphicsPipelineDesc tessellationPipelineDesc = graphicsPipelineDesc;
        tessellationPipelineDesc.inputAssembly.topology = nri::Topology::PATCH_LIST;
        tessellationPipelineDesc.inputAssembly.tessControlPointNum = 3;
        tessellationPipelineDesc.shaders = tessellationShaders;
        tessellationPipelineDesc.shaderNum = 4;
        TEST_CHECK(CreateCachedPipeline(context, tessellationPipelineDesc, tessellationPipeline));
        tessellationPipelineDesc.shaderNum = 3;
        TEST_CHECK(CreatePipeline(context, tessellationPipelineDesc));

        if (context.deviceDesc->features.geometryShader) {
            const nri::ShaderDesc combinedShaders[] = {
                tessellationShaders[0],
                tessellationShaders[1],
                tessellationShaders[2],
                utils::LoadShader(graphicsAPI, "GraphicsPipelineStates.gs", context.shaderStorage),
                shaders[1],
            };
            tessellationPipelineDesc.shaders = combinedShaders;
            tessellationPipelineDesc.shaderNum = 5;
            TEST_CHECK(CreateCachedPipeline(context, tessellationPipelineDesc, combinedPipeline));
        }
    }

    nri::TextureDesc colorTextureDesc = {};
    colorTextureDesc.type = nri::TextureType::TEXTURE_2D;
    colorTextureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    colorTextureDesc.format = colorDesc.format;
    colorTextureDesc.width = 64;
    colorTextureDesc.height = 64;
    nri::Texture* colorTexture = nullptr;
    TEST_CHECK(context.CreateTexture(colorTextureDesc, nri::MemoryLocation::DEVICE, colorTexture));

    nri::TextureDesc depthTextureDesc = colorTextureDesc;
    depthTextureDesc.usage = nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT;
    depthTextureDesc.format = depthStencilFormat;
    nri::Texture* depthTexture = nullptr;
    TEST_CHECK(context.CreateTexture(depthTextureDesc, nri::MemoryLocation::DEVICE, depthTexture));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = colorTexture;
    viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
    viewDesc.format = colorTextureDesc.format;
    viewDesc.mipNum = 1;
    viewDesc.layerNum = 1;
    viewDesc.sliceNum = 1;
    nri::Descriptor* colorAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, colorAttachment));
    context.Track(colorAttachment);

    viewDesc.texture = depthTexture;
    viewDesc.type = nri::TextureView::DEPTH_STENCIL_ATTACHMENT;
    viewDesc.format = depthStencilFormat;
    viewDesc.planes = nri::PlaneBits::ALL;
    nri::Descriptor* depthAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, depthAttachment));
    context.Track(depthAttachment);

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = 256 * 64;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));
    nri::BufferDesc argumentDesc = {};
    argumentDesc.size = 256;
    argumentDesc.usage = nri::BufferUsageBits::ARGUMENT | nri::BufferUsageBits::INDEX;
    nri::Buffer* arguments = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentDesc, nri::MemoryLocation::DEVICE, arguments));
    uint8_t argumentData[256] = {};
    const nri::DrawDesc draw = {3, 1, 3, 7};
    const nri::DrawIndexedDesc indexedDraw = {3, 1, 1, -3, 7};
    const uint16_t indices[] = {65535, 65535, 6, 7, 8};
    const uint32_t count = 1;
    memcpy(argumentData + 16, &draw, sizeof(draw));
    memcpy(argumentData + 96, &indexedDraw, sizeof(indexedDraw));
    memcpy(argumentData + 128, &indexedDraw, sizeof(indexedDraw));
    memcpy(argumentData + 192, indices, sizeof(indices));
    memcpy(argumentData + 224, &count, sizeof(count));
    const nri::BufferUploadDesc argumentUpload = {argumentData, arguments, {nri::AccessBits::ARGUMENT_BUFFER | nri::AccessBits::INDEX_BUFFER, nri::StageBits::INDIRECT | nri::StageBits::INDEX_INPUT}};
    TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, &argumentUpload, 1));
    const float vertexData[][4] = {
        {99, 10, 10, 99}, // Buffer prefix, excluded by the bound offset.
        {99, 10, 10, 99},
        {99, 10, 10, 99},
        {99, 10, 10, 99}, // Guard vertices.
        {99, -0.75f, 0.75f, 99},
        {99, 0.75f, 0.75f, 99},
        {99, 0.0f, -0.75f, 99},
    };
    nri::BufferDesc vertexDesc = {};
    vertexDesc.size = sizeof(vertexData);
    vertexDesc.usage = nri::BufferUsageBits::VERTEX;
    nri::Buffer* vertices = nullptr;
    TEST_CHECK(context.CreateBuffer(vertexDesc, nri::MemoryLocation::HOST_UPLOAD, vertices));
    void* vertexMapping = context.core.MapBuffer(*vertices, 0, vertexDesc.size);
    TEST_CHECK(vertexMapping != nullptr);
    memcpy(vertexMapping, vertexData, sizeof(vertexData));
    context.core.UnmapBuffer(*vertices);
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::TextureBarrierDesc textureBarriers[2] = {};
    textureBarriers[0].texture = colorTexture;
    textureBarriers[0].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
    textureBarriers[0].mipNum = 1;
    textureBarriers[0].layerNum = 1;
    textureBarriers[1].texture = depthTexture;
    textureBarriers[1].after = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT};
    textureBarriers[1].mipNum = 1;
    textureBarriers[1].layerNum = 1;
    textureBarriers[1].planes = nri::PlaneBits::ALL;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = textureBarriers;
    barrierDesc.textureNum = 2;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    nri::AttachmentDesc color = {};
    color.descriptor = colorAttachment;
    color.loadOp = nri::LoadOp::CLEAR;
    color.storeOp = nri::StoreOp::STORE;
    nri::AttachmentDesc depthStencil = {};
    depthStencil.descriptor = depthAttachment;
    depthStencil.clearValue.depthStencil = {1.0f, 0};
    depthStencil.loadOp = nri::LoadOp::CLEAR;
    depthStencil.storeOp = nri::StoreOp::STORE;
    nri::RenderingDesc renderingDesc = {};
    renderingDesc.colors = &color;
    renderingDesc.colorNum = 1;
    renderingDesc.depth = depthStencil;
    renderingDesc.stencil = depthStencil;
    context.core.CmdBeginRendering(*commandBuffer, renderingDesc);

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *pipelineLayout);
    context.core.CmdSetPipeline(*commandBuffer, *pipeline);
    const nri::Viewport viewport = {0.0f, 0.0f, 64.0f, 64.0f, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, 64, 64};
    const nri::Color32f blendConstants = {0.25f, 0.5f, 0.75f, 1.0f};
    context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
    context.core.CmdSetStencilReference(*commandBuffer, 1, 2);
    context.core.CmdSetBlendConstants(*commandBuffer, blendConstants);
    if (context.deviceDesc->features.depthBoundsTest)
        context.core.CmdSetDepthBounds(*commandBuffer, 0.0f, 1.0f);
    if (context.deviceDesc->features.dynamicDepthBias) {
        const nri::DepthBiasDesc depthBias = {1.0f, 0.0f, 1.0f};
        context.core.CmdSetDepthBias(*commandBuffer, depthBias);
    }
    nri::Pipeline* drawPipelines[] = {pipeline, geometryPipeline, tessellationPipeline, combinedPipeline};
    for (uint32_t i = 0; i < 4; i++) {
        if (!drawPipelines[i])
            continue;

        context.core.CmdSetPipeline(*commandBuffer, *drawPipelines[i]);
        const nri::VertexBufferDesc vertexBuffer = {vertices, 16, context.deviceDesc->features.extendedDynamicState ? 16u : 0u};
        context.core.CmdSetVertexBuffers(*commandBuffer, 0, &vertexBuffer, 1);
        context.core.CmdSetIndexBuffer(*commandBuffer, *arguments, 194, nri::IndexType::UINT16);
        for (uint32_t mode = 0; mode < 4; mode++) {
            const nri::Viewport drawViewport = {float(i * 16), float(mode * 16), 16.0f, 16.0f, 0.0f, 1.0f};
            context.core.CmdSetViewports(*commandBuffer, &drawViewport, 1);
            if (mode == 0)
                context.core.CmdDraw(*commandBuffer, draw);
            else if (mode == 1)
                context.core.CmdDrawIndexed(*commandBuffer, indexedDraw);
            else if (mode == 2)
                context.core.CmdDrawIndirect(*commandBuffer, *arguments, 16, 2, 32, nullptr, 0);
            else
                context.core.CmdDrawIndexedIndirect(*commandBuffer, *arguments, 96, context.deviceDesc->features.drawIndirectCount ? 2 : 1, 32, context.deviceDesc->features.drawIndirectCount ? arguments : nullptr, 224);
        }
    }
    context.core.CmdEndRendering(*commandBuffer);
    textureBarriers[0].before = textureBarriers[0].after;
    textureBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);
    nri::TextureDataLayoutDesc readbackLayout = {};
    readbackLayout.rowPitch = 256;
    readbackLayout.slicePitch = 256 * 64;
    nri::TextureRegionDesc region = {};
    region.width = 64;
    region.height = 64;
    region.depth = 1;
    context.core.CmdReadbackTextureToBuffer(*commandBuffer, *readback, readbackLayout, *colorTexture, region);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    const uint8_t* pixels = (const uint8_t*)context.core.MapBuffer(*readback, 0, readbackDesc.size);
    if (!pixels)
        return false;

    bool passed = true;
    const char* names[] = {"vertex", "geometry", "tessellation", "tessellation + geometry"};
    for (uint32_t i = 0; i < 4; i++) {
        if (!drawPipelines[i])
            continue;

        for (uint32_t mode = 0; mode < 4; mode++) {
            const uint8_t* center = pixels + (mode * 16 + 8) * 256 + (i * 16 + 8) * 4;
            // Fragment (0.8, 0.3, 0.1) multiplied by blend constants (0.25, 0.5, 0.75).
            const bool matches = abs(int(center[0]) - 51) <= 1 && abs(int(center[1]) - 38) <= 1 && abs(int(center[2]) - 19) <= 1 && center[3] == 255;
            printf("%s mode %u center: %u %u %u %u\n", names[i], mode, center[0], center[1], center[2], center[3]);
            passed &= test::Report(names[i], matches);
        }
    }
    passed &= pixels[0] == 0 && pixels[1] == 0 && pixels[2] == 0 && pixels[3] == 0;
    context.core.UnmapBuffer(*readback);

    return test::Report("graphics vertex input", passed);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
