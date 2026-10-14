// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool CreatePipeline(test::Context& context, const nri::GraphicsPipelineDesc& desc) {
    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, desc, pipeline));
    context.Track(pipeline);

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
        utils::LoadShader(graphicsAPI, "GraphicsPipelineStates.vs", context.shaderStorage),
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

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = pipelineLayout;
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

    if (context.deviceDesc->features.geometryShader) {
        const nri::ShaderDesc geometryShaders[] = {
            shaders[0],
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStates.gs", context.shaderStorage),
            shaders[1],
        };
        nri::GraphicsPipelineDesc geometryPipelineDesc = graphicsPipelineDesc;
        geometryPipelineDesc.shaders = geometryShaders;
        geometryPipelineDesc.shaderNum = 3;
        TEST_CHECK(CreatePipeline(context, geometryPipelineDesc));
    }

    if (context.deviceDesc->features.tessellationShader) {
        const nri::ShaderDesc tessellationShaders[] = {
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStatesTess.vs", context.shaderStorage),
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStatesTess.tcs", context.shaderStorage),
            utils::LoadShader(graphicsAPI, "GraphicsPipelineStatesTess.tes", context.shaderStorage),
            shaders[1],
        };
        nri::GraphicsPipelineDesc tessellationPipelineDesc = graphicsPipelineDesc;
        tessellationPipelineDesc.inputAssembly.topology = nri::Topology::PATCH_LIST;
        tessellationPipelineDesc.inputAssembly.tessControlPointNum = 3;
        tessellationPipelineDesc.shaders = tessellationShaders;
        tessellationPipelineDesc.shaderNum = 4;
        TEST_CHECK(CreatePipeline(context, tessellationPipelineDesc));
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
    context.core.CmdDraw(*commandBuffer, {3, 1, 0, 0});
    context.core.CmdEndRendering(*commandBuffer);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    return test::Report("graphics pipeline states", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
