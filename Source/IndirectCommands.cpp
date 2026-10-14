// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include <array>

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::PipelineLayoutDesc graphicsLayoutDesc = {};
    graphicsLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* graphicsLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, graphicsLayoutDesc, graphicsLayout));
    context.Track(graphicsLayout);

    nri::PipelineLayoutDesc computeLayoutDesc = {};
    computeLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    nri::PipelineLayout* computeLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, computeLayoutDesc, computeLayout));
    context.Track(computeLayout);

    const nri::GraphicsAPI graphicsAPI = context.deviceDesc->graphicsAPI;
    const nri::ShaderDesc graphicsShaders[] = {
        utils::LoadShader(graphicsAPI, "IndirectCommands.vs", context.shaderStorage),
        utils::LoadShader(graphicsAPI, "IndirectCommands.fs", context.shaderStorage),
    };

    nri::ColorAttachmentDesc colorAttachmentDesc = {};
    colorAttachmentDesc.format = nri::Format::RGBA8_UNORM;
    colorAttachmentDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = graphicsLayout;
    graphicsPipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    graphicsPipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    graphicsPipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    graphicsPipelineDesc.outputMerger.colors = &colorAttachmentDesc;
    graphicsPipelineDesc.outputMerger.colorNum = 1;
    graphicsPipelineDesc.shaders = graphicsShaders;
    graphicsPipelineDesc.shaderNum = 2;

    nri::Pipeline* graphicsPipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, graphicsPipeline));
    context.Track(graphicsPipeline);

    nri::ComputePipelineDesc computePipelineDesc = {};
    computePipelineDesc.pipelineLayout = computeLayout;
    computePipelineDesc.shader = utils::LoadShader(graphicsAPI, "IndirectCommands.cs", context.shaderStorage);
    nri::Pipeline* computePipeline = nullptr;
    TEST_CHECK(context.core.CreateComputePipeline(*context.device, computePipelineDesc, computePipeline));
    context.Track(computePipeline);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.format = colorAttachmentDesc.format;
    textureDesc.width = 32;
    textureDesc.height = 32;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureViewDesc textureViewDesc = {};
    textureViewDesc.texture = texture;
    textureViewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
    textureViewDesc.format = textureDesc.format;
    textureViewDesc.mipNum = 1;
    textureViewDesc.layerNum = 1;
    textureViewDesc.sliceNum = 1;
    nri::Descriptor* colorAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(textureViewDesc, colorAttachment));
    context.Track(colorAttachment);

    struct Arguments {
        nri::DrawDesc draw;
        uint8_t padding0[64 - sizeof(nri::DrawDesc)];
        nri::DrawIndexedDesc drawIndexed;
        uint8_t padding1[64 - sizeof(nri::DrawIndexedDesc)];
        nri::DispatchDesc dispatch;
    } arguments = {};
    arguments.draw = {3, 1, 0, 0};
    arguments.drawIndexed = {3, 1, 0, 0, 0};
    arguments.dispatch = {1, 1, 1};

    nri::BufferDesc argumentBufferDesc = {};
    argumentBufferDesc.size = sizeof(arguments);
    argumentBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
    nri::Buffer* argumentBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentBufferDesc, nri::MemoryLocation::DEVICE, argumentBuffer));

    const uint16_t indices[] = {0, 1, 2, 0}; // WGPU: source offset, destination offset, and copy size must be multiples of 4
    nri::BufferDesc indexBufferDesc = {};
    indexBufferDesc.size = sizeof(indices);
    indexBufferDesc.usage = nri::BufferUsageBits::INDEX;
    nri::Buffer* indexBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(indexBufferDesc, nri::MemoryLocation::DEVICE, indexBuffer));

    const uint32_t indirectCount = 1;
    nri::Buffer* countBuffer = nullptr;
    if (context.deviceDesc->features.drawIndirectCount) {
        nri::BufferDesc countBufferDesc = {};
        countBufferDesc.size = sizeof(indirectCount);
        countBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
        TEST_CHECK(context.CreateBuffer(countBufferDesc, nri::MemoryLocation::DEVICE, countBuffer));
    }

    const nri::BufferUploadDesc uploads[] = {
        {&arguments, argumentBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}},
        {indices, indexBuffer, {nri::AccessBits::INDEX_BUFFER, nri::StageBits::INDEX_INPUT}},
    };
    TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, uploads, 2));
    if (countBuffer) {
        const nri::BufferUploadDesc countUpload = {&indirectCount, countBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}};
        TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, &countUpload, 1));
    }

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
    textureBarrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
    textureBarrier.mipNum = 1;
    textureBarrier.layerNum = 1;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    nri::AttachmentDesc attachmentDesc = {};
    attachmentDesc.descriptor = colorAttachment;
    attachmentDesc.loadOp = nri::LoadOp::CLEAR;
    attachmentDesc.storeOp = nri::StoreOp::STORE;
    nri::RenderingDesc renderingDesc = {};
    renderingDesc.colors = &attachmentDesc;
    renderingDesc.colorNum = 1;
    context.core.CmdBeginRendering(*commandBuffer, renderingDesc);

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *graphicsLayout);
    context.core.CmdSetPipeline(*commandBuffer, *graphicsPipeline);
    const nri::Viewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, 32, 32};
    context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
    context.core.CmdSetIndexBuffer(*commandBuffer, *indexBuffer, 0, nri::IndexType::UINT16);
    context.core.CmdDrawIndirect(*commandBuffer, *argumentBuffer, 0, 1, sizeof(nri::DrawDesc), nullptr, 0);
    context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, 64, 1, sizeof(nri::DrawIndexedDesc), nullptr, 0);
    if (context.deviceDesc->features.drawIndirectCount)
        context.core.CmdDrawIndexedIndirect(*commandBuffer, *argumentBuffer, 64, 1, sizeof(nri::DrawIndexedDesc), countBuffer, 0);
    context.core.CmdEndRendering(*commandBuffer);

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *computeLayout);
    context.core.CmdSetPipeline(*commandBuffer, *computePipeline);
    context.core.CmdDispatchIndirect(*commandBuffer, *argumentBuffer, 128);

    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    return test::Report("indirect draw and dispatch", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
