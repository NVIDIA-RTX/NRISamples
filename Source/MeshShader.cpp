// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include "Extensions/NRIMeshShader.h"

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    if (!context.deviceDesc->features.meshShader) {
        printf("SKIP  Mesh shaders are unsupported\n");

        return true;
    }

    nri::MeshShaderInterface mesh = {};
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::MeshShaderInterface), &mesh));

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.shaderStages = nri::StageBits::MESH_SHADER | nri::StageBits::FRAGMENT_SHADER;
    nri::PipelineLayout* pipelineLayout = nullptr;
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayout));
    context.Track(pipelineLayout);

    const nri::ShaderDesc shaders[] = {
        utils::LoadShader(context.deviceDesc->graphicsAPI, "MeshShader.ms", context.shaderStorage),
        utils::LoadShader(context.deviceDesc->graphicsAPI, "MeshShader.fs", context.shaderStorage),
    };

    nri::ColorAttachmentDesc colorDesc = {};
    colorDesc.format = nri::Format::RGBA8_UNORM;
    colorDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = pipelineLayout;
    graphicsPipelineDesc.inputAssembly.topology = nri::Topology::TRIANGLE_LIST;
    graphicsPipelineDesc.rasterization.fillMode = nri::FillMode::SOLID;
    graphicsPipelineDesc.rasterization.cullMode = nri::CullMode::NONE;
    graphicsPipelineDesc.outputMerger.colors = &colorDesc;
    graphicsPipelineDesc.outputMerger.colorNum = 1;
    graphicsPipelineDesc.shaders = shaders;
    graphicsPipelineDesc.shaderNum = 2;

    nri::Pipeline* pipeline = nullptr;
    TEST_CHECK(context.core.CreateGraphicsPipeline(*context.device, graphicsPipelineDesc, pipeline));
    context.Track(pipeline);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.format = colorDesc.format;
    textureDesc.width = 32;
    textureDesc.height = 32;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureViewDesc viewDesc = {};
    viewDesc.texture = texture;
    viewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
    viewDesc.format = textureDesc.format;
    viewDesc.mipNum = 1;
    viewDesc.layerNum = 1;
    viewDesc.sliceNum = 1;
    nri::Descriptor* colorAttachment = nullptr;
    TEST_CHECK(context.core.CreateTextureView(viewDesc, colorAttachment));
    context.Track(colorAttachment);

    const nri::DrawMeshTasksDesc arguments = {1, 1, 1};
    nri::BufferDesc argumentBufferDesc = {};
    argumentBufferDesc.size = sizeof(arguments);
    argumentBufferDesc.usage = nri::BufferUsageBits::ARGUMENT;
    nri::Buffer* argumentBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(argumentBufferDesc, nri::MemoryLocation::DEVICE, argumentBuffer));

    const nri::BufferUploadDesc uploadDesc = {&arguments, argumentBuffer, {nri::AccessBits::ARGUMENT_BUFFER, nri::StageBits::INDIRECT}};
    TEST_CHECK(context.helper.UploadData(*queue, nullptr, 0, &uploadDesc, 1));

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
    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *pipelineLayout);
    context.core.CmdSetPipeline(*commandBuffer, *pipeline);
    const nri::Viewport viewport = {0.0f, 0.0f, 32.0f, 32.0f, 0.0f, 1.0f};
    const nri::Rect scissor = {0, 0, 32, 32};
    context.core.CmdSetViewports(*commandBuffer, &viewport, 1);
    context.core.CmdSetScissors(*commandBuffer, &scissor, 1);
    mesh.CmdDrawMeshTasks(*commandBuffer, arguments);
    mesh.CmdDrawMeshTasksIndirect(*commandBuffer, *argumentBuffer, 0, 1, sizeof(arguments), nullptr, 0);
    context.core.CmdEndRendering(*commandBuffer);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    return test::Report("mesh shader direct and indirect draws", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
