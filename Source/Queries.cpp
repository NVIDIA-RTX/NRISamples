// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));

    nri::QueryPool* timestampPool = nullptr;
    nri::QueryPool* occlusionPool = nullptr;
    nri::Buffer* timestampReadback = nullptr;
    nri::Buffer* occlusionReadback = nullptr;

    if (context.deviceDesc->features.timestamp) {
        const nri::QueryPoolDesc queryPoolDesc = {nri::QueryType::TIMESTAMP, 2};
        TEST_CHECK(context.core.CreateQueryPool(*context.device, queryPoolDesc, timestampPool));
        context.Track(timestampPool);

        const uint32_t querySize = context.core.GetQuerySize(*timestampPool);
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = querySize * 2;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, timestampReadback));

        context.core.ResetQueries(*timestampPool, 0, 2);
    }

    nri::Texture* colorTexture = nullptr;
    nri::Descriptor* colorAttachment = nullptr;
    if (context.deviceDesc->features.occlusion) {
        const nri::QueryPoolDesc queryPoolDesc = {nri::QueryType::OCCLUSION, 1};
        TEST_CHECK(context.core.CreateQueryPool(*context.device, queryPoolDesc, occlusionPool));
        context.Track(occlusionPool);

        const uint32_t querySize = context.core.GetQuerySize(*occlusionPool);
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = querySize;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, occlusionReadback));

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
        textureDesc.format = nri::Format::RGBA8_UNORM;
        textureDesc.width = 16;
        textureDesc.height = 16;
        TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, colorTexture));

        nri::TextureViewDesc textureViewDesc = {};
        textureViewDesc.texture = colorTexture;
        textureViewDesc.type = nri::TextureView::COLOR_ATTACHMENT;
        textureViewDesc.format = textureDesc.format;
        textureViewDesc.mipNum = 1;
        textureViewDesc.layerNum = 1;
        textureViewDesc.sliceNum = 1;
        TEST_CHECK(context.core.CreateTextureView(textureViewDesc, colorAttachment));
        context.Track(colorAttachment);
    }

    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    if (timestampPool) {
        context.core.CmdResetQueries(*commandBuffer, *timestampPool, 0, 2);
        context.core.CmdEndQuery(*commandBuffer, *timestampPool, 0);
    }

    if (occlusionPool) {
        context.core.CmdResetQueries(*commandBuffer, *occlusionPool, 0, 1);

        nri::TextureBarrierDesc textureBarrier = {};
        textureBarrier.texture = colorTexture;
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
        context.core.CmdBeginQuery(*commandBuffer, *occlusionPool, 0);
        context.core.CmdEndQuery(*commandBuffer, *occlusionPool, 0);
        context.core.CmdEndRendering(*commandBuffer);
        context.core.CmdCopyQueries(*commandBuffer, *occlusionPool, 0, 1, *occlusionReadback, 0);
    }

    if (timestampPool) {
        context.core.CmdEndQuery(*commandBuffer, *timestampPool, 1);
        context.core.CmdCopyQueries(*commandBuffer, *timestampPool, 0, 2, *timestampReadback, 0);
    }

    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    bool passed = true;
    if (timestampReadback) {
        const uint64_t* timestamps = (const uint64_t*)context.core.MapBuffer(*timestampReadback, 0, nri::WHOLE_SIZE);
        passed &= timestamps && timestamps[1] >= timestamps[0];
        context.core.UnmapBuffer(*timestampReadback);
    } else
        printf("SKIP  Graphics timestamps are unsupported\n");

    if (occlusionReadback) {
        const uint64_t* samples = (const uint64_t*)context.core.MapBuffer(*occlusionReadback, 0, nri::WHOLE_SIZE);
        passed &= samples && samples[0] == 0;
        context.core.UnmapBuffer(*occlusionReadback);
    } else
        printf("SKIP  Occlusion queries are unsupported\n");

    if (context.deviceDesc->features.calibratedTimestamps) {
        uint64_t timestampGPU = 0;
        uint64_t timestampCPU = 0;
        context.core.GetCalibratedTimestamps(*queue, timestampGPU, timestampCPU);
        passed &= timestampGPU != 0 && timestampCPU != 0;
    } else
        printf("SKIP  Calibrated timestamps are unsupported\n");

    return test::Report("queries", passed);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
