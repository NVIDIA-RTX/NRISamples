// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include <array>

namespace {

static inline bool SubmitTextureCommands(test::Context& context, nri::Queue& queue, nri::CommandBuffer& commandBuffer, nri::Fence& fence, uint64_t value) {
    TEST_CHECK(context.core.EndCommandBuffer(commandBuffer));

    nri::FenceSubmitDesc waitFence = {&fence, value - 1, nri::StageBits::ALL};
    nri::FenceSubmitDesc signalFence = {&fence, value, nri::StageBits::ALL};
    nri::CommandBuffer* commandBufferPtr = &commandBuffer;
    nri::QueueSubmitDesc submitDesc = {};
    submitDesc.commandBuffers = &commandBufferPtr;
    submitDesc.commandBufferNum = 1;
    submitDesc.waitFences = &waitFence;
    submitDesc.waitFenceNum = value > 1 ? 1 : 0;
    submitDesc.signalFences = &signalFence;
    submitDesc.signalFenceNum = 1;
    TEST_CHECK(context.core.QueueSubmit(queue, submitDesc));

    return true;
}

static inline bool TestTextureQueues(test::Context& context, nri::Queue& graphicsQueue, nri::Queue& copyQueue, nri::SharingMode sharingMode) {
    constexpr uint32_t width = 16;
    constexpr uint32_t height = 16;
    constexpr uint32_t rowSize = width * 4;
    const uint32_t rowAlignment = std::max(context.deviceDesc->memoryAlignment.uploadBufferTextureRow, 1u);
    const uint32_t sliceAlignment = std::max(context.deviceDesc->memoryAlignment.uploadBufferTextureSlice, 1u);
    nri::TextureDataLayoutDesc dataLayout = {};
    dataLayout.rowPitch = (rowSize + rowAlignment - 1) / rowAlignment * rowAlignment;
    dataLayout.slicePitch = (dataLayout.rowPitch * height + sliceAlignment - 1) / sliceAlignment * sliceAlignment;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = uint64_t(dataLayout.slicePitch) * 2;
    nri::Buffer* uploadBuffer = nullptr;
    nri::Buffer* readbackBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer));
    bufferDesc.size = uint64_t(dataLayout.slicePitch) * 3;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readbackBuffer));

    uint8_t* uploadData = (uint8_t*)context.core.MapBuffer(*uploadBuffer, 0, nri::WHOLE_SIZE);
    TEST_CHECK(uploadData);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < rowSize; x++) {
            const uint32_t i = y * rowSize + x;
            uploadData[y * dataLayout.rowPitch + x] = uint8_t(i * 37 + 13);
            uploadData[dataLayout.slicePitch + y * dataLayout.rowPitch + x] = uint8_t(i * 23 + 71);
        }
    }
    context.core.UnmapBuffer(*uploadBuffer);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
    if (sharingMode == nri::SharingMode::EXCLUSIVE)
        textureDesc.usage |= nri::TextureUsageBits::COLOR_ATTACHMENT;
    textureDesc.sharingMode = sharingMode;
    nri::Texture* texture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

    nri::TextureRegionDesc region = {};
    region.width = width;
    region.height = height;
    region.depth = 1;
    region.planes = nri::PlaneBits::COLOR;

    nri::Queue* queues[] = {&copyQueue, &graphicsQueue, &copyQueue, &graphicsQueue, &copyQueue};
    nri::CommandBuffer* commands[5] = {};
    for (uint32_t i = 0; i < 5; i++) {
        nri::CommandAllocator* allocator = nullptr;
        TEST_CHECK(context.CreateCommandObjects(*queues[i], allocator, commands[i]));
        TEST_CHECK(context.core.BeginCommandBuffer(*commands[i], nullptr));
    }

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
    textureBarrier.mipNum = 1;
    textureBarrier.layerNum = 1;
    textureBarrier.planes = nri::PlaneBits::COLOR;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;
    auto barrier = [&](uint32_t i, const nri::AccessLayoutStage& before, const nri::AccessLayoutStage& after, nri::Queue* srcQueue = nullptr, nri::Queue* dstQueue = nullptr) {
        textureBarrier.before = before;
        textureBarrier.after = after;
        textureBarrier.srcQueue = srcQueue;
        textureBarrier.dstQueue = dstQueue;
        context.core.CmdBarrier(*commands[i], barrierDesc);
    };

    const nri::AccessLayoutStage undefined = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::NONE};
    const nri::AccessLayoutStage copyDst = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
    const nri::AccessLayoutStage copySrc = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    const nri::AccessLayoutStage shaderRead = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::FRAGMENT_SHADER};
    const nri::AccessLayoutStage firstHandoff = sharingMode == nri::SharingMode::EXCLUSIVE ? copyDst : copySrc;
    nri::Fence* fence = nullptr;
    TEST_CHECK(context.core.CreateFence(*context.device, {}, fence));
    context.Track(fence);

    // COPY first use and COPY_DESTINATION hand-off to GRAPHICS.
    barrier(0, undefined, copyDst);
    context.core.CmdUploadBufferToTexture(*commands[0], *texture, region, *uploadBuffer, dataLayout);
    // Concurrent images transition on the source queue; the destination does not repeat the transition.
    barrier(0, copyDst, firstHandoff, &copyQueue, &graphicsQueue);
    TEST_CHECK(SubmitTextureCommands(context, copyQueue, *commands[0], *fence, 1));

    // Each exclusive Vulkan hand-off repeats the same layouts on release and acquire.
    // Graphics-only transitions are separate, so no shader layout or stage reaches a COPY list.
    barrier(1, firstHandoff, firstHandoff, &copyQueue, &graphicsQueue);
    barrier(1, firstHandoff, copySrc);
    context.core.CmdReadbackTextureToBuffer(*commands[1], *readbackBuffer, dataLayout, *texture, region);
    barrier(1, copySrc, shaderRead);
    barrier(1, shaderRead, copyDst);
    barrier(1, copyDst, copyDst, &graphicsQueue, &copyQueue);
    TEST_CHECK(SubmitTextureCommands(context, graphicsQueue, *commands[1], *fence, 2));

    // GRAPHICS -> COPY with COPY_DESTINATION, followed by a native copy-queue access transition.
    barrier(2, copyDst, copyDst, &graphicsQueue, &copyQueue);
    dataLayout.offset = dataLayout.slicePitch;
    context.core.CmdUploadBufferToTexture(*commands[2], *texture, region, *uploadBuffer, dataLayout);
    barrier(2, copyDst, copySrc);
    barrier(2, copySrc, copySrc, &copyQueue, &graphicsQueue);
    TEST_CHECK(SubmitTextureCommands(context, copyQueue, *commands[2], *fence, 3));

    // COPY -> GRAPHICS with COPY_SOURCE; use explicit graphics layouts before handing it back.
    barrier(3, copySrc, copySrc, &copyQueue, &graphicsQueue);
    barrier(3, copySrc, shaderRead);
    barrier(3, shaderRead, copySrc);
    context.core.CmdReadbackTextureToBuffer(*commands[3], *readbackBuffer, dataLayout, *texture, region);
    barrier(3, copySrc, copySrc, &graphicsQueue, &copyQueue);
    TEST_CHECK(SubmitTextureCommands(context, graphicsQueue, *commands[3], *fence, 4));

    // GRAPHICS -> COPY with COPY_SOURCE, followed by readback on COPY.
    barrier(4, copySrc, copySrc, &graphicsQueue, &copyQueue);
    dataLayout.offset = uint64_t(dataLayout.slicePitch) * 2;
    context.core.CmdReadbackTextureToBuffer(*commands[4], *readbackBuffer, dataLayout, *texture, region);
    TEST_CHECK(SubmitTextureCommands(context, copyQueue, *commands[4], *fence, 5));
    context.core.Wait(*fence, 5);
    TEST_CHECK(context.core.GetFenceValue(*fence) >= 5);

    const uint8_t* readbackData = (const uint8_t*)context.core.MapBuffer(*readbackBuffer, 0, nri::WHOLE_SIZE);
    TEST_CHECK(readbackData);
    bool passed = true;
    for (uint32_t slot = 0; slot < 3; slot++) {
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < rowSize; x++) {
                const uint32_t i = y * rowSize + x;
                const uint8_t expected = slot == 0 ? uint8_t(i * 37 + 13) : uint8_t(i * 23 + 71);
                passed &= readbackData[slot * dataLayout.slicePitch + y * dataLayout.rowPitch + x] == expected;
            }
        }
    }
    context.core.UnmapBuffer(*readbackBuffer);

    return test::Report(sharingMode == nri::SharingMode::EXCLUSIVE ? "exclusive texture copy-queue hand-offs" : "concurrent texture copy-queue hand-offs", passed);
}

bool Run(test::Context& context, const test::Settings& settings) {
    const nri::QueueFamilyDesc queueFamilies[] = {
        {nullptr, 1, nri::QueueType::GRAPHICS},
        {nullptr, 1, nri::QueueType::COPY},
    };

    if (!context.Initialize(settings, queueFamilies, 2) || context.skipped)
        return context.skipped;

    nri::Queue* graphicsQueue = nullptr;
    nri::Queue* copyQueue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, graphicsQueue));
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::COPY, 0, copyQueue));

    constexpr uint32_t dataSize = 4096;
    std::array<uint8_t, dataSize> expected = {};
    for (uint32_t i = 0; i < dataSize; i++)
        expected[i] = uint8_t(i * 37 + 13);

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = dataSize;

    nri::Buffer* uploadBuffer = nullptr;
    nri::Buffer* deviceBuffer = nullptr;
    nri::Buffer* readbackBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, deviceBuffer));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readbackBuffer));

    void* uploadData = context.core.MapBuffer(*uploadBuffer, 0, dataSize);
    memcpy(uploadData, expected.data(), dataSize);
    context.core.UnmapBuffer(*uploadBuffer);

    nri::CommandAllocator* copyCommandAllocator = nullptr;
    nri::CommandBuffer* copyCommandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*copyQueue, copyCommandAllocator, copyCommandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*copyCommandBuffer, nullptr));
    context.core.CmdCopyBuffer(*copyCommandBuffer, *deviceBuffer, 0, *uploadBuffer, 0, dataSize);

    nri::BufferBarrierDesc bufferBarrier = {};
    bufferBarrier.buffer = deviceBuffer;
    bufferBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.buffers = &bufferBarrier;
    barrierDesc.bufferNum = 1;
    context.core.CmdBarrier(*copyCommandBuffer, barrierDesc);
    context.core.CmdCopyBuffer(*copyCommandBuffer, *readbackBuffer, 0, *deviceBuffer, 0, dataSize);

    nri::QueryPool* timestampQueryPool = nullptr;
    if (context.deviceDesc->features.timestampCopyQueue) {
        nri::QueryPoolDesc queryPoolDesc = {nri::QueryType::TIMESTAMP_COPY_QUEUE, 1};
        TEST_CHECK(context.core.CreateQueryPool(*context.device, queryPoolDesc, timestampQueryPool));
        context.Track(timestampQueryPool);

        context.core.ResetQueries(*timestampQueryPool, 0, 1);
        context.core.CmdEndQuery(*copyCommandBuffer, *timestampQueryPool, 0);
    } else
        printf("SKIP  Copy-queue timestamps are unsupported\n");

    TEST_CHECK(context.core.EndCommandBuffer(*copyCommandBuffer));

    nri::Fence* fence = nullptr;
    TEST_CHECK(context.core.CreateFence(*context.device, {}, fence));
    context.Track(fence);

    nri::FenceSubmitDesc signalFence = {};
    signalFence.fence = fence;
    signalFence.value = 1;

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &copyCommandBuffer;
    queueSubmitDesc.commandBufferNum = 1;
    queueSubmitDesc.signalFences = &signalFence;
    queueSubmitDesc.signalFenceNum = 1;
    TEST_CHECK(context.core.QueueSubmit(*copyQueue, queueSubmitDesc));
    context.core.Wait(*fence, 1);

    if (context.core.GetFenceValue(*fence) < 1) {
        printf("FAIL  Fence value did not advance\n");

        return false;
    }

    bool passed = test::VerifyBytes(context.core, *readbackBuffer, expected.data(), dataSize);

    if (timestampQueryPool) {
        nri::BufferDesc timestampBufferDesc = {};
        timestampBufferDesc.size = sizeof(uint64_t);
        nri::Buffer* timestampReadbackBuffer = nullptr;
        TEST_CHECK(context.CreateBuffer(timestampBufferDesc, nri::MemoryLocation::HOST_READBACK, timestampReadbackBuffer));

        nri::Queue& resolveQueue = context.deviceDesc->other.timestampCopyQueueResolveOnCopyQueue ? *copyQueue : *graphicsQueue;
        nri::CommandAllocator* resolveCommandAllocator = nullptr;
        nri::CommandBuffer* resolveCommandBuffer = nullptr;
        TEST_CHECK(context.CreateCommandObjects(resolveQueue, resolveCommandAllocator, resolveCommandBuffer));
        TEST_CHECK(context.core.BeginCommandBuffer(*resolveCommandBuffer, nullptr));
        context.core.CmdCopyQueries(*resolveCommandBuffer, *timestampQueryPool, 0, 1, *timestampReadbackBuffer, 0);
        TEST_CHECK(context.SubmitAndWait(resolveQueue, *resolveCommandBuffer));

        const uint64_t* timestampData = (const uint64_t*)context.core.MapBuffer(*timestampReadbackBuffer, 0, sizeof(uint64_t));
        bool timestampPassed = timestampData && *timestampData != 0;
        if (timestampData)
            context.core.UnmapBuffer(*timestampReadbackBuffer);
        passed &= test::Report("copy queue timestamp", timestampPassed);
    }

    TEST_CHECK(context.core.QueueWaitIdle(graphicsQueue));

    passed &= TestTextureQueues(context, *graphicsQueue, *copyQueue, nri::SharingMode::EXCLUSIVE);
    passed &= TestTextureQueues(context, *graphicsQueue, *copyQueue, nri::SharingMode::CONCURRENT);

    return test::Report("dedicated copy queue", passed);
}

} // namespace

int main(int argc, char** argv) {
    return test::RunSample(argc, argv, Run);
}

