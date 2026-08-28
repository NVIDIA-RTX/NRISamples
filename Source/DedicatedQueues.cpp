// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include <array>

namespace {

bool Run(const test::Settings& settings) {
    const nri::QueueFamilyDesc queueFamilies[] = {
        {nullptr, 1, nri::QueueType::GRAPHICS},
        {nullptr, 1, nri::QueueType::COPY},
    };

    test::Context context;
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
    nri::Buffer* timestampReadback = nullptr;
    if (context.deviceDesc->features.timestampCopyQueue) {
        nri::QueryPoolDesc queryPoolDesc = {nri::QueryType::TIMESTAMP_COPY_QUEUE, 1};
        TEST_CHECK(context.core.CreateQueryPool(*context.device, queryPoolDesc, timestampQueryPool));
        context.Track(timestampQueryPool);

        const uint32_t querySize = context.core.GetQuerySize(*timestampQueryPool);
        nri::BufferDesc queryBufferDesc = {};
        queryBufferDesc.size = querySize;
        TEST_CHECK(context.CreateBuffer(queryBufferDesc, nri::MemoryLocation::HOST_READBACK, timestampReadback));

        context.core.CmdResetQueries(*copyCommandBuffer, *timestampQueryPool, 0, 1);
        context.core.CmdEndQuery(*copyCommandBuffer, *timestampQueryPool, 0);
        context.core.CmdCopyQueries(*copyCommandBuffer, *timestampQueryPool, 0, 1, *timestampReadback, 0);
    }

    TEST_CHECK(context.core.EndCommandBuffer(*copyCommandBuffer));

    nri::Fence* fence = nullptr;
    TEST_CHECK(context.core.CreateFence(*context.device, 0, fence));
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

    if (timestampReadback) {
        const uint64_t* timestamp = (const uint64_t*)context.core.MapBuffer(*timestampReadback, 0, sizeof(uint64_t));
        passed &= timestamp != nullptr;
        context.core.UnmapBuffer(*timestampReadback);
    } else
        printf("SKIP  Copy-queue timestamps are unsupported\n");

    TEST_CHECK(context.core.QueueWaitIdle(graphicsQueue));

    return test::Report("dedicated copy queue", passed);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}

