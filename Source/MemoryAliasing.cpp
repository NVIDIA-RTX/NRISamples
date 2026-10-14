// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include <array>

namespace {

bool RecordCopy(test::Context& context, nri::Queue& queue, nri::CommandAllocator& commandAllocator, nri::CommandBuffer& commandBuffer, nri::Buffer& upload, nri::Buffer& aliased, nri::Buffer& readback) {
    context.core.ResetCommandAllocator(commandAllocator);
    TEST_CHECK(context.core.BeginCommandBuffer(commandBuffer, nullptr));
    context.core.CmdCopyBuffer(commandBuffer, aliased, 0, upload, 0, nri::WHOLE_SIZE);

    nri::GlobalBarrierDesc aliasingBarrier = {};
    aliasingBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    aliasingBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.globals = &aliasingBarrier;
    barrierDesc.globalNum = 1;
    context.core.CmdBarrier(commandBuffer, barrierDesc);
    context.core.CmdCopyBuffer(commandBuffer, readback, 0, aliased, 0, nri::WHOLE_SIZE);
    TEST_CHECK(context.SubmitAndWait(queue, commandBuffer));

    return true;
}

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    if (!context.deviceDesc->features.resourceAliasing) {
        printf("SKIP  Resource aliasing is unsupported\n");

        return true;
    }

    constexpr uint32_t dataSize = 4096;
    std::array<uint8_t, dataSize> dataA = {};
    std::array<uint8_t, dataSize> dataB = {};
    for (uint32_t i = 0; i < dataSize; i++) {
        dataA[i] = uint8_t(i * 7 + 3);
        dataB[i] = uint8_t(i * 23 + 17);
    }

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = dataSize;

    nri::Buffer* aliasedA = nullptr;
    nri::Buffer* aliasedB = nullptr;
    TEST_CHECK(context.core.CreateBuffer(*context.device, bufferDesc, aliasedA));
    TEST_CHECK(context.core.CreateBuffer(*context.device, bufferDesc, aliasedB));
    context.Track(aliasedA);
    context.Track(aliasedB);

    nri::MemoryDesc memoryDescA = {};
    nri::MemoryDesc memoryDescB = {};
    context.core.GetBufferMemoryDesc(*aliasedA, nri::MemoryLocation::DEVICE, memoryDescA);
    context.core.GetBufferMemoryDesc(*aliasedB, nri::MemoryLocation::DEVICE, memoryDescB);
    if (memoryDescA.type != memoryDescB.type) {
        printf("FAIL  Identical buffers returned incompatible memory types\n");

        return false;
    }

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = std::max(memoryDescA.size, memoryDescB.size);
    allocateMemoryDesc.type = memoryDescA.type;
    allocateMemoryDesc.priority = 0.5f;

    nri::Memory* memory = nullptr;
    TEST_CHECK(context.core.AllocateMemory(*context.device, allocateMemoryDesc, memory));
    context.Track(memory);

    const nri::BindBufferMemoryDesc bindings[] = {
        {aliasedA, memory, 0},
        {aliasedB, memory, 0},
    };
    TEST_CHECK(context.core.BindBufferMemory(bindings, 2));

    nri::Buffer* uploadA = nullptr;
    nri::Buffer* uploadB = nullptr;
    nri::Buffer* readbackA = nullptr;
    nri::Buffer* readbackB = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, uploadA));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, uploadB));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readbackA));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readbackB));

    void* mapped = context.core.MapBuffer(*uploadA, 0, dataSize);
    memcpy(mapped, dataA.data(), dataSize);
    context.core.UnmapBuffer(*uploadA);
    mapped = context.core.MapBuffer(*uploadB, 0, dataSize);
    memcpy(mapped, dataB.data(), dataSize);
    context.core.UnmapBuffer(*uploadB);

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));
    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));

    TEST_CHECK(RecordCopy(context, *queue, *commandAllocator, *commandBuffer, *uploadA, *aliasedA, *readbackA));
    bool passed = test::VerifyBytes(context.core, *readbackA, dataA.data(), dataSize);

    TEST_CHECK(RecordCopy(context, *queue, *commandAllocator, *commandBuffer, *uploadB, *aliasedB, *readbackB));
    passed &= test::VerifyBytes(context.core, *readbackB, dataB.data(), dataSize);

    return test::Report("overlapping placed resources", passed);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
