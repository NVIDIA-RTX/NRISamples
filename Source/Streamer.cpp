// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include "Extensions/NRIStreamer.h"

#include <array>

namespace {

struct StreamerGuard {
    ~StreamerGuard() {
        if (streamer)
            interface.DestroyStreamer(streamer);
    }

    nri::StreamerInterface interface = {};
    nri::Streamer* streamer = nullptr;
};

uint32_t Align(uint32_t value, uint32_t alignment) {
    alignment = std::max(alignment, 1u);

    return (value + alignment - 1) / alignment * alignment;
}

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    StreamerGuard streamerGuard;
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::StreamerInterface), &streamerGuard.interface));

    nri::StreamerDesc streamerDesc = {};
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.constantBufferSize = 4096;
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferDesc.usage = nri::BufferUsageBits::VERTEX | nri::BufferUsageBits::INDEX;
    streamerDesc.queuedFrameNum = 2;
    TEST_CHECK(streamerGuard.interface.CreateStreamer(*context.device, streamerDesc, streamerGuard.streamer));

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    constexpr uint32_t bufferSize = 512;
    std::array<uint8_t, bufferSize> expected = {};
    for (uint32_t i = 0; i < bufferSize; i++)
        expected[i] = uint8_t(i * 19 + 5);

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = bufferSize;
    nri::Buffer* destinationBuffer = nullptr;
    nri::Buffer* readbackBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::DEVICE, destinationBuffer));
    TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_READBACK, readbackBuffer));

    const nri::DataSize chunks[] = {
        {expected.data(), 173},
        {expected.data() + 173, bufferSize - 173},
    };
    nri::StreamBufferDataDesc streamBufferDesc = {};
    streamBufferDesc.dataChunks = chunks;
    streamBufferDesc.dataChunkNum = 2;
    streamBufferDesc.placementAlignment = 256;
    streamBufferDesc.dstBuffer = destinationBuffer;

    const nri::BufferOffset streamedBuffer = streamerGuard.interface.StreamBufferData(*streamerGuard.streamer, streamBufferDesc);
    if (!streamedBuffer.buffer || (streamedBuffer.offset % 256)) {
        printf("FAIL  StreamBufferData returned an invalid placement\n");

        return false;
    }

    streamBufferDesc.dstBuffer = nullptr;
    const nri::BufferOffset directBuffer = streamerGuard.interface.StreamBufferData(*streamerGuard.streamer, streamBufferDesc);
    if (!directBuffer.buffer || (directBuffer.offset % 256)) {
        printf("FAIL  Destination-less StreamBufferData failed\n");

        return false;
    }

    constexpr uint32_t textureWidth = 4;
    constexpr uint32_t textureHeight = 4;
    std::array<uint8_t, textureWidth * textureHeight * 4> textureData = {};
    for (uint32_t i = 0; i < textureData.size(); i++)
        textureData[i] = uint8_t(i * 11 + 7);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.width = textureWidth;
    textureDesc.height = textureHeight;
    nri::Texture* destinationTexture = nullptr;
    TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, destinationTexture));

    nri::StreamTextureDataDesc streamTextureDesc = {};
    streamTextureDesc.data = textureData.data();
    streamTextureDesc.dataRowPitch = textureWidth * 4;
    streamTextureDesc.dataSlicePitch = (uint32_t)textureData.size();
    streamTextureDesc.dstTexture = destinationTexture;
    streamTextureDesc.dstRegion = {0, 0, 0, textureWidth, textureHeight, 1, 0, 0, nri::PlaneBits::ALL};
    const nri::BufferOffset streamedTexture = streamerGuard.interface.StreamTextureData(*streamerGuard.streamer, streamTextureDesc);
    if (!streamedTexture.buffer) {
        printf("FAIL  StreamTextureData failed\n");

        return false;
    }

    const uint32_t rowPitch = Align(textureWidth * 4, context.deviceDesc->memoryAlignment.uploadBufferTextureRow);
    const uint32_t slicePitch = Align(rowPitch * textureHeight, context.deviceDesc->memoryAlignment.uploadBufferTextureSlice);
    nri::BufferDesc textureReadbackDesc = {};
    textureReadbackDesc.size = slicePitch;
    nri::Buffer* textureReadback = nullptr;
    TEST_CHECK(context.CreateBuffer(textureReadbackDesc, nri::MemoryLocation::HOST_READBACK, textureReadback));

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = destinationTexture;
    textureBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
    textureBarrier.mipNum = 1;
    textureBarrier.layerNum = 1;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    streamerGuard.interface.CmdCopyStreamedData(*commandBuffer, *streamerGuard.streamer);

    nri::BufferBarrierDesc bufferBarrier = {};
    bufferBarrier.buffer = destinationBuffer;
    bufferBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    textureBarrier.before = textureBarrier.after;
    textureBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    barrierDesc.buffers = &bufferBarrier;
    barrierDesc.bufferNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    context.core.CmdCopyBuffer(*commandBuffer, *readbackBuffer, 0, *destinationBuffer, 0, bufferSize);
    const nri::TextureDataLayoutDesc textureDataLayout = {0, rowPitch, slicePitch};
    context.core.CmdReadbackTextureToBuffer(*commandBuffer, *textureReadback, textureDataLayout, *destinationTexture, streamTextureDesc.dstRegion);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));
    streamerGuard.interface.EndStreamerFrame(*streamerGuard.streamer);

    bool passed = test::VerifyBytes(context.core, *readbackBuffer, expected.data(), bufferSize);

    const uint8_t* readback = (const uint8_t*)context.core.MapBuffer(*textureReadback, 0, slicePitch);
    passed &= readback != nullptr;
    for (uint32_t y = 0; readback && y < textureHeight; y++)
        passed &= memcmp(readback + y * rowPitch, textureData.data() + y * textureWidth * 4, textureWidth * 4) == 0;
    context.core.UnmapBuffer(*textureReadback);

    return test::Report("streamed buffer and texture data", passed);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}

