// © 2026 NVIDIA Corporation

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"

#if NRI_ENABLE_AGILITY_SDK_SUPPORT
#    include "NRIAgilitySDK.h"
#endif

namespace {

struct Settings {
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
    uint32_t adapterIndex = 0;
    bool debugAPI = false;
    bool debugNRI = false;
};

uint32_t Align(uint32_t value, uint32_t alignment) {
    alignment = std::max(alignment, 1u);

    return (value + alignment - 1) / alignment * alignment;
}

uint8_t GetBufferValue(uint32_t i) {
    return uint8_t(i * 29 + 17);
}

uint8_t GetTextureValue(uint32_t x, uint32_t y, uint32_t channel) {
    return uint8_t(x * 17 + y * 31 + channel * 53 + 11);
}

#define RETURN_ON_FAILURE(expression) \
    do { \
        nri::Result result = (expression); \
        if (result != nri::Result::SUCCESS) { \
            printf("FAIL  %s returned %d\n", #expression, int(result)); \
            \
            return false; \
        } \
    } while (0)

class CopyTests {
public:
    ~CopyTests();

    bool Initialize(const Settings& settings);
    bool Run();

private:
    bool CreateBuffer(uint64_t size, nri::MemoryLocation memoryLocation, nri::Buffer*& buffer);
    bool CreateTexture(const nri::TextureDesc& textureDesc, nri::Texture*& texture);
    bool BeginCommandBuffer();
    bool SubmitAndWait();
    bool Report(const char* name, bool passed);
    bool TestBuffers();
    bool VerifyTextureReadback(nri::Buffer& buffer, uint32_t width, uint32_t height, const nri::TextureDataLayoutDesc& layout);
    bool TestTextures();
    bool TransitionHostTexture(nri::Texture& texture, uint16_t mipNum, const nri::AccessLayoutStage& before, const nri::AccessLayoutStage& after);
    bool TestHostCopyFormat(nri::Format format, uint32_t blockWidth, uint32_t blockHeight, uint32_t blockSize, bool& uploadPassed, bool& readbackPassed);
    bool TestHostCopies();

    nri::Device* m_Device = nullptr;
    nri::CoreInterface m_Core = {};
    const nri::DeviceDesc* m_DeviceDesc = nullptr;
    nri::Queue* m_Queue = nullptr;
    nri::CommandAllocator* m_CommandAllocator = nullptr;
    nri::CommandBuffer* m_CommandBuffer = nullptr;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Texture*> m_Textures;
};

CopyTests::~CopyTests() {
    if (!m_Device)
        return;

    if (m_CommandBuffer)
        m_Core.DestroyCommandBuffer(m_CommandBuffer);
    if (m_CommandAllocator)
        m_Core.DestroyCommandAllocator(m_CommandAllocator);

    for (nri::Buffer* buffer : m_Buffers)
        m_Core.DestroyBuffer(buffer);
    for (nri::Texture* texture : m_Textures)
        m_Core.DestroyTexture(texture);

    nri::nriDestroyDevice(m_Device);
}

bool CopyTests::Initialize(const Settings& settings) {
    uint32_t adapterNum = 0;
    RETURN_ON_FAILURE(nri::nriEnumerateAdapters(nullptr, adapterNum));
    if (!adapterNum) {
        printf("FAIL  No adapters found\n");

        return false;
    }

    std::vector<nri::AdapterDesc> adapters(adapterNum);
    RETURN_ON_FAILURE(nri::nriEnumerateAdapters(adapters.data(), adapterNum));

    uint32_t adapterIndex = std::min(settings.adapterIndex, adapterNum - 1);

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = settings.graphicsAPI;
    deviceCreationDesc.adapterDesc = &adapters[adapterIndex];
    deviceCreationDesc.enableGraphicsAPIValidation = settings.debugAPI;
    deviceCreationDesc.enableNRIValidation = settings.debugNRI;
    RETURN_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    RETURN_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), &m_Core));
    RETURN_ON_FAILURE(m_Core.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_Queue));
    RETURN_ON_FAILURE(m_Core.CreateCommandAllocator(*m_Queue, m_CommandAllocator));
    RETURN_ON_FAILURE(m_Core.CreateCommandBuffer(*m_CommandAllocator, m_CommandBuffer));

    m_DeviceDesc = &m_Core.GetDeviceDesc(*m_Device);
    printf("CopyTests: %s, %s\n", nri::nriGetGraphicsAPIString(m_DeviceDesc->graphicsAPI), m_DeviceDesc->adapterDesc.name);

    return true;
}

bool CopyTests::Run() {
    bool result = true;
    result &= TestBuffers();
    result &= TestTextures();
    result &= TestHostCopies();

    return result;
}

bool CopyTests::CreateBuffer(uint64_t size, nri::MemoryLocation memoryLocation, nri::Buffer*& buffer) {
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.usage = nri::BufferUsageBits::NONE;

    RETURN_ON_FAILURE(m_Core.CreateCommittedBuffer(*m_Device, memoryLocation, 0.0f, bufferDesc, buffer));
    m_Buffers.push_back(buffer);

    return true;
}

bool CopyTests::CreateTexture(const nri::TextureDesc& textureDesc, nri::Texture*& texture) {
    RETURN_ON_FAILURE(m_Core.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, texture));
    m_Textures.push_back(texture);

    return true;
}

bool CopyTests::BeginCommandBuffer() {
    RETURN_ON_FAILURE(m_Core.BeginCommandBuffer(*m_CommandBuffer, nullptr));

    return true;
}

bool CopyTests::SubmitAndWait() {
    RETURN_ON_FAILURE(m_Core.EndCommandBuffer(*m_CommandBuffer));

    nri::CommandBuffer* commandBuffer = m_CommandBuffer;
    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;

    RETURN_ON_FAILURE(m_Core.QueueSubmit(*m_Queue, queueSubmitDesc));
    RETURN_ON_FAILURE(m_Core.QueueWaitIdle(m_Queue));
    m_Core.ResetCommandAllocator(*m_CommandAllocator);

    return true;
}

bool CopyTests::Report(const char* name, bool passed) {
    printf("%s  %s\n", passed ? "PASS" : "FAIL", name);

    return passed;
}

bool CopyTests::TestBuffers() {
    constexpr uint32_t bufferSize = 4096;
    constexpr uint32_t zeroOffset = 512;
    constexpr uint32_t zeroSize = 1536;

    nri::Buffer* uploadBuffer = nullptr;
    nri::Buffer* copyBuffer = nullptr;
    nri::Buffer* zeroBuffer = nullptr;
    nri::Buffer* copyReadbackBuffer = nullptr;
    nri::Buffer* zeroReadbackBuffer = nullptr;
    if (!CreateBuffer(bufferSize, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::DEVICE, copyBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::DEVICE, zeroBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::HOST_READBACK, copyReadbackBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::HOST_READBACK, zeroReadbackBuffer))
        return false;

    uint8_t* uploadData = (uint8_t*)m_Core.MapBuffer(*uploadBuffer, 0, bufferSize);
    if (!uploadData) {
        printf("FAIL  MapBuffer(uploadBuffer) returned NULL\n");

        return false;
    }

    for (uint32_t i = 0; i < bufferSize; i++)
        uploadData[i] = GetBufferValue(i);
    m_Core.UnmapBuffer(*uploadBuffer);

    if (!BeginCommandBuffer())
        return false;

    std::array<nri::BufferBarrierDesc, 2> initialBarriers = {};
    initialBarriers[0].buffer = copyBuffer;
    initialBarriers[0].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    initialBarriers[1].buffer = zeroBuffer;
    initialBarriers[1].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.buffers = initialBarriers.data();
    barrierDesc.bufferNum = (uint32_t)initialBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdCopyBuffer(*m_CommandBuffer, *copyBuffer, 0, *uploadBuffer, 0, nri::WHOLE_SIZE);
    m_Core.CmdCopyBuffer(*m_CommandBuffer, *zeroBuffer, 0, *uploadBuffer, 0, nri::WHOLE_SIZE);

    nri::BufferBarrierDesc zeroBefore = {};
    zeroBefore.buffer = zeroBuffer;
    zeroBefore.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    zeroBefore.after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    barrierDesc.buffers = &zeroBefore;
    barrierDesc.bufferNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdZeroBuffer(*m_CommandBuffer, *zeroBuffer, zeroOffset, zeroSize);

    std::array<nri::BufferBarrierDesc, 2> readbackBarriers = {};
    readbackBarriers[0].buffer = copyBuffer;
    readbackBarriers[0].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    readbackBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    readbackBarriers[1].buffer = zeroBuffer;
    readbackBarriers[1].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    readbackBarriers[1].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    barrierDesc.buffers = readbackBarriers.data();
    barrierDesc.bufferNum = (uint32_t)readbackBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdCopyBuffer(*m_CommandBuffer, *copyReadbackBuffer, 0, *copyBuffer, 0, nri::WHOLE_SIZE);
    m_Core.CmdCopyBuffer(*m_CommandBuffer, *zeroReadbackBuffer, 0, *zeroBuffer, 0, nri::WHOLE_SIZE);

    if (!SubmitAndWait())
        return false;

    const uint8_t* copyData = (const uint8_t*)m_Core.MapBuffer(*copyReadbackBuffer, 0, bufferSize);
    const uint8_t* zeroData = (const uint8_t*)m_Core.MapBuffer(*zeroReadbackBuffer, 0, bufferSize);
    if (!copyData || !zeroData) {
        printf("FAIL  MapBuffer(readbackBuffer) returned NULL\n");
        if (copyData)
            m_Core.UnmapBuffer(*copyReadbackBuffer);
        if (zeroData)
            m_Core.UnmapBuffer(*zeroReadbackBuffer);

        return false;
    }

    bool copyPassed = true;
    bool zeroPassed = true;
    for (uint32_t i = 0; i < bufferSize; i++) {
        copyPassed &= copyData[i] == GetBufferValue(i);

        uint8_t expected = i >= zeroOffset && i < zeroOffset + zeroSize ? 0 : GetBufferValue(i);
        zeroPassed &= zeroData[i] == expected;
    }

    m_Core.UnmapBuffer(*copyReadbackBuffer);
    m_Core.UnmapBuffer(*zeroReadbackBuffer);

    return Report("CmdCopyBuffer", copyPassed) && Report("CmdZeroBuffer", zeroPassed);
}

bool CopyTests::VerifyTextureReadback(nri::Buffer& buffer, uint32_t width, uint32_t height, const nri::TextureDataLayoutDesc& layout) {
    const uint8_t* data = (const uint8_t*)m_Core.MapBuffer(buffer, 0, layout.slicePitch);
    if (!data)
        return false;

    bool passed = true;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            for (uint32_t channel = 0; channel < 4; channel++)
                passed &= data[y * layout.rowPitch + x * 4 + channel] == GetTextureValue(x, y, channel);
        }
    }

    m_Core.UnmapBuffer(buffer);

    return passed;
}

bool CopyTests::TestTextures() {
    constexpr uint32_t width = 13;
    constexpr uint32_t height = 7;
    constexpr uint32_t texelSize = 4;

    uint32_t rowAlignment = std::max(m_DeviceDesc->memoryAlignment.uploadBufferTextureRow, 1u);
    uint32_t sliceAlignment = std::max(m_DeviceDesc->memoryAlignment.uploadBufferTextureSlice, 1u);

    nri::TextureDataLayoutDesc dataLayout = {};
    dataLayout.rowPitch = Align(width * texelSize, rowAlignment);
    dataLayout.slicePitch = dataLayout.rowPitch * height;
    while (dataLayout.slicePitch % sliceAlignment)
        dataLayout.slicePitch += dataLayout.rowPitch;

    nri::Buffer* uploadBuffer = nullptr;
    nri::Buffer* uploadReadbackBuffer = nullptr;
    nri::Buffer* copyReadbackBuffer = nullptr;
    if (!CreateBuffer(dataLayout.slicePitch, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer)
        || !CreateBuffer(dataLayout.slicePitch, nri::MemoryLocation::HOST_READBACK, uploadReadbackBuffer)
        || !CreateBuffer(dataLayout.slicePitch, nri::MemoryLocation::HOST_READBACK, copyReadbackBuffer))
        return false;

    uint8_t* uploadData = (uint8_t*)m_Core.MapBuffer(*uploadBuffer, 0, dataLayout.slicePitch);
    if (!uploadData) {
        printf("FAIL  MapBuffer(textureUploadBuffer) returned NULL\n");

        return false;
    }

    memset(uploadData, 0xCD, dataLayout.slicePitch);
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            for (uint32_t channel = 0; channel < 4; channel++)
                uploadData[y * dataLayout.rowPitch + x * 4 + channel] = GetTextureValue(x, y, channel);
        }
    }
    m_Core.UnmapBuffer(*uploadBuffer);

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.depth = 1;
    textureDesc.mipNum = 1;
    textureDesc.layerNum = 1;
    textureDesc.sampleNum = 1;

    nri::Texture* uploadTexture = nullptr;
    nri::Texture* copyTexture = nullptr;
    if (!CreateTexture(textureDesc, uploadTexture) || !CreateTexture(textureDesc, copyTexture))
        return false;

    nri::TextureRegionDesc wholeTexture = {};
    wholeTexture.width = nri::WHOLE_SIZE;
    wholeTexture.height = nri::WHOLE_SIZE;
    wholeTexture.depth = nri::WHOLE_SIZE;
    wholeTexture.planes = nri::PlaneBits::COLOR;

    if (!BeginCommandBuffer())
        return false;

    std::array<nri::TextureBarrierDesc, 2> initialBarriers = {};
    initialBarriers[0].texture = uploadTexture;
    initialBarriers[0].before.layout = nri::Layout::UNDEFINED;
    initialBarriers[0].after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
    initialBarriers[0].mipNum = 1;
    initialBarriers[0].layerNum = 1;
    initialBarriers[0].planes = nri::PlaneBits::COLOR;
    initialBarriers[1] = initialBarriers[0];
    initialBarriers[1].texture = copyTexture;

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = initialBarriers.data();
    barrierDesc.textureNum = (uint32_t)initialBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdUploadBufferToTexture(*m_CommandBuffer, *uploadTexture, wholeTexture, *uploadBuffer, dataLayout);

    nri::TextureBarrierDesc uploadBarrier = {};
    uploadBarrier.texture = uploadTexture;
    uploadBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
    uploadBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    uploadBarrier.mipNum = 1;
    uploadBarrier.layerNum = 1;
    uploadBarrier.planes = nri::PlaneBits::COLOR;
    barrierDesc.textures = &uploadBarrier;
    barrierDesc.textureNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdReadbackTextureToBuffer(*m_CommandBuffer, *uploadReadbackBuffer, dataLayout, *uploadTexture, wholeTexture);
    m_Core.CmdCopyTexture(*m_CommandBuffer, *copyTexture, nullptr, *uploadTexture, nullptr);

    nri::TextureBarrierDesc copyBarrier = uploadBarrier;
    copyBarrier.texture = copyTexture;
    barrierDesc.textures = &copyBarrier;
    barrierDesc.textureNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdReadbackTextureToBuffer(*m_CommandBuffer, *copyReadbackBuffer, dataLayout, *copyTexture, wholeTexture);

    if (!SubmitAndWait())
        return false;

    bool uploadPassed = VerifyTextureReadback(*uploadReadbackBuffer, width, height, dataLayout);
    bool copyPassed = VerifyTextureReadback(*copyReadbackBuffer, width, height, dataLayout);

    return Report("CmdUploadBufferToTexture", uploadPassed)
        && Report("CmdReadbackTextureToBuffer", uploadPassed)
        && Report("CmdCopyTexture", copyPassed);
}

bool CopyTests::TransitionHostTexture(nri::Texture& texture, uint16_t mipNum, const nri::AccessLayoutStage& before, const nri::AccessLayoutStage& after) {
    if (!BeginCommandBuffer())
        return false;

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = &texture;
    textureBarrier.before = before;
    textureBarrier.after = after;
    textureBarrier.mipNum = mipNum;
    textureBarrier.layerNum = 1;
    textureBarrier.planes = nri::PlaneBits::COLOR;

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    return SubmitAndWait();
}

bool CopyTests::TestHostCopyFormat(nri::Format format, uint32_t blockWidth, uint32_t blockHeight, uint32_t blockSize, bool& uploadPassed, bool& readbackPassed) {
    constexpr uint32_t width = 64;
    constexpr uint32_t height = 64;
    constexpr uint16_t mipNum = 7;
    constexpr uint8_t readbackSentinel = 0xCD;

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::HOST_TRANSFER;
    textureDesc.format = format;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.depth = 1;
    textureDesc.mipNum = mipNum;
    textureDesc.layerNum = 1;
    textureDesc.sampleNum = 1;

    nri::Texture* uploadTexture = nullptr;
    nri::Texture* readbackTexture = nullptr;
    if (!CreateTexture(textureDesc, uploadTexture) || !CreateTexture(textureDesc, readbackTexture))
        return false;

    struct MipData {
        uint32_t rowSize = 0;
        uint32_t rowNum = 0;
        uint32_t rowPitch = 0;
        uint32_t slicePitch = 0;
        nri::TextureDataLayoutDesc gpuLayout = {};
        nri::Buffer* gpuUploadBuffer = nullptr;
        nri::Buffer* gpuReadbackBuffer = nullptr;
        nri::Texture* gpuReadbackTexture = nullptr;
        std::vector<uint8_t> expected;
        std::vector<uint8_t> hostUpload;
        std::vector<uint8_t> hostReadback;
    };

    uint32_t rowAlignment = std::max(m_DeviceDesc->memoryAlignment.uploadBufferTextureRow, 1u);
    uint32_t sliceAlignment = std::max(m_DeviceDesc->memoryAlignment.uploadBufferTextureSlice, 1u);

    std::array<MipData, mipNum> mipData;
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        uint32_t mipWidth = std::max(width >> mip, 1u);
        uint32_t mipHeight = std::max(height >> mip, 1u);

        MipData& data = mipData[mip];
        data.rowSize = Align(mipWidth, blockWidth) / blockWidth * blockSize;
        data.rowNum = Align(mipHeight, blockHeight) / blockHeight;
        data.rowPitch = data.rowSize + blockSize;
        data.slicePitch = data.rowPitch * (data.rowNum + 1);
        data.expected.resize(data.slicePitch, uint8_t(0x80 + mip));
        data.hostReadback.resize(data.slicePitch, readbackSentinel);

        data.gpuLayout.rowPitch = Align(data.rowSize, rowAlignment);
        data.gpuLayout.slicePitch = data.gpuLayout.rowPitch * data.rowNum;
        while (data.gpuLayout.slicePitch % sliceAlignment)
            data.gpuLayout.slicePitch += data.gpuLayout.rowPitch;

        if (!CreateBuffer(data.gpuLayout.slicePitch, nri::MemoryLocation::HOST_UPLOAD, data.gpuUploadBuffer)
            || !CreateBuffer(data.gpuLayout.slicePitch, nri::MemoryLocation::HOST_READBACK, data.gpuReadbackBuffer))
            return false;

        nri::TextureDesc gpuReadbackTextureDesc = textureDesc;
        gpuReadbackTextureDesc.usage = nri::TextureUsageBits::NONE;
        gpuReadbackTextureDesc.width = (uint16_t)std::max(mipWidth, blockWidth);
        gpuReadbackTextureDesc.height = (uint16_t)std::max(mipHeight, blockHeight);
        gpuReadbackTextureDesc.mipNum = 1;
        if (!CreateTexture(gpuReadbackTextureDesc, data.gpuReadbackTexture))
            return false;

        for (uint32_t row = 0; row < data.rowNum; row++) {
            for (uint32_t i = 0; i < data.rowSize; i++)
                data.expected[row * data.rowPitch + i] = uint8_t(mip * 43 + row * 19 + i * 7 + 3);
        }

        data.hostUpload = data.expected;

        uint8_t* gpuUploadData = (uint8_t*)m_Core.MapBuffer(*data.gpuUploadBuffer, 0, data.gpuLayout.slicePitch);
        if (!gpuUploadData) {
            printf("FAIL  MapBuffer(gpuUploadBuffer) returned NULL\n");

            return false;
        }

        memset(gpuUploadData, 0xCD, data.gpuLayout.slicePitch);
        for (uint32_t row = 0; row < data.rowNum; row++)
            memcpy(gpuUploadData + row * data.gpuLayout.rowPitch, data.expected.data() + row * data.rowPitch, data.rowSize);
        m_Core.UnmapBuffer(*data.gpuUploadBuffer);
    }

    constexpr nri::AccessLayoutStage hostWrite = {nri::AccessBits::HOST_WRITE, nri::Layout::GENERAL, nri::StageBits::HOST};
    constexpr nri::AccessLayoutStage hostRead = {nri::AccessBits::HOST_READ, nri::Layout::GENERAL, nri::StageBits::HOST};
    constexpr nri::AccessLayoutStage copySource = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    constexpr nri::AccessLayoutStage copyDestination = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};

    nri::AccessLayoutStage undefined = {};
    undefined.layout = nri::Layout::UNDEFINED;

    std::array<nri::TextureRegionDesc, mipNum> regions = {};
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        regions[mip].width = nri::WHOLE_SIZE;
        regions[mip].height = nri::WHOLE_SIZE;
        regions[mip].depth = 1;
        regions[mip].mipOffset = (uint16_t)mip;
        regions[mip].planes = nri::PlaneBits::COLOR;
    }

    if (!TransitionHostTexture(*uploadTexture, mipNum, undefined, hostWrite))
        return false;

    std::array<nri::UploadHostMemoryToTextureDesc, mipNum> uploadDescs = {};
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        uploadDescs[mip].srcData = mipData[mip].hostUpload.data();
        uploadDescs[mip].dstTexture = uploadTexture;
        uploadDescs[mip].dstRegion = regions[mip];
        uploadDescs[mip].srcRowPitch = mipData[mip].rowPitch;
        uploadDescs[mip].srcSlicePitch = mipData[mip].slicePitch;
    }

    RETURN_ON_FAILURE(m_Core.UploadHostMemoryToTexture(*m_Queue, uploadDescs.data(), (uint32_t)uploadDescs.size()));

    for (MipData& data : mipData)
        memset(data.hostUpload.data(), 0xFE, data.hostUpload.size());

    if (!BeginCommandBuffer())
        return false;

    std::array<nri::TextureBarrierDesc, mipNum + 1> copyBarriers = {};
    copyBarriers[0].texture = uploadTexture;
    copyBarriers[0].before = hostWrite;
    copyBarriers[0].after = copySource;
    copyBarriers[0].mipNum = mipNum;
    copyBarriers[0].layerNum = 1;
    copyBarriers[0].planes = nri::PlaneBits::COLOR;

    for (uint32_t mip = 0; mip < mipNum; mip++) {
        copyBarriers[mip + 1].texture = mipData[mip].gpuReadbackTexture;
        copyBarriers[mip + 1].before.layout = nri::Layout::UNDEFINED;
        copyBarriers[mip + 1].after = copyDestination;
        copyBarriers[mip + 1].mipNum = 1;
        copyBarriers[mip + 1].layerNum = 1;
        copyBarriers[mip + 1].planes = nri::PlaneBits::COLOR;
    }

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = copyBarriers.data();
    barrierDesc.textureNum = (uint32_t)copyBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    for (uint32_t mip = 0; mip < mipNum; mip++)
        m_Core.CmdCopyTexture(*m_CommandBuffer, *mipData[mip].gpuReadbackTexture, &regions[0], *uploadTexture, &regions[mip]);

    std::array<nri::TextureBarrierDesc, mipNum> readbackBarriers = {};
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        readbackBarriers[mip].texture = mipData[mip].gpuReadbackTexture;
        readbackBarriers[mip].before = copyDestination;
        readbackBarriers[mip].after = copySource;
        readbackBarriers[mip].mipNum = 1;
        readbackBarriers[mip].layerNum = 1;
        readbackBarriers[mip].planes = nri::PlaneBits::COLOR;
    }

    barrierDesc.textures = readbackBarriers.data();
    barrierDesc.textureNum = (uint32_t)readbackBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    for (uint32_t mip = 0; mip < mipNum; mip++)
        m_Core.CmdReadbackTextureToBuffer(*m_CommandBuffer, *mipData[mip].gpuReadbackBuffer, mipData[mip].gpuLayout, *mipData[mip].gpuReadbackTexture, regions[0]);

    if (!SubmitAndWait())
        return false;

    uploadPassed = true;
    for (const MipData& data : mipData) {
        const uint8_t* gpuReadbackData = (const uint8_t*)m_Core.MapBuffer(*data.gpuReadbackBuffer, 0, data.gpuLayout.slicePitch);
        if (!gpuReadbackData) {
            printf("FAIL  MapBuffer(gpuReadbackBuffer) returned NULL\n");

            return false;
        }

        for (uint32_t row = 0; row < data.rowNum; row++)
            uploadPassed &= memcmp(data.expected.data() + row * data.rowPitch, gpuReadbackData + row * data.gpuLayout.rowPitch, data.rowSize) == 0;
        m_Core.UnmapBuffer(*data.gpuReadbackBuffer);
    }

    if (!TransitionHostTexture(*readbackTexture, mipNum, undefined, copyDestination) || !BeginCommandBuffer())
        return false;

    for (uint32_t mip = 0; mip < mipNum; mip++)
        m_Core.CmdUploadBufferToTexture(*m_CommandBuffer, *readbackTexture, regions[mip], *mipData[mip].gpuUploadBuffer, mipData[mip].gpuLayout);

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = readbackTexture;
    textureBarrier.before = copyDestination;
    textureBarrier.after = hostRead;
    textureBarrier.mipNum = mipNum;
    textureBarrier.layerNum = 1;
    textureBarrier.planes = nri::PlaneBits::COLOR;

    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    if (!SubmitAndWait())
        return false;

    std::array<nri::ReadbackTextureToHostMemoryDesc, mipNum> readbackDescs = {};
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        readbackDescs[mip].srcTexture = readbackTexture;
        readbackDescs[mip].dstData = mipData[mip].hostReadback.data();
        readbackDescs[mip].srcRegion = regions[mip];
        readbackDescs[mip].dstRowPitch = mipData[mip].rowPitch;
        readbackDescs[mip].dstSlicePitch = mipData[mip].slicePitch;
    }

    RETURN_ON_FAILURE(m_Core.ReadbackTextureToHostMemory(*m_Queue, readbackDescs.data(), (uint32_t)readbackDescs.size()));

    readbackPassed = true;
    for (const MipData& data : mipData) {
        for (uint32_t row = 0; row < data.rowNum; row++) {
            readbackPassed &= memcmp(data.expected.data() + row * data.rowPitch, data.hostReadback.data() + row * data.rowPitch, data.rowSize) == 0;

            for (uint32_t i = data.rowSize; i < data.rowPitch; i++)
                readbackPassed &= data.hostReadback[row * data.rowPitch + i] == readbackSentinel;
        }

        for (uint32_t i = data.rowPitch * data.rowNum; i < data.slicePitch; i++)
            readbackPassed &= data.hostReadback[i] == readbackSentinel;
    }

    return true;
}

bool CopyTests::TestHostCopies() {
    if (!(m_Core.GetFormatSupport(*m_Device, nri::Format::RGBA8_UNORM) & nri::FormatSupportBits::HOST_COPY)) {
        printf("FAIL  RGBA8_UNORM does not support host copies\n");

        return false;
    }

    bool uploadPassed = false;
    bool readbackPassed = false;
    if (!TestHostCopyFormat(nri::Format::RGBA8_UNORM, 1, 1, 4, uploadPassed, readbackPassed))
        return false;

    if (m_Core.GetFormatSupport(*m_Device, nri::Format::BC1_RGBA_UNORM) & nri::FormatSupportBits::HOST_COPY) {
        bool bcUploadPassed = false;
        bool bcReadbackPassed = false;
        if (!TestHostCopyFormat(nri::Format::BC1_RGBA_UNORM, 4, 4, 8, bcUploadPassed, bcReadbackPassed))
            return false;

        uploadPassed &= bcUploadPassed;
        readbackPassed &= bcReadbackPassed;
    } else
        printf("SKIP  BC1_RGBA_UNORM host copies are unsupported\n");

    return Report("UploadHostMemoryToTexture", uploadPassed) && Report("ReadbackTextureToHostMemory", readbackPassed);
}

#undef RETURN_ON_FAILURE

Settings ParseSettings(int argc, char** argv) {
    Settings settings;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--api=D3D11"))
            settings.graphicsAPI = nri::GraphicsAPI::D3D11;
        else if (!strcmp(argv[i], "--api=D3D12"))
            settings.graphicsAPI = nri::GraphicsAPI::D3D12;
        else if (!strcmp(argv[i], "--api=VULKAN"))
            settings.graphicsAPI = nri::GraphicsAPI::VK;
        else if (!strcmp(argv[i], "--api=WGPU"))
            settings.graphicsAPI = nri::GraphicsAPI::WGPU;
        else if (!strcmp(argv[i], "--debugAPI"))
            settings.debugAPI = true;
        else if (!strcmp(argv[i], "--debugNRI"))
            settings.debugNRI = true;
        else if (!strncmp(argv[i], "--adapter=", 10))
            settings.adapterIndex = (uint32_t)strtoul(argv[i] + 10, nullptr, 10);
    }

    return settings;
}

} // namespace

int main(int argc, char** argv) {
    CopyTests copyTests;
    if (!copyTests.Initialize(ParseSettings(argc, argv)))
        return 1;

    return copyTests.Run() ? 0 : 1;
}
