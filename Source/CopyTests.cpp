// © 2026 NVIDIA Corporation

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "NRIFramework.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"

#if NRI_ENABLE_AGILITY_SDK_SUPPORT
#    include "NRIAgilitySDK.h"
#endif

namespace {

struct Settings {
#if (defined(__APPLE__) && NRI_ENABLE_METAL_SHADER_CONVERTER)
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::METAL;
#else
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
#endif
    uint32_t adapterIndex = 0;
    nri::QueueType queueType = nri::QueueType::GRAPHICS;
    bool debugAPI = false;
    bool debugNRI = false;
    bool legacyBarriers = false;
};

static inline uint32_t Align(uint32_t value, uint32_t alignment) {
    alignment = std::max(alignment, 1u);

    return (value + alignment - 1) / alignment * alignment;
}

static inline uint8_t GetBufferValue(uint32_t i) {
    return uint8_t(i * 29 + 17);
}

static inline uint8_t GetTextureValue(uint32_t x, uint32_t y, uint32_t channel) {
    return uint8_t(x * 17 + y * 31 + channel * 53 + 11);
}

static inline bool Report(const char* name, bool passed) {
    printf("%s  %s\n", passed ? "PASS" : "FAIL", name);
    return passed;
}

static inline bool IsInsideRects(uint32_t x, uint32_t y, const nri::Rect* rects, uint32_t rectNum) {
    for (uint32_t i = 0; i < rectNum; i++) {
        const uint32_t left = uint32_t(rects[i].x);
        const uint32_t top = uint32_t(rects[i].y);
        if (x >= left && x < left + rects[i].width && y >= top && y < top + rects[i].height)
            return true;
    }
    return false;
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
    void TransitionBuffer(nri::Buffer& buffer, const nri::AccessStage& before, const nri::AccessStage& after);
    bool VerifyBufferReadback(nri::Buffer& buffer, uint32_t size, uint32_t zeroOffset, uint32_t zeroSize);
    bool VerifyAttachmentReadback(nri::Buffer& buffer, const nri::TextureDesc& textureDesc, uint32_t attachmentIndex, const nri::TextureDataLayoutDesc& layout, const nri::Rect* rects, uint32_t rectNum);
    bool TestBuffers();
    bool TestAttachmentClears(nri::Format format);
    bool VerifyTextureReadback(nri::Buffer& buffer, uint32_t width, uint32_t height, const nri::TextureDataLayoutDesc& layout);
    bool TestTextures();
    bool TestTextureBarrierAccess();
    bool TransitionHostTexture(nri::Texture& texture, uint16_t mipNum, const nri::AccessLayoutStage& before, const nri::AccessLayoutStage& after);
    bool TestHostCopyFormat(nri::Format format, uint32_t blockWidth, uint32_t blockHeight, uint32_t blockSize, bool& uploadPassed, bool& readbackPassed, bool& wholeCopyPassed);
    bool TestHostCopies();

    nri::Device* m_Device = nullptr;
    nri::CoreInterface m_Core = {};
    const nri::DeviceDesc* m_DeviceDesc = nullptr;
    nri::Queue* m_Queue = nullptr;
    nri::QueueType m_QueueType = nri::QueueType::GRAPHICS;
    nri::CommandAllocator* m_CommandAllocator = nullptr;
    nri::CommandBuffer* m_CommandBuffer = nullptr;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::Descriptor*> m_Descriptors;
};

CopyTests::~CopyTests() {
    if (!m_Device)
        return;

    if (m_CommandBuffer)
        m_Core.DestroyCommandBuffer(m_CommandBuffer);
    if (m_CommandAllocator)
        m_Core.DestroyCommandAllocator(m_CommandAllocator);

    for (nri::Descriptor* descriptor : m_Descriptors)
        m_Core.DestroyDescriptor(descriptor);
    for (nri::Buffer* buffer : m_Buffers)
        m_Core.DestroyBuffer(buffer);
    for (nri::Texture* texture : m_Textures)
        m_Core.DestroyTexture(texture);

    nri::nriDestroyDevice(m_Device);
}

bool CopyTests::Initialize(const Settings& settings) {
    m_QueueType = settings.queueType;

    uint32_t adapterNum = 0;
    RETURN_ON_FAILURE(nri::nriEnumerateAdapters(nullptr, adapterNum, settings.graphicsAPI));
    if (!adapterNum) {
        printf("SKIP  No adapters found\n");
        exit(NRIF_SKIP);
    }

    std::vector<nri::AdapterDesc> adapters(adapterNum);
    RETURN_ON_FAILURE(nri::nriEnumerateAdapters(adapters.data(), adapterNum, settings.graphicsAPI));
    const uint32_t adapterIndex = std::min(settings.adapterIndex, adapterNum - 1);
    const nri::AdapterDesc& adapter = adapters[adapterIndex];
    if (!(adapter.supportedGraphicsAPIs & settings.graphicsAPI))
        exit(NRIF_SKIP);
    if (!adapter.queueNum[size_t(m_QueueType)]) {
        printf("SKIP  Requested queue type is unsupported\n");
        exit(NRIF_SKIP);
    }

    nri::QueueFamilyDesc queueFamily = {};
    queueFamily.queueType = m_QueueType;
    queueFamily.queueNum = 1;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = settings.graphicsAPI;
    deviceCreationDesc.adapterDesc = &adapter;
    deviceCreationDesc.enableGraphicsAPIValidation = settings.debugAPI;
    deviceCreationDesc.enableNRIValidation = settings.debugNRI;
    deviceCreationDesc.disableD3D12EnhancedBarriers = settings.legacyBarriers;
    deviceCreationDesc.queueFamilies = &queueFamily;
    deviceCreationDesc.queueFamilyNum = 1;
    RETURN_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    RETURN_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), &m_Core));
    m_DeviceDesc = &m_Core.GetDeviceDesc(*m_Device);

    RETURN_ON_FAILURE(m_Core.GetQueue(*m_Device, m_QueueType, 0, m_Queue));
    RETURN_ON_FAILURE(m_Core.CreateCommandAllocator(*m_Queue, m_CommandAllocator));
    RETURN_ON_FAILURE(m_Core.CreateCommandBuffer(*m_CommandAllocator, m_CommandBuffer));

    printf("CopyTests: %s, %s\n", nri::nriGetGraphicsAPIString(m_DeviceDesc->graphicsAPI), m_DeviceDesc->adapterDesc.name);
    return true;
}

bool CopyTests::Run() {
    bool result = true;
    result &= TestBuffers();
    if (m_QueueType != nri::QueueType::GRAPHICS)
        return result;

    result &= TestAttachmentClears(nri::Format::RGBA8_UNORM);
    result &= TestAttachmentClears(nri::Format::RGBA8_UINT);
    result &= TestAttachmentClears(nri::Format::D32_SFLOAT);
    result &= TestTextures();
    result &= TestTextureBarrierAccess();
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

void CopyTests::TransitionBuffer(nri::Buffer& buffer, const nri::AccessStage& before, const nri::AccessStage& after) {
    nri::BufferBarrierDesc bufferBarrier = {};
    bufferBarrier.buffer = &buffer;
    bufferBarrier.before = before;
    bufferBarrier.after = after;

    nri::BarrierDesc barriers = {};
    barriers.buffers = &bufferBarrier;
    barriers.bufferNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barriers);
}

bool CopyTests::VerifyBufferReadback(nri::Buffer& buffer, uint32_t size, uint32_t zeroOffset, uint32_t zeroSize) {
    const uint8_t* data = (const uint8_t*)m_Core.MapBuffer(buffer, 0, size);
    if (!data) {
        printf("FAIL  MapBuffer(readback) returned NULL\n");
        return false;
    }

    bool passed = true;
    for (uint32_t i = 0; i < size; i++) {
        const bool isZeroed = i >= zeroOffset && i - zeroOffset < zeroSize;
        const uint8_t expected = isZeroed ? 0 : GetBufferValue(i);
        if (data[i] != expected) {
            printf("Buffer mismatch byte=%u value=%u expected=%u\n", i, data[i], expected);
            passed = false;
            break;
        }
    }

    m_Core.UnmapBuffer(buffer);
    return passed;
}

bool CopyTests::TestBuffers() {
    constexpr uint32_t bufferSize = 4096;
    const struct {
        const char* name;
        bool zeroBuffer;
        uint32_t offset;
        uint32_t size;
    } cases[] = {
        {"CmdCopyBuffer", false, 0, 0},
        {"CmdZeroBuffer (4-byte aligned)", true, 516, 1532},
        {"CmdZeroBuffer (16-byte aligned)", true, 512, 1536},
        {"CmdZeroBuffer (WHOLE_SIZE)", true, 0, nri::WHOLE_SIZE},
    };

    nri::Buffer* uploadBuffer = nullptr;
    nri::Buffer* deviceBuffer = nullptr;
    nri::Buffer* readbackBuffer = nullptr;
    if (!CreateBuffer(bufferSize, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::DEVICE, deviceBuffer)
        || !CreateBuffer(bufferSize, nri::MemoryLocation::HOST_READBACK, readbackBuffer))
        return false;

    uint8_t* uploadData = (uint8_t*)m_Core.MapBuffer(*uploadBuffer, 0, bufferSize);
    if (!uploadData) {
        printf("FAIL  MapBuffer(upload) returned NULL\n");
        return false;
    }
    for (uint32_t i = 0; i < bufferSize; i++)
        uploadData[i] = GetBufferValue(i);
    m_Core.UnmapBuffer(*uploadBuffer);

    const nri::AccessStage copySource = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    const nri::AccessStage copyDestination = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    nri::AccessStage currentState = {};
    bool passed = true;
    for (const auto& test : cases) {
        if (!BeginCommandBuffer())
            return false;

        TransitionBuffer(*deviceBuffer, currentState, copyDestination);
        m_Core.CmdCopyBuffer(*m_CommandBuffer, *deviceBuffer, 0, *uploadBuffer, 0, nri::WHOLE_SIZE);

        uint32_t zeroSize = 0;
        if (test.zeroBuffer) {
            TransitionBuffer(*deviceBuffer, copyDestination, copyDestination);
            m_Core.CmdZeroBuffer(*m_CommandBuffer, *deviceBuffer, test.offset, test.size);
            zeroSize = test.size == nri::WHOLE_SIZE ? bufferSize : test.size;
        }

        TransitionBuffer(*deviceBuffer, copyDestination, copySource);
        m_Core.CmdCopyBuffer(*m_CommandBuffer, *readbackBuffer, 0, *deviceBuffer, 0, nri::WHOLE_SIZE);
        if (!SubmitAndWait())
            return false;

        passed &= Report(test.name, VerifyBufferReadback(*readbackBuffer, bufferSize, test.offset, zeroSize));
        currentState = copySource;
    }
    return passed;
}

bool CopyTests::TestAttachmentClears(nri::Format format) {
    const bool isDepth = format == nri::Format::D32_SFLOAT;
    const bool isInteger = format == nri::Format::RGBA8_UINT;
    const char* testName = "CmdClearAttachments (MRT array)";
    if (isDepth)
        testName = "CmdClearAttachments (depth array)";
    else if (isInteger)
        testName = "CmdClearAttachments (integer MRT array)";
    const nri::FormatSupportBits requiredSupport = isDepth ? nri::FormatSupportBits::DEPTH_STENCIL_ATTACHMENT : nri::FormatSupportBits::COLOR_ATTACHMENT;
    if (!(m_Core.GetFormatSupport(*m_Device, format) & requiredSupport)) {
        printf("SKIP  %s: attachment format is unsupported\n", testName);
        return true;
    }

    constexpr uint16_t width = 8;
    constexpr uint16_t height = 6;
    const uint16_t layerNum = std::min<uint16_t>(2, m_DeviceDesc->dimensions.attachmentLayerMaxNum);
    const uint32_t attachmentNum = isDepth ? 1 : std::min<uint32_t>(2, m_DeviceDesc->shaderStage.fragment.attachmentMaxNum);
    if (!layerNum || !attachmentNum) {
        printf("SKIP  %s: attachment layers or color attachments are unsupported\n", testName);
        return true;
    }
    const bool hasRectClears = isDepth ? m_DeviceDesc->features.rectDepthStencilClears : m_DeviceDesc->features.rectColorClears;
    if (!hasRectClears) {
        printf("SKIP  %s: rectangle clears are unsupported\n", testName);
        return true;
    }
    const nri::PlaneBits planes = isDepth ? nri::PlaneBits::DEPTH : nri::PlaneBits::COLOR;
    const nri::AccessLayoutStage attachmentState = isDepth
        ? nri::AccessLayoutStage{nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_WRITE, nri::Layout::DEPTH_STENCIL_ATTACHMENT, nri::StageBits::DEPTH_STENCIL_ATTACHMENT}
        : nri::AccessLayoutStage{nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
    const nri::AccessLayoutStage uploadState = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};

    nri::TextureDataLayoutDesc dataLayout = {};
    dataLayout.rowPitch = Align(width * 4, m_DeviceDesc->memoryAlignment.uploadBufferTextureRow);
    dataLayout.slicePitch = dataLayout.rowPitch * height;
    while (dataLayout.slicePitch % std::max(m_DeviceDesc->memoryAlignment.uploadBufferTextureSlice, 1u))
        dataLayout.slicePitch += dataLayout.rowPitch;

    // Seed integer attachments through a copy so typed zero clears have nonzero guard pixels.
    nri::Buffer* uploadBuffer = nullptr;
    if (isInteger) {
        if (!CreateBuffer(dataLayout.slicePitch, nri::MemoryLocation::HOST_UPLOAD, uploadBuffer))
            return false;
        void* data = m_Core.MapBuffer(*uploadBuffer, 0, dataLayout.slicePitch);
        if (!data)
            return false;
        memset(data, 1, dataLayout.slicePitch);
        m_Core.UnmapBuffer(*uploadBuffer);
    }

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = format;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.layerNum = layerNum;
    textureDesc.usage = isDepth ? nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT : nri::TextureUsageBits::COLOR_ATTACHMENT;

    nri::Texture* textures[2] = {};
    nri::Buffer* readbacks[2] = {};
    nri::AttachmentDesc attachments[2][2] = {};
    nri::TextureBarrierDesc textureBarriers[2] = {};
    for (uint32_t i = 0; i < attachmentNum; i++) {
        if (!CreateTexture(textureDesc, textures[i]) || !CreateBuffer(uint64_t(dataLayout.slicePitch) * layerNum, nri::MemoryLocation::HOST_READBACK, readbacks[i]))
            return false;

        for (uint16_t layer = 0; layer < layerNum; layer++) {
            nri::TextureViewDesc viewDesc = {};
            viewDesc.texture = textures[i];
            viewDesc.type = isDepth ? nri::TextureView::DEPTH_STENCIL_ATTACHMENT : nri::TextureView::COLOR_ATTACHMENT;
            viewDesc.layerOffset = layer;
            viewDesc.layerNum = 1;
            viewDesc.planes = planes;
            nri::AttachmentDesc& attachment = attachments[layer][i];
            RETURN_ON_FAILURE(m_Core.CreateTextureView(viewDesc, attachment.descriptor));
            m_Descriptors.push_back(attachment.descriptor);
            attachment.loadOp = isInteger ? nri::LoadOp::LOAD : nri::LoadOp::CLEAR;
            attachment.storeOp = nri::StoreOp::STORE;
            if (isDepth)
                attachment.clearValue.depthStencil.depth = 1.0f;
            else if (!isInteger)
                attachment.clearValue.color.f = {1.0f, 1.0f, 1.0f, 1.0f};
        }

        textureBarriers[i].texture = textures[i];
        textureBarriers[i].before.layout = nri::Layout::UNDEFINED;
        textureBarriers[i].after = isInteger ? uploadState : attachmentState;
        textureBarriers[i].mipNum = 1;
        textureBarriers[i].layerNum = layerNum;
        textureBarriers[i].planes = planes;
    }

    if (!BeginCommandBuffer())
        return false;
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = textureBarriers;
    barrierDesc.textureNum = attachmentNum;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    if (isInteger) {
        for (uint32_t i = 0; i < attachmentNum; i++) {
            for (uint16_t layer = 0; layer < layerNum; layer++) {
                nri::TextureRegionDesc region = {};
                region.width = width;
                region.height = height;
                region.depth = 1;
                region.layerOffset = layer;
                m_Core.CmdUploadBufferToTexture(*m_CommandBuffer, *textures[i], region, *uploadBuffer, dataLayout);
            }
            textureBarriers[i].before = uploadState;
            textureBarriers[i].after = attachmentState;
        }
        m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);
    }

    const nri::Rect rects[] = {{1, 1, 3, 2}, {2, 4, 4, 1}};
    const uint32_t rectNum = uint32_t(std::size(rects));
    for (uint16_t layer = 0; layer < layerNum; layer++) {
        nri::RenderingDesc renderingDesc = {};
        if (isDepth)
            renderingDesc.depth = attachments[layer][0];
        else {
            renderingDesc.colors = attachments[layer];
            renderingDesc.colorNum = attachmentNum;
        }
        m_Core.CmdBeginRendering(*m_CommandBuffer, renderingDesc);

        nri::ClearAttachmentDesc clears[2] = {};
        for (uint32_t i = 0; i < attachmentNum; i++) {
            clears[i].planes = planes;
            clears[i].colorAttachmentIndex = uint8_t(i);
            if (isDepth)
                clears[i].value.depthStencil.depth = 1.0f;
            else if (isInteger)
                clears[i].value.color.ui = {0, 0, 0, 0};
            else
                clears[i].value.color.f = {1.0f, 1.0f, 1.0f, 1.0f};
        }
        if (!isInteger) {
            // Clear slot 0 again, then overwrite its rectangles; slot 1 keeps its load clear outside them.
            nri::ClearAttachmentDesc repeatClear = {};
            repeatClear.planes = planes;
            m_Core.CmdClearAttachments(*m_CommandBuffer, &repeatClear, 1, nullptr, 0);
        }

        // Both rectangles are wholly inside the attachment; every array layer uses its own view.
        if (attachmentNum > 1)
            clears[1].value = {};
        m_Core.CmdClearAttachments(*m_CommandBuffer, clears, attachmentNum, rects, rectNum);
        m_Core.CmdEndRendering(*m_CommandBuffer);
    }

    for (uint32_t i = 0; i < attachmentNum; i++) {
        textureBarriers[i].before = attachmentState;
        textureBarriers[i].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
    }
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);
    for (uint32_t i = 0; i < attachmentNum; i++) {
        for (uint16_t layer = 0; layer < layerNum; layer++) {
            nri::TextureRegionDesc region = {};
            region.width = width;
            region.height = height;
            region.depth = 1;
            region.layerOffset = layer;
            region.planes = planes;
            dataLayout.offset = uint64_t(dataLayout.slicePitch) * layer;
            m_Core.CmdReadbackTextureToBuffer(*m_CommandBuffer, *readbacks[i], dataLayout, *textures[i], region);
        }
    }
    if (!SubmitAndWait())
        return false;

    bool passed = true;
    for (uint32_t i = 0; i < attachmentNum; i++)
        passed &= VerifyAttachmentReadback(*readbacks[i], textureDesc, i, dataLayout, rects, rectNum);
    return Report(testName, passed);
}

bool CopyTests::VerifyAttachmentReadback(nri::Buffer& buffer, const nri::TextureDesc& textureDesc, uint32_t attachmentIndex,
    const nri::TextureDataLayoutDesc& layout, const nri::Rect* rects, uint32_t rectNum) {
    const uint8_t* data = (const uint8_t*)m_Core.MapBuffer(buffer, 0, uint64_t(layout.slicePitch) * textureDesc.layerNum);
    if (!data) {
        printf("FAIL  MapBuffer(attachment readback) returned NULL\n");
        return false;
    }

    const bool isDepth = textureDesc.format == nri::Format::D32_SFLOAT;
    const bool isInteger = textureDesc.format == nri::Format::RGBA8_UINT;
    bool passed = true;
    for (uint32_t layer = 0; layer < textureDesc.layerNum; layer++) {
        for (uint32_t y = 0; y < textureDesc.height; y++) {
            for (uint32_t x = 0; x < textureDesc.width; x++) {
                const bool inside = IsInsideRects(x, y, rects, rectNum);
                bool expectedOne = !inside;
                if (!isInteger && attachmentIndex == 0)
                    expectedOne = inside;
                const uint8_t* texel = data + layer * layout.slicePitch + y * layout.rowPitch + x * 4;
                if (isDepth) {
                    float actualDepth = 0.0f;
                    memcpy(&actualDepth, texel, sizeof(actualDepth));
                    const float expectedDepth = expectedOne ? 1.0f : 0.0f;
                    if (passed && actualDepth != expectedDepth)
                        printf("Clear mismatch layer=%u x=%u y=%u depth=%g expected=%g\n", layer, x, y, actualDepth, expectedDepth);
                    passed &= actualDepth == expectedDepth;
                } else {
                    uint8_t expected = 0;
                    if (expectedOne)
                        expected = isInteger ? 1 : 255;
                    for (uint32_t channel = 0; channel < 4; channel++) {
                        if (passed && texel[channel] != expected)
                            printf("Clear mismatch attachment=%u layer=%u x=%u y=%u channel=%u value=%u expected=%u\n", attachmentIndex, layer, x, y, channel, texel[channel], expected);
                        passed &= texel[channel] == expected;
                    }
                }
            }
        }
    }

    m_Core.UnmapBuffer(buffer);
    return passed;
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

bool CopyTests::TestTextureBarrierAccess() {
    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = nri::Format::RGBA8_UNORM;
    textureDesc.width = 16;
    textureDesc.height = 16;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;

    nri::Texture* texture = nullptr;
    if (!CreateTexture(textureDesc, texture))
        return false;

    const nri::Layout layouts[] = {nri::Layout::GENERAL, nri::Layout::COPY_DESTINATION, nri::Layout::COPY_SOURCE, nri::Layout::SHADER_RESOURCE};
    const nri::StageBits stages[] = {nri::StageBits::ALL, nri::StageBits::COPY, nri::StageBits::FRAGMENT_SHADER, nri::StageBits::NONE};

    nri::TextureBarrierDesc textureBarrier = {};
    textureBarrier.texture = texture;
    textureBarrier.mipNum = 1;
    textureBarrier.layerNum = 1;
    textureBarrier.planes = nri::PlaneBits::COLOR;

    nri::BarrierDesc barrierDesc = {};
    barrierDesc.textures = &textureBarrier;
    barrierDesc.textureNum = 1;

    for (nri::Layout layout : layouts) {
        nri::AccessBits access = nri::AccessBits::COPY_DESTINATION;
        if (layout == nri::Layout::COPY_SOURCE)
            access = nri::AccessBits::COPY_SOURCE;
        else if (layout == nri::Layout::SHADER_RESOURCE)
            access = nri::AccessBits::SHADER_RESOURCE;

        for (nri::StageBits stage : stages) {
            if (!BeginCommandBuffer())
                return false;

            // Keep the requested synchronization even for discarded contents and empty access masks.
            textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, stage};
            textureBarrier.after = {nri::AccessBits::NONE, layout, stage};
            m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

            // NONE scopes cannot have another barrier on the same resource in the same QueueSubmit scope.
            if (stage == nri::StageBits::NONE) {
                if (!SubmitAndWait() || !BeginCommandBuffer())
                    return false;
            } else {
                // Adjacent barriers expose the invalid scope restriction introduced by forcing SYNC_NONE.
                textureBarrier.before = textureBarrier.after;
                m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);
            }

            textureBarrier.before = textureBarrier.after;
            textureBarrier.after = {access, layout, nri::StageBits::ALL};
            m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

            // Preserve ordinary access conversion, and exercise empty access on the after side independently.
            textureBarrier.before = textureBarrier.after;
            textureBarrier.after = {nri::AccessBits::NONE, layout, stage};
            m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

            if (!SubmitAndWait())
                return false;
        }
    }

    return Report("Texture barriers with empty access and defined layouts", true);
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

bool CopyTests::TestHostCopyFormat(nri::Format format, uint32_t blockWidth, uint32_t blockHeight, uint32_t blockSize, bool& uploadPassed, bool& readbackPassed, bool& wholeCopyPassed) {
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
    nri::Texture* copyTexture = nullptr;
    if (!CreateTexture(textureDesc, uploadTexture) || !CreateTexture(textureDesc, readbackTexture) || !CreateTexture(textureDesc, copyTexture))
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

    auto verifyGpuReadbacks = [&](bool& passed) {
        passed = true;
        for (const MipData& data : mipData) {
            const uint8_t* gpuReadbackData = (const uint8_t*)m_Core.MapBuffer(*data.gpuReadbackBuffer, 0, data.gpuLayout.slicePitch);
            if (!gpuReadbackData) {
                printf("FAIL  MapBuffer(gpuReadbackBuffer) returned NULL\n");

                return false;
            }

            for (uint32_t row = 0; row < data.rowNum; row++)
                passed &= memcmp(data.expected.data() + row * data.rowPitch, gpuReadbackData + row * data.gpuLayout.rowPitch, data.rowSize) == 0;
            m_Core.UnmapBuffer(*data.gpuReadbackBuffer);
        }

        return true;
    };

    if (!verifyGpuReadbacks(uploadPassed))
        return false;

    if (!TransitionHostTexture(*copyTexture, mipNum, undefined, hostWrite))
        return false;

    for (nri::UploadHostMemoryToTextureDesc& uploadDesc : uploadDescs)
        uploadDesc.dstTexture = copyTexture;

    RETURN_ON_FAILURE(m_Core.UploadHostMemoryToTexture(*m_Queue, uploadDescs.data(), (uint32_t)uploadDescs.size()));

    if (!TransitionHostTexture(*copyTexture, mipNum, hostWrite, copyDestination) || !BeginCommandBuffer())
        return false;

    std::array<nri::TextureBarrierDesc, mipNum> wholeCopyBarriers = {};
    for (uint32_t mip = 0; mip < mipNum; mip++) {
        wholeCopyBarriers[mip].texture = mipData[mip].gpuReadbackTexture;
        wholeCopyBarriers[mip].before = copySource;
        wholeCopyBarriers[mip].after = copyDestination;
        wholeCopyBarriers[mip].mipNum = 1;
        wholeCopyBarriers[mip].layerNum = 1;
        wholeCopyBarriers[mip].planes = nri::PlaneBits::COLOR;
    }

    barrierDesc.textures = wholeCopyBarriers.data();
    barrierDesc.textureNum = (uint32_t)wholeCopyBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    m_Core.CmdCopyTexture(*m_CommandBuffer, *copyTexture, nullptr, *uploadTexture, nullptr);

    nri::TextureBarrierDesc wholeCopyBarrier = {};
    wholeCopyBarrier.texture = copyTexture;
    wholeCopyBarrier.before = copyDestination;
    wholeCopyBarrier.after = copySource;
    wholeCopyBarrier.mipNum = mipNum;
    wholeCopyBarrier.layerNum = 1;
    wholeCopyBarrier.planes = nri::PlaneBits::COLOR;
    barrierDesc.textures = &wholeCopyBarrier;
    barrierDesc.textureNum = 1;
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    for (uint32_t mip = 0; mip < mipNum; mip++)
        m_Core.CmdCopyTexture(*m_CommandBuffer, *mipData[mip].gpuReadbackTexture, &regions[0], *copyTexture, &regions[mip]);

    barrierDesc.textures = readbackBarriers.data();
    barrierDesc.textureNum = (uint32_t)readbackBarriers.size();
    m_Core.CmdBarrier(*m_CommandBuffer, barrierDesc);

    for (uint32_t mip = 0; mip < mipNum; mip++)
        m_Core.CmdReadbackTextureToBuffer(*m_CommandBuffer, *mipData[mip].gpuReadbackBuffer, mipData[mip].gpuLayout, *mipData[mip].gpuReadbackTexture, regions[0]);

    if (!SubmitAndWait())
        return false;

    if (!verifyGpuReadbacks(wholeCopyPassed))
        return false;

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
        printf("SKIP  RGBA8_UNORM host copies are unsupported\n");

        return true;
    }

    bool uploadPassed = false;
    bool readbackPassed = false;
    bool wholeCopyPassed = false;
    if (!TestHostCopyFormat(nri::Format::RGBA8_UNORM, 1, 1, 4, uploadPassed, readbackPassed, wholeCopyPassed))
        return false;

    const struct {
        nri::Format format;
        uint32_t width, height, size;
        const char* name;
    } compressedFormats[] = {
        {nri::Format::BC1_RGBA_UNORM, 4, 4, 8, "BC1"},
        {nri::Format::ETC2_RGB8_A8_UNORM, 4, 4, 16, "ETC2"},
        {nri::Format::ASTC_5X4_UNORM, 5, 4, 16, "ASTC 5x4"},
    };

    for (const auto& format : compressedFormats) {
        if (!(m_Core.GetFormatSupport(*m_Device, format.format) & nri::FormatSupportBits::HOST_COPY)) {
            printf("SKIP  %s host copies are unsupported\n", format.name);

            continue;
        }

        bool compressedUploadPassed = false;
        bool compressedReadbackPassed = false;
        bool compressedWholeCopyPassed = false;
        if (!TestHostCopyFormat(format.format, format.width, format.height, format.size, compressedUploadPassed, compressedReadbackPassed, compressedWholeCopyPassed))
            return false;

        uploadPassed &= compressedUploadPassed;
        readbackPassed &= compressedReadbackPassed;
        wholeCopyPassed &= compressedWholeCopyPassed;
        Report(format.name, compressedUploadPassed && compressedReadbackPassed && compressedWholeCopyPassed);
    }

    return Report("UploadHostMemoryToTexture", uploadPassed)
        && Report("ReadbackTextureToHostMemory", readbackPassed)
        && Report("CmdCopyTexture (whole mip chain)", wholeCopyPassed);
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
        else if (!strcmp(argv[i], "--api=METAL"))
            settings.graphicsAPI = nri::GraphicsAPI::METAL;
        else if (!strcmp(argv[i], "--api=WGPU"))
            settings.graphicsAPI = nri::GraphicsAPI::WGPU;
        else if (!strcmp(argv[i], "--debugAPI"))
            settings.debugAPI = true;
        else if (!strcmp(argv[i], "--debugNRI"))
            settings.debugNRI = true;
        else if (!strcmp(argv[i], "--legacyBarriers"))
            settings.legacyBarriers = true;
        else if (!strcmp(argv[i], "--queue=COMPUTE"))
            settings.queueType = nri::QueueType::COMPUTE;
        else if (!strcmp(argv[i], "--queue=COPY"))
            settings.queueType = nri::QueueType::COPY;
        else if (!strncmp(argv[i], "--adapter=", 10))
            settings.adapterIndex = (uint32_t)strtoul(argv[i] + 10, nullptr, 10);
    }

    return settings;
}

} // namespace

int main(int argc, char** argv) {
    CopyTests copyTests;
    if (!copyTests.Initialize(ParseSettings(argc, argv)))
        return NRIF_FAIL;

    return copyTests.Run() ? NRIF_OK : NRIF_FAIL;
}
