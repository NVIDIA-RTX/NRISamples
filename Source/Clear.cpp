// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

class Sample : public SampleBase {
public:
    Sample() {
    }

    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI, bool) override;
    void LatencySleep(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

private:
    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocator);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        NRI.DestroyFence(m_FrameFence);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    nri::nriDestroyDevice(m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    // Adapters
    nri::AdapterDesc adapterDesc[2] = {};
    uint32_t adapterDescsNum = helper::GetCountOf(adapterDesc);
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(adapterDesc, adapterDescsNum));

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_COMMANDBUFFER_EMULATION;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &adapterDesc[std::min(m_AdapterIndex, adapterDescsNum - 1)];
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    // NRI
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));

    // Command queue
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));

    // Fences
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Swap chain
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.window = GetWindow();
        swapChainDesc.queue = m_GraphicsQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
        swapChainDesc.flags = (m_Vsync ? nri::SwapChainBits::VSYNC : nri::SwapChainBits::NONE) | nri::SwapChainBits::ALLOW_TEARING;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = GetOptimalSwapChainTextureNum();
        swapChainDesc.queuedFrameNum = GetQueuedFrameNum();
        NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

        uint32_t swapChainTextureNum;
        nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);

        nri::Format swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

        for (uint32_t i = 0; i < swapChainTextureNum; i++) {
            nri::Texture2DViewDesc textureViewDesc = {swapChainTextures[i], nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};

            nri::Descriptor* colorAttachment = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, colorAttachment));

            nri::Fence* acquireSemaphore = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, acquireSemaphore));

            nri::Fence* releaseSemaphore = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, releaseSemaphore));

            SwapChainTexture& swapChainTexture = m_SwapChainTextures.emplace_back();

            swapChainTexture = {};
            swapChainTexture.acquireSemaphore = acquireSemaphore;
            swapChainTexture.releaseSemaphore = releaseSemaphore;
            swapChainTexture.texture = swapChainTextures[i];
            swapChainTexture.colorAttachment = colorAttachment;
            swapChainTexture.attachmentFormat = swapChainFormat;
        }
    }

    // Queued frames
    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    return true;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Record
    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, nullptr);
    {
        nri::TextureBarrierDesc textureBarriers = {};
        textureBarriers.texture = swapChainTexture.texture;
        textureBarriers.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        textureBarriers.layerNum = 1;
        textureBarriers.mipNum = 1;

        nri::BarrierGroupDesc barrierGroupDesc = {};
        barrierGroupDesc.textureNum = 1;
        barrierGroupDesc.textures = &textureBarriers;
        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);

        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &swapChainTexture.colorAttachment;

        NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
        {
            helper::Annotation annotation(NRI, commandBuffer, "Clear");

            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;
            clearDesc.planes = nri::PlaneBits::COLOR;

            nri::Dim_t w = (nri::Dim_t)GetWindowResolution().x;
            nri::Dim_t h = (nri::Dim_t)GetWindowResolution().y;
            nri::Dim_t h3 = h / 3;
            int16_t y = (int16_t)h3;

            clearDesc.value.color.f = {1.0f, 0.0f, 0.0f, 1.0f};
            nri::Rect rect1 = {0, 0, w, h3};
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect1, 1);

            clearDesc.value.color.f = {0.0f, 1.0f, 0.0f, 1.0f};
            nri::Rect rect2 = {0, y, w, h3};
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect2, 1);

            clearDesc.value.color.f = {0.0f, 0.0f, 1.0f, 1.0f};
            nri::Rect rect3 = {0, static_cast<int16_t>(y * 2), w, h3};
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect3, 1);
        }
        NRI.CmdEndRendering(commandBuffer);

        textureBarriers.before = textureBarriers.after;
        textureBarriers.after = {nri::AccessBits::NONE, nri::Layout::PRESENT};

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
    }
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::FenceSubmitDesc textureAcquiredFence = {};
        textureAcquiredFence.fence = swapChainAcquireSemaphore;
        textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

        nri::FenceSubmitDesc renderingFinishedFence = {};
        renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.waitFences = &textureAcquiredFence;
        queueSubmitDesc.waitFenceNum = 1;
        queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = &renderingFinishedFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    // Present
    NRI.QueuePresent(*m_SwapChain, *swapChainTexture.releaseSemaphore);

    { // Signaling after "Present" improves D3D11 performance a bit
        nri::FenceSubmitDesc signalFence = {};
        signalFence.fence = m_FrameFence;
        signalFence.value = 1 + frameIndex;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.signalFences = &signalFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }
}

SAMPLE_MAIN(Sample, 0);
