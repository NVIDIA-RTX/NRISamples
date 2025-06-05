// © 2021 NVIDIA Corporation

#if _WIN32
#    include <windows.h>
#endif

#include "NRIFramework.h"

#include <array>
#include <atomic>
#include <thread>

// Found in sse2neon
// _mm_pause is already defined in sse2neon.h for ARM platforms
#if !(defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM))
#    include <xmmintrin.h>
#endif

constexpr uint32_t BOX_NUM = 30000;
constexpr uint32_t DRAW_CALLS_PER_PIPELINE = 4;
constexpr size_t QUEUED_FRAME_MAX_NUM = 4;
constexpr size_t THREAD_MAX_NUM = 64;

constexpr uint32_t HALT = 0;
constexpr uint32_t GO = 1;
constexpr uint32_t STOP = 2;

struct Vertex {
    float position[3];
    float texCoords[2];
};

struct Box {
    uint32_t dynamicConstantBufferOffset;
    nri::DescriptorSet* descriptorSet;
    nri::Pipeline* pipeline;
};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;

    // Used by the main thread only
    nri::CommandBuffer* commandBufferPre;
    nri::CommandBuffer* commandBufferPost;
};

struct ThreadContext {
    std::array<QueuedFrame, QUEUED_FRAME_MAX_NUM> queuedFrames;
    std::thread thread;
    std::atomic_uint32_t control;
};

class Sample : public SampleBase {
public:
    inline Sample() {
    }

    ~Sample();

private:
    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipeline(nri::Format swapChainFormat);
    void CreateTextures();
    void CreateDepthTexture();
    void CreateVertexBuffer();
    void CreateDescriptorPool();
    void CreateTransformConstantBuffer();
    void CreateDescriptorSets();
    void CreateFakeConstantBuffers();
    void CreateViewConstantBuffer();
    void RenderBoxes(nri::CommandBuffer& commandBuffer, uint32_t offset, uint32_t number);
    void ThreadEntryPoint(uint32_t threadIndex);
    void SetupProjViewMatrix(float4x4& projViewMatrix);

private:
    std::array<ThreadContext, THREAD_MAX_NUM> m_ThreadContexts = {};
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::Descriptor*> m_TextureViews;
    std::vector<nri::Descriptor*> m_FakeConstantBufferViews;
    std::vector<Box> m_Boxes;
    std::vector<SwapChainTexture> m_SwapChainTextures;
    std::vector<nri::Memory*> m_MemoryAllocations;
    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::Texture* m_DepthTexture = nullptr;
    nri::Descriptor* m_DepthTextureView = nullptr;
    nri::Descriptor* m_TransformConstantBufferView = nullptr;
    nri::Descriptor* m_ViewConstantBufferView = nullptr;
    nri::Descriptor* m_Sampler = nullptr;
    nri::DescriptorSet* m_DescriptorSetWithSharedSampler = nullptr;
    nri::Buffer* m_VertexBuffer = nullptr;
    nri::Buffer* m_IndexBuffer = nullptr;
    nri::Buffer* m_TransformConstantBuffer = nullptr;
    nri::Buffer* m_ViewConstantBuffer = nullptr;
    nri::Buffer* m_FakeConstantBuffer = nullptr;
    const SwapChainTexture* m_BackBuffer = nullptr;
    double m_FrameTime = 0.0;
    nri::Format m_DepthFormat = nri::Format::UNKNOWN;
    uint32_t m_ThreadNum = 0;
    uint32_t m_FrameIndex = 0;
    uint32_t m_BoxesPerThread = 0;
    uint32_t m_IndexNum = 0;
    bool m_MultiThreading = true;
    bool m_MultiSubmit = false;

    std::atomic_uint32_t m_ReadyCount;
};

Sample::~Sample() {
    for (size_t i = 1; m_MultiThreading && i < m_ThreadNum; i++) {
        ThreadContext& threadContext = m_ThreadContexts[i];
        threadContext.control.store(STOP);
        threadContext.thread.join();
    }

    if (NRI.HasHelper())
        NRI.WaitForIdle(*m_GraphicsQueue);

    if (NRI.HasCore()) {
        for (uint32_t i = 0; i < m_ThreadNum; i++) {
            ThreadContext& threadContext = m_ThreadContexts[i];

            for (uint32_t j = 0; j < GetQueuedFrameNum(); j++) {
                QueuedFrame& queuedFrame = threadContext.queuedFrames[j];

                NRI.DestroyCommandBuffer(*queuedFrame.commandBuffer);
                NRI.DestroyCommandBuffer(*queuedFrame.commandBufferPre);
                NRI.DestroyCommandBuffer(*queuedFrame.commandBufferPost);
                NRI.DestroyCommandAllocator(*queuedFrame.commandAllocator);
            }
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(*swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(*swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(*swapChainTexture.colorAttachment);
        }

        for (size_t i = 0; i < m_Textures.size(); i++)
            NRI.DestroyDescriptor(*m_TextureViews[i]);

        for (size_t i = 0; i < m_Textures.size(); i++)
            NRI.DestroyTexture(*m_Textures[i]);

        for (size_t i = 0; i < m_FakeConstantBufferViews.size(); i++)
            NRI.DestroyDescriptor(*m_FakeConstantBufferViews[i]);

        for (size_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(*m_Pipelines[i]);

        NRI.DestroyDescriptor(*m_Sampler);
        NRI.DestroyDescriptor(*m_DepthTextureView);
        NRI.DestroyDescriptor(*m_TransformConstantBufferView);
        NRI.DestroyDescriptor(*m_ViewConstantBufferView);
        NRI.DestroyTexture(*m_DepthTexture);
        NRI.DestroyBuffer(*m_TransformConstantBuffer);
        NRI.DestroyBuffer(*m_ViewConstantBuffer);
        NRI.DestroyBuffer(*m_FakeConstantBuffer);
        NRI.DestroyBuffer(*m_VertexBuffer);
        NRI.DestroyBuffer(*m_IndexBuffer);
        NRI.DestroyPipelineLayout(*m_PipelineLayout);
        NRI.DestroyDescriptorPool(*m_DescriptorPool);
        NRI.DestroyFence(*m_FrameFence);

        for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
            NRI.FreeMemory(*m_MemoryAllocations[i]);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(*m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(*m_Streamer);

    DestroyImgui();

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
    uint32_t concurrentThreadMaxNum = std::thread::hardware_concurrency();
    m_ThreadNum = std::min((concurrentThreadMaxNum * 3) / 4, (uint32_t)THREAD_MAX_NUM);

    for (ThreadContext& threadContext : m_ThreadContexts)
        threadContext.control.store(HALT, std::memory_order_relaxed);

    m_Boxes.resize(std::max(BOX_NUM, m_ThreadNum));
    m_BoxesPerThread = (uint32_t)m_Boxes.size() / (uint32_t)m_ThreadNum;

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
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));

    // Create streamer
    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferUsageBits = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.queuedFrameNum = GetQueuedFrameNum();
    NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    m_DepthFormat = nri::GetSupportedDepthFormat(NRI, *m_Device, 24, false);
    nri::Format swapChainFormat = CreateSwapChain();

    CreateCommandBuffers();
    CreateDepthTexture();
    CreatePipeline(swapChainFormat);
    CreateTextures();
    CreateFakeConstantBuffers();
    CreateViewConstantBuffer();
    CreateVertexBuffer();
    CreateDescriptorPool();
    CreateTransformConstantBuffer();
    CreateDescriptorSets();

    if (m_MultiThreading) {
        for (uint32_t i = 1; i < m_ThreadNum; i++)
            m_ThreadContexts[i].thread = std::thread(&Sample::ThreadEntryPoint, this, i);
    }

    return InitImgui(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);

    uint32_t threadNum = m_MultiThreading ? m_ThreadNum : 1;
    for (uint32_t i = 0; i < threadNum; i++) {
        ThreadContext& threadContext = m_ThreadContexts[i];
        NRI.ResetCommandAllocator(*threadContext.queuedFrames[queuedFrameIndex].commandAllocator);
    }
}

void Sample::PrepareFrame(uint32_t) {
    bool multiThreadingPrev = m_MultiThreading;

    ImGui::NewFrame();
    {
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoResize);
        {
            ImGui::Text("Box number: %u", (uint32_t)m_Boxes.size());
            ImGui::Text("Draw calls per pipeline: %u", DRAW_CALLS_PER_PIPELINE);
            ImGui::Text("Frame time: %.2f ms", m_FrameTime);
            ImGui::Checkbox("Multi-threading", &m_MultiThreading);
            ImGui::Checkbox("Multi-submit", &m_MultiSubmit);
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();

    if (m_MultiThreading != multiThreadingPrev) {
        if (m_MultiThreading) {
            for (uint32_t i = 1; i < m_ThreadNum; i++) {
                ThreadContext& threadContext = m_ThreadContexts[i];
                threadContext.control.store(HALT);
                threadContext.thread = std::thread(&Sample::ThreadEntryPoint, this, i);
            }
        } else {
            for (size_t i = 1; i < m_ThreadNum; i++) {
                ThreadContext& threadContext = m_ThreadContexts[i];
                threadContext.control.store(STOP);
                threadContext.thread.join();
            }
        }
    }
}

void Sample::RenderFrame(uint32_t frameIndex) {
    ThreadContext& context0 = m_ThreadContexts[0];

    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = context0.queuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    m_BackBuffer = &m_SwapChainTextures[currentSwapChainTextureIndex];
    m_FrameIndex = frameIndex;

    m_FrameTime = m_Timer.GetTimeStamp();

    { // Record pre
        nri::CommandBuffer& commandBufferPre = *queuedFrame.commandBufferPre;
        NRI.BeginCommandBuffer(commandBufferPre, m_DescriptorPool);
        {
            helper::Annotation annotation(NRI, commandBufferPre, "Pre");

            nri::TextureBarrierDesc swapChainTextureTransition = {};
            swapChainTextureTransition.texture = m_BackBuffer->texture;
            swapChainTextureTransition.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};

            nri::BarrierGroupDesc barrierGroupDesc = {};
            barrierGroupDesc.textures = &swapChainTextureTransition;
            barrierGroupDesc.textureNum = 1;

            NRI.CmdBarrier(commandBufferPre, barrierGroupDesc);

            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &m_BackBuffer->colorAttachment;
            attachmentsDesc.depthStencil = m_DepthTextureView;

            NRI.CmdBeginRendering(commandBufferPre, attachmentsDesc);
            {
                nri::ClearDesc clearDescs[2] = {};
                clearDescs[0].planes = nri::PlaneBits::COLOR;
                clearDescs[1].planes = nri::PlaneBits::DEPTH;
                clearDescs[1].value.depthStencil.depth = 1.0f;

                NRI.CmdClearAttachments(commandBufferPre, clearDescs, helper::GetCountOf(clearDescs), nullptr, 0);
            }
            NRI.CmdEndRendering(commandBufferPre);
        }
        NRI.EndCommandBuffer(commandBufferPre);

        // Submit pre
        if (m_MultiSubmit) {
            nri::FenceSubmitDesc textureAcquiredFence = {};
            textureAcquiredFence.fence = swapChainAcquireSemaphore;
            textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

            nri::QueueSubmitDesc queueSubmitDesc = {};
            queueSubmitDesc.waitFences = &textureAcquiredFence;
            queueSubmitDesc.waitFenceNum = 1;
            queueSubmitDesc.commandBuffers = &queuedFrame.commandBufferPre;
            queueSubmitDesc.commandBufferNum = 1;

            NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
        }
    }

    // Pass "GO" to workers
    if (m_MultiThreading) {
        m_ReadyCount.store(0, std::memory_order_seq_cst);

        for (uint32_t i = 1; i < m_ThreadNum; i++) {
            ThreadContext& threadContext = m_ThreadContexts[i];
            threadContext.control.store(GO, std::memory_order_relaxed);
        }
    }

    { // Record
        nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
        NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
        {
            helper::Annotation annotation(NRI, commandBuffer, "Render boxes");

            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &m_BackBuffer->colorAttachment;
            attachmentsDesc.depthStencil = m_DepthTextureView;

            NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
            {
                uint32_t boxNum = m_MultiThreading ? m_BoxesPerThread : (uint32_t)m_Boxes.size();

                RenderBoxes(commandBuffer, 0, boxNum);
            }
            NRI.CmdEndRendering(commandBuffer);
        }
        NRI.EndCommandBuffer(commandBuffer);

        // Submit
        if (m_MultiSubmit) {
            nri::QueueSubmitDesc queueSubmitDesc = {};
            queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
            queueSubmitDesc.commandBufferNum = 1;

            NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
        }
    }

    // Wait for completion
    if (m_MultiThreading) {
        while (m_ReadyCount.load(std::memory_order_relaxed) != m_ThreadNum - 1)
            _mm_pause();
    }

    { // Record post
        nri::CommandBuffer& commandBufferPost = *queuedFrame.commandBufferPost;
        NRI.BeginCommandBuffer(commandBufferPost, m_DescriptorPool);
        {
            helper::Annotation annotation(NRI, commandBufferPost, "Post");

            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &m_BackBuffer->colorAttachment;

            NRI.CmdBeginRendering(commandBufferPost, attachmentsDesc);
            {
                RenderImgui(commandBufferPost, *m_Streamer, m_BackBuffer->attachmentFormat, 1.0f, true);
            }
            NRI.CmdEndRendering(commandBufferPost);

            nri::TextureBarrierDesc swapChainTextureTransition = {};
            swapChainTextureTransition.texture = m_BackBuffer->texture;
            swapChainTextureTransition.before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
            swapChainTextureTransition.after = {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT};
            swapChainTextureTransition.layerNum = 1;
            swapChainTextureTransition.mipNum = 1;

            nri::BarrierGroupDesc barrierGroupDesc = {};
            barrierGroupDesc.textures = &swapChainTextureTransition;
            barrierGroupDesc.textureNum = 1;

            NRI.CmdBarrier(commandBufferPost, barrierGroupDesc);
        }
        NRI.EndCommandBuffer(commandBufferPost);

        // Submit post
        if (m_MultiSubmit) {
            nri::FenceSubmitDesc renderingFinishedFence = {};
            renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

            nri::QueueSubmitDesc queueSubmitDesc = {};
            queueSubmitDesc.commandBuffers = &queuedFrame.commandBufferPost;
            queueSubmitDesc.commandBufferNum = 1;
            queueSubmitDesc.signalFences = &renderingFinishedFence;
            queueSubmitDesc.signalFenceNum = 1;

            NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
        }
    }

    // Submit all
    if (!m_MultiSubmit) {
        uint32_t threadNum = m_MultiThreading ? (uint32_t)m_ThreadNum : 1;
        nri::CommandBuffer* commandBuffers[THREAD_MAX_NUM + 2] = {};

        commandBuffers[0] = queuedFrame.commandBufferPre;
        commandBuffers[1 + threadNum] = queuedFrame.commandBufferPost;
        for (uint32_t i = 0; i < threadNum; i++)
            commandBuffers[1 + i] = m_ThreadContexts[i].queuedFrames[queuedFrameIndex].commandBuffer;

        nri::FenceSubmitDesc textureAcquiredFence = {};
        textureAcquiredFence.fence = swapChainAcquireSemaphore;
        textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

        nri::FenceSubmitDesc renderingFinishedFence = {};
        renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.waitFences = &textureAcquiredFence;
        queueSubmitDesc.waitFenceNum = 1;
        queueSubmitDesc.commandBuffers = commandBuffers;
        queueSubmitDesc.commandBufferNum = threadNum + 2;
        queueSubmitDesc.signalFences = &renderingFinishedFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    m_FrameTime = m_Timer.GetTimeStamp() - m_FrameTime;

    NRI.EndStreamerFrame(*m_Streamer);

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

void Sample::RenderBoxes(nri::CommandBuffer& commandBuffer, uint32_t offset, uint32_t number) {
    helper::Annotation annotation(NRI, commandBuffer, "RenderBoxes");

    const nri::Rect scissorRect = {0, 0, (nri::Dim_t)GetWindowResolution().x, (nri::Dim_t)GetWindowResolution().y};
    const nri::Viewport viewport = {0.0f, 0.0f, (float)scissorRect.width, (float)scissorRect.height, 0.0f, 1.0f};
    NRI.CmdSetViewports(commandBuffer, &viewport, 1);
    NRI.CmdSetScissors(commandBuffer, &scissorRect, 1);
    NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);

    nri::VertexBufferDesc vertexBufferDesc = {};
    vertexBufferDesc.buffer = m_VertexBuffer;
    vertexBufferDesc.offset = 0;
    vertexBufferDesc.stride = sizeof(Vertex);

    for (uint32_t i = 0; i < number; i++) {
        const Box& box = m_Boxes[offset + i];

        NRI.CmdSetPipeline(commandBuffer, *box.pipeline);
        NRI.CmdSetDescriptorSet(commandBuffer, 0, *box.descriptorSet, &box.dynamicConstantBufferOffset);
        NRI.CmdSetDescriptorSet(commandBuffer, 1, *m_DescriptorSetWithSharedSampler, nullptr);
        NRI.CmdSetIndexBuffer(commandBuffer, *m_IndexBuffer, 0, nri::IndexType::UINT16);
        NRI.CmdSetVertexBuffers(commandBuffer, 0, &vertexBufferDesc, 1);

        NRI.CmdDrawIndexed(commandBuffer, {m_IndexNum, 1, 0, 0, 0});
    }
}

void Sample::ThreadEntryPoint(uint32_t threadIndex) {
    ThreadContext& threadContext = m_ThreadContexts[threadIndex];

    while (true) {
        uint32_t control = threadContext.control.load(std::memory_order_relaxed);
        if (control == HALT)
            continue;
        else if (control == STOP)
            break;

        threadContext.control.store(HALT, std::memory_order_seq_cst);

        uint32_t queuedFrameIndex = m_FrameIndex % GetQueuedFrameNum();
        nri::CommandBuffer& commandBuffer = *threadContext.queuedFrames[queuedFrameIndex].commandBuffer;

        // Record
        NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
        {
            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &m_BackBuffer->colorAttachment;
            attachmentsDesc.depthStencil = m_DepthTextureView;

            NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
            {
                uint32_t baseBoxIndex = threadIndex * m_BoxesPerThread;
                uint32_t boxNum = std::min(m_BoxesPerThread, (uint32_t)m_Boxes.size() - baseBoxIndex);

                RenderBoxes(commandBuffer, baseBoxIndex, boxNum);
            }
            NRI.CmdEndRendering(commandBuffer);
        }
        NRI.EndCommandBuffer(commandBuffer);

        // Submit
        if (m_MultiSubmit) {
            nri::QueueSubmitDesc queueSubmitDesc = {};
            queueSubmitDesc.commandBuffers = &threadContext.queuedFrames[queuedFrameIndex].commandBuffer;
            queueSubmitDesc.commandBufferNum = 1;

            NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
        }

        // Signal "done" and stay in "HALT" mode (wait for instructions from the main thread)
        m_ReadyCount.fetch_add(1, std::memory_order_release);
    }
}

nri::Format Sample::CreateSwapChain() {
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

    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);

    nri::Format swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

    m_SwapChainTextures.clear();
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

    return swapChainFormat;
}

void Sample::CreateCommandBuffers() {
    for (uint32_t i = 0; i < m_ThreadNum; i++) {
        ThreadContext& threadContext = m_ThreadContexts[i];

        for (uint32_t j = 0; j < GetQueuedFrameNum(); j++) {
            QueuedFrame& queuedFrame = threadContext.queuedFrames[j];

            NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
            NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));

            if (i == 0) {
                NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBufferPre));
                NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBufferPost));
            }
        }
    }
}

void Sample::CreatePipeline(nri::Format swapChainFormat) {
    nri::DescriptorRangeDesc descriptorRanges0[] = {
        {1, 3, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::ALL},
        {0, 3, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER}};

    nri::DescriptorRangeDesc descriptorRanges1[] = {
        {0, 1, nri::DescriptorType::SAMPLER, nri::StageBits::FRAGMENT_SHADER}};

    nri::SamplerDesc samplerDesc = {};
    samplerDesc.addressModes = {nri::AddressMode::MIRRORED_REPEAT, nri::AddressMode::MIRRORED_REPEAT};
    samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
    samplerDesc.anisotropy = 4;
    samplerDesc.mipMax = 16.0f;

    NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, m_Sampler));

    nri::DynamicConstantBufferDesc dynamicConstantBufferDesc = {0, nri::StageBits::VERTEX_SHADER};

    nri::DescriptorSetDesc descriptorSetDescs[] = {
        {0, descriptorRanges0, helper::GetCountOf(descriptorRanges0), &dynamicConstantBufferDesc, 1},
        {1, descriptorRanges1, helper::GetCountOf(descriptorRanges1)},
    };

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
    pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
    pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;

    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

    constexpr uint32_t pipelineNum = 8;

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    utils::ShaderCodeStorage shaderCodeStorage;

    nri::ShaderDesc shaders[1 + pipelineNum];
    shaders[0] = utils::LoadShader(deviceDesc.graphicsAPI, "Box.vs", shaderCodeStorage);
    for (uint32_t i = 0; i < pipelineNum; i++)
        shaders[1 + i] = utils::LoadShader(deviceDesc.graphicsAPI, "Box" + std::to_string(i) + ".fs", shaderCodeStorage);

    nri::VertexStreamDesc vertexStreamDesc = {};
    vertexStreamDesc.bindingSlot = 0;

    nri::VertexAttributeDesc vertexAttributeDesc[2] = {
        {
            {"POSITION", 0},
            {0},
            helper::GetOffsetOf(&Vertex::position),
            nri::Format::RGB32_SFLOAT,
        },
        {
            {"TEXCOORD", 0},
            {1},
            helper::GetOffsetOf(&Vertex::texCoords),
            nri::Format::RG32_SFLOAT,
        }};

    nri::VertexInputDesc vertexInputDesc = {};
    vertexInputDesc.attributes = vertexAttributeDesc;
    vertexInputDesc.attributeNum = (uint8_t)helper::GetCountOf(vertexAttributeDesc);
    vertexInputDesc.streams = &vertexStreamDesc;
    vertexInputDesc.streamNum = 1;

    nri::InputAssemblyDesc inputAssemblyDesc = {};
    inputAssemblyDesc.topology = nri::Topology::TRIANGLE_LIST;

    nri::RasterizationDesc rasterizationDesc = {};
    rasterizationDesc.fillMode = nri::FillMode::SOLID;
    rasterizationDesc.cullMode = nri::CullMode::NONE;

    nri::ColorAttachmentDesc colorAttachmentDesc = {};
    colorAttachmentDesc.format = swapChainFormat;
    colorAttachmentDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

    nri::DepthAttachmentDesc depthAttachmentDesc = {};
    depthAttachmentDesc.compareFunc = nri::CompareFunc::LESS;
    depthAttachmentDesc.write = true;

    nri::OutputMergerDesc outputMergerDesc = {};
    outputMergerDesc.colors = &colorAttachmentDesc;
    outputMergerDesc.colorNum = 1;

    outputMergerDesc.depthStencilFormat = m_DepthFormat;
    outputMergerDesc.depth.compareFunc = nri::CompareFunc::LESS;
    outputMergerDesc.depth.write = true;

    nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
    graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
    graphicsPipelineDesc.vertexInput = &vertexInputDesc;
    graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
    graphicsPipelineDesc.rasterization = rasterizationDesc;
    graphicsPipelineDesc.outputMerger = outputMergerDesc;

    m_Pipelines.resize(pipelineNum);

    for (size_t i = 0; i < m_Pipelines.size(); i++) {
        nri::ShaderDesc shaderStages[] = {shaders[0], shaders[1 + i]};
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);

        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_Pipelines[i]));
    }
}

void Sample::CreateDepthTexture() {
    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT;
    textureDesc.format = m_DepthFormat;
    textureDesc.width = (uint16_t)GetWindowResolution().x;
    textureDesc.height = (uint16_t)GetWindowResolution().y;
    textureDesc.mipNum = 1;

    NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_DepthTexture));

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.textureNum = 1;
    resourceGroupDesc.textures = &m_DepthTexture;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::Texture2DViewDesc texture2DViewDesc = {m_DepthTexture, nri::Texture2DViewType::DEPTH_STENCIL_ATTACHMENT, m_DepthFormat};
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_DepthTextureView));

    nri::TextureUploadDesc textureData = {};
    textureData.texture = m_DepthTexture;
    textureData.after = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_WRITE, nri::Layout::DEPTH_STENCIL_ATTACHMENT};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureData, 1, nullptr, 0));
}

void Sample::CreateVertexBuffer() {
    const float boxHalfSize = 0.5f;

    std::vector<Vertex> vertices{
        {{-boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 0.0f}},
        {{-boxHalfSize, -boxHalfSize, boxHalfSize}, {4.0f, 0.0f}},
        {{-boxHalfSize, boxHalfSize, -boxHalfSize}, {0.0f, 4.0f}},
        {{-boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 4.0f}},
        {{boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 0.0f}},
        {{boxHalfSize, -boxHalfSize, boxHalfSize}, {4.0f, 0.0f}},
        {{boxHalfSize, boxHalfSize, -boxHalfSize}, {0.0f, 4.0f}},
        {{boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 4.0f}},
        {{-boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 0.0f}},
        {{-boxHalfSize, -boxHalfSize, boxHalfSize}, {4.0f, 0.0f}},
        {{boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 4.0f}},
        {{boxHalfSize, -boxHalfSize, boxHalfSize}, {4.0f, 4.0f}},
        {{-boxHalfSize, boxHalfSize, -boxHalfSize}, {0.0f, 0.0f}},
        {{-boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 0.0f}},
        {{boxHalfSize, boxHalfSize, -boxHalfSize}, {0.0f, 4.0f}},
        {{boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 4.0f}},
        {{-boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 0.0f}},
        {{-boxHalfSize, boxHalfSize, -boxHalfSize}, {4.0f, 0.0f}},
        {{boxHalfSize, -boxHalfSize, -boxHalfSize}, {0.0f, 4.0f}},
        {{boxHalfSize, boxHalfSize, -boxHalfSize}, {4.0f, 4.0f}},
        {{-boxHalfSize, -boxHalfSize, boxHalfSize}, {0.0f, 0.0f}},
        {{-boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 0.0f}},
        {{boxHalfSize, -boxHalfSize, boxHalfSize}, {0.0f, 4.0f}},
        {{boxHalfSize, boxHalfSize, boxHalfSize}, {4.0f, 4.0f}},
    };

    std::vector<uint16_t> indices{
        0, 1, 2, 1, 2, 3,
        4, 5, 6, 5, 6, 7,
        8, 9, 10, 9, 10, 11,
        12, 13, 14, 13, 14, 15,
        16, 17, 18, 17, 18, 19,
        20, 21, 22, 21, 22, 23};

    m_IndexNum = (uint32_t)indices.size();

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = helper::GetByteSizeOf(vertices);
    bufferDesc.usage = nri::BufferUsageBits::VERTEX_BUFFER;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_VertexBuffer));

    bufferDesc.size = helper::GetByteSizeOf(indices);
    bufferDesc.usage = nri::BufferUsageBits::INDEX_BUFFER;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_IndexBuffer));

    nri::Buffer* const buffers[] = {m_VertexBuffer, m_IndexBuffer};

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = helper::GetCountOf(buffers);
    resourceGroupDesc.buffers = buffers;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::BufferUploadDesc vertexBufferUpdate = {};
    vertexBufferUpdate.buffer = m_VertexBuffer;
    vertexBufferUpdate.data = vertices.data();
    vertexBufferUpdate.after = {nri::AccessBits::VERTEX_BUFFER};

    nri::BufferUploadDesc indexBufferUpdate = {};
    indexBufferUpdate.buffer = m_IndexBuffer;
    indexBufferUpdate.data = indices.data();
    indexBufferUpdate.after = {nri::AccessBits::INDEX_BUFFER};

    const nri::BufferUploadDesc bufferUpdates[] = {vertexBufferUpdate, indexBufferUpdate};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, bufferUpdates, helper::GetCountOf(bufferUpdates)));
}

void Sample::CreateTransformConstantBuffer() {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    uint32_t matrixSize = uint32_t(sizeof(float4x4));
    uint32_t alignedMatrixSize = helper::Align(matrixSize, deviceDesc.memoryAlignment.constantBufferOffset);

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = m_Boxes.size() * alignedMatrixSize;
    bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_TransformConstantBuffer));

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_TransformConstantBuffer;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::BufferViewDesc constantBufferViewDesc = {};
    constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
    constantBufferViewDesc.buffer = m_TransformConstantBuffer;
    constantBufferViewDesc.size = alignedMatrixSize;
    NRI.CreateBufferView(constantBufferViewDesc, m_TransformConstantBufferView);

    uint32_t dynamicConstantBufferOffset = 0;

    std::vector<uint8_t> bufferContent((size_t)bufferDesc.size, 0);
    uint8_t* bufferContentRange = bufferContent.data();

    constexpr uint32_t lineSize = 17;

    for (size_t i = 0; i < m_Boxes.size(); i++) {
        Box& box = m_Boxes[i];

        float4x4& matrix = *(float4x4*)(bufferContentRange + dynamicConstantBufferOffset);
        matrix = float4x4::Identity();

        const size_t x = i % lineSize;
        const size_t y = i / lineSize;
        matrix.PreTranslation(float3(-1.35f * 0.5f * (lineSize - 1) + 1.35f * x, 8.0f + 1.25f * y, 0.0f));
        matrix.AddScale(float3(1.0f + 0.0001f * (rand() % 2001)));

        box.dynamicConstantBufferOffset = dynamicConstantBufferOffset;
        dynamicConstantBufferOffset += alignedMatrixSize;
    }

    nri::BufferUploadDesc bufferUpdate = {};
    bufferUpdate.buffer = m_TransformConstantBuffer;
    bufferUpdate.data = bufferContent.data();
    bufferUpdate.after = {nri::AccessBits::CONSTANT_BUFFER};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, &bufferUpdate, 1));
}

void Sample::CreateDescriptorSets() {
    { // DescriptorSet 0 (per box)
        std::vector<nri::DescriptorSet*> descriptorSets(m_Boxes.size());
        NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, descriptorSets.data(), (uint32_t)descriptorSets.size(), 0);

        for (size_t i = 0; i < m_Boxes.size(); i++) {
            Box& box = m_Boxes[i];

            nri::Descriptor* constantBuffers[] = {
                m_FakeConstantBufferViews[0],
                m_ViewConstantBufferView,
                m_FakeConstantBufferViews[rand() % m_FakeConstantBufferViews.size()],
            };

            const nri::Descriptor* textureViews[3] = {};
            for (size_t j = 0; j < helper::GetCountOf(textureViews); j++)
                textureViews[j] = m_TextureViews[rand() % m_TextureViews.size()];

            const nri::DescriptorRangeUpdateDesc rangeUpdates[] = {
                {constantBuffers, helper::GetCountOf(constantBuffers)},
                {textureViews, helper::GetCountOf(textureViews)}};

            box.pipeline = m_Pipelines[(i / DRAW_CALLS_PER_PIPELINE) % m_Pipelines.size()];
            box.descriptorSet = descriptorSets[i];

            NRI.UpdateDescriptorRanges(*box.descriptorSet, 0, helper::GetCountOf(rangeUpdates), rangeUpdates);
            NRI.UpdateDynamicConstantBuffers(*box.descriptorSet, 0, 1, &m_TransformConstantBufferView);
        }
    }

    { // DescriptorSet 1 (shared)
        const nri::DescriptorRangeUpdateDesc rangeUpdates[] = {
            {&m_Sampler, 1}};

        NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &m_DescriptorSetWithSharedSampler, 1, 0);
        NRI.UpdateDescriptorRanges(*m_DescriptorSetWithSharedSampler, 0, helper::GetCountOf(rangeUpdates), rangeUpdates);
    }
}

void Sample::CreateDescriptorPool() {
    uint32_t boxNum = (uint32_t)m_Boxes.size();

    nri::DescriptorPoolDesc descriptorPoolDesc = {};
    descriptorPoolDesc.constantBufferMaxNum = 3 * boxNum;
    descriptorPoolDesc.dynamicConstantBufferMaxNum = 1 * boxNum;
    descriptorPoolDesc.textureMaxNum = 3 * boxNum;
    descriptorPoolDesc.descriptorSetMaxNum = boxNum + 1;
    descriptorPoolDesc.samplerMaxNum = 1;

    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
}

void Sample::CreateTextures() {
    constexpr uint32_t textureNum = 8;

    std::vector<utils::Texture> loadedTextures(textureNum);
    std::string texturePath = utils::GetFullPath("", utils::DataFolder::TEXTURES);

    for (uint32_t i = 0; i < loadedTextures.size(); i++) {
        if (!utils::LoadTexture(texturePath + "checkerboard" + std::to_string(i) + ".dds", loadedTextures[i]))
            std::abort();
    }

    uint32_t textureVariations = 1024;

    m_Textures.resize(textureVariations);
    for (size_t i = 0; i < m_Textures.size(); i++) {
        const utils::Texture& texture = loadedTextures[i % textureNum];

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        textureDesc.format = texture.GetFormat();
        textureDesc.width = texture.GetWidth();
        textureDesc.height = texture.GetHeight();
        textureDesc.mipNum = texture.GetMipNum();

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_Textures[i]));
    }

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.textureNum = (uint32_t)m_Textures.size();
    resourceGroupDesc.textures = m_Textures.data();

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    constexpr uint32_t MAX_MIP_NUM = 16;
    std::vector<nri::TextureUploadDesc> textureUpdates(m_Textures.size());
    std::vector<nri::TextureSubresourceUploadDesc> subresources(m_Textures.size() * MAX_MIP_NUM);

    for (size_t i = 0; i < textureUpdates.size(); i++) {
        const size_t subresourceOffset = MAX_MIP_NUM * i;
        const utils::Texture& texture = loadedTextures[i % textureNum];

        for (uint32_t mip = 0; mip < texture.GetMipNum(); mip++)
            texture.GetSubresource(subresources[subresourceOffset + mip], mip);

        nri::TextureUploadDesc& textureUpdate = textureUpdates[i];
        textureUpdate.subresources = &subresources[subresourceOffset];
        textureUpdate.texture = m_Textures[i];
        textureUpdate.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};
    }

    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, textureUpdates.data(), (uint32_t)textureUpdates.size(), nullptr, 0));

    m_TextureViews.resize(m_Textures.size());
    for (size_t i = 0; i < m_Textures.size(); i++) {
        const utils::Texture& texture = loadedTextures[i % textureNum];

        nri::Texture2DViewDesc texture2DViewDesc = {m_Textures[i], nri::Texture2DViewType::SHADER_RESOURCE_2D, texture.GetFormat()};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_TextureViews[i]));
    }
}

void Sample::CreateFakeConstantBuffers() {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    uint32_t constantRangeSize = (uint32_t)helper::Align(sizeof(float4), deviceDesc.memoryAlignment.constantBufferOffset);
    constexpr uint32_t fakeConstantBufferRangeNum = 16384;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = fakeConstantBufferRangeNum * constantRangeSize;
    bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_FakeConstantBuffer));

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_FakeConstantBuffer;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::BufferViewDesc constantBufferViewDesc = {};
    constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
    constantBufferViewDesc.buffer = m_FakeConstantBuffer;
    constantBufferViewDesc.size = constantRangeSize;

    m_FakeConstantBufferViews.resize(fakeConstantBufferRangeNum);
    for (size_t i = 0; i < m_FakeConstantBufferViews.size(); i++) {
        NRI.CreateBufferView(constantBufferViewDesc, m_FakeConstantBufferViews[i]);
        constantBufferViewDesc.offset += constantRangeSize;
    }

    std::vector<uint8_t> bufferContent((size_t)bufferDesc.size, 0);

    nri::BufferUploadDesc bufferUpdate = {};
    bufferUpdate.buffer = m_FakeConstantBuffer;
    bufferUpdate.data = bufferContent.data();
    bufferUpdate.after = {nri::AccessBits::CONSTANT_BUFFER};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, &bufferUpdate, 1));
}

void Sample::CreateViewConstantBuffer() {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    uint32_t constantRangeSize = (uint32_t)helper::Align(sizeof(float4x4), deviceDesc.memoryAlignment.constantBufferOffset);

    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = constantRangeSize;
    bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ViewConstantBuffer));

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_ViewConstantBuffer;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::BufferViewDesc constantBufferViewDesc = {};
    constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
    constantBufferViewDesc.buffer = m_ViewConstantBuffer;
    constantBufferViewDesc.size = constantRangeSize;
    NRI.CreateBufferView(constantBufferViewDesc, m_ViewConstantBufferView);

    std::vector<uint8_t> bufferContent((size_t)bufferDesc.size, 0);
    SetupProjViewMatrix(*(float4x4*)(bufferContent.data()));

    nri::BufferUploadDesc bufferUpdate = {};
    bufferUpdate.buffer = m_ViewConstantBuffer;
    bufferUpdate.data = bufferContent.data();
    bufferUpdate.after = {nri::AccessBits::CONSTANT_BUFFER};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, &bufferUpdate, 1));
}

void Sample::SetupProjViewMatrix(float4x4& projViewMatrix) {
    const float aspect = float(GetWindowResolution().x) / float(GetWindowResolution().y);

    float4x4 projectionMatrix;
    projectionMatrix.SetupByHalfFovxInf(radians(45.0f), aspect, 0.1f, 0);

    float4x4 viewMatrix = float4x4::Identity();
    viewMatrix.SetupByRotationYPR(radians(0.0f), radians(0.0f), 0.0f);
    viewMatrix.WorldToView();

    const float3 cameraPosition = float3(0.0f, -2.5f, 2.0f);
    viewMatrix.PreTranslation(-cameraPosition);

    projViewMatrix = projectionMatrix * viewMatrix;
}

SAMPLE_MAIN(Sample, 0);
