// © 2024 NVIDIA Corporation

#include "NRIFramework.h"

// Tweakables, which must be set only once
constexpr bool ALLOW_LOW_LATENCY = true;
constexpr bool WAITABLE_SWAP_CHAIN = false;
constexpr bool EMULATE_BAD_PRACTICE = false;
constexpr bool VSYNC = false;
constexpr uint32_t WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY = 1; // 2 helps to avoid "TOTAL = GPU + CPU" time issue
constexpr uint32_t QUEUED_FRAMES_MAX_NUM = 3;
constexpr uint32_t CTA_NUM = 38000; // TODO: tuned to reach ~1ms on RTX 4080
constexpr uint32_t COLOR_LATENCY_SLEEP = NriBgra(255, 0, 0);
constexpr uint32_t COLOR_SIMULATION = NriBgra(0, 255, 0);
constexpr uint32_t COLOR_RENDER = NriBgra(0, 0, 255);

struct NRIInterface
    : public nri::CoreInterface,
      public nri::HelperInterface,
      public nri::StreamerInterface,
      public nri::SwapChainInterface,
      public nri::LowLatencyInterface {};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

class Sample : public SampleBase {
public:
    Sample() {
        m_QueuedFrames.resize(QUEUED_FRAMES_MAX_NUM);
    }

    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void LatencySleep(uint32_t) override;
    void PrepareFrame(uint32_t) override;
    void RenderFrame(uint32_t frameIndex) override;

private:
    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Buffer* m_Buffer = nullptr;
    nri::Memory* m_Memory = nullptr;
    nri::Descriptor* m_BufferStorage = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
    float m_CpuWorkload = 4.0f;                        // ms
    uint32_t m_GpuWorkload = 10;                       // in pigeons, current settings give ~10 ms on RTX 4080
    uint32_t m_QueuedFrameNum = QUEUED_FRAMES_MAX_NUM; // [1; QUEUED_FRAMES_MAX_NUM]
    bool m_AllowLowLatency = false;
    bool m_EnableLowLatency = false;
};

Sample::~Sample() {
    NRI.WaitForIdle(*m_GraphicsQueue);

    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI.DestroyCommandBuffer(*queuedFrame.commandBuffer);
        NRI.DestroyCommandAllocator(*queuedFrame.commandAllocator);
    }

    for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
        NRI.DestroyFence(*swapChainTexture.acquireSemaphore);
        NRI.DestroyFence(*swapChainTexture.releaseSemaphore);
        NRI.DestroyDescriptor(*swapChainTexture.colorAttachment);
    }

    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyDescriptor(*m_BufferStorage);
    NRI.DestroyBuffer(*m_Buffer);
    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);
    NRI.DestroyStreamer(*m_Streamer);

    NRI.FreeMemory(*m_Memory);

    DestroyImgui();

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
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

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    // Create streamer
    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferUsageBits = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.queuedFrameNum = QUEUED_FRAMES_MAX_NUM;
    NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));

    // Low latency
    m_AllowLowLatency = ALLOW_LOW_LATENCY && deviceDesc.features.lowLatency;

    if (m_AllowLowLatency)
        NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::LowLatencyInterface), (nri::LowLatencyInterface*)&NRI));

    // Command queue
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));

    // Fence
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Swap chain
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.window = GetWindow();
        swapChainDesc.queue = m_GraphicsQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
        swapChainDesc.flags = (m_Vsync ? nri::SwapChainBits::VSYNC : nri::SwapChainBits::NONE) | nri::SwapChainBits::ALLOW_TEARING;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = QUEUED_FRAMES_MAX_NUM + 1;
        swapChainDesc.queuedFrameNum = WAITABLE_SWAP_CHAIN ? WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY : QUEUED_FRAMES_MAX_NUM;

        if constexpr (VSYNC)
            swapChainDesc.flags |= nri::SwapChainBits::VSYNC;
        if constexpr (WAITABLE_SWAP_CHAIN)
            swapChainDesc.flags |= nri::SwapChainBits::WAITABLE;
        if (m_AllowLowLatency)
            swapChainDesc.flags |= nri::SwapChainBits::ALLOW_LOW_LATENCY;

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

    { // Buffer
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = CTA_NUM * 256 * sizeof(float);
        bufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;

        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_Buffer));

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_Buffer;

        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, &m_Memory));

        nri::BufferViewDesc bufferViewDesc = {};
        bufferViewDesc.buffer = m_Buffer;
        bufferViewDesc.format = nri::Format::R16_SFLOAT;
        bufferViewDesc.viewType = nri::BufferViewType::SHADER_RESOURCE_STORAGE;

        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, m_BufferStorage));
    }

    { // Compute pipeline
        utils::ShaderCodeStorage shaderCodeStorage;

        nri::DescriptorRangeDesc descriptorRangeStorage = {0, 1, nri::DescriptorType::STORAGE_BUFFER, nri::StageBits::COMPUTE_SHADER};
        nri::DescriptorSetDesc descriptorSetDesc = {0, &descriptorRangeStorage, 1};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

        nri::ComputePipelineDesc computePipelineDesc = {};
        computePipelineDesc.pipelineLayout = m_PipelineLayout;
        computePipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Compute.cs", shaderCodeStorage);
        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_Pipeline));
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.storageBufferMaxNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0));

        nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_BufferStorage, 1, 0};
        NRI.UpdateDescriptorRanges(*m_DescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
    }

    // Queued frames
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    return InitImgui(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    nri::nriBeginAnnotation("LatencySleep", COLOR_LATENCY_SLEEP);

    // Marker
    if (m_AllowLowLatency)
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::SIMULATION_START);

    // Wait for present
    if constexpr (WAITABLE_SWAP_CHAIN)
        NRI.WaitForPresent(*m_SwapChain);

    // Preserve frame queue (optimal place for "non-waitable" swap chain)
    if constexpr (WAITABLE_SWAP_CHAIN == EMULATE_BAD_PRACTICE) {
        const QueuedFrame& queuedFrame = m_QueuedFrames[frameIndex % m_QueuedFrameNum];

        NRI.Wait(*m_FrameFence, frameIndex >= m_QueuedFrameNum ? 1 + frameIndex - m_QueuedFrameNum : 0);
        NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
    }

    // Sleep just before sampling input
    if (m_AllowLowLatency) {
        NRI.LatencySleep(*m_SwapChain);
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::INPUT_SAMPLE);
    }

    nri::nriEndAnnotation();
}

void Sample::PrepareFrame(uint32_t) {
    nri::nriBeginAnnotation("Simulation", COLOR_SIMULATION);

    // Emulate CPU workload
    double begin = m_Timer.GetTimeStamp() + m_CpuWorkload;
    while (m_Timer.GetTimeStamp() < begin)
        ;

    bool enableLowLatencyPrev = m_EnableLowLatency;
    uint32_t queuedFrameNumPrev = m_QueuedFrameNum;

    ImGui::NewFrame();
    {
        // Lagometer
        ImVec2 p = ImGui::GetIO().MousePos;
        ImGui::GetForegroundDrawList()->AddRectFilled(p, ImVec2(p.x + 20, p.y + 20), IM_COL32(128, 10, 10, 255));

        // Stats
        nri::LatencyReport latencyReport = {};
        if (m_AllowLowLatency)
            NRI.GetLatencyReport(*m_SwapChain, latencyReport);

        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::Begin("Low latency");
        {
            ImGui::Text("X (end) - Input    =   .... ms");
            ImGui::Separator();
            ImGui::Text("  Input            : %+6.2f", 0.0);
            ImGui::Text("  Simulation       : %+6.2f", (int64_t)(latencyReport.simulationEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Text("  Render           : %+6.2f", (int64_t)(latencyReport.renderSubmitEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Text("  Present          : %+6.2f", (int64_t)(latencyReport.presentEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Text("  Driver           : %+6.2f", (int64_t)(latencyReport.driverEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Text("  OS render queue  : %+6.2f", (int64_t)(latencyReport.osRenderQueueEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Text("  GPU render       : %+6.2f", (int64_t)(latencyReport.gpuRenderEndTimeUs - latencyReport.inputSampleTimeUs) / 1000.0);
            ImGui::Separator();
            ImGui::Text("Frame time         : %6.2f ms", m_Timer.GetSmoothedFrameTime());
            ImGui::Separator();

            ImGui::Text("CPU workload (ms):");
            ImGui::SetNextItemWidth(210.0f);
            ImGui::SliderFloat("##CPU", &m_CpuWorkload, 0.0f, 1000.0f / 30.0f, "%.1f", ImGuiSliderFlags_NoInput);
            ImGui::Text("GPU workload (pigeons):");
            ImGui::SetNextItemWidth(210.0f);
            ImGui::SliderInt("##GPU", (int32_t*)&m_GpuWorkload, 1, 20, "%d", ImGuiSliderFlags_NoInput);
            ImGui::Text("Queued frames:");
            ImGui::SetNextItemWidth(210.0f);
            ImGui::SliderInt("##Frames", (int32_t*)&m_QueuedFrameNum, 1, QUEUED_FRAMES_MAX_NUM, "%d", ImGuiSliderFlags_NoInput);

            if (!m_AllowLowLatency)
                ImGui::BeginDisabled();
            ImGui::Checkbox("Low latency (F1)", &m_EnableLowLatency);
            if (m_AllowLowLatency && IsKeyToggled(Key::F1))
                m_EnableLowLatency = !m_EnableLowLatency;
            if (!m_AllowLowLatency)
                ImGui::EndDisabled();

            char s[64];
            snprintf(s, sizeof(s), "Waitable swapchain (%u)", WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY);

            ImGui::BeginDisabled();
            bool waitable = WAITABLE_SWAP_CHAIN;
            ImGui::Checkbox(s, &waitable);
            bool badPractice = EMULATE_BAD_PRACTICE;
            ImGui::Checkbox("Bad practice", &badPractice);
            ImGui::EndDisabled();
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();

    if (enableLowLatencyPrev != m_EnableLowLatency) {
        nri::LatencySleepMode sleepMode = {};
        sleepMode.lowLatencyMode = m_EnableLowLatency;
        sleepMode.lowLatencyBoost = m_EnableLowLatency;

        NRI.SetLatencySleepMode(*m_SwapChain, sleepMode);
    }

    if (queuedFrameNumPrev != m_QueuedFrameNum)
        NRI.WaitForIdle(*m_GraphicsQueue);

    // Marker
    if (m_AllowLowLatency)
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::SIMULATION_END);

    nri::nriEndAnnotation();
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::nriBeginAnnotation("Render", COLOR_RENDER);

    const QueuedFrame& queuedFrame = m_QueuedFrames[frameIndex % m_QueuedFrameNum];

    // Preserve frame queue (optimal place for "waitable" swapchain)
    if constexpr (WAITABLE_SWAP_CHAIN != EMULATE_BAD_PRACTICE) {
        NRI.Wait(*m_FrameFence, frameIndex >= m_QueuedFrameNum ? 1 + frameIndex - m_QueuedFrameNum : 0);
        NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
    }

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Record
    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        NRI.CmdBeginAnnotation(commandBuffer, "Render", COLOR_RENDER);

        nri::TextureBarrierDesc swapchainBarrier = {};
        swapchainBarrier.texture = swapChainTexture.texture;
        swapchainBarrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        swapchainBarrier.layerNum = 1;
        swapchainBarrier.mipNum = 1;

        { // Barrier
            nri::BarrierGroupDesc barriers = {};
            barriers.textureNum = 1;
            barriers.textures = &swapchainBarrier;

            NRI.CmdBarrier(commandBuffer, barriers);
        }

        // Compute workload (main, resolution independent)
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
        NRI.CmdSetPipeline(commandBuffer, *m_Pipeline);
        NRI.CmdSetDescriptorSet(commandBuffer, 0, *m_DescriptorSet, nullptr);

        for (uint32_t i = 0; i < m_GpuWorkload; i++) {
            NRI.CmdDispatch(commandBuffer, {CTA_NUM, 1, 1});

            { // Barrier
                nri::GlobalBarrierDesc storageBarrier = {};
                storageBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
                storageBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

                nri::BarrierGroupDesc barriers = {};
                barriers.globalNum = 1;
                barriers.globals = &storageBarrier;

                NRI.CmdBarrier(commandBuffer, barriers);
            }
        }

        // Clear and UI
        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &swapChainTexture.colorAttachment;

        NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
        {
            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;
            clearDesc.planes = nri::PlaneBits::COLOR;
            clearDesc.value.color.f = {0.0f, 0.1f, 0.0f, 1.0f};

            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, nullptr, 0);

            RenderImgui(commandBuffer, *m_Streamer, swapChainTexture.attachmentFormat, 1.0f, true);
        }
        NRI.CmdEndRendering(commandBuffer);

        { // Barrier
            swapchainBarrier.before = swapchainBarrier.after;
            swapchainBarrier.after = {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT};

            nri::BarrierGroupDesc barriers = {};
            barriers.textureNum = 1;
            barriers.textures = &swapchainBarrier;

            NRI.CmdBarrier(commandBuffer, barriers);
        }

        NRI.CmdEndAnnotation(commandBuffer);
    }
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::FenceSubmitDesc frameFence = {};
        frameFence.fence = m_FrameFence;
        frameFence.value = 1 + frameIndex;

        nri::FenceSubmitDesc textureAcquiredFence = {};
        textureAcquiredFence.fence = swapChainAcquireSemaphore;
        textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

        nri::FenceSubmitDesc renderingFinishedFence = {};
        renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

        nri::FenceSubmitDesc signalFences[] = {renderingFinishedFence, frameFence};

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.waitFences = &textureAcquiredFence;
        queueSubmitDesc.waitFenceNum = 1;
        queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = signalFences;
        queueSubmitDesc.signalFenceNum = helper::GetCountOf(signalFences);

        NRI.QueueAnnotation(*m_GraphicsQueue, "Submit", COLOR_RENDER);

        if (m_AllowLowLatency) {
            NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::RENDER_SUBMIT_START);
            NRI.QueueSubmitTrackable(*m_GraphicsQueue, queueSubmitDesc, *m_SwapChain);
            NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::RENDER_SUBMIT_END);
        } else
            NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    NRI.EndStreamerFrame(*m_Streamer);

    // Present
    NRI.QueuePresent(*m_SwapChain, *swapChainTexture.releaseSemaphore);

    nri::nriEndAnnotation();
}

SAMPLE_MAIN(Sample, 0);
