// © 2024 NVIDIA Corporation

#include "NRIFramework.h"
#include "Extensions/NRILowLatency.h"

#include <array>

// Tweakables, which must be set only once
#define ALLOW_LOW_LATENCY true
#define WAITABLE_SWAP_CHAIN false
#define WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY 1 // 2 helps to avoid "TOTAL = GPU + CPU" time issue
#define EMULATE_BAD_PRACTICE true
#define VSYNC_INTERVAL 0
#define QUEUED_FRAMES_MAX_NUM 3
#define CTA_NUM 38000 // tuned to reach ~1ms on RTX 4080

struct NRIInterface
    : public nri::CoreInterface
    , public nri::HelperInterface
    , public nri::StreamerInterface
    , public nri::SwapChainInterface
    , public nri::LowLatencyInterface
{};

struct Frame
{
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

class Sample : public SampleBase
{
public:

    Sample()
    {}

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
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Buffer* m_Buffer = nullptr;
    nri::Memory* m_Memory = nullptr;
    nri::Descriptor* m_BufferStorage = nullptr;

    std::array<Frame, QUEUED_FRAMES_MAX_NUM> m_Frames = {};
    std::vector<BackBuffer> m_SwapChainBuffers;
    float m_CpuWorkload = 4.0f; // ms
    uint32_t m_GpuWorkload = 10; // in pigeons, current settings give ~10 ms on RTX 4080
    uint32_t m_QueuedFrameNum = QUEUED_FRAMES_MAX_NUM; // [1; QUEUED_FRAMES_MAX_NUM]
    bool m_AllowLowLatency = false;
    bool m_EnableLowLatency = false;
};

Sample::~Sample()
{
    NRI.WaitForIdle(*m_CommandQueue);

    for (Frame& frame : m_Frames)
    {
        NRI.DestroyCommandBuffer(*frame.commandBuffer);
        NRI.DestroyCommandAllocator(*frame.commandAllocator);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);

    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyDescriptor(*m_BufferStorage);
    NRI.DestroyBuffer(*m_Buffer);
    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);
    NRI.DestroyStreamer(*m_Streamer);

    NRI.FreeMemory(*m_Memory);

    DestroyUI(NRI);

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI)
{
    nri::AdapterDesc bestAdapterDesc = {};
    uint32_t adapterDescsNum = 1;
    NRI_ABORT_ON_FAILURE( nri::nriEnumerateAdapters(&bestAdapterDesc, adapterDescsNum) );

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_COMMANDBUFFER_EMULATION;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &bestAdapterDesc;
    deviceCreationDesc.memoryAllocatorInterface = m_MemoryAllocatorInterface;
    NRI_ABORT_ON_FAILURE( nri::nriCreateDevice(deviceCreationDesc, m_Device) );

    // NRI
    NRI_ABORT_ON_FAILURE( nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI) );

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    // Create streamer
    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferUsageBits = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.frameInFlightNum = QUEUED_FRAMES_MAX_NUM;
    NRI_ABORT_ON_FAILURE( NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer) );

    // Low latency
    m_AllowLowLatency = ALLOW_LOW_LATENCY && deviceDesc.isLowLatencySupported;

    if (m_AllowLowLatency)
        NRI_ABORT_ON_FAILURE( nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::LowLatencyInterface), (nri::LowLatencyInterface*)&NRI) );

    // Command queue
    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue) );

    // Fence
    NRI_ABORT_ON_FAILURE( NRI.CreateFence(*m_Device, 0, m_FrameFence) );

    // Swap chain
    nri::Format swapChainFormat;
    {
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.window = GetWindow();
        swapChainDesc.commandQueue = m_CommandQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
        swapChainDesc.verticalSyncInterval = m_VsyncInterval;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = (uint8_t)m_Frames.size();
        swapChainDesc.verticalSyncInterval = VSYNC_INTERVAL;
        swapChainDesc.queuedFrameNum = WAITABLE_SWAP_CHAIN ? WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY : (uint8_t)m_Frames.size();
        swapChainDesc.waitable = WAITABLE_SWAP_CHAIN;
        swapChainDesc.allowLowLatency = m_AllowLowLatency;
        NRI_ABORT_ON_FAILURE( NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain) );

        uint32_t swapChainTextureNum;
        nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
        swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

        for (uint32_t i = 0; i < swapChainTextureNum; i++)
        {
            nri::Texture2DViewDesc textureViewDesc = {swapChainTextures[i], nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};

            nri::Descriptor* colorAttachment;
            NRI_ABORT_ON_FAILURE( NRI.CreateTexture2DView(textureViewDesc, colorAttachment) );

            const BackBuffer backBuffer = { colorAttachment, swapChainTextures[i] };
            m_SwapChainBuffers.push_back(backBuffer);
        }
    }

    { // Buffer
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = CTA_NUM * 256 * sizeof(float);
        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;

        NRI_ABORT_ON_FAILURE( NRI.CreateBuffer(*m_Device, bufferDesc, m_Buffer) );

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_Buffer;

        NRI_ABORT_ON_FAILURE( NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, &m_Memory) );

        nri::BufferViewDesc bufferViewDesc = {};
        bufferViewDesc.buffer = m_Buffer;
        bufferViewDesc.format = nri::Format::R16_SFLOAT;
        bufferViewDesc.viewType = nri::BufferViewType::SHADER_RESOURCE_STORAGE;

        NRI_ABORT_ON_FAILURE( NRI.CreateBufferView(bufferViewDesc, m_BufferStorage) );
    }

    { // Compute pipeline
        utils::ShaderCodeStorage shaderCodeStorage;

        nri::DescriptorRangeDesc descriptorRangeStorage = {0, 1, nri::DescriptorType::STORAGE_BUFFER, nri::StageBits::COMPUTE_SHADER};
        nri::DescriptorSetDesc descriptorSetDesc = {0, &descriptorRangeStorage, 1};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        NRI_ABORT_ON_FAILURE( NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout) );

        nri::ComputePipelineDesc computePipelineDesc = {};
        computePipelineDesc.pipelineLayout = m_PipelineLayout;
        computePipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Compute.cs", shaderCodeStorage);
        NRI_ABORT_ON_FAILURE( NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_Pipeline) );
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.storageBufferMaxNum = 1;

        NRI_ABORT_ON_FAILURE( NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool) );    
        NRI_ABORT_ON_FAILURE( NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0) );

        nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_BufferStorage, 1, 0};
        NRI.UpdateDescriptorRanges(*m_DescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
    }

    // Buffered resources
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandAllocator(*m_CommandQueue, frame.commandAllocator) );
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer) );
    }

    return InitUI(NRI, NRI, *m_Device, swapChainFormat);
}

void Sample::LatencySleep(uint32_t frameIndex)
{
    // Marker
    if (m_AllowLowLatency)
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::SIMULATION_START);

    // Wait for present
    if constexpr (WAITABLE_SWAP_CHAIN)
        NRI.WaitForPresent(*m_SwapChain);

    // Preserve frame queue (optimal place for "non-waitable" swap chain)
    if constexpr (WAITABLE_SWAP_CHAIN == EMULATE_BAD_PRACTICE)
    {
        const Frame& frame = m_Frames[frameIndex % m_QueuedFrameNum];
        if (frameIndex >= m_QueuedFrameNum)
        {
            NRI.Wait(*m_FrameFence, 1 + frameIndex - m_QueuedFrameNum);
            NRI.ResetCommandAllocator(*frame.commandAllocator);
        }
    }

    // Sleep just before sampling input
    if (m_AllowLowLatency)
    {
        NRI.LatencySleep(*m_SwapChain);
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::INPUT_SAMPLE);
    }
}

void Sample::PrepareFrame(uint32_t)
{
    // Emulate CPU workload
    double begin = m_Timer.GetTimeStamp() + m_CpuWorkload;
    while(m_Timer.GetTimeStamp() < begin)
        ;

    BeginUI();

    // Lagometer
    ImVec2 p = ImGui::GetIO().MousePos;
    ImGui::GetForegroundDrawList()->AddRectFilled(p, ImVec2(p.x + 20, p.y + 20), IM_COL32(128, 10, 10, 255));

    // Stats
    bool enableLowLatencyPrev = m_EnableLowLatency;
    uint32_t queuedFrameNumPrev = m_QueuedFrameNum;

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
        ImGui::SliderInt("##Frames", (int32_t*)&m_QueuedFrameNum, 1, (int32_t)m_Frames.size(), "%d", ImGuiSliderFlags_NoInput);

        if (!m_AllowLowLatency) ImGui::BeginDisabled();
        ImGui::Checkbox("Low latency (F1)", &m_EnableLowLatency);
        if (m_AllowLowLatency && IsKeyToggled(Key::F1))
            m_EnableLowLatency = !m_EnableLowLatency;
        if (!m_AllowLowLatency) ImGui::EndDisabled();

        ImGui::BeginDisabled();
        bool waitable = WAITABLE_SWAP_CHAIN;
        ImGui::Checkbox("Waitable swapchain (" STRINGIFY(WAITABLE_SWAP_CHAIN_MAX_FRAME_LATENCY) ")", &waitable);
        bool badPractice = EMULATE_BAD_PRACTICE;
        ImGui::Checkbox("Bad practice", &badPractice);
        ImGui::EndDisabled();
    }
    ImGui::End();

    EndUI(NRI, *m_Streamer);
    NRI.CopyStreamerUpdateRequests(*m_Streamer);

    if (enableLowLatencyPrev != m_EnableLowLatency)
    {
        nri::LatencySleepMode sleepMode = {};
        sleepMode.lowLatencyMode = m_EnableLowLatency;
        sleepMode.lowLatencyBoost = m_EnableLowLatency;

        NRI.SetLatencySleepMode(*m_SwapChain, sleepMode);
    }

    if (queuedFrameNumPrev != m_QueuedFrameNum)
        NRI.WaitForIdle(*m_CommandQueue);

    // Marker
    if (m_AllowLowLatency)
        NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::SIMULATION_END);
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
    const BackBuffer& backBuffer = m_SwapChainBuffers[backBufferIndex];
    const Frame& frame = m_Frames[frameIndex % m_QueuedFrameNum];

    // Preserve frame queue (optimal place for "waitable" swapchain)
    if constexpr (WAITABLE_SWAP_CHAIN != EMULATE_BAD_PRACTICE)
    {
        if (frameIndex >= m_QueuedFrameNum)
        {
            NRI.Wait(*m_FrameFence, 1 + frameIndex - m_QueuedFrameNum);
            NRI.ResetCommandAllocator(*frame.commandAllocator);
        }
    }

    // Record
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        nri::TextureBarrierDesc swapchainBarrier = {};
        swapchainBarrier.texture = backBuffer.texture;
        swapchainBarrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        swapchainBarrier.arraySize = 1;
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

        for (uint32_t i = 0; i < m_GpuWorkload; i++)
        {
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
        attachmentsDesc.colors = &backBuffer.colorAttachment;

        NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
        {
            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;
            clearDesc.value.color32f = {0.0f, 0.1f, 0.0f, 1.0f};

            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, nullptr, 0);

            RenderUI(NRI, NRI, *m_Streamer, commandBuffer, 1.0f, true);
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
    }
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::FenceSubmitDesc signalFence = {};
        signalFence.fence = m_FrameFence;
        signalFence.value = 1 + frameIndex;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &frame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;
        queueSubmitDesc.signalFences = &signalFence;
        queueSubmitDesc.signalFenceNum = 1;

        if (m_AllowLowLatency)
        {
            NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::RENDER_SUBMIT_START);
            NRI.QueueSubmitTrackable(*m_CommandQueue, queueSubmitDesc, *m_SwapChain);
            NRI.SetLatencyMarker(*m_SwapChain, nri::LatencyMarker::RENDER_SUBMIT_END);
        }
        else
            NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);
    }

    // Present
    NRI.QueuePresent(*m_SwapChain);
}

SAMPLE_MAIN(Sample, 0);
