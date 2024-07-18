// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

struct NRIInterface
    : public nri::CoreInterface
    , public nri::HelperInterface
    , public nri::StreamerInterface
    , public nri::SwapChainInterface
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
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

private:

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::Buffer* m_ReadbackBuffer = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<BackBuffer> m_SwapChainBuffers;

    nri::Format m_SwapChainFormat;
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

    NRI.DestroyBuffer(*m_ReadbackBuffer);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);
    NRI.DestroyStreamer(*m_Streamer);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

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

    // Create streamer
    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferUsageBits = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.frameInFlightNum = BUFFERED_FRAME_MAX_NUM;
    NRI_ABORT_ON_FAILURE( NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer) );

    // Command queue
    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue) );

    // Fences
    NRI_ABORT_ON_FAILURE( NRI.CreateFence(*m_Device, 0, m_FrameFence) );

    { // Swap chain
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.window = GetWindow();
        swapChainDesc.commandQueue = m_CommandQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
        swapChainDesc.verticalSyncInterval = m_VsyncInterval;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;
        NRI_ABORT_ON_FAILURE( NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain) );

        uint32_t swapChainTextureNum;
        nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
        m_SwapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

        for (uint32_t i = 0; i < swapChainTextureNum; i++)
        {
            nri::Texture2DViewDesc textureViewDesc = {swapChainTextures[i], nri::Texture2DViewType::COLOR_ATTACHMENT, m_SwapChainFormat};

            nri::Descriptor* colorAttachment;
            NRI_ABORT_ON_FAILURE( NRI.CreateTexture2DView(textureViewDesc, colorAttachment) );

            const BackBuffer backBuffer = { colorAttachment, swapChainTextures[i] };
            m_SwapChainBuffers.push_back(backBuffer);
        }
    }

    // Buffered resources
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandAllocator(*m_CommandQueue, frame.commandAllocator) );
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer) );
    }

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    { // Readback buffer
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = helper::Align(4, deviceDesc.uploadBufferTextureRowAlignment);
        NRI_ABORT_ON_FAILURE( NRI.CreateBuffer(*m_Device, bufferDesc, m_ReadbackBuffer) );

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_READBACK;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_ReadbackBuffer;

        m_MemoryAllocations.resize(1, nullptr);
        NRI_ABORT_ON_FAILURE( NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data()) );
    }

    return InitUI(NRI, NRI, *m_Device, m_SwapChainFormat);
}

void Sample::PrepareFrame(uint32_t)
{
    BeginUI();

    uint32_t color = 0;
    const uint32_t* data = (uint32_t*)NRI.MapBuffer(*m_ReadbackBuffer, 0, nri::WHOLE_SIZE);
    if (data)
    {
        color = *data | 0xFF000000;
        NRI.UnmapBuffer(*m_ReadbackBuffer);
    }

    if (m_SwapChainFormat == nri::Format::BGRA8_UNORM)
    {
        uint8_t* bgra = (uint8_t*)&color;
        Swap(bgra[0], bgra[2]);
    }

    ImVec2 p = ImGui::GetIO().MousePos;
    p.x += 24;

    float sz = ImGui::GetTextLineHeight();
    ImGui::SetNextWindowPos(p, ImGuiCond_Always);
    ImGui::Begin("ColorWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize);
    {
        p = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x+sz, p.y+sz), color);
        ImGui::Dummy(ImVec2(sz, sz));
        ImGui::SameLine();
        ImGui::Text("Color");
    }
    ImGui::End();

    EndUI(NRI, *m_Streamer);
    NRI.CopyStreamerUpdateRequests(*m_Streamer);
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    const uint32_t windowWidth = GetWindowResolution().x;
    const uint32_t windowHeight = GetWindowResolution().y;
    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM)
    {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
        NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
    BackBuffer& backBuffer = m_SwapChainBuffers[backBufferIndex];

    // Record
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, nullptr);
    {
        nri::TextureBarrierDesc textureBarrierDescs = {};
        textureBarrierDescs.texture = backBuffer.texture;
        textureBarrierDescs.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE};
        textureBarrierDescs.arraySize = 1;
        textureBarrierDescs.mipNum = 1;

        nri::BarrierGroupDesc barrierGroupDesc = {};
        barrierGroupDesc.textureNum = 1;
        barrierGroupDesc.textures = &textureBarrierDescs;
        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);

        nri::TextureDataLayoutDesc dstDataLayoutDesc = {};
        dstDataLayoutDesc.rowPitch = helper::Align(4, NRI.GetDeviceDesc(*m_Device).uploadBufferTextureRowAlignment);

        nri::TextureRegionDesc srcRegionDesc = {};
        srcRegionDesc.x = (uint16_t)clamp(ImGui::GetMousePos().x, 0.0f, float(windowWidth - 1));
        srcRegionDesc.y = (uint16_t)clamp(ImGui::GetMousePos().y, 0.0f, float(windowHeight - 1));
        srcRegionDesc.width = 1;
        srcRegionDesc.height = 1;
        srcRegionDesc.depth = 1;

        // before clearing the texture read back contents under the mouse cursor
        NRI.CmdReadbackTextureToBuffer(commandBuffer, *m_ReadbackBuffer, dstDataLayoutDesc, *backBuffer.texture, srcRegionDesc);

        textureBarrierDescs.before = textureBarrierDescs.after;
        textureBarrierDescs.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);

        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &backBuffer.colorAttachment;

        NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
        {
            helper::Annotation annotation(NRI, commandBuffer, "Clear");

            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;

            nri::Dim_t w = (nri::Dim_t)GetWindowResolution().x;
            nri::Dim_t h = (nri::Dim_t)GetWindowResolution().y;
            nri::Dim_t h3 = h / 3;
            int16_t y = (int16_t)h3;

            clearDesc.value.color32f = {1.0f, 0.0f, 0.0f, 1.0f};
            nri::Rect rect1 = { 0, 0, w, h3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect1, 1);

            clearDesc.value.color32f = {0.0f, 1.0f, 0.0f, 1.0f};
            nri::Rect rect2 = { 0, y, w, h3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect2, 1);

            clearDesc.value.color32f = {0.0f, 0.0f, 1.0f, 1.0f};
            nri::Rect rect3 = { 0, y * 2, w, h3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect3, 1);

            RenderUI(NRI, NRI, *m_Streamer, commandBuffer, 1.0f, true);
        }
        NRI.CmdEndRendering(commandBuffer);

        textureBarrierDescs.before = textureBarrierDescs.after;
        textureBarrierDescs.after = {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT};

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
    }
    NRI.EndCommandBuffer(commandBuffer);

    { // Submit
        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &frame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;

        NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);
    }

    // Present
    NRI.QueuePresent(*m_SwapChain);

    { // Signaling after "Present" improves D3D11 performance a bit
        nri::FenceSubmitDesc signalFence = {};
        signalFence.fence = m_FrameFence;
        signalFence.value = 1 + frameIndex;

        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.signalFences = &signalFence;
        queueSubmitDesc.signalFenceNum = 1;

        NRI.QueueSubmit(*m_CommandQueue, queueSubmitDesc);
    }
}

SAMPLE_MAIN(Sample, 0);
