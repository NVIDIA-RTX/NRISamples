/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "SampleBase.h"

#include <array>

struct NRIInterface
    : public nri::CoreInterface
    , public nri::SwapChainInterface
    , public nri::HelperInterface
{};

struct Frame
{
    nri::DeviceSemaphore* deviceSemaphore;
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

class Sample : public SampleBase
{
public:

    Sample()
    {}

    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI);
    void PrepareFrame(uint32_t frameIndex);
    void RenderFrame(uint32_t frameIndex);

private:

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::QueueSemaphore* m_AcquireSemaphore = nullptr;
    nri::QueueSemaphore* m_ReleaseSemaphore = nullptr;
    nri::Buffer* m_ReadbackBuffer = nullptr;

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
        NRI.DestroyDeviceSemaphore(*frame.deviceSemaphore);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
    {
        NRI.DestroyFrameBuffer(*backBuffer.frameBuffer);
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);
    }

    NRI.DestroyBuffer(*m_ReadbackBuffer);
    NRI.DestroyQueueSemaphore(*m_AcquireSemaphore);
    NRI.DestroyQueueSemaphore(*m_ReleaseSemaphore);
    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

    DestroyUserInterface();

    nri::DestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI)
{
    nri::PhysicalDeviceGroup physicalDeviceGroup = {};
    if (!helper::FindPhysicalDeviceGroup(physicalDeviceGroup))
        return false;

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.D3D11CommandBufferEmulation = D3D11_COMMANDBUFFER_EMULATION;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.physicalDeviceGroup = &physicalDeviceGroup;
    deviceCreationDesc.memoryAllocatorInterface = m_MemoryAllocatorInterface;
    NRI_ABORT_ON_FAILURE( nri::CreateDevice(deviceCreationDesc, m_Device) );

    // NRI
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) );

    // Command queue
    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue) );

    // Swap chain
    {
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.windowSystemType = GetWindowSystemType();
        swapChainDesc.window = GetWindow();
        swapChainDesc.commandQueue = m_CommandQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
        swapChainDesc.verticalSyncInterval = m_SwapInterval;
        swapChainDesc.width = GetWindowWidth();
        swapChainDesc.height = GetWindowHeight();
        swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;
        NRI_ABORT_ON_FAILURE( NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain) );

        uint32_t swapChainTextureNum;
        nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum, m_SwapChainFormat);

        for (uint32_t i = 0; i < swapChainTextureNum; i++)
        {
            nri::Texture2DViewDesc textureViewDesc = {swapChainTextures[i], nri::Texture2DViewType::COLOR_ATTACHMENT, m_SwapChainFormat};

            nri::Descriptor* colorAttachment;
            NRI_ABORT_ON_FAILURE( NRI.CreateTexture2DView(textureViewDesc, colorAttachment) );

            nri::FrameBufferDesc frameBufferDesc = {};
            frameBufferDesc.colorAttachmentNum = 1;
            frameBufferDesc.colorAttachments = &colorAttachment;
            nri::FrameBuffer* frameBuffer;
            NRI_ABORT_ON_FAILURE( NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, frameBuffer) );

            const BackBuffer backBuffer = { frameBuffer, frameBuffer, colorAttachment, swapChainTextures[i] };
            m_SwapChainBuffers.push_back(backBuffer);
        }
    }

    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_AcquireSemaphore) );
    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_ReleaseSemaphore) );

    // Buffered resources
    for (Frame& frame : m_Frames)
    {
        NRI_ABORT_ON_FAILURE( NRI.CreateDeviceSemaphore(*m_Device, true, frame.deviceSemaphore) );
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandAllocator(*m_CommandQueue, nri::WHOLE_DEVICE_GROUP, frame.commandAllocator) );
        NRI_ABORT_ON_FAILURE( NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer) );
    }

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    // Readback buffer
    {
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

    return CreateUserInterface(*m_Device, NRI, NRI, GetWindowWidth(), GetWindowHeight(), m_SwapChainFormat);
}

void Sample::PrepareFrame(uint32_t frameIndex)
{
    uint32_t color = 0;
    const uint32_t* data = (uint32_t*)NRI.MapBuffer(*m_ReadbackBuffer, 0, 128);
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

    PrepareUserInterface();

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
}

void Sample::RenderFrame(uint32_t frameIndex)
{
    const uint32_t windowWidth = GetWindowWidth();
    const uint32_t windowHeight = GetWindowHeight();
    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];

    const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain, *m_AcquireSemaphore);
    BackBuffer& backBuffer = m_SwapChainBuffers[backBufferIndex];

    NRI.WaitForSemaphore(*m_CommandQueue, *frame.deviceSemaphore);
    NRI.ResetCommandAllocator(*frame.commandAllocator);

    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, nullptr, 0);
    {
        nri::TextureTransitionBarrierDesc textureTransitionBarrierDesc = {};
        textureTransitionBarrierDesc.texture = backBuffer.texture;
        textureTransitionBarrierDesc.prevAccess = nri::AccessBits::UNKNOWN;
        textureTransitionBarrierDesc.nextAccess = nri::AccessBits::COPY_SOURCE;
        textureTransitionBarrierDesc.prevLayout = nri::TextureLayout::UNKNOWN;
        textureTransitionBarrierDesc.nextLayout = nri::TextureLayout::GENERAL;
        textureTransitionBarrierDesc.arraySize = 1;
        textureTransitionBarrierDesc.mipNum = 1;

        nri::TransitionBarrierDesc transitionBarriers = {};
        transitionBarriers.textureNum = 1;
        transitionBarriers.textures = &textureTransitionBarrierDesc;
        NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

        nri::TextureDataLayoutDesc dstDataLayoutDesc = {};
        dstDataLayoutDesc.rowPitch = NRI.GetDeviceDesc(*m_Device).uploadBufferTextureRowAlignment;

        nri::TextureRegionDesc srcRegionDesc = {};
        srcRegionDesc.offset[0] = (uint16_t)Clamp(ImGui::GetMousePos().x, 0.0f, float(windowWidth - 1));
        srcRegionDesc.offset[1] = (uint16_t)Clamp(ImGui::GetMousePos().y, 0.0f, float(windowHeight - 1));
        srcRegionDesc.size[0] = 1;
        srcRegionDesc.size[1] = 1;
        srcRegionDesc.size[2] = 1;

        // before clearing the texture read back contents under the mouse cursor
        NRI.CmdReadbackTextureToBuffer(commandBuffer, *m_ReadbackBuffer, dstDataLayoutDesc, *backBuffer.texture, srcRegionDesc);

        textureTransitionBarrierDesc.prevLayout = nri::TextureLayout::UNKNOWN;
        textureTransitionBarrierDesc.nextLayout = nri::TextureLayout::COLOR_ATTACHMENT;
        textureTransitionBarrierDesc.prevAccess = textureTransitionBarrierDesc.nextAccess;
        textureTransitionBarrierDesc.nextAccess = nri::AccessBits::COLOR_ATTACHMENT;
        NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

        NRI.CmdBeginRenderPass(commandBuffer, *backBuffer.frameBuffer, nri::RenderPassBeginFlag::NONE);
        {
            helper::Annotation annotation(NRI, commandBuffer, "Clear");

            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;

            clearDesc.value.rgba32f = {1.0f, 0.0f, 0.0f, 1.0f};
            nri::Rect rect1 = { 0, 0, windowWidth, windowHeight / 3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect1, 1);

            clearDesc.value.rgba32f = {0.0f, 1.0f, 0.0f, 1.0f};
            nri::Rect rect2 = { 0, (int32_t)windowHeight / 3, windowWidth, windowHeight / 3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect2, 1);

            clearDesc.value.rgba32f = {0.0f, 0.0f, 1.0f, 1.0f};
            nri::Rect rect3 = { 0, (int32_t)(windowHeight * 2) / 3, windowWidth, windowHeight / 3 };
            NRI.CmdClearAttachments(commandBuffer, &clearDesc, 1, &rect3, 1);

            RenderUserInterface(commandBuffer);
        }
        NRI.CmdEndRenderPass(commandBuffer);

        textureTransitionBarrierDesc.prevAccess = textureTransitionBarrierDesc.nextAccess;
        textureTransitionBarrierDesc.nextAccess = nri::AccessBits::UNKNOWN;
        textureTransitionBarrierDesc.prevLayout = textureTransitionBarrierDesc.nextLayout;
        textureTransitionBarrierDesc.nextLayout = nri::TextureLayout::PRESENT;

        NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);
    }
    NRI.EndCommandBuffer(commandBuffer);

    const nri::CommandBuffer* commandBufferArray[] = { &commandBuffer };

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBufferNum = helper::GetCountOf(commandBufferArray);
    workSubmissionDesc.commandBuffers = commandBufferArray;
    workSubmissionDesc.wait = &m_AcquireSemaphore;
    workSubmissionDesc.waitNum = 1;
    workSubmissionDesc.signal = &m_ReleaseSemaphore;
    workSubmissionDesc.signalNum = 1;
    NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, frame.deviceSemaphore);

    NRI.SwapChainPresent(*m_SwapChain, *m_ReleaseSemaphore);
}

SAMPLE_MAIN(Sample, 0);
