// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

constexpr uint32_t VERTEX_NUM = 100000 * 3;

struct QueuedFrame {
    nri::CommandAllocator* commandAllocatorGraphics;
    nri::CommandAllocator* commandAllocatorCompute;
    std::array<nri::CommandBuffer*, 3> commandBufferGraphics;
    nri::CommandBuffer* commandBufferCompute;
};

struct Vertex {
    float position[3];
};

class Sample : public SampleBase {
public:
    Sample() {
    }

    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI, bool) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

private:
    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Queue* m_ComputeQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::Fence* m_ComputeFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_SharedPipelineLayout = nullptr;
    nri::Pipeline* m_GraphicsPipeline = nullptr;
    nri::Pipeline* m_ComputePipeline = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Texture* m_Texture = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Descriptor* m_Descriptor = nullptr;
    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
    std::vector<nri::Memory*> m_MemoryAllocations;
    bool m_IsAsyncMode = false;
    bool m_HasComputeQueue = false;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            for (size_t i = 0; i < queuedFrame.commandBufferGraphics.size(); i++)
                NRI.DestroyCommandBuffer(queuedFrame.commandBufferGraphics[i]);

            NRI.DestroyCommandBuffer(queuedFrame.commandBufferCompute);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocatorCompute);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocatorGraphics);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        NRI.DestroyDescriptor(m_Descriptor);
        NRI.DestroyTexture(m_Texture);
        NRI.DestroyBuffer(m_GeometryBuffer);
        NRI.DestroyPipeline(m_GraphicsPipeline);
        NRI.DestroyPipeline(m_ComputePipeline);
        NRI.DestroyPipelineLayout(m_SharedPipelineLayout);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
        NRI.DestroyFence(m_ComputeFence);
        NRI.DestroyFence(m_FrameFence);

        for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
            NRI.FreeMemory(m_MemoryAllocations[i]);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(m_Streamer);

    DestroyImgui();

    nri::nriDestroyDevice(m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    // Adapters
    nri::AdapterDesc adapterDesc[2] = {};
    uint32_t adapterDescsNum = helper::GetCountOf(adapterDesc);
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(adapterDesc, adapterDescsNum));

    // Device
    nri::QueueFamilyDesc queueFamilies[2] = {};
    queueFamilies[0].queueNum = 1;
    queueFamilies[0].queueType = nri::QueueType::GRAPHICS;
    queueFamilies[1].queueNum = 1;
    queueFamilies[1].queueType = nri::QueueType::COMPUTE;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = helper::GetCountOf(queueFamilies);
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
    streamerDesc.queuedFrameNum = GetQueuedFrameNum();
    NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));

    // Command queues
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI.SetDebugName(m_GraphicsQueue, "GraphicsQueue");

    NRI.GetQueue(*m_Device, nri::QueueType::COMPUTE, 0, m_ComputeQueue);
    if (m_ComputeQueue)
        NRI.SetDebugName(m_ComputeQueue, "ComputeQueue");

    m_HasComputeQueue = m_ComputeQueue && graphicsAPI != nri::GraphicsAPI::D3D11;
    m_IsAsyncMode = m_HasComputeQueue;

    // Fences
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_ComputeFence));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    // Swap chain
    nri::Format swapChainFormat;
    {
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

        swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

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
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocatorGraphics));
        for (size_t i = 0; i < queuedFrame.commandBufferGraphics.size(); i++)
            NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocatorGraphics, queuedFrame.commandBufferGraphics[i]));

        if (m_IsAsyncMode) {
            NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_ComputeQueue, queuedFrame.commandAllocatorCompute));
            NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocatorCompute, queuedFrame.commandBufferCompute));
        }
    }

    { // Pipeline layout
        nri::DescriptorRangeDesc descriptorRangeStorage = {0, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER};

        nri::DescriptorSetDesc descriptorSetDesc = {0, &descriptorRangeStorage, 1};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER | nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;
        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_SharedPipelineLayout));
    }

    utils::ShaderCodeStorage shaderCodeStorage;
    { // Graphics pipeline
        nri::VertexStreamDesc vertexStreamDesc = {};
        vertexStreamDesc.bindingSlot = 0;

        nri::VertexAttributeDesc vertexAttributeDesc[1] = {};
        {
            vertexAttributeDesc[0].format = nri::Format::RGB32_SFLOAT;
            vertexAttributeDesc[0].streamIndex = 0;
            vertexAttributeDesc[0].offset = helper::GetOffsetOf(&Vertex::position);
            vertexAttributeDesc[0].d3d = {"POSITION", 0};
            vertexAttributeDesc[0].vk.location = {0};
        }

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

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = &colorAttachmentDesc;
        outputMergerDesc.colorNum = 1;

        nri::ShaderDesc shaderStages[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "Triangles.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "Triangles.fs", shaderCodeStorage),
        };

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_SharedPipelineLayout;
        graphicsPipelineDesc.vertexInput = &vertexInputDesc;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);
        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_GraphicsPipeline));
    }

    { // Compute pipeline
        nri::ComputePipelineDesc computePipelineDesc = {};
        computePipelineDesc.pipelineLayout = m_SharedPipelineLayout;
        computePipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "Surface.cs", shaderCodeStorage);
        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_ComputePipeline));
    }

    { // Storage texture
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
        textureDesc.format = swapChainFormat;
        textureDesc.width = (uint16_t)GetWindowResolution().x / 2;
        textureDesc.height = (uint16_t)GetWindowResolution().y;
        textureDesc.mipNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_Texture));
    }

    { // Geometry buffer
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = sizeof(Vertex) * VERTEX_NUM;
        bufferDesc.usage = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_GeometryBuffer));
    }

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_GeometryBuffer;
    resourceGroupDesc.textureNum = 1;
    resourceGroupDesc.textures = &m_Texture;

    m_MemoryAllocations.resize(NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data()))

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.storageTextureMaxNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    }

    { // Storage descriptor
        nri::Texture2DViewDesc texture2DViewDesc = {m_Texture, nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D, swapChainFormat};

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_Descriptor));
    }

    { // Descriptor set
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_SharedPipelineLayout, 0, &m_DescriptorSet, 1, 0));

        nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_Descriptor, 1, 0};
        NRI.UpdateDescriptorRanges(*m_DescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
    }

    Rng::Hash::Initialize(m_RngState, 567, 57);

    { // Upload data
        std::vector<Vertex> geometryBufferData(VERTEX_NUM);
        for (uint32_t i = 0; i < VERTEX_NUM; i += 3) {
            Vertex& v0 = geometryBufferData[i];
            v0.position[0] = Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f;
            v0.position[1] = Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f;
            v0.position[2] = Rng::Hash::GetFloat(m_RngState);

            Vertex& v1 = geometryBufferData[i + 1];
            v1.position[0] = v0.position[0] + (Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f) * 0.3f;
            v1.position[1] = v0.position[1] + (Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f) * 0.3f;
            v1.position[2] = Rng::Hash::GetFloat(m_RngState);

            Vertex& v2 = geometryBufferData[i + 2];
            v2.position[0] = v0.position[0] + (Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f) * 0.3f;
            v2.position[1] = v0.position[1] + (Rng::Hash::GetFloat(m_RngState) * 2.0f - 1.0f) * 0.3f;
            v2.position[2] = Rng::Hash::GetFloat(m_RngState);
        }

        nri::TextureUploadDesc textureData = {};
        textureData.subresources = nullptr;
        textureData.texture = m_Texture;
        textureData.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE};

        nri::BufferUploadDesc bufferData = {};
        bufferData.buffer = m_GeometryBuffer;
        bufferData.data = geometryBufferData.data();
        bufferData.after = {nri::AccessBits::VERTEX_BUFFER};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureData, 1, &bufferData, 1));
    }

    return InitImgui(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocatorGraphics);

    if (m_IsAsyncMode)
        NRI.ResetCommandAllocator(*queuedFrame.commandAllocatorCompute);
}

void Sample::PrepareFrame(uint32_t) {
    ImGui::NewFrame();
    {
        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoResize);
        {
            ImGui::Text("Left - graphics, Right - compute");
            ImGui::BeginDisabled(!m_HasComputeQueue);
            ImGui::Checkbox("Use ASYNC compute", &m_IsAsyncMode);
            ImGui::EndDisabled();
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
}

void Sample::RenderFrame(uint32_t frameIndex) {
    const uint32_t windowWidth = GetWindowResolution().x;
    const uint32_t windowHeight = GetWindowResolution().y;
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Record command buffer #0 (graphics or compute)
    nri::TextureBarrierDesc textureBarriers[2] = {};

    textureBarriers[0].texture = swapChainTexture.texture;
    textureBarriers[0].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
    textureBarriers[0].layerNum = 1;
    textureBarriers[0].mipNum = 1;

    textureBarriers[1].texture = m_Texture;
    textureBarriers[1].layerNum = 1;
    textureBarriers[1].mipNum = 1;

    nri::BarrierGroupDesc barrierGroupDesc = {};
    barrierGroupDesc.textures = textureBarriers;

    nri::CommandBuffer& commandBuffer0 = m_IsAsyncMode ? *queuedFrame.commandBufferCompute : *queuedFrame.commandBufferGraphics[0];
    NRI.BeginCommandBuffer(commandBuffer0, m_DescriptorPool);
    {
        helper::Annotation annotation(NRI, commandBuffer0, "Compute");

        const uint32_t nx = ((windowWidth / 2) + 15) / 16;
        const uint32_t ny = (windowHeight + 15) / 16;

        NRI.CmdSetPipelineLayout(commandBuffer0, nri::BindPoint::COMPUTE, *m_SharedPipelineLayout);
        NRI.CmdSetPipeline(commandBuffer0, *m_ComputePipeline);

        nri::DescriptorSetBindingDesc descriptorSet0 = {0, m_DescriptorSet};
        NRI.CmdSetDescriptorSet(commandBuffer0, descriptorSet0);

        NRI.CmdDispatch(commandBuffer0, {nx, ny, 1});
    }
    NRI.EndCommandBuffer(commandBuffer0);

    // Record command buffer #1 (graphics)
    nri::CommandBuffer& commandBuffer1 = *queuedFrame.commandBufferGraphics[1];
    NRI.BeginCommandBuffer(commandBuffer1, m_DescriptorPool);
    {
        helper::Annotation annotation(NRI, commandBuffer1, "Graphics");

        barrierGroupDesc.textureNum = 1;
        NRI.CmdBarrier(commandBuffer1, barrierGroupDesc);

        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &swapChainTexture.colorAttachment;

        CmdCopyImguiData(commandBuffer1, *m_Streamer);

        NRI.CmdBeginRendering(commandBuffer1, attachmentsDesc);
        {
            const nri::Viewport viewport = {0.0f, 0.0f, (float)windowWidth, (float)windowHeight, 0.0f, 1.0f};
            const nri::Rect scissorRect = {0, 0, (nri::Dim_t)windowWidth, (nri::Dim_t)windowHeight};
            NRI.CmdSetViewports(commandBuffer1, &viewport, 1);
            NRI.CmdSetScissors(commandBuffer1, &scissorRect, 1);

            nri::ClearDesc clearDesc = {};
            clearDesc.colorAttachmentIndex = 0;
            clearDesc.planes = nri::PlaneBits::COLOR;
            NRI.CmdClearAttachments(commandBuffer1, &clearDesc, 1, nullptr, 0);

            NRI.CmdSetPipelineLayout(commandBuffer1, nri::BindPoint::GRAPHICS, *m_SharedPipelineLayout);
            NRI.CmdSetPipeline(commandBuffer1, *m_GraphicsPipeline);
            NRI.CmdSetIndexBuffer(commandBuffer1, *m_GeometryBuffer, 0, nri::IndexType::UINT16);

            nri::VertexBufferDesc vertexBufferDesc = {};
            vertexBufferDesc.buffer = m_GeometryBuffer;
            vertexBufferDesc.offset = 0;
            vertexBufferDesc.stride = sizeof(Vertex);
            NRI.CmdSetVertexBuffers(commandBuffer1, 0, &vertexBufferDesc, 1);

            NRI.CmdDraw(commandBuffer1, {VERTEX_NUM, 1, 0, 0});

            CmdDrawImgui(commandBuffer1, swapChainTexture.attachmentFormat, 1.0f, true);
        }
        NRI.CmdEndRendering(commandBuffer1);
    }
    NRI.EndCommandBuffer(commandBuffer1);

    // Record command buffer #2 (graphics)
    nri::CommandBuffer& commandBuffer2 = *queuedFrame.commandBufferGraphics[2];
    NRI.BeginCommandBuffer(commandBuffer2, nullptr);
    {
        helper::Annotation annotation(NRI, commandBuffer2, "Composition");

        // Resource transitions
        textureBarriers[0].before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
        textureBarriers[0].after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};

        textureBarriers[1].before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        textureBarriers[1].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};

        barrierGroupDesc.textureNum = 2;
        NRI.CmdBarrier(commandBuffer2, barrierGroupDesc);

        // Copy texture produced by compute to back buffer
        nri::TextureRegionDesc dstRegion = {};
        dstRegion.x = (uint16_t)windowWidth / 2;

        nri::TextureRegionDesc srcRegion = {};
        srcRegion.width = (uint16_t)windowWidth / 2;
        srcRegion.height = (uint16_t)windowHeight;
        srcRegion.depth = 1;

        NRI.CmdCopyTexture(commandBuffer2, *swapChainTexture.texture, &dstRegion, *m_Texture, &srcRegion);

        // Resource transitions
        textureBarriers[0].before = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        textureBarriers[0].after = {nri::AccessBits::NONE, nri::Layout::PRESENT};

        textureBarriers[1].before = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
        textureBarriers[1].after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

        barrierGroupDesc.textureNum = 2;
        NRI.CmdBarrier(commandBuffer2, barrierGroupDesc);
    }
    NRI.EndCommandBuffer(commandBuffer2);

    nri::CommandBuffer* commandBufferArray[3] = {&commandBuffer0, &commandBuffer1, &commandBuffer2};

    { // Submit work
        nri::FenceSubmitDesc textureAcquiredFence = {};
        textureAcquiredFence.fence = swapChainAcquireSemaphore;
        textureAcquiredFence.stages = nri::StageBits::COPY;

        nri::FenceSubmitDesc renderingFinishedFence = {};
        renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

        if (m_IsAsyncMode) {
            nri::FenceSubmitDesc computeFinishedFence = {};
            computeFinishedFence.fence = m_ComputeFence;
            computeFinishedFence.value = 1 + frameIndex;

            { // Submit the Compute task into the COMPUTE queue
                nri::FenceSubmitDesc waitFence = {};
                waitFence.fence = m_FrameFence;
                waitFence.value = frameIndex;

                nri::QueueSubmitDesc computeTask = {};
                computeTask.waitFences = &waitFence; // Wait for the previous frame completion before execution
                computeTask.waitFenceNum = 1;
                computeTask.commandBuffers = &commandBufferArray[0];
                computeTask.commandBufferNum = 1;
                computeTask.signalFences = &computeFinishedFence;
                computeTask.signalFenceNum = 1;

                NRI.QueueSubmit(*m_ComputeQueue, computeTask);
            }

            { // Submit the Graphics task into the GRAPHICS queue
                nri::QueueSubmitDesc graphicsTask = {};
                graphicsTask.commandBuffers = &commandBufferArray[1];
                graphicsTask.commandBufferNum = 1;

                NRI.QueueSubmit(*m_GraphicsQueue, graphicsTask);
            }

            { // Submit the Composition task into the GRAPHICS queue
                nri::FenceSubmitDesc waitFences[] = {textureAcquiredFence, computeFinishedFence};

                nri::QueueSubmitDesc compositionTask = {};
                compositionTask.waitFences = waitFences; // Wait for the Compute task completion before execution
                compositionTask.waitFenceNum = helper::GetCountOf(waitFences);
                compositionTask.commandBuffers = &commandBufferArray[2];
                compositionTask.commandBufferNum = 1;
                compositionTask.signalFences = &renderingFinishedFence;
                compositionTask.signalFenceNum = 1;

                NRI.QueueSubmit(*m_GraphicsQueue, compositionTask);
            }
        } else {
            // Submit all tasks to the GRAPHICS queue
            nri::QueueSubmitDesc allTasks = {};
            allTasks.waitFences = &textureAcquiredFence;
            allTasks.waitFenceNum = 1;
            allTasks.commandBuffers = commandBufferArray;
            allTasks.commandBufferNum = helper::GetCountOf(commandBufferArray);
            allTasks.signalFences = &renderingFinishedFence;
            allTasks.signalFenceNum = 1;

            NRI.QueueSubmit(*m_GraphicsQueue, allTasks);
        }
    }

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

SAMPLE_MAIN(Sample, 0);
