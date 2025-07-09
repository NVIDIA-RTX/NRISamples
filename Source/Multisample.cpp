// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

struct ConstantBufferLayout {
    float color[3];
    float scale;
};

struct Vertex {
    float position[2];
    float uv[2];
};

static const Vertex g_VertexData[] = {
    {-0.71f, -0.50f, 0.0f, 0.0f},
    {0.00f, 0.71f, 1.0f, 1.0f},
    {0.71f, -0.50f, 0.0f, 1.0f}};

static const uint16_t g_IndexData[] = {0, 1, 2};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
    nri::Descriptor* constantBufferView;
    nri::DescriptorSet* constantBufferDescriptorSet;
    uint64_t constantBufferViewOffset;
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
    nri::Fence* m_FrameFence = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;
    nri::DescriptorSet* m_TextureDescriptorSet = nullptr;
    nri::Descriptor* m_TextureShaderResource = nullptr;
    nri::Descriptor* m_Sampler = nullptr;
    nri::Descriptor* m_AttachmentMsaa = nullptr;
    nri::Buffer* m_ConstantBuffer = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Texture* m_Texture = nullptr;
    nri::Texture* m_TextureMsaa = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
    std::vector<nri::Memory*> m_MemoryAllocations;

    uint64_t m_GeometryOffset = 0;
    float m_Scale = 1.0f;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(*m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(*queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(*queuedFrame.commandAllocator);
            NRI.DestroyDescriptor(*queuedFrame.constantBufferView);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(*swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(*swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(*swapChainTexture.colorAttachment);
        }

        NRI.DestroyPipeline(*m_Pipeline);
        NRI.DestroyPipelineLayout(*m_PipelineLayout);
        NRI.DestroyDescriptor(*m_TextureShaderResource);
        NRI.DestroyDescriptor(*m_Sampler);
        NRI.DestroyDescriptor(*m_AttachmentMsaa);
        NRI.DestroyBuffer(*m_ConstantBuffer);
        NRI.DestroyBuffer(*m_GeometryBuffer);
        NRI.DestroyTexture(*m_Texture);
        NRI.DestroyTexture(*m_TextureMsaa);
        NRI.DestroyDescriptorPool(*m_DescriptorPool);
        NRI.DestroyFence(*m_FrameFence);

        for (nri::Memory* memory : m_MemoryAllocations)
            NRI.FreeMemory(*memory);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(*m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(*m_Streamer);

    DestroyImgui();

    nri::nriDestroyDevice(*m_Device);
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

    // Command queue
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));

    // Fences
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

    // Multisampling support
    nri::FormatSupportBits formatSupportBits = NRI.GetFormatSupport(*m_Device, swapChainFormat);
    nri::Sample_t sampleNum = 1;
    if (formatSupportBits & nri::FormatSupportBits::MULTISAMPLE_8X)
        sampleNum = 8;
    else if (formatSupportBits & nri::FormatSupportBits::MULTISAMPLE_4X)
        sampleNum = 4;
    else if (formatSupportBits & nri::FormatSupportBits::MULTISAMPLE_2X)
        sampleNum = 2;

    if (sampleNum == 1) {
        printf("Multisampling is not supported\n");
        return false;
    }

    // Queued frames
    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    // Pipeline
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    utils::ShaderCodeStorage shaderCodeStorage;
    {
        nri::DescriptorRangeDesc descriptorRangeConstant[1];
        descriptorRangeConstant[0] = {0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::ALL};

        nri::DescriptorRangeDesc descriptorRangeTexture[2];
        descriptorRangeTexture[0] = {0, 1, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER};
        descriptorRangeTexture[1] = {0, 1, nri::DescriptorType::SAMPLER, nri::StageBits::FRAGMENT_SHADER};

        nri::DescriptorSetDesc descriptorSetDescs[] = {
            {0, descriptorRangeConstant, helper::GetCountOf(descriptorRangeConstant)},
            {1, descriptorRangeTexture, helper::GetCountOf(descriptorRangeTexture)},
        };

        nri::RootConstantDesc rootConstant = {1, sizeof(float), nri::StageBits::FRAGMENT_SHADER};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.rootConstantNum = 1;
        pipelineLayoutDesc.rootConstants = &rootConstant;
        pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

        nri::VertexStreamDesc vertexStreamDesc = {};
        vertexStreamDesc.bindingSlot = 0;

        nri::VertexAttributeDesc vertexAttributeDesc[2] = {};
        {
            vertexAttributeDesc[0].format = nri::Format::RG32_SFLOAT;
            vertexAttributeDesc[0].streamIndex = 0;
            vertexAttributeDesc[0].offset = helper::GetOffsetOf(&Vertex::position);
            vertexAttributeDesc[0].d3d = {"POSITION", 0};
            vertexAttributeDesc[0].vk.location = {0};

            vertexAttributeDesc[1].format = nri::Format::RG32_SFLOAT;
            vertexAttributeDesc[1].streamIndex = 0;
            vertexAttributeDesc[1].offset = helper::GetOffsetOf(&Vertex::uv);
            vertexAttributeDesc[1].d3d = {"TEXCOORD", 0};
            vertexAttributeDesc[1].vk.location = {1};
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
        colorAttachmentDesc.blendEnabled = true;
        colorAttachmentDesc.colorBlend = {nri::BlendFactor::SRC_ALPHA, nri::BlendFactor::ONE_MINUS_SRC_ALPHA, nri::BlendOp::ADD};

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = &colorAttachmentDesc;
        outputMergerDesc.colorNum = 1;

        nri::ShaderDesc shaderStages[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "TriangleFlexibleMultiview.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "Triangle.fs", shaderCodeStorage),
        };

        nri::MultisampleDesc multisampleDesc = {};
        multisampleDesc.sampleMask = nri::ALL_SAMPLES;
        multisampleDesc.sampleNum = sampleNum;
        multisampleDesc.alphaToCoverage = false;

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
        graphicsPipelineDesc.vertexInput = &vertexInputDesc;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.multisample = &multisampleDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);

        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_Pipeline));
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = GetQueuedFrameNum() + 1;
        descriptorPoolDesc.constantBufferMaxNum = GetQueuedFrameNum();
        descriptorPoolDesc.textureMaxNum = 1;
        descriptorPoolDesc.samplerMaxNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    }

    // Load texture
    utils::Texture texture;
    std::string path = utils::GetFullPath("wood.dds", utils::DataFolder::TEXTURES);
    if (!utils::LoadTexture(path, texture))
        return false;

    // Resources
    const uint32_t constantBufferSize = helper::Align((uint32_t)sizeof(ConstantBufferLayout), deviceDesc.memoryAlignment.constantBufferOffset);
    constexpr uint64_t indexDataSize = sizeof(g_IndexData);
    constexpr uint64_t indexDataAlignedSize = helper::Align(indexDataSize, 16);
    constexpr uint64_t vertexDataSize = sizeof(g_VertexData);
    {
        { // Texture MSAA
            nri::TextureDesc textureDesc = {};
            textureDesc.type = nri::TextureType::TEXTURE_2D;
            textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT;
            textureDesc.format = swapChainFormat;
            textureDesc.width = (nri::Dim_t)GetWindowResolution().x;
            textureDesc.height = (nri::Dim_t)GetWindowResolution().y;
            textureDesc.sampleNum = sampleNum;
            textureDesc.mipNum = 1;

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_TextureMsaa));
        }

        { // Read-only texture
            nri::TextureDesc textureDesc = {};
            textureDesc.type = nri::TextureType::TEXTURE_2D;
            textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
            textureDesc.format = texture.GetFormat();
            textureDesc.width = texture.GetWidth();
            textureDesc.height = texture.GetHeight();
            textureDesc.mipNum = texture.GetMipNum();

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_Texture));
        }

        { // Constant buffer
            nri::BufferDesc bufferDesc = {};
            bufferDesc.size = constantBufferSize * GetQueuedFrameNum();
            bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;

            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));
        }

        { // Geometry buffer
            nri::BufferDesc bufferDesc = {};
            bufferDesc.size = indexDataAlignedSize + vertexDataSize;
            bufferDesc.usage = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;

            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_GeometryBuffer));

            m_GeometryOffset = indexDataAlignedSize;
        }

        // Bind to memory
        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_ConstantBuffer;

        m_MemoryAllocations.resize(1, nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data()));

        nri::Texture* textures[] = {m_TextureMsaa, m_Texture};

        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_GeometryBuffer;
        resourceGroupDesc.textureNum = helper::GetCountOf(textures);
        resourceGroupDesc.textures = textures;

        m_MemoryAllocations.resize(1 + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + 1));
    }

    // Descriptors
    {
        { // Attachment MSAA
            nri::Texture2DViewDesc texture2DViewDesc = {m_TextureMsaa, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_AttachmentMsaa));
        }

        { // Read-only texture
            nri::Texture2DViewDesc texture2DViewDesc = {m_Texture, nri::Texture2DViewType::SHADER_RESOURCE_2D, texture.GetFormat()};

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_TextureShaderResource));
        }

        { // Sampler
            nri::SamplerDesc samplerDesc = {};
            samplerDesc.addressModes = {nri::AddressMode::MIRRORED_REPEAT, nri::AddressMode::MIRRORED_REPEAT};
            samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
            samplerDesc.anisotropy = 4;
            samplerDesc.mipMax = 16.0f;

            NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, m_Sampler));
        }

        // Constant buffer
        for (uint32_t i = 0; i < GetQueuedFrameNum(); i++) {
            nri::BufferViewDesc bufferViewDesc = {};
            bufferViewDesc.buffer = m_ConstantBuffer;
            bufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
            bufferViewDesc.offset = i * constantBufferSize;
            bufferViewDesc.size = constantBufferSize;

            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, m_QueuedFrames[i].constantBufferView));

            m_QueuedFrames[i].constantBufferViewOffset = bufferViewDesc.offset;
        }
    }

    { // Descriptor sets
        // Texture
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &m_TextureDescriptorSet, 1, 0));

        nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDescs[2] = {};
        descriptorRangeUpdateDescs[0].descriptorNum = 1;
        descriptorRangeUpdateDescs[0].descriptors = &m_TextureShaderResource;

        descriptorRangeUpdateDescs[1].descriptorNum = 1;
        descriptorRangeUpdateDescs[1].descriptors = &m_Sampler;
        NRI.UpdateDescriptorRanges(*m_TextureDescriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDescs), descriptorRangeUpdateDescs);

        // Constant buffer
        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &queuedFrame.constantBufferDescriptorSet, 1, 0));

            nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&queuedFrame.constantBufferView, 1};
            NRI.UpdateDescriptorRanges(*queuedFrame.constantBufferDescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
        }
    }

    { // Upload data
        std::vector<uint8_t> geometryBufferData(indexDataAlignedSize + vertexDataSize);
        memcpy(&geometryBufferData[0], g_IndexData, indexDataSize);
        memcpy(&geometryBufferData[indexDataAlignedSize], g_VertexData, vertexDataSize);

        std::array<nri::TextureSubresourceUploadDesc, 16> subresources;
        for (uint32_t mip = 0; mip < texture.GetMipNum(); mip++)
            texture.GetSubresource(subresources[mip], mip);

        nri::TextureUploadDesc textureData = {};
        textureData.subresources = subresources.data();
        textureData.texture = m_Texture;
        textureData.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

        nri::BufferUploadDesc bufferData = {};
        bufferData.buffer = m_GeometryBuffer;
        bufferData.data = geometryBufferData.data();
        bufferData.after = {nri::AccessBits::INDEX_BUFFER | nri::AccessBits::VERTEX_BUFFER};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureData, 1, &bufferData, 1));
    }

    // User interface
    bool initialized = InitImgui(*m_Device);

    return initialized;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::PrepareFrame(uint32_t) {
    ImGui::NewFrame();
    {
        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoResize);
        {
            ImGui::SliderFloat("Scale", &m_Scale, 0.75f, 1.25f);
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::Dim_t w = (nri::Dim_t)GetWindowResolution().x;
    nri::Dim_t h = (nri::Dim_t)GetWindowResolution().y;

    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Update constants
    ConstantBufferLayout* commonConstants = (ConstantBufferLayout*)NRI.MapBuffer(*m_ConstantBuffer, queuedFrame.constantBufferViewOffset, sizeof(ConstantBufferLayout));
    if (commonConstants) {
        commonConstants->color[0] = 0.8f;
        commonConstants->color[1] = 0.5f;
        commonConstants->color[2] = 0.1f;
        commonConstants->scale = m_Scale;

        NRI.UnmapBuffer(*m_ConstantBuffer);
    }

    // Record commands
    nri::CommandBuffer* commandBuffer = queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(*commandBuffer, m_DescriptorPool);
    {
        { // Barriers
            nri::TextureBarrierDesc textureBarriers[2] = {};

            textureBarriers[0].texture = m_TextureMsaa;
            if (frameIndex != 0)
                textureBarriers[0].before = {nri::AccessBits::RESOLVE_SOURCE, nri::Layout::RESOLVE_SOURCE};
            textureBarriers[0].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};

            textureBarriers[1].texture = swapChainTexture.texture;
            textureBarriers[1].after = {nri::AccessBits::RESOLVE_DESTINATION, nri::Layout::RESOLVE_DESTINATION};

            nri::BarrierGroupDesc barrierGroup = {};
            barrierGroup.textureNum = helper::GetCountOf(textureBarriers);
            barrierGroup.textures = textureBarriers;

            NRI.CmdBarrier(*commandBuffer, barrierGroup);
        }

        { // Multisampling rendering
            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &m_AttachmentMsaa;

            NRI.CmdBeginRendering(*commandBuffer, attachmentsDesc);
            {
                {
                    helper::Annotation annotation(NRI, *commandBuffer, "Clears");

                    nri::ClearDesc clearDesc = {};
                    clearDesc.planes = nri::PlaneBits::COLOR;
                    clearDesc.value.color.f = {1.0f, 1.0f, 1.0f, 1.0f};

                    NRI.CmdClearAttachments(*commandBuffer, &clearDesc, 1, nullptr, 0);
                }

                {
                    helper::Annotation annotation(NRI, *commandBuffer, "Triangle");

                    const float transparency = 1.0f;
                    NRI.CmdSetPipelineLayout(*commandBuffer, *m_PipelineLayout);
                    NRI.CmdSetPipeline(*commandBuffer, *m_Pipeline);
                    NRI.CmdSetRootConstants(*commandBuffer, 0, &transparency, 4);
                    NRI.CmdSetIndexBuffer(*commandBuffer, *m_GeometryBuffer, 0, nri::IndexType::UINT16);

                    nri::VertexBufferDesc vertexBufferDesc = {};
                    vertexBufferDesc.buffer = m_GeometryBuffer;
                    vertexBufferDesc.offset = m_GeometryOffset;
                    vertexBufferDesc.stride = sizeof(Vertex);
                    NRI.CmdSetVertexBuffers(*commandBuffer, 0, &vertexBufferDesc, 1);

                    NRI.CmdSetDescriptorSet(*commandBuffer, 0, *queuedFrame.constantBufferDescriptorSet, nullptr);
                    NRI.CmdSetDescriptorSet(*commandBuffer, 1, *m_TextureDescriptorSet, nullptr);

                    const nri::Viewport viewport = {0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
                    NRI.CmdSetViewports(*commandBuffer, &viewport, 1);

                    nri::Rect scissor = {0, 0, w, h};
                    NRI.CmdSetScissors(*commandBuffer, &scissor, 1);

                    NRI.CmdDrawIndexed(*commandBuffer, {3, 1, 0, 0, 0});
                }
            }
            NRI.CmdEndRendering(*commandBuffer);
        }

        { // Barriers
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = m_TextureMsaa;
            textureBarrier.before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
            textureBarrier.after = {nri::AccessBits::RESOLVE_SOURCE, nri::Layout::RESOLVE_SOURCE};

            nri::BarrierGroupDesc barrierGroup = {};
            barrierGroup.textureNum = 1;
            barrierGroup.textures = &textureBarrier;

            NRI.CmdBarrier(*commandBuffer, barrierGroup);
        }

        { // Resolve
            NRI.CmdResolveTexture(*commandBuffer, *swapChainTexture.texture, nullptr, *m_TextureMsaa, nullptr);
        }

        { // Barriers
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = swapChainTexture.texture;
            textureBarrier.before = {nri::AccessBits::RESOLVE_DESTINATION, nri::Layout::RESOLVE_DESTINATION};
            textureBarrier.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};

            nri::BarrierGroupDesc barrierGroup = {};
            barrierGroup.textureNum = 1;
            barrierGroup.textures = &textureBarrier;

            NRI.CmdBarrier(*commandBuffer, barrierGroup);
        }

        { // Composition
            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &swapChainTexture.colorAttachment;

            CmdCopyImguiData(*commandBuffer, *m_Streamer);

            NRI.CmdBeginRendering(*commandBuffer, attachmentsDesc);
            {
                helper::Annotation annotation(NRI, *commandBuffer, "UI");

                CmdDrawImgui(*commandBuffer, swapChainTexture.attachmentFormat, 1.0f, true);
            }
            NRI.CmdEndRendering(*commandBuffer);
        }

        { // Barriers
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = swapChainTexture.texture;
            textureBarrier.before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
            textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::PRESENT};

            nri::BarrierGroupDesc barrierGroup = {};
            barrierGroup.textureNum = 1;
            barrierGroup.textures = &textureBarrier;

            NRI.CmdBarrier(*commandBuffer, barrierGroup);
        }
    }
    NRI.EndCommandBuffer(*commandBuffer);

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
