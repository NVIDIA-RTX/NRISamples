// © 2021 NVIDIA Corporation

#include "NRI.hlsl"
#include "NRIFramework.h"

#include <array>

constexpr uint32_t GLOBAL_DESCRIPTOR_SET = 0;
constexpr uint32_t MATERIAL_DESCRIPTOR_SET = 1;
constexpr float CLEAR_DEPTH = 0.0f;
constexpr uint32_t TEXTURES_PER_MATERIAL = 4;

constexpr uint32_t CONSTANT_BUFFER = 0;
constexpr uint32_t READBACK_BUFFER = 1;
constexpr uint32_t INDEX_BUFFER = 2;
constexpr uint32_t VERTEX_BUFFER = 3;

struct GlobalConstantBufferLayout {
    float4x4 gWorldToClip;
    float3 gCameraPos;
};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
    uint32_t globalConstantBufferViewOffsets;
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
    nri::Descriptor* m_DepthAttachment = nullptr;
    nri::Descriptor* m_ShadingRateAttachment = nullptr;
    nri::QueryPool* m_QueryPool = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<SwapChainTexture> m_SwapChainTextures;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<nri::Descriptor*> m_Descriptors;

    nri::Format m_DepthFormat = nri::Format::UNKNOWN;

    utils::Scene m_Scene;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(*m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(*queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(*queuedFrame.commandAllocator);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(*swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(*swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(*swapChainTexture.colorAttachment);
        }

        for (size_t i = 0; i < m_Descriptors.size(); i++)
            NRI.DestroyDescriptor(*m_Descriptors[i]);

        for (size_t i = 0; i < m_Textures.size(); i++)
            NRI.DestroyTexture(*m_Textures[i]);

        for (size_t i = 0; i < m_Buffers.size(); i++)
            NRI.DestroyBuffer(*m_Buffers[i]);

        for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
            NRI.FreeMemory(*m_MemoryAllocations[i]);

        for (size_t i = 0; i < m_Pipelines.size(); i++)
            NRI.DestroyPipeline(*m_Pipelines[i]);

        NRI.DestroyQueryPool(*m_QueryPool);
        NRI.DestroyPipelineLayout(*m_PipelineLayout);
        NRI.DestroyDescriptorPool(*m_DescriptorPool);
        NRI.DestroyFence(*m_FrameFence);
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

    m_DepthFormat = nri::GetSupportedDepthFormat(NRI, *m_Device, 24, true);

    { // Swap chain
        nri::SwapChainDesc swapChainDesc = {};
        swapChainDesc.window = GetWindow();
        swapChainDesc.queue = m_GraphicsQueue;
        swapChainDesc.format = nri::SwapChainFormat::BT709_G22_10BIT;
        swapChainDesc.flags = (m_Vsync ? nri::SwapChainBits::VSYNC : nri::SwapChainBits::NONE) | nri::SwapChainBits::ALLOW_TEARING;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = GetOptimalSwapChainTextureNum();
        swapChainDesc.queuedFrameNum = GetQueuedFrameNum();
        NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));
    }

    uint32_t swapChainTextureNum;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);

    nri::Format swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

    // Queued frames
    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    { // Pipeline layout
        nri::DescriptorRangeDesc globalDescriptorRange[2];
        globalDescriptorRange[0] = {0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::ALL};
        globalDescriptorRange[1] = {0, 1, nri::DescriptorType::SAMPLER, nri::StageBits::FRAGMENT_SHADER};

        nri::DescriptorRangeDesc materialDescriptorRange[1];
        materialDescriptorRange[0] = {0, TEXTURES_PER_MATERIAL, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER};

        nri::DescriptorSetDesc descriptorSetDescs[] = {
            {0, globalDescriptorRange, helper::GetCountOf(globalDescriptorRange)},
            {1, materialDescriptorRange, helper::GetCountOf(materialDescriptorRange)},
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    // Pipeline
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    utils::ShaderCodeStorage shaderCodeStorage;
    {
        nri::VertexStreamDesc vertexStreamDesc = {};
        vertexStreamDesc.bindingSlot = 0;

        nri::VertexAttributeDesc vertexAttributeDesc[4] = {};
        {
            vertexAttributeDesc[0].format = nri::Format::RGB32_SFLOAT;
            vertexAttributeDesc[0].offset = helper::GetOffsetOf(&utils::Vertex::pos);
            vertexAttributeDesc[0].d3d = {"POSITION", 0};
            vertexAttributeDesc[0].vk = {0};

            vertexAttributeDesc[1].format = nri::Format::RG16_SFLOAT;
            vertexAttributeDesc[1].offset = helper::GetOffsetOf(&utils::Vertex::uv);
            vertexAttributeDesc[1].d3d = {"TEXCOORD", 0};
            vertexAttributeDesc[1].vk = {1};

            vertexAttributeDesc[2].format = nri::Format::R10_G10_B10_A2_UNORM;
            vertexAttributeDesc[2].offset = helper::GetOffsetOf(&utils::Vertex::N);
            vertexAttributeDesc[2].d3d = {"NORMAL", 0};
            vertexAttributeDesc[2].vk = {2};

            vertexAttributeDesc[3].format = nri::Format::R10_G10_B10_A2_UNORM;
            vertexAttributeDesc[3].offset = helper::GetOffsetOf(&utils::Vertex::T);
            vertexAttributeDesc[3].d3d = {"TANGENT", 0};
            vertexAttributeDesc[3].vk = {3};
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
        rasterizationDesc.frontCounterClockwise = true;
        rasterizationDesc.shadingRate = deviceDesc.tiers.shadingRate != 0;

        nri::MultisampleDesc multisampleDesc = {};
        multisampleDesc.sampleNum = 1;
        multisampleDesc.sampleMask = nri::ALL_SAMPLES;
        multisampleDesc.sampleLocations = deviceDesc.tiers.sampleLocations >= 2;

        nri::ColorAttachmentDesc colorAttachmentDesc = {};
        colorAttachmentDesc.format = swapChainFormat;
        colorAttachmentDesc.colorWriteMask = nri::ColorWriteBits::RGBA;

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = &colorAttachmentDesc;
        outputMergerDesc.colorNum = 1;
        outputMergerDesc.depthStencilFormat = m_DepthFormat;
        outputMergerDesc.depth.write = true;
        outputMergerDesc.depth.compareOp = CLEAR_DEPTH == 1.0f ? nri::CompareOp::LESS : nri::CompareOp::GREATER;

        nri::ShaderDesc shaderStages[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "Forward.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "Forward.fs", shaderCodeStorage),
        };

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
        graphicsPipelineDesc.vertexInput = &vertexInputDesc;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.multisample = &multisampleDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);

        nri::Pipeline* pipeline;

        { // Opaque
            NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, pipeline));
            m_Pipelines.push_back(pipeline);
        }

        { // Alpha opaque
            shaderStages[1] = utils::LoadShader(deviceDesc.graphicsAPI, "ForwardDiscard.fs", shaderCodeStorage);

            rasterizationDesc.cullMode = nri::CullMode::NONE;
            outputMergerDesc.depth.write = true;
            colorAttachmentDesc.blendEnabled = false;
            NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, pipeline));
            m_Pipelines.push_back(pipeline);
        }

        shaderStages[1] = utils::LoadShader(deviceDesc.graphicsAPI, "ForwardTransparent.fs", shaderCodeStorage);

        { // Transparent
            rasterizationDesc.cullMode = nri::CullMode::NONE;
            outputMergerDesc.depth.write = false;
            colorAttachmentDesc.blendEnabled = true;
            colorAttachmentDesc.colorBlend = {nri::BlendFactor::SRC_ALPHA, nri::BlendFactor::ONE_MINUS_SRC_ALPHA, nri::BlendOp::ADD};
            NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, pipeline));
            m_Pipelines.push_back(pipeline);
        }
    }

    // Scene
    std::string sceneFile = utils::GetFullPath(m_SceneFile, utils::DataFolder::SCENES);
    NRI_ABORT_ON_FALSE(utils::LoadScene(sceneFile, m_Scene, false));

    // Camera
    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, false);

    const uint32_t textureNum = (uint32_t)m_Scene.textures.size();
    const uint32_t materialNum = (uint32_t)m_Scene.materials.size();

    // Textures
    for (const utils::Texture* textureData : m_Scene.textures) {
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        textureDesc.format = textureData->GetFormat();
        textureDesc.width = textureData->GetWidth();
        textureDesc.height = textureData->GetHeight();
        textureDesc.mipNum = textureData->GetMipNum();
        textureDesc.layerNum = textureData->GetArraySize();

        nri::Texture* texture;
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, texture));
        m_Textures.push_back(texture);
    }

    // Depth attachment
    nri::Texture* depthTexture = nullptr;
    {
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::DEPTH_STENCIL_ATTACHMENT;
        textureDesc.format = m_DepthFormat;
        textureDesc.width = (uint16_t)GetWindowResolution().x;
        textureDesc.height = (uint16_t)GetWindowResolution().y;
        textureDesc.mipNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, depthTexture));
        m_Textures.push_back(depthTexture);
    }

    // Shading rate attachment
    nri::Texture* shadingRateTexture = nullptr;
    uint8_t* shadingRateData = nullptr;
    uint32_t shadingRateTexWidth = 0;
    uint32_t shadingRateTexHeight = 0;
    if (deviceDesc.tiers.shadingRate >= 2) {
        shadingRateTexWidth = (GetWindowResolution().x + deviceDesc.other.shadingRateAttachmentTileSize - 1) / deviceDesc.other.shadingRateAttachmentTileSize;
        shadingRateTexHeight = (GetWindowResolution().y + deviceDesc.other.shadingRateAttachmentTileSize - 1) / deviceDesc.other.shadingRateAttachmentTileSize;

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADING_RATE_ATTACHMENT;
        textureDesc.format = nri::Format::R8_UINT;
        textureDesc.width = (uint16_t)shadingRateTexWidth;
        textureDesc.height = (uint16_t)shadingRateTexHeight;
        textureDesc.mipNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, shadingRateTexture));
        m_Textures.push_back(shadingRateTexture);

        // Fill with some values
        shadingRateData = (uint8_t*)malloc(shadingRateTexWidth * shadingRateTexHeight);

        uint8_t* p = shadingRateData;
        for (uint32_t j = 0; j < shadingRateTexHeight; j++) {
            for (uint32_t i = 0; i < shadingRateTexWidth; i++) {
                *p = i < shadingRateTexWidth / 2 ? NRI_SHADING_RATE(0, 0) : NRI_SHADING_RATE(2, 2);
                p++;
            }
        }
    }

    const uint32_t constantBufferSize = helper::Align((uint32_t)sizeof(GlobalConstantBufferLayout), deviceDesc.memoryAlignment.constantBufferOffset);

    { // Buffers
        // CONSTANT_BUFFER
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = constantBufferSize * GetQueuedFrameNum();
        bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
        nri::Buffer* buffer;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
        m_Buffers.push_back(buffer);

        // READBACK_BUFFER
        bufferDesc.size = sizeof(nri::PipelineStatisticsDesc);
        bufferDesc.usage = nri::BufferUsageBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
        m_Buffers.push_back(buffer);

        // INDEX_BUFFER
        bufferDesc.size = helper::GetByteSizeOf(m_Scene.indices);
        bufferDesc.usage = nri::BufferUsageBits::INDEX_BUFFER;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
        m_Buffers.push_back(buffer);

        // VERTEX_BUFFER
        bufferDesc.size = helper::GetByteSizeOf(m_Scene.vertices);
        bufferDesc.usage = nri::BufferUsageBits::VERTEX_BUFFER;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
        m_Buffers.push_back(buffer);
    }

    { // Memory
        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_Buffers[CONSTANT_BUFFER];

        size_t baseAllocation = m_MemoryAllocations.size();
        m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_READBACK;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_Buffers[READBACK_BUFFER];

        baseAllocation = m_MemoryAllocations.size();
        m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        resourceGroupDesc.bufferNum = 2;
        resourceGroupDesc.buffers = &m_Buffers[INDEX_BUFFER];
        resourceGroupDesc.textureNum = (uint32_t)m_Textures.size();
        resourceGroupDesc.textures = m_Textures.data();

        baseAllocation = m_MemoryAllocations.size();
        uint32_t allocationNum = NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc);
        m_MemoryAllocations.resize(baseAllocation + allocationNum, nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));
    }

    // Create descriptors
    nri::Descriptor* anisotropicSampler;
    nri::Descriptor* constantBufferViews[8] = {};
    {
        // Material textures
        m_Descriptors.resize(textureNum);
        for (uint32_t i = 0; i < textureNum; i++) {
            const utils::Texture& texture = *m_Scene.textures[i];

            nri::Texture2DViewDesc texture2DViewDesc = {m_Textures[i], nri::Texture2DViewType::SHADER_RESOURCE_2D, texture.GetFormat()};
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_Descriptors[i]));
        }

        // Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
        samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
        samplerDesc.anisotropy = 8;
        samplerDesc.mipMax = 16.0f;
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, anisotropicSampler));
        m_Descriptors.push_back(anisotropicSampler);

        // Constant buffer
        for (uint32_t i = 0; i < GetQueuedFrameNum(); i++) {
            m_QueuedFrames[i].globalConstantBufferViewOffsets = i * constantBufferSize;

            nri::BufferViewDesc bufferViewDesc = {};
            bufferViewDesc.buffer = m_Buffers[CONSTANT_BUFFER];
            bufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
            bufferViewDesc.offset = i * constantBufferSize;
            bufferViewDesc.size = constantBufferSize;
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, constantBufferViews[i]));
            m_Descriptors.push_back(constantBufferViews[i]);
        }

        { // Depth buffer
            nri::Texture2DViewDesc texture2DViewDesc = {depthTexture, nri::Texture2DViewType::DEPTH_STENCIL_ATTACHMENT, m_DepthFormat};

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_DepthAttachment));
            m_Descriptors.push_back(m_DepthAttachment);
        }

        // Shading rate attachment
        if (shadingRateTexture) {
            nri::Texture2DViewDesc texture2DViewDesc = {shadingRateTexture, nri::Texture2DViewType::SHADING_RATE_ATTACHMENT, nri::Format::R8_UINT};

            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_ShadingRateAttachment));
            m_Descriptors.push_back(m_ShadingRateAttachment);
        }

        // Swap chain
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

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = materialNum + GetQueuedFrameNum();
        descriptorPoolDesc.textureMaxNum = materialNum * TEXTURES_PER_MATERIAL;
        descriptorPoolDesc.samplerMaxNum = GetQueuedFrameNum();
        descriptorPoolDesc.constantBufferMaxNum = GetQueuedFrameNum();

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    }

    { // Descriptor sets
        m_DescriptorSets.resize(GetQueuedFrameNum() + materialNum);

        // Global
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, GLOBAL_DESCRIPTOR_SET, &m_DescriptorSets[0], GetQueuedFrameNum(), 0));

        for (uint32_t i = 0; i < GetQueuedFrameNum(); i++) {
            nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDescs[2] = {};
            descriptorRangeUpdateDescs[0].descriptorNum = 1;
            descriptorRangeUpdateDescs[0].descriptors = &constantBufferViews[i];
            descriptorRangeUpdateDescs[1].descriptorNum = 1;
            descriptorRangeUpdateDescs[1].descriptors = &anisotropicSampler;

            NRI.UpdateDescriptorRanges(*m_DescriptorSets[i], 0, helper::GetCountOf(descriptorRangeUpdateDescs), descriptorRangeUpdateDescs);
        }

        // Material
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, MATERIAL_DESCRIPTOR_SET, &m_DescriptorSets[GetQueuedFrameNum()], materialNum, 0));

        for (uint32_t i = 0; i < materialNum; i++) {
            const utils::Material& material = m_Scene.materials[i];

            nri::Descriptor* materialTextures[TEXTURES_PER_MATERIAL] = {
                m_Descriptors[material.baseColorTexIndex],
                m_Descriptors[material.roughnessMetalnessTexIndex],
                m_Descriptors[material.normalTexIndex],
                m_Descriptors[material.emissiveTexIndex],
            };

            nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDescs = {};
            descriptorRangeUpdateDescs.descriptorNum = helper::GetCountOf(materialTextures);
            descriptorRangeUpdateDescs.descriptors = materialTextures;
            NRI.UpdateDescriptorRanges(*m_DescriptorSets[GetQueuedFrameNum() + i], 0, 1, &descriptorRangeUpdateDescs);
        }
    }

    { // Upload data
        std::vector<nri::TextureUploadDesc> textureData(textureNum + 2);

        uint32_t subresourceNum = 0;
        for (uint32_t i = 0; i < textureNum; i++) {
            const utils::Texture& texture = *m_Scene.textures[i];
            subresourceNum += texture.GetArraySize() * texture.GetMipNum();
        }

        std::vector<nri::TextureSubresourceUploadDesc> subresources(subresourceNum);
        nri::TextureSubresourceUploadDesc* subresourceBegin = subresources.data();

        // Material textures
        uint32_t i = 0;
        for (; i < textureNum; i++) {
            const utils::Texture& texture = *m_Scene.textures[i];

            for (uint32_t slice = 0; slice < texture.GetArraySize(); slice++) {
                for (uint32_t mip = 0; mip < texture.GetMipNum(); mip++)
                    texture.GetSubresource(subresourceBegin[slice * texture.GetMipNum() + mip], mip, slice);
            }

            textureData[i] = {};
            textureData[i].subresources = subresourceBegin;
            textureData[i].texture = m_Textures[i];
            textureData[i].after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

            subresourceBegin += texture.GetArraySize() * texture.GetMipNum();
        }

        // Depth attachment
        textureData[i] = {};
        textureData[i].subresources = nullptr;
        textureData[i].texture = depthTexture;
        textureData[i].after = {nri::AccessBits::DEPTH_STENCIL_ATTACHMENT_WRITE, nri::Layout::DEPTH_STENCIL_ATTACHMENT};
        i++;

        // Shading rate attachment
        nri::TextureSubresourceUploadDesc shadingRateSubresource = {};
        shadingRateSubresource.slices = shadingRateData;
        shadingRateSubresource.sliceNum = 1;
        shadingRateSubresource.rowPitch = shadingRateTexWidth;
        shadingRateSubresource.slicePitch = shadingRateTexWidth * shadingRateTexHeight;

        if (shadingRateTexture) {
            textureData[i] = {};
            textureData[i].subresources = &shadingRateSubresource;
            textureData[i].texture = shadingRateTexture;
            textureData[i].after = {nri::AccessBits::SHADING_RATE_ATTACHMENT, nri::Layout::SHADING_RATE_ATTACHMENT};
            i++;
        }

        // Buffers
        nri::BufferUploadDesc bufferData[] = {
            {m_Scene.vertices.data(), m_Buffers[VERTEX_BUFFER], {nri::AccessBits::VERTEX_BUFFER}},
            {m_Scene.indices.data(), m_Buffers[INDEX_BUFFER], {nri::AccessBits::INDEX_BUFFER}},
        };

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, textureData.data(), i, bufferData, helper::GetCountOf(bufferData)));
    }

    // Pipeline statistics
    if (deviceDesc.features.pipelineStatistics) {
        nri::QueryPoolDesc queryPoolDesc = {};
        queryPoolDesc.queryType = nri::QueryType::PIPELINE_STATISTICS;
        queryPoolDesc.capacity = GetQueuedFrameNum();

        NRI_ABORT_ON_FAILURE(NRI.CreateQueryPool(*m_Device, queryPoolDesc, m_QueryPool));
    }

    m_Scene.UnloadGeometryData();
    m_Scene.UnloadTextureData();

    if (shadingRateData)
        free(shadingRateData);

    return InitImgui(*m_Device);
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::PrepareFrame(uint32_t frameIndex) {
    ImGui::NewFrame();
    {
        nri::PipelineStatisticsDesc* pipelineStats = (nri::PipelineStatisticsDesc*)NRI.MapBuffer(*m_Buffers[READBACK_BUFFER], 0, sizeof(nri::PipelineStatisticsDesc));

        ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(0, 0));
        ImGui::Begin("Stats");
        {
            ImGui::Text("Input vertices               : %" PRIu64, pipelineStats->inputVertexNum);
            ImGui::Text("Input primitives             : %" PRIu64, pipelineStats->inputPrimitiveNum);
            ImGui::Text("Vertex shader invocations    : %" PRIu64, pipelineStats->vertexShaderInvocationNum);
            ImGui::Text("Rasterizer input primitives  : %" PRIu64, pipelineStats->rasterizerInPrimitiveNum);
            ImGui::Text("Rasterizer output primitives : %" PRIu64, pipelineStats->rasterizerOutPrimitiveNum);
            ImGui::Text("Fragment shader invocations  : %" PRIu64, pipelineStats->fragmentShaderInvocationNum);
        }
        ImGui::End();

        NRI.UnmapBuffer(*m_Buffers[READBACK_BUFFER]);
    }
    ImGui::EndFrame();
    ImGui::Render();

    CameraDesc desc = {};
    desc.aspectRatio = float(GetWindowResolution().x) / float(GetWindowResolution().y);
    desc.horizontalFov = 90.0f;
    desc.nearZ = 0.1f;
    desc.isReversedZ = (CLEAR_DEPTH == 0.0f);
    GetCameraDescFromInputDevices(desc);

    m_Camera.Update(desc, frameIndex);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const uint32_t nextQueuedFrameIndex = (frameIndex + 1) % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];
    const uint32_t windowWidth = GetWindowResolution().x;
    const uint32_t windowHeight = GetWindowResolution().y;
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Update constants
    const uint64_t rangeOffset = m_QueuedFrames[queuedFrameIndex].globalConstantBufferViewOffsets;
    auto constants = (GlobalConstantBufferLayout*)NRI.MapBuffer(*m_Buffers[CONSTANT_BUFFER], rangeOffset, sizeof(GlobalConstantBufferLayout));
    if (constants) {
        constants->gWorldToClip = m_Camera.state.mWorldToClip * m_Scene.mSceneToWorld;
        constants->gCameraPos = m_Camera.state.position;

        NRI.UnmapBuffer(*m_Buffers[CONSTANT_BUFFER]);
    }

    // Record
    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        helper::Annotation annotation(NRI, commandBuffer, "Scene");

        nri::TextureBarrierDesc textureBarriers = {};
        textureBarriers.texture = swapChainTexture.texture;
        textureBarriers.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        textureBarriers.layerNum = 1;
        textureBarriers.mipNum = 1;

        nri::BarrierGroupDesc barrierGroupDesc = {};
        barrierGroupDesc.textureNum = 1;
        barrierGroupDesc.textures = &textureBarriers;

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);

        // Test PSL // TODO: D3D11 gets DEVICE_REMOVED if VRS is used with PSL...
        if (deviceDesc.tiers.sampleLocations >= 2 && deviceDesc.graphicsAPI != nri::GraphicsAPI::D3D11) {
            static const nri::SampleLocation samplePos[4] = {
                {-6, -2},
                {-2, 6},
                {6, 2},
                {2, -6},
            };

            NRI.CmdSetSampleLocations(commandBuffer, samplePos + (frameIndex % 4), 1, 1);
        }

        // Test VRS (per pipeline)
        if (deviceDesc.tiers.shadingRate) {
            nri::ShadingRateDesc shadingRateDesc = {};
            if (deviceDesc.tiers.shadingRate >= 2) {
                shadingRateDesc.shadingRate = nri::ShadingRate::FRAGMENT_SIZE_1X1;
                shadingRateDesc.primitiveCombiner = nri::ShadingRateCombiner::REPLACE;
                shadingRateDesc.attachmentCombiner = nri::ShadingRateCombiner::REPLACE;
            } else
                shadingRateDesc.shadingRate = nri::ShadingRate::FRAGMENT_SIZE_2X2;

            NRI.CmdSetShadingRate(commandBuffer, shadingRateDesc);
        }

        // Test pipeline stats query
        if (m_QueryPool) {
            NRI.CmdResetQueries(commandBuffer, *m_QueryPool, queuedFrameIndex, 1);
            NRI.CmdBeginQuery(commandBuffer, *m_QueryPool, queuedFrameIndex);
        }

        { // Rendering
            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &swapChainTexture.colorAttachment;
            attachmentsDesc.depthStencil = m_DepthAttachment;

            if (deviceDesc.tiers.shadingRate >= 2)
                attachmentsDesc.shadingRate = m_ShadingRateAttachment;

            NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
            {
                nri::ClearDesc clearDescs[2] = {};
                clearDescs[0].planes = nri::PlaneBits::COLOR;
                clearDescs[0].value.color.f = {0.0f, 0.63f, 1.0f};
                clearDescs[1].planes = nri::PlaneBits::DEPTH;
                clearDescs[1].value.depthStencil.depth = CLEAR_DEPTH;

                NRI.CmdClearAttachments(commandBuffer, clearDescs, helper::GetCountOf(clearDescs), nullptr, 0);

                const nri::Viewport viewport = {0.0f, 0.0f, (float)windowWidth, (float)windowHeight, 0.0f, 1.0f};
                NRI.CmdSetViewports(commandBuffer, &viewport, 1);

                const nri::Rect scissor = {0, 0, (nri::Dim_t)windowWidth, (nri::Dim_t)windowHeight};
                NRI.CmdSetScissors(commandBuffer, &scissor, 1);

                NRI.CmdSetIndexBuffer(commandBuffer, *m_Buffers[INDEX_BUFFER], 0, sizeof(utils::Index) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32);

                NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
                NRI.CmdSetDescriptorSet(commandBuffer, GLOBAL_DESCRIPTOR_SET, *m_DescriptorSets[queuedFrameIndex], nullptr);

                // TODO: no sorting per pipeline / material, transparency is not last
                for (const utils::Instance& instance : m_Scene.instances) {
                    const utils::Material& material = m_Scene.materials[instance.materialIndex];
                    uint32_t pipelineIndex = material.IsAlphaOpaque() ? 1 : (material.IsTransparent() ? 2 : 0);
                    NRI.CmdSetPipeline(commandBuffer, *m_Pipelines[pipelineIndex]);

                    nri::VertexBufferDesc vertexBufferDesc = {};
                    vertexBufferDesc.buffer = m_Buffers[VERTEX_BUFFER];
                    vertexBufferDesc.offset = 0;
                    vertexBufferDesc.stride = sizeof(utils::Vertex);
                    NRI.CmdSetVertexBuffers(commandBuffer, 0, &vertexBufferDesc, 1);

                    nri::DescriptorSet* descriptorSet = m_DescriptorSets[GetQueuedFrameNum() + instance.materialIndex];
                    NRI.CmdSetDescriptorSet(commandBuffer, MATERIAL_DESCRIPTOR_SET, *descriptorSet, nullptr);

                    const utils::Mesh& mesh = m_Scene.meshes[instance.meshInstanceIndex];
                    NRI.CmdDrawIndexed(commandBuffer, {mesh.indexNum, 1, mesh.indexOffset, (int32_t)mesh.vertexOffset, 0});
                }
            }
            NRI.CmdEndRendering(commandBuffer);
        }

        // End query
        if (m_QueryPool) {
            NRI.CmdEndQuery(commandBuffer, *m_QueryPool, queuedFrameIndex);
            if (frameIndex >= GetQueuedFrameNum())
                NRI.CmdCopyQueries(commandBuffer, *m_QueryPool, nextQueuedFrameIndex, 1, *m_Buffers[READBACK_BUFFER], 0);
        }

        // Reset VRS
        if (deviceDesc.tiers.shadingRate) {
            nri::ShadingRateDesc shadingRateDesc = {};
            shadingRateDesc.shadingRate = nri::ShadingRate::FRAGMENT_SIZE_1X1;
            shadingRateDesc.primitiveCombiner = nri::ShadingRateCombiner::KEEP;
            shadingRateDesc.attachmentCombiner = nri::ShadingRateCombiner::KEEP;

            NRI.CmdSetShadingRate(commandBuffer, shadingRateDesc);
        }

        { // UI
            nri::AttachmentsDesc attachmentsDesc = {};
            attachmentsDesc.colorNum = 1;
            attachmentsDesc.colors = &swapChainTexture.colorAttachment;

            CmdCopyImguiData(commandBuffer, *m_Streamer);

            NRI.CmdBeginRendering(commandBuffer, attachmentsDesc);
            {
                CmdDrawImgui(commandBuffer, swapChainTexture.attachmentFormat, 1.0f, true);
            }
            NRI.CmdEndRendering(commandBuffer);
        }

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
