// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

constexpr uint32_t VIEW_MASK = 0b11;
constexpr nri::Color32f COLOR_0 = {1.0f, 1.0f, 0.0f, 1.0f};
constexpr nri::Color32f COLOR_1 = {0.46f, 0.72f, 0.0f, 1.0f};

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

struct NRIInterface
    : public nri::CoreInterface,
      public nri::HelperInterface,
      public nri::StreamerInterface,
      public nri::SwapChainInterface {};

struct Frame {
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

    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
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
    nri::Pipeline* m_PipelineMultiview = nullptr;
    nri::DescriptorSet* m_TextureDescriptorSet = nullptr;
    nri::Descriptor* m_TextureShaderResource = nullptr;
    nri::Descriptor* m_Sampler = nullptr;
    nri::Buffer* m_ConstantBuffer = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Texture* m_Texture = nullptr;

    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<nri::Memory*> m_MemoryAllocations;

    uint64_t m_GeometryOffset = 0;
    bool m_Multiview = false;
    float m_Transparency = 1.0f;
    float m_Scale = 1.0f;
};

Sample::~Sample() {
    NRI.WaitForIdle(*m_GraphicsQueue);

    for (Frame& frame : m_Frames) {
        NRI.DestroyCommandBuffer(*frame.commandBuffer);
        NRI.DestroyCommandAllocator(*frame.commandAllocator);
        NRI.DestroyDescriptor(*frame.constantBufferView);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);

    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipeline(*m_PipelineMultiview);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyDescriptor(*m_TextureShaderResource);
    NRI.DestroyDescriptor(*m_Sampler);
    NRI.DestroyBuffer(*m_ConstantBuffer);
    NRI.DestroyBuffer(*m_GeometryBuffer);
    NRI.DestroyTexture(*m_Texture);
    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyFence(*m_FrameFence);
    NRI.DestroySwapChain(*m_SwapChain);
    NRI.DestroyStreamer(*m_Streamer);

    for (nri::Memory* memory : m_MemoryAllocations)
        NRI.FreeMemory(*memory);

    DestroyUI(NRI);

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
    nri::AdapterDesc bestAdapterDesc = {};
    uint32_t adapterDescsNum = 1;
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(&bestAdapterDesc, adapterDescsNum));

    // Device
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_COMMANDBUFFER_EMULATION;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &bestAdapterDesc;
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
    streamerDesc.frameInFlightNum = BUFFERED_FRAME_MAX_NUM;
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
        swapChainDesc.verticalSyncInterval = m_VsyncInterval;
        swapChainDesc.width = (uint16_t)GetWindowResolution().x;
        swapChainDesc.height = (uint16_t)GetWindowResolution().y;
        swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;
        NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

        uint32_t swapChainTextureNum;
        nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
        swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

        for (uint32_t i = 0; i < swapChainTextureNum; i++) {
            nri::Texture2DViewDesc textureViewDesc = {swapChainTextures[i], nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};

            nri::Descriptor* colorAttachment;
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, colorAttachment));

            const BackBuffer backBuffer = {colorAttachment, swapChainTextures[i]};
            m_SwapChainBuffers.push_back(backBuffer);
        }
    }

    // Buffered resources
    for (Frame& frame : m_Frames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, frame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer));
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
        colorAttachmentDesc.colorBlend = {nri::BlendFactor::SRC_ALPHA, nri::BlendFactor::ONE_MINUS_SRC_ALPHA, nri::BlendFunc::ADD};

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = &colorAttachmentDesc;
        outputMergerDesc.colorNum = 1;

        nri::ShaderDesc shaderStages[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "TriangleFlexibleMultiview.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "Triangle.fs", shaderCodeStorage),
        };

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
        graphicsPipelineDesc.vertexInput = &vertexInputDesc;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);

        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_Pipeline));

        // Multiview
        if (deviceDesc.features.flexibleMultiview) {
            graphicsPipelineDesc.outputMerger.viewMask = VIEW_MASK;
            graphicsPipelineDesc.outputMerger.multiview = nri::Multiview::FLEXIBLE;

            NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_PipelineMultiview));
        }
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = BUFFERED_FRAME_MAX_NUM + 1;
        descriptorPoolDesc.constantBufferMaxNum = BUFFERED_FRAME_MAX_NUM;
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
    const uint64_t indexDataSize = sizeof(g_IndexData);
    const uint64_t indexDataAlignedSize = helper::Align(indexDataSize, 16);
    const uint64_t vertexDataSize = sizeof(g_VertexData);
    {
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
            bufferDesc.size = constantBufferSize * BUFFERED_FRAME_MAX_NUM;
            bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));
        }

        { // Geometry buffer
            nri::BufferDesc bufferDesc = {};
            bufferDesc.size = indexDataAlignedSize + vertexDataSize;
            bufferDesc.usage = nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_GeometryBuffer));
        }
        m_GeometryOffset = indexDataAlignedSize;
    }

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_ConstantBuffer;

    m_MemoryAllocations.resize(1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data()));

    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_GeometryBuffer;
    resourceGroupDesc.textureNum = 1;
    resourceGroupDesc.textures = &m_Texture;

    m_MemoryAllocations.resize(1 + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + 1));

    {     // Descriptors
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
        for (uint32_t i = 0; i < BUFFERED_FRAME_MAX_NUM; i++) {
            nri::BufferViewDesc bufferViewDesc = {};
            bufferViewDesc.buffer = m_ConstantBuffer;
            bufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
            bufferViewDesc.offset = i * constantBufferSize;
            bufferViewDesc.size = constantBufferSize;
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, m_Frames[i].constantBufferView));

            m_Frames[i].constantBufferViewOffset = bufferViewDesc.offset;
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
        for (Frame& frame : m_Frames) {
            NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &frame.constantBufferDescriptorSet, 1, 0));

            nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&frame.constantBufferView, 1};
            NRI.UpdateDescriptorRanges(*frame.constantBufferDescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
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
        bufferData.data = &geometryBufferData[0];
        bufferData.dataSize = geometryBufferData.size();
        bufferData.after = {nri::AccessBits::INDEX_BUFFER | nri::AccessBits::VERTEX_BUFFER};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureData, 1, &bufferData, 1));
    }

    // User interface
    bool initialized = InitUI(NRI, NRI, *m_Device, swapChainFormat);

    return initialized;
}

void Sample::PrepareFrame(uint32_t) {
    BeginUI();

    ImGui::SetNextWindowPos(ImVec2(30, 30), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(0, 0));
    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoResize);
    {
        ImGui::SliderFloat("Transparency", &m_Transparency, 0.0f, 1.0f);
        ImGui::SliderFloat("Scale", &m_Scale, 0.75f, 1.25f);

        const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
        ImGui::BeginDisabled(!deviceDesc.features.flexibleMultiview);
        ImGui::Checkbox("Multiview", &m_Multiview);
        ImGui::EndDisabled();
    }
    ImGui::End();

    EndUI(NRI, *m_Streamer);
    NRI.CopyStreamerUpdateRequests(*m_Streamer);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::Dim_t w = (nri::Dim_t)GetWindowResolution().x;
    nri::Dim_t h = (nri::Dim_t)GetWindowResolution().y;
    nri::Dim_t w2 = w / 2;
    nri::Dim_t h2 = h / 2;
    nri::Dim_t w4 = w / 4;

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM) {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
        NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    const uint32_t currentTextureIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
    BackBuffer& currentBackBuffer = m_SwapChainBuffers[currentTextureIndex];

    ConstantBufferLayout* commonConstants = (ConstantBufferLayout*)NRI.MapBuffer(*m_ConstantBuffer, frame.constantBufferViewOffset, sizeof(ConstantBufferLayout));
    if (commonConstants) {
        commonConstants->color[0] = 0.8f;
        commonConstants->color[1] = 0.5f;
        commonConstants->color[2] = 0.1f;
        commonConstants->scale = m_Scale;

        NRI.UnmapBuffer(*m_ConstantBuffer);
    }

    // Record
    nri::CommandBuffer* commandBuffer = frame.commandBuffer;
    NRI.BeginCommandBuffer(*commandBuffer, m_DescriptorPool);
    {
        nri::TextureBarrierDesc textureBarrierDescs = {};
        textureBarrierDescs.texture = currentBackBuffer.texture;
        textureBarrierDescs.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};

        nri::BarrierGroupDesc barrierGroupDesc = {};
        barrierGroupDesc.textureNum = 1;
        barrierGroupDesc.textures = &textureBarrierDescs;

        NRI.CmdBarrier(*commandBuffer, barrierGroupDesc);

        // Single- or multi- view
        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &currentBackBuffer.colorAttachment;
        attachmentsDesc.viewMask = m_Multiview ? VIEW_MASK : 0;

        NRI.CmdBeginRendering(*commandBuffer, attachmentsDesc);
        {
            {
                helper::Annotation annotation(NRI, *commandBuffer, "Clears");

                nri::ClearDesc clearDesc = {};
                clearDesc.planes = nri::PlaneBits::COLOR;
                clearDesc.value.color.f = COLOR_0;

                NRI.CmdClearAttachments(*commandBuffer, &clearDesc, 1, nullptr, 0);

                clearDesc.value.color.f = COLOR_1;

                nri::Rect rects[2];
                rects[0] = {0, 0, w2, h2};
                rects[1] = {(int16_t)w2, (int16_t)h2, w2, h2};

                NRI.CmdClearAttachments(*commandBuffer, &clearDesc, 1, rects, helper::GetCountOf(rects));
            }

            {
                helper::Annotation annotation(NRI, *commandBuffer, "Triangle");

                NRI.CmdSetPipelineLayout(*commandBuffer, *m_PipelineLayout);
                NRI.CmdSetPipeline(*commandBuffer, m_Multiview ? *m_PipelineMultiview : *m_Pipeline);
                NRI.CmdSetRootConstants(*commandBuffer, 0, &m_Transparency, 4);
                NRI.CmdSetIndexBuffer(*commandBuffer, *m_GeometryBuffer, 0, nri::IndexType::UINT16);

                nri::VertexBufferDesc vertexBufferDesc = {};
                vertexBufferDesc.buffer = m_GeometryBuffer;
                vertexBufferDesc.offset = m_GeometryOffset;
                vertexBufferDesc.stride = sizeof(Vertex);
                NRI.CmdSetVertexBuffers(*commandBuffer, 0, &vertexBufferDesc, 1);

                NRI.CmdSetDescriptorSet(*commandBuffer, 0, *frame.constantBufferDescriptorSet, nullptr);
                NRI.CmdSetDescriptorSet(*commandBuffer, 1, *m_TextureDescriptorSet, nullptr);

                if (m_Multiview) {
                    const nri::Viewport viewports[2] = {
                        {0.0f, 0.0f, (float)w2, (float)h, 0.0f, 1.0f},
                        {(float)w2, 0.0f, (float)w2, (float)h, 0.0f, 1.0f},
                    };
                    NRI.CmdSetViewports(*commandBuffer, viewports, helper::GetCountOf(viewports));

                    nri::Rect scissors[2] = {
                        {(int16_t)0, 0, w4, h},
                        {(int16_t)w2, 0, w4, h},
                    };
                    NRI.CmdSetScissors(*commandBuffer, scissors, helper::GetCountOf(scissors));
                } else {
                    const nri::Viewport viewport = {0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
                    NRI.CmdSetViewports(*commandBuffer, &viewport, 1);

                    nri::Rect scissor = {0, 0, w2, h};
                    NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                }

                NRI.CmdDrawIndexed(*commandBuffer, {3, 1, 0, 0, 0});

                if (m_Multiview) {
                    nri::Rect scissor[2] = {
                        {(int16_t)0 + w4, (int16_t)h2, w4, h2},
                        {(int16_t)w2 + w4, (int16_t)h2, w4, h2},
                    };
                    NRI.CmdSetScissors(*commandBuffer, scissor, helper::GetCountOf(scissor));
                } else {
                    nri::Rect scissor = {(int16_t)w2, (int16_t)h2, w2, h2};
                    NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                }

                NRI.CmdDraw(*commandBuffer, {3, 1, 0, 0});
            }
        }
        NRI.CmdEndRendering(*commandBuffer);

        // Singleview
        attachmentsDesc.viewMask = 0;

        NRI.CmdBeginRendering(*commandBuffer, attachmentsDesc);
        {
            helper::Annotation annotation(NRI, *commandBuffer, "UI");

            RenderUI(NRI, NRI, *m_Streamer, *commandBuffer, 1.0f, true);
        }
        NRI.CmdEndRendering(*commandBuffer);

        textureBarrierDescs.before = textureBarrierDescs.after;
        textureBarrierDescs.after = {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT};

        NRI.CmdBarrier(*commandBuffer, barrierGroupDesc);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    { // Submit
        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &frame.commandBuffer;
        queueSubmitDesc.commandBufferNum = 1;

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
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

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }
}

SAMPLE_MAIN(Sample, 0);
