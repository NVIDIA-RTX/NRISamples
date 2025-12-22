// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

constexpr nri::Format GBUFFER_FORMAT = nri::Format::RGBA8_UNORM;

struct ConstantBufferLayout {
    float L[3];
};

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
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

private:
    NRIInterface NRI = {};
    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::Texture* m_Material = nullptr;
    nri::Texture* m_Gbuffer = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Pipeline* m_GbufferFill = nullptr;
    nri::Pipeline* m_GbufferUse = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Descriptor* m_Buffer_Constant = nullptr;
    nri::Descriptor* m_Material_ShaderResource = nullptr;
    nri::Descriptor* m_Gbuffer_ColorAttachment = nullptr;
    nri::Descriptor* m_Gbuffer_InputAttachment = nullptr;
    uint32_t m_ConstantBufferOffset = 0;
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

        NRI.DestroyPipeline(m_GbufferFill);
        NRI.DestroyPipeline(m_GbufferUse);
        NRI.DestroyPipelineLayout(m_PipelineLayout);
        NRI.DestroyDescriptor(m_Buffer_Constant);
        NRI.DestroyDescriptor(m_Material_ShaderResource);
        NRI.DestroyDescriptor(m_Gbuffer_ColorAttachment);
        NRI.DestroyDescriptor(m_Gbuffer_InputAttachment);
        NRI.DestroyTexture(m_Material);
        NRI.DestroyTexture(m_Gbuffer);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
        NRI.DestroyFence(m_FrameFence);
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
    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_ENABLE_COMMAND_BUFFER_EMULATION;
    deviceCreationDesc.disableD3D12EnhancedBarriers = D3D12_DISABLE_ENHANCED_BARRIERS;
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
    if (!deviceDesc.shaderFeatures.inputAttachments) {
        printf("Input attachments are not supported!\n");
        exit(0);
    }

    // Create streamer
    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER};
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.constantBufferSize = 1024;
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
        swapChainDesc.width = (uint16_t)GetOutputResolution().x;
        swapChainDesc.height = (uint16_t)GetOutputResolution().y;
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
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    { // Pipeline layout
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::MIRRORED_REPEAT, nri::AddressMode::MIRRORED_REPEAT};
        samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
        samplerDesc.anisotropy = 4;
        samplerDesc.mipMax = 16.0f;

        nri::RootSamplerDesc rootSampler = {0, samplerDesc, nri::StageBits::FRAGMENT_SHADER};
        nri::RootDescriptorDesc rootConstantBuffer = {1, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::ALL};

        nri::DescriptorRangeDesc descriptorRanges[2] = {};
        descriptorRanges[0] = {0, 1, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER};
        descriptorRanges[1] = {1, 1, nri::DescriptorType::INPUT_ATTACHMENT, nri::StageBits::FRAGMENT_SHADER};

        nri::DescriptorSetDesc descriptorSetDesc = {0, descriptorRanges, helper::GetCountOf(descriptorRanges)};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.rootRegisterSpace = 1; // see shader
        pipelineLayoutDesc.rootSamplerNum = 1;
        pipelineLayoutDesc.rootSamplers = &rootSampler;
        pipelineLayoutDesc.rootDescriptorNum = 1;
        pipelineLayoutDesc.rootDescriptors = &rootConstantBuffer;
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
        pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    // Pipelines
    utils::ShaderCodeStorage shaderCodeStorage;
    {
        nri::InputAssemblyDesc inputAssemblyDesc = {};
        inputAssemblyDesc.topology = nri::Topology::TRIANGLE_STRIP;

        nri::RasterizationDesc rasterizationDesc = {};
        rasterizationDesc.fillMode = nri::FillMode::SOLID;
        rasterizationDesc.cullMode = nri::CullMode::NONE;

        nri::ColorAttachmentDesc colorAttachmentDescs[2] = {};
        colorAttachmentDescs[0].format = swapChainFormat;
        colorAttachmentDescs[0].colorWriteMask = nri::ColorWriteBits::NONE;
        colorAttachmentDescs[1].format = GBUFFER_FORMAT;
        colorAttachmentDescs[1].colorWriteMask = nri::ColorWriteBits::RGBA;

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = colorAttachmentDescs;
        outputMergerDesc.colorNum = helper::GetCountOf(colorAttachmentDescs);

        nri::ShaderDesc shaderStages[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "ScreenQuad.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "GbufferFill.fs", shaderCodeStorage),
        };

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaderStages;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaderStages);

        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_GbufferFill));

        colorAttachmentDescs[0].colorWriteMask = nri::ColorWriteBits::RGBA;
        colorAttachmentDescs[1].colorWriteMask = nri::ColorWriteBits::NONE;

        shaderStages[1] = utils::LoadShader(deviceDesc.graphicsAPI, "GbufferUse.fs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, graphicsPipelineDesc, m_GbufferUse));
    }

    // Load texture
    utils::Texture materialTexture;
    std::string path = utils::GetFullPath("svbbbdi4_normal.jpg", utils::DataFolder::TEXTURES);
    if (!utils::LoadTexture(path, materialTexture))
        return false;

    // Resources
    {
        { // Material
            nri::TextureDesc textureDesc = {};
            textureDesc.type = nri::TextureType::TEXTURE_2D;
            textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
            textureDesc.format = materialTexture.GetFormat();
            textureDesc.width = materialTexture.GetWidth();
            textureDesc.height = materialTexture.GetHeight();
            textureDesc.mipNum = materialTexture.GetMipNum();

            NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, textureDesc, m_Material));

            NRI.SetDebugName(m_Material, "Material");
        }

        { // Gbuffer
            nri::TextureDesc textureDesc = {};
            textureDesc.type = nri::TextureType::TEXTURE_2D;
            textureDesc.usage = nri::TextureUsageBits::COLOR_ATTACHMENT | nri::TextureUsageBits::INPUT_ATTACHMENT;
            textureDesc.format = GBUFFER_FORMAT;
            textureDesc.width = (nri::Dim_t)GetOutputResolution().x;
            textureDesc.height = (nri::Dim_t)GetOutputResolution().y;

            NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, textureDesc, m_Gbuffer));

            NRI.SetDebugName(m_Gbuffer, "Gbuffer");
        }
    }

    // Descriptors
    {
        { // Material
            nri::Texture2DViewDesc texture2DViewDesc = {m_Material, nri::Texture2DViewType::SHADER_RESOURCE, materialTexture.GetFormat()};
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_Material_ShaderResource));
        }

        { // Gbuffer
            nri::Texture2DViewDesc texture2DViewDesc = {m_Gbuffer, nri::Texture2DViewType::COLOR_ATTACHMENT, GBUFFER_FORMAT};
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_Gbuffer_ColorAttachment));

            texture2DViewDesc.viewType = nri::Texture2DViewType::INPUT_ATTACHMENT;
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_Gbuffer_InputAttachment));
        }

        { // Constant buffer
            nri::BufferViewDesc bufferViewDesc = {};
            bufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
            bufferViewDesc.buffer = NRI.GetStreamerConstantBuffer(*m_Streamer);
            bufferViewDesc.size = helper::Align((uint32_t)sizeof(ConstantBufferLayout), deviceDesc.memoryAlignment.constantBufferOffset);
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, m_Buffer_Constant));
        }
    }

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.textureMaxNum = 1;
        descriptorPoolDesc.inputAttachmentMaxNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    }

    { // Descriptor set
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0));

        nri::UpdateDescriptorRangeDesc updates[2] = {};
        updates[0] = {m_DescriptorSet, 0, 0, &m_Material_ShaderResource, 1};
        updates[1] = {m_DescriptorSet, 1, 0, &m_Gbuffer_InputAttachment, 1};

        NRI.UpdateDescriptorRanges(updates, helper::GetCountOf(updates));
    }

    { // Upload data
        std::array<nri::TextureSubresourceUploadDesc, 16> subresources;
        for (uint32_t mip = 0; mip < materialTexture.GetMipNum(); mip++)
            materialTexture.GetSubresource(subresources[mip], mip);

        nri::TextureUploadDesc textureData = {};
        textureData.subresources = subresources.data();
        textureData.texture = m_Material;
        textureData.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureData, 1, nullptr, 0));
    }

    return true;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::PrepareFrame(uint32_t frameIndex) {
    float t = frameIndex * 0.001f;
    float x = 5.0f * cosf(t);
    float y = 5.0f * sinf(t);

    // Update constants
    ConstantBufferLayout constants = {};
    constants.L[0] = x;
    constants.L[1] = y;
    constants.L[2] = 1.0f;

    m_ConstantBufferOffset = NRI.StreamConstantData(*m_Streamer, &constants, sizeof(constants));
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::Dim_t w = (nri::Dim_t)GetOutputResolution().x;
    nri::Dim_t h = (nri::Dim_t)GetOutputResolution().y;

    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Record
    nri::CommandBuffer* commandBuffer = queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(*commandBuffer, m_DescriptorPool);
    {
        { // Barriers
            nri::TextureBarrierDesc textureBarriers[2] = {};

            textureBarriers[0].texture = swapChainTexture.texture;
            textureBarriers[0].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};

            textureBarriers[1].texture = m_Gbuffer;
            //if (frameIndex)
            //    textureBarriers[1].before = {nri::AccessBits::INPUT_ATTACHMENT, nri::Layout::INPUT_ATTACHMENT};
            textureBarriers[1].after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::INPUT_ATTACHMENT};

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);
            barrierDesc.textures = textureBarriers;

            NRI.CmdBarrier(*commandBuffer, barrierDesc);
        }

        // Render passes
        nri::AttachmentDesc colorAttachmentDescs[2] = {};

        colorAttachmentDescs[0].descriptor = swapChainTexture.colorAttachment;
        colorAttachmentDescs[0].loadOp = nri::LoadOp::CLEAR;

        colorAttachmentDescs[1].descriptor = m_Gbuffer_ColorAttachment;
        colorAttachmentDescs[1].loadOp = nri::LoadOp::CLEAR;

        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colorNum = helper::GetCountOf(colorAttachmentDescs);
        renderingDesc.colors = colorAttachmentDescs;

        NRI.CmdBeginRendering(*commandBuffer, renderingDesc);
        {
            const nri::Viewport viewport = {0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
            NRI.CmdSetViewports(*commandBuffer, &viewport, 1);

            const nri::Rect scissor = {0, 0, w, h};
            NRI.CmdSetScissors(*commandBuffer, &scissor, 1);

            NRI.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *m_PipelineLayout);

            nri::SetRootDescriptorDesc rootDescriptorDesc = {0, m_Buffer_Constant, m_ConstantBufferOffset};
            NRI.CmdSetRootDescriptor(*commandBuffer, rootDescriptorDesc);

            nri::SetDescriptorSetDesc descriptorSetDesc = {0, m_DescriptorSet};
            NRI.CmdSetDescriptorSet(*commandBuffer, descriptorSetDesc);

            { // Gbuffer fill
                helper::Annotation annotation(NRI, *commandBuffer, "Gbuffer fill");

                NRI.CmdSetPipeline(*commandBuffer, *m_GbufferFill);
                NRI.CmdDraw(*commandBuffer, {4, 1, 0, 0});
            }

            { // Barrier
                nri::TextureBarrierDesc textureBarrier = {};
                textureBarrier.texture = m_Gbuffer;
                textureBarrier.before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::INPUT_ATTACHMENT, nri::StageBits::COLOR_ATTACHMENT};
                textureBarrier.after = {nri::AccessBits::INPUT_ATTACHMENT, nri::Layout::INPUT_ATTACHMENT, nri::StageBits::FRAGMENT_SHADER};

                nri::BarrierDesc barrierDesc = {};
                barrierDesc.textureNum = 1;
                barrierDesc.textures = &textureBarrier;

                NRI.CmdBarrier(*commandBuffer, barrierDesc);
            }

            { // Gbuffer use
                helper::Annotation annotation(NRI, *commandBuffer, "Gbuffer use");

                NRI.CmdSetPipeline(*commandBuffer, *m_GbufferUse);
                NRI.CmdDraw(*commandBuffer, {4, 1, 0, 0});
            }
        }
        NRI.CmdEndRendering(*commandBuffer);

        { // Barrier
            nri::TextureBarrierDesc textureBarrier = {};

            textureBarrier.texture = swapChainTexture.texture;
            textureBarrier.before = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
            textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::PRESENT, nri::StageBits::NONE};

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textureNum = 1;
            barrierDesc.textures = &textureBarrier;

            NRI.CmdBarrier(*commandBuffer, barrierDesc);
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
