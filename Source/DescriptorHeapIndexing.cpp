// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

constexpr uint32_t RESOURCE_NUM = 16; // more than needed

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
    void RenderFrame(uint32_t frameIndex) override;

private:
    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;
    nri::Buffer* m_Buffer = nullptr;
    nri::Texture* m_Tex0 = nullptr;
    nri::Texture* m_Tex1 = nullptr;
    nri::Texture* m_Output = nullptr;
    nri::Descriptor* m_Buffer_Constant = nullptr;
    nri::Descriptor* m_Tex0_Texture = nullptr;
    nri::Descriptor* m_Tex1_Texture = nullptr;
    nri::Descriptor* m_Linear_Sampler = nullptr;
    nri::Descriptor* m_Nearest_Sampler = nullptr;
    nri::Descriptor* m_Output_StorageTexture = nullptr;
    nri::Pipeline* m_ComputePipeline = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
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

        NRI.DestroyFence(m_FrameFence);

        NRI.DestroyDescriptor(m_Buffer_Constant);
        NRI.DestroyDescriptor(m_Tex0_Texture);
        NRI.DestroyDescriptor(m_Tex1_Texture);
        NRI.DestroyDescriptor(m_Linear_Sampler);
        NRI.DestroyDescriptor(m_Nearest_Sampler);
        NRI.DestroyDescriptor(m_Output_StorageTexture);

        NRI.DestroyBuffer(m_Buffer);
        NRI.DestroyTexture(m_Tex0);
        NRI.DestroyTexture(m_Tex1);
        NRI.DestroyTexture(m_Output);

        NRI.DestroyPipeline(m_ComputePipeline);
        NRI.DestroyPipelineLayout(m_PipelineLayout);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    nri::nriDestroyDevice(m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    if (graphicsAPI == nri::GraphicsAPI::D3D11) {
        printf("This sample is not supported by D3D11\n");
        exit(0);
    }

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
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));

    // Command queue
    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));

    { // Swap chain
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

    // Queued frames
    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    { // Output
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.format = m_SwapChainTextures[0].attachmentFormat;
        textureDesc.width = (uint16_t)GetOutputResolution().x;
        textureDesc.height = (uint16_t)GetOutputResolution().y;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, textureDesc, m_Output));

        nri::Texture2DViewDesc textureViewDesc = {m_Output, nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D, textureDesc.format};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, m_Output_StorageTexture));
    }

    { // Constant buffer
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = 256;
        bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedBuffer(*m_Device, NriDeviceUploadHeap, bufferDesc, m_Buffer));

        nri::BufferViewDesc bufferViewDesc = {m_Buffer, nri::BufferViewType::CONSTANT};
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferViewDesc, m_Buffer_Constant));
    }

    { // Texture 0
        utils::Texture textureData;
        std::string path = utils::GetFullPath("svbbbdi4_2.jpg", utils::DataFolder::TEXTURES);
        if (!utils::LoadTexture(path, textureData))
            return false;

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        textureDesc.format = textureData.GetFormat();
        textureDesc.width = textureData.GetWidth();
        textureDesc.height = textureData.GetHeight();

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, textureDesc, m_Tex0));

        nri::Texture2DViewDesc textureViewDesc = {m_Tex0, nri::Texture2DViewType::SHADER_RESOURCE_2D, textureDesc.format};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, m_Tex0_Texture));

        nri::TextureSubresourceUploadDesc subresource0 = {};
        textureData.GetSubresource(subresource0, 0);

        nri::TextureUploadDesc textureUploadDesc = {};
        textureUploadDesc.subresources = &subresource0;
        textureUploadDesc.texture = m_Tex0;
        textureUploadDesc.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureUploadDesc, 1, nullptr, 0));
    }

    { // Texture 1
        utils::Texture textureData;
        std::string path = utils::GetFullPath("svbbbdi4_normal.jpg", utils::DataFolder::TEXTURES);
        if (!utils::LoadTexture(path, textureData))
            return false;

        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        textureDesc.format = textureData.GetFormat();
        textureDesc.width = textureData.GetWidth();
        textureDesc.height = textureData.GetHeight();

        NRI_ABORT_ON_FAILURE(NRI.CreatePlacedTexture(*m_Device, NriDeviceHeap, textureDesc, m_Tex1));

        nri::Texture2DViewDesc textureViewDesc = {m_Tex1, nri::Texture2DViewType::SHADER_RESOURCE_2D, textureDesc.format};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, m_Tex1_Texture));

        nri::TextureSubresourceUploadDesc subresource0 = {};
        textureData.GetSubresource(subresource0, 0);

        nri::TextureUploadDesc textureUploadDesc = {};
        textureUploadDesc.subresources = &subresource0;
        textureUploadDesc.texture = m_Tex1;
        textureUploadDesc.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

        NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, &textureUploadDesc, 1, nullptr, 0));
    }

    { // Samplers
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, {nri::Filter::LINEAR, nri::Filter::LINEAR}, m_Linear_Sampler));
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, {nri::Filter::NEAREST, nri::Filter::NEAREST}, m_Nearest_Sampler));
    }

    { // Pipeline layout
        nri::DescriptorRangeDesc heaps[2] = {
            { // Resource heap
                0, // VK binding for "-fvk-bind-resource-heap"
                RESOURCE_NUM,
                nri::DescriptorType::MUTABLE,
                nri::StageBits::COMPUTE_SHADER,
                nri::DescriptorRangeBits::ARRAY | nri::DescriptorRangeBits::PARTIALLY_BOUND,
            },
            { // Sampler heap
                1, // VK binding for "-fvk-bind-sampler-heap"
                2,
                nri::DescriptorType::SAMPLER,
                nri::StageBits::COMPUTE_SHADER,
                nri::DescriptorRangeBits::ARRAY | nri::DescriptorRangeBits::PARTIALLY_BOUND,
            },
        };

        nri::DescriptorSetDesc descriptorSetDesc = {0, heaps, helper::GetCountOf(heaps)};

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        pipelineLayoutDesc.flags = nri::PipelineLayoutBits::RESOURCE_HEAP_DIRECTLY_INDEXED | nri::PipelineLayoutBits::SAMPLER_HEAP_DIRECTLY_INDEXED;

        NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));
    }

    { // Compute pipeline
        utils::ShaderCodeStorage shaderCodeStorage;

        nri::ComputePipelineDesc computePipelineDesc = {};
        computePipelineDesc.pipelineLayout = m_PipelineLayout;
        computePipelineDesc.shader = utils::LoadShader(graphicsAPI, "DescriptorHeapIndexing.cs", shaderCodeStorage);

        NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_ComputePipeline));
    }

    { // Descriptor pool (ala resource heap) and a descriptor set, working as "an interface" for updating descriptors in the heap
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.mutableMaxNum = RESOURCE_NUM;
        descriptorPoolDesc.samplerMaxNum = 2;
        descriptorPoolDesc.descriptorSetMaxNum = 1;

        NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));

        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0));

        // The descriptor set is the 1st allocated from the pool, so "GetDescriptorSetOffsets" returns {0}
        uint32_t resourceHeapOffset = uint32_t(-1);
        uint32_t samplerHeapOffset = uint32_t(-1);
        NRI.GetDescriptorSetOffsets(*m_DescriptorSet, resourceHeapOffset, samplerHeapOffset);

        if (resourceHeapOffset != 0 || samplerHeapOffset != 0) {
            printf("ERROR: heap offsets are expected to be 0!\n");
            exit(1);
        }
    }

    { // Update descriptors in the resource heap
        const nri::Descriptor* textures[] = {
            m_Tex0_Texture,
            m_Tex1_Texture,
        };

        const nri::Descriptor* samplers[] = {
            m_Nearest_Sampler,
            m_Linear_Sampler,
        };

        const nri::UpdateDescriptorRangeDesc updateDescriptorRangeDesc[] = {
            // 0 range is "resource heap"
            {m_DescriptorSet, 0, 0, &m_Output_StorageTexture, 1, nri::DescriptorType::STORAGE_TEXTURE},
            {m_DescriptorSet, 0, 1, &m_Buffer_Constant, 1, nri::DescriptorType::CONSTANT_BUFFER},
            {m_DescriptorSet, 0, 2, textures, helper::GetCountOf(textures), nri::DescriptorType::TEXTURE},

            // 1 range is "sampler heap"
            {m_DescriptorSet, 1, 0, samplers, helper::GetCountOf(samplers)},
        };

        NRI.UpdateDescriptorRanges(updateDescriptorRangeDesc, helper::GetCountOf(updateDescriptorRangeDesc));
    }

    return true;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    // Acquire a swap chain texture
    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    // Update buffer
    float* data = (float*)NRI.MapBuffer(*m_Buffer, 0, nri::WHOLE_SIZE);
    {
        data[0] = 0.0f;
        data[1] = 0.0f;
        data[2] = 0.0f;
        data[3] = 0.0f;

        data[4] = (float)sin((double)frameIndex * 0.0002) * 0.5f + 0.5f;
        data[5] = 1.0f;
        data[6] = 0.0f;
        data[7] = 1.0f;
    }
    NRI.UnmapBuffer(*m_Buffer);

    // Record
    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        nri::TextureBarrierDesc textureTransitions[2] = {};
        nri::BarrierDesc barrierDesc = {};

        // Barriers
        textureTransitions[0].texture = swapChainTexture.texture;
        textureTransitions[0].after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION};
        textureTransitions[0].layerNum = 1;
        textureTransitions[0].mipNum = 1;

        textureTransitions[1].texture = m_Output;
        textureTransitions[1].before = {frameIndex == 0 ? nri::AccessBits::NONE : nri::AccessBits::COPY_SOURCE, frameIndex == 0 ? nri::Layout::UNDEFINED : nri::Layout::COPY_SOURCE};
        textureTransitions[1].after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE};
        textureTransitions[1].layerNum = 1;
        textureTransitions[1].mipNum = 1;

        barrierDesc.textures = textureTransitions;
        barrierDesc.textureNum = 2;

        NRI.CmdBarrier(commandBuffer, barrierDesc);

        // Rendering
        NRI.CmdSetPipelineLayout(commandBuffer, nri::BindPoint::COMPUTE, *m_PipelineLayout);
        NRI.CmdSetPipeline(commandBuffer, *m_ComputePipeline);

        nri::SetDescriptorSetDesc descriptorSet0 = {0, m_DescriptorSet};
        NRI.CmdSetDescriptorSet(commandBuffer, descriptorSet0);

        uint32_t nx = (GetOutputResolution().x + 15) / 16;
        uint32_t ny = (GetOutputResolution().y + 15) / 16;

        NRI.CmdDispatch(commandBuffer, {nx, ny, 1});

        // Barriers
        textureTransitions[1].before = textureTransitions[1].after;
        textureTransitions[1].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE};

        barrierDesc.textures = textureTransitions + 1;
        barrierDesc.textureNum = 1;

        NRI.CmdBarrier(commandBuffer, barrierDesc);

        // Copy
        NRI.CmdCopyTexture(commandBuffer, *swapChainTexture.texture, nullptr, *m_Output, nullptr);

        // Barriers
        textureTransitions[0].before = textureTransitions[0].after;
        textureTransitions[0].after = {nri::AccessBits::NONE, nri::Layout::PRESENT, nri::StageBits::NONE};

        barrierDesc.textures = textureTransitions;
        barrierDesc.textureNum = 1;

        NRI.CmdBarrier(commandBuffer, barrierDesc);
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
