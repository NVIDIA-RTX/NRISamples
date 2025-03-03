// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

constexpr auto BUILD_FLAGS = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr uint32_t BOX_NUM = 100000;
constexpr float BOX_HALF_SIZE = 0.5f;

static const float positions[12 * 6] = {
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    -BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
    BOX_HALF_SIZE,
};

static const float texCoords[12 * 4] = {
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
};

static const uint16_t indices[12 * 3] = {
    0, 1, 2,
    1, 2, 3,
    4, 5, 6,
    5, 6, 7,
    8, 9, 10,
    9, 10, 11,
    12, 13, 14,
    13, 14, 15,
    16, 17, 18,
    17, 18, 19,
    20, 21, 22,
    21, 22, 23};

struct NRIInterface
    : public nri::CoreInterface,
      public nri::SwapChainInterface,
      public nri::HelperInterface,
      public nri::RayTracingInterface {};

struct Frame {
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
};

class Sample : public SampleBase {
public:
    Sample() {
    }

    ~Sample();

private:
    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    void CreateSwapChain(nri::Format& swapChainFormat);
    void CreateCommandBuffers();
    void CreateRayTracingPipeline();
    void CreateRayTracingOutput(nri::Format swapChainFormat);
    void CreateDescriptorSets();
    void CreateBottomLevelAccelerationStructure();
    void CreateTopLevelAccelerationStructure();
    void CreateShaderTable();
    void CreateUploadBuffer(uint64_t size, nri::BufferUsageBits usage, nri::Buffer*& buffer, nri::Memory*& memory);
    void CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory);
    void BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum);
    void BuildTopLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, uint32_t instanceNum, nri::Buffer& instanceBuffer);
    void CreateShaderResources();

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};

    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;

    nri::Buffer* m_ShaderTable = nullptr;
    uint64_t m_ShaderGroupIdentifierSize = 0;
    uint64_t m_MissShaderOffset = 0;
    uint64_t m_HitShaderGroupOffset = 0;

    nri::Texture* m_RayTracingOutput = nullptr;
    nri::Descriptor* m_RayTracingOutputView = nullptr;

    nri::Buffer* m_TexCoordBuffer = nullptr;
    nri::Buffer* m_IndexBuffer = nullptr;
    nri::Descriptor* m_TexCoordBufferView = nullptr;
    nri::Descriptor* m_IndexBufferView = nullptr;

    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSets[3] = {};

    nri::AccelerationStructure* m_BLAS = nullptr;
    nri::AccelerationStructure* m_TLAS = nullptr;
    nri::Descriptor* m_TLASDescriptor = nullptr;

    const BackBuffer* m_BackBuffer = nullptr;
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<nri::Memory*> m_MemoryAllocations;
};

Sample::~Sample() {
    NRI.WaitForIdle(*m_GraphicsQueue);

    for (uint32_t i = 0; i < m_Frames.size(); i++) {
        NRI.DestroyCommandBuffer(*m_Frames[i].commandBuffer);
        NRI.DestroyCommandAllocator(*m_Frames[i].commandAllocator);
    }

    for (uint32_t i = 0; i < m_SwapChainBuffers.size(); i++)
        NRI.DestroyDescriptor(*m_SwapChainBuffers[i].colorAttachment);

    NRI.DestroyDescriptor(*m_RayTracingOutputView);
    NRI.DestroyTexture(*m_RayTracingOutput);

    NRI.DestroyDescriptorPool(*m_DescriptorPool);

    NRI.DestroyAccelerationStructure(*m_BLAS);
    NRI.DestroyAccelerationStructure(*m_TLAS);
    NRI.DestroyDescriptor(*m_TLASDescriptor);
    NRI.DestroyBuffer(*m_ShaderTable);

    NRI.DestroyDescriptor(*m_TexCoordBufferView);
    NRI.DestroyDescriptor(*m_IndexBufferView);
    NRI.DestroyBuffer(*m_TexCoordBuffer);
    NRI.DestroyBuffer(*m_IndexBuffer);

    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);

    NRI.DestroyFence(*m_FrameFence);

    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

    DestroyUI(NRI);

    nri::nriDestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
    nri::AdapterDesc bestAdapterDesc = {};
    uint32_t adapterDescsNum = 1;
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(&bestAdapterDesc, adapterDescsNum));

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &bestAdapterDesc;
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    CreateCommandBuffers();

    nri::Format swapChainFormat = nri::Format::UNKNOWN;
    CreateSwapChain(swapChainFormat);

    CreateRayTracingPipeline();
    CreateDescriptorSets();
    CreateRayTracingOutput(swapChainFormat);
    CreateBottomLevelAccelerationStructure();
    CreateTopLevelAccelerationStructure();
    CreateShaderTable();
    CreateShaderResources();

    return true;
}

void Sample::PrepareFrame(uint32_t) {
}

void Sample::RenderFrame(uint32_t frameIndex) {
    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM) {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
        NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    const uint32_t backBufferIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
    m_BackBuffer = &m_SwapChainBuffers[backBufferIndex];

    nri::TextureBarrierDesc textureTransitions[2] = {};
    nri::BarrierGroupDesc barrierGroupDesc = {};

    // Record
    nri::CommandBuffer& commandBuffer = *frame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, m_DescriptorPool);
    {
        // Rendering
        textureTransitions[0].texture = m_BackBuffer->texture;
        textureTransitions[0].after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION};
        textureTransitions[0].layerNum = 1;
        textureTransitions[0].mipNum = 1;

        textureTransitions[1].texture = m_RayTracingOutput;
        textureTransitions[1].before = {frameIndex == 0 ? nri::AccessBits::UNKNOWN : nri::AccessBits::COPY_SOURCE, frameIndex == 0 ? nri::Layout::UNKNOWN : nri::Layout::COPY_SOURCE};
        textureTransitions[1].after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE};
        textureTransitions[1].layerNum = 1;
        textureTransitions[1].mipNum = 1;

        barrierGroupDesc.textures = textureTransitions;
        barrierGroupDesc.textureNum = 2;

        nri::BufferBarrierDesc bufferBarrier = {};
        if (frameIndex == 0) {
            bufferBarrier.buffer = m_ShaderTable;
            bufferBarrier.after = {nri::AccessBits::SHADER_RESOURCE, nri::StageBits::RAYGEN_SHADER};

            barrierGroupDesc.bufferNum = 1;
            barrierGroupDesc.buffers = &bufferBarrier;
        }

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
        NRI.CmdSetPipeline(commandBuffer, *m_Pipeline);

        for (uint32_t i = 0; i < helper::GetCountOf(m_DescriptorSets); i++)
            NRI.CmdSetDescriptorSet(commandBuffer, i, *m_DescriptorSets[i], nullptr);

        nri::DispatchRaysDesc dispatchRaysDesc = {};
        dispatchRaysDesc.raygenShader = {m_ShaderTable, 0, m_ShaderGroupIdentifierSize, m_ShaderGroupIdentifierSize};
        dispatchRaysDesc.missShaders = {m_ShaderTable, m_MissShaderOffset, m_ShaderGroupIdentifierSize, m_ShaderGroupIdentifierSize};
        dispatchRaysDesc.hitShaderGroups = {m_ShaderTable, m_HitShaderGroupOffset, m_ShaderGroupIdentifierSize, m_ShaderGroupIdentifierSize};
        dispatchRaysDesc.x = (uint16_t)GetWindowResolution().x;
        dispatchRaysDesc.y = (uint16_t)GetWindowResolution().y;
        dispatchRaysDesc.z = 1;
        NRI.CmdDispatchRays(commandBuffer, dispatchRaysDesc);

        // Copy
        textureTransitions[1].before = textureTransitions[1].after;
        textureTransitions[1].after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE};

        barrierGroupDesc.textures = textureTransitions + 1;
        barrierGroupDesc.textureNum = 1;

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
        NRI.CmdCopyTexture(commandBuffer, *m_BackBuffer->texture, nullptr, *m_RayTracingOutput, nullptr);

        // Present
        textureTransitions[0].before = textureTransitions[0].after;
        textureTransitions[0].after = {nri::AccessBits::UNKNOWN, nri::Layout::PRESENT};

        barrierGroupDesc.textures = textureTransitions;
        barrierGroupDesc.textureNum = 1;

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
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

        NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    }

    // Present
    NRI.QueuePresent(*m_SwapChain);
}

void Sample::CreateSwapChain(nri::Format& swapChainFormat) {
    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.window = GetWindow();
    swapChainDesc.queue = m_GraphicsQueue;
    swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.verticalSyncInterval = m_VsyncInterval;
    swapChainDesc.width = (uint16_t)GetWindowResolution().x;
    swapChainDesc.height = (uint16_t)GetWindowResolution().y;
    swapChainDesc.textureNum = SWAP_CHAIN_TEXTURE_NUM;

    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

    uint32_t swapChainTextureNum = 0;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
    swapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

    for (uint32_t i = 0; i < swapChainTextureNum; i++) {
        m_SwapChainBuffers.emplace_back();
        BackBuffer& backBuffer = m_SwapChainBuffers.back();

        backBuffer = {};
        backBuffer.texture = swapChainTextures[i];

        nri::Texture2DViewDesc textureViewDesc = {backBuffer.texture, nri::Texture2DViewType::COLOR_ATTACHMENT, swapChainFormat};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, backBuffer.colorAttachment));
    }
}

void Sample::CreateCommandBuffers() {
    for (Frame& frame : m_Frames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, frame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*frame.commandAllocator, frame.commandBuffer));
    }
}

void Sample::CreateRayTracingPipeline() {
    nri::DescriptorRangeDesc descriptorRanges[] = {
        {0, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::RAYGEN_SHADER},
        {1, 1, nri::DescriptorType::ACCELERATION_STRUCTURE, nri::StageBits::RAYGEN_SHADER},
        {0, BOX_NUM, nri::DescriptorType::BUFFER, nri::StageBits::CLOSEST_HIT_SHADER, nri::DescriptorRangeBits::VARIABLE_SIZED_ARRAY | nri::DescriptorRangeBits::PARTIALLY_BOUND},
    };

    nri::DescriptorSetDesc descriptorSetDescs[] = {
        {0, descriptorRanges, 2},
        {1, descriptorRanges + 2, 1},
        {2, descriptorRanges + 2, 1},
    };

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
    pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
    pipelineLayoutDesc.shaderStages = nri::StageBits::RAYGEN_SHADER | nri::StageBits::CLOSEST_HIT_SHADER;

    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    utils::ShaderCodeStorage shaderCodeStorage;
    nri::ShaderDesc shaders[] = {
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingBox.rgen", shaderCodeStorage, "raygen"),
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingBox.rmiss", shaderCodeStorage, "miss"),
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingBox.rchit", shaderCodeStorage, "closest_hit"),
    };

    nri::ShaderLibrary shaderLibrary = {};
    shaderLibrary.shaders = shaders;
    shaderLibrary.shaderNum = helper::GetCountOf(shaders);

    const nri::ShaderGroupDesc shaderGroupDescs[] = {{1}, {2}, {3}};

    nri::RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.recursionDepthMax = 1;
    pipelineDesc.payloadAttributeSizeMax = 3 * sizeof(float);
    pipelineDesc.intersectionAttributeSizeMax = 2 * sizeof(float);
    pipelineDesc.pipelineLayout = m_PipelineLayout;
    pipelineDesc.shaderGroupDescs = shaderGroupDescs;
    pipelineDesc.shaderGroupDescNum = helper::GetCountOf(shaderGroupDescs);
    pipelineDesc.shaderLibrary = &shaderLibrary;

    NRI_ABORT_ON_FAILURE(NRI.CreateRayTracingPipeline(*m_Device, pipelineDesc, m_Pipeline));
}

void Sample::CreateRayTracingOutput(nri::Format swapChainFormat) {
    nri::TextureDesc rayTracingOutputDesc = {};
    rayTracingOutputDesc.type = nri::TextureType::TEXTURE_2D;
    rayTracingOutputDesc.format = swapChainFormat;
    rayTracingOutputDesc.width = (uint16_t)GetWindowResolution().x;
    rayTracingOutputDesc.height = (uint16_t)GetWindowResolution().y;
    rayTracingOutputDesc.depth = 1;
    rayTracingOutputDesc.layerNum = 1;
    rayTracingOutputDesc.mipNum = 1;
    rayTracingOutputDesc.sampleNum = 1;
    rayTracingOutputDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, rayTracingOutputDesc, m_RayTracingOutput));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetTextureMemoryDesc(*m_RayTracingOutput, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;

    nri::Memory* memory = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, memory));
    m_MemoryAllocations.push_back(memory);

    const nri::TextureMemoryBindingDesc memoryBindingDesc = {memory, m_RayTracingOutput};
    NRI_ABORT_ON_FAILURE(NRI.BindTextureMemory(*m_Device, &memoryBindingDesc, 1));

    nri::Texture2DViewDesc textureViewDesc = {m_RayTracingOutput, nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D, swapChainFormat};
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, m_RayTracingOutputView));

    const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_RayTracingOutputView, 1, 0};
    NRI.UpdateDescriptorRanges(*m_DescriptorSets[0], 0, 1, &descriptorRangeUpdateDesc);
}

void Sample::CreateDescriptorSets() {
    nri::DescriptorPoolDesc descriptorPoolDesc = {};
    descriptorPoolDesc.storageTextureMaxNum = 1;
    descriptorPoolDesc.accelerationStructureMaxNum = 1;
    descriptorPoolDesc.bufferMaxNum = BOX_NUM * 2;
    descriptorPoolDesc.descriptorSetMaxNum = helper::GetCountOf(m_DescriptorSets);

    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSets[0], 1, 0));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 1, &m_DescriptorSets[1], 1, BOX_NUM));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 2, &m_DescriptorSets[2], 1, BOX_NUM));
}

void Sample::CreateShaderResources() {
    const uint32_t triangleNum = helper::GetCountOf(indices) / 3;
    std::vector<uint16_t> paddedIndices(triangleNum * 4, 0);
    for (uint32_t i = 0; i < triangleNum; i++) {
        paddedIndices[i * 4] = indices[i * 3];
        paddedIndices[i * 4 + 1] = indices[i * 3 + 1];
        paddedIndices[i * 4 + 2] = indices[i * 3 + 2];
    }

    const nri::BufferDesc texCoordBufferDesc = {sizeof(texCoords), 0, nri::BufferUsageBits::SHADER_RESOURCE};
    const nri::BufferDesc indexBufferDesc = {helper::GetByteSizeOf(paddedIndices), 0, nri::BufferUsageBits::SHADER_RESOURCE};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, texCoordBufferDesc, m_TexCoordBuffer));
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, indexBufferDesc, m_IndexBuffer));

    nri::Buffer* buffers[] = {m_TexCoordBuffer, m_IndexBuffer};

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = helper::GetCountOf(buffers);
    resourceGroupDesc.buffers = buffers;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    nri::BufferUploadDesc dataDescArray[] = {
        {texCoords, texCoordBufferDesc.size, m_TexCoordBuffer, 0, {nri::AccessBits::SHADER_RESOURCE}},
        {paddedIndices.data(), indexBufferDesc.size, m_IndexBuffer, 0, {nri::AccessBits::SHADER_RESOURCE}},
    };
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, dataDescArray, helper::GetCountOf(dataDescArray)));

    nri::BufferViewDesc texCoordBufferViewDesc = {};
    texCoordBufferViewDesc.buffer = m_TexCoordBuffer;
    texCoordBufferViewDesc.viewType = nri::BufferViewType::SHADER_RESOURCE;
    texCoordBufferViewDesc.format = nri::Format::RG32_SFLOAT;
    texCoordBufferViewDesc.size = texCoordBufferDesc.size;

    nri::BufferViewDesc indexBufferViewDesc = {};
    indexBufferViewDesc.buffer = m_IndexBuffer;
    indexBufferViewDesc.viewType = nri::BufferViewType::SHADER_RESOURCE;
    indexBufferViewDesc.format = nri::Format::RGBA16_UINT;
    indexBufferViewDesc.size = indexBufferDesc.size;

    NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(texCoordBufferViewDesc, m_TexCoordBufferView));
    NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(indexBufferViewDesc, m_IndexBufferView));

    nri::DescriptorRangeUpdateDesc rangeUpdateDesc = {};
    rangeUpdateDesc.descriptorNum = 1;
    rangeUpdateDesc.descriptors = &m_TexCoordBufferView;

    for (uint32_t i = 0; i < BOX_NUM; i++) {
        rangeUpdateDesc.baseDescriptor = i;
        NRI.UpdateDescriptorRanges(*m_DescriptorSets[1], 0, 1, &rangeUpdateDesc);
    }

    rangeUpdateDesc.descriptorNum = 1;
    rangeUpdateDesc.descriptors = &m_IndexBufferView;

    for (uint32_t i = 0; i < BOX_NUM; i++) {
        rangeUpdateDesc.baseDescriptor = i;
        NRI.UpdateDescriptorRanges(*m_DescriptorSets[2], 0, 1, &rangeUpdateDesc);
    }
}

void Sample::CreateBottomLevelAccelerationStructure() {
    nri::Buffer* buffer = nullptr;
    nri::Memory* memory = nullptr;
    CreateUploadBuffer(sizeof(positions) + sizeof(indices), nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT, buffer, memory);

    uint8_t* data = (uint8_t*)NRI.MapBuffer(*buffer, 0, sizeof(positions) + sizeof(indices));
    memcpy(data, positions, sizeof(positions));
    memcpy(data + sizeof(positions), indices, sizeof(indices));
    NRI.UnmapBuffer(*buffer);

    nri::GeometryObject object = {};
    object.type = nri::GeometryType::TRIANGLES;
    object.flags = nri::BottomLevelGeometryBits::OPAQUE_GEOMETRY;
    object.geometry.triangles.vertexBuffer = buffer;
    object.geometry.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
    object.geometry.triangles.vertexNum = helper::GetCountOf(positions) / 3;
    object.geometry.triangles.vertexStride = 3 * sizeof(float);
    object.geometry.triangles.indexBuffer = buffer;
    object.geometry.triangles.indexOffset = sizeof(positions);
    object.geometry.triangles.indexNum = helper::GetCountOf(indices);
    object.geometry.triangles.indexType = nri::IndexType::UINT16;

    nri::AccelerationStructureDesc accelerationStructureBLASDesc = {};
    accelerationStructureBLASDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
    accelerationStructureBLASDesc.flags = BUILD_FLAGS;
    accelerationStructureBLASDesc.instanceOrGeometryObjectNum = 1;
    accelerationStructureBLASDesc.geometryObjects = &object;

    NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureBLASDesc, m_BLAS));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetAccelerationStructureMemoryDesc(*m_BLAS, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;

    nri::Memory* ASMemory = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, ASMemory));
    m_MemoryAllocations.push_back(ASMemory);

    const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {ASMemory, m_BLAS};
    NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

    BuildBottomLevelAccelerationStructure(*m_BLAS, &object, 1);

    NRI.DestroyBuffer(*buffer);
    NRI.FreeMemory(*memory);
}

void Sample::CreateTopLevelAccelerationStructure() {
    nri::AccelerationStructureDesc accelerationStructureTLASDesc = {};
    accelerationStructureTLASDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
    accelerationStructureTLASDesc.flags = BUILD_FLAGS;
    accelerationStructureTLASDesc.instanceOrGeometryObjectNum = BOX_NUM;

    NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureTLASDesc, m_TLAS));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetAccelerationStructureMemoryDesc(*m_TLAS, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;

    nri::Memory* ASMemory = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, ASMemory));
    m_MemoryAllocations.push_back(ASMemory);

    const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {ASMemory, m_TLAS};
    NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

    std::vector<nri::GeometryObjectInstance> geometryObjectInstances(BOX_NUM, nri::GeometryObjectInstance{});

    const float lineWidth = 120.0f;
    const uint32_t lineSize = 100;
    const float step = lineWidth / (lineSize - 1);

    for (uint32_t i = 0; i < geometryObjectInstances.size(); i++) {
        nri::GeometryObjectInstance& instance = geometryObjectInstances[i];
        instance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_BLAS);
        instance.instanceId = i;
        instance.transform[0][0] = 1.0f;
        instance.transform[1][1] = 1.0f;
        instance.transform[2][2] = 1.0f;
        instance.transform[0][3] = -lineWidth * 0.5f + (i % lineSize) * step;
        instance.transform[1][3] = -10.0f + (i / lineSize) * step;
        instance.transform[2][3] = 10.0f + (i / lineSize) * step;
        instance.mask = 0xff;
    }

    nri::Buffer* buffer = nullptr;
    nri::Memory* memory = nullptr;
    CreateUploadBuffer(helper::GetByteSizeOf(geometryObjectInstances), nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT, buffer, memory);

    void* data = NRI.MapBuffer(*buffer, 0, nri::WHOLE_SIZE);
    memcpy(data, geometryObjectInstances.data(), helper::GetByteSizeOf(geometryObjectInstances));
    NRI.UnmapBuffer(*buffer);

    BuildTopLevelAccelerationStructure(*m_TLAS, (uint32_t)geometryObjectInstances.size(), *buffer);

    NRI.DestroyBuffer(*buffer);
    NRI.FreeMemory(*memory);

    NRI.CreateAccelerationStructureDescriptor(*m_TLAS, m_TLASDescriptor);

    const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_TLASDescriptor, 1, 0};
    NRI.UpdateDescriptorRanges(*m_DescriptorSets[0], 1, 1, &descriptorRangeUpdateDesc);
}

void Sample::CreateUploadBuffer(uint64_t size, nri::BufferUsageBits usage, nri::Buffer*& buffer, nri::Memory*& memory) {
    const nri::BufferDesc bufferDesc = {size, 0, usage};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryDesc(*buffer, nri::MemoryLocation::HOST_UPLOAD, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = {memory, buffer};
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory) {
    const uint64_t scratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(accelerationStructure);

    const nri::BufferDesc bufferDesc = {scratchBufferSize, 0, nri::BufferUsageBits::SCRATCH_BUFFER};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryDesc(*buffer, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, memory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = {memory, buffer};
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum) {
    nri::Buffer* scratchBuffer = nullptr;
    nri::Memory* scratchBufferMemory = nullptr;
    CreateScratchBuffer(accelerationStructure, scratchBuffer, scratchBufferMemory);

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandAllocator(*m_GraphicsQueue, commandAllocator);
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;

    NRI.BeginCommandBuffer(*commandBuffer, nullptr);
    {
        NRI.CmdBuildBottomLevelAccelerationStructure(*commandBuffer, objectNum, objects, BUILD_FLAGS, accelerationStructure, *scratchBuffer, 0);
    }
    NRI.EndCommandBuffer(*commandBuffer);
    NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    NRI.WaitForIdle(*m_GraphicsQueue);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    NRI.DestroyBuffer(*scratchBuffer);
    NRI.FreeMemory(*scratchBufferMemory);
}

void Sample::BuildTopLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, uint32_t instanceNum, nri::Buffer& instanceBuffer) {
    nri::Buffer* scratchBuffer = nullptr;
    nri::Memory* scratchBufferMemory = nullptr;
    CreateScratchBuffer(accelerationStructure, scratchBuffer, scratchBufferMemory);

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandAllocator(*m_GraphicsQueue, commandAllocator);
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;

    NRI.BeginCommandBuffer(*commandBuffer, nullptr);
    {
        NRI.CmdBuildTopLevelAccelerationStructure(*commandBuffer, instanceNum, instanceBuffer, 0, BUILD_FLAGS, accelerationStructure, *scratchBuffer, 0);
    }
    NRI.EndCommandBuffer(*commandBuffer);
    NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    NRI.WaitForIdle(*m_GraphicsQueue);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    NRI.DestroyBuffer(*scratchBuffer);
    NRI.FreeMemory(*scratchBufferMemory);
}

void Sample::CreateShaderTable() {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    const uint64_t identifierSize = deviceDesc.rayTracingShaderGroupIdentifierSize;
    const uint64_t tableAlignment = deviceDesc.shaderBindingTableAlignment;

    m_ShaderGroupIdentifierSize = identifierSize;
    m_MissShaderOffset = helper::Align(identifierSize, tableAlignment);
    m_HitShaderGroupOffset = helper::Align(m_MissShaderOffset + identifierSize, tableAlignment);
    const uint64_t shaderTableSize = helper::Align(m_HitShaderGroupOffset + identifierSize, tableAlignment);

    const nri::BufferDesc bufferDesc = {shaderTableSize, 0, nri::BufferUsageBits::SHADER_BINDING_TABLE};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ShaderTable));

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.bufferNum = 1;
    resourceGroupDesc.buffers = &m_ShaderTable;

    const size_t baseAllocation = m_MemoryAllocations.size();
    m_MemoryAllocations.resize(baseAllocation + 1, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_MemoryAllocations.data() + baseAllocation));

    std::vector<uint8_t> content((size_t)shaderTableSize, 0);
    for (uint32_t i = 0; i < 3; i++)
        NRI.WriteShaderGroupIdentifiers(*m_Pipeline, i, 1, content.data() + i * helper::Align(identifierSize, tableAlignment));

    nri::BufferUploadDesc dataDesc = {};
    dataDesc.data = content.data();
    dataDesc.dataSize = content.size();
    dataDesc.buffer = m_ShaderTable;
    dataDesc.after = {nri::AccessBits::UNKNOWN};
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*m_GraphicsQueue, nullptr, 0, &dataDesc, 1));
}

SAMPLE_MAIN(Sample, 0);
