// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#include <array>

constexpr auto BUILD_FLAGS = nri::AccelerationStructureBits::PREFER_FAST_TRACE;

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
    void CreateDescriptorSet();
    void CreateBottomLevelAccelerationStructure();
    void CreateTopLevelAccelerationStructure();
    void CreateShaderTable();
    void CreateUploadBuffer(uint64_t size, nri::BufferUsageBits usage, nri::Buffer*& buffer, nri::Memory*& memory);
    void CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory);
    void BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::BottomLevelGeometryDesc* objects, const uint32_t objectNum);
    void BuildTopLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, uint32_t instanceNum, nri::Buffer& instanceBuffer);

    NRIInterface NRI = {};
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};

    nri::Pipeline* m_Pipeline = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;

    nri::Buffer* m_ShaderTable = nullptr;
    nri::Memory* m_ShaderTableMemory = nullptr;
    uint64_t m_ShaderGroupIdentifierSize = 0;
    uint64_t m_MissShaderOffset = 0;
    uint64_t m_HitShaderGroupOffset = 0;

    nri::Texture* m_RayTracingOutput = nullptr;
    nri::Descriptor* m_RayTracingOutputView = nullptr;

    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;

    nri::AccelerationStructure* m_BLAS = nullptr;
    nri::AccelerationStructure* m_TLAS = nullptr;
    nri::Descriptor* m_TLASDescriptor = nullptr;
    nri::Memory* m_BLASMemory = nullptr;
    nri::Memory* m_TLASMemory = nullptr;

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

    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);

    NRI.DestroyFence(*m_FrameFence);

    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

    NRI.FreeMemory(*m_BLASMemory);
    NRI.FreeMemory(*m_TLASMemory);
    NRI.FreeMemory(*m_ShaderTableMemory);

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
    CreateDescriptorSet();
    CreateRayTracingOutput(swapChainFormat);
    CreateBottomLevelAccelerationStructure();
    CreateTopLevelAccelerationStructure();
    CreateShaderTable();

    return InitUI(NRI, NRI, *m_Device, swapChainFormat);
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

        NRI.CmdBarrier(commandBuffer, barrierGroupDesc);
        NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
        NRI.CmdSetPipeline(commandBuffer, *m_Pipeline);
        NRI.CmdSetDescriptorSet(commandBuffer, 0, *m_DescriptorSet, nullptr);

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
    nri::DescriptorRangeDesc descriptorRanges[2] = {};
    descriptorRanges[0].descriptorNum = 1;
    descriptorRanges[0].descriptorType = nri::DescriptorType::STORAGE_TEXTURE;
    descriptorRanges[0].baseRegisterIndex = 0;
    descriptorRanges[0].shaderStages = nri::StageBits::RAYGEN_SHADER;

    descriptorRanges[1].descriptorNum = 1;
    descriptorRanges[1].descriptorType = nri::DescriptorType::ACCELERATION_STRUCTURE;
    descriptorRanges[1].baseRegisterIndex = 1;
    descriptorRanges[1].shaderStages = nri::StageBits::RAYGEN_SHADER;

    nri::DescriptorSetDesc descriptorSetDesc = {0, descriptorRanges, helper::GetCountOf(descriptorRanges)};

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = &descriptorSetDesc;
    pipelineLayoutDesc.descriptorSetNum = 1;
    pipelineLayoutDesc.shaderStages = nri::StageBits::RAYGEN_SHADER;

    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_PipelineLayout));

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    utils::ShaderCodeStorage shaderCodeStorage;
    nri::ShaderDesc shaders[] = {
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingTriangle.rgen", shaderCodeStorage, "raygen"),
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingTriangle.rmiss", shaderCodeStorage, "miss"),
        utils::LoadShader(deviceDesc.graphicsAPI, "RayTracingTriangle.rchit", shaderCodeStorage, "closest_hit"),
    };

    nri::ShaderLibraryDesc shaderLibrary = {};
    shaderLibrary.shaders = shaders;
    shaderLibrary.shaderNum = helper::GetCountOf(shaders);

    const nri::ShaderGroupDesc shaderGroupDescs[] = {{1}, {2}, {3}};

    nri::RayTracingPipelineDesc pipelineDesc = {};
    pipelineDesc.recursionMaxDepth = 1;
    pipelineDesc.rayPayloadMaxSize = 3 * sizeof(float);
    pipelineDesc.rayHitAttributeMaxSize = 2 * sizeof(float);
    pipelineDesc.pipelineLayout = m_PipelineLayout;
    pipelineDesc.shaderGroups = shaderGroupDescs;
    pipelineDesc.shaderGroupNum = helper::GetCountOf(shaderGroupDescs);
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

    const nri::TextureMemoryBindingDesc memoryBindingDesc = {m_RayTracingOutput, memory};
    NRI_ABORT_ON_FAILURE(NRI.BindTextureMemory(*m_Device, &memoryBindingDesc, 1));

    nri::Texture2DViewDesc textureViewDesc = {m_RayTracingOutput, nri::Texture2DViewType::SHADER_RESOURCE_STORAGE_2D, swapChainFormat};
    NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(textureViewDesc, m_RayTracingOutputView));

    const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_RayTracingOutputView, 1, 0};
    NRI.UpdateDescriptorRanges(*m_DescriptorSet, 0, 1, &descriptorRangeUpdateDesc);
}

void Sample::CreateDescriptorSet() {
    nri::DescriptorPoolDesc descriptorPoolDesc = {};
    descriptorPoolDesc.storageTextureMaxNum = 1;
    descriptorPoolDesc.accelerationStructureMaxNum = 1;
    descriptorPoolDesc.descriptorSetMaxNum = 1;

    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_DescriptorPool));
    NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0));
}

void Sample::CreateBottomLevelAccelerationStructure() {
    const uint64_t vertexDataSize = 3 * 3 * sizeof(float);
    const uint64_t indexDataSize = 3 * sizeof(uint16_t);

    nri::Buffer* buffer = nullptr;
    nri::Memory* memory = nullptr;
    CreateUploadBuffer(vertexDataSize + indexDataSize, nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT, buffer, memory);

    const float positions[] = {-0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.5f, -0.5f, 0.0f};
    const uint16_t indices[] = {0, 1, 2};

    uint8_t* data = (uint8_t*)NRI.MapBuffer(*buffer, 0, vertexDataSize + indexDataSize);
    memcpy(data, positions, sizeof(positions));
    memcpy(data + vertexDataSize, indices, sizeof(indices));
    NRI.UnmapBuffer(*buffer);

    nri::BottomLevelGeometryDesc object = {};
    object.type = nri::BottomLevelGeometryType::TRIANGLES;
    object.flags = nri::BottomLevelGeometryBits::OPAQUE_GEOMETRY;
    object.triangles.vertexBuffer = buffer;
    object.triangles.vertexFormat = nri::Format::RGB32_SFLOAT;
    object.triangles.vertexNum = 3;
    object.triangles.vertexStride = 3 * sizeof(float);
    object.triangles.indexBuffer = buffer;
    object.triangles.indexOffset = vertexDataSize;
    object.triangles.indexNum = 3;
    object.triangles.indexType = nri::IndexType::UINT16;

    nri::AccelerationStructureDesc accelerationStructureDesc = {};
    accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;
    accelerationStructureDesc.flags = BUILD_FLAGS;
    accelerationStructureDesc.geometryOrInstanceNum = 1;
    accelerationStructureDesc.geometries = &object;

    NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, m_BLAS));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetAccelerationStructureMemoryDesc(*m_BLAS, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, m_BLASMemory));

    const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {m_BLAS, m_BLASMemory};
    NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

    BuildBottomLevelAccelerationStructure(*m_BLAS, &object, 1);

    NRI.DestroyBuffer(*buffer);
    NRI.FreeMemory(*memory);
}

void Sample::CreateTopLevelAccelerationStructure() {
    nri::AccelerationStructureDesc accelerationStructureDesc = {};
    accelerationStructureDesc.type = nri::AccelerationStructureType::TOP_LEVEL;
    accelerationStructureDesc.flags = BUILD_FLAGS;
    accelerationStructureDesc.geometryOrInstanceNum = 1;
    NRI_ABORT_ON_FAILURE(NRI.CreateAccelerationStructure(*m_Device, accelerationStructureDesc, m_TLAS));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetAccelerationStructureMemoryDesc(*m_TLAS, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, m_TLASMemory));

    const nri::AccelerationStructureMemoryBindingDesc memoryBindingDesc = {m_TLAS, m_TLASMemory};
    NRI_ABORT_ON_FAILURE(NRI.BindAccelerationStructureMemory(*m_Device, &memoryBindingDesc, 1));

    nri::Buffer* buffer = nullptr;
    nri::Memory* memory = nullptr;
    CreateUploadBuffer(sizeof(nri::TopLevelInstance), nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT, buffer, memory);

    nri::TopLevelInstance geometryObjectInstance = {};
    geometryObjectInstance.accelerationStructureHandle = NRI.GetAccelerationStructureHandle(*m_BLAS);
    geometryObjectInstance.transform[0][0] = 1.0f;
    geometryObjectInstance.transform[1][1] = 1.0f;
    geometryObjectInstance.transform[2][2] = 1.0f;
    geometryObjectInstance.mask = 0xFF;
    geometryObjectInstance.flags = nri::TopLevelInstanceBits::FORCE_OPAQUE;

    void* data = NRI.MapBuffer(*buffer, 0, sizeof(geometryObjectInstance));
    memcpy(data, &geometryObjectInstance, sizeof(geometryObjectInstance));
    NRI.UnmapBuffer(*buffer);

    BuildTopLevelAccelerationStructure(*m_TLAS, 1, *buffer);

    NRI.DestroyBuffer(*buffer);
    NRI.FreeMemory(*memory);

    NRI.CreateAccelerationStructureDescriptor(*m_TLAS, m_TLASDescriptor);

    const nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc = {&m_TLASDescriptor, 1, 0};
    NRI.UpdateDescriptorRanges(*m_DescriptorSet, 1, 1, &descriptorRangeUpdateDesc);
}

void Sample::CreateUploadBuffer(uint64_t size, nri::BufferUsageBits usage, nri::Buffer*& buffer, nri::Memory*& memory) {
    nri::BufferDesc bufferDesc = {size, 0, usage};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryDesc(*buffer, nri::MemoryLocation::HOST_UPLOAD, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, memory));

    nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = {buffer, memory};
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory) {
    uint64_t scratchBufferSize = NRI.GetAccelerationStructureBuildScratchBufferSize(accelerationStructure);

    nri::BufferDesc bufferDesc = {scratchBufferSize, 0, nri::BufferUsageBits::SCRATCH_BUFFER};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryDesc(*buffer, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, memory));

    nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = {buffer, memory};
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));
}

void Sample::BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::BottomLevelGeometryDesc* objects, const uint32_t objectNum) {
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
        nri::BuildBottomLevelAccelerationStructureDesc desc = {};
        desc.dst = &accelerationStructure;
        desc.geometries = objects;
        desc.geometryNum = objectNum;
        desc.scratchBuffer = scratchBuffer;

        NRI.CmdBuildBottomLevelAccelerationStructures(*commandBuffer, &desc, 1);
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
        nri::BuildTopLevelAccelerationStructureDesc desc = {};
        desc.dst = &accelerationStructure;
        desc.instanceNum = instanceNum;
        desc.instanceBuffer = &instanceBuffer;
        desc.scratchBuffer = scratchBuffer;

        NRI.CmdBuildTopLevelAccelerationStructures(*commandBuffer, &desc, 1);
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
    const uint64_t identifierSize = deviceDesc.shaderStage.rayTracing.shaderGroupIdentifierSize;
    const uint64_t tableAlignment = deviceDesc.memoryAlignment.shaderBindingTable;

    m_ShaderGroupIdentifierSize = identifierSize;
    m_MissShaderOffset = helper::Align(identifierSize, tableAlignment);
    m_HitShaderGroupOffset = helper::Align(m_MissShaderOffset + identifierSize, tableAlignment);
    const uint64_t shaderTableSize = helper::Align(m_HitShaderGroupOffset + identifierSize, tableAlignment);

    const nri::BufferDesc bufferDesc = {shaderTableSize, 0, nri::BufferUsageBits::SHADER_BINDING_TABLE};
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ShaderTable));

    nri::MemoryDesc memoryDesc = {};
    NRI.GetBufferMemoryDesc(*m_ShaderTable, nri::MemoryLocation::DEVICE, memoryDesc);

    nri::AllocateMemoryDesc allocateMemoryDesc = {};
    allocateMemoryDesc.size = memoryDesc.size;
    allocateMemoryDesc.type = memoryDesc.type;
    NRI_ABORT_ON_FAILURE(NRI.AllocateMemory(*m_Device, allocateMemoryDesc, m_ShaderTableMemory));

    const nri::BufferMemoryBindingDesc bufferMemoryBindingDesc = {m_ShaderTable, m_ShaderTableMemory};
    NRI_ABORT_ON_FAILURE(NRI.BindBufferMemory(*m_Device, &bufferMemoryBindingDesc, 1));

    nri::Buffer* buffer = nullptr;
    nri::Memory* memory = nullptr;
    CreateUploadBuffer(shaderTableSize, nri::BufferUsageBits::NONE, buffer, memory);

    uint8_t* data = (uint8_t*)NRI.MapBuffer(*buffer, 0, shaderTableSize);
    for (uint32_t i = 0; i < 3; i++)
        NRI.WriteShaderGroupIdentifiers(*m_Pipeline, i, 1, data + i * helper::Align(identifierSize, tableAlignment));
    NRI.UnmapBuffer(*buffer);

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    NRI.CreateCommandAllocator(*m_GraphicsQueue, commandAllocator);
    NRI.CreateCommandBuffer(*commandAllocator, commandBuffer);

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.commandBuffers = &commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;

    NRI.BeginCommandBuffer(*commandBuffer, nullptr);
    {
        NRI.CmdCopyBuffer(*commandBuffer, *m_ShaderTable, 0, *buffer, 0, shaderTableSize);
    }
    NRI.EndCommandBuffer(*commandBuffer);
    NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);
    NRI.WaitForIdle(*m_GraphicsQueue);

    NRI.DestroyCommandBuffer(*commandBuffer);
    NRI.DestroyCommandAllocator(*commandAllocator);

    NRI.DestroyBuffer(*buffer);
    NRI.FreeMemory(*memory);
}

SAMPLE_MAIN(Sample, 0);
