// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#define VK_MINOR_VERSION 3

#ifdef _WIN32
#    undef APIENTRY // defined in GLFW

#    include <d3d11.h>
#    include "Extensions/NRIWrapperD3D11.h"

#    include <d3d12.h>
#    include "Extensions/NRIWrapperD3D12.h"

#    define VK_USE_PLATFORM_WIN32_KHR
const char* VULKAN_LIB = "vulkan-1.dll";
#elif defined(__APPLE__)
#    define VK_USE_PLATFORM_METAL_EXT
const char* VULKAN_LIB = "libvulkan.dynlib";
#else
#    define VK_USE_PLATFORM_XLIB_KHR
const char* VULKAN_LIB = "libvulkan.so";
#endif

#define VK_NO_PROTOTYPES 1
#include "vulkan/vulkan.h"

#include "Extensions/NRIWrapperVK.h"

struct Library;

#ifdef _WIN32
#    include <windows.h>

static Library* LoadSharedLibrary(const char* path) {
    return (Library*)LoadLibraryA(path);
}

static void* GetSharedLibraryFunction(Library& library, const char* name) {
    return (void*)GetProcAddress((HMODULE)&library, name);
}

static void UnloadSharedLibrary(Library& library) {
    FreeLibrary((HMODULE)&library);
}

#elif defined(__linux__) || defined(__APPLE__)
#    include <dlfcn.h>

static Library* LoadSharedLibrary(const char* path) {
    return (Library*)dlopen(path, RTLD_NOW);
}

static void* GetSharedLibraryFunction(Library& library, const char* name) {
    return dlsym((void*)&library, name);
}

static void UnloadSharedLibrary(Library& library) {
    dlclose((void*)&library);
}

#else
#    error unknown platform
#endif

constexpr nri::Color32f COLOR_0 = {0.5f, 0.0f, 1.0f, 1.0f};
constexpr nri::Color32f COLOR_1 = {0.72f, 0.46f, 0.0f, 1.0f};

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
    void CreateD3D11Device();
    void CreateD3D12Device();
    void CreateVulkanDevice();

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
    nri::Buffer* m_ConstantBuffer = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Texture* m_Texture = nullptr;

    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<nri::Memory*> m_MemoryAllocations;

#ifdef _WIN32
    ID3D11Device* m_D3D11Device = nullptr;
    ID3D12Device* m_D3D12Device = nullptr;
#endif
    VkInstance m_VKInstance = VK_NULL_HANDLE;
    VkDevice m_VKDevice = VK_NULL_HANDLE;
    Library* m_VulkanLoader = nullptr;

    uint64_t m_GeometryOffset = 0;
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

    if (m_VulkanLoader) {
        auto vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetSharedLibraryFunction(*m_VulkanLoader, "vkGetInstanceProcAddr");
        auto vkDestroyInstance = (PFN_vkDestroyInstance)vkGetInstanceProcAddr(m_VKInstance, "vkDestroyInstance");
        auto vkDestroyDevice = (PFN_vkDestroyDevice)vkGetInstanceProcAddr(m_VKInstance, "vkDestroyDevice");

        vkDestroyDevice(m_VKDevice, nullptr);
        vkDestroyInstance(m_VKInstance, nullptr);
        UnloadSharedLibrary(*m_VulkanLoader);
    }

#ifdef _WIN32
    if (m_D3D11Device)
        m_D3D11Device->Release();

    if (m_D3D12Device)
        m_D3D12Device->Release();
#endif
}

void Sample::CreateD3D11Device() {
#ifdef _WIN32
    const HRESULT result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, m_DebugAPI ? D3D11_CREATE_DEVICE_DEBUG : 0, nullptr, 0, D3D11_SDK_VERSION, &m_D3D11Device, nullptr, nullptr);

    NRI_ABORT_ON_FALSE(SUCCEEDED(result));

    nri::DeviceCreationD3D11Desc deviceDesc = {};
    deviceDesc.d3d11Device = m_D3D11Device;
    deviceDesc.allocationCallbacks = m_AllocationCallbacks;
    deviceDesc.enableNRIValidation = m_DebugNRI;

    NRI_ABORT_ON_FAILURE(nri::nriCreateDeviceFromD3D11Device(deviceDesc, m_Device));
#endif
}

void Sample::CreateD3D12Device() {
#ifdef _WIN32
    if (m_DebugAPI) {
        ID3D12Debug* debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            debugController->Release();
        }
    }

    const HRESULT result = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(m_D3D12Device), (void**)&m_D3D12Device);

    NRI_ABORT_ON_FALSE(SUCCEEDED(result));

    nri::QueueFamilyD3D12Desc queueFamily = {};
    queueFamily.queueType = nri::QueueType::GRAPHICS;
    queueFamily.queueNum = 1;

    nri::DeviceCreationD3D12Desc deviceDesc = {};
    deviceDesc.d3d12Device = m_D3D12Device;
    deviceDesc.allocationCallbacks = m_AllocationCallbacks;
    deviceDesc.enableNRIValidation = m_DebugNRI;
    deviceDesc.queueFamilies = &queueFamily;
    deviceDesc.queueFamilyNum = 1;

    NRI_ABORT_ON_FAILURE(nri::nriCreateDeviceFromD3D12Device(deviceDesc, m_Device));
#endif
}

void Sample::CreateVulkanDevice() {
    m_VulkanLoader = LoadSharedLibrary(VULKAN_LIB);

    auto vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetSharedLibraryFunction(*m_VulkanLoader, "vkGetInstanceProcAddr");
    auto vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");

    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.apiVersion = VK_MAKE_API_VERSION(0, 1, VK_MINOR_VERSION, 0);

    const char* instanceExtensions[] = {
#ifdef _WIN32
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif defined(__APPLE__)
        VK_EXT_METAL_SURFACE_EXTENSION_NAME,
#else
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
        VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
    };
    const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if (VK_MINOR_VERSION == 2)
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
#endif
    };
    const char* layers[] = {"VK_LAYER_KHRONOS_validation"};

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions;
    instanceCreateInfo.enabledExtensionCount = helper::GetCountOf(instanceExtensions);
    instanceCreateInfo.ppEnabledLayerNames = layers;
    instanceCreateInfo.enabledLayerCount = m_DebugAPI ? 1 : 0;

    VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &m_VKInstance);
    NRI_ABORT_ON_FALSE(result == VK_SUCCESS);

    auto vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr(m_VKInstance, "vkEnumeratePhysicalDevices");
    auto vkCreateDevice = (PFN_vkCreateDevice)vkGetInstanceProcAddr(m_VKInstance, "vkCreateDevice");
    auto vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(m_VKInstance, "vkGetPhysicalDeviceFeatures2");

    uint32_t physicalDeviceNum = 0;
    vkEnumeratePhysicalDevices(m_VKInstance, &physicalDeviceNum, nullptr);

    NRI_ABORT_ON_FALSE(physicalDeviceNum != 0);

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceNum);
    vkEnumeratePhysicalDevices(m_VKInstance, &physicalDeviceNum, physicalDevices.data());

    VkPhysicalDevice physicalDevice = physicalDevices[0];

    VkPhysicalDeviceFeatures2 deviceFeatures2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    VkPhysicalDeviceVulkan11Features featuresVulkan11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    deviceFeatures2.pNext = &featuresVulkan11;

    VkPhysicalDeviceVulkan12Features featuresVulkan12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    featuresVulkan11.pNext = &featuresVulkan12;

#if (VK_MINOR_VERSION == 2)
    VkPhysicalDeviceSynchronization2Features synchronization2features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    featuresVulkan12.pNext = &synchronization2features;

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    synchronization2features.pNext = &dynamicRenderingFeatures;

    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    dynamicRenderingFeatures.pNext = &extendedDynamicStateFeatures;
#else
    VkPhysicalDeviceVulkan13Features featuresVulkan13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    featuresVulkan12.pNext = &featuresVulkan13;
#endif

    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);

    const float priority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.pQueuePriorities = &priority;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.queueFamilyIndex = 0; // blind shot!

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = &deviceFeatures2;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.enabledExtensionCount = helper::GetCountOf(deviceExtensions);
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

    result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &m_VKDevice);
    NRI_ABORT_ON_FALSE(result == VK_SUCCESS);

    // Wrap the device
    nri::QueueFamilyVKDesc queueFamily = {};
    queueFamily.queueType = nri::QueueType::GRAPHICS;
    queueFamily.queueNum = queueCreateInfo.queueCount;
    queueFamily.familyIndex = queueCreateInfo.queueFamilyIndex;

    nri::DeviceCreationVKDesc deviceDesc = {};
    deviceDesc.allocationCallbacks = m_AllocationCallbacks;
    deviceDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceDesc.vkExtensions.instanceExtensions = instanceExtensions;
    deviceDesc.vkExtensions.instanceExtensionNum = helper::GetCountOf(instanceExtensions);
    deviceDesc.vkExtensions.deviceExtensions = deviceExtensions;
    deviceDesc.vkExtensions.deviceExtensionNum = helper::GetCountOf(deviceExtensions);
    deviceDesc.vkInstance = (VKHandle)m_VKInstance;
    deviceDesc.vkDevice = (VKHandle)m_VKDevice;
    deviceDesc.vkPhysicalDevice = (VKHandle)physicalDevice;
    deviceDesc.queueFamilies = &queueFamily;
    deviceDesc.queueFamilyNum = 1;
    deviceDesc.minorVersion = VK_MINOR_VERSION;

    NRI_ABORT_ON_FAILURE(nri::nriCreateDeviceFromVkDevice(deviceDesc, m_Device));
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI) {
    switch (graphicsAPI) {
        case nri::GraphicsAPI::D3D11:
            CreateD3D11Device();
            break;
        case nri::GraphicsAPI::D3D12:
            CreateD3D12Device();
            break;
        case nri::GraphicsAPI::VK:
            CreateVulkanDevice();
            break;
    }

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
        vertexStreamDesc.stride = sizeof(Vertex);

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
            utils::LoadShader(deviceDesc.graphicsAPI, "Triangle.vs", shaderCodeStorage),
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
    }

    // Descriptor pool
    {
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
        // Texture
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;
        textureDesc.format = texture.GetFormat();
        textureDesc.width = texture.GetWidth();
        textureDesc.height = texture.GetHeight();
        textureDesc.mipNum = texture.GetMipNum();

        NRI_ABORT_ON_FAILURE(NRI.CreateTexture(*m_Device, textureDesc, m_Texture));

        // Constant buffer
        {
            nri::BufferDesc bufferDesc = {};
            bufferDesc.size = constantBufferSize * BUFFERED_FRAME_MAX_NUM;
            bufferDesc.usage = nri::BufferUsageBits::CONSTANT_BUFFER;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));
        }

        // Geometry buffer
        {
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

    // Descriptors
    {
        // Texture
        nri::Texture2DViewDesc texture2DViewDesc = {m_Texture, nri::Texture2DViewType::SHADER_RESOURCE_2D, texture.GetFormat()};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_TextureShaderResource));

        // Sampler
        nri::SamplerDesc samplerDesc = {};
        samplerDesc.addressModes = {nri::AddressMode::MIRRORED_REPEAT, nri::AddressMode::MIRRORED_REPEAT};
        samplerDesc.filters = {nri::Filter::LINEAR, nri::Filter::LINEAR, nri::Filter::LINEAR};
        samplerDesc.anisotropy = 4;
        samplerDesc.mipMax = 16.0f;
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, m_Sampler));

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

    // Descriptor sets
    {
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

    // Upload data
    {
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
    }
    ImGui::End();

    EndUI(NRI, *m_Streamer);
    NRI.CopyStreamerUpdateRequests(*m_Streamer);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::Dim_t windowWidth = (nri::Dim_t)GetWindowResolution().x;
    nri::Dim_t windowHeight = (nri::Dim_t)GetWindowResolution().y;
    nri::Dim_t halfWidth = windowWidth / 2;
    nri::Dim_t halfHeight = windowHeight / 2;

    const uint32_t bufferedFrameIndex = frameIndex % BUFFERED_FRAME_MAX_NUM;
    const Frame& frame = m_Frames[bufferedFrameIndex];

    if (frameIndex >= BUFFERED_FRAME_MAX_NUM) {
        NRI.Wait(*m_FrameFence, 1 + frameIndex - BUFFERED_FRAME_MAX_NUM);
        NRI.ResetCommandAllocator(*frame.commandAllocator);
    }

    ConstantBufferLayout* commonConstants = (ConstantBufferLayout*)NRI.MapBuffer(*m_ConstantBuffer, frame.constantBufferViewOffset, sizeof(ConstantBufferLayout));
    if (commonConstants) {
        commonConstants->color[0] = 0.8f;
        commonConstants->color[1] = 0.5f;
        commonConstants->color[2] = 0.1f;
        commonConstants->scale = m_Scale;

        NRI.UnmapBuffer(*m_ConstantBuffer);
    }

    const uint32_t currentTextureIndex = NRI.AcquireNextSwapChainTexture(*m_SwapChain);
    BackBuffer& currentBackBuffer = m_SwapChainBuffers[currentTextureIndex];

    nri::TextureBarrierDesc textureBarrierDescs = {};
    textureBarrierDescs.texture = currentBackBuffer.texture;
    textureBarrierDescs.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
    textureBarrierDescs.layerNum = 1;
    textureBarrierDescs.mipNum = 1;

    // Record
    nri::CommandBuffer* commandBuffer = frame.commandBuffer;
    NRI.BeginCommandBuffer(*commandBuffer, m_DescriptorPool);
    {
        nri::BarrierGroupDesc barrierGroupDesc = {};
        barrierGroupDesc.textureNum = 1;
        barrierGroupDesc.textures = &textureBarrierDescs;
        NRI.CmdBarrier(*commandBuffer, barrierGroupDesc);

        nri::AttachmentsDesc attachmentsDesc = {};
        attachmentsDesc.colorNum = 1;
        attachmentsDesc.colors = &currentBackBuffer.colorAttachment;

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
                rects[0] = {0, 0, halfWidth, halfHeight};
                rects[1] = {(int16_t)halfWidth, (int16_t)halfHeight, halfWidth, halfHeight};

                NRI.CmdClearAttachments(*commandBuffer, &clearDesc, 1, rects, helper::GetCountOf(rects));
            }

            {
                helper::Annotation annotation(NRI, *commandBuffer, "Triangle");

                const nri::Viewport viewport = {0.0f, 0.0f, (float)windowWidth, (float)windowHeight, 0.0f, 1.0f};
                NRI.CmdSetViewports(*commandBuffer, &viewport, 1);

                NRI.CmdSetPipelineLayout(*commandBuffer, *m_PipelineLayout);
                NRI.CmdSetPipeline(*commandBuffer, *m_Pipeline);
                NRI.CmdSetRootConstants(*commandBuffer, 0, &m_Transparency, 4);
                NRI.CmdSetIndexBuffer(*commandBuffer, *m_GeometryBuffer, 0, nri::IndexType::UINT16);
                NRI.CmdSetVertexBuffers(*commandBuffer, 0, 1, &m_GeometryBuffer, &m_GeometryOffset);
                NRI.CmdSetDescriptorSet(*commandBuffer, 0, *frame.constantBufferDescriptorSet, nullptr);
                NRI.CmdSetDescriptorSet(*commandBuffer, 1, *m_TextureDescriptorSet, nullptr);

                nri::Rect scissor = {0, 0, halfWidth, windowHeight};
                NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                NRI.CmdDrawIndexed(*commandBuffer, {3, 1, 0, 0, 0});

                scissor = {(int16_t)halfWidth, (int16_t)halfHeight, halfWidth, halfHeight};
                NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                NRI.CmdDraw(*commandBuffer, {3, 1, 0, 0});
            }

            {
                helper::Annotation annotation(NRI, *commandBuffer, "UI");

                RenderUI(NRI, NRI, *m_Streamer, *commandBuffer, 1.0f, true);
            }
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
