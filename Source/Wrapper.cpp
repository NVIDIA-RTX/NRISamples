// © 2021 NVIDIA Corporation

#include "NRIFramework.h"
#undef APIENTRY // defined in GLFW

#define VK_MINOR_VERSION 4

#define VK_NO_PROTOTYPES 1
#include "vulkan/vulkan.h"

#include "Extensions/NRIWrapperVK.h"

struct Library;

#if (NRIF_PLATFORM == NRIF_WINDOWS)
#    include <d3d11.h>
#    include "Extensions/NRIWrapperD3D11.h"

#    include <d3d12.h>
#    include "Extensions/NRIWrapperD3D12.h"

static Library* LoadSharedLibrary(const char* path) {
    return (Library*)LoadLibraryA(path);
}

static void* GetSharedLibraryFunction(Library& library, const char* name) {
    return (void*)GetProcAddress((HMODULE)&library, name);
}

static void UnloadSharedLibrary(Library& library) {
    FreeLibrary((HMODULE)&library);
}

#else
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
    {{-0.71f, -0.50f}, {0.0f, 0.0f}},
    {{0.00f, 0.71f}, {1.0f, 1.0f}},
    {{0.71f, -0.50f}, {0.0f, 1.0f}},
};

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
    nri::Buffer* m_ConstantBuffer = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Texture* m_Texture = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames = {};
    std::vector<SwapChainTexture> m_SwapChainTextures;
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
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocator);
            NRI.DestroyDescriptor(queuedFrame.constantBufferView);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        NRI.DestroyPipeline(m_Pipeline);
        NRI.DestroyPipelineLayout(m_PipelineLayout);
        NRI.DestroyDescriptor(m_TextureShaderResource);
        NRI.DestroyBuffer(m_ConstantBuffer);
        NRI.DestroyBuffer(m_GeometryBuffer);
        NRI.DestroyTexture(m_Texture);
        NRI.DestroyDescriptorPool(m_DescriptorPool);
        NRI.DestroyFence(m_FrameFence);

        for (nri::Memory* memory : m_MemoryAllocations)
            NRI.FreeMemory(memory);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(m_Streamer);

    DestroyImgui();

    nri::nriDestroyDevice(m_Device);

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
#if (NRIF_PLATFORM == NRIF_WINDOWS)
    const char* libraryName = "vulkan-1.dll";
#elif (NRIF_PLATFORM == NRIF_COCOA)
    const char* libraryName = "libvulkan.1.dylib";
#else
    const char* libraryName = "libvulkan.so.1";
#endif

    m_VulkanLoader = LoadSharedLibrary(libraryName);

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
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
#else
        VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
        VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME,
    };

    const char* deviceExtensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
#if (defined(__APPLE__) || VK_MINOR_VERSION < 4)
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_6_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
#endif
#if (defined(__APPLE__) || VK_MINOR_VERSION < 3)
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_COPY_COMMANDS_2_EXTENSION_NAME,
        VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
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

#ifdef __APPLE__
    instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

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

#if (VK_MINOR_VERSION < 3)
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

#if (VK_MINOR_VERSION > 3)
    VkPhysicalDeviceVulkan14Features featuresVulkan14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
    featuresVulkan13.pNext = &featuresVulkan14;
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

    NRI_ABORT_ON_FAILURE(nri::nriCreateDeviceFromVKDevice(deviceDesc, m_Device));
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    switch (graphicsAPI) {
        case nri::GraphicsAPI::VK:
            CreateVulkanDevice();
            break;
        case nri::GraphicsAPI::D3D12:
            CreateD3D12Device();
            break;
        default:
            CreateD3D11Device();
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
    streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER};
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

        nri::RootConstantDesc rootConstant = {1, sizeof(float), nri::StageBits::FRAGMENT_SHADER};
        nri::RootSamplerDesc rootSampler = {0, samplerDesc, nri::StageBits::FRAGMENT_SHADER};
        nri::DescriptorRangeDesc setConstantBuffer = {0, 1, nri::DescriptorType::CONSTANT_BUFFER, nri::StageBits::ALL};
        nri::DescriptorRangeDesc setTexture = {0, 1, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER};

        nri::DescriptorSetDesc descriptorSetDescs[] = {
            {0, &setConstantBuffer, 1},
            {1, &setTexture, 1},
        };

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.rootRegisterSpace = 2; // see shader
        pipelineLayoutDesc.rootConstantNum = 1;
        pipelineLayoutDesc.rootConstants = &rootConstant;
        pipelineLayoutDesc.rootSamplerNum = 1;
        pipelineLayoutDesc.rootSamplers = &rootSampler;
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

        nri::VertexAttributeDesc vertexAttributeDesc[2] = {};
        {
            vertexAttributeDesc[0].format = nri::Format::RG32_SFLOAT;
            vertexAttributeDesc[0].streamIndex = 0;
            vertexAttributeDesc[0].offset = offsetof(Vertex, position);
            vertexAttributeDesc[0].d3d = {"POSITION", 0};
            vertexAttributeDesc[0].vk.location = {0};

            vertexAttributeDesc[1].format = nri::Format::RG32_SFLOAT;
            vertexAttributeDesc[1].streamIndex = 0;
            vertexAttributeDesc[1].offset = offsetof(Vertex, uv);
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

    { // Descriptor pool
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = GetQueuedFrameNum() + 1;
        descriptorPoolDesc.constantBufferMaxNum = GetQueuedFrameNum();
        descriptorPoolDesc.textureMaxNum = 1;

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
            bufferDesc.size = constantBufferSize * GetQueuedFrameNum();
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

    { // Descriptors
        // Read-only texture
        nri::Texture2DViewDesc texture2DViewDesc = {m_Texture, nri::Texture2DViewType::SHADER_RESOURCE, texture.GetFormat()};
        NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texture2DViewDesc, m_TextureShaderResource));

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

        nri::UpdateDescriptorRangeDesc updateTexture = {m_TextureDescriptorSet, 0, 0, &m_TextureShaderResource, 1};
        NRI.UpdateDescriptorRanges(&updateTexture, 1);

        // Constant buffer
        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &queuedFrame.constantBufferDescriptorSet, 1, 0));

            nri::UpdateDescriptorRangeDesc updateDescriptorRangeDesc = {queuedFrame.constantBufferDescriptorSet, 0, 0, &queuedFrame.constantBufferView, 1};
            NRI.UpdateDescriptorRanges(&updateDescriptorRangeDesc, 1);
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
            ImGui::SliderFloat("Transparency", &m_Transparency, 0.0f, 1.0f);
            ImGui::SliderFloat("Scale", &m_Scale, 0.75f, 1.25f);
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
}

void Sample::RenderFrame(uint32_t frameIndex) {
    nri::Dim_t windowWidth = (nri::Dim_t)GetOutputResolution().x;
    nri::Dim_t windowHeight = (nri::Dim_t)GetOutputResolution().y;
    nri::Dim_t halfWidth = windowWidth / 2;
    nri::Dim_t halfHeight = windowHeight / 2;

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

    // Record
    nri::TextureBarrierDesc textureBarriers = {};
    textureBarriers.texture = swapChainTexture.texture;
    textureBarriers.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
    textureBarriers.layerNum = 1;
    textureBarriers.mipNum = 1;

    nri::CommandBuffer* commandBuffer = queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(*commandBuffer, m_DescriptorPool);
    {
        nri::BarrierDesc barrierDesc = {};
        barrierDesc.textureNum = 1;
        barrierDesc.textures = &textureBarriers;
        NRI.CmdBarrier(*commandBuffer, barrierDesc);

        nri::AttachmentDesc colorAttachmentDesc = {};
        colorAttachmentDesc.descriptor = swapChainTexture.colorAttachment;

        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colorNum = 1;
        renderingDesc.colors = &colorAttachmentDesc;

        CmdCopyImguiData(*commandBuffer, *m_Streamer);

        NRI.CmdBeginRendering(*commandBuffer, renderingDesc);
        {
            {
                helper::Annotation annotation(NRI, *commandBuffer, "Clears");

                nri::ClearAttachmentDesc clearDesc = {};
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

                NRI.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::GRAPHICS, *m_PipelineLayout);
                NRI.CmdSetPipeline(*commandBuffer, *m_Pipeline);

                nri::SetRootConstantsDesc rootConstants = {0, &m_Transparency, 4};
                NRI.CmdSetRootConstants(*commandBuffer, rootConstants);

                NRI.CmdSetIndexBuffer(*commandBuffer, *m_GeometryBuffer, 0, nri::IndexType::UINT16);

                nri::VertexBufferDesc vertexBufferDesc = {};
                vertexBufferDesc.buffer = m_GeometryBuffer;
                vertexBufferDesc.offset = m_GeometryOffset;
                vertexBufferDesc.stride = sizeof(Vertex);
                NRI.CmdSetVertexBuffers(*commandBuffer, 0, &vertexBufferDesc, 1);

                nri::SetDescriptorSetDesc descriptorSet0 = {0, queuedFrame.constantBufferDescriptorSet};
                NRI.CmdSetDescriptorSet(*commandBuffer, descriptorSet0);

                nri::SetDescriptorSetDesc descriptorSet1 = {1, m_TextureDescriptorSet};
                NRI.CmdSetDescriptorSet(*commandBuffer, descriptorSet1);

                nri::Rect scissor = {0, 0, halfWidth, windowHeight};
                NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                NRI.CmdDrawIndexed(*commandBuffer, {3, 1, 0, 0, 0});

                scissor = {(int16_t)halfWidth, (int16_t)halfHeight, halfWidth, halfHeight};
                NRI.CmdSetScissors(*commandBuffer, &scissor, 1);
                NRI.CmdDraw(*commandBuffer, {3, 1, 0, 0});
            }

            {
                helper::Annotation annotation(NRI, *commandBuffer, "UI");

                CmdDrawImgui(*commandBuffer, swapChainTexture.attachmentFormat, 1.0f, true);
            }
        }
        NRI.CmdEndRendering(*commandBuffer);

        textureBarriers.before = textureBarriers.after;
        textureBarriers.after = {nri::AccessBits::NONE, nri::Layout::PRESENT, nri::StageBits::NONE};

        NRI.CmdBarrier(*commandBuffer, barrierDesc);
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
