// © 2026 NVIDIA Corporation

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"

#include "ml.h"
#include "ml.hlsli"

#include "Utils.h"

#if NRI_ENABLE_AGILITY_SDK_SUPPORT
#    include "NRIAgilitySDK.h"
#endif

namespace test {

struct Settings {
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
    uint32_t adapterIndex = 0;
    bool debugAPI = false;
    bool debugNRI = false;
};

inline Settings ParseSettings(int argc, char** argv) {
    Settings settings;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--api=D3D11"))
            settings.graphicsAPI = nri::GraphicsAPI::D3D11;
        else if (!strcmp(argv[i], "--api=D3D12"))
            settings.graphicsAPI = nri::GraphicsAPI::D3D12;
        else if (!strcmp(argv[i], "--api=VULKAN"))
            settings.graphicsAPI = nri::GraphicsAPI::VK;
        else if (!strcmp(argv[i], "--api=WGPU"))
            settings.graphicsAPI = nri::GraphicsAPI::WGPU;
        else if (!strcmp(argv[i], "--debugAPI"))
            settings.debugAPI = true;
        else if (!strcmp(argv[i], "--debugNRI"))
            settings.debugNRI = true;
        else if (!strncmp(argv[i], "--adapter=", 10))
            settings.adapterIndex = (uint32_t)strtoul(argv[i] + 10, nullptr, 10);
    }

    return settings;
}

inline bool Check(nri::Result result, const char* expression, const char* file, uint32_t line) {
    if (result == nri::Result::SUCCESS)
        return true;

    printf("FAIL  %s returned %d (%s:%u)\n", expression, int(result), file, line);

    return false;
}

inline bool Check(bool result, const char* expression, const char* file, uint32_t line) {
    if (result)
        return true;

    printf("FAIL  %s (%s:%u)\n", expression, file, line);

    return false;
}

#define TEST_CHECK(expression) \
    do { \
        if (!test::Check((expression), #expression, __FILE__, __LINE__)) \
            return false; \
    } while (0)

class Context {
public:
    ~Context() {
        if (!device)
            return;

        core.DeviceWaitIdle(device);

        for (nri::Descriptor* descriptor : descriptors)
            core.DestroyDescriptor(descriptor);
        for (nri::Pipeline* pipeline : pipelines)
            core.DestroyPipeline(pipeline);
        for (nri::PipelineLayout* pipelineLayout : pipelineLayouts)
            core.DestroyPipelineLayout(pipelineLayout);
        for (nri::DescriptorPool* descriptorPool : descriptorPools)
            core.DestroyDescriptorPool(descriptorPool);
        for (nri::QueryPool* queryPool : queryPools)
            core.DestroyQueryPool(queryPool);
        for (nri::CommandBuffer* commandBuffer : commandBuffers)
            core.DestroyCommandBuffer(commandBuffer);
        for (nri::CommandAllocator* commandAllocator : commandAllocators)
            core.DestroyCommandAllocator(commandAllocator);
        for (nri::Buffer* buffer : buffers)
            core.DestroyBuffer(buffer);
        for (nri::Texture* texture : textures)
            core.DestroyTexture(texture);
        for (nri::Fence* fence : fences)
            core.DestroyFence(fence);
        for (nri::Memory* memory : memories)
            core.FreeMemory(memory);

        nri::nriDestroyDevice(device);
    }

    bool Initialize(const Settings& settings, const nri::QueueFamilyDesc* queueFamilies = nullptr, uint32_t queueFamilyNum = 0) {
        uint32_t adapterNum = 0;
        TEST_CHECK(nri::nriEnumerateAdapters(nullptr, adapterNum));
        if (!adapterNum) {
            printf("FAIL  No adapters found\n");

            return false;
        }

        std::vector<nri::AdapterDesc> adapters(adapterNum);
        TEST_CHECK(nri::nriEnumerateAdapters(adapters.data(), adapterNum));

        const uint32_t adapterIndex = std::min(settings.adapterIndex, adapterNum - 1);
        adapterDesc = adapters[adapterIndex];
        if (!(adapterDesc.supportedGraphicsAPIs & settings.graphicsAPI)) {
            printf("SKIP  API is unsupported by the selected adapter\n");

            skipped = true;
            return true;
        }

        for (uint32_t i = 0; i < queueFamilyNum; i++) {
            const nri::QueueFamilyDesc& queueFamily = queueFamilies[i];
            if (adapterDesc.queueNum[(uint32_t)queueFamily.queueType] < queueFamily.queueNum) {
                printf("SKIP  Requested queue family is unsupported\n");

                skipped = true;
                return true;
            }
        }

        nri::DeviceCreationDesc deviceCreationDesc = {};
        deviceCreationDesc.graphicsAPI = settings.graphicsAPI;
        deviceCreationDesc.adapterDesc = &adapterDesc;
        deviceCreationDesc.queueFamilies = queueFamilies;
        deviceCreationDesc.queueFamilyNum = queueFamilyNum;
        deviceCreationDesc.enableGraphicsAPIValidation = settings.debugAPI;
        deviceCreationDesc.enableNRIValidation = settings.debugNRI;
        deviceCreationDesc.vkBindingOffsets = {0, 128, 32, 64};

        TEST_CHECK(nri::nriCreateDevice(deviceCreationDesc, device));
        TEST_CHECK(nri::nriGetInterface(*device, NRI_INTERFACE(nri::CoreInterface), &core));
        TEST_CHECK(nri::nriGetInterface(*device, NRI_INTERFACE(nri::HelperInterface), &helper));

        deviceDesc = &core.GetDeviceDesc(*device);
        printf("%s, %s\n", nri::nriGetGraphicsAPIString(deviceDesc->graphicsAPI), deviceDesc->adapterDesc.name);

        return true;
    }

    bool CreateCommandObjects(nri::Queue& queue, nri::CommandAllocator*& commandAllocator, nri::CommandBuffer*& commandBuffer) {
        TEST_CHECK(core.CreateCommandAllocator(queue, commandAllocator));
        commandAllocators.push_back(commandAllocator);
        TEST_CHECK(core.CreateCommandBuffer(*commandAllocator, commandBuffer));
        commandBuffers.push_back(commandBuffer);

        return true;
    }

    bool SubmitAndWait(nri::Queue& queue, nri::CommandBuffer& commandBuffer) {
        TEST_CHECK(core.EndCommandBuffer(commandBuffer));

        nri::CommandBuffer* commandBufferPtr = &commandBuffer;
        nri::QueueSubmitDesc queueSubmitDesc = {};
        queueSubmitDesc.commandBuffers = &commandBufferPtr;
        queueSubmitDesc.commandBufferNum = 1;
        TEST_CHECK(core.QueueSubmit(queue, queueSubmitDesc));
        TEST_CHECK(core.QueueWaitIdle(&queue));

        return true;
    }

    bool CreateBuffer(const nri::BufferDesc& desc, nri::MemoryLocation memoryLocation, nri::Buffer*& buffer) {
        TEST_CHECK(core.CreateCommittedBuffer(*device, memoryLocation, 0.0f, desc, buffer));
        buffers.push_back(buffer);

        return true;
    }

    bool CreateTexture(const nri::TextureDesc& desc, nri::MemoryLocation memoryLocation, nri::Texture*& texture) {
        TEST_CHECK(core.CreateCommittedTexture(*device, memoryLocation, 0.0f, desc, texture));
        textures.push_back(texture);

        return true;
    }

    template <typename T>
    void Track(T* object);

    nri::Device* device = nullptr;
    nri::CoreInterface core = {};
    nri::HelperInterface helper = {};
    const nri::DeviceDesc* deviceDesc = nullptr;
    nri::AdapterDesc adapterDesc = {};
    bool skipped = false;
    utils::ShaderCodeStorage shaderStorage;

    std::vector<nri::CommandAllocator*> commandAllocators;
    std::vector<nri::CommandBuffer*> commandBuffers;
    std::vector<nri::Fence*> fences;
    std::vector<nri::DescriptorPool*> descriptorPools;
    std::vector<nri::PipelineLayout*> pipelineLayouts;
    std::vector<nri::Pipeline*> pipelines;
    std::vector<nri::QueryPool*> queryPools;
    std::vector<nri::Buffer*> buffers;
    std::vector<nri::Texture*> textures;
    std::vector<nri::Descriptor*> descriptors;
    std::vector<nri::Memory*> memories;
};

template <>
inline void Context::Track(nri::Fence* object) {
    fences.push_back(object);
}

template <>
inline void Context::Track(nri::DescriptorPool* object) {
    descriptorPools.push_back(object);
}

template <>
inline void Context::Track(nri::PipelineLayout* object) {
    pipelineLayouts.push_back(object);
}

template <>
inline void Context::Track(nri::Pipeline* object) {
    pipelines.push_back(object);
}

template <>
inline void Context::Track(nri::QueryPool* object) {
    queryPools.push_back(object);
}

template <>
inline void Context::Track(nri::Buffer* object) {
    buffers.push_back(object);
}

template <>
inline void Context::Track(nri::Texture* object) {
    textures.push_back(object);
}

template <>
inline void Context::Track(nri::Descriptor* object) {
    descriptors.push_back(object);
}

template <>
inline void Context::Track(nri::Memory* object) {
    memories.push_back(object);
}

inline bool VerifyBytes(nri::CoreInterface& core, nri::Buffer& buffer, const void* expected, uint64_t size) {
    const void* data = core.MapBuffer(buffer, 0, size);
    const bool passed = data && memcmp(data, expected, (size_t)size) == 0;
    core.UnmapBuffer(buffer);

    return passed;
}

inline bool Report(const char* name, bool passed) {
    printf("%s  %s\n", passed ? "PASS" : "FAIL", name);

    return passed;
}

} // namespace test
