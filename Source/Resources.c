// © 2021 NVIDIA Corporation

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"

#if NRI_ENABLE_AGILITY_SDK_SUPPORT
#    include "NRIAgilitySDK.h"
#endif

#define NRI_ABORT_ON_FAILURE(result) \
    if (result != NriResult_SUCCESS) \
        exit(1);

int main(int argc, char** argv) {
    // Settings
    NriGraphicsAPI graphicsAPI = NriGraphicsAPI_VK;
    bool debugAPI = false;
    bool debugNRI = false;
    uint32_t adapterIndex = 0;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--api=D3D11"))
            graphicsAPI = NriGraphicsAPI_D3D11;
        else if (!strcmp(argv[i], "--api=D3D12"))
            graphicsAPI = NriGraphicsAPI_D3D12;
        else if (!strcmp(argv[i], "--api=VULKAN"))
            graphicsAPI = NriGraphicsAPI_VK;
        else if (!strcmp(argv[i], "--debugAPI"))
            debugAPI = true;
        else if (!strcmp(argv[i], "--debugNRI"))
            debugNRI = true;
        else if (!strcmp(argv[i], "--adapter=1"))
            adapterIndex = 1;
    }

    // Create device
    NriDevice* device = NULL;
    {
        NriAdapterDesc adapterDescs[2] = {0};
        uint32_t adapterDescsNum = 2;
        NRI_ABORT_ON_FAILURE(nriEnumerateAdapters(adapterDescs, &adapterDescsNum));

        NRI_ABORT_ON_FAILURE(nriCreateDevice(
            &(NriDeviceCreationDesc){
                .graphicsAPI = graphicsAPI,
                .enableGraphicsAPIValidation = debugAPI,
                .enableNRIValidation = debugNRI,
                .adapterDesc = &adapterDescs[adapterIndex < adapterDescsNum ? adapterIndex : adapterDescsNum - 1],
            },
            &device));
    }

    // Query interfaces
    NriCoreInterface iCore = {0};
    {
        NRI_ABORT_ON_FAILURE(nriGetInterface(device, NRI_INTERFACE(NriCoreInterface), &iCore));
    }

    const NriDeviceDesc* deviceDesc = iCore.GetDeviceDesc(device);

    // Create a placed buffer
    NriMemory* placedBufferMemory = NULL;
    NriBuffer* placedBuffer = NULL;
    if (deviceDesc->features.getMemoryDesc2) {
        NriBufferDesc bufferDesc = {
            .size = 32 * 1024 * 1024,
            .usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE,
            .structureStride = 4,
        };

        NriMemoryDesc memoryDesc = {0};
        iCore.GetBufferMemoryDesc2(device, &bufferDesc, NriMemoryLocation_DEVICE, &memoryDesc);

        NRI_ABORT_ON_FAILURE(iCore.AllocateMemory(device,
            &(NriAllocateMemoryDesc){
                .size = memoryDesc.size,
                .type = memoryDesc.type,
            },
            &placedBufferMemory));

        NRI_ABORT_ON_FAILURE(iCore.CreatePlacedBuffer(device, placedBufferMemory, 0, &bufferDesc, &placedBuffer));
    }

    { // Test views
        NriDescriptor* bufferView_Typed = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE,
                .offset = 0,
                .size = 1024,
                .format = NriFormat_RGBA32_SFLOAT,
            },
            &bufferView_Typed);

        NriDescriptor* bufferView_TypedStorage = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE_STORAGE,
                .offset = 0,
                .size = 1024,
                .format = NriFormat_RG32_UINT,
            },
            &bufferView_TypedStorage);

        NriDescriptor* bufferView_Raw = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE_RAW,
                .offset = 0,
                .size = 1024,
            },
            &bufferView_Raw);

        NriDescriptor* bufferView_RawStorage = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE_STORAGE_RAW,
                .offset = 0,
                .size = 1024,
            },
            &bufferView_RawStorage);

        NriDescriptor* bufferView_Structured = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE_STRUCTURED,
                .offset = 0,
                .size = 1024,
                .structureStride = 16,
            },
            &bufferView_Structured);

        NriDescriptor* bufferView_StructuredStorage = NULL;
        iCore.CreateBufferView(
            &(NriBufferViewDesc){
                .buffer = placedBuffer,
                .viewType = NriBufferViewType_SHADER_RESOURCE_STORAGE_STRUCTURED,
                .offset = 0,
                .size = 1024,
                .structureStride = 32,
            },
            &bufferView_StructuredStorage);

        iCore.DestroyDescriptor(bufferView_Typed);
        iCore.DestroyDescriptor(bufferView_TypedStorage);
        iCore.DestroyDescriptor(bufferView_Raw);
        iCore.DestroyDescriptor(bufferView_RawStorage);
        iCore.DestroyDescriptor(bufferView_Structured);
        iCore.DestroyDescriptor(bufferView_StructuredStorage);
    }

    { // Cleanup
        iCore.DestroyBuffer(placedBuffer);
        iCore.FreeMemory(placedBufferMemory);

        nriDestroyDevice(device);
    }

    return 0;
}
