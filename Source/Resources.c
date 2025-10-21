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
            .size = 100 * 1024 * 1024,
            .usage = NriBufferUsageBits_INDEX_BUFFER | NriBufferUsageBits_VERTEX_BUFFER,
        };

        NriMemoryDesc memoryDesc = {0};
        iCore.GetBufferMemoryDesc2(device, &bufferDesc, NriMemoryLocation_DEVICE_UPLOAD, &memoryDesc);

        NRI_ABORT_ON_FAILURE(iCore.AllocateMemory(device,
            &(NriAllocateMemoryDesc){
                .size = memoryDesc.size,
                .type = memoryDesc.type,
            },
            &placedBufferMemory));

        NRI_ABORT_ON_FAILURE(iCore.CreatePlacedBuffer(device, placedBufferMemory, 0, &bufferDesc, &placedBuffer));
    }

    { // Cleanup
        iCore.DestroyBuffer(placedBuffer);
        iCore.FreeMemory(placedBufferMemory);

        nriDestroyDevice(device);
    }

    return 0;
}
