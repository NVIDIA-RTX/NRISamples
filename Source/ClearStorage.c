// © 2021 NVIDIA Corporation

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIResourceAllocator.h"

#if NRI_ENABLE_AGILITY_SDK_SUPPORT
#    include "NRIAgilitySDK.h"
#endif

#define NRI_ABORT_ON_FAILURE(result) \
    if (result != NriResult_SUCCESS) \
        exit(1);

int main(int argc, char** argv) {
    // Settings
    const NriDim_t width = 1024;
    const bool disableD3D12EnhancedBarriers = false;
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
                .disableD3D12EnhancedBarriers = disableD3D12EnhancedBarriers,
                .adapterDesc = &adapterDescs[adapterIndex < adapterDescsNum ? adapterIndex : adapterDescsNum - 1],
            },
            &device));
    }

    // Query interfaces
    NriCoreInterface iCore = {0};
    NriResourceAllocatorInterface iResourceAllocator = {0};
    {
        NRI_ABORT_ON_FAILURE(nriGetInterface(device, NRI_INTERFACE(NriCoreInterface), &iCore));
        NRI_ABORT_ON_FAILURE(nriGetInterface(device, NRI_INTERFACE(NriResourceAllocatorInterface), &iResourceAllocator));
    }

    // Create resources
    NriBuffer* buffer = NULL;
    NriTexture* texture = NULL;
    {
        NRI_ABORT_ON_FAILURE(iResourceAllocator.AllocateBuffer(device,
            &(NriAllocateBufferDesc){
                .memoryLocation = NriMemoryLocation_DEVICE,
                .desc = (NriBufferDesc){
                    .size = width * sizeof(float),
                    .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE,
                },
            },
            &buffer));
        iCore.SetDebugName(buffer, "Buffer");

        NRI_ABORT_ON_FAILURE(iResourceAllocator.AllocateTexture(device,
            &(NriAllocateTextureDesc){
                .memoryLocation = NriMemoryLocation_DEVICE,
                .desc = (NriTextureDesc){
                    .type = NriTextureType_TEXTURE_1D,
                    .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE,
                    .format = NriFormat_R32_SFLOAT,
                    .width = width,
                },
            },
            &texture));
        iCore.SetDebugName(texture, "Texture");
    }

    // Create storage views
    NriDescriptor* storageBuffer = NULL;
    NriDescriptor* storageTexture = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.CreateBufferView(&(NriBufferViewDesc){
                                                        .buffer = buffer,
                                                        .viewType = NriBufferViewType_SHADER_RESOURCE_STORAGE,
                                                        .format = NriFormat_R32_SFLOAT,
                                                    },
            &storageBuffer));

        NRI_ABORT_ON_FAILURE(iCore.CreateTexture1DView(&(NriTexture1DViewDesc){
                                                           .texture = texture,
                                                           .viewType = NriTexture1DViewType_SHADER_RESOURCE_STORAGE_1D,
                                                           .format = NriFormat_R32_SFLOAT,
                                                       },
            &storageTexture));
    }

    // Create descriptor pool
    NriDescriptorPool* descriptorPool = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.CreateDescriptorPool(device,
            &(NriDescriptorPoolDesc){
                .descriptorSetMaxNum = 1,
                .storageBufferMaxNum = 1,
                .storageTextureMaxNum = 1,
            },
            &descriptorPool));
    }

    // Create pipeline layout
    NriPipelineLayout* pipelineLayout = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.CreatePipelineLayout(device,
            &(NriPipelineLayoutDesc){
                .descriptorSets = &(NriDescriptorSetDesc){
                    .ranges = (NriDescriptorRangeDesc[2]){
                        {
                            .baseRegisterIndex = 0,
                            .descriptorNum = 1,
                            .descriptorType = NriDescriptorType_STORAGE_BUFFER,
                            .shaderStages = NriStageBits_COMPUTE_SHADER,
                        },
                        {
                            .baseRegisterIndex = 1,
                            .descriptorNum = 1,
                            .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
                            .shaderStages = NriStageBits_COMPUTE_SHADER,
                        },
                    },
                    .rangeNum = 2,
                },
                .descriptorSetNum = 1,
                .shaderStages = NriStageBits_COMPUTE_SHADER,
            },
            &pipelineLayout));
    }

    // Create descriptor set
    NriDescriptorSet* descriptorSet = NULL;
    NRI_ABORT_ON_FAILURE(iCore.AllocateDescriptorSets(descriptorPool, pipelineLayout, 0, &descriptorSet, 1, 0));

    // Finally put storage descriptors into this set
    iCore.UpdateDescriptorRanges(descriptorSet, 0, 2,
        (NriDescriptorRangeUpdateDesc[2]){
            {
                .descriptors = &storageBuffer,
                .descriptorNum = 1,
            },
            {
                .descriptors = &storageTexture,
                .descriptorNum = 1,
            },
        });

    // Get queue
    NriQueue* queue = NULL;
    NRI_ABORT_ON_FAILURE(iCore.GetQueue(device, NriQueueType_GRAPHICS, 0, &queue));

    // Main
    NriCommandAllocator* commandAllocator = NULL;
    NriCommandBuffer* commandBuffer = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.CreateCommandAllocator(queue, &commandAllocator));
        NRI_ABORT_ON_FAILURE(iCore.CreateCommandBuffer(commandAllocator, &commandBuffer));

        iCore.BeginCommandBuffer(commandBuffer, descriptorPool); // A descriptor pool with the resources must be bound
        {
            // Required synchronization
            // Variant 1: "SHADER_RESOURCE_STORAGE" access/layout and "CLEAR_STORAGE" + any shader stage (or "ALL")
            // Variant 2: "CLEAR_STORAGE" access/layout and "CLEAR_STORAGE" stage
            iCore.CmdBarrier(commandBuffer,
                &(NriBarrierGroupDesc){
                    .buffers = &(NriBufferBarrierDesc){
                        .before = (NriAccessStage){
                            .access = NriAccessBits_NONE,
                            .stages = NriStageBits_NONE,
                        },
                        .after = (NriAccessStage){
                            // Variant 1
                            .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
                            .stages = NriStageBits_ALL,
                        },
                        .buffer = buffer,
                    },
                    .bufferNum = 1,
                    .textures = &(NriTextureBarrierDesc){
                        .before = (NriAccessLayoutStage){
                            .access = NriAccessBits_NONE,
                            .layout = NriLayout_UNDEFINED,
                            .stages = NriStageBits_NONE,
                        },
                        .after = (NriAccessLayoutStage){
                            // Variant 2
                            .access = NriAccessBits_CLEAR_STORAGE,
                            .layout = NriLayout_SHADER_RESOURCE_STORAGE,
                            .stages = NriStageBits_CLEAR_STORAGE,
                        },
                        .texture = texture,
                    },
                    .textureNum = 1,
                });

            // A corresponding pipeline layout must be bound
            iCore.CmdSetPipelineLayout(commandBuffer, pipelineLayout);

            // A set with the resources must be bound
            iCore.CmdSetDescriptorSet(commandBuffer, 0, descriptorSet, NULL);

            // Clear buffer storage
            iCore.CmdClearStorage(commandBuffer,
                &(NriClearStorageDesc){
                    .storage = storageBuffer,
                    .value = (NriColor){
                        .f = (NriColor32f){0},
                    },
                    .setIndex = 0,
                    .rangeIndex = 0,
                    .descriptorIndex = 0,
                });

            // Clear texture storage
            iCore.CmdClearStorage(commandBuffer,
                &(NriClearStorageDesc){
                    .storage = storageTexture,
                    .value = (NriColor){
                        .f = (NriColor32f){0},
                    },
                    .setIndex = 0,
                    .rangeIndex = 1,
                    .descriptorIndex = 0,
                });
        }
        iCore.EndCommandBuffer(commandBuffer);

        // Submit
        iCore.QueueSubmit(queue,
            &(NriQueueSubmitDesc){
                .commandBufferNum = 1,
                .commandBuffers = &commandBuffer,
            });

        // Wait for idle
        iCore.QueueWaitIdle(queue);
    }

    { // Cleanup
        iCore.DestroyCommandBuffer(commandBuffer);
        iCore.DestroyCommandAllocator(commandAllocator);
        iCore.DestroyPipelineLayout(pipelineLayout);
        iCore.DestroyDescriptorPool(descriptorPool);
        iCore.DestroyDescriptor(storageBuffer);
        iCore.DestroyDescriptor(storageTexture);
        iCore.DestroyBuffer(buffer);
        iCore.DestroyTexture(texture);

        nriDestroyDevice(device);
    }

    return 0;
}
