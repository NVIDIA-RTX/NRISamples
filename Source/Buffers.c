// © 2021 NVIDIA Corporation

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIResourceAllocator.h"

#define NRI_ABORT_ON_FAILURE(result) \
    if (result != NriResult_SUCCESS) \
        exit(1);

int main(int argc, char** argv) {
    // Settings
    bool useSelfCopies = true; // works only with barriers in-between
    const bool useBarriersBetweenSelfCopies = true;
    const uint32_t bufferZeroSize = 1024;
    const uint32_t bufferOneSize = 64 * 1024;

    NriGraphicsAPI graphicsAPI = NriGraphicsAPI_D3D11;
    bool debugAPI = false;
    bool debugNRI = false;
    uint32_t adapterIndex = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--api=D3D12"))
            graphicsAPI = NriGraphicsAPI_D3D12;
        else if (!strcmp(argv[i], "--api=VULKAN"))
            graphicsAPI = NriGraphicsAPI_VK;
        else if (!strcmp(argv[i], "--debugAPI"))
            debugAPI = true;
        else if (!strcmp(argv[i], "--debugNRI"))
            debugNRI = true;
        else if(!strcmp(argv[i], "--adapter=1"))
            adapterIndex = 1;
    }

    // Create device
    NriDevice* device = NULL;
    {
        NriAdapterDesc adapterDescs[2] = {0};
        uint32_t adapterDescsNum = 2;
        NRI_ABORT_ON_FAILURE(nriEnumerateAdapters(
            adapterDescs, &adapterDescsNum));

        NRI_ABORT_ON_FAILURE(nriCreateDevice(
            &(NriDeviceCreationDesc){
                .graphicsAPI = graphicsAPI,
                .enableGraphicsAPIValidation = debugAPI,
                .enableNRIValidation = debugNRI,
                .adapterDesc = &adapterDescs[min(adapterIndex, adapterDescsNum - 1)],
            },
            &device));
    }

    // Query interfaces
    NriCoreInterface iCore = {0};
    NriHelperInterface iHelper = {0};
    NriResourceAllocatorInterface iResourceAllocator = {0};
    {
        NRI_ABORT_ON_FAILURE(nriGetInterface(
            device, NRI_INTERFACE(NriCoreInterface), &iCore));

        NRI_ABORT_ON_FAILURE(nriGetInterface(
            device, NRI_INTERFACE(NriHelperInterface), &iHelper));

        NRI_ABORT_ON_FAILURE(nriGetInterface(
            device, NRI_INTERFACE(NriResourceAllocatorInterface), &iResourceAllocator));

        const NriDeviceDesc* deviceDesc = iCore.GetDeviceDesc(device);
        if (deviceDesc->graphicsAPI == NriGraphicsAPI_D3D11 || !deviceDesc->features.enchancedBarrier)
            useSelfCopies = false; // Vulkan or D3D12 with AgilitySDK required
    }

    // Create buffers
    NriBuffer* bufferZero;
    NriBuffer* bufferOne;
    NriBuffer* bufferReadback;
    {
        NRI_ABORT_ON_FAILURE(iResourceAllocator.AllocateBuffer(
            device, &(NriAllocateBufferDesc){
                        .memoryLocation = NriMemoryLocation_DEVICE,
                        .desc = (NriBufferDesc){
                            .size = bufferZeroSize,
                            .usage = NriBufferUsageBits_NONE,
                        },
                    },
            &bufferZero));

        NRI_ABORT_ON_FAILURE(iResourceAllocator.AllocateBuffer(
            device, &(NriAllocateBufferDesc){
                        .memoryLocation = NriMemoryLocation_DEVICE,
                        .desc = (NriBufferDesc){
                            .size = bufferOneSize,
                            .usage = NriBufferUsageBits_NONE,
                        },
                    },
            &bufferOne));

        NRI_ABORT_ON_FAILURE(iResourceAllocator.AllocateBuffer(
            device, &(NriAllocateBufferDesc){
                        .memoryLocation = NriMemoryLocation_HOST_READBACK,
                        .desc = (NriBufferDesc){
                            .size = bufferOneSize,
                            .usage = NriBufferUsageBits_NONE,
                        },
                    },
            &bufferReadback));
    }

    // Fill buffers
    NriQueue* queue = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.GetQueue(device, NriQueueType_GRAPHICS, 0, &queue));

        uint8_t* zeroData = (uint8_t*)malloc(bufferZeroSize);
        memset(zeroData, 0, bufferZeroSize);

        uint8_t* garbageData = (uint8_t*)malloc(bufferOneSize);
        memset(garbageData, 1, bufferOneSize);

        NriBufferUploadDesc bufferUploads[2] = {
            (NriBufferUploadDesc){
                // fill "bufferZero" with "0"
                .data = zeroData,
                .buffer = bufferZero,
                .after = (NriAccessStage){
                    .access = NriAccessBits_COPY_SOURCE,
                },
            },
            (NriBufferUploadDesc){
                // fill "bufferOne" with "1"
                .data = garbageData,
                .buffer = bufferOne,
                .after = (NriAccessStage){
                    .access = NriAccessBits_COPY_DESTINATION,
                },
            },
        };

        iHelper.UploadData(queue, NULL, 0, bufferUploads, 2);

        free(zeroData);
        free(garbageData);
    }

    // Main
    NriCommandAllocator* commandAllocator = NULL;
    NriCommandBuffer* commandBuffer = NULL;
    {
        NRI_ABORT_ON_FAILURE(iCore.CreateCommandAllocator(queue, &commandAllocator));
        NRI_ABORT_ON_FAILURE(iCore.CreateCommandBuffer(commandAllocator, &commandBuffer));

        iCore.BeginCommandBuffer(commandBuffer, NULL);
        {
            // Clear "bufferOne" using "bufferZero"
            uint32_t size = bufferOneSize;
            uint32_t offset = 0;

            if (useSelfCopies) {
                // Self copies
                uint32_t blockSize = size < bufferZeroSize ? size : bufferZeroSize;
                uint32_t offsetOrig = offset;

                iCore.CmdCopyBuffer(commandBuffer, bufferOne, offset, bufferZero, 0, blockSize);

                offset += blockSize;
                size -= blockSize;

                while (size >= blockSize) {
                    if (useBarriersBetweenSelfCopies) {
                        iCore.CmdBarrier(commandBuffer,
                            &(NriBarrierGroupDesc){
                                .bufferNum = 1,
                                .buffers = &(NriBufferBarrierDesc){
                                    .buffer = bufferOne,
                                    .before = (NriAccessStage){
                                        .access = NriAccessBits_COPY_DESTINATION | NriAccessBits_COPY_SOURCE,
                                        .stages = NriStageBits_COPY,
                                    },
                                    .after = (NriAccessStage){
                                        .access = NriAccessBits_COPY_DESTINATION | NriAccessBits_COPY_SOURCE,
                                        .stages = NriStageBits_COPY,
                                    },
                                },
                            });
                    }

                    iCore.CmdCopyBuffer(commandBuffer, bufferOne, offset, bufferOne, offsetOrig, blockSize);

                    offset += blockSize;
                    size -= blockSize;

                    blockSize <<= 1;
                }

                if (size)
                    iCore.CmdCopyBuffer(commandBuffer, bufferOne, offset, bufferOne, offsetOrig, size);
            } else {
                // No self copies
                while (size) {
                    uint32_t blockSize = size < bufferZeroSize ? size : bufferZeroSize;

                    iCore.CmdCopyBuffer(commandBuffer, bufferOne, offset, bufferZero, 0, blockSize);

                    offset += blockSize;
                    size -= blockSize;
                }
            }

            // Copy to readback
            iCore.CmdBarrier(commandBuffer,
                &(NriBarrierGroupDesc){
                    .bufferNum = 1,
                    .buffers = &(NriBufferBarrierDesc){
                        .buffer = bufferOne,
                        .before = (NriAccessStage){
                            .access = NriAccessBits_COPY_DESTINATION,
                        },
                        .after = (NriAccessStage){
                            .access = NriAccessBits_COPY_SOURCE,
                        },
                    },
                });

            iCore.CmdCopyBuffer(commandBuffer, bufferReadback, 0, bufferOne, 0, NRI_WHOLE_SIZE);
        }
        iCore.EndCommandBuffer(commandBuffer);

        // Submit
        iCore.QueueSubmit(queue,
            &(NriQueueSubmitDesc){
                .commandBufferNum = 1,
                .commandBuffers = &commandBuffer,
            });

        // Wait for idle
        iHelper.WaitForIdle(queue);
    }

    { // Validate result
        const uint8_t* result = (uint8_t*)iCore.MapBuffer(bufferReadback, 0, NRI_WHOLE_SIZE);
        {
            uint32_t sum = 0;
            for (uint32_t i = 0; i < bufferOneSize; i++)
                sum += result[i];

            printf("Result = %u (0 expected)\n", sum);
        }
        iCore.UnmapBuffer(bufferReadback);
    }

    { // Cleanup
        iCore.DestroyCommandBuffer(commandBuffer);
        iCore.DestroyCommandAllocator(commandAllocator);

        iCore.DestroyBuffer(bufferZero);
        iCore.DestroyBuffer(bufferOne);
        iCore.DestroyBuffer(bufferReadback);

        nriDestroyDevice(device);
    }

    return 0;
}
