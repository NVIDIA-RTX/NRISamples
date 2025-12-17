// © 2021 NVIDIA Corporation

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if _WIN32
#    include <malloc.h>
#    define alloca _alloca
#else
#    include <alloca.h>
#endif

#include "NRI.h"

#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRIHelper.h"

static const char* vendors[] = {
    "unknown",
    "NVIDIA",
    "AMD",
    "INTEL",
};

static const char* architectures[] = {
    "unknown",
    "INTEGRATED",
    "DESCRETE",
};

#define NRI_ABORT_ON_FAILURE(result) \
    if (result != NriResult_SUCCESS) \
        exit(1);

void SilencePlease(NriMessage messageType, const char* file, uint32_t line, const char* message, void* userArg) {
    (void)messageType;
    (void)file;
    (void)line;
    (void)message;
    (void)userArg;
}

int main(int argc, char** argv) {
    // Settings
    NriGraphicsAPI graphicsAPI = NriGraphicsAPI_D3D11;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--api=D3D12"))
            graphicsAPI = NriGraphicsAPI_D3D12;
        else if (!strcmp(argv[i], "--api=VULKAN"))
            graphicsAPI = NriGraphicsAPI_VK;
    }

    // Query adapters number
    uint32_t adaptersNum = 0;
    NRI_ABORT_ON_FAILURE(nriEnumerateAdapters(
        NULL, &adaptersNum));

    // Query adapters
    size_t bytes = adaptersNum * sizeof(NriAdapterDesc);
    NriAdapterDesc* adapterDescs = (NriAdapterDesc*)alloca(bytes);
    NRI_ABORT_ON_FAILURE(nriEnumerateAdapters(
        adapterDescs, &adaptersNum));

    // Print adapters info
    printf("nriEnumerateAdapters: %u adapters reported\n", adaptersNum);

    for (uint32_t i = 0; i < adaptersNum; i++) {
        const NriAdapterDesc* adapterDesc = adapterDescs + i;

        printf("\nAdapter #%u\n", i + 1);
        printf("\tName                 : %s\n", adapterDesc->name);
        printf("\tVendor               : %s\n", vendors[adapterDesc->vendor]);
        printf("\tArchitecture         : %s\n", architectures[adapterDesc->architecture]);
        printf("\tVideo memory         : %" PRIu64 " Mb\n", adapterDesc->videoMemorySize >> 20);
        printf("\tShared system memory : %" PRIu64 " Mb\n", adapterDesc->sharedSystemMemorySize >> 20);
        printf("\tQueues               : {%u, %u, %u}\n", adapterDesc->queueNum[0], adapterDesc->queueNum[1], adapterDesc->queueNum[2]);
        printf("\tID                   : 0x%08X\n", adapterDesc->deviceId);
        printf("\tUID.low              : 0x%016" PRIX64 "\n", adapterDesc->uid.low);
        printf("\tUID.high             : 0x%016" PRIX64 "\n", adapterDesc->uid.high);

        // Print formats info
        NriDevice* device = NULL;
        NRI_ABORT_ON_FAILURE(nriCreateDevice(
            &(NriDeviceCreationDesc){
                .graphicsAPI = graphicsAPI,
                .adapterDesc = adapterDesc,
                .callbackInterface.MessageCallback = SilencePlease,
            },
            &device));

        NriCoreInterface iCore = {0};
        NRI_ABORT_ON_FAILURE(nriGetInterface(
            device, NRI_INTERFACE(NriCoreInterface), &iCore));

        printf("\n");
        printf("%54.54s\n", "STORAGE_WRITE_WITHOUT_FORMAT");
        printf("%54.54s\n", "STORAGE_READ_WITHOUT_FORMAT |");
        printf("%54.54s\n", "VERTEX_BUFFER | |");
        printf("%54.54s\n", "STORAGE_BUFFER_ATOMICS | | |");
        printf("%54.54s\n", "STORAGE_BUFFER | | | |");
        printf("%54.54s\n", "BUFFER | | | | |");
        printf("%54.54s\n", "MULTISAMPLE_RESOLVE | | | | | |");
        printf("%54.54s\n", "MULTISAMPLE_8X | | | | | | |");
        printf("%54.54s\n", "MULTISAMPLE_4X | | | | | | | |");
        printf("%54.54s\n", "MULTISAMPLE_2X | | | | | | | | |");
        printf("%54.54s\n", "BLEND | | | | | | | | | |");
        printf("%54.54s\n", "DEPTH_STENCIL_ATTACHMENT | | | | | | | | | | |");
        printf("%54.54s\n", "COLOR_ATTACHMENT | | | | | | | | | | | |");
        printf("%54.54s\n", "STORAGE_TEXTURE_ATOMICS | | | | | | | | | | | | |");
        printf("%54.54s\n", "STORAGE_TEXTURE | | | | | | | | | | | | | |");
        printf("%54.54s\n", "TEXTURE | | | | | | | | | | | | | | |");
        printf("%54.54s\n", "| | | | | | | | | | | | | | | |");

        for (uint32_t f = 0; f < NriFormat_MAX_NUM; f++) {
            const NriFormatProps* formatProps = nriGetFormatProps((NriFormat)f);
            NriFormatSupportBits formatSupportBits = iCore.GetFormatSupport(device, (NriFormat)f);

            printf("%20.20s   ", formatProps->name);

            for (uint16_t bit = 0; bit < 16; bit++) {
                if (formatSupportBits & (1 << bit))
                    printf("+ ");
                else
                    printf(". ");
            }

            printf("\n");
        }

        nriDestroyDevice(device);
    }

    return 0;
}
