// © 2021 NVIDIA Corporation

#include <stdio.h>
#include <malloc.h>

#define NRI_FORCE_C

#include "NRI.h"
#include "Extensions/NRIDeviceCreation.h"

static const char* vendors[] =
{
    "unknown",
    "NVIDIA",
    "AMD",
    "INTEL"
};

#if _WIN32
    #define alloca _alloca
#else
    #include <alloca.h>
#endif

bool EnumerateAdapters()
{
    // Query device adapterDescs number
    uint32_t adaptersNum = 0;
    NriResult result = nriEnumerateAdapters(NULL, &adaptersNum);
    if (result != NriResult_SUCCESS)
        return false;

    printf("NriGetPhysicalDevices: %u adapterDescs reported\n", adaptersNum);
    if (!adaptersNum)
        return true;

    // Query device adapterDescs
    size_t bytes = adaptersNum * sizeof(NriAdapterDesc);
    NriAdapterDesc* adapterDescs = (NriAdapterDesc*)alloca(bytes);
    result = nriEnumerateAdapters(adapterDescs, &adaptersNum);
    if (result != NriResult_SUCCESS)
        return false;

    // Print information
    for (uint32_t i = 0; i < adaptersNum; i++)
    {
        const NriAdapterDesc* p = adapterDescs + i;

        printf("\nGroup #%u\n", i + 1);
        printf("\tDescription: %s\n", p->description);
        printf("\tLUID: 0x%016llX\n", p->luid);
        printf("\tVideo memory (Mb): %llu\n", p->videoMemorySize >> 20);
        printf("\tSystem memory (Mb): %llu\n", p->systemMemorySize >> 20);
        printf("\tID: 0x%08X\n", p->deviceId);
        printf("\tVendor: %s\n", vendors[p->vendor]);
    }

    return true;
}

int main()
{
    return EnumerateAdapters() ? 0 : 1;
}
