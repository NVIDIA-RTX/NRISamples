// © 2026 NVIDIA Corporation

#include "TestShared.h"

#include "Extensions/NRIRayTracing.h"

#include <algorithm>
#include <vector>

namespace {

struct AccelerationStructureGuard {
    ~AccelerationStructureGuard() {
        for (nri::AccelerationStructure* accelerationStructure : accelerationStructures)
            interface.DestroyAccelerationStructure(accelerationStructure);
    }

    nri::RayTracingInterface interface = {};
    std::vector<nri::AccelerationStructure*> accelerationStructures;
};

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    if (!context.deviceDesc->tiers.rayTracing) {
        printf("SKIP  Ray tracing is unsupported\n");

        return true;
    }

    AccelerationStructureGuard guard;
    TEST_CHECK(nri::nriGetInterface(*context.device, NRI_INTERFACE(nri::RayTracingInterface), &guard.interface));

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    nri::BottomLevelAabb aabb = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    nri::BufferDesc aabbBufferDesc = {};
    aabbBufferDesc.size = sizeof(aabb);
    aabbBufferDesc.usage = nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_INPUT;
    nri::Buffer* aabbBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(aabbBufferDesc, nri::MemoryLocation::HOST_UPLOAD, aabbBuffer));
    void* mapped = context.core.MapBuffer(*aabbBuffer, 0, sizeof(aabb));
    memcpy(mapped, &aabb, sizeof(aabb));
    context.core.UnmapBuffer(*aabbBuffer);

    nri::BottomLevelGeometryDesc geometry = {};
    geometry.type = nri::BottomLevelGeometryType::AABBS;
    geometry.flags = nri::BottomLevelGeometryBits::NO_DUPLICATE_ANY_HIT_INVOCATION;
    geometry.aabbs.buffer = aabbBuffer;
    geometry.aabbs.num = 1;
    geometry.aabbs.stride = sizeof(aabb);

    nri::AccelerationStructureDesc accelerationStructureDesc = {};
    accelerationStructureDesc.geometries = &geometry;
    accelerationStructureDesc.geometryOrInstanceNum = 1;
    accelerationStructureDesc.flags = nri::AccelerationStructureBits::ALLOW_UPDATE | nri::AccelerationStructureBits::ALLOW_COMPACTION | nri::AccelerationStructureBits::PREFER_FAST_TRACE;
    accelerationStructureDesc.type = nri::AccelerationStructureType::BOTTOM_LEVEL;

    if (context.deviceDesc->features.getMemoryDesc2) {
        nri::MemoryDesc memoryDesc = {};
        guard.interface.GetAccelerationStructureMemoryDesc2(*context.device, accelerationStructureDesc, nri::MemoryLocation::DEVICE, memoryDesc);
        if (!memoryDesc.size) {
            printf("FAIL  GetAccelerationStructureMemoryDesc2 returned an empty description\n");

            return false;
        }
    }

    nri::AccelerationStructure* source = nullptr;
    TEST_CHECK(guard.interface.CreateCommittedAccelerationStructure(*context.device, nri::MemoryLocation::DEVICE, 0.0f, accelerationStructureDesc, source));
    guard.accelerationStructures.push_back(source);

    if (!guard.interface.GetAccelerationStructureBuffer(*source)) {
        printf("FAIL  GetAccelerationStructureBuffer returned null\n");

        return false;
    }

    const uint64_t buildScratchSize = guard.interface.GetAccelerationStructureBuildScratchBufferSize(*source);
    const uint64_t updateScratchSize = guard.interface.GetAccelerationStructureUpdateScratchBufferSize(*source);
    nri::BufferDesc scratchBufferDesc = {};
    scratchBufferDesc.size = std::max(buildScratchSize, updateScratchSize);
    scratchBufferDesc.usage = nri::BufferUsageBits::SCRATCH;
    nri::Buffer* scratchBuffer = nullptr;
    TEST_CHECK(context.CreateBuffer(scratchBufferDesc, nri::MemoryLocation::DEVICE, scratchBuffer));

    const nri::QueryPoolDesc queryPoolDesc = {nri::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE, 1};
    nri::QueryPool* queryPool = nullptr;
    TEST_CHECK(context.core.CreateQueryPool(*context.device, queryPoolDesc, queryPool));
    context.Track(queryPool);

    const uint32_t querySize = context.core.GetQuerySize(*queryPool);
    nri::BufferDesc readbackDesc = {};
    readbackDesc.size = querySize;
    nri::Buffer* readback = nullptr;
    TEST_CHECK(context.CreateBuffer(readbackDesc, nri::MemoryLocation::HOST_READBACK, readback));

    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));
    context.core.CmdResetQueries(*commandBuffer, *queryPool, 0, 1);

    nri::BuildBottomLevelAccelerationStructureDesc buildDesc = {};
    buildDesc.dst = source;
    buildDesc.geometries = &geometry;
    buildDesc.geometryNum = 1;
    buildDesc.scratchBuffer = scratchBuffer;
    guard.interface.CmdBuildBottomLevelAccelerationStructures(*commandBuffer, &buildDesc, 1);

    nri::GlobalBarrierDesc globalBarrier = {};
    globalBarrier.before = {nri::AccessBits::ACCELERATION_STRUCTURE_WRITE, nri::StageBits::ACCELERATION_STRUCTURE};
    globalBarrier.after = {nri::AccessBits::ACCELERATION_STRUCTURE_READ, nri::StageBits::ACCELERATION_STRUCTURE};
    nri::BarrierDesc barrierDesc = {};
    barrierDesc.globals = &globalBarrier;
    barrierDesc.globalNum = 1;
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    nri::AccelerationStructure* sourceArray[] = {source};
    guard.interface.CmdWriteAccelerationStructuresSizes(*commandBuffer, sourceArray, 1, *queryPool, 0);
    context.core.CmdCopyQueries(*commandBuffer, *queryPool, 0, 1, *readback, 0);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    const uint64_t* compactedSizePtr = (const uint64_t*)context.core.MapBuffer(*readback, 0, querySize);
    const uint64_t compactedSize = compactedSizePtr ? compactedSizePtr[0] : 0;
    context.core.UnmapBuffer(*readback);
    if (!compactedSize) {
        printf("FAIL  Compacted acceleration-structure size is zero\n");

        return false;
    }

    nri::AccelerationStructure* clone = nullptr;
    TEST_CHECK(guard.interface.CreateCommittedAccelerationStructure(*context.device, nri::MemoryLocation::DEVICE, 0.0f, accelerationStructureDesc, clone));
    guard.accelerationStructures.push_back(clone);

    nri::AccelerationStructureDesc compactDesc = accelerationStructureDesc;
    compactDesc.optimizedSize = compactedSize;
    compactDesc.flags = nri::AccelerationStructureBits::PREFER_FAST_TRACE;
    nri::AccelerationStructure* compact = nullptr;
    if (context.deviceDesc->features.getMemoryDesc2) {
        nri::MemoryDesc compactMemoryDesc = {};
        guard.interface.GetAccelerationStructureMemoryDesc2(*context.device, compactDesc, nri::MemoryLocation::DEVICE, compactMemoryDesc);

        nri::AllocateMemoryDesc allocateMemoryDesc = {};
        allocateMemoryDesc.size = compactMemoryDesc.size;
        allocateMemoryDesc.type = compactMemoryDesc.type;
        nri::Memory* compactMemory = nullptr;
        TEST_CHECK(context.core.AllocateMemory(*context.device, allocateMemoryDesc, compactMemory));
        context.Track(compactMemory);
        TEST_CHECK(guard.interface.CreatePlacedAccelerationStructure(*context.device, compactMemory, 0, compactDesc, compact));
    } else
        TEST_CHECK(guard.interface.CreateCommittedAccelerationStructure(*context.device, nri::MemoryLocation::DEVICE, 0.0f, compactDesc, compact));
    guard.accelerationStructures.push_back(compact);

    context.core.ResetCommandAllocator(*commandAllocator);
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, nullptr));
    guard.interface.CmdCopyAccelerationStructure(*commandBuffer, *clone, *source, nri::CopyMode::CLONE);
    guard.interface.CmdCopyAccelerationStructure(*commandBuffer, *compact, *source, nri::CopyMode::COMPACT);

    globalBarrier.before = {nri::AccessBits::ACCELERATION_STRUCTURE_READ, nri::StageBits::ACCELERATION_STRUCTURE};
    globalBarrier.after = {nri::AccessBits::ACCELERATION_STRUCTURE_WRITE, nri::StageBits::ACCELERATION_STRUCTURE};
    context.core.CmdBarrier(*commandBuffer, barrierDesc);

    aabb.maxX = 2.0f;
    mapped = context.core.MapBuffer(*aabbBuffer, 0, sizeof(aabb));
    memcpy(mapped, &aabb, sizeof(aabb));
    context.core.UnmapBuffer(*aabbBuffer);
    buildDesc.src = source;
    guard.interface.CmdBuildBottomLevelAccelerationStructures(*commandBuffer, &buildDesc, 1);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    return test::Report("advanced acceleration structures", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
