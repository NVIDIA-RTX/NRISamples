// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

    // Create and destroy shading-rate attachment descriptors
    if (context.deviceDesc->tiers.shadingRate >= 2) {
        nri::TextureDesc textureDesc = {};
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        textureDesc.usage = nri::TextureUsageBits::SHADING_RATE_ATTACHMENT;
        textureDesc.format = nri::Format::R8_UINT;
        textureDesc.width = 1;
        textureDesc.height = 1;
        textureDesc.mipNum = 1;

        nri::Texture* texture = nullptr;
        TEST_CHECK(context.CreateTexture(textureDesc, nri::MemoryLocation::DEVICE, texture));

        const nri::TextureViewDesc textureViewDesc = {texture, nri::TextureView::SHADING_RATE_ATTACHMENT, nri::Format::R8_UINT};
        for (uint32_t i = 0; i < 2; i++) {
            nri::Descriptor* shadingRateAttachment = nullptr;
            TEST_CHECK(context.core.CreateTextureView(textureViewDesc, shadingRateAttachment));
            context.core.DestroyDescriptor(shadingRateAttachment);
        }
    }

    // Create source descriptors
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = 256;
    bufferDesc.usage = nri::BufferUsageBits::CONSTANT;

    nri::Descriptor* views[3] = {};
    for (uint32_t i = 0; i < 3; i++) {
        nri::Buffer* buffer = nullptr;
        TEST_CHECK(context.CreateBuffer(bufferDesc, nri::MemoryLocation::HOST_UPLOAD, buffer));

        nri::BufferViewDesc viewDesc = {};
        viewDesc.buffer = buffer;
        viewDesc.type = nri::BufferView::CONSTANT_BUFFER;
        viewDesc.size = nri::WHOLE_SIZE;
        TEST_CHECK(context.core.CreateBufferView(viewDesc, views[i]));
        context.Track(views[i]);
    }

    // Copy descriptors and update a bound descriptor set
    nri::DescriptorRangeDesc rangeDesc = {};
    rangeDesc.descriptorNum = 2;
    rangeDesc.descriptorType = nri::DescriptorType::CONSTANT_BUFFER;
    rangeDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
    rangeDesc.flags = nri::DescriptorRangeBits::ARRAY | nri::DescriptorRangeBits::ALLOW_UPDATE_AFTER_SET;

    nri::DescriptorRangeDesc copyRangeDesc = rangeDesc;
    copyRangeDesc.flags = nri::DescriptorRangeBits::ARRAY;

    nri::DescriptorSetDesc setDesc = {};
    setDesc.ranges = &rangeDesc;
    setDesc.rangeNum = 1;
    setDesc.flags = nri::DescriptorSetBits::ALLOW_UPDATE_AFTER_SET;

    nri::DescriptorSetDesc copySetDesc = setDesc;
    copySetDesc.ranges = &copyRangeDesc;
    copySetDesc.flags = nri::DescriptorSetBits::NONE;

    nri::PipelineLayoutDesc pipelineLayoutDesc = {};
    pipelineLayoutDesc.descriptorSets = &setDesc;
    pipelineLayoutDesc.descriptorSetNum = 1;
    pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;

    nri::PipelineLayoutDesc copyPipelineLayoutDesc = pipelineLayoutDesc;
    copyPipelineLayoutDesc.descriptorSets = &copySetDesc;

    nri::PipelineLayout* pipelineLayouts[2] = {};
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, copyPipelineLayoutDesc, pipelineLayouts[0]));
    TEST_CHECK(context.core.CreatePipelineLayout(*context.device, pipelineLayoutDesc, pipelineLayouts[1]));
    context.Track(pipelineLayouts[0]);
    context.Track(pipelineLayouts[1]);

    nri::DescriptorPoolDesc copySourcePoolDesc = {};
    copySourcePoolDesc.descriptorSetMaxNum = 2;
    copySourcePoolDesc.constantBufferMaxNum = 8;
    copySourcePoolDesc.flags = nri::DescriptorPoolBits::COPY_SOURCE;

    nri::DescriptorPoolDesc poolDesc = copySourcePoolDesc;
    poolDesc.flags = nri::DescriptorPoolBits::NONE;

    nri::DescriptorPoolDesc updateAfterSetPoolDesc = poolDesc;
    updateAfterSetPoolDesc.flags = nri::DescriptorPoolBits::ALLOW_UPDATE_AFTER_SET;

    nri::DescriptorPool* pools[3] = {};
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, copySourcePoolDesc, pools[0]));
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, poolDesc, pools[1]));
    TEST_CHECK(context.core.CreateDescriptorPool(*context.device, updateAfterSetPoolDesc, pools[2]));
    for (nri::DescriptorPool* pool : pools)
        context.Track(pool);

    nri::DescriptorSet* sets[2] = {};
    TEST_CHECK(context.core.AllocateDescriptorSets(*pools[0], *pipelineLayouts[0], 0, &sets[0], 1, 0));
    TEST_CHECK(context.core.AllocateDescriptorSets(*pools[1], *pipelineLayouts[0], 0, &sets[1], 1, 0));

    nri::DescriptorSet* updateAfterSet = nullptr;
    TEST_CHECK(context.core.AllocateDescriptorSets(*pools[2], *pipelineLayouts[1], 0, &updateAfterSet, 1, 0));

    // Initialize and copy descriptor ranges
    const nri::UpdateDescriptorRangeDesc updateDesc = {sets[0], 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&updateDesc, 1);

    nri::CopyDescriptorRangeDesc copyDesc = {};
    copyDesc.dstDescriptorSet = sets[1];
    copyDesc.srcDescriptorSet = sets[0];
    copyDesc.descriptorNum = 2;
    context.core.CopyDescriptorRanges(&copyDesc, 1);

    const nri::UpdateDescriptorRangeDesc updateAfterSetInit = {updateAfterSet, 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&updateAfterSetInit, 1);

    // Bind descriptor sets from multiple pools and update a set after binding
    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    TEST_CHECK(context.CreateCommandObjects(*queue, commandAllocator, commandBuffer));
    TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, pools[1]));

    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *pipelineLayouts[0]);
    nri::SetDescriptorSetDesc setDescriptorSetDesc = {0, sets[1], nri::BindPoint::COMPUTE};
    context.core.CmdSetDescriptorSet(*commandBuffer, setDescriptorSetDesc);

    context.core.CmdSetDescriptorPool(*commandBuffer, *pools[2]);
    context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *pipelineLayouts[1]);
    setDescriptorSetDesc.descriptorSet = updateAfterSet;
    context.core.CmdSetDescriptorSet(*commandBuffer, setDescriptorSetDesc);

    const nri::UpdateDescriptorRangeDesc updateAfterSetDesc = {updateAfterSet, 0, 1, &views[2], 1};
    context.core.UpdateDescriptorRanges(&updateAfterSetDesc, 1);
    TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));

    // Reset a descriptor pool and reuse its storage
    context.core.ResetDescriptorPool(*pools[0]);
    nri::DescriptorSet* recycledSet = nullptr;
    TEST_CHECK(context.core.AllocateDescriptorSets(*pools[0], *pipelineLayouts[0], 0, &recycledSet, 1, 0));
    const nri::UpdateDescriptorRangeDesc recycledUpdate = {recycledSet, 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&recycledUpdate, 1);

    // Allocate, update, copy and bind variable-sized descriptor arrays
    if (context.deviceDesc->tiers.bindless != 0 && context.deviceDesc->tiers.resourceBinding == 2) {
        nri::DescriptorRangeDesc variableRangeDesc = {};
        variableRangeDesc.descriptorNum = 3;
        variableRangeDesc.descriptorType = nri::DescriptorType::CONSTANT_BUFFER;
        variableRangeDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        variableRangeDesc.flags = nri::DescriptorRangeBits::VARIABLE_SIZED_ARRAY;

        nri::DescriptorRangeDesc fixedRangeDesc = variableRangeDesc;
        fixedRangeDesc.descriptorNum = 1;
        fixedRangeDesc.flags = nri::DescriptorRangeBits::NONE;
        variableRangeDesc.baseRegisterIndex = 1;

        const nri::DescriptorRangeDesc variableRangeDescs[] = {fixedRangeDesc, variableRangeDesc};

        nri::DescriptorSetDesc variableSetDescs[2] = {};
        variableSetDescs[0].ranges = variableRangeDescs;
        variableSetDescs[0].rangeNum = 2;
        variableSetDescs[1].ranges = &fixedRangeDesc;
        variableSetDescs[1].rangeNum = 1;
        variableSetDescs[1].registerSpace = 1;

        nri::PipelineLayoutDesc variablePipelineLayoutDesc = {};
        variablePipelineLayoutDesc.descriptorSets = variableSetDescs;
        variablePipelineLayoutDesc.descriptorSetNum = 2;
        variablePipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;

        nri::PipelineLayout* variablePipelineLayout = nullptr;
        TEST_CHECK(context.core.CreatePipelineLayout(*context.device, variablePipelineLayoutDesc, variablePipelineLayout));
        context.Track(variablePipelineLayout);

        nri::DescriptorPoolDesc variableAllocationPoolDesc = {};
        variableAllocationPoolDesc.descriptorSetMaxNum = 3;
        variableAllocationPoolDesc.constantBufferMaxNum = 5;

        nri::DescriptorPoolDesc variableSourcePoolDesc = {};
        variableSourcePoolDesc.descriptorSetMaxNum = 1;
        variableSourcePoolDesc.constantBufferMaxNum = 2;
        variableSourcePoolDesc.flags = nri::DescriptorPoolBits::COPY_SOURCE;

        nri::DescriptorPoolDesc variableDestinationPoolDesc = {};
        variableDestinationPoolDesc.descriptorSetMaxNum = 1;
        variableDestinationPoolDesc.constantBufferMaxNum = 2;

        nri::DescriptorPool* variableAllocationPool = nullptr;
        nri::DescriptorPool* variableSourcePool = nullptr;
        nri::DescriptorPool* variableDestinationPool = nullptr;
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, variableAllocationPoolDesc, variableAllocationPool));
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, variableSourcePoolDesc, variableSourcePool));
        TEST_CHECK(context.core.CreateDescriptorPool(*context.device, variableDestinationPoolDesc, variableDestinationPool));
        context.Track(variableAllocationPool);
        context.Track(variableSourcePool);
        context.Track(variableDestinationPool);

        nri::DescriptorSet* variableAllocationSets[2] = {};
        nri::DescriptorSet* variableSourceSet = nullptr;
        nri::DescriptorSet* fixedSet = nullptr;
        nri::DescriptorSet* variableDestinationSet = nullptr;
        TEST_CHECK(context.core.AllocateDescriptorSets(*variableAllocationPool, *variablePipelineLayout, 0, variableAllocationSets, 2, 1));
        TEST_CHECK(context.core.AllocateDescriptorSets(*variableAllocationPool, *variablePipelineLayout, 1, &fixedSet, 1, 0));
        TEST_CHECK(context.core.AllocateDescriptorSets(*variableSourcePool, *variablePipelineLayout, 0, &variableSourceSet, 1, 1));
        TEST_CHECK(context.core.AllocateDescriptorSets(*variableDestinationPool, *variablePipelineLayout, 0, &variableDestinationSet, 1, 1));

        // Verify that variable-sized allocations consume only the requested number of descriptors
        if (context.deviceDesc->graphicsAPI == nri::GraphicsAPI::D3D12) {
            uint32_t resourceHeapOffset = 0;
            uint32_t samplerHeapOffset = 0;
            context.core.GetDescriptorSetOffsets(*fixedSet, resourceHeapOffset, samplerHeapOffset);
            TEST_CHECK(resourceHeapOffset == 4);
        }

        // Update and copy variable-sized descriptor ranges
        const nri::UpdateDescriptorRangeDesc variableUpdateDescs[] = {
            {variableAllocationSets[0], 0, 0, views, 1},
            {variableAllocationSets[0], 1, 0, views, 1},
            {variableSourceSet, 0, 0, views, 1},
            {variableSourceSet, 1, 0, views, 1},
            {fixedSet, 0, 0, views, 1},
            {variableDestinationSet, 0, 0, views, 1},
        };
        context.core.UpdateDescriptorRanges(variableUpdateDescs, (uint32_t)std::size(variableUpdateDescs));

        nri::CopyDescriptorRangeDesc variableCopyDesc = {};
        variableCopyDesc.dstDescriptorSet = variableDestinationSet;
        variableCopyDesc.dstRangeIndex = 1;
        variableCopyDesc.srcDescriptorSet = variableSourceSet;
        variableCopyDesc.srcRangeIndex = 1;
        variableCopyDesc.descriptorNum = 1;
        context.core.CopyDescriptorRanges(&variableCopyDesc, 1);

        // Bind a variable-sized descriptor set
        context.core.ResetCommandAllocator(*commandAllocator);
        TEST_CHECK(context.core.BeginCommandBuffer(*commandBuffer, variableAllocationPool));
        context.core.CmdSetPipelineLayout(*commandBuffer, nri::BindPoint::COMPUTE, *variablePipelineLayout);
        const nri::SetDescriptorSetDesc variableSetDescriptorSetDesc = {0, variableAllocationSets[0], nri::BindPoint::COMPUTE};
        context.core.CmdSetDescriptorSet(*commandBuffer, variableSetDescriptorSetDesc);
        TEST_CHECK(context.SubmitAndWait(*queue, *commandBuffer));
    }

    return test::Report("descriptor pool management", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
