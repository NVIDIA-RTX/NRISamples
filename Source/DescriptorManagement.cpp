// © 2026 NVIDIA Corporation

#include "TestShared.h"

namespace {

bool Run(const test::Settings& settings) {
    test::Context context;
    if (!context.Initialize(settings) || context.skipped)
        return context.skipped;

    nri::Queue* queue = nullptr;
    TEST_CHECK(context.core.GetQueue(*context.device, nri::QueueType::GRAPHICS, 0, queue));

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

    const nri::UpdateDescriptorRangeDesc updateDesc = {sets[0], 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&updateDesc, 1);

    nri::CopyDescriptorRangeDesc copyDesc = {};
    copyDesc.dstDescriptorSet = sets[1];
    copyDesc.srcDescriptorSet = sets[0];
    copyDesc.descriptorNum = 2;
    context.core.CopyDescriptorRanges(&copyDesc, 1);

    const nri::UpdateDescriptorRangeDesc updateAfterSetInit = {updateAfterSet, 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&updateAfterSetInit, 1);

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

    context.core.ResetDescriptorPool(*pools[0]);
    nri::DescriptorSet* recycledSet = nullptr;
    TEST_CHECK(context.core.AllocateDescriptorSets(*pools[0], *pipelineLayouts[0], 0, &recycledSet, 1, 0));
    const nri::UpdateDescriptorRangeDesc recycledUpdate = {recycledSet, 0, 0, views, 2};
    context.core.UpdateDescriptorRanges(&recycledUpdate, 1);

    return test::Report("descriptor pool management", true);
}

} // namespace

int main(int argc, char** argv) {
    return Run(test::ParseSettings(argc, argv)) ? 0 : 1;
}
