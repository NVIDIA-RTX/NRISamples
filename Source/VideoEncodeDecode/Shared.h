// © 2021 NVIDIA Corporation

#pragma once

#if defined(_WIN32)
#    include <d3d12.h>
#endif

#include "NRIFramework.h"

#include "Extensions/NRIVideo.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace video_sample {

constexpr uint32_t DEFAULT_VIDEO_WIDTH = 1920;
constexpr uint32_t DEFAULT_VIDEO_HEIGHT = 1080;
constexpr double ROUND_TRIP_INTERVAL_SEC = 1.0 / 60.0;
constexpr uint64_t BITSTREAM_SIZE = 2 * 1024 * 1024;
constexpr uint64_t ENCODED_SLICE_OFFSET = 4096;
constexpr uint64_t METADATA_SIZE = 4 * 1024 * 1024;
constexpr uint64_t RESOLVED_METADATA_SIZE = 4096;

enum PatternOperation : uint32_t {
    OP_GENERATE_PATTERN = 0,
    OP_NV12_TO_PREVIEW = 1,
};

enum class SampleCodec : uint8_t {
    H264,
    H265,
    AV1,
};

enum class VisualFrameMode : uint8_t {
    IDR,
    P,
    B,
};

struct Av1SequenceOptions {
    bool enableCdef = true;
    bool enableRestoration = true;
    uint8_t seqForceScreenContentTools = 2;
};

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
};

struct PatternConstants {
    uint32_t width = DEFAULT_VIDEO_WIDTH;
    uint32_t height = DEFAULT_VIDEO_HEIGHT;
    uint32_t yOffsetBytes = 0;
    uint32_t yRowPitchBytes = 0;
    uint32_t uvRowPitchBytes = 0;
    uint32_t uvOffsetBytes = 0;
    uint32_t operation = OP_GENERATE_PATTERN;
    float time = 0.0f;
    uint32_t _padding = 0;
    uint32_t _padding1 = 0;
};

struct Nv12BufferLayout {
    uint32_t yRowPitchBytes = DEFAULT_VIDEO_WIDTH;
    uint32_t ySlicePitchBytes = DEFAULT_VIDEO_WIDTH * DEFAULT_VIDEO_HEIGHT;
    uint64_t uvOffsetBytes = uint64_t(DEFAULT_VIDEO_WIDTH) * DEFAULT_VIDEO_HEIGHT;
    uint32_t uvRowPitchBytes = DEFAULT_VIDEO_WIDTH;
    uint32_t uvSlicePitchBytes = DEFAULT_VIDEO_WIDTH * DEFAULT_VIDEO_HEIGHT / 2;
    uint64_t totalSizeBytes = uint64_t(DEFAULT_VIDEO_WIDTH) * DEFAULT_VIDEO_HEIGHT * 3 / 2;
};

struct VideoConfig {
    SampleCodec codec = SampleCodec::H264;
    uint32_t videoWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t videoHeight = DEFAULT_VIDEO_HEIGHT;
    bool av1PFrameVisual = false;
    VisualFrameMode h26FrameMode = VisualFrameMode::IDR;
};

struct VideoQuality {
    uint32_t qpI = 20;
    uint32_t qpP = 22;
    uint32_t qpB = 24;
    uint32_t av1BaseQIndex = 20;
    bool lossless = false;
};

struct VideoSize {
    uint32_t videoWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t videoHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t codedWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t codedHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t decodeBitstreamSizeAlignment = 256;
};

struct VideoContext {
    NRIInterface* nri = nullptr;
    nri::VideoInterface* video = nullptr;
    nri::Device* device = nullptr;
    nri::Queue* graphicsQueue = nullptr;
    nri::Queue* encodeQueue = nullptr;
    nri::Queue* decodeQueue = nullptr;
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::NONE;
};

struct CodecParameters {
    nri::VideoH264SequenceParameterSetDesc h264Sps = {};
    nri::VideoH264PictureParameterSetDesc h264Pps = {};
    nri::VideoH265VideoParameterSetDesc h265Vps = {};
    nri::VideoH265SequenceParameterSetDesc h265Sps = {};
    nri::VideoH265PictureParameterSetDesc h265Pps = {};
    nri::VideoAV1SequenceDesc av1Sequence = {};
};

struct EncodedFrame {
    nri::VideoEncodeFeedback feedback = {};
    nri::Buffer* bitstreamBuffer = nullptr;
    nri::VideoAV1EncodeDecodeInfo av1DecodeInfo = {};
    nri::VideoEncodeFrameType frameType = nri::VideoEncodeFrameType::IDR;
    uint32_t frameIndex = 0;
    int32_t pictureOrderCount = 0;
    uint32_t outputSlot = 0;
    bool hasAv1DecodeInfo = false;
    bool isAv1PFrame = false;
    bool isDisplayFrame = true;
};

struct DecodedFrame {
    nri::Texture* texture = nullptr;
    nri::VideoPicture* picture = nullptr;
    nri::VideoDecodePictureStates pictureStates = {};
    bool isAv1PFrame = false;
};

inline const char* GetCodecName(SampleCodec codec) {
    switch (codec) {
        case SampleCodec::H265:
            return "H.265";
        case SampleCodec::AV1:
            return "AV1";
        case SampleCodec::H264:
        default:
            return "H.264";
    }
}

inline nri::VideoCodec GetNriCodec(SampleCodec codec) {
    switch (codec) {
        case SampleCodec::H265:
            return nri::VideoCodec::H265;
        case SampleCodec::AV1:
            return nri::VideoCodec::AV1;
        case SampleCodec::H264:
        default:
            return nri::VideoCodec::H264;
    }
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

inline uint8_t GetAv1FrameSizeBitsMinus1(uint32_t value) {
    uint32_t bits = 0;
    uint32_t maxValue = value ? value - 1 : 0;
    do {
        bits++;
        maxValue >>= 1;
    } while (maxValue);
    return (uint8_t)(bits - 1);
}

inline nri::VideoAV1SequenceDesc MakeAV1SequenceDesc(uint32_t width, uint32_t height, const Av1SequenceOptions& options) {
    nri::VideoAV1SequenceDesc desc = {};
    desc.flags = nri::VideoAV1SequenceBits::ENABLE_ORDER_HINT | nri::VideoAV1SequenceBits::COLOR_DESCRIPTION_PRESENT;
    if (options.enableCdef)
        desc.flags |= nri::VideoAV1SequenceBits::ENABLE_CDEF;
    if (options.enableRestoration)
        desc.flags |= nri::VideoAV1SequenceBits::ENABLE_RESTORATION;
    desc.bitDepth = 8;
    desc.subsamplingX = 1;
    desc.subsamplingY = 1;
    desc.maxFrameWidthMinus1 = (uint16_t)(width - 1);
    desc.maxFrameHeightMinus1 = (uint16_t)(height - 1);
    desc.frameWidthBitsMinus1 = GetAv1FrameSizeBitsMinus1(width);
    desc.frameHeightBitsMinus1 = GetAv1FrameSizeBitsMinus1(height);
    desc.orderHintBitsMinus1 = 7;
    desc.seqForceIntegerMv = 2;
    desc.seqForceScreenContentTools = options.seqForceScreenContentTools;
    desc.colorPrimaries = 1;
    desc.transferCharacteristics = 1;
    desc.matrixCoefficients = 1;
    desc.chromaSamplePosition = 1;
    return desc;
}

inline Nv12BufferLayout MakeNv12BufferLayout(const nri::DeviceDesc& deviceDesc, uint32_t width, uint32_t height) {
    const uint32_t rowAlignment = std::max(deviceDesc.memoryAlignment.uploadBufferTextureRow, 1u);
    const uint32_t sliceAlignment = std::max(deviceDesc.memoryAlignment.uploadBufferTextureSlice, 1u);

    Nv12BufferLayout layout = {};
    layout.yRowPitchBytes = (uint32_t)AlignUp(width, rowAlignment);
    layout.ySlicePitchBytes = (uint32_t)AlignUp(uint64_t(layout.yRowPitchBytes) * height, sliceAlignment);
    layout.uvOffsetBytes = layout.ySlicePitchBytes;
    layout.uvRowPitchBytes = (uint32_t)AlignUp(width, rowAlignment);
    layout.uvSlicePitchBytes = (uint32_t)AlignUp(uint64_t(layout.uvRowPitchBytes) * (height / 2), sliceAlignment);
    layout.totalSizeBytes = layout.uvOffsetBytes + layout.uvSlicePitchBytes;
    return layout;
}

inline CodecParameters MakeCodecParameters(nri::GraphicsAPI graphicsAPI, uint32_t codedWidth, uint32_t codedHeight) {
    CodecParameters params = {};

    params.h264Sps.flags = nri::VideoH264SequenceParameterSetBits::DIRECT_8X8_INFERENCE | nri::VideoH264SequenceParameterSetBits::FRAME_MBS_ONLY | nri::VideoH264SequenceParameterSetBits::QPPRIME_Y_ZERO_TRANSFORM_BYPASS;
    params.h264Sps.profileIdc = 100;
    params.h264Sps.levelIdc = 42;
    params.h264Sps.chromaFormatIdc = 1;
    params.h264Sps.sequenceParameterSetId = 0;
    params.h264Sps.log2MaxFrameNumMinus4 = 0;
    params.h264Sps.pictureOrderCountType = 0;
    params.h264Sps.log2MaxPictureOrderCountLsbMinus4 = 0;
    params.h264Sps.referenceFrameNum = 1;
    params.h264Sps.pictureWidthInMbsMinus1 = (uint16_t)((codedWidth + 15) / 16 - 1);
    params.h264Sps.pictureHeightInMapUnitsMinus1 = (uint16_t)((codedHeight + 15) / 16 - 1);

    params.h264Pps.flags = nri::VideoH264PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT;
    params.h264Pps.sequenceParameterSetId = 0;
    params.h264Pps.pictureParameterSetId = 0;
    params.h264Pps.refIndexL0DefaultActiveMinus1 = 0;
    params.h264Pps.refIndexL1DefaultActiveMinus1 = 0;

    params.h265Vps.flags = nri::VideoH265VideoParameterSetBits::TEMPORAL_ID_NESTING;
    params.h265Vps.videoParameterSetId = 0;
    params.h265Vps.maxSubLayersMinus1 = 0;
    params.h265Vps.profileTierLevel.flags = nri::VideoH265ProfileTierLevelBits::FRAME_ONLY_CONSTRAINT;
    params.h265Vps.profileTierLevel.generalProfileIdc = 1;
    params.h265Vps.profileTierLevel.generalLevelIdc = 90;
    params.h265Vps.decPicBufMgr.maxDecPicBufferingMinus1[0] = 2;
    params.h265Vps.decPicBufMgr.maxNumReorderPics[0] = 1;

    params.h265Sps.flags = nri::VideoH265SequenceParameterSetBits::TEMPORAL_ID_NESTING | nri::VideoH265SequenceParameterSetBits::AMP_ENABLED | nri::VideoH265SequenceParameterSetBits::SAMPLE_ADAPTIVE_OFFSET_ENABLED;
    params.h265Sps.videoParameterSetId = params.h265Vps.videoParameterSetId;
    params.h265Sps.maxSubLayersMinus1 = params.h265Vps.maxSubLayersMinus1;
    params.h265Sps.sequenceParameterSetId = 0;
    params.h265Sps.chromaFormatIdc = 1;
    params.h265Sps.pictureWidthInLumaSamples = codedWidth;
    params.h265Sps.pictureHeightInLumaSamples = codedHeight;
    params.h265Sps.log2MaxPictureOrderCountLsbMinus4 = 3;
    params.h265Sps.log2MinLumaCodingBlockSizeMinus3 = 0;
    params.h265Sps.log2DiffMaxMinLumaCodingBlockSize = 2;
    params.h265Sps.log2MinLumaTransformBlockSizeMinus2 = 0;
    params.h265Sps.log2DiffMaxMinLumaTransformBlockSize = 3;
    params.h265Sps.maxTransformHierarchyDepthInter = 3;
    params.h265Sps.maxTransformHierarchyDepthIntra = 3;
    params.h265Sps.profileTierLevel = params.h265Vps.profileTierLevel;
    params.h265Sps.decPicBufMgr = params.h265Vps.decPicBufMgr;

    params.h265Pps.flags = nri::VideoH265PictureParameterSetBits::CABAC_INIT_PRESENT | nri::VideoH265PictureParameterSetBits::TRANSFORM_SKIP_ENABLED | nri::VideoH265PictureParameterSetBits::CU_QP_DELTA_ENABLED | nri::VideoH265PictureParameterSetBits::SLICE_CHROMA_QP_OFFSETS_PRESENT | nri::VideoH265PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT | nri::VideoH265PictureParameterSetBits::LISTS_MODIFICATION_PRESENT;
    params.h265Pps.pictureParameterSetId = 0;
    params.h265Pps.sequenceParameterSetId = params.h265Sps.sequenceParameterSetId;
    params.h265Pps.videoParameterSetId = params.h265Vps.videoParameterSetId;

    Av1SequenceOptions av1SequenceOptions = {};
    if (graphicsAPI == nri::GraphicsAPI::VK) {
        av1SequenceOptions.enableCdef = false;
        av1SequenceOptions.enableRestoration = false;
        av1SequenceOptions.seqForceScreenContentTools = 0;
    }
    params.av1Sequence = MakeAV1SequenceDesc(codedWidth, codedHeight, av1SequenceOptions);
    return params;
}

template <typename Record>
bool SubmitOneTime(nri::CoreInterface& core, nri::Queue& queue, Record&& record) {
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    bool ok = core.CreateCommandAllocator(queue, allocator) == nri::Result::SUCCESS && allocator && core.CreateCommandBuffer(*allocator, commandBuffer) == nri::Result::SUCCESS && commandBuffer && core.BeginCommandBuffer(*commandBuffer, nullptr) == nri::Result::SUCCESS;
    if (ok) {
        std::forward<Record>(record)(*commandBuffer);
        ok = core.EndCommandBuffer(*commandBuffer) == nri::Result::SUCCESS;
    }
    if (ok) {
        const nri::CommandBuffer* commandBuffers[] = {commandBuffer};
        nri::QueueSubmitDesc submit = {};
        submit.commandBuffers = commandBuffers;
        submit.commandBufferNum = 1;
        ok = core.QueueSubmit(queue, submit) == nri::Result::SUCCESS && core.QueueWaitIdle(&queue) == nri::Result::SUCCESS;
    }
    if (commandBuffer)
        core.DestroyCommandBuffer(commandBuffer);
    if (allocator)
        core.DestroyCommandAllocator(allocator);
    return ok;
}

template <typename Record>
bool SubmitOneTime(nri::CoreInterface& core, nri::Queue& queue, nri::DescriptorPool* descriptorPool, Record&& record) {
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    bool ok = core.CreateCommandAllocator(queue, allocator) == nri::Result::SUCCESS && allocator && core.CreateCommandBuffer(*allocator, commandBuffer) == nri::Result::SUCCESS && commandBuffer && core.BeginCommandBuffer(*commandBuffer, descriptorPool) == nri::Result::SUCCESS;
    if (ok) {
        std::forward<decltype(record)>(record)(*commandBuffer);
        ok = core.EndCommandBuffer(*commandBuffer) == nri::Result::SUCCESS;
    }
    if (ok) {
        const nri::CommandBuffer* commandBuffers[] = {commandBuffer};
        nri::QueueSubmitDesc submit = {};
        submit.commandBuffers = commandBuffers;
        submit.commandBufferNum = 1;
        ok = core.QueueSubmit(queue, submit) == nri::Result::SUCCESS && core.QueueWaitIdle(&queue) == nri::Result::SUCCESS;
    }
    if (commandBuffer)
        core.DestroyCommandBuffer(commandBuffer);
    if (allocator)
        core.DestroyCommandAllocator(allocator);
    return ok;
}

inline bool CopyNv12BufferToTexture(nri::CoreInterface& core, nri::Queue& queue, const Nv12BufferLayout& layout, nri::Buffer& src, nri::Texture& dst, uint32_t width, uint32_t height) {
    return SubmitOneTime(core, queue, [&](nri::CommandBuffer& commandBuffer) {
        nri::BufferBarrierDesc bufferBarrier = {};
        bufferBarrier.buffer = &src;
        bufferBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};

        nri::TextureBarrierDesc textureBarrier = {};
        textureBarrier.texture = &dst;
        textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
        textureBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        textureBarrier.mipNum = nri::REMAINING;
        textureBarrier.layerNum = nri::REMAINING;
        textureBarrier.planes = nri::PlaneBits::ALL;

        nri::BarrierDesc barrierDesc = {};
        barrierDesc.buffers = &bufferBarrier;
        barrierDesc.bufferNum = 1;
        barrierDesc.textures = &textureBarrier;
        barrierDesc.textureNum = 1;
        core.CmdBarrier(commandBuffer, barrierDesc);

        nri::TextureRegionDesc lumaRegion = {};
        lumaRegion.width = (nri::Dim_t)width;
        lumaRegion.height = (nri::Dim_t)height;
        lumaRegion.depth = 1;
        lumaRegion.planes = nri::PlaneBits::PLANE_0;

        nri::TextureDataLayoutDesc lumaLayout = {};
        lumaLayout.rowPitch = layout.yRowPitchBytes;
        lumaLayout.slicePitch = layout.ySlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, dst, lumaRegion, src, lumaLayout);

        nri::TextureRegionDesc chromaRegion = {};
        chromaRegion.width = (nri::Dim_t)width;
        chromaRegion.height = (nri::Dim_t)height;
        chromaRegion.depth = 1;
        chromaRegion.planes = nri::PlaneBits::PLANE_1;

        nri::TextureDataLayoutDesc chromaLayout = {};
        chromaLayout.offset = layout.uvOffsetBytes;
        chromaLayout.rowPitch = layout.uvRowPitchBytes;
        chromaLayout.slicePitch = layout.uvSlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, dst, chromaRegion, src, chromaLayout);

        textureBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
        bufferBarrier.before = bufferBarrier.after;
        bufferBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        core.CmdBarrier(commandBuffer, barrierDesc);
    });
}

inline nri::Result CreateEncodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_READBACK, priority, bufferDesc, buffer);
}

inline nri::Result CreateDecodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_UPLOAD, priority, bufferDesc, buffer);
}

} // namespace video_sample
