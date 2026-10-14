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
constexpr uint64_t ENCODED_SLICE_OFFSET = 4096;
constexpr uint64_t RESOLVED_METADATA_SIZE = 4096;
constexpr uint32_t MAX_REFERENCE_NUM = 8;
constexpr uint32_t MAX_DPB_SLOT_NUM = MAX_REFERENCE_NUM + 1; // references + one setup slot

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

enum class RateControlKind : uint8_t {
    CQP,
    CBR,
    VBR,
};

enum class DecodeDpbMode : uint8_t {
    AUTO,     // coincide if supported
    COINCIDE, // decode output is the DPB setup picture
    DISTINCT, // decode output ("VIDEO_OUTPUT_ONLY") differs from the DPB setup picture
};

enum class DpbLayout : uint8_t {
    AUTO,     // texture array if the backend requires it, otherwise separate textures
    ARRAY,    // one texture array, one layer per DPB slot
    SEPARATE, // one texture per DPB slot
};

enum class Av1InfoMode : uint8_t {
    AUTO,     // VK: payload header, D3D12: metadata only
    HEADER,   // "encodedPayloadHeader"
    METADATA, // submitted picture descriptions only
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

    // Stream configuration
    uint32_t referenceNum = 1; // "VideoSessionDesc::maxReferenceNum" of both sessions, DPB has "referenceNum + 1" slots
    bool hasBFrames = false;
    DecodeDpbMode decodeDpb = DecodeDpbMode::AUTO;
    DpbLayout dpbLayout = DpbLayout::AUTO;
    bool encodeInputOutputOnly = true; // "VIDEO_OUTPUT_ONLY" encode source
    bool h265ListsModification = true; // PPS lists_modification_present_flag
    bool h265ScalingListEnabled = false; // SPS scaling_list_enabled_flag without lists (default scaling lists, the encoders do not apply them)
    bool reverseDecodeReferences = true; // H.265: pass the decode references of every second picture in the opposite POC order, also without list modification
    bool reverseReferences = true;     // H.265: pass the references of every second picture in the opposite POC order (the backend sorts them)
    bool encodeFeedback = true;        // "CmdResolveVideoEncodeFeedback" + "GetVideoEncodeFeedback", otherwise the bitstream is scanned
    Av1InfoMode av1Info = Av1InfoMode::AUTO;
    bool h265DefaultParameters = false; // encode session parameters are created by the backend (decode and Annex-B headers mirror them)
    bool decodePictureAtVideoSize = false; // decode pictures cover the video size (external streams) instead of the coded texture size
    bool h264TransformBypass = false;   // SPS qpprime_y_zero_transform_bypass_flag (lossless with QP 0), not allowed by the High profile
    uint32_t codedAlignment = 1;       // minimum alignment of the coded size, e.g. 16 (macroblocks) for drivers that code aligned pictures
    uint8_t h264SpsId = 0;
    uint8_t h264PpsRefIdxL0DefaultActiveMinus1 = 0; // PPS num_ref_idx_l0_default_active_minus1
    uint8_t h264PpsId = 0;
    uint8_t h265VpsId = 0;
    uint8_t h265SpsId = 0;
    uint8_t h265PpsId = 0;
};

struct RateControlSetting {
    RateControlKind kind = RateControlKind::CQP;
    uint32_t bitrateKbps = 4000;
    uint32_t maxBitrateKbps = 0; // VBR peak, 0 = target
    uint32_t qpMin = 0;          // CBR/VBR QP range, 0 = backend default
    uint32_t qpMax = 0;
    uint32_t maxFrameBits = 0;   // 0 = unlimited
    uint32_t vbvMs = 0;          // virtual buffer size, 0 = backend default
    uint32_t vbvInitialMs = 0;
};

struct VideoQuality {
    uint32_t qpI = 20;
    uint32_t qpP = 22;
    uint32_t qpB = 24;
    uint32_t av1BaseQIndex = 20;
    bool lossless = false;
    RateControlSetting rateControl = {};
};

struct VideoSize {
    uint32_t videoWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t videoHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t codedWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t codedHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t decodeBitstreamSizeAlignment = 256;
    uint32_t bitstreamOffsetAlignment = 1;
    uint32_t bitstreamSizeAlignment = 1;
    uint64_t bitstreamSize = 2 * 1024 * 1024;
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

// One coded picture in decode order
struct PlanReference {
    uint32_t slot = 0;
    int32_t poc = 0;
    uint32_t frameNum = 0;
    uint32_t frameId = 0; // AV1
    uint8_t orderHint = 0;
    nri::VideoFrameType type = nri::VideoFrameType::IDR;
    uint8_t listIndex = 0; // H.264/H.265: 0 - List0 (past), 1 - List1 (future)
};

struct FramePlan {
    nri::VideoFrameType type = nri::VideoFrameType::IDR;
    uint32_t codingIndex = 0;
    uint32_t displayIndex = 0;
    int32_t poc = 0;
    uint32_t frameNum = 0; // H.264 "frame_num", AV1 frame id
    uint32_t idrPictureId = 0;
    uint32_t setupSlot = 0; // always allocated, a non-reference picture takes a free slot
    bool isReference = true;
    bool isDisplay = true; // interactive: the last picture of a cycle
    bool reverseReferenceOrder = false; // H.265 RPS order independence (references are passed in POC-unsorted order)
    bool reverseDecodeReferenceOrder = false; // the same for the decode only (encode lists are order dependent without "lists_modification_present_flag")
    uint8_t av1RefreshFrameFlags = 0xFF;
    uint8_t av1OrderHint = 0;
    uint32_t referenceNum = 0;
    PlanReference references[MAX_REFERENCE_NUM] = {};
};

struct EncodedFrame {
    FramePlan plan = {};
    nri::VideoEncodeFeedback feedback = {};
    std::vector<uint8_t> headers; // Annex-B parameter sets, AV1: temporal delimiter + sequence header (key frames only)
    std::vector<uint8_t> payload;
    nri::VideoAV1EncodeDecodeInfo av1DecodeInfo = {};
    bool hasAv1DecodeInfo = false;
    bool feedbackResolved = false;
    bool av1MetadataUnsupported = false;
    bool hasAv1SavedOrderHints = false;                         // external stream decode: "SavedOrderHints" of the DPB pictures (indexed by DPB slot) instead of zeros
    uint8_t av1SavedOrderHints[MAX_DPB_SLOT_NUM][8] = {};
};

struct DecodedFrame {
    uint32_t displayIndex = 0;
    bool isDisplay = true;
};

// Video picture (texture layer) with the state tracked between operations
struct PictureSlot {
    nri::Texture* texture = nullptr;
    nri::VideoPicture* picture = nullptr;
    uint32_t layer = 0;
    nri::AccessLayoutStage state = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};         // source of the next barrier on the same queue
    nri::AccessLayoutStage consumerState = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE}; // source of a barrier on another queue
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

inline uint64_t AlignDown(uint64_t value, uint64_t alignment) {
    return alignment == 0 ? value : (value / alignment) * alignment;
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

// "general_level_idc" is 30 times the level number
inline uint8_t GetH265GeneralLevelIdc(uint32_t width, uint32_t height) {
    struct Level {
        uint8_t generalLevelIdc;
        uint64_t maxLumaPs;
    };

    static const Level levels[] = {{93, 983040}, {123, 2228224}, {153, 8912896}, {183, 35651584}};
    const uint64_t samples = uint64_t(width) * height;
    for (const Level& level : levels) {
        if (samples <= level.maxLumaPs)
            return level.generalLevelIdc;
    }
    return levels[3].generalLevelIdc;
}

inline uint8_t GetH264LevelIdc(uint32_t width, uint32_t height) {
    const uint64_t macroblockNum = uint64_t((width + 15) / 16) * ((height + 15) / 16);
    if (macroblockNum <= 8704)
        return 42;
    if (macroblockNum <= 36864)
        return 51;
    return 62;
}

// Mirrors the parameters the Vulkan backend creates for an H.265 encode session without "h265Parameters"
inline void MakeH265DefaultParameters(CodecParameters& params, uint32_t codedWidth, uint32_t codedHeight, uint32_t maxReferenceNum) {
    params.h265Vps = {};
    params.h265Sps = {};
    params.h265Pps = {};

    params.h265Vps.flags = nri::VideoH265VideoParameterSetBits::TEMPORAL_ID_NESTING;
    params.h265Vps.profileTierLevel.generalProfileIdc = 1;
    params.h265Vps.profileTierLevel.generalLevelIdc = GetH265GeneralLevelIdc(codedWidth, codedHeight);
    params.h265Vps.decPicBufMgr.maxDecPicBufferingMinus1[0] = (uint8_t)std::min(maxReferenceNum ? maxReferenceNum : 1u, 15u);
    params.h265Vps.decPicBufMgr.maxNumReorderPics[0] = maxReferenceNum > 1 ? 1 : 0;

    params.h265Sps.flags = nri::VideoH265SequenceParameterSetBits::TEMPORAL_ID_NESTING | nri::VideoH265SequenceParameterSetBits::SAMPLE_ADAPTIVE_OFFSET_ENABLED | nri::VideoH265SequenceParameterSetBits::STRONG_INTRA_SMOOTHING_ENABLED;
    params.h265Sps.chromaFormatIdc = 1;
    params.h265Sps.pictureWidthInLumaSamples = codedWidth;
    params.h265Sps.pictureHeightInLumaSamples = codedHeight;
    params.h265Sps.log2MaxPictureOrderCountLsbMinus4 = 4;
    params.h265Sps.log2MinLumaCodingBlockSizeMinus3 = 0;
    params.h265Sps.log2DiffMaxMinLumaCodingBlockSize = 2;
    params.h265Sps.log2MinLumaTransformBlockSizeMinus2 = 0;
    params.h265Sps.log2DiffMaxMinLumaTransformBlockSize = 2;
    params.h265Sps.maxTransformHierarchyDepthInter = 2;
    params.h265Sps.maxTransformHierarchyDepthIntra = 2;
    params.h265Sps.profileTierLevel = params.h265Vps.profileTierLevel;
    params.h265Sps.decPicBufMgr = params.h265Vps.decPicBufMgr;

    params.h265Pps.flags = nri::VideoH265PictureParameterSetBits::LOOP_FILTER_ACROSS_SLICES_ENABLED | nri::VideoH265PictureParameterSetBits::LISTS_MODIFICATION_PRESENT;
    params.h265Pps.log2ParallelMergeLevelMinus2 = 2;
}

inline CodecParameters MakeCodecParameters(nri::GraphicsAPI graphicsAPI, const VideoConfig& config, uint32_t codedWidth, uint32_t codedHeight) {
    CodecParameters params = {};

    const bool isCropped = codedWidth != config.videoWidth || codedHeight != config.videoHeight;
    const uint32_t referenceNum = std::max(config.referenceNum, 1u);

    // H.264
    params.h264Sps.flags = nri::VideoH264SequenceParameterSetBits::DIRECT_8X8_INFERENCE | nri::VideoH264SequenceParameterSetBits::FRAME_MBS_ONLY;
    if (config.h264TransformBypass)
        params.h264Sps.flags |= nri::VideoH264SequenceParameterSetBits::QPPRIME_Y_ZERO_TRANSFORM_BYPASS;
    params.h264Sps.profileIdc = 100;
    params.h264Sps.levelIdc = GetH264LevelIdc(codedWidth, codedHeight);
    params.h264Sps.chromaFormatIdc = 1;
    params.h264Sps.sequenceParameterSetId = config.h264SpsId;
    params.h264Sps.log2MaxFrameNumMinus4 = 4;
    params.h264Sps.pictureOrderCountType = 0;
    params.h264Sps.log2MaxPictureOrderCountLsbMinus4 = 4;
    params.h264Sps.referenceFrameNum = (uint8_t)referenceNum;
    params.h264Sps.pictureWidthInMbsMinus1 = (uint16_t)((codedWidth + 15) / 16 - 1);
    params.h264Sps.pictureHeightInMapUnitsMinus1 = (uint16_t)((codedHeight + 15) / 16 - 1);
    // The coded size can be smaller than the macroblock-aligned picture (D3D12 codes the video size)
    const uint32_t h264AlignedWidth = (codedWidth + 15) & ~15u;
    const uint32_t h264AlignedHeight = (codedHeight + 15) & ~15u;
    if (h264AlignedWidth != config.videoWidth || h264AlignedHeight != config.videoHeight) {
        // CropUnitX/Y are 2 luma samples (4:2:0, frame_mbs_only)
        params.h264Sps.flags |= nri::VideoH264SequenceParameterSetBits::FRAME_CROPPING;
        params.h264Sps.frameCropRightOffset = (uint16_t)((h264AlignedWidth - config.videoWidth) / 2);
        params.h264Sps.frameCropBottomOffset = (uint16_t)((h264AlignedHeight - config.videoHeight) / 2);
    }

    params.h264Pps.flags = nri::VideoH264PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT;
    params.h264Pps.sequenceParameterSetId = config.h264SpsId;
    params.h264Pps.pictureParameterSetId = config.h264PpsId;
    params.h264Pps.refIndexL0DefaultActiveMinus1 = config.h264PpsRefIdxL0DefaultActiveMinus1;
    params.h264Pps.refIndexL1DefaultActiveMinus1 = 0;

    // H.265
    if (config.h265DefaultParameters)
        MakeH265DefaultParameters(params, codedWidth, codedHeight, referenceNum);
    else {
        params.h265Vps.flags = nri::VideoH265VideoParameterSetBits::TEMPORAL_ID_NESTING;
        params.h265Vps.maxSubLayersMinus1 = 0;
        params.h265Vps.profileTierLevel.flags = nri::VideoH265ProfileTierLevelBits::FRAME_ONLY_CONSTRAINT;
        params.h265Vps.profileTierLevel.generalProfileIdc = 1;
        params.h265Vps.profileTierLevel.generalLevelIdc = GetH265GeneralLevelIdc(codedWidth, codedHeight);
        params.h265Vps.decPicBufMgr.maxDecPicBufferingMinus1[0] = (uint8_t)std::min(referenceNum + 1, 15u);
        params.h265Vps.decPicBufMgr.maxNumReorderPics[0] = config.hasBFrames ? 1 : 0;

        params.h265Sps.flags = nri::VideoH265SequenceParameterSetBits::TEMPORAL_ID_NESTING | nri::VideoH265SequenceParameterSetBits::AMP_ENABLED | nri::VideoH265SequenceParameterSetBits::SAMPLE_ADAPTIVE_OFFSET_ENABLED;
        params.h265Sps.maxSubLayersMinus1 = params.h265Vps.maxSubLayersMinus1;
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

        params.h265Pps.flags = nri::VideoH265PictureParameterSetBits::CABAC_INIT_PRESENT | nri::VideoH265PictureParameterSetBits::TRANSFORM_SKIP_ENABLED | nri::VideoH265PictureParameterSetBits::CU_QP_DELTA_ENABLED | nri::VideoH265PictureParameterSetBits::SLICE_CHROMA_QP_OFFSETS_PRESENT | nri::VideoH265PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT;
        if (config.h265ListsModification)
            params.h265Pps.flags |= nri::VideoH265PictureParameterSetBits::LISTS_MODIFICATION_PRESENT;
    }
    if (config.h265ScalingListEnabled)
        params.h265Sps.flags |= nri::VideoH265SequenceParameterSetBits::SCALING_LIST_ENABLED;
    params.h265Vps.videoParameterSetId = config.h265VpsId;
    params.h265Sps.videoParameterSetId = config.h265VpsId;
    params.h265Sps.sequenceParameterSetId = config.h265SpsId;
    params.h265Pps.videoParameterSetId = config.h265VpsId;
    params.h265Pps.sequenceParameterSetId = config.h265SpsId;
    params.h265Pps.pictureParameterSetId = config.h265PpsId;
    if (isCropped) {
        // Conformance window offsets are in chroma sample units (2 luma samples)
        params.h265Sps.flags |= nri::VideoH265SequenceParameterSetBits::CONFORMANCE_WINDOW;
        params.h265Sps.confWinRightOffset = (codedWidth - config.videoWidth) / 2;
        params.h265Sps.confWinBottomOffset = (codedHeight - config.videoHeight) / 2;
    }

    // AV1
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

// "src" is in "srcState" before and after the copy
inline bool CopyNv12BufferToTexture(nri::CoreInterface& core, nri::Queue& queue, const Nv12BufferLayout& layout, nri::Buffer& src, nri::AccessStage srcState, PictureSlot& dst, uint32_t width, uint32_t height) {
    return SubmitOneTime(core, queue, [&](nri::CommandBuffer& commandBuffer) {
        nri::BufferBarrierDesc bufferBarrier = {};
        bufferBarrier.buffer = &src;
        bufferBarrier.before = srcState;
        bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};

        nri::TextureBarrierDesc textureBarrier = {};
        textureBarrier.texture = dst.texture;
        textureBarrier.before = dst.state;
        textureBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        textureBarrier.layerOffset = (nri::Dim_t)dst.layer;
        textureBarrier.layerNum = 1;
        textureBarrier.mipNum = 1;
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
        lumaRegion.layerOffset = (nri::Dim_t)dst.layer;
        lumaRegion.planes = nri::PlaneBits::PLANE_0;

        nri::TextureDataLayoutDesc lumaLayout = {};
        lumaLayout.rowPitch = layout.yRowPitchBytes;
        lumaLayout.slicePitch = layout.ySlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, *dst.texture, lumaRegion, src, lumaLayout);

        nri::TextureRegionDesc chromaRegion = {};
        chromaRegion.width = (nri::Dim_t)width;
        chromaRegion.height = (nri::Dim_t)height;
        chromaRegion.depth = 1;
        chromaRegion.layerOffset = (nri::Dim_t)dst.layer;
        chromaRegion.planes = nri::PlaneBits::PLANE_1;

        nri::TextureDataLayoutDesc chromaLayout = {};
        chromaLayout.offset = layout.uvOffsetBytes;
        chromaLayout.rowPitch = layout.uvRowPitchBytes;
        chromaLayout.slicePitch = layout.uvSlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, *dst.texture, chromaRegion, src, chromaLayout);

        textureBarrier.before = textureBarrier.after;
        textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
        bufferBarrier.before = bufferBarrier.after;
        bufferBarrier.after = srcState;
        core.CmdBarrier(commandBuffer, barrierDesc);
        dst.state = textureBarrier.after;
        dst.consumerState = textureBarrier.after;
    });
}

inline nri::Result CreateEncodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_READBACK, priority, bufferDesc, buffer);
}

inline nri::Result CreateDecodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_UPLOAD, priority, bufferDesc, buffer);
}

// Creates "slotNum" DPB pictures: layers of one texture or separate textures
inline bool CreatePictureSlots(NRIInterface& nri, nri::VideoInterface& video, nri::Device& device, const nri::TextureDesc& textureDesc, nri::VideoPictureUsage usage, bool useArray, uint32_t slotNum, const char* name, PictureSlot* slots, std::vector<nri::Texture*>& textures, uint32_t pictureWidth = 0, uint32_t pictureHeight = 0) {
    nri::TextureDesc desc = textureDesc;
    nri::Texture* arrayTexture = nullptr;
    if (useArray) {
        desc.layerNum = (nri::Dim_t)slotNum;
        nri::Texture* texture = nullptr;
        if (nri.CreateCommittedTexture(device, nri::MemoryLocation::DEVICE, 0.0f, desc, texture) != nri::Result::SUCCESS)
            return false;
        nri.SetDebugName(texture, name);
        textures.push_back(texture);
        arrayTexture = texture;
    }

    for (uint32_t i = 0; i < slotNum; i++) {
        PictureSlot& slot = slots[i];
        if (useArray) {
            slot.texture = arrayTexture;
            slot.layer = i;
        } else {
            desc.layerNum = 1;
            if (nri.CreateCommittedTexture(device, nri::MemoryLocation::DEVICE, 0.0f, desc, slot.texture) != nri::Result::SUCCESS)
                return false;
            nri.SetDebugName(slot.texture, name);
            textures.push_back(slot.texture);
            slot.layer = 0;
        }

        nri::VideoPictureDesc pictureDesc = {};
        pictureDesc.texture = slot.texture;
        pictureDesc.usage = usage;
        pictureDesc.width = (nri::Dim_t)(pictureWidth ? pictureWidth : textureDesc.width);
        pictureDesc.height = (nri::Dim_t)(pictureHeight ? pictureHeight : textureDesc.height);
        pictureDesc.layer = (nri::Dim_t)slot.layer;
        if (video.CreateVideoPicture(device, pictureDesc, slot.picture) != nri::Result::SUCCESS)
            return false;
    }
    return true;
}

// Moves new textures from "UNDEFINED" to the resting state
inline bool InitializePictureLayouts(nri::CoreInterface& core, nri::Queue& queue, const std::vector<nri::Texture*>& textures) {
    return SubmitOneTime(core, queue, [&](nri::CommandBuffer& commandBuffer) {
        std::vector<nri::TextureBarrierDesc> barriers(textures.size());
        for (size_t i = 0; i < textures.size(); i++) {
            barriers[i] = {};
            barriers[i].texture = textures[i];
            barriers[i].before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
            barriers[i].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            barriers[i].mipNum = nri::REMAINING;
            barriers[i].layerNum = nri::REMAINING;
            barriers[i].planes = nri::PlaneBits::ALL;
        }

        nri::BarrierDesc barrierDesc = {};
        barrierDesc.textures = barriers.data();
        barrierDesc.textureNum = (uint32_t)barriers.size();
        core.CmdBarrier(commandBuffer, barrierDesc);
    });
}

// "ALL" by default; "COLOR" and "PLANE_0 | PLANE_1" cover the other ways to address both planes of an NV12 picture
inline nri::PlaneBits g_PicturePlanes = nri::PlaneBits::ALL;

inline nri::TextureBarrierDesc MakePictureBarrier(const PictureSlot& slot, nri::AccessLayoutStage before, nri::AccessLayoutStage after) {
    nri::TextureBarrierDesc barrier = {};
    barrier.texture = slot.texture;
    barrier.before = before;
    barrier.after = after;
    barrier.layerOffset = (nri::Dim_t)slot.layer;
    barrier.layerNum = 1;
    barrier.mipNum = 1;
    barrier.planes = g_PicturePlanes;
    return barrier;
}

// Updates the tracked state after an operation recorded on a video queue
inline void ApplyVideoPictureState(PictureSlot& slot, const nri::VideoPictureState& state) {
    if (state.transitionOnVideoQueue) {
        slot.state = state.videoQueueAfter;
        slot.consumerState = state.consumerQueueBefore;
    } else {
        slot.state = state.required;
        slot.consumerState = state.consumerQueueBefore;
    }
}

} // namespace video_sample
