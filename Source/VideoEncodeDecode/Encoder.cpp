// © 2021 NVIDIA Corporation

#include "Encoder.h"

#include <algorithm>
#include <cstdio>

namespace video_sample {

static const nri::VideoAV1ReferenceName AV1_REFERENCE_NAMES[] = {
    nri::VideoAV1ReferenceName::LAST,
    nri::VideoAV1ReferenceName::LAST2,
    nri::VideoAV1ReferenceName::LAST3,
    nri::VideoAV1ReferenceName::GOLDEN,
    nri::VideoAV1ReferenceName::BWDREF,
    nri::VideoAV1ReferenceName::ALTREF2,
    nri::VideoAV1ReferenceName::ALTREF,
};

Encoder::~Encoder() {
    if (!m_Context.nri)
        return;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    if (video.DestroyVideoPicture) {
        for (PictureSlot& slot : m_Dpb) {
            if (slot.picture)
                video.DestroyVideoPicture(slot.picture);
        }
        if (m_Input.picture)
            video.DestroyVideoPicture(m_Input.picture);
        if (m_Parameters)
            video.DestroyVideoSessionParameters(m_Parameters);
        if (m_Session)
            video.DestroyVideoSession(m_Session);
    }

    if (m_ResolvedMetadataReadbackBuffer && m_ResolvedMetadataReadbackBuffer != m_ResolvedMetadataBuffer)
        nri.DestroyBuffer(m_ResolvedMetadataReadbackBuffer);
    if (m_ResolvedMetadataBuffer)
        nri.DestroyBuffer(m_ResolvedMetadataBuffer);
    if (m_MetadataBuffer)
        nri.DestroyBuffer(m_MetadataBuffer);
    if (m_BitstreamBuffer)
        nri.DestroyBuffer(m_BitstreamBuffer);
    if (m_BitstreamHeaderUploadBuffer)
        nri.DestroyBuffer(m_BitstreamHeaderUploadBuffer);
    for (nri::Texture* texture : m_Textures)
        nri.DestroyTexture(texture);
}

bool Encoder::Initialize(const VideoContext& context, const VideoConfig& config, const VideoSize& size, const CodecParameters& codecParameters) {
    m_Context = context;
    m_Config = config;
    m_Size = size;
    m_CodecParameters = codecParameters;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    nri::VideoSessionDesc sessionDesc = {};
    sessionDesc.type = nri::VideoSessionType::ENCODE;
    sessionDesc.codec = GetNriCodec(m_Config.codec);
    sessionDesc.format = nri::Format::NV12_UNORM;
    sessionDesc.width = m_Size.codedWidth;
    sessionDesc.height = m_Size.codedHeight;
    sessionDesc.maxReferenceNum = m_Config.referenceNum;

    if (video.GetVideoCapabilities(*m_Context.device, sessionDesc, m_Capabilities) != nri::Result::SUCCESS || m_Config.referenceNum > m_Capabilities.maxReferenceNum) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " encode with " + std::to_string(m_Config.referenceNum) + " reference(s) is unsupported";
        m_Unsupported = true;
        return false;
    }

    m_UseFeedback = m_Config.encodeFeedback && m_Capabilities.encodeFeedbackSupported;
    const bool useArray = m_Config.dpbLayout == DpbLayout::ARRAY || (m_Config.dpbLayout == DpbLayout::AUTO && m_Capabilities.dpbTextureArrayRequired);
    if (m_Capabilities.dpbTextureArrayRequired && !useArray) {
        m_Status = "The backend requires DPB texture arrays";
        m_Unsupported = true;
        return false;
    }

    if (video.CreateVideoSession(*m_Context.device, sessionDesc, m_Session) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to create ") + GetCodecName(m_Config.codec) + " encode session";
        return false;
    }

    nri::VideoH264SessionParametersDesc h264Parameters = {};
    h264Parameters.sequenceParameterSets = &m_CodecParameters.h264Sps;
    h264Parameters.sequenceParameterSetNum = 1;
    h264Parameters.pictureParameterSets = &m_CodecParameters.h264Pps;
    h264Parameters.pictureParameterSetNum = 1;
    h264Parameters.maxSequenceParameterSetNum = 1;
    h264Parameters.maxPictureParameterSetNum = 1;

    nri::VideoH265SessionParametersDesc h265Parameters = {};
    h265Parameters.videoParameterSets = &m_CodecParameters.h265Vps;
    h265Parameters.videoParameterSetNum = 1;
    h265Parameters.sequenceParameterSets = &m_CodecParameters.h265Sps;
    h265Parameters.sequenceParameterSetNum = 1;
    h265Parameters.pictureParameterSets = &m_CodecParameters.h265Pps;
    h265Parameters.pictureParameterSetNum = 1;
    h265Parameters.maxVideoParameterSetNum = 1;
    h265Parameters.maxSequenceParameterSetNum = 1;
    h265Parameters.maxPictureParameterSetNum = 1;

    nri::VideoAV1SessionParametersDesc av1Parameters = {};
    av1Parameters.sequence = m_CodecParameters.av1Sequence;

    nri::VideoSessionParametersDesc parametersDesc = {};
    parametersDesc.session = m_Session;
    parametersDesc.h264Parameters = m_Config.codec == SampleCodec::H264 ? &h264Parameters : nullptr;
    parametersDesc.h265Parameters = (m_Config.codec == SampleCodec::H265 && !m_Config.h265DefaultParameters) ? &h265Parameters : nullptr;
    parametersDesc.av1Parameters = m_Config.codec == SampleCodec::AV1 ? &av1Parameters : nullptr;

    const nri::Result parametersResult = video.CreateVideoSessionParameters(*m_Context.device, parametersDesc, m_Parameters);
    if (parametersResult != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to create ") + GetCodecName(m_Config.codec) + " encode parameters";
        m_Unsupported = parametersResult == nri::Result::UNSUPPORTED;
        return false;
    }

    // Source
    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::VIDEO_ENCODE;
    if (m_Config.encodeInputOutputOnly)
        textureDesc.usage |= nri::TextureUsageBits::VIDEO_OUTPUT_ONLY;
    textureDesc.format = nri::Format::NV12_UNORM;
    textureDesc.width = (nri::Dim_t)m_Size.codedWidth;
    textureDesc.height = (nri::Dim_t)m_Size.codedHeight;
    textureDesc.mipNum = 1;
    textureDesc.layerNum = 1;
    textureDesc.videoCodec = GetNriCodec(m_Config.codec);

    if (!CreatePictureSlots(nri, video, *m_Context.device, textureDesc, nri::VideoPictureUsage::ENCODE_INPUT, false, 1, "VideoEncodeInput", &m_Input, m_Textures)) {
        m_Status = "Failed to create NV12 encode source";
        return false;
    }

    // Reconstructed pictures (DPB)
    textureDesc.usage = nri::TextureUsageBits::VIDEO_ENCODE | nri::TextureUsageBits::VIDEO_REFERENCE_ONLY;
    m_DpbSlotNum = std::max(m_Config.referenceNum + 1, useArray ? m_Capabilities.dpbTextureArrayMinLayerNum : 0u);
    m_DpbSlotNum = std::min(m_DpbSlotNum, MAX_DPB_SLOT_NUM);
    if (!CreatePictureSlots(nri, video, *m_Context.device, textureDesc, nri::VideoPictureUsage::ENCODE_REFERENCE, useArray, m_DpbSlotNum, "VideoEncodeDpb", m_Dpb, m_Textures)) {
        m_Status = "Failed to create NV12 reconstructed pictures";
        return false;
    }

    if (!InitializePictureLayouts(nri, *m_Context.graphicsQueue, m_Textures)) {
        m_Status = "Failed to initialize encode texture layouts";
        return false;
    }

    // Buffers
    nri::BufferDesc bitstreamHeaderUploadBufferDesc = {};
    bitstreamHeaderUploadBufferDesc.size = ENCODED_SLICE_OFFSET;

    nri::BufferDesc bitstreamBufferDesc = {};
    bitstreamBufferDesc.size = m_Size.bitstreamSize;
    bitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc metadataBufferDesc = {};
    metadataBufferDesc.size = std::max<uint64_t>(m_Capabilities.metadataSize, 4096);
    metadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc resolvedMetadataBufferDesc = {};
    resolvedMetadataBufferDesc.size = std::max<uint64_t>(m_Capabilities.resolvedMetadataSize, RESOLVED_METADATA_SIZE);
    resolvedMetadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::HOST_UPLOAD, 0.0f, bitstreamHeaderUploadBufferDesc, m_BitstreamHeaderUploadBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create bitstream header upload buffer";
        return false;
    }
    if (CreateEncodeBitstreamBuffer(nri, *m_Context.device, 0.0f, bitstreamBufferDesc, m_BitstreamBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create encode bitstream buffer";
        return false;
    }
    if (m_Capabilities.metadataSize && nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, metadataBufferDesc, m_MetadataBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create encode metadata buffer";
        return false;
    }
    {
        // VK: the query index is recorded in the host-visible buffer. D3D12: resolved on the device, then copied
        if (m_Context.graphicsAPI == nri::GraphicsAPI::VK) {
            if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::HOST_READBACK, 0.0f, resolvedMetadataBufferDesc, m_ResolvedMetadataBuffer) != nri::Result::SUCCESS) {
                m_Status = "Failed to create resolved encode metadata buffer";
                return false;
            }
            m_ResolvedMetadataReadbackBuffer = m_ResolvedMetadataBuffer;
        } else {
            nri::BufferDesc readbackDesc = {};
            readbackDesc.size = resolvedMetadataBufferDesc.size;
            if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, resolvedMetadataBufferDesc, m_ResolvedMetadataBuffer) != nri::Result::SUCCESS || nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::HOST_READBACK, 0.0f, readbackDesc, m_ResolvedMetadataReadbackBuffer) != nri::Result::SUCCESS) {
                m_Status = "Failed to create resolved encode metadata buffers";
                return false;
            }
        }

        if (m_UseFeedback && m_Capabilities.encodeFeedbackResolveRequired && nri.GetQueue(*m_Context.device, m_Capabilities.resolvedMetadataQueueType, 0, m_ResolveQueue) != nri::Result::SUCCESS) {
            m_Status = "The resolved metadata queue is unavailable";
            m_Unsupported = true;
            return false;
        }
    }

    m_Ready = true;
    m_Status = std::string(GetCodecName(m_Config.codec)) + " encoder initialized";
    return true;
}

bool Encoder::UploadHeaders(const FramePlan& plan, std::vector<uint8_t>& headers) {
    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;
    const bool isKeyFrame = plan.type == nri::VideoFrameType::IDR;

    headers.clear();

    if (m_Config.codec == SampleCodec::AV1) {
        // Temporal delimiter, sequence header on key frames. Not written into the buffer, "VideoEncodeDesc::bitstreamMetadataSize" bytes are skipped
        if (isKeyFrame) {
            nri::VideoAV1ObuHeadersDesc obuDesc = {};
            obuDesc.sequence = m_CodecParameters.av1Sequence;
            if (video.WriteVideoAV1ObuHeaders(obuDesc) != nri::Result::SUCCESS || !obuDesc.writtenSize) {
                m_Status = "Failed to query AV1 OBU headers size";
                return false;
            }
            headers.resize((size_t)obuDesc.writtenSize);
            obuDesc.dst = headers.data();
            obuDesc.dstSize = headers.size();
            if (video.WriteVideoAV1ObuHeaders(obuDesc) != nri::Result::SUCCESS) {
                m_Status = "Failed to build AV1 OBU headers";
                return false;
            }
        } else
            headers = {0x12, 0x00};

        void* headerPtr = nri.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        if (!headerPtr) {
            m_Status = "Failed to map bitstream header upload buffer";
            return false;
        }
        std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
        nri.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);
        return true;
    }

    nri::VideoAnnexBParameterSetsDesc annexBDesc = {};
    annexBDesc.codec = GetNriCodec(m_Config.codec);
    annexBDesc.h264Sps = &m_CodecParameters.h264Sps;
    annexBDesc.h264Pps = &m_CodecParameters.h264Pps;
    annexBDesc.h265Vps = &m_CodecParameters.h265Vps;
    annexBDesc.h265Sps = &m_CodecParameters.h265Sps;
    annexBDesc.h265Pps = &m_CodecParameters.h265Pps;

    std::vector<uint8_t> annexBHeaders;
    if (video.WriteVideoAnnexBParameterSets(annexBDesc) != nri::Result::SUCCESS || annexBDesc.writtenSize == 0 || annexBDesc.writtenSize >= ENCODED_SLICE_OFFSET) {
        m_Status = std::string("Failed to query ") + GetCodecName(m_Config.codec) + " Annex-B parameter-set size";
        return false;
    }

    annexBHeaders.resize((size_t)annexBDesc.writtenSize);
    annexBDesc.dst = annexBHeaders.data();
    annexBDesc.dstSize = annexBHeaders.size();
    if (video.WriteVideoAnnexBParameterSets(annexBDesc) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to build ") + GetCodecName(m_Config.codec) + " Annex-B parameter sets";
        return false;
    }

    void* headerPtr = nri.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
    if (!headerPtr) {
        m_Status = "Failed to map bitstream header upload buffer";
        return false;
    }
    std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
    std::memcpy(headerPtr, annexBHeaders.data(), annexBHeaders.size());
    nri.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);

    if (isKeyFrame)
        headers = annexBHeaders;
    return true;
}

bool Encoder::ResolveFeedback(nri::VideoEncodeFeedback& feedback, bool& feedbackResolved) {
    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;
    feedbackResolved = false;

    if (m_Capabilities.encodeFeedbackResolveRequired) {
        const bool onEncodeQueue = m_Capabilities.resolvedMetadataQueueType == nri::QueueType::VIDEO_ENCODE;
        if (!SubmitOneTime(nri, *m_ResolveQueue, [&](nri::CommandBuffer& commandBuffer) {
                nri::BufferBarrierDesc bufferBarriers[2] = {};
                uint32_t bufferBarrierNum = 0;
                bufferBarriers[bufferBarrierNum].buffer = m_ResolvedMetadataBuffer;
                bufferBarriers[bufferBarrierNum++].after = m_Capabilities.resolvedMetadataState;
                if (onEncodeQueue && m_MetadataBuffer) {
                    bufferBarriers[bufferBarrierNum].buffer = m_MetadataBuffer;
                    bufferBarriers[bufferBarrierNum++].after = {nri::AccessBits::VIDEO_ENCODE_READ, nri::StageBits::VIDEO_ENCODE};
                }

                nri::BarrierDesc barrierDesc = {};
                barrierDesc.buffers = bufferBarriers;
                barrierDesc.bufferNum = bufferBarrierNum;
                nri.CmdBarrier(commandBuffer, barrierDesc);
                m_Context.video->CmdResolveVideoEncodeFeedback(commandBuffer, *m_Session, *m_ResolvedMetadataBuffer, 0);
                for (uint32_t i = 0; i < bufferBarrierNum; i++) {
                    bufferBarriers[i].before = bufferBarriers[i].after;
                    bufferBarriers[i].after = {};
                }
                nri.CmdBarrier(commandBuffer, barrierDesc);
            })) {
            m_Status = "Encode feedback resolve submission failed";
            return false;
        }

        if (m_ResolvedMetadataReadbackBuffer != m_ResolvedMetadataBuffer) {
            if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
                    nri::BufferBarrierDesc bufferBarriers[2] = {};
                    bufferBarriers[0].buffer = m_ResolvedMetadataBuffer;
                    bufferBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
                    bufferBarriers[1].buffer = m_ResolvedMetadataReadbackBuffer;
                    bufferBarriers[1].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

                    nri::BarrierDesc barrierDesc = {};
                    barrierDesc.buffers = bufferBarriers;
                    barrierDesc.bufferNum = helper::GetCountOf(bufferBarriers);
                    nri.CmdBarrier(commandBuffer, barrierDesc);
                    nri.CmdCopyBuffer(commandBuffer, *m_ResolvedMetadataReadbackBuffer, 0, *m_ResolvedMetadataBuffer, 0, nri.GetBufferDesc(*m_ResolvedMetadataBuffer).size);
                    for (nri::BufferBarrierDesc& bufferBarrier : bufferBarriers) {
                        bufferBarrier.before = bufferBarrier.after;
                        bufferBarrier.after = {};
                    }
                    nri.CmdBarrier(commandBuffer, barrierDesc);
                })) {
                m_Status = "Encode feedback readback submission failed";
                return false;
            }
        }
    }

    const nri::Result result = video.GetVideoEncodeFeedback(*m_Session, *m_ResolvedMetadataReadbackBuffer, 0, feedback);
    if (result != nri::Result::SUCCESS) {
        m_Status = result == nri::Result::UNSUPPORTED ? std::string(GetCodecName(m_Config.codec)) + " encode metadata feedback is unsupported" : "Failed to read resolved encode metadata";
        return false;
    }

    feedbackResolved = true;
    return true;
}

// No feedback: the bitstream buffer is zeroed before each frame, so the payload ends at the last non-zero byte
bool Encoder::ScanFeedback(nri::VideoEncodeFeedback& feedback) {
    NRIInterface& nri = *m_Context.nri;
    const uint8_t* bitstream = (const uint8_t*)nri.MapBuffer(*m_BitstreamBuffer, ENCODED_SLICE_OFFSET, m_Size.bitstreamSize - ENCODED_SLICE_OFFSET);
    if (!bitstream) {
        m_Status = "Failed to map the encode bitstream";
        return false;
    }

    uint64_t size = m_Size.bitstreamSize - ENCODED_SLICE_OFFSET;
    while (size && !bitstream[size - 1])
        size--;
    nri.UnmapBuffer(*m_BitstreamBuffer);

    feedback = {};
    feedback.encodedBitstreamWrittenBytes = size;
    feedback.writtenSubregionNum = 1;
    return true;
}

bool Encoder::BuildAv1DecodeInfo(EncodedFrame& frame, const nri::VideoEncodePictureDesc& pictureDesc, const nri::VideoAV1EncodePictureDesc& av1PictureDesc, const nri::VideoAV1ReferenceDesc* av1References, uint32_t av1ReferenceNum) {
    nri::VideoInterface& video = *m_Context.video;

    nri::VideoAV1EncodeDecodeInfoDesc av1InfoDesc = {};
    av1InfoDesc.feedback = &frame.feedback;
    av1InfoDesc.sequence = &m_CodecParameters.av1Sequence;
    av1InfoDesc.references = av1References;
    av1InfoDesc.referenceNum = av1ReferenceNum;
    av1InfoDesc.pictureDesc = &pictureDesc;
    av1InfoDesc.av1PictureDesc = &av1PictureDesc;

    Av1InfoMode mode = m_Config.av1Info;
    if (mode == Av1InfoMode::AUTO)
        mode = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? Av1InfoMode::HEADER : Av1InfoMode::METADATA;

    nri::Result result = nri::Result::UNSUPPORTED;
    if (mode == Av1InfoMode::METADATA) {
        // Submitted descriptions only, no payload header
        result = video.GetVideoAV1EncodeDecodeInfo(*m_Session, *m_ResolvedMetadataReadbackBuffer, 0, av1InfoDesc, frame.av1DecodeInfo);
        if (result == nri::Result::UNSUPPORTED)
            frame.av1MetadataUnsupported = true;
    }

    if (result != nri::Result::SUCCESS) {
        av1InfoDesc.encodedPayloadHeader = frame.payload.data();
        av1InfoDesc.encodedPayloadHeaderSize = frame.payload.size();
        result = video.GetVideoAV1EncodeDecodeInfo(*m_Session, *m_ResolvedMetadataReadbackBuffer, 0, av1InfoDesc, frame.av1DecodeInfo);
    }

    if (result != nri::Result::SUCCESS) {
        m_Status = "Failed to prepare AV1 decode metadata";
        return false;
    }

    frame.hasAv1DecodeInfo = true;
    return true;
}

bool Encoder::Encode(const EncodeRequest& request, EncodedFrame& frame) {
    if (!m_Ready || !request.nv12Buffer || !request.nv12Layout) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " encoder is not ready";
        return false;
    }

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;
    const FramePlan& plan = request.plan;
    const bool isH26 = m_Config.codec != SampleCodec::AV1;
    const bool isKeyFrame = plan.type == nri::VideoFrameType::IDR;
    const uint32_t minQ = m_Config.codec == SampleCodec::AV1 ? 1 : 0;

    frame = {};
    frame.plan = plan;

    // Source
    if (!CopyNv12BufferToTexture(nri, *m_Context.graphicsQueue, *request.nv12Layout, *request.nv12Buffer, request.nv12BufferState, m_Input, request.nv12Width, request.nv12Height)) {
        m_Status = "Failed to upload NV12 source to video texture";
        return false;
    }

    if (!UploadHeaders(plan, frame.headers))
        return false;

    if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri.CmdZeroBuffer(commandBuffer, *m_BitstreamBuffer, 0, m_Size.bitstreamSize);
            nri.CmdCopyBuffer(commandBuffer, *m_BitstreamBuffer, 0, *m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        })) {
        m_Status = std::string("Failed to upload ") + GetCodecName(m_Config.codec) + " Annex-B parameter sets";
        return false;
    }

    // References, in the order of the plan (reversed on odd H.265 pictures to test RPS ordering)
    nri::VideoReference references[MAX_REFERENCE_NUM] = {};
    nri::VideoH264EncodeReferenceDesc h264References[MAX_REFERENCE_NUM] = {};
    nri::VideoH265ReferenceDesc h265References[MAX_REFERENCE_NUM] = {};
    PlanReference orderedReferences[MAX_REFERENCE_NUM] = {};
    for (uint32_t i = 0; i < plan.referenceNum; i++)
        orderedReferences[i] = plan.references[plan.reverseReferenceOrder ? plan.referenceNum - 1 - i : i];
    for (uint32_t i = 0; i < plan.referenceNum; i++) {
        const PlanReference& reference = orderedReferences[i];
        references[i] = {m_Dpb[reference.slot].picture, reference.slot};
        h264References[i] = {reference.type, 0, reference.listIndex, 0, reference.frameNum, reference.poc, reference.slot, 0, 0};
        h265References[i] = {reference.slot, reference.poc, 0, reference.type, 0, reference.listIndex};
    }

    nri::VideoEncodePictureDesc pictureDesc = {};
    pictureDesc.frameType = plan.type;
    pictureDesc.frameIndex = plan.frameNum;
    pictureDesc.pictureOrderCount = plan.poc;
    pictureDesc.idrPictureId = (uint16_t)plan.idrPictureId;

    nri::VideoH264EncodePictureDesc h264PictureDesc = {};
    h264PictureDesc.sequenceParameterSetId = m_CodecParameters.h264Sps.sequenceParameterSetId;
    h264PictureDesc.pictureParameterSetId = m_CodecParameters.h264Pps.pictureParameterSetId;
    h264PictureDesc.references = plan.referenceNum ? h264References : nullptr;
    h264PictureDesc.referenceNum = plan.referenceNum;

    nri::VideoH265EncodePictureDesc h265PictureDesc = {};
    h265PictureDesc.videoParameterSetId = m_CodecParameters.h265Vps.videoParameterSetId;
    h265PictureDesc.sequenceParameterSetId = m_CodecParameters.h265Sps.sequenceParameterSetId;
    h265PictureDesc.pictureParameterSetId = m_CodecParameters.h265Pps.pictureParameterSetId;

    // AV1: all reference names map to the latest reference picture (DPB slots alternate, "refFrameIndex" 0 is always the latest)
    nri::VideoAV1ReferenceDesc av1References[8] = {};
    uint32_t av1ReferenceNum = 0;
    if (m_Config.codec == SampleCodec::AV1 && plan.referenceNum) {
        const PlanReference& reference = plan.references[0];
        for (nri::VideoAV1ReferenceName name : AV1_REFERENCE_NAMES) {
            nri::VideoAV1ReferenceDesc& av1Reference = av1References[av1ReferenceNum++];
            av1Reference.name = name;
            av1Reference.refFrameIndex = 0;
            av1Reference.frameType = reference.type;
            av1Reference.orderHint = reference.orderHint;
            av1Reference.frameId = reference.frameId;
            av1Reference.slot = reference.slot;
        }
    }

    uint16_t av1MiColumnStarts[] = {0, (uint16_t)(2 * ((m_Size.codedWidth + 7) >> 3))};
    uint16_t av1MiRowStarts[] = {0, (uint16_t)(2 * ((m_Size.codedHeight + 7) >> 3))};
    uint16_t av1WidthInSuperblocksMinus1[] = {(uint16_t)(((m_Size.codedWidth + 63) / 64) - 1)};
    uint16_t av1HeightInSuperblocksMinus1[] = {(uint16_t)(((m_Size.codedHeight + 63) / 64) - 1)};
    nri::VideoAV1TileLayoutDesc av1TileLayout = {};
    av1TileLayout.columnNum = 1;
    av1TileLayout.rowNum = 1;
    av1TileLayout.tileSizeBytesMinus1 = 3;
    av1TileLayout.uniformSpacing = 1;
    av1TileLayout.miColumnStarts = av1MiColumnStarts;
    av1TileLayout.miRowStarts = av1MiRowStarts;
    av1TileLayout.widthInSuperblocksMinus1 = av1WidthInSuperblocksMinus1;
    av1TileLayout.heightInSuperblocksMinus1 = av1HeightInSuperblocksMinus1;
    nri::VideoAV1LoopFilterDesc av1LoopFilter = {};
    av1LoopFilter.refDeltas[0] = 1;
    av1LoopFilter.refDeltas[4] = -1;
    av1LoopFilter.refDeltas[6] = -1;
    av1LoopFilter.refDeltas[7] = -1;
    nri::VideoAV1CdefDesc av1Cdef = {};
    nri::VideoAV1LoopRestorationDesc av1LoopRestoration = {};
    nri::VideoAV1GlobalMotionDesc av1GlobalMotion = {};
    for (auto& params : av1GlobalMotion.params) {
        params[2] = 1 << 16;
        params[5] = 1 << 16;
    }

    const bool isVK = m_Context.graphicsAPI == nri::GraphicsAPI::VK;
    const bool av1LosslessFallback = m_Config.codec == SampleCodec::AV1 && request.quality.lossless;
    nri::VideoAV1PictureBits av1CommonPictureFlags = nri::VideoAV1PictureBits::SHOW_FRAME;
    if (m_Context.graphicsAPI == nri::GraphicsAPI::D3D12)
        av1CommonPictureFlags |= nri::VideoAV1PictureBits::SEGMENTATION_ENABLED;

    nri::VideoAV1EncodePictureDesc av1PictureDesc = {};
    av1PictureDesc.currentFrameId = plan.frameNum;
    av1PictureDesc.orderHint = plan.av1OrderHint;
    av1PictureDesc.refreshFrameFlags = plan.av1RefreshFrameFlags;
    av1PictureDesc.primaryReferenceName = isKeyFrame ? nri::VideoAV1ReferenceName::NONE : nri::VideoAV1ReferenceName::LAST;
    av1PictureDesc.flags = isKeyFrame ? av1CommonPictureFlags | nri::VideoAV1PictureBits::ERROR_RESILIENT_MODE : av1CommonPictureFlags;
    av1PictureDesc.renderWidthMinus1 = (uint16_t)((isVK ? m_Size.codedWidth : m_Size.videoWidth) - 1);
    av1PictureDesc.renderHeightMinus1 = (uint16_t)((isVK ? m_Size.codedHeight : m_Size.videoHeight) - 1);
    av1PictureDesc.baseQIndex = (uint8_t)(request.quality.lossless ? 1 : std::max(request.quality.av1BaseQIndex, minQ));
    av1PictureDesc.interpolationFilter = 0;
    av1PictureDesc.txMode = 2;
    av1PictureDesc.cdefDampingMinus3 = 3;
    av1PictureDesc.tileLayout = isVK ? nullptr : &av1TileLayout;
    av1PictureDesc.loopFilter = isVK ? nullptr : &av1LoopFilter;
    av1PictureDesc.cdef = isVK ? nullptr : &av1Cdef;
    av1PictureDesc.loopRestoration = isVK ? nullptr : &av1LoopRestoration;
    av1PictureDesc.globalMotion = isVK ? nullptr : &av1GlobalMotion;
    av1PictureDesc.references = av1ReferenceNum ? av1References : nullptr;
    av1PictureDesc.referenceNum = av1ReferenceNum;

    // Rate control
    const RateControlSetting& rateControl = request.quality.rateControl;
    auto getQp = [&](uint32_t qp) {
        return (uint8_t)(request.quality.lossless ? (av1LosslessFallback ? 1 : 0) : std::max(qp, minQ));
    };
    nri::VideoEncodeRateControlDesc rateControlDesc = {};
    rateControlDesc.mode = rateControl.kind == RateControlKind::CBR ? nri::VideoEncodeRateControlMode::CBR : (rateControl.kind == RateControlKind::VBR ? nri::VideoEncodeRateControlMode::VBR : nri::VideoEncodeRateControlMode::CQP);
    rateControlDesc.qpI = getQp(request.quality.qpI);
    rateControlDesc.qpP = getQp(request.quality.qpP);
    rateControlDesc.qpB = getQp(request.quality.qpB);
    rateControlDesc.frameRateNumerator = 30;
    rateControlDesc.frameRateDenominator = 1;
    if (rateControl.kind != RateControlKind::CQP) {
        rateControlDesc.targetBitrate = uint64_t(rateControl.bitrateKbps) * 1000;
        rateControlDesc.maxBitrate = uint64_t(rateControl.maxBitrateKbps ? rateControl.maxBitrateKbps : rateControl.bitrateKbps) * 1000;
    }

    // Encode
    // Non-reference pictures (B frames, AV1 "refreshFrameFlags = 0") still write the setup slot
    PictureSlot* reconstructed = &m_Dpb[plan.setupSlot];

    nri::VideoEncodeDesc encodeDesc = {};
    encodeDesc.session = m_Session;
    encodeDesc.parameters = m_Parameters;
    encodeDesc.srcPicture = m_Input.picture;
    encodeDesc.dstBitstream.buffer = m_BitstreamBuffer;
    encodeDesc.dstBitstream.offset = ENCODED_SLICE_OFFSET;
    encodeDesc.dstBitstream.size = m_Size.bitstreamSize - ENCODED_SLICE_OFFSET;
    encodeDesc.bitstreamMetadataSize = ENCODED_SLICE_OFFSET;
    encodeDesc.pictureDesc = &pictureDesc;
    encodeDesc.rateControlDesc = &rateControlDesc;
    encodeDesc.reconstructedPicture = reconstructed ? reconstructed->picture : nullptr;
    encodeDesc.reconstructedSlot = plan.setupSlot;
    encodeDesc.references = plan.referenceNum ? references : nullptr;
    encodeDesc.referenceNum = plan.referenceNum;
    encodeDesc.metadata = m_MetadataBuffer;
    encodeDesc.resolvedMetadata = m_UseFeedback ? m_ResolvedMetadataBuffer : nullptr;
    encodeDesc.h264PictureDesc = m_Config.codec == SampleCodec::H264 ? &h264PictureDesc : nullptr;
    encodeDesc.h265PictureDesc = m_Config.codec == SampleCodec::H265 ? &h265PictureDesc : nullptr;
    encodeDesc.h265ReferenceDescs = (m_Config.codec == SampleCodec::H265 && plan.referenceNum) ? h265References : nullptr;
    encodeDesc.av1PictureDesc = m_Config.codec == SampleCodec::AV1 ? &av1PictureDesc : nullptr;

    // Picture states
    PictureSlot* pictures[2 + MAX_REFERENCE_NUM] = {};
    nri::VideoPictureState states[2 + MAX_REFERENCE_NUM] = {};
    uint32_t pictureNum = 0;
    bool statesOk = true;
    auto addPicture = [&](PictureSlot* slot, nri::VideoPictureRole role) {
        statesOk = statesOk && video.GetVideoPictureState(*slot->picture, role, states[pictureNum]) == nri::Result::SUCCESS;
        pictures[pictureNum++] = slot;
    };
    addPicture(&m_Input, nri::VideoPictureRole::ENCODE_INPUT);
    if (reconstructed)
        addPicture(reconstructed, nri::VideoPictureRole::ENCODE_RECONSTRUCTED);
    for (uint32_t i = 0; i < plan.referenceNum; i++)
        addPicture(&m_Dpb[plan.references[i].slot], nri::VideoPictureRole::ENCODE_REFERENCE);
    if (!statesOk) {
        m_Status = "Failed to query video encode picture states";
        return false;
    }

    if (!SubmitOneTime(nri, *m_Context.encodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::BufferBarrierDesc bufferBarrier = {};
            bufferBarrier.buffer = m_MetadataBuffer;
            bufferBarrier.after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};

            nri::TextureBarrierDesc textureBarriers[2 + MAX_REFERENCE_NUM] = {};
            for (uint32_t i = 0; i < pictureNum; i++)
                textureBarriers[i] = MakePictureBarrier(*pictures[i], pictures[i]->state, states[i].required);

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = m_MetadataBuffer ? &bufferBarrier : nullptr;
            barrierDesc.bufferNum = m_MetadataBuffer ? 1 : 0;
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = pictureNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);

            video.CmdEncodeVideo(commandBuffer, encodeDesc);

            bufferBarrier.before = bufferBarrier.after;
            bufferBarrier.after = {};
            uint32_t postTextureNum = 0;
            for (uint32_t i = 0; i < pictureNum; i++) {
                if (!states[i].transitionOnVideoQueue)
                    continue;
                textureBarriers[postTextureNum++] = MakePictureBarrier(*pictures[i], states[i].required, states[i].videoQueueAfter);
            }
            barrierDesc.textureNum = postTextureNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " encode submission failed";
        return false;
    }
    for (uint32_t i = 0; i < pictureNum; i++)
        ApplyVideoPictureState(*pictures[i], states[i]);

    // Feedback
    if (m_UseFeedback) {
        if (!ResolveFeedback(frame.feedback, frame.feedbackResolved))
            return false;
    } else if (!ScanFeedback(frame.feedback))
        return false;

    if (frame.feedback.errorFlags || !frame.feedback.encodedBitstreamWrittenBytes) {
        char message[160] = {};
        std::snprintf(message, sizeof(message), "Encoder returned errorFlags=0x%llX bytes=%llu",
            (unsigned long long)frame.feedback.errorFlags, (unsigned long long)frame.feedback.encodedBitstreamWrittenBytes);
        m_Status = message;
        return false;
    }

    // Payload
    const uint64_t payloadOffset = ENCODED_SLICE_OFFSET + frame.feedback.encodedBitstreamOffset;
    const uint64_t payloadSize = frame.feedback.encodedBitstreamWrittenBytes;
    if (payloadOffset > m_Size.bitstreamSize || payloadSize > m_Size.bitstreamSize - payloadOffset) {
        m_Status = "Encoded bitstream range exceeds the bitstream buffer";
        return false;
    }
    const uint8_t* payload = (const uint8_t*)nri.MapBuffer(*m_BitstreamBuffer, payloadOffset, payloadSize);
    if (!payload) {
        m_Status = "Failed to map the encode bitstream";
        return false;
    }
    frame.payload.assign(payload, payload + payloadSize);
    nri.UnmapBuffer(*m_BitstreamBuffer);

    if (m_Config.codec == SampleCodec::AV1) {
        if (!BuildAv1DecodeInfo(frame, pictureDesc, av1PictureDesc, av1ReferenceNum ? av1References : nullptr, av1ReferenceNum))
            return false;
    }
    (void)isH26;

    m_Status = std::string(GetCodecName(m_Config.codec)) + " encode complete";
    return true;
}

} // namespace video_sample
