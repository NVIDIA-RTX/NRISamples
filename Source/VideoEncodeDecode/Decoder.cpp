// © 2021 NVIDIA Corporation

#include "Decoder.h"

#include <cstdio>

namespace video_sample {

Decoder::~Decoder() {
    if (!m_Context.nri)
        return;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    if (video.DestroyVideoPicture) {
        for (PictureSlot& slot : m_Dpb) {
            if (slot.picture)
                video.DestroyVideoPicture(slot.picture);
        }
        if (m_Output.picture)
            video.DestroyVideoPicture(m_Output.picture);
        if (m_Parameters)
            video.DestroyVideoSessionParameters(m_Parameters);
        if (m_Session)
            video.DestroyVideoSession(m_Session);
    }

    if (m_DecodeBitstreamBuffer)
        nri.DestroyBuffer(m_DecodeBitstreamBuffer);
    for (nri::Texture* texture : m_Textures)
        nri.DestroyTexture(texture);
}

bool Decoder::Initialize(const VideoContext& context, const VideoConfig& config, const VideoSize& size, const CodecParameters& codecParameters) {
    m_Context = context;
    m_Config = config;
    m_Size = size;
    m_CodecParameters = codecParameters;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    nri::VideoSessionDesc sessionDesc = {};
    sessionDesc.type = nri::VideoSessionType::DECODE;
    sessionDesc.codec = GetNriCodec(m_Config.codec);
    sessionDesc.format = nri::Format::NV12_UNORM;
    sessionDesc.width = m_Size.codedWidth;
    sessionDesc.height = m_Size.codedHeight;
    sessionDesc.maxReferenceNum = m_Config.referenceNum;

    if (video.GetVideoCapabilities(*m_Context.device, sessionDesc, m_Capabilities) != nri::Result::SUCCESS || m_Config.referenceNum > m_Capabilities.maxReferenceNum) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " decode with " + std::to_string(m_Config.referenceNum) + " reference(s) is unsupported";
        m_Unsupported = true;
        return false;
    }

    // Decode output and DPB setup picture: the same subresource (coincide) or separate pictures (distinct, "VIDEO_OUTPUT_ONLY" output)
    if (m_Config.decodeDpb == DecodeDpbMode::COINCIDE)
        m_IsDistinct = false;
    else if (m_Config.decodeDpb == DecodeDpbMode::DISTINCT)
        m_IsDistinct = true;
    else
        m_IsDistinct = !m_Capabilities.decodeDpbAndOutputCoincide;

    if (m_IsDistinct ? !m_Capabilities.decodeDpbAndOutputDistinct : !m_Capabilities.decodeDpbAndOutputCoincide) {
        m_Status = std::string("Decode DPB and output ") + (m_IsDistinct ? "distinct" : "coincide") + " mode is unsupported";
        m_Unsupported = true;
        return false;
    }

    if (!(m_Capabilities.decodeBitstreamSourceMask & nri::VideoDecodeBitstreamSourceBits::BUFFER)) {
        m_Status = "Decode from a bitstream buffer is unsupported";
        m_Unsupported = true;
        return false;
    }

    const bool useArray = m_Config.dpbLayout == DpbLayout::ARRAY || (m_Config.dpbLayout == DpbLayout::AUTO && m_Capabilities.dpbTextureArrayRequired);
    if (m_Capabilities.dpbTextureArrayRequired && !useArray) {
        m_Status = "The backend requires DPB texture arrays";
        m_Unsupported = true;
        return false;
    }

    if (video.CreateVideoSession(*m_Context.device, sessionDesc, m_Session) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to create ") + GetCodecName(m_Config.codec) + " decode session";
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
    parametersDesc.h265Parameters = m_Config.codec == SampleCodec::H265 ? &h265Parameters : nullptr;
    parametersDesc.av1Parameters = m_Config.codec == SampleCodec::AV1 ? &av1Parameters : nullptr;

    if (video.CreateVideoSessionParameters(*m_Context.device, parametersDesc, m_Parameters) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to create ") + GetCodecName(m_Config.codec) + " decode parameters";
        return false;
    }

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::VIDEO_DECODE;
    textureDesc.format = nri::Format::NV12_UNORM;
    textureDesc.width = (nri::Dim_t)m_Size.codedWidth;
    textureDesc.height = (nri::Dim_t)m_Size.codedHeight;
    textureDesc.mipNum = 1;
    textureDesc.layerNum = 1;
    textureDesc.videoCodec = GetNriCodec(m_Config.codec);

    m_DpbSlotNum = std::max(m_Config.referenceNum + 1, useArray ? m_Capabilities.dpbTextureArrayMinLayerNum : 0u);
    m_DpbSlotNum = std::min(m_DpbSlotNum, MAX_DPB_SLOT_NUM);

    const uint32_t pictureWidth = m_Config.decodePictureAtVideoSize ? m_Size.videoWidth : 0;
    const uint32_t pictureHeight = m_Config.decodePictureAtVideoSize ? m_Size.videoHeight : 0;
    bool isCreated = false;
    if (m_IsDistinct) {
        textureDesc.usage |= nri::TextureUsageBits::VIDEO_REFERENCE_ONLY;
        isCreated = CreatePictureSlots(nri, video, *m_Context.device, textureDesc, nri::VideoPictureUsage::DECODE_REFERENCE, useArray, m_DpbSlotNum, "VideoDecodeDpb", m_Dpb, m_Textures, pictureWidth, pictureHeight);

        textureDesc.usage = nri::TextureUsageBits::VIDEO_DECODE | nri::TextureUsageBits::VIDEO_OUTPUT_ONLY;
        isCreated = isCreated && CreatePictureSlots(nri, video, *m_Context.device, textureDesc, nri::VideoPictureUsage::DECODE_OUTPUT, false, 1, "VideoDecodeOutput", &m_Output, m_Textures, pictureWidth, pictureHeight);
    } else
        isCreated = CreatePictureSlots(nri, video, *m_Context.device, textureDesc, nri::VideoPictureUsage::DECODE_OUTPUT, useArray, m_DpbSlotNum, "VideoDecodeDpb", m_Dpb, m_Textures, pictureWidth, pictureHeight);
    if (!isCreated) {
        m_Status = "Failed to create NV12 decode pictures";
        return false;
    }

    if (!InitializePictureLayouts(nri, *m_Context.graphicsQueue, m_Textures)) {
        m_Status = "Failed to initialize decode texture layouts";
        return false;
    }

    nri::BufferDesc decodeBitstreamBufferDesc = {};
    decodeBitstreamBufferDesc.size = m_Size.bitstreamSize;
    decodeBitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_DECODE;
    if (CreateDecodeBitstreamBuffer(nri, *m_Context.device, 0.0f, decodeBitstreamBufferDesc, m_DecodeBitstreamBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create decode bitstream buffer";
        return false;
    }

    m_Ready = true;
    m_Status = std::string(GetCodecName(m_Config.codec)) + " decoder initialized";
    return true;
}

bool Decoder::WriteAnnexBEndOfStream(std::vector<uint8_t>& annexBEndOfStream) {
    annexBEndOfStream.clear();
    if (m_Config.codec == SampleCodec::AV1)
        return true;

    nri::VideoInterface& video = *m_Context.video;
    nri::VideoAnnexBEndOfStreamDesc annexBDesc = {};
    annexBDesc.codec = GetNriCodec(m_Config.codec);
    if (video.WriteVideoAnnexBEndOfStream(annexBDesc) != nri::Result::SUCCESS || annexBDesc.writtenSize == 0) {
        m_Status = std::string("Failed to query ") + GetCodecName(m_Config.codec) + " Annex-B end-of-stream size";
        return false;
    }

    annexBEndOfStream.resize((size_t)annexBDesc.writtenSize);
    annexBDesc.dst = annexBEndOfStream.data();
    annexBDesc.dstSize = annexBEndOfStream.size();
    if (video.WriteVideoAnnexBEndOfStream(annexBDesc) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to build ") + GetCodecName(m_Config.codec) + " Annex-B end-of-stream marker";
        return false;
    }

    return true;
}

bool Decoder::BuildDecodeBitstream(const EncodedFrame& frame, uint64_t& decodeBitstreamRange, uint32_t& pictureOffset) {
    NRIInterface& nri = *m_Context.nri;

    std::vector<uint8_t> annexBHeaders;
    if (!frame.hasAv1DecodeInfo) {
        nri::VideoInterface& video = *m_Context.video;
        nri::VideoAnnexBParameterSetsDesc annexBDesc = {};
        annexBDesc.codec = GetNriCodec(m_Config.codec);
        annexBDesc.h264Sps = &m_CodecParameters.h264Sps;
        annexBDesc.h264Pps = &m_CodecParameters.h264Pps;
        annexBDesc.h265Vps = &m_CodecParameters.h265Vps;
        annexBDesc.h265Sps = &m_CodecParameters.h265Sps;
        annexBDesc.h265Pps = &m_CodecParameters.h265Pps;
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
    }

    std::vector<uint8_t> annexBEndOfStream;
    if (!WriteAnnexBEndOfStream(annexBEndOfStream))
        return false;
    if (m_Context.graphicsAPI == nri::GraphicsAPI::VK)
        annexBEndOfStream.clear();

    const nri::VideoAV1EncodeDecodeInfo* av1DecodeInfo = frame.hasAv1DecodeInfo ? &frame.av1DecodeInfo : nullptr;
    const uint64_t encodedPayloadSkip = av1DecodeInfo ? av1DecodeInfo->bitstreamOffset : 0;
    const uint64_t encodedPayloadBytes = av1DecodeInfo ? av1DecodeInfo->bitstreamSize : frame.payload.size() - encodedPayloadSkip;
    const uint64_t decodeSliceOffset = annexBHeaders.size();
    pictureOffset = (uint32_t)decodeSliceOffset;
    const uint64_t decodeBitstreamSize = AlignUp(decodeSliceOffset + encodedPayloadBytes + annexBEndOfStream.size(), m_Size.decodeBitstreamSizeAlignment);
    if (encodedPayloadSkip + encodedPayloadBytes > frame.payload.size() || decodeBitstreamSize > m_Size.bitstreamSize) {
        m_Status = std::string("Encoded ") + GetCodecName(m_Config.codec) + " bitstream exceeded decode buffer size";
        return false;
    }

    uint8_t* decodeBitstream = (uint8_t*)nri.MapBuffer(*m_DecodeBitstreamBuffer, 0, decodeBitstreamSize);
    if (!decodeBitstream) {
        m_Status = std::string("Failed to map exact ") + GetCodecName(m_Config.codec) + " decode bitstream";
        return false;
    }
    std::memset(decodeBitstream, 0, (size_t)decodeBitstreamSize);
    if (!annexBHeaders.empty())
        std::memcpy(decodeBitstream, annexBHeaders.data(), annexBHeaders.size());
    std::memcpy(decodeBitstream + decodeSliceOffset, frame.payload.data() + encodedPayloadSkip, (size_t)encodedPayloadBytes);
    if (!annexBEndOfStream.empty())
        std::memcpy(decodeBitstream + decodeSliceOffset + encodedPayloadBytes, annexBEndOfStream.data(), annexBEndOfStream.size());
    decodeBitstreamRange = decodeBitstreamSize;
    nri.UnmapBuffer(*m_DecodeBitstreamBuffer);
    return true;
}

bool Decoder::Decode(const DecodeRequest& request, DecodedFrame& decodedFrame) {
    if (!m_Ready || !request.frame || !request.nv12ReadbackBuffer || !request.nv12Layout) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " decoder is not ready";
        return false;
    }

    const EncodedFrame& frame = *request.frame;
    const FramePlan& plan = frame.plan;
    const nri::VideoAV1EncodeDecodeInfo* av1DecodeInfo = frame.hasAv1DecodeInfo ? &frame.av1DecodeInfo : nullptr;
    const bool isKeyFrame = plan.type == nri::VideoFrameType::IDR;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    uint64_t decodeBitstreamRange = 0;
    uint32_t pictureOffset = 0;
    if (!BuildDecodeBitstream(frame, decodeBitstreamRange, pictureOffset))
        return false;

    PictureSlot& setup = m_Dpb[plan.setupSlot];
    PictureSlot& output = m_IsDistinct ? m_Output : setup;
    const uint32_t pictureOffsets[] = {pictureOffset};

    // References, in the order of the plan (reversed on odd H.265 pictures to test RPS ordering)
    PlanReference orderedReferences[MAX_REFERENCE_NUM] = {};
    nri::VideoReference references[MAX_REFERENCE_NUM] = {};
    nri::VideoH264DecodeReferenceDesc h264References[MAX_REFERENCE_NUM] = {};
    nri::VideoH265ReferenceDesc h265References[MAX_REFERENCE_NUM] = {};
    for (uint32_t i = 0; i < plan.referenceNum; i++)
        orderedReferences[i] = plan.references[plan.reverseReferenceOrder ? plan.referenceNum - 1 - i : i];
    for (uint32_t i = 0; i < plan.referenceNum; i++) {
        const PlanReference& reference = orderedReferences[i];
        references[i] = {m_Dpb[reference.slot].picture, reference.slot};
        h264References[i] = {nri::VideoH264DecodeReferenceBits::TOP_FIELD | nri::VideoH264DecodeReferenceBits::BOTTOM_FIELD, (uint16_t)reference.frameNum, reference.slot, reference.poc, reference.poc};
        h265References[i] = {reference.slot, reference.poc, 0, reference.type, 0, reference.listIndex};
    }

    nri::VideoH264DecodePictureDesc h264DecodePicture = {};
    h264DecodePicture.flags = nri::VideoH264DecodePictureBits::NONE;
    if (isKeyFrame)
        h264DecodePicture.flags |= nri::VideoH264DecodePictureBits::IDR | nri::VideoH264DecodePictureBits::INTRA;
    if (plan.isReference)
        h264DecodePicture.flags |= nri::VideoH264DecodePictureBits::REFERENCE;
    h264DecodePicture.sequenceParameterSetId = m_CodecParameters.h264Sps.sequenceParameterSetId;
    h264DecodePicture.pictureParameterSetId = m_CodecParameters.h264Pps.pictureParameterSetId;
    h264DecodePicture.frameNum = (uint16_t)plan.frameNum;
    h264DecodePicture.idrPictureId = (uint16_t)plan.idrPictureId;
    h264DecodePicture.topFieldOrderCount = plan.poc;
    h264DecodePicture.bottomFieldOrderCount = plan.poc;
    h264DecodePicture.sliceOffsets = pictureOffsets;
    h264DecodePicture.sliceOffsetNum = helper::GetCountOf(pictureOffsets);
    h264DecodePicture.references = plan.referenceNum ? h264References : nullptr;
    h264DecodePicture.referenceNum = plan.referenceNum;

    nri::VideoH265DecodePictureDesc h265DecodePicture = {};
    h265DecodePicture.flags = nri::VideoH265DecodePictureBits::NONE;
    if (isKeyFrame)
        h265DecodePicture.flags |= nri::VideoH265DecodePictureBits::IRAP | nri::VideoH265DecodePictureBits::IDR;
    if (plan.isReference)
        h265DecodePicture.flags |= nri::VideoH265DecodePictureBits::REFERENCE;
    h265DecodePicture.videoParameterSetId = m_CodecParameters.h265Vps.videoParameterSetId;
    h265DecodePicture.sequenceParameterSetId = m_CodecParameters.h265Sps.sequenceParameterSetId;
    h265DecodePicture.pictureParameterSetId = m_CodecParameters.h265Pps.pictureParameterSetId;
    h265DecodePicture.pictureOrderCount = plan.poc;
    h265DecodePicture.sliceSegmentOffsets = pictureOffsets;
    h265DecodePicture.sliceSegmentOffsetNum = helper::GetCountOf(pictureOffsets);
    h265DecodePicture.references = plan.referenceNum ? h265References : nullptr;
    h265DecodePicture.referenceNum = plan.referenceNum;

    nri::VideoAV1EncodeDecodeInfo av1Info = {};
    uint8_t av1DecodeOrderHints[8] = {};
    if (av1DecodeInfo) {
        av1Info = *av1DecodeInfo;
        av1Info.picture.tiles = av1Info.tiles;
        av1Info.picture.tileLayout = &av1Info.tileLayout;
        av1Info.picture.quantization = &av1Info.quantization;
        av1Info.picture.loopFilter = &av1Info.loopFilter;
        av1Info.picture.cdef = &av1Info.cdef;
        av1Info.picture.segmentation = av1DecodeInfo->picture.segmentation ? &av1Info.segmentation : nullptr;
        av1Info.picture.loopRestoration = &av1Info.loopRestoration;
        av1Info.picture.globalMotion = &av1Info.globalMotion;
        av1Info.tileLayout.miColumnStarts = av1Info.miColumnStarts;
        av1Info.tileLayout.miRowStarts = av1Info.miRowStarts;
        av1Info.tileLayout.widthInSuperblocksMinus1 = av1Info.widthInSuperblocksMinus1;
        av1Info.tileLayout.heightInSuperblocksMinus1 = av1Info.heightInSuperblocksMinus1;
        if (av1Info.picture.references && av1Info.picture.referenceNum)
            av1Info.picture.references = av1Info.references;
        if (!isKeyFrame) {
            av1DecodeOrderHints[0] = frame.hasAv1SavedOrderHints ? av1Info.picture.orderHint : 0;
            av1Info.picture.orderHints = av1DecodeOrderHints;
            for (uint32_t i = 0; i < av1Info.picture.referenceNum && i < helper::GetCountOf(av1Info.references); i++)
                av1Info.references[i].savedOrderHints = frame.hasAv1SavedOrderHints ? frame.av1SavedOrderHints[av1Info.references[i].slot % MAX_DPB_SLOT_NUM] : av1DecodeOrderHints;
        }
    }

    nri::VideoDecodeDesc decodeDesc = {};
    decodeDesc.session = m_Session;
    decodeDesc.parameters = m_Parameters;
    decodeDesc.bitstream.buffer = m_DecodeBitstreamBuffer;
    decodeDesc.bitstream.size = decodeBitstreamRange;
    decodeDesc.dstPicture = output.picture;
    decodeDesc.setupPicture = m_IsDistinct ? setup.picture : nullptr;
    decodeDesc.references = plan.referenceNum ? references : nullptr;
    decodeDesc.referenceNum = plan.referenceNum;
    decodeDesc.dstSlot = plan.setupSlot;
    decodeDesc.h264PictureDesc = m_Config.codec == SampleCodec::H264 ? &h264DecodePicture : nullptr;
    decodeDesc.h265PictureDesc = m_Config.codec == SampleCodec::H265 ? &h265DecodePicture : nullptr;
    decodeDesc.av1PictureDesc = av1DecodeInfo ? &av1Info.picture : nullptr;

    // Picture states
    PictureSlot* pictures[2 + MAX_REFERENCE_NUM] = {};
    nri::VideoPictureState states[2 + MAX_REFERENCE_NUM] = {};
    uint32_t pictureNum = 0;
    bool statesOk = true;
    auto addPicture = [&](PictureSlot* slot, nri::VideoPictureRole role) {
        statesOk = statesOk && video.GetVideoPictureState(*slot->picture, role, states[pictureNum]) == nri::Result::SUCCESS;
        pictures[pictureNum++] = slot;
    };
    if (m_IsDistinct) {
        addPicture(&m_Output, nri::VideoPictureRole::DECODE_OUTPUT);
        addPicture(&setup, nri::VideoPictureRole::DECODE_SETUP);
    } else
        addPicture(&setup, nri::VideoPictureRole::DECODE_OUTPUT_AND_SETUP);
    for (uint32_t i = 0; i < plan.referenceNum; i++)
        addPicture(&m_Dpb[plan.references[i].slot], nri::VideoPictureRole::DECODE_REFERENCE);
    if (!statesOk) {
        m_Status = "Failed to query video decode picture states";
        return false;
    }

    if (!SubmitOneTime(nri, *m_Context.decodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::BufferBarrierDesc bufferBarrier = {};
            bufferBarrier.buffer = m_DecodeBitstreamBuffer;
            bufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            bufferBarrier.after = {nri::AccessBits::VIDEO_DECODE_READ, nri::StageBits::VIDEO_DECODE};

            nri::TextureBarrierDesc textureBarriers[2 + MAX_REFERENCE_NUM] = {};
            for (uint32_t i = 0; i < pictureNum; i++)
                textureBarriers[i] = MakePictureBarrier(*pictures[i], pictures[i]->state, states[i].required);

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = m_Context.graphicsAPI == nri::GraphicsAPI::D3D12 ? nullptr : &bufferBarrier;
            barrierDesc.bufferNum = m_Context.graphicsAPI == nri::GraphicsAPI::D3D12 ? 0 : 1;
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = pictureNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);

            video.CmdDecodeVideo(commandBuffer, decodeDesc);

            uint32_t postTextureNum = 0;
            for (uint32_t i = 0; i < pictureNum; i++) {
                if (!states[i].transitionOnVideoQueue)
                    continue;
                textureBarriers[postTextureNum++] = MakePictureBarrier(*pictures[i], states[i].required, states[i].videoQueueAfter);
            }
            barrierDesc.buffers = nullptr;
            barrierDesc.bufferNum = 0;
            barrierDesc.textureNum = postTextureNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " decode submission failed";
        return false;
    }
    for (uint32_t i = 0; i < pictureNum; i++)
        ApplyVideoPictureState(*pictures[i], states[i]);

    // Copy NV12 into the readback buffer
    if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarrier = MakePictureBarrier(output, output.consumerState, {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY});

            nri::BufferBarrierDesc nv12BufferBarrier = {};
            nv12BufferBarrier.buffer = request.nv12ReadbackBuffer;
            nv12BufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            nv12BufferBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = &nv12BufferBarrier;
            barrierDesc.bufferNum = 1;
            barrierDesc.textures = &textureBarrier;
            barrierDesc.textureNum = 1;
            nri.CmdBarrier(commandBuffer, barrierDesc);

            nri::TextureRegionDesc lumaRegion = {};
            lumaRegion.width = (nri::Dim_t)m_Size.videoWidth;
            lumaRegion.height = (nri::Dim_t)m_Size.videoHeight;
            lumaRegion.depth = 1;
            lumaRegion.layerOffset = (nri::Dim_t)output.layer;
            lumaRegion.planes = nri::PlaneBits::PLANE_0;

            nri::TextureDataLayoutDesc lumaLayout = {};
            lumaLayout.rowPitch = request.nv12Layout->yRowPitchBytes;
            lumaLayout.slicePitch = request.nv12Layout->ySlicePitchBytes;
            nri.CmdReadbackTextureToBuffer(commandBuffer, *request.nv12ReadbackBuffer, lumaLayout, *output.texture, lumaRegion);

            nri::TextureRegionDesc chromaRegion = lumaRegion;
            chromaRegion.planes = nri::PlaneBits::PLANE_1;

            nri::TextureDataLayoutDesc chromaLayout = {};
            chromaLayout.offset = request.nv12Layout->uvOffsetBytes;
            chromaLayout.rowPitch = request.nv12Layout->uvRowPitchBytes;
            chromaLayout.slicePitch = request.nv12Layout->uvSlicePitchBytes;
            nri.CmdReadbackTextureToBuffer(commandBuffer, *request.nv12ReadbackBuffer, chromaLayout, *output.texture, chromaRegion);

            nv12BufferBarrier.before = nv12BufferBarrier.after;
            nv12BufferBarrier.after = {nri::AccessBits::NONE, nri::StageBits::NONE};
            textureBarrier.before = textureBarrier.after;
            textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            nri.CmdBarrier(commandBuffer, barrierDesc);
            output.state = textureBarrier.after;
            output.consumerState = textureBarrier.after;
        })) {
        m_Status = "Failed to copy decoded NV12";
        return false;
    }

    decodedFrame = {};
    decodedFrame.displayIndex = plan.displayIndex;
    decodedFrame.isDisplay = plan.isDisplay;

    char message[128] = {};
    std::snprintf(message, sizeof(message), "%s decode complete, encoded %llu bytes", GetCodecName(m_Config.codec), (unsigned long long)frame.feedback.encodedBitstreamWrittenBytes);
    m_Status = message;
    return true;
}

} // namespace video_sample
