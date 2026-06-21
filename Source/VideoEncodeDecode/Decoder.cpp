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
        if (m_AV1PDecodePicture)
            video.DestroyVideoPicture(m_AV1PDecodePicture);
        if (m_DecodePicture)
            video.DestroyVideoPicture(m_DecodePicture);
        if (m_Parameters)
            video.DestroyVideoSessionParameters(m_Parameters);
        if (m_Session)
            video.DestroyVideoSession(m_Session);
    }

    if (m_DecodeBitstreamBuffer)
        nri.DestroyBuffer(m_DecodeBitstreamBuffer);
    if (m_DecodeTexture)
        nri.DestroyTexture(m_DecodeTexture);
    if (m_AV1PDecodeTexture)
        nri.DestroyTexture(m_AV1PDecodeTexture);
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
    sessionDesc.maxReferenceNum = m_Config.codec == SampleCodec::AV1 ? 1 : 16;

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

    if (nri.CreateCommittedTexture(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, m_DecodeTexture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create NV12 decode texture";
        return false;
    }
    nri.SetDebugName(m_DecodeTexture, "VideoDecodeTexture");

    if (m_Config.av1PFrameVisual) {
        if (nri.CreateCommittedTexture(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, m_AV1PDecodeTexture) != nri::Result::SUCCESS) {
            m_Status = "Failed to create second NV12 decode texture";
            return false;
        }
        nri.SetDebugName(m_AV1PDecodeTexture, "VideoAV1PDecodeTexture");
    }

    if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarriers[2] = {};
            uint32_t textureBarrierNum = 0;
            textureBarriers[textureBarrierNum++].texture = m_DecodeTexture;
            if (m_AV1PDecodeTexture)
                textureBarriers[textureBarrierNum++].texture = m_AV1PDecodeTexture;

            for (nri::TextureBarrierDesc& textureBarrier : textureBarriers) {
                if (!textureBarrier.texture)
                    continue;
                textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
                textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
                textureBarrier.mipNum = nri::REMAINING;
                textureBarrier.layerNum = nri::REMAINING;
                textureBarrier.planes = nri::PlaneBits::ALL;
            }

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = textureBarrierNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_Status = "Failed to initialize decode texture layouts";
        return false;
    }

    nri::BufferDesc decodeBitstreamBufferDesc = {};
    decodeBitstreamBufferDesc.size = BITSTREAM_SIZE;
    decodeBitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_DECODE;
    if (CreateDecodeBitstreamBuffer(nri, *m_Context.device, 0.0f, decodeBitstreamBufferDesc, m_DecodeBitstreamBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create decode bitstream buffer";
        return false;
    }

    nri::VideoPictureDesc decodePictureDesc = {};
    decodePictureDesc.texture = m_DecodeTexture;
    decodePictureDesc.usage = nri::VideoPictureUsage::DECODE_OUTPUT;
    decodePictureDesc.width = (nri::Dim_t)m_Size.codedWidth;
    decodePictureDesc.height = (nri::Dim_t)m_Size.codedHeight;

    if (video.CreateVideoPicture(*m_Context.device, decodePictureDesc, m_DecodePicture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create decode picture";
        return false;
    }
    if (m_Config.av1PFrameVisual) {
        decodePictureDesc.texture = m_AV1PDecodeTexture;
        if (video.CreateVideoPicture(*m_Context.device, decodePictureDesc, m_AV1PDecodePicture) != nri::Result::SUCCESS) {
            m_Status = "Failed to create second decode picture";
            return false;
        }
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
    const uint64_t encodedPayloadBytes = av1DecodeInfo ? av1DecodeInfo->bitstreamSize : frame.feedback.encodedBitstreamWrittenBytes - encodedPayloadSkip;
    const uint64_t decodeSliceOffset = annexBHeaders.size();
    pictureOffset = (uint32_t)decodeSliceOffset;
    const uint64_t decodeBitstreamSize = AlignUp(decodeSliceOffset + encodedPayloadBytes + annexBEndOfStream.size(), m_Size.decodeBitstreamSizeAlignment);
    const uint64_t encodedSourceOffset = ENCODED_SLICE_OFFSET + frame.feedback.encodedBitstreamOffset + encodedPayloadSkip;
    if (frame.feedback.encodedBitstreamOffset > BITSTREAM_SIZE - ENCODED_SLICE_OFFSET || encodedSourceOffset > BITSTREAM_SIZE || encodedPayloadBytes > BITSTREAM_SIZE - encodedSourceOffset || decodeBitstreamSize > BITSTREAM_SIZE) {
        m_Status = std::string("Encoded ") + GetCodecName(m_Config.codec) + " bitstream exceeded decode buffer size";
        return false;
    }

    const uint8_t* encodedPayload = (const uint8_t*)nri.MapBuffer(*frame.bitstreamBuffer, encodedSourceOffset, encodedPayloadBytes);
    uint8_t* decodeBitstream = (uint8_t*)nri.MapBuffer(*m_DecodeBitstreamBuffer, 0, decodeBitstreamSize);
    if (!encodedPayload || !decodeBitstream) {
        if (encodedPayload)
            nri.UnmapBuffer(*frame.bitstreamBuffer);
        if (decodeBitstream)
            nri.UnmapBuffer(*m_DecodeBitstreamBuffer);
        m_Status = std::string("Failed to map exact ") + GetCodecName(m_Config.codec) + " decode bitstream";
        return false;
    }
    std::memset(decodeBitstream, 0, (size_t)decodeBitstreamSize);
    if (!annexBHeaders.empty())
        std::memcpy(decodeBitstream, annexBHeaders.data(), annexBHeaders.size());
    std::memcpy(decodeBitstream + decodeSliceOffset, encodedPayload, (size_t)encodedPayloadBytes);
    if (!annexBEndOfStream.empty())
        std::memcpy(decodeBitstream + decodeSliceOffset + encodedPayloadBytes, annexBEndOfStream.data(), annexBEndOfStream.size());
    decodeBitstreamRange = AlignUp(decodeSliceOffset + encodedPayloadBytes + annexBEndOfStream.size(), m_Size.decodeBitstreamSizeAlignment);
    nri.UnmapBuffer(*frame.bitstreamBuffer);
    nri.UnmapBuffer(*m_DecodeBitstreamBuffer);
    return true;
}

bool Decoder::Decode(const DecodeRequest& request, DecodedFrame& decodedFrame) {
    if (!m_Ready || !request.frame || !request.nv12ReadbackBuffer || !request.nv12Layout) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " decoder is not ready";
        return false;
    }

    const EncodedFrame& frame = *request.frame;
    const nri::VideoAV1EncodeDecodeInfo* av1DecodeInfo = frame.hasAv1DecodeInfo ? &frame.av1DecodeInfo : nullptr;
    const bool av1PFrame = frame.isAv1PFrame;

    uint64_t decodeBitstreamRange = 0;
    uint32_t pictureOffset = 0;
    if (!BuildDecodeBitstream(frame, decodeBitstreamRange, pictureOffset))
        return false;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    const uint32_t decodeFrameIndex = m_DecodeFrameIndex++;
    const uint32_t decodeSlot = m_Config.codec == SampleCodec::AV1 ? 0 : decodeFrameIndex % 16;
    const uint32_t pictureOffsets[] = {pictureOffset};

    nri::VideoH264DecodePictureDesc h264DecodePicture = {};
    h264DecodePicture.flags = nri::VideoH264DecodePictureBits::IDR | nri::VideoH264DecodePictureBits::INTRA | nri::VideoH264DecodePictureBits::REFERENCE;
    h264DecodePicture.sequenceParameterSetId = m_CodecParameters.h264Sps.sequenceParameterSetId;
    h264DecodePicture.pictureParameterSetId = m_CodecParameters.h264Pps.pictureParameterSetId;
    h264DecodePicture.frameNum = (uint16_t)(decodeFrameIndex & 0xF);
    h264DecodePicture.idrPictureId = (uint16_t)(1 + (decodeFrameIndex & 0xFFFF));
    h264DecodePicture.topFieldOrderCount = 0;
    h264DecodePicture.bottomFieldOrderCount = 0;
    h264DecodePicture.sliceOffsets = pictureOffsets;
    h264DecodePicture.sliceOffsetNum = helper::GetCountOf(pictureOffsets);
    h264DecodePicture.referenceSlot = decodeSlot;

    nri::VideoH265DecodePictureDesc h265DecodePicture = {};
    h265DecodePicture.flags = nri::VideoH265DecodePictureBits::IRAP | nri::VideoH265DecodePictureBits::IDR | nri::VideoH265DecodePictureBits::REFERENCE;
    h265DecodePicture.videoParameterSetId = m_CodecParameters.h265Vps.videoParameterSetId;
    h265DecodePicture.sequenceParameterSetId = m_CodecParameters.h265Sps.sequenceParameterSetId;
    h265DecodePicture.pictureParameterSetId = m_CodecParameters.h265Pps.pictureParameterSetId;
    h265DecodePicture.pictureOrderCount = (int32_t)decodeFrameIndex;
    h265DecodePicture.sliceSegmentOffsets = pictureOffsets;
    h265DecodePicture.sliceSegmentOffsetNum = helper::GetCountOf(pictureOffsets);

    nri::VideoAV1EncodeDecodeInfo av1Info = {};
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
    }
    nri::VideoReference av1DecodeReference = {m_DecodePicture, 0};
    uint8_t av1DecodeOrderHints[8] = {};
    if (av1PFrame) {
        av1Info.picture.orderHints = av1DecodeOrderHints;
        for (uint32_t i = 0; i < av1Info.picture.referenceNum && i < helper::GetCountOf(av1Info.references); i++)
            av1Info.references[i].savedOrderHints = av1DecodeOrderHints;
    }

    nri::VideoDecodeDesc decodeDesc = {};
    decodeDesc.session = m_Session;
    decodeDesc.parameters = m_Parameters;
    decodeDesc.bitstream.buffer = m_DecodeBitstreamBuffer;
    decodeDesc.bitstream.size = decodeBitstreamRange;
    decodeDesc.dstPicture = av1PFrame ? m_AV1PDecodePicture : m_DecodePicture;
    decodeDesc.references = av1PFrame ? &av1DecodeReference : nullptr;
    decodeDesc.referenceNum = av1PFrame ? 1u : 0u;
    decodeDesc.dstSlot = av1PFrame ? 1u : decodeSlot;
    decodeDesc.h264PictureDesc = m_Config.codec == SampleCodec::H264 ? &h264DecodePicture : nullptr;
    decodeDesc.h265PictureDesc = m_Config.codec == SampleCodec::H265 ? &h265DecodePicture : nullptr;
    decodeDesc.av1PictureDesc = av1DecodeInfo ? &av1Info.picture : nullptr;

    nri::VideoDecodePictureStates decodePictureStates = {};
    if (video.GetVideoDecodePictureStates(*(av1PFrame ? m_AV1PDecodePicture : m_DecodePicture), decodePictureStates) != nri::Result::SUCCESS) {
        m_Status = "Failed to query video decode picture states";
        return false;
    }
    if (m_Context.graphicsAPI == nri::GraphicsAPI::VK && !av1DecodeInfo) {
        decodePictureStates.decodeWrite = {nri::AccessBits::VIDEO_DECODE_WRITE, nri::Layout::VIDEO_DECODE_DPB, nri::StageBits::VIDEO_DECODE};
        decodePictureStates.graphicsBefore = {nri::AccessBits::VIDEO_DECODE, nri::Layout::VIDEO_DECODE_DPB, nri::StageBits::VIDEO_DECODE};
    }

    if (!SubmitOneTime(nri, *m_Context.decodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::BufferBarrierDesc bufferBarrier = {};
            bufferBarrier.buffer = m_DecodeBitstreamBuffer;
            bufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            bufferBarrier.after = {nri::AccessBits::VIDEO_DECODE_READ, nri::StageBits::VIDEO_DECODE};

            nri::TextureBarrierDesc textureBarriers[3] = {};
            uint32_t textureBarrierNum = 0;
            textureBarriers[textureBarrierNum].texture = av1PFrame ? m_AV1PDecodeTexture : m_DecodeTexture;
            textureBarriers[textureBarrierNum].before = {nri::AccessBits::NONE, m_Context.graphicsAPI == nri::GraphicsAPI::VK && !av1DecodeInfo ? nri::Layout::UNDEFINED : nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[textureBarrierNum].after = decodePictureStates.decodeWrite;
            textureBarriers[textureBarrierNum].mipNum = nri::REMAINING;
            textureBarriers[textureBarrierNum].layerNum = nri::REMAINING;
            textureBarriers[textureBarrierNum].planes = nri::PlaneBits::ALL;
            textureBarrierNum++;
            if (av1PFrame) {
                textureBarriers[textureBarrierNum].texture = m_DecodeTexture;
                textureBarriers[textureBarrierNum].before = {nri::AccessBits::VIDEO_DECODE_READ, nri::Layout::VIDEO_DECODE_DPB, nri::StageBits::VIDEO_DECODE};
                textureBarriers[textureBarrierNum].after = {nri::AccessBits::VIDEO_DECODE_READ, nri::Layout::VIDEO_DECODE_DPB, nri::StageBits::VIDEO_DECODE};
                textureBarriers[textureBarrierNum].mipNum = nri::REMAINING;
                textureBarriers[textureBarrierNum].layerNum = nri::REMAINING;
                textureBarriers[textureBarrierNum].planes = nri::PlaneBits::ALL;
                textureBarrierNum++;
            }

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = m_Context.graphicsAPI == nri::GraphicsAPI::D3D12 ? nullptr : &bufferBarrier;
            barrierDesc.bufferNum = m_Context.graphicsAPI == nri::GraphicsAPI::D3D12 ? 0 : 1;
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = textureBarrierNum;
            nri.CmdBarrier(commandBuffer, barrierDesc);
            video.CmdDecodeVideo(commandBuffer, decodeDesc);

            if (decodePictureStates.releaseAfterDecode) {
                textureBarriers[0].before = decodePictureStates.decodeWrite;
                textureBarriers[0].after = decodePictureStates.afterDecode;
                if (av1PFrame) {
                    textureBarriers[1].before = {nri::AccessBits::VIDEO_DECODE_READ, nri::Layout::VIDEO_DECODE_DPB, nri::StageBits::VIDEO_DECODE};
                    textureBarriers[1].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
                }
                barrierDesc.buffers = nullptr;
                barrierDesc.bufferNum = 0;
                nri.CmdBarrier(commandBuffer, barrierDesc);
            }
        })) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " decode submission failed";
        return false;
    }

    nri::Queue* readbackQueue = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? m_Context.decodeQueue : m_Context.graphicsQueue;
    if (!SubmitOneTime(nri, *readbackQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = av1PFrame ? m_AV1PDecodeTexture : m_DecodeTexture;
            textureBarrier.before = decodePictureStates.graphicsBefore;
            textureBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
            textureBarrier.mipNum = nri::REMAINING;
            textureBarrier.layerNum = nri::REMAINING;
            textureBarrier.planes = nri::PlaneBits::ALL;

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = &textureBarrier;
            barrierDesc.textureNum = 1;
            nri.CmdBarrier(commandBuffer, barrierDesc);

            nri::BufferBarrierDesc nv12BufferBarrier = {};
            nv12BufferBarrier.buffer = request.nv12ReadbackBuffer;
            nv12BufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            nv12BufferBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

            nri::BarrierDesc copyBarrierDesc = {};
            copyBarrierDesc.buffers = &nv12BufferBarrier;
            copyBarrierDesc.bufferNum = 1;
            nri.CmdBarrier(commandBuffer, copyBarrierDesc);

            nri::TextureRegionDesc lumaRegion = {};
            lumaRegion.width = (nri::Dim_t)m_Size.videoWidth;
            lumaRegion.height = (nri::Dim_t)m_Size.videoHeight;
            lumaRegion.depth = 1;
            lumaRegion.planes = nri::PlaneBits::PLANE_0;

            nri::TextureDataLayoutDesc lumaLayout = {};
            lumaLayout.rowPitch = request.nv12Layout->yRowPitchBytes;
            lumaLayout.slicePitch = request.nv12Layout->ySlicePitchBytes;
            nri.CmdReadbackTextureToBuffer(commandBuffer, *request.nv12ReadbackBuffer, lumaLayout, *(av1PFrame ? m_AV1PDecodeTexture : m_DecodeTexture), lumaRegion);

            nri::TextureRegionDesc chromaRegion = {};
            chromaRegion.width = (nri::Dim_t)m_Size.videoWidth;
            chromaRegion.height = (nri::Dim_t)m_Size.videoHeight;
            chromaRegion.depth = 1;
            chromaRegion.planes = nri::PlaneBits::PLANE_1;

            nri::TextureDataLayoutDesc chromaLayout = {};
            chromaLayout.offset = request.nv12Layout->uvOffsetBytes;
            chromaLayout.rowPitch = request.nv12Layout->uvRowPitchBytes;
            chromaLayout.slicePitch = request.nv12Layout->uvSlicePitchBytes;
            nri.CmdReadbackTextureToBuffer(commandBuffer, *request.nv12ReadbackBuffer, chromaLayout, *(av1PFrame ? m_AV1PDecodeTexture : m_DecodeTexture), chromaRegion);

            nv12BufferBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
            nv12BufferBarrier.after = {nri::AccessBits::NONE, nri::StageBits::NONE};
            nri.CmdBarrier(commandBuffer, copyBarrierDesc);

            textureBarrier.before = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
            textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            nri.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_Status = "Failed to copy decoded NV12 for preview";
        return false;
    }

    decodedFrame = {};
    decodedFrame.texture = av1PFrame ? m_AV1PDecodeTexture : m_DecodeTexture;
    decodedFrame.picture = av1PFrame ? m_AV1PDecodePicture : m_DecodePicture;
    decodedFrame.pictureStates = decodePictureStates;
    decodedFrame.isAv1PFrame = av1PFrame;

    char message[128] = {};
    std::snprintf(message, sizeof(message), "%s decode complete, encoded %llu bytes", GetCodecName(m_Config.codec), (unsigned long long)frame.feedback.encodedBitstreamWrittenBytes);
    m_Status = message;
    return true;
}

} // namespace video_sample
