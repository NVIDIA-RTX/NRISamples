// © 2021 NVIDIA Corporation

#include "Encoder.h"

#include <cstdio>

namespace video_sample {

Encoder::~Encoder() {
    if (!m_Context.nri)
        return;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    if (video.DestroyVideoPicture) {
        if (m_AV1PReconstructedPicture)
            video.DestroyVideoPicture(m_AV1PReconstructedPicture);
        if (m_ReconstructedPicture)
            video.DestroyVideoPicture(m_ReconstructedPicture);
        if (m_EncodePicture)
            video.DestroyVideoPicture(m_EncodePicture);
        if (m_Parameters)
            video.DestroyVideoSessionParameters(m_Parameters);
        if (m_Session)
            video.DestroyVideoSession(m_Session);
    }

    if (m_MetadataReadbackCommandBuffer)
        nri.DestroyCommandBuffer(m_MetadataReadbackCommandBuffer);
    if (m_MetadataReadbackCommandAllocator)
        nri.DestroyCommandAllocator(m_MetadataReadbackCommandAllocator);
    if (m_MetadataReadbackFence)
        nri.DestroyFence(m_MetadataReadbackFence);
    if (m_ResolvedMetadataReadbackBuffer)
        nri.DestroyBuffer(m_ResolvedMetadataReadbackBuffer);
    if (m_ResolvedMetadataBuffer)
        nri.DestroyBuffer(m_ResolvedMetadataBuffer);
    if (m_MetadataBuffer)
        nri.DestroyBuffer(m_MetadataBuffer);
    if (m_BitstreamBuffer)
        nri.DestroyBuffer(m_BitstreamBuffer);
    if (m_BitstreamHeaderUploadBuffer)
        nri.DestroyBuffer(m_BitstreamHeaderUploadBuffer);
    if (m_ReconstructedTexture)
        nri.DestroyTexture(m_ReconstructedTexture);
    if (m_AV1PReconstructedTexture)
        nri.DestroyTexture(m_AV1PReconstructedTexture);
    if (m_EncodeTexture)
        nri.DestroyTexture(m_EncodeTexture);
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
    sessionDesc.maxReferenceNum = 1;

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
    parametersDesc.h265Parameters = m_Config.codec == SampleCodec::H265 ? &h265Parameters : nullptr;
    parametersDesc.av1Parameters = m_Config.codec == SampleCodec::AV1 ? &av1Parameters : nullptr;

    if (video.CreateVideoSessionParameters(*m_Context.device, parametersDesc, m_Parameters) != nri::Result::SUCCESS) {
        m_Status = std::string("Failed to create ") + GetCodecName(m_Config.codec) + " encode parameters";
        return false;
    }

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.usage = nri::TextureUsageBits::VIDEO_ENCODE;
    textureDesc.format = nri::Format::NV12_UNORM;
    textureDesc.width = (nri::Dim_t)m_Size.codedWidth;
    textureDesc.height = (nri::Dim_t)m_Size.codedHeight;
    textureDesc.mipNum = 1;
    textureDesc.layerNum = 1;
    textureDesc.videoCodec = GetNriCodec(m_Config.codec);

    if (nri.CreateCommittedTexture(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, m_EncodeTexture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create NV12 encode texture";
        return false;
    }
    nri.SetDebugName(m_EncodeTexture, "VideoEncodeTexture");

    if (nri.CreateCommittedTexture(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, m_ReconstructedTexture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create NV12 reconstructed texture";
        return false;
    }
    nri.SetDebugName(m_ReconstructedTexture, "VideoReconstructedTexture");

    if (m_Config.av1PFrameVisual) {
        if (nri.CreateCommittedTexture(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, textureDesc, m_AV1PReconstructedTexture) != nri::Result::SUCCESS) {
            m_Status = "Failed to create second NV12 reconstructed texture";
            return false;
        }
        nri.SetDebugName(m_AV1PReconstructedTexture, "VideoAV1PReconstructedTexture");
    }

    if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarriers[3] = {};
            uint32_t textureBarrierNum = 0;
            textureBarriers[textureBarrierNum++].texture = m_EncodeTexture;
            textureBarriers[textureBarrierNum++].texture = m_ReconstructedTexture;
            if (m_AV1PReconstructedTexture)
                textureBarriers[textureBarrierNum++].texture = m_AV1PReconstructedTexture;

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
        m_Status = "Failed to initialize encode texture layouts";
        return false;
    }

    nri::BufferDesc bitstreamHeaderUploadBufferDesc = {};
    bitstreamHeaderUploadBufferDesc.size = ENCODED_SLICE_OFFSET;

    nri::BufferDesc bitstreamBufferDesc = {};
    bitstreamBufferDesc.size = BITSTREAM_SIZE;
    bitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc metadataBufferDesc = {};
    metadataBufferDesc.size = METADATA_SIZE;
    metadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc resolvedMetadataBufferDesc = {};
    resolvedMetadataBufferDesc.size = RESOLVED_METADATA_SIZE;
    resolvedMetadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc resolvedMetadataReadbackBufferDesc = {};
    resolvedMetadataReadbackBufferDesc.size = RESOLVED_METADATA_SIZE;
    resolvedMetadataReadbackBufferDesc.usage = nri::BufferUsageBits::NONE;

    if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::HOST_UPLOAD, 0.0f, bitstreamHeaderUploadBufferDesc, m_BitstreamHeaderUploadBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create bitstream header upload buffer";
        return false;
    }
    if (CreateEncodeBitstreamBuffer(nri, *m_Context.device, 0.0f, bitstreamBufferDesc, m_BitstreamBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create encode bitstream buffer";
        return false;
    }
    if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, metadataBufferDesc, m_MetadataBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create encode metadata buffer";
        return false;
    }
    if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::DEVICE, 0.0f, resolvedMetadataBufferDesc, m_ResolvedMetadataBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create resolved encode metadata buffer";
        return false;
    }
    if (nri.CreateCommittedBuffer(*m_Context.device, nri::MemoryLocation::HOST_READBACK, 0.0f, resolvedMetadataReadbackBufferDesc, m_ResolvedMetadataReadbackBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create resolved encode metadata readback buffer";
        return false;
    }
    if (nri.CreateFence(*m_Context.device, 0, m_MetadataReadbackFence) != nri::Result::SUCCESS) {
        m_Status = "Failed to create metadata readback fence";
        return false;
    }
    if (nri.CreateCommandAllocator(*m_Context.graphicsQueue, m_MetadataReadbackCommandAllocator) != nri::Result::SUCCESS || nri.CreateCommandBuffer(*m_MetadataReadbackCommandAllocator, m_MetadataReadbackCommandBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to create metadata readback command buffer";
        return false;
    }

    nri::VideoPictureDesc encodePictureDesc = {};
    encodePictureDesc.texture = m_EncodeTexture;
    encodePictureDesc.usage = nri::VideoPictureUsage::ENCODE_INPUT;
    encodePictureDesc.width = (nri::Dim_t)m_Size.codedWidth;
    encodePictureDesc.height = (nri::Dim_t)m_Size.codedHeight;

    nri::VideoPictureDesc reconstructedPictureDesc = encodePictureDesc;
    reconstructedPictureDesc.texture = m_ReconstructedTexture;
    reconstructedPictureDesc.usage = nri::VideoPictureUsage::ENCODE_REFERENCE;

    if (video.CreateVideoPicture(*m_Context.device, encodePictureDesc, m_EncodePicture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create encode picture";
        return false;
    }
    if (video.CreateVideoPicture(*m_Context.device, reconstructedPictureDesc, m_ReconstructedPicture) != nri::Result::SUCCESS) {
        m_Status = "Failed to create reconstructed picture";
        return false;
    }
    if (m_Config.av1PFrameVisual) {
        reconstructedPictureDesc.texture = m_AV1PReconstructedTexture;
        if (video.CreateVideoPicture(*m_Context.device, reconstructedPictureDesc, m_AV1PReconstructedPicture) != nri::Result::SUCCESS) {
            m_Status = "Failed to create second reconstructed picture";
            return false;
        }
    }

    m_Ready = true;
    m_Status = std::string(GetCodecName(m_Config.codec)) + " encoder initialized";
    return true;
}

bool Encoder::IsEncodingAv1PFrame() const {
    return m_Config.av1PFrameVisual && m_AV1PFrameStage == 1;
}

bool Encoder::WriteAnnexBHeadersToUploadBuffer(std::vector<uint8_t>& annexBHeaders) {
    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    if (m_Config.codec == SampleCodec::AV1) {
        void* headerPtr = nri.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        if (!headerPtr) {
            m_Status = "Failed to map bitstream header upload buffer";
            return false;
        }
        std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
        nri.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);
        annexBHeaders.clear();
        return true;
    }

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

    void* headerPtr = nri.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
    if (!headerPtr) {
        m_Status = "Failed to map bitstream header upload buffer";
        return false;
    }
    std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
    std::memcpy(headerPtr, annexBHeaders.data(), annexBHeaders.size());
    nri.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);
    return true;
}

bool Encoder::Encode(const EncodeRequest& request) {
    if (!m_Ready || !request.nv12Buffer || !request.nv12Layout) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " encoder is not ready";
        return false;
    }

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;
    const bool av1PFrame = IsEncodingAv1PFrame();

    if (!CopyNv12BufferToTexture(nri, *m_Context.graphicsQueue, *request.nv12Layout, *request.nv12Buffer, *m_EncodeTexture, m_Size.videoWidth, m_Size.videoHeight)) {
        m_Status = "Failed to upload NV12 source to video texture";
        return false;
    }

    std::vector<uint8_t> annexBHeaders;
    if (!WriteAnnexBHeadersToUploadBuffer(annexBHeaders))
        return false;

    if (!SubmitOneTime(nri, *m_Context.graphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri.CmdZeroBuffer(commandBuffer, *m_BitstreamBuffer, 0, BITSTREAM_SIZE);
            nri.CmdCopyBuffer(commandBuffer, *m_BitstreamBuffer, 0, *m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        })) {
        m_Status = std::string("Failed to upload ") + GetCodecName(m_Config.codec) + " Annex-B parameter sets";
        return false;
    }

    nri::VideoEncodePictureDesc pictureDesc = {};
    pictureDesc.frameType = av1PFrame ? nri::VideoEncodeFrameType::P : nri::VideoEncodeFrameType::IDR;
    pictureDesc.frameIndex = av1PFrame ? 1 : 0;
    pictureDesc.pictureOrderCount = av1PFrame ? 1 : 0;
    pictureDesc.idrPictureId = av1PFrame ? 0 : 1;

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
    nri::VideoAV1PictureDesc av1PictureDesc = {};
    av1PictureDesc.currentFrameId = av1PFrame ? 1 : 0;
    av1PictureDesc.orderHint = av1PFrame ? 1 : 0;
    av1PictureDesc.refreshFrameFlags = av1PFrame ? 0x1 : 0xFF;
    av1PictureDesc.primaryReferenceName = av1PFrame ? nri::VideoAV1ReferenceName::LAST : nri::VideoAV1ReferenceName::NONE;
    nri::VideoAV1PictureBits av1CommonPictureFlags = nri::VideoAV1PictureBits::SHOW_FRAME;
    if (m_Context.graphicsAPI == nri::GraphicsAPI::D3D12)
        av1CommonPictureFlags |= nri::VideoAV1PictureBits::SEGMENTATION_ENABLED;
    av1PictureDesc.flags = av1PFrame ? av1CommonPictureFlags
                                     : av1CommonPictureFlags | nri::VideoAV1PictureBits::ERROR_RESILIENT_MODE;
    av1PictureDesc.renderWidthMinus1 = (uint16_t)((m_Context.graphicsAPI == nri::GraphicsAPI::VK ? m_Size.codedWidth : m_Size.videoWidth) - 1);
    av1PictureDesc.renderHeightMinus1 = (uint16_t)((m_Context.graphicsAPI == nri::GraphicsAPI::VK ? m_Size.codedHeight : m_Size.videoHeight) - 1);
    av1PictureDesc.baseQIndex = (uint8_t)m_Config.av1BaseQIndex;
    av1PictureDesc.interpolationFilter = 0;
    av1PictureDesc.txMode = 2;
    av1PictureDesc.cdefDampingMinus3 = 3;
    av1PictureDesc.tileLayout = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? nullptr : &av1TileLayout;
    av1PictureDesc.loopFilter = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? nullptr : &av1LoopFilter;
    av1PictureDesc.cdef = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? nullptr : &av1Cdef;
    av1PictureDesc.loopRestoration = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? nullptr : &av1LoopRestoration;
    av1PictureDesc.globalMotion = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? nullptr : &av1GlobalMotion;
    nri::VideoReference av1Reference = {m_ReconstructedPicture, 0};
    nri::VideoAV1ReferenceDesc av1References[8] = {};
    if (av1PFrame) {
        const nri::VideoAV1ReferenceName av1ReferenceNames[] = {
            nri::VideoAV1ReferenceName::LAST,
            nri::VideoAV1ReferenceName::LAST2,
            nri::VideoAV1ReferenceName::LAST3,
            nri::VideoAV1ReferenceName::GOLDEN,
            nri::VideoAV1ReferenceName::BWDREF,
            nri::VideoAV1ReferenceName::ALTREF2,
            nri::VideoAV1ReferenceName::ALTREF,
        };
        for (uint32_t i = 0; i < helper::GetCountOf(av1ReferenceNames); i++) {
            av1References[i].name = av1ReferenceNames[i];
            av1References[i].refFrameIndex = 0;
            av1References[i].frameType = nri::VideoEncodeFrameType::IDR;
            av1References[i].orderHint = 0;
            av1References[i].frameId = 0;
            av1References[i].slot = 0;
        }
        av1PictureDesc.references = av1References;
        av1PictureDesc.referenceNum = helper::GetCountOf(av1ReferenceNames);
    }

    nri::VideoEncodeRateControlDesc rateControlDesc = {};
    rateControlDesc.mode = nri::VideoEncodeRateControlMode::CQP;
    rateControlDesc.qpI = (uint8_t)m_Config.qpI;
    rateControlDesc.qpP = (uint8_t)m_Config.qpP;
    rateControlDesc.qpB = (uint8_t)m_Config.qpB;
    rateControlDesc.frameRateNumerator = 30;
    rateControlDesc.frameRateDenominator = 1;

    nri::VideoEncodeDesc encodeDesc = {};
    encodeDesc.session = m_Session;
    encodeDesc.parameters = m_Parameters;
    encodeDesc.srcPicture = m_EncodePicture;
    encodeDesc.dstBitstream.buffer = m_BitstreamBuffer;
    encodeDesc.dstBitstream.offset = ENCODED_SLICE_OFFSET;
    encodeDesc.dstBitstream.size = BITSTREAM_SIZE - ENCODED_SLICE_OFFSET;
    encodeDesc.bitstreamMetadataSize = ENCODED_SLICE_OFFSET;
    encodeDesc.pictureDesc = &pictureDesc;
    encodeDesc.rateControlDesc = &rateControlDesc;
    encodeDesc.reconstructedPicture = m_ReconstructedPicture;
    if (av1PFrame) {
        encodeDesc.reconstructedPicture = m_AV1PReconstructedPicture;
        encodeDesc.references = &av1Reference;
        encodeDesc.referenceNum = 1;
        encodeDesc.reconstructedSlot = 1;
    }
    encodeDesc.metadata = m_MetadataBuffer;
    encodeDesc.resolvedMetadata = m_Context.graphicsAPI == nri::GraphicsAPI::VK ? m_ResolvedMetadataReadbackBuffer : m_ResolvedMetadataBuffer;
    encodeDesc.av1PictureDesc = m_Config.codec == SampleCodec::AV1 ? &av1PictureDesc : nullptr;

    nri::VideoEncodePictureStates srcPictureStates = {};
    nri::VideoEncodePictureStates reconstructedPictureStates = {};
    if (video.GetVideoEncodePictureStates(*m_EncodePicture, srcPictureStates) != nri::Result::SUCCESS || video.GetVideoEncodePictureStates(*(av1PFrame ? m_AV1PReconstructedPicture : m_ReconstructedPicture), reconstructedPictureStates) != nri::Result::SUCCESS) {
        m_Status = "Failed to query video encode picture states";
        return false;
    }

    if (!SubmitOneTime(nri, *m_Context.encodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::BufferBarrierDesc bufferBarriers[2] = {};
            bufferBarriers[0].buffer = m_MetadataBuffer;
            bufferBarriers[0].after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[1].buffer = encodeDesc.resolvedMetadata;
            bufferBarriers[1].after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};

            nri::TextureBarrierDesc textureBarriers[3] = {};
            textureBarriers[0].texture = m_EncodeTexture;
            textureBarriers[0].before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[0].after = srcPictureStates.encodeRead;
            textureBarriers[0].mipNum = nri::REMAINING;
            textureBarriers[0].layerNum = nri::REMAINING;
            textureBarriers[0].planes = nri::PlaneBits::ALL;
            textureBarriers[1].texture = av1PFrame ? m_AV1PReconstructedTexture : m_ReconstructedTexture;
            textureBarriers[1].before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[1].after = reconstructedPictureStates.encodeWrite;
            textureBarriers[1].mipNum = nri::REMAINING;
            textureBarriers[1].layerNum = nri::REMAINING;
            textureBarriers[1].planes = nri::PlaneBits::ALL;
            textureBarriers[2].texture = m_ReconstructedTexture;
            textureBarriers[2].before = reconstructedPictureStates.graphicsBefore;
            textureBarriers[2].after = {nri::AccessBits::VIDEO_ENCODE_READ, nri::Layout::VIDEO_ENCODE_DPB, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[2].mipNum = nri::REMAINING;
            textureBarriers[2].layerNum = nri::REMAINING;
            textureBarriers[2].planes = nri::PlaneBits::ALL;

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = bufferBarriers;
            barrierDesc.bufferNum = helper::GetCountOf(bufferBarriers);
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = av1PFrame ? helper::GetCountOf(textureBarriers) : 2;
            nri.CmdBarrier(commandBuffer, barrierDesc);
            video.CmdEncodeVideo(commandBuffer, encodeDesc);
            bufferBarriers[0].before = {nri::AccessBits::VIDEO_ENCODE_READ, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[0].after = {};
            bufferBarriers[1].before = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[1].after = {};
            textureBarriers[0].before = srcPictureStates.encodeRead;
            textureBarriers[0].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[1].before = reconstructedPictureStates.encodeWrite;
            textureBarriers[1].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[2].before = {nri::AccessBits::VIDEO_ENCODE_READ, nri::Layout::VIDEO_ENCODE_DPB, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[2].after = reconstructedPictureStates.afterEncode;
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = av1PFrame && reconstructedPictureStates.releaseAfterEncode ? helper::GetCountOf(textureBarriers) : 2;
            nri.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_Status = std::string(GetCodecName(m_Config.codec)) + " encode submission failed";
        return false;
    }

    if (m_MetadataReadbackPending)
        return true;

    if (m_Context.graphicsAPI == nri::GraphicsAPI::VK) {
        m_MetadataReadbackPending = true;
        return true;
    }

    return SubmitMetadataReadback();
}

bool Encoder::SubmitMetadataReadback() {
    NRIInterface& nri = *m_Context.nri;

    nri.ResetCommandAllocator(*m_MetadataReadbackCommandAllocator);
    if (nri.BeginCommandBuffer(*m_MetadataReadbackCommandBuffer, nullptr) != nri::Result::SUCCESS) {
        m_Status = "Failed to begin metadata readback command buffer";
        return false;
    }

    nri::BufferBarrierDesc metadataBarriers[2] = {};
    metadataBarriers[0].buffer = m_ResolvedMetadataBuffer;
    metadataBarriers[0].before = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarriers[1].buffer = m_ResolvedMetadataReadbackBuffer;
    metadataBarriers[1].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

    nri::BarrierDesc metadataBarrierDesc = {};
    metadataBarrierDesc.buffers = metadataBarriers;
    metadataBarrierDesc.bufferNum = helper::GetCountOf(metadataBarriers);
    nri.CmdBarrier(*m_MetadataReadbackCommandBuffer, metadataBarrierDesc);
    nri.CmdCopyBuffer(*m_MetadataReadbackCommandBuffer, *m_ResolvedMetadataReadbackBuffer, 0, *m_ResolvedMetadataBuffer, 0, RESOLVED_METADATA_SIZE);
    metadataBarriers[0].before = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarriers[0].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[1].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[1].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    nri.CmdBarrier(*m_MetadataReadbackCommandBuffer, metadataBarrierDesc);

    if (nri.EndCommandBuffer(*m_MetadataReadbackCommandBuffer) != nri::Result::SUCCESS) {
        m_Status = "Failed to end metadata readback command buffer";
        return false;
    }

    m_MetadataReadbackFenceValue++;
    nri::FenceSubmitDesc signalFence = {};
    signalFence.fence = m_MetadataReadbackFence;
    signalFence.value = m_MetadataReadbackFenceValue;

    const nri::CommandBuffer* commandBuffers[] = {m_MetadataReadbackCommandBuffer};
    nri::QueueSubmitDesc submit = {};
    submit.commandBuffers = commandBuffers;
    submit.commandBufferNum = helper::GetCountOf(commandBuffers);
    submit.signalFences = &signalFence;
    submit.signalFenceNum = 1;
    if (nri.QueueSubmit(*m_Context.graphicsQueue, submit) != nri::Result::SUCCESS) {
        m_Status = "Failed to submit async metadata readback";
        return false;
    }

    m_MetadataReadbackPending = true;
    return true;
}

bool Encoder::BuildAv1DecodeInfo(nri::VideoEncodeFeedback& feedback, nri::VideoAV1EncodeDecodeInfo& av1DecodeInfo) {
    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    nri::VideoAV1EncodeDecodeInfoDesc av1InfoDesc = {};
    av1InfoDesc.feedback = &feedback;
    av1InfoDesc.sequence = &m_CodecParameters.av1Sequence;
    nri::VideoAV1ReferenceDesc av1InfoReferences[8] = {};
    if (IsEncodingAv1PFrame()) {
        const nri::VideoAV1ReferenceName av1ReferenceNames[] = {
            nri::VideoAV1ReferenceName::LAST,
            nri::VideoAV1ReferenceName::LAST2,
            nri::VideoAV1ReferenceName::LAST3,
            nri::VideoAV1ReferenceName::GOLDEN,
            nri::VideoAV1ReferenceName::BWDREF,
            nri::VideoAV1ReferenceName::ALTREF2,
            nri::VideoAV1ReferenceName::ALTREF,
        };
        for (uint32_t i = 0; i < helper::GetCountOf(av1ReferenceNames); i++) {
            av1InfoReferences[i].name = av1ReferenceNames[i];
            av1InfoReferences[i].refFrameIndex = 0;
            av1InfoReferences[i].frameType = nri::VideoEncodeFrameType::IDR;
            av1InfoReferences[i].orderHint = 0;
            av1InfoReferences[i].frameId = 0;
            av1InfoReferences[i].slot = 0;
        }
        av1InfoDesc.references = av1InfoReferences;
        av1InfoDesc.referenceNum = helper::GetCountOf(av1ReferenceNames);
    }

    const uint8_t* encodedPayloadHeader = nullptr;
    const uint64_t encodedPayloadOffset = ENCODED_SLICE_OFFSET + feedback.encodedBitstreamOffset;
    const uint64_t encodedPayloadSize = feedback.encodedBitstreamWrittenBytes;
    if (m_Context.graphicsAPI == nri::GraphicsAPI::VK && encodedPayloadOffset <= BITSTREAM_SIZE && encodedPayloadSize <= BITSTREAM_SIZE - encodedPayloadOffset) {
        encodedPayloadHeader = (const uint8_t*)nri.MapBuffer(*m_BitstreamBuffer, encodedPayloadOffset, encodedPayloadSize);
        av1InfoDesc.encodedPayloadHeader = encodedPayloadHeader;
        av1InfoDesc.encodedPayloadHeaderSize = encodedPayloadHeader ? encodedPayloadSize : 0;
    }
    const nri::Result av1InfoResult = video.GetVideoEncodeAV1DecodeInfo(*m_Session, *m_ResolvedMetadataReadbackBuffer, 0, av1InfoDesc, av1DecodeInfo);
    if (encodedPayloadHeader)
        nri.UnmapBuffer(*m_BitstreamBuffer);
    if (av1InfoResult != nri::Result::SUCCESS) {
        m_Status = "Failed to prepare AV1 decode metadata";
        return false;
    }
    feedback.encodedBitstreamWrittenBytes = av1DecodeInfo.bitstreamOffset + av1DecodeInfo.bitstreamSize;
    return true;
}

bool Encoder::Poll(EncodedFrame& frame) {
    if (!m_MetadataReadbackPending)
        return false;

    NRIInterface& nri = *m_Context.nri;
    nri::VideoInterface& video = *m_Context.video;

    const uint64_t completedFence = nri.GetFenceValue(*m_MetadataReadbackFence);
    if (completedFence < m_MetadataReadbackFenceValue)
        return false;

    m_MetadataReadbackPending = false;

    frame = {};
    frame.bitstreamBuffer = m_BitstreamBuffer;
    const nri::Result feedbackResult = video.GetVideoEncodeFeedback(*m_Session, *m_ResolvedMetadataReadbackBuffer, 0, frame.feedback);
    if (feedbackResult != nri::Result::SUCCESS) {
        if (feedbackResult == nri::Result::UNSUPPORTED)
            m_Status = std::string(GetCodecName(m_Config.codec)) + " encode metadata feedback is unsupported";
        else
            m_Status = "Failed to read resolved encode metadata";
        return false;
    }

    if (frame.feedback.errorFlags || !frame.feedback.encodedBitstreamWrittenBytes) {
        char message[160] = {};
        std::snprintf(message, sizeof(message), "Encoder returned errorFlags=0x%llX bytes=%llu",
            (unsigned long long)frame.feedback.errorFlags, (unsigned long long)frame.feedback.encodedBitstreamWrittenBytes);
        m_Status = message;
        return false;
    }

    if (m_Config.codec == SampleCodec::AV1) {
        frame.hasAv1DecodeInfo = true;
        if (!BuildAv1DecodeInfo(frame.feedback, frame.av1DecodeInfo))
            return false;
    }

    frame.isAv1PFrame = IsEncodingAv1PFrame();
    if (m_Config.av1PFrameVisual && m_AV1PFrameStage == 0)
        m_AV1PFrameStage = 1;
    else if (frame.isAv1PFrame)
        m_AV1PFrameStage = 0;

    m_Status = std::string(GetCodecName(m_Config.codec)) + " encode complete";
    return true;
}

} // namespace video_sample
