// © 2021 NVIDIA Corporation

#pragma once

#include "Shared.h"

namespace video_sample {

struct EncodeRequest {
    nri::Buffer* nv12Buffer = nullptr;
    const Nv12BufferLayout* nv12Layout = nullptr;
    VideoQuality quality = {};
    float timeSec = 0.0f;
};

class Encoder {
public:
    Encoder() = default;
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    bool Initialize(const VideoContext& context, const VideoConfig& config, const VideoSize& size, const CodecParameters& codecParameters);
    bool Encode(const EncodeRequest& request);
    bool Poll(EncodedFrame& frame);

    bool IsReady() const {
        return m_Ready;
    }

    bool HasPendingFeedback() const {
        return m_MetadataReadbackPending;
    }

    nri::Texture* GetInputTexture() const {
        return m_EncodeTexture;
    }

    const std::string& GetStatus() const {
        return m_Status;
    }

private:
    bool WriteAnnexBHeadersToUploadBuffer(std::vector<uint8_t>& annexBHeaders);
    bool SubmitMetadataReadback();
    bool BuildAv1DecodeInfo(nri::VideoEncodeFeedback& feedback, nri::VideoAV1EncodeDecodeInfo& av1DecodeInfo);
    bool IsEncodingAv1PFrame() const;
    bool NeedsSecondReconstructedPicture() const;
    bool NeedsThirdReconstructedPicture() const;

private:
    VideoContext m_Context = {};
    VideoConfig m_Config = {};
    VideoSize m_Size = {};
    CodecParameters m_CodecParameters = {};
    std::string m_Status = "Initializing encoder";

    nri::VideoSession* m_Session = nullptr;
    nri::VideoSessionParameters* m_Parameters = nullptr;
    nri::Texture* m_EncodeTexture = nullptr;
    nri::Texture* m_ReconstructedTexture = nullptr;
    nri::Texture* m_AV1PReconstructedTexture = nullptr;
    nri::Texture* m_BReconstructedTexture = nullptr;
    nri::Buffer* m_BitstreamHeaderUploadBuffer = nullptr;
    nri::Buffer* m_BitstreamBuffer = nullptr;
    nri::Buffer* m_MetadataBuffer = nullptr;
    nri::Buffer* m_ResolvedMetadataBuffer = nullptr;
    nri::Buffer* m_ResolvedMetadataReadbackBuffer = nullptr;
    nri::VideoPicture* m_EncodePicture = nullptr;
    nri::VideoPicture* m_ReconstructedPicture = nullptr;
    nri::VideoPicture* m_AV1PReconstructedPicture = nullptr;
    nri::VideoPicture* m_BReconstructedPicture = nullptr;
    nri::CommandAllocator* m_MetadataReadbackCommandAllocator = nullptr;
    nri::CommandBuffer* m_MetadataReadbackCommandBuffer = nullptr;
    nri::Fence* m_MetadataReadbackFence = nullptr;

    bool m_Ready = false;
    bool m_MetadataReadbackPending = false;
    uint32_t m_AV1PFrameStage = 0;
    uint32_t m_H26FrameStage = 0;
    nri::VideoEncodeFrameType m_PendingFrameType = nri::VideoEncodeFrameType::IDR;
    uint32_t m_PendingFrameIndex = 0;
    int32_t m_PendingPictureOrderCount = 0;
    uint32_t m_PendingOutputSlot = 0;
    bool m_PendingDisplayFrame = true;
    uint64_t m_MetadataReadbackFenceValue = 0;
};

} // namespace video_sample
