// © 2021 NVIDIA Corporation

#pragma once

#include "Shared.h"

namespace video_sample {

struct EncodeRequest {
    nri::Buffer* nv12Buffer = nullptr;
    nri::AccessStage nv12BufferState = {};
    const Nv12BufferLayout* nv12Layout = nullptr;
    uint32_t nv12Width = 0;
    uint32_t nv12Height = 0;
    VideoQuality quality = {};
    FramePlan plan = {};
};

class Encoder {
public:
    Encoder() = default;
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    bool Initialize(const VideoContext& context, const VideoConfig& config, const VideoSize& size, const CodecParameters& codecParameters);
    bool Encode(const EncodeRequest& request, EncodedFrame& encodedFrame);

    bool IsReady() const {
        return m_Ready;
    }

    bool IsUnsupported() const {
        return m_Unsupported;
    }

    const nri::VideoCapabilities& GetCapabilities() const {
        return m_Capabilities;
    }

    const std::string& GetStatus() const {
        return m_Status;
    }

private:
    bool UploadHeaders(const FramePlan& plan, std::vector<uint8_t>& headers);
    bool ResolveFeedback(nri::VideoEncodeFeedback& feedback, bool& feedbackResolved);
    bool ScanFeedback(nri::VideoEncodeFeedback& feedback);
    bool BuildAv1DecodeInfo(EncodedFrame& frame, const nri::VideoEncodePictureDesc& pictureDesc, const nri::VideoAV1EncodePictureDesc& av1PictureDesc, const nri::VideoAV1ReferenceDesc* av1References, uint32_t av1ReferenceNum);

private:
    VideoContext m_Context = {};
    VideoConfig m_Config = {};
    VideoSize m_Size = {};
    CodecParameters m_CodecParameters = {};
    nri::VideoCapabilities m_Capabilities = {};
    std::string m_Status = "Initializing encoder";

    nri::VideoSession* m_Session = nullptr;
    nri::VideoSessionParameters* m_Parameters = nullptr;
    std::vector<nri::Texture*> m_Textures;
    PictureSlot m_Input = {};
    PictureSlot m_Dpb[MAX_DPB_SLOT_NUM] = {};
    uint32_t m_DpbSlotNum = 0;
    nri::Buffer* m_BitstreamHeaderUploadBuffer = nullptr;
    nri::Buffer* m_BitstreamBuffer = nullptr;
    nri::Buffer* m_MetadataBuffer = nullptr;
    nri::Buffer* m_ResolvedMetadataBuffer = nullptr;         // VK: the readback buffer
    nri::Buffer* m_ResolvedMetadataReadbackBuffer = nullptr;
    nri::Queue* m_ResolveQueue = nullptr;

    bool m_Ready = false;
    bool m_Unsupported = false;
    bool m_UseFeedback = false;
    std::vector<uint8_t> m_Headers;
};

} // namespace video_sample
