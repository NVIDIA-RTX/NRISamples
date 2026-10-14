// © 2021 NVIDIA Corporation

#pragma once

#include "Shared.h"

namespace video_sample {

struct DecodeRequest {
    const EncodedFrame* frame = nullptr;
    nri::Buffer* nv12ReadbackBuffer = nullptr;
    const Nv12BufferLayout* nv12Layout = nullptr;
};

class Decoder {
public:
    Decoder() = default;
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    bool Initialize(const VideoContext& context, const VideoConfig& config, const VideoSize& size, const CodecParameters& codecParameters);
    bool Decode(const DecodeRequest& request, DecodedFrame& decodedFrame);

    bool IsReady() const {
        return m_Ready;
    }

    bool IsUnsupported() const {
        return m_Unsupported;
    }

    bool IsDistinct() const {
        return m_IsDistinct;
    }

    const nri::VideoCapabilities& GetCapabilities() const {
        return m_Capabilities;
    }

    const std::string& GetStatus() const {
        return m_Status;
    }

private:
    bool WriteAnnexBEndOfStream(std::vector<uint8_t>& annexBEndOfStream);
    bool BuildDecodeBitstream(const EncodedFrame& frame, uint64_t& decodeBitstreamRange, uint32_t& pictureOffset);

private:
    VideoContext m_Context = {};
    VideoConfig m_Config = {};
    VideoSize m_Size = {};
    CodecParameters m_CodecParameters = {};
    nri::VideoCapabilities m_Capabilities = {};
    std::string m_Status = "Initializing decoder";

    nri::VideoSession* m_Session = nullptr;
    nri::VideoSessionParameters* m_Parameters = nullptr;
    std::vector<nri::Texture*> m_Textures;
    PictureSlot m_Dpb[MAX_DPB_SLOT_NUM] = {};
    PictureSlot m_Output = {}; // distinct mode only
    uint32_t m_DpbSlotNum = 0;
    nri::Buffer* m_DecodeBitstreamBuffer = nullptr;

    bool m_Ready = false;
    bool m_Unsupported = false;
    bool m_IsDistinct = false;
};

} // namespace video_sample
