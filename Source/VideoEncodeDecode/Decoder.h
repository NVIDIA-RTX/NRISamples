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
    std::string m_Status = "Initializing decoder";

    nri::VideoSession* m_Session = nullptr;
    nri::VideoSessionParameters* m_Parameters = nullptr;
    nri::Texture* m_DecodeTexture = nullptr;
    nri::Texture* m_AV1PDecodeTexture = nullptr;
    nri::Buffer* m_DecodeBitstreamBuffer = nullptr;
    nri::VideoPicture* m_DecodePicture = nullptr;
    nri::VideoPicture* m_AV1PDecodePicture = nullptr;

    uint32_t m_DecodeFrameIndex = 0;
    bool m_Ready = false;
};

} // namespace video_sample
