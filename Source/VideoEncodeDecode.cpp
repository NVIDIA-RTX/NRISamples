// © 2021 NVIDIA Corporation

#if defined(_WIN32)
#    include <d3d12.h>
#endif

#include "NRIFramework.h"
#include "VideoEncodeDecode/Decoder.h"
#include "VideoEncodeDecode/Encoder.h"
#include "VideoEncodeDecode/Stream.h"

#include "Extensions/NRIVideo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

using video_sample::AlignUp;
using video_sample::DEFAULT_VIDEO_HEIGHT;
using video_sample::DEFAULT_VIDEO_WIDTH;
using video_sample::GetCodecName;
using video_sample::GetNriCodec;
using video_sample::MakeNv12BufferLayout;
using video_sample::Nv12BufferLayout;
using video_sample::OP_GENERATE_PATTERN;
using video_sample::OP_NV12_TO_PREVIEW;
using video_sample::PatternOperation;
using video_sample::QueuedFrame;
using video_sample::ROUND_TRIP_INTERVAL_SEC;
using video_sample::SampleCodec;
using video_sample::SubmitOneTime;

struct PatternConstants {
    uint32_t width = DEFAULT_VIDEO_WIDTH;
    uint32_t height = DEFAULT_VIDEO_HEIGHT;
    uint32_t yOffsetBytes = 0;
    uint32_t yRowPitchBytes = 0;
    uint32_t uvRowPitchBytes = 0;
    uint32_t uvOffsetBytes = 0;
    uint32_t operation = OP_GENERATE_PATTERN;
    float time = 0.0f;
    float motionSpeed = 1.0f;
    float detailStrength = 1.0f;
    float diffStrength = 6.0f;
    uint32_t showDifference = 0;
    uint32_t frameIndex = 0;
};

} // namespace

class Sample : public SampleBase {
public:
    Sample() = default;
    ~Sample();

    bool Initialize(nri::GraphicsAPI graphicsAPI, bool) override;
    void InitCmdLine(cmdline::parser& cmdLine) override;
    void ReadCmdLine(cmdline::parser& cmdLine) override;
    void LatencySleep(uint32_t frameIndex) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;
    bool AppShouldClose() override;

private:
    bool InitializeGraphics(nri::GraphicsAPI graphicsAPI);
    bool TryInitializePreviewTextures(nri::GraphicsAPI graphicsAPI);
    void InitializeGeneratedFrames(float timeSec);
    bool CanRunRoundTrip() const;
    void TryInitializeVideo(nri::GraphicsAPI graphicsAPI);
    PatternConstants MakePatternConstants(PatternOperation operation, float timeSec) const;
    bool GeneratePatternWithCompute(const PatternConstants& constants, nri::Descriptor* previewTexture, bool returnSourceBufferToShaderStorage = false, nri::AccessStage uploadBufferBefore = {});
    bool TryRunRoundTrip(float timeSec);
    int32_t RunHeadless(nri::GraphicsAPI graphicsAPI);
    void DrawTexturePanel(const char* label, nri::Descriptor* texture, const ImVec2& size);

private:
    NRIInterface NRI = {};
    nri::VideoInterface Video = {};
    std::unique_ptr<video_sample::Encoder> m_Encoder;
    std::unique_ptr<video_sample::Decoder> m_Decoder;
    video_sample::VideoConfig m_VideoConfig = {};
    video_sample::VideoQuality m_VideoQuality = {};
    video_sample::VideoSize m_VideoSize = {};
    video_sample::CodecParameters m_CodecParameters = {};
    nri::GraphicsAPI m_GraphicsAPI = nri::GraphicsAPI::NONE;

    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Queue* m_VideoEncodeQueue = nullptr;
    nri::Queue* m_VideoDecodeQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    nri::Texture* m_SourcePreviewTexture = nullptr;
    nri::Texture* m_DecodePreviewTexture = nullptr;
    nri::Buffer* m_UploadBuffer = nullptr;
    nri::Descriptor* m_UploadBufferView = nullptr;
    nri::Descriptor* m_SourcePreviewStorage = nullptr;
    nri::Descriptor* m_DecodePreviewStorage = nullptr;
    nri::Descriptor* m_SourcePreviewTextureView = nullptr;
    nri::Descriptor* m_DecodePreviewTextureView = nullptr;
    nri::PipelineLayout* m_GeneratePipelineLayout = nullptr;
    nri::Pipeline* m_GenerateComputePipeline = nullptr;
    nri::DescriptorPool* m_GenerateDescriptorPool = nullptr;
    nri::DescriptorSet* m_GenerateDescriptorSet = nullptr;
    std::vector<QueuedFrame> m_QueuedFrames;
    std::vector<SwapChainTexture> m_SwapChainTextures;

    Nv12BufferLayout m_Nv12Layout = {};
    std::vector<video_sample::FramePlan> m_Plans;
    size_t m_PlanIndex = 0;
    video_sample::StreamOptions m_StreamOptions = {};
    std::string m_StreamMode; // "", "stream", "matrix", "list", "decodeFile"
    std::string m_DecodeFile;
    std::string m_RefYuv;
    std::string m_CaseFilter;
    std::string m_GopArg;
    std::string m_DecodeDpbArg = "auto";
    std::string m_DpbLayoutArg = "auto";
    std::string m_EncodeInputArg = "outputOnly";
    std::string m_FeedbackArg = "on";
    std::string m_Av1InfoArg = "auto";
    std::string m_RateControlArg = "cqp";
    std::string m_RateControlChangeArg;
    std::string m_H264IdsArg = "0,0";
    std::string m_H265IdsArg = "0,0,0";
    std::string m_DumpArg;
    std::string m_NameArg = "stream";
    nri::Format m_SwapChainFormat = nri::Format::UNKNOWN;
    std::string m_VideoStatus = "Initializing video";
    std::string m_PreviewStatus = "Initializing preview";
    std::string m_CodecArg = "H264";
    std::string m_AV1FrameArg = "IDR";
    std::string m_H26FrameArg = "IDR";
    uint32_t m_VideoWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t m_VideoHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t m_CodedVideoWidth = DEFAULT_VIDEO_WIDTH;
    uint32_t m_CodedVideoHeight = DEFAULT_VIDEO_HEIGHT;
    uint32_t m_DecodeBitstreamSizeAlignment = 256;
    uint32_t m_QpI = 20;
    uint32_t m_QpP = 22;
    uint32_t m_QpB = 24;
    uint32_t m_AV1BaseQIndex = 20;
    bool m_Lossless = false;
    float m_PatternMotionSpeed = 1.0f;
    float m_PatternDetailStrength = 1.0f;
    float m_DiffStrength = 6.0f;
    PatternConstants m_PendingEncodedPatternConstants = {};
    SampleCodec m_Codec = SampleCodec::H264;
    video_sample::VisualFrameMode m_H26FrameMode = video_sample::VisualFrameMode::IDR;
    double m_StartTimeSec = 0.0;
    double m_LastRoundTripTimeSec = -ROUND_TRIP_INTERVAL_SEC;
    bool m_VideoReady = false;
    bool m_VideoQueuesRequested = false;
    bool m_EnhancedBarriers = false;
    bool m_DecodePreviewReady = false;
    bool m_PreviewTexturesShaderReadable = false;
    bool m_SourcePreviewReady = false;
    bool m_AV1PFrameVisual = false;
    bool m_ShowDifference = false;
    bool m_HasPendingEncodedPatternConstants = false;
    bool m_RequireVideoRoundTrip = false;
    bool m_VideoSizeExplicit = false;
    bool m_SmokeRoundTripComplete = false;
    bool m_SmokeRoundTripFailed = false;
    uint32_t m_PatternFrameIndex = 0;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        m_Decoder.reset();
        m_Encoder.reset();

        if (m_GenerateDescriptorPool)
            NRI.DestroyDescriptorPool(m_GenerateDescriptorPool);
        if (m_GenerateComputePipeline)
            NRI.DestroyPipeline(m_GenerateComputePipeline);
        if (m_GeneratePipelineLayout)
            NRI.DestroyPipelineLayout(m_GeneratePipelineLayout);
        if (m_UploadBufferView)
            NRI.DestroyDescriptor(m_UploadBufferView);
        if (m_UploadBuffer)
            NRI.DestroyBuffer(m_UploadBuffer);
        if (m_SourcePreviewStorage)
            NRI.DestroyDescriptor(m_SourcePreviewStorage);
        if (m_DecodePreviewStorage)
            NRI.DestroyDescriptor(m_DecodePreviewStorage);
        if (m_SourcePreviewTextureView)
            NRI.DestroyDescriptor(m_SourcePreviewTextureView);
        if (m_DecodePreviewTextureView)
            NRI.DestroyDescriptor(m_DecodePreviewTextureView);
        if (m_SourcePreviewTexture)
            NRI.DestroyTexture(m_SourcePreviewTexture);
        if (m_DecodePreviewTexture)
            NRI.DestroyTexture(m_DecodePreviewTexture);
        for (QueuedFrame& queuedFrame : m_QueuedFrames) {
            NRI.DestroyCommandBuffer(queuedFrame.commandBuffer);
            NRI.DestroyCommandAllocator(queuedFrame.commandAllocator);
        }

        for (SwapChainTexture& swapChainTexture : m_SwapChainTextures) {
            NRI.DestroyFence(swapChainTexture.acquireSemaphore);
            NRI.DestroyFence(swapChainTexture.releaseSemaphore);
            NRI.DestroyDescriptor(swapChainTexture.colorAttachment);
        }

        NRI.DestroyFence(m_FrameFence);
    }

    if (NRI.HasSwapChain())
        NRI.DestroySwapChain(m_SwapChain);

    if (NRI.HasStreamer())
        NRI.DestroyStreamer(m_Streamer);

    DestroyImgui();

    if (m_Device)
        nri::nriDestroyDevice(m_Device);
}

void Sample::InitCmdLine(cmdline::parser& cmdLine) {
    cmdLine.add<std::string>("codec", 0, "video codec: H264, H265, or AV1", false, m_CodecArg, cmdline::oneof<std::string>("H264", "H265", "AV1"));
    cmdLine.add<std::string>("av1Frame", 0, "AV1 visual frame permutation: IDR or P", false, m_AV1FrameArg, cmdline::oneof<std::string>("IDR", "P"));
    cmdLine.add<std::string>("h26Frame", 0, "H.264/H.265 visual frame permutation: IDR, P, or B", false, m_H26FrameArg, cmdline::oneof<std::string>("IDR", "P", "B"));
    cmdLine.add<uint32_t>("videoWidth", 0, "NV12 video encode/decode width", false, m_VideoWidth);
    cmdLine.add<uint32_t>("videoHeight", 0, "NV12 video encode/decode height", false, m_VideoHeight);
    cmdLine.add<uint32_t>("qpI", 0, "CQP quantizer for I/IDR frames", false, m_QpI);
    cmdLine.add<uint32_t>("qpP", 0, "CQP quantizer for P frames", false, m_QpP);
    cmdLine.add<uint32_t>("qpB", 0, "CQP quantizer for B frames", false, m_QpB);
    cmdLine.add<uint32_t>("av1BaseQIndex", 0, "AV1 base quantizer index", false, m_AV1BaseQIndex);
    cmdLine.add("lossless", 0, "force zero quantizers for lossless-capable codec modes");
    cmdLine.add("requireVideoRoundTrip", 0, "exit smoke runs only after a real encode/decode round trip");

    // Headless runs: encode -> decode of a generated frame sequence, PSNR check, exit code 0 = pass, 1 = fail, 2 = unsupported (NRI validation is always on)
    cmdLine.add("stream", 0, "headless: encode and decode a frame sequence with the options below, then exit");
    cmdLine.add("matrix", 0, "headless: run the built-in case matrix (640x360 base size), then exit");
    cmdLine.add("listCases", 0, "list the built-in matrix cases and exit");
    cmdLine.add<std::string>("case", 0, "matrix: run only the case with this name, or the cases whose name contains this text", false, "");
    cmdLine.add<uint32_t>("frames", 0, "stream: number of frames", false, 8);
    cmdLine.add<std::string>("gop", 0, "stream: kinds of the frames after the first IDR in display order, repeated: P, p (non-reference P), B (non-reference), I (IDR); e.g. BBP", false, m_GopArg);
    cmdLine.add<uint32_t>("refs", 0, "stream: maximum number of reference frames, 1..8 (B pictures need 2, AV1 uses 1)", false, 1);
    cmdLine.add<uint32_t>("idrPeriod", 0, "stream: IDR period in frames, 0 - only the first frame", false, 0);
    cmdLine.add<std::string>("decodeDpb", 0, "stream: decode output and DPB setup picture", false, m_DecodeDpbArg, cmdline::oneof<std::string>("auto", "coincide", "distinct"));
    cmdLine.add<std::string>("dpbLayout", 0, "stream: DPB pictures as layers of one texture or separate textures", false, m_DpbLayoutArg, cmdline::oneof<std::string>("auto", "array", "separate"));
    cmdLine.add<std::string>("encodeInput", 0, "stream: encode source texture usage", false, m_EncodeInputArg, cmdline::oneof<std::string>("outputOnly", "plain"));
    cmdLine.add<std::string>("rc", 0, "stream: rate control mode", false, m_RateControlArg, cmdline::oneof<std::string>("cqp", "cbr", "vbr"));
    cmdLine.add<uint32_t>("bitrate", 0, "stream: CBR/VBR target bitrate in kbit/s", false, 4000);
    cmdLine.add<uint32_t>("maxBitrate", 0, "stream: VBR peak bitrate in kbit/s, 0 - target", false, 0);
    cmdLine.add<std::string>("rcChange", 0, "stream: change rate control at a coding index, FRAME:MODE:KBPS[:MAXKBPS], e.g. 4:vbr:2000:4000", false, "");
    cmdLine.add<std::string>("feedback", 0, "stream: resolve encode feedback, 'off' scans the bitstream", false, m_FeedbackArg, cmdline::oneof<std::string>("on", "off"));
    cmdLine.add<std::string>("av1Info", 0, "stream: AV1 decode info source, metadata uses the submitted descriptions only (D3D12)", false, m_Av1InfoArg, cmdline::oneof<std::string>("auto", "header", "metadata"));
    cmdLine.add<std::string>("h264Ids", 0, "stream: H.264 SPS,PPS ids", false, m_H264IdsArg);
    cmdLine.add<uint32_t>("h264PpsRefs", 0, "stream: H.264 PPS num_ref_idx_l0_default_active_minus1", false, 0);
    cmdLine.add<uint32_t>("codedAlign", 0, "stream: minimum alignment of the coded size", false, 1);
    cmdLine.add<uint32_t>("qpMin", 0, "stream: CBR/VBR QP range minimum", false, 0);
    cmdLine.add<uint32_t>("qpMax", 0, "stream: CBR/VBR QP range maximum", false, 0);
    cmdLine.add<uint32_t>("maxFrameBits", 0, "stream: CBR/VBR maximum frame size in bits", false, 0);
    cmdLine.add<uint32_t>("vbvMs", 0, "stream: CBR/VBR virtual buffer size in ms", false, 0);
    cmdLine.add<uint32_t>("vbvInitialMs", 0, "stream: CBR/VBR initial virtual buffer fullness in ms", false, 0);
    cmdLine.add<std::string>("picturePlanes", 0, "video picture barriers address the planes with ALL, COLOR or PLANES (PLANE_0 | PLANE_1)", false, "all", cmdline::oneof<std::string>("all", "color", "planes"));
    cmdLine.add("enhancedBarriers", 0, "D3D12: use enhanced barriers (off by default for the video queues)");
    cmdLine.add<std::string>("h265Ids", 0, "stream: H.265 VPS,SPS,PPS ids", false, m_H265IdsArg);
    cmdLine.add("h265DefaultParams", 0, "stream: create the H.265 encode session without parameter sets (backend defaults, level by size)");
    cmdLine.add("noH265ListsModification", 0, "stream: clear the H.265 PPS lists_modification_present_flag");
    cmdLine.add("noReverseRefs", 0, "stream: H.265 encode references are always passed in the same order");
    cmdLine.add("noReverseDecodeRefs", 0, "stream: H.265 decode references are always passed in the same order");
    cmdLine.add<std::string>("decodeFile", 0, "headless: decode an AV1 OBU file (one OBU_FRAME per picture, e.g. from libaom) instead of encoding", false, "");
    cmdLine.add<std::string>("refYuv", 0, "decodeFile: NV12 reference of the shown pictures, the decode must be bit-exact", false, "");
    cmdLine.add("verbose", 0, "headless: log every picture before it is submitted");
    cmdLine.add("encodeOnly", 0, "stream: skip the decoder (the encoded stream is still dumped)");
    cmdLine.add<std::string>("dump", 0, "headless: write stream, decoded.yuv, source.yuv and frames.csv of every case into this directory", false, "");
    cmdLine.add<double>("minPsnr", 0, "stream: minimum luma PSNR in dB per frame", false, 30.0);
    cmdLine.add<std::string>("name", 0, "stream: name of the run (dump subdirectory)", false, m_NameArg);
}

static bool ParseIds(const std::string& text, uint32_t* values, uint32_t count) {
    uint32_t parsed = 0;
    const char* p = text.c_str();
    while (*p && parsed < count) {
        char* end = nullptr;
        const unsigned long value = std::strtoul(p, &end, 10);
        if (end == p || value > 255)
            return false;
        values[parsed++] = (uint32_t)value;
        p = *end == ',' ? end + 1 : end;
    }
    return parsed == count && !*p;
}

void Sample::ReadCmdLine(cmdline::parser& cmdLine) {
    m_CodecArg = cmdLine.get<std::string>("codec");
    m_AV1FrameArg = cmdLine.get<std::string>("av1Frame");
    m_H26FrameArg = cmdLine.get<std::string>("h26Frame");
    m_VideoWidth = cmdLine.get<uint32_t>("videoWidth");
    m_VideoHeight = cmdLine.get<uint32_t>("videoHeight");
    if (!cmdLine.exist("videoWidth") && cmdLine.exist("width"))
        m_VideoWidth = m_OutputResolution.x;
    if (!cmdLine.exist("videoHeight") && cmdLine.exist("height"))
        m_VideoHeight = m_OutputResolution.y;
    m_QpI = cmdLine.get<uint32_t>("qpI");
    m_QpP = cmdLine.get<uint32_t>("qpP");
    m_QpB = cmdLine.get<uint32_t>("qpB");
    m_AV1BaseQIndex = cmdLine.get<uint32_t>("av1BaseQIndex");
    m_Lossless = cmdLine.exist("lossless");
    m_RequireVideoRoundTrip = cmdLine.exist("requireVideoRoundTrip");
    m_Codec = m_CodecArg == "H265" ? SampleCodec::H265 : (m_CodecArg == "AV1" ? SampleCodec::AV1 : SampleCodec::H264);
    m_H26FrameMode = m_H26FrameArg == "B" ? video_sample::VisualFrameMode::B : (m_H26FrameArg == "P" ? video_sample::VisualFrameMode::P : video_sample::VisualFrameMode::IDR);
    m_AV1PFrameVisual = m_Codec == SampleCodec::AV1 && m_AV1FrameArg == "P";

    // Headless
    m_StreamMode = cmdLine.exist("listCases") ? "list" : (cmdLine.exist("matrix") ? "matrix" : (cmdLine.exist("stream") ? "stream" : ""));
    if (cmdLine.exist("decodeFile")) {
        m_StreamMode = "decodeFile";
        m_DecodeFile = cmdLine.get<std::string>("decodeFile");
        m_RefYuv = cmdLine.get<std::string>("refYuv");
    }
    m_CaseFilter = cmdLine.get<std::string>("case");
    m_VideoSizeExplicit = cmdLine.exist("videoWidth") || cmdLine.exist("videoHeight");

    video_sample::StreamOptions& options = m_StreamOptions;
    options.name = cmdLine.get<std::string>("name");
    options.config.codec = m_Codec;
    options.config.videoWidth = m_VideoWidth;
    options.config.videoHeight = m_VideoHeight;
    options.config.referenceNum = cmdLine.get<uint32_t>("refs");
    options.config.h264TransformBypass = m_Lossless;
    options.config.h265ListsModification = !cmdLine.exist("noH265ListsModification");
    options.config.reverseReferences = !cmdLine.exist("noReverseRefs");
    options.config.reverseDecodeReferences = !cmdLine.exist("noReverseDecodeRefs");
    options.config.codedAlignment = std::max(cmdLine.get<uint32_t>("codedAlign"), 1u);
    options.config.h264PpsRefIdxL0DefaultActiveMinus1 = (uint8_t)cmdLine.get<uint32_t>("h264PpsRefs");
    const std::string decodeDpb = cmdLine.get<std::string>("decodeDpb");
    options.config.decodeDpb = decodeDpb == "coincide" ? video_sample::DecodeDpbMode::COINCIDE : (decodeDpb == "distinct" ? video_sample::DecodeDpbMode::DISTINCT : video_sample::DecodeDpbMode::AUTO);
    const std::string dpbLayout = cmdLine.get<std::string>("dpbLayout");
    options.config.dpbLayout = dpbLayout == "array" ? video_sample::DpbLayout::ARRAY : (dpbLayout == "separate" ? video_sample::DpbLayout::SEPARATE : video_sample::DpbLayout::AUTO);
    options.config.encodeInputOutputOnly = cmdLine.get<std::string>("encodeInput") == "outputOnly";
    options.config.encodeFeedback = cmdLine.get<std::string>("feedback") == "on";
    const std::string av1Info = cmdLine.get<std::string>("av1Info");
    options.config.av1Info = av1Info == "header" ? video_sample::Av1InfoMode::HEADER : (av1Info == "metadata" ? video_sample::Av1InfoMode::METADATA : video_sample::Av1InfoMode::AUTO);
    options.config.h265DefaultParameters = cmdLine.exist("h265DefaultParams");
    uint32_t h264Ids[2] = {};
    uint32_t h265Ids[3] = {};
    if (!ParseIds(cmdLine.get<std::string>("h264Ids"), h264Ids, 2) || !ParseIds(cmdLine.get<std::string>("h265Ids"), h265Ids, 3))
        std::fprintf(stderr, "Invalid --h264Ids / --h265Ids, expected comma-separated numbers\n");
    options.config.h264SpsId = (uint8_t)h264Ids[0];
    options.config.h264PpsId = (uint8_t)h264Ids[1];
    options.config.h265VpsId = (uint8_t)h265Ids[0];
    options.config.h265SpsId = (uint8_t)h265Ids[1];
    options.config.h265PpsId = (uint8_t)h265Ids[2];

    options.frameNum = cmdLine.get<uint32_t>("frames");
    options.gop = cmdLine.get<std::string>("gop");
    options.idrPeriod = cmdLine.get<uint32_t>("idrPeriod");
    options.minPsnr = cmdLine.get<double>("minPsnr");
    options.encodeOnly = cmdLine.exist("encodeOnly");
    options.verbose = cmdLine.exist("verbose");
    options.dumpDir = cmdLine.get<std::string>("dump");

    options.quality.qpI = m_QpI;
    options.quality.qpP = m_QpP;
    options.quality.qpB = m_QpB;
    options.quality.av1BaseQIndex = m_AV1BaseQIndex;
    options.quality.lossless = m_Lossless;
    const std::string rc = cmdLine.get<std::string>("rc");
    options.quality.rateControl.kind = rc == "cbr" ? video_sample::RateControlKind::CBR : (rc == "vbr" ? video_sample::RateControlKind::VBR : video_sample::RateControlKind::CQP);
    options.quality.rateControl.bitrateKbps = cmdLine.get<uint32_t>("bitrate");
    options.quality.rateControl.maxBitrateKbps = cmdLine.get<uint32_t>("maxBitrate");
    options.quality.rateControl.qpMin = cmdLine.get<uint32_t>("qpMin");
    options.quality.rateControl.qpMax = cmdLine.get<uint32_t>("qpMax");
    options.quality.rateControl.maxFrameBits = cmdLine.get<uint32_t>("maxFrameBits");
    options.quality.rateControl.vbvMs = cmdLine.get<uint32_t>("vbvMs");
    options.quality.rateControl.vbvInitialMs = cmdLine.get<uint32_t>("vbvInitialMs");
    m_EnhancedBarriers = cmdLine.exist("enhancedBarriers");
    const std::string picturePlanes = cmdLine.get<std::string>("picturePlanes");
    video_sample::g_PicturePlanes = picturePlanes == "color" ? nri::PlaneBits::COLOR : (picturePlanes == "planes" ? (nri::PlaneBits::PLANE_0 | nri::PlaneBits::PLANE_1) : nri::PlaneBits::ALL);

    // FRAME:MODE:KBPS[:MAXKBPS]
    const std::string rcChange = cmdLine.get<std::string>("rcChange");
    if (!rcChange.empty()) {
        char mode[8] = {};
        uint32_t frame = 0;
        uint32_t kbps = 0;
        uint32_t maxKbps = 0;
        const int32_t parsed = std::sscanf(rcChange.c_str(), "%u:%7[a-z]:%u:%u", &frame, mode, &kbps, &maxKbps);
        if (parsed < 2) {
            std::fprintf(stderr, "Invalid --rcChange, expected FRAME:MODE:KBPS[:MAXKBPS]\n");
        } else {
            options.rateControlChangeFrame = (int32_t)frame;
            options.rateControlB.kind = !std::strcmp(mode, "cbr") ? video_sample::RateControlKind::CBR : (!std::strcmp(mode, "vbr") ? video_sample::RateControlKind::VBR : video_sample::RateControlKind::CQP);
            options.rateControlB.bitrateKbps = parsed >= 3 ? kbps : 4000;
            options.rateControlB.maxBitrateKbps = parsed >= 4 ? maxKbps : 0;
        }
    }

    if (!m_VideoWidth || !m_VideoHeight || (m_VideoWidth % 4) != 0 || (m_VideoHeight % 2) != 0 || m_VideoWidth > 65535 || m_VideoHeight > 65535) {
        m_VideoStatus = "Video size must be non-zero, width must be divisible by 4, height must be even, and both dimensions must fit 16-bit video descriptors";
        std::fprintf(stderr, "%s\n", m_VideoStatus.c_str());
        std::exit(1);
    }
    const uint32_t minCodecQp = m_Codec == SampleCodec::AV1 ? 1 : 0;
    const uint32_t maxCodecQp = m_Codec == SampleCodec::AV1 ? 255 : 51;
    if (m_QpI < minCodecQp || m_QpI > maxCodecQp || m_QpP < minCodecQp || m_QpP > maxCodecQp || m_QpB < minCodecQp || m_QpB > maxCodecQp || m_AV1BaseQIndex < minCodecQp || m_AV1BaseQIndex > 255) {
        m_VideoStatus = m_Codec == SampleCodec::AV1 ? "AV1 quantizers must be in the 1..255 range on current hardware encode paths" : "H.264/H.265 QP values must be in the 0..51 range";
        std::fprintf(stderr, "%s\n", m_VideoStatus.c_str());
        std::exit(1);
    }

    // Headless modes finish before SampleBase initializes GLFW or creates a window.
    if (!m_StreamMode.empty()) {
        const std::string& api = cmdLine.get<std::string>("api");
        nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
        if (api == "D3D11")
            graphicsAPI = nri::GraphicsAPI::D3D11;
        else if (api == "D3D12")
            graphicsAPI = nri::GraphicsAPI::D3D12;
        else if (api == "METAL")
            graphicsAPI = nri::GraphicsAPI::METAL;
        else if (api == "WGPU")
            graphicsAPI = nri::GraphicsAPI::WGPU;

        std::exit(RunHeadless(graphicsAPI));
    }
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    m_GraphicsAPI = graphicsAPI;
    m_VideoConfig.codec = m_Codec;
    m_VideoConfig.videoWidth = m_VideoWidth;
    m_VideoConfig.videoHeight = m_VideoHeight;
    m_VideoConfig.h264TransformBypass = m_Lossless;
    m_VideoConfig.av1PFrameVisual = m_AV1PFrameVisual;
    m_VideoConfig.h26FrameMode = m_H26FrameMode;
    m_VideoConfig.hasBFrames = m_Codec != SampleCodec::AV1 && m_H26FrameMode == video_sample::VisualFrameMode::B;
    m_VideoConfig.referenceNum = m_VideoConfig.hasBFrames ? 2 : 1;
    m_VideoQuality.qpI = m_QpI;
    m_VideoQuality.qpP = m_QpP;
    m_VideoQuality.qpB = m_QpB;
    m_VideoQuality.av1BaseQIndex = m_AV1BaseQIndex;
    m_VideoQuality.lossless = m_Lossless;

    // Interactive cycle: IDR, IDR + P, or IDR + P + B (decoded as IDR, P, B), the last picture is displayed
    const char* gop = m_Codec == SampleCodec::AV1 ? (m_AV1PFrameVisual ? "P" : "I") : (m_H26FrameMode == video_sample::VisualFrameMode::B ? "BP" : (m_H26FrameMode == video_sample::VisualFrameMode::P ? "P" : "I"));
    m_Plans = video_sample::BuildFramePlans(m_VideoConfig, gop, 1 + (uint32_t)std::strlen(gop) - (gop[0] == 'I' ? 1 : 0), 0);
    m_VideoConfig.reorderNum = video_sample::GetReorderNum(m_Plans);
    for (video_sample::FramePlan& plan : m_Plans)
        plan.isDisplay = &plan == &m_Plans.back();

    if (!InitializeGraphics(graphicsAPI))
        return false;

    m_StartTimeSec = m_Timer.GetTimeStamp() * 0.001;
    if (!TryInitializePreviewTextures(graphicsAPI))
        return false;
    TryInitializeVideo(graphicsAPI);
    InitializeGeneratedFrames(0.0f);

    return InitImgui(*m_Device);
}

bool Sample::InitializeGraphics(nri::GraphicsAPI graphicsAPI) {
    nri::AdapterDesc adapterDesc[2] = {};
    uint32_t adapterDescsNum = helper::GetCountOf(adapterDesc);
    NRI_ABORT_ON_FAILURE(nri::nriEnumerateAdapters(adapterDesc, adapterDescsNum));

    const nri::AdapterDesc& selectedAdapter = adapterDesc[std::min(m_AdapterIndex, adapterDescsNum - 1)];
    m_VideoQueuesRequested = graphicsAPI != nri::GraphicsAPI::D3D11 && selectedAdapter.queueNum[(uint32_t)nri::QueueType::VIDEO_ENCODE] && selectedAdapter.queueNum[(uint32_t)nri::QueueType::VIDEO_DECODE];

    nri::QueueFamilyDesc queueFamilies[3] = {};
    uint32_t queueFamilyNum = 0;
    queueFamilies[queueFamilyNum].queueNum = 1;
    queueFamilies[queueFamilyNum++].queueType = nri::QueueType::GRAPHICS;
    if (m_VideoQueuesRequested) {
        queueFamilies[queueFamilyNum].queueNum = 1;
        queueFamilies[queueFamilyNum++].queueType = nri::QueueType::VIDEO_ENCODE;
        queueFamilies[queueFamilyNum].queueNum = 1;
        queueFamilies[queueFamilyNum++].queueType = nri::QueueType::VIDEO_DECODE;
    }

    nri::DeviceCreationDesc deviceCreationDesc = {};

    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_ENABLE_COMMAND_BUFFER_EMULATION;
    // D3D12 video encode command lists fail to close with enhanced barriers enabled.
    deviceCreationDesc.disableD3D12EnhancedBarriers = graphicsAPI == nri::GraphicsAPI::D3D12 ? !m_EnhancedBarriers : D3D12_DISABLE_ENHANCED_BARRIERS;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &selectedAdapter;
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = queueFamilyNum;
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));
    m_Nv12Layout = MakeNv12BufferLayout(NRI.GetDeviceDesc(*m_Device), m_VideoWidth, m_VideoHeight);

    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX | nri::BufferUsageBits::INDEX};
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.queuedFrameNum = GetQueuedFrameNum();
    streamerDesc.hostDataCapacity = IMGUI_HOST_DATA_CAPACITY;
    NRI_ABORT_ON_FAILURE(NRI.CreateStreamer(*m_Device, streamerDesc, m_Streamer));

    NRI_ABORT_ON_FAILURE(NRI.GetQueue(*m_Device, nri::QueueType::GRAPHICS, 0, m_GraphicsQueue));
    NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, 0, m_FrameFence));

    nri::SwapChainDesc swapChainDesc = {};
    swapChainDesc.window = GetWindow();
    swapChainDesc.queue = m_GraphicsQueue;
    swapChainDesc.format = nri::SwapChainFormat::BT709_G22_8BIT;
    swapChainDesc.flags = (m_Vsync ? nri::SwapChainBits::VSYNC : nri::SwapChainBits::NONE) | nri::SwapChainBits::ALLOW_TEARING;
    swapChainDesc.width = (uint16_t)GetOutputResolution().x;
    swapChainDesc.height = (uint16_t)GetOutputResolution().y;
    swapChainDesc.textureNum = GetOptimalSwapChainTextureNum();
    swapChainDesc.queuedFrameNum = GetQueuedFrameNum();
    NRI_ABORT_ON_FAILURE(NRI.CreateSwapChain(*m_Device, swapChainDesc, m_SwapChain));

    uint32_t swapChainTextureNum;
    nri::Texture* const* swapChainTextures = NRI.GetSwapChainTextures(*m_SwapChain, swapChainTextureNum);
    m_SwapChainFormat = NRI.GetTextureDesc(*swapChainTextures[0]).format;

    for (uint32_t i = 0; i < swapChainTextureNum; i++) {
        nri::TextureViewDesc textureViewDesc = {swapChainTextures[i], nri::TextureView::COLOR_ATTACHMENT, m_SwapChainFormat};

        nri::Descriptor* colorAttachment = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateTextureView(textureViewDesc, colorAttachment));

        nri::Fence* acquireSemaphore = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, acquireSemaphore));

        nri::Fence* releaseSemaphore = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateFence(*m_Device, nri::SWAPCHAIN_SEMAPHORE, releaseSemaphore));

        SwapChainTexture& swapChainTexture = m_SwapChainTextures.emplace_back();
        swapChainTexture = {};
        swapChainTexture.acquireSemaphore = acquireSemaphore;
        swapChainTexture.releaseSemaphore = releaseSemaphore;
        swapChainTexture.texture = swapChainTextures[i];
        swapChainTexture.colorAttachment = colorAttachment;
        swapChainTexture.attachmentFormat = m_SwapChainFormat;
    }

    m_QueuedFrames.resize(GetQueuedFrameNum() + swapChainTextureNum);
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    return true;
}

PatternConstants Sample::MakePatternConstants(PatternOperation operation, float timeSec) const {
    PatternConstants patternConstants = {};
    patternConstants.width = m_VideoWidth;
    patternConstants.height = m_VideoHeight;
    patternConstants.yRowPitchBytes = m_Nv12Layout.yRowPitchBytes;
    patternConstants.uvRowPitchBytes = m_Nv12Layout.uvRowPitchBytes;
    patternConstants.uvOffsetBytes = (uint32_t)m_Nv12Layout.uvOffsetBytes;
    patternConstants.operation = operation;
    patternConstants.time = timeSec;
    patternConstants.motionSpeed = m_PatternMotionSpeed;
    patternConstants.detailStrength = m_PatternDetailStrength;
    patternConstants.diffStrength = m_DiffStrength;
    patternConstants.showDifference = m_ShowDifference ? 1 : 0;
    patternConstants.frameIndex = m_PatternFrameIndex;
    return patternConstants;
}

void Sample::InitializeGeneratedFrames(float timeSec) {
    if (!m_SourcePreviewTexture || !m_SourcePreviewStorage || !m_UploadBuffer || !m_UploadBufferView)
        return;

    PatternConstants patternConstants = MakePatternConstants(OP_GENERATE_PATTERN, timeSec);
    if (!GeneratePatternWithCompute(patternConstants, m_SourcePreviewStorage, true)) {
        m_PreviewStatus = "Failed to generate source pattern via compute";
        return;
    }

    m_PreviewStatus = "Source preview is generated by compute";
    m_SourcePreviewReady = true;
}

bool Sample::GeneratePatternWithCompute(const PatternConstants& constants, nri::Descriptor* previewTexture, bool returnSourceBufferToShaderStorage, nri::AccessStage uploadBufferBefore) {
    if (!m_GeneratePipelineLayout || !m_GenerateComputePipeline || !m_GenerateDescriptorSet || !m_UploadBufferView || !previewTexture)
        return false;
    if ((constants.width % 4) != 0 || (constants.height % 2) != 0)
        return false;

    const uint32_t dispatchX = (constants.width + 3) / 4;
    const uint32_t dispatchY = constants.height;

    const bool previewTexturesShaderReadable = m_PreviewTexturesShaderReadable;
    const bool submitted = SubmitOneTime(
        NRI,
        *m_GraphicsQueue,
        m_GenerateDescriptorPool,
        [this, &constants, dispatchX, dispatchY, previewTexturesShaderReadable, returnSourceBufferToShaderStorage, uploadBufferBefore](nri::CommandBuffer& commandBuffer) {
            nri::SetDescriptorSetDesc descriptorSet = {0, m_GenerateDescriptorSet};

            nri::BufferBarrierDesc bufferBarrier = {};
            bufferBarrier.buffer = m_UploadBuffer;
            bufferBarrier.before = uploadBufferBefore;
            bufferBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

            nri::TextureBarrierDesc textureBarriers[2] = {};
            textureBarriers[0].texture = m_SourcePreviewTexture;
            textureBarriers[1].texture = m_DecodePreviewTexture;
            for (nri::TextureBarrierDesc& textureBarrier : textureBarriers) {
                if (previewTexturesShaderReadable)
                    textureBarrier.before = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::ALL};
                else
                    textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
                textureBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
                textureBarrier.mipNum = nri::REMAINING;
                textureBarrier.layerNum = nri::REMAINING;
                textureBarrier.planes = nri::PlaneBits::ALL;
            }

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = &bufferBarrier;
            barrierDesc.bufferNum = 1;
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);

            nri::SetRootConstantsDesc rootConstants = {0, &constants, sizeof(PatternConstants)};
            NRI.CmdBarrier(commandBuffer, barrierDesc);
            NRI.CmdSetPipelineLayout(commandBuffer, nri::BindPoint::COMPUTE, *m_GeneratePipelineLayout);
            NRI.CmdSetDescriptorSet(commandBuffer, descriptorSet);
            NRI.CmdSetPipeline(commandBuffer, *m_GenerateComputePipeline);
            NRI.CmdSetRootConstants(commandBuffer, rootConstants);
            NRI.CmdDispatch(commandBuffer, {dispatchX, dispatchY, 1});

            bufferBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
            if (returnSourceBufferToShaderStorage)
                bufferBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
            else
                bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
            for (nri::TextureBarrierDesc& textureBarrier : textureBarriers) {
                textureBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
                textureBarrier.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::ALL};
            }
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        });

    if (submitted)
        m_PreviewTexturesShaderReadable = true;

    return submitted;
}

bool Sample::TryInitializePreviewTextures(nri::GraphicsAPI) {
    if (m_SourcePreviewTexture && m_DecodePreviewTexture && m_SourcePreviewTextureView && m_DecodePreviewTextureView)
        return true;

    nri::TextureDesc previewTextureDesc = {};
    previewTextureDesc.type = nri::TextureType::TEXTURE_2D;
    previewTextureDesc.format = nri::Format::RGBA8_UNORM;
    previewTextureDesc.width = (nri::Dim_t)m_VideoWidth;
    previewTextureDesc.height = (nri::Dim_t)m_VideoHeight;
    previewTextureDesc.mipNum = 1;
    previewTextureDesc.layerNum = 1;
    previewTextureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE | nri::TextureUsageBits::SHADER_RESOURCE_STORAGE;

    if (!m_SourcePreviewTexture) {
        if (NRI.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, previewTextureDesc, m_SourcePreviewTexture) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create source preview texture";
            return false;
        }
        NRI.SetDebugName(m_SourcePreviewTexture, "VideoSourcePreviewTexture");
    }

    if (!m_DecodePreviewTexture) {
        if (NRI.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, previewTextureDesc, m_DecodePreviewTexture) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create decode preview texture";
            return false;
        }
        NRI.SetDebugName(m_DecodePreviewTexture, "VideoDecodePreviewTexture");
    }

    if (!m_PreviewTexturesShaderReadable) {
        const bool initialized = SubmitOneTime(NRI, *m_GraphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarriers[2] = {};
            textureBarriers[0].texture = m_SourcePreviewTexture;
            textureBarriers[1].texture = m_DecodePreviewTexture;

            for (nri::TextureBarrierDesc& textureBarrier : textureBarriers) {
                textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
                textureBarrier.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::ALL};
                textureBarrier.mipNum = nri::REMAINING;
                textureBarrier.layerNum = nri::REMAINING;
                textureBarrier.planes = nri::PlaneBits::ALL;
            }

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        });

        if (!initialized) {
            m_PreviewStatus = "Failed to initialize preview texture layouts";
            return false;
        }

        m_PreviewTexturesShaderReadable = true;
    }

    nri::TextureViewDesc sourceTextureViewDesc = {m_SourcePreviewTexture, nri::TextureView::TEXTURE, previewTextureDesc.format};
    nri::TextureViewDesc decodeTextureViewDesc = {m_DecodePreviewTexture, nri::TextureView::TEXTURE, previewTextureDesc.format};

    if (!m_SourcePreviewTextureView) {
        if (NRI.CreateTextureView(sourceTextureViewDesc, m_SourcePreviewTextureView) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create source preview ImGui texture view";
            return false;
        }
    }

    if (!m_DecodePreviewTextureView) {
        if (NRI.CreateTextureView(decodeTextureViewDesc, m_DecodePreviewTextureView) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create decode preview ImGui texture view";
            return false;
        }
    }

    nri::TextureViewDesc sourceStorageTextureViewDesc = {m_SourcePreviewTexture, nri::TextureView::STORAGE_TEXTURE, nri::Format::RGBA8_UNORM};
    nri::TextureViewDesc decodeStorageTextureViewDesc = {m_DecodePreviewTexture, nri::TextureView::STORAGE_TEXTURE, nri::Format::RGBA8_UNORM};

    if (!m_SourcePreviewStorage) {
        if (NRI.CreateTextureView(sourceStorageTextureViewDesc, m_SourcePreviewStorage) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create source preview storage texture view";
            return false;
        }
    }

    if (!m_DecodePreviewStorage) {
        if (NRI.CreateTextureView(decodeStorageTextureViewDesc, m_DecodePreviewStorage) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create decode preview storage texture view";
            return false;
        }
    }

    if (!m_UploadBuffer) {
        nri::BufferDesc uploadBufferDesc = {};
        uploadBufferDesc.size = m_Nv12Layout.totalSizeBytes;
        uploadBufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;

        if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, uploadBufferDesc, m_UploadBuffer) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create compute NV12 buffer";
            return false;
        }
    }

    if (!m_UploadBufferView) {
        nri::BufferViewDesc uploadBufferViewDesc = {};
        uploadBufferViewDesc.buffer = m_UploadBuffer;
        uploadBufferViewDesc.format = nri::Format::R32_UINT;
        uploadBufferViewDesc.type = nri::BufferView::STORAGE_BUFFER;
        uploadBufferViewDesc.size = NRI.GetBufferDesc(*m_UploadBuffer).size;

        if (NRI.CreateBufferView(uploadBufferViewDesc, m_UploadBufferView) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create compute NV12 buffer view";
            return false;
        }
    }

    if (!m_GeneratePipelineLayout) {
        nri::DescriptorRangeDesc descriptorRanges[] = {
            {0, 1, nri::DescriptorType::STORAGE_BUFFER, nri::StageBits::COMPUTE_SHADER},
            {1, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER},
            {2, 1, nri::DescriptorType::STORAGE_TEXTURE, nri::StageBits::COMPUTE_SHADER},
        };
        nri::DescriptorSetDesc descriptorSetDescs[] = {{0, descriptorRanges, helper::GetCountOf(descriptorRanges)}};

        nri::RootConstantDesc rootConstantDesc = {};
        rootConstantDesc.registerIndex = 0;
        rootConstantDesc.size = sizeof(PatternConstants);
        rootConstantDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.rootConstantNum = 1;
        pipelineLayoutDesc.rootConstants = &rootConstantDesc;
        pipelineLayoutDesc.descriptorSetNum = helper::GetCountOf(descriptorSetDescs);
        pipelineLayoutDesc.descriptorSets = descriptorSetDescs;
        pipelineLayoutDesc.shaderStages = nri::StageBits::COMPUTE_SHADER;
        if (NRI.CreatePipelineLayout(*m_Device, pipelineLayoutDesc, m_GeneratePipelineLayout) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create compute pipeline layout for pattern generation";
            return false;
        }
    }

    if (!m_GenerateComputePipeline) {
        const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
        utils::ShaderCodeStorage shaderCodeStorage;
        nri::ComputePipelineDesc computePipelineDesc = {};
        computePipelineDesc.pipelineLayout = m_GeneratePipelineLayout;
        computePipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "VideoEncodePattern.cs", shaderCodeStorage);
        if (NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_GenerateComputePipeline) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create pattern generation compute pipeline";
            return false;
        }
    }

    if (!m_GenerateDescriptorPool) {
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.storageBufferMaxNum = 1;
        descriptorPoolDesc.storageTextureMaxNum = 2;
        if (NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_GenerateDescriptorPool) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to create compute descriptor pool for pattern generation";
            return false;
        }
    }

    if (!m_GenerateDescriptorSet) {
        if (NRI.AllocateDescriptorSets(*m_GenerateDescriptorPool, *m_GeneratePipelineLayout, 0, &m_GenerateDescriptorSet, 1, 0) != nri::Result::SUCCESS) {
            m_PreviewStatus = "Failed to allocate compute descriptor set for pattern generation";
            return false;
        }

        nri::UpdateDescriptorRangeDesc updateDescriptorRangeDescs[] = {
            {m_GenerateDescriptorSet, 0, 0, &m_UploadBufferView, 1},
            {m_GenerateDescriptorSet, 1, 0, &m_SourcePreviewStorage, 1},
            {m_GenerateDescriptorSet, 2, 0, &m_DecodePreviewStorage, 1},
        };
        NRI.UpdateDescriptorRanges(updateDescriptorRangeDescs, helper::GetCountOf(updateDescriptorRangeDescs));
    }

    return true;
}

void Sample::TryInitializeVideo(nri::GraphicsAPI graphicsAPI) {
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);

    if (graphicsAPI == nri::GraphicsAPI::D3D11) {
        m_VideoStatus = "D3D11 does not expose NRI video queues";
        return;
    }

    if (!m_VideoQueuesRequested || !deviceDesc.adapterDesc.queueNum[(uint32_t)nri::QueueType::VIDEO_ENCODE] || !deviceDesc.adapterDesc.queueNum[(uint32_t)nri::QueueType::VIDEO_DECODE]) {
        m_VideoStatus = "Adapter has no NRI video encode/decode queues";
        return;
    }

    if (nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::VideoInterface), &Video) != nri::Result::SUCCESS || !Video.CreateVideoSession) {
        m_VideoStatus = "NRI video interface is unavailable";
        return;
    }

    if (NRI.GetQueue(*m_Device, nri::QueueType::VIDEO_ENCODE, 0, m_VideoEncodeQueue) != nri::Result::SUCCESS || NRI.GetQueue(*m_Device, nri::QueueType::VIDEO_DECODE, 0, m_VideoDecodeQueue) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to get video queues";
        return;
    }

    video_sample::VideoContext videoContext = {};
    videoContext.nri = &NRI;
    videoContext.video = &Video;
    videoContext.device = m_Device;
    videoContext.graphicsQueue = m_GraphicsQueue;
    videoContext.encodeQueue = m_VideoEncodeQueue;
    videoContext.decodeQueue = m_VideoDecodeQueue;
    videoContext.graphicsAPI = graphicsAPI;

    bool isUnsupported = false;
    if (!video_sample::QueryVideoSize(videoContext, m_VideoConfig, m_VideoSize, isUnsupported, m_VideoStatus))
        return;

    m_CodedVideoWidth = m_VideoSize.codedWidth;
    m_CodedVideoHeight = m_VideoSize.codedHeight;
    m_CodecParameters = video_sample::MakeCodecParameters(graphicsAPI, m_VideoConfig, m_CodedVideoWidth, m_CodedVideoHeight);

    m_Encoder = std::make_unique<video_sample::Encoder>();
    if (!m_Encoder->Initialize(videoContext, m_VideoConfig, m_VideoSize, m_CodecParameters)) {
        m_VideoStatus = m_Encoder->GetStatus();
        return;
    }

    m_Decoder = std::make_unique<video_sample::Decoder>();
    if (!m_Decoder->Initialize(videoContext, m_VideoConfig, m_VideoSize, m_CodecParameters)) {
        m_VideoStatus = m_Decoder->GetStatus();
        return;
    }

    m_VideoReady = true;
    m_VideoStatus = std::string("NRI video queues and ") + GetCodecName(m_Codec) + " encode/decode objects initialized";
}

// Encodes and decodes the next picture of the cycle, returns true after the displayed picture
bool Sample::TryRunRoundTrip(float timeSec) {
    if (!CanRunRoundTrip()) {
        m_VideoStatus = std::string(GetCodecName(m_Codec)) + " round trip is not currently supported in this configuration";
        return false;
    }

    const video_sample::FramePlan& plan = m_Plans[m_PlanIndex];
    m_PlanIndex = (m_PlanIndex + 1) % m_Plans.size();

    m_PatternFrameIndex++;
    PatternConstants patternConstants = MakePatternConstants(OP_GENERATE_PATTERN, timeSec);
    if (!GeneratePatternWithCompute(patternConstants, m_SourcePreviewStorage, true)) {
        m_VideoStatus = "Failed to generate NV12 source pattern via compute";
        return false;
    }
    m_PendingEncodedPatternConstants = patternConstants;
    m_HasPendingEncodedPatternConstants = true;

    video_sample::EncodeRequest encodeRequest = {};
    encodeRequest.nv12Buffer = m_UploadBuffer;
    encodeRequest.nv12BufferState = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
    encodeRequest.nv12Layout = &m_Nv12Layout;
    encodeRequest.nv12Width = m_VideoWidth;
    encodeRequest.nv12Height = m_VideoHeight;
    encodeRequest.quality = m_VideoQuality;
    encodeRequest.plan = plan;
    video_sample::EncodedFrame encodedFrame = {};
    if (!m_Encoder->Encode(encodeRequest, encodedFrame)) {
        m_VideoStatus = m_Encoder->GetStatus();
        return false;
    }

    video_sample::DecodedFrame decodedFrame = {};
    video_sample::DecodeRequest decodeRequest = {};
    decodeRequest.frame = &encodedFrame;
    decodeRequest.nv12ReadbackBuffer = m_UploadBuffer;
    decodeRequest.nv12Layout = &m_Nv12Layout;
    if (!m_Decoder->Decode(decodeRequest, decodedFrame)) {
        m_VideoStatus = m_Decoder->GetStatus();
        return false;
    }

    if (!plan.isDisplay)
        return false;

    patternConstants = m_HasPendingEncodedPatternConstants ? m_PendingEncodedPatternConstants : MakePatternConstants(OP_NV12_TO_PREVIEW, timeSec);
    patternConstants.operation = OP_NV12_TO_PREVIEW;
    patternConstants.diffStrength = m_DiffStrength;
    patternConstants.showDifference = m_ShowDifference ? 1 : 0;
    if (!GeneratePatternWithCompute(patternConstants, m_DecodePreviewStorage, true)) {
        m_VideoStatus = "Failed to convert decoded NV12 to preview texture";
        return false;
    }
    m_HasPendingEncodedPatternConstants = false;
    m_DecodePreviewReady = true;

    char message[128] = {};
    std::snprintf(message, sizeof(message), "%s encode/decode round trip complete, encoded %llu bytes", GetCodecName(m_Codec), (unsigned long long)encodedFrame.feedback.encodedBitstreamWrittenBytes);
    m_VideoStatus = message;
    if (!m_SmokeRoundTripComplete) {
        std::printf("VIDEO_ROUND_TRIP_OK: %s\n", message);
        m_SmokeRoundTripComplete = true;
    }
    return true;
}

int32_t Sample::RunHeadless(nri::GraphicsAPI graphicsAPI) {
    using namespace video_sample;

    RunEnvironment environment = {};
    environment.graphicsAPI = graphicsAPI;
    environment.adapterIndex = m_AdapterIndex;
    environment.debugAPI = m_DebugAPI;
    environment.debugNRI = true;
    environment.enhancedBarriers = m_EnhancedBarriers;

    if (m_StreamMode == "decodeFile") {
        const StreamResult result = RunDecodeFile(environment, m_DecodeFile, m_RefYuv, m_StreamOptions.name, m_StreamOptions.dumpDir, m_StreamOptions.verbose);
        std::printf("[%s] %s api=%s adapter=\"%s\" frames=%u bytes=%llu psnrY(min/avg)=%.2f/%.2f nriErrors=%u nriWarnings=%u %s%s%s\n",
            GetStreamStatusName(result.status), m_StreamOptions.name.c_str(), nri::nriGetGraphicsAPIString(graphicsAPI), result.adapterName.c_str(), result.frameNum, (unsigned long long)result.byteNum, result.psnrMin, result.psnrAvg, result.nriErrorNum, result.nriWarningNum,
            result.notes.c_str(), result.message.empty() ? "" : " | ", result.message.c_str());
        std::fflush(stdout);
        return result.status == StreamStatus::PASS ? 0 : (result.status == StreamStatus::FAIL ? 1 : 2);
    }

    StreamOptions base = m_StreamOptions;
    if (m_StreamMode != "stream") {
        // Matrix base: small, not 16-aligned (cropping), CQP
        base.config = VideoConfig();
        base.config.videoWidth = m_VideoSizeExplicit ? m_VideoWidth : 640;
        base.config.videoHeight = m_VideoSizeExplicit ? m_VideoHeight : 360;
        base.gop = "P";
        base.frameNum = 8;
        base.idrPeriod = 0;
        base.minPsnr = 30.0;
        base.rateControlChangeFrame = -1;
        base.verbose = m_StreamOptions.verbose;
        base.encodeOnly = m_StreamOptions.encodeOnly;
    }

    std::vector<StreamOptions> cases;
    if (m_StreamMode == "stream")
        cases.push_back(base);
    else {
        // An exact name selects one case, otherwise the filter is a substring
        const std::vector<StreamOptions> all = MakeMatrixCases(base);
        bool isExact = false;
        for (const StreamOptions& options : all)
            isExact = isExact || options.name == m_CaseFilter;
        for (const StreamOptions& options : all) {
            if (m_CaseFilter.empty() || (isExact ? options.name == m_CaseFilter : options.name.find(m_CaseFilter) != std::string::npos))
                cases.push_back(options);
        }
    }

    if (m_StreamMode == "list") {
        for (const StreamOptions& options : cases)
            std::printf("%s\n", options.name.c_str());
        return 0;
    }

    uint32_t passNum = 0;
    uint32_t failNum = 0;
    uint32_t skipNum = 0;
    for (const StreamOptions& options : cases) {
        const StreamResult result = RunStream(environment, options);

        if (result.status == StreamStatus::PASS)
            passNum++;
        else if (result.status == StreamStatus::FAIL)
            failNum++;
        else
            skipNum++;

        std::printf("[%s] %s api=%s adapter=\"%s\" frames=%u bytes=%llu psnrY(min/avg)=%.2f/%.2f nriErrors=%u nriWarnings=%u %s%s%s\n",
            GetStreamStatusName(result.status), options.name.c_str(), nri::nriGetGraphicsAPIString(graphicsAPI), result.adapterName.c_str(), result.frameNum, (unsigned long long)result.byteNum, result.psnrMin, result.psnrAvg, result.nriErrorNum, result.nriWarningNum,
            result.notes.c_str(), result.message.empty() ? "" : " | ", result.message.c_str());
        std::fflush(stdout);
    }

    std::printf("SUMMARY: pass=%u fail=%u skip=%u\n", passNum, failNum, skipNum);
    std::fflush(stdout);
    return failNum ? 1 : (passNum ? 0 : 2);
}

bool Sample::CanRunRoundTrip() const {
    return m_VideoReady && m_Encoder && m_Encoder->IsReady() && m_Decoder && m_Decoder->IsReady() && m_GraphicsQueue && m_VideoEncodeQueue && m_VideoDecodeQueue && m_UploadBuffer && m_UploadBufferView && m_SourcePreviewStorage && m_DecodePreviewStorage && m_GeneratePipelineLayout && m_GenerateComputePipeline && m_GenerateDescriptorSet;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    const uint32_t commandFrameNum = (uint32_t)m_QueuedFrames.size();
    const uint32_t queuedFrameNum = GetQueuedFrameNum();
    uint32_t queuedFrameIndex = frameIndex % commandFrameNum;
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];
    const uint64_t waitValue = frameIndex >= queuedFrameNum ? 1 + frameIndex - queuedFrameNum : 0;

    NRI.Wait(*m_FrameFence, waitValue);
    NRI.ResetCommandAllocator(*queuedFrame.commandAllocator);
}

void Sample::DrawTexturePanel(const char* label, nri::Descriptor* texture, const ImVec2& size) {
    ImGui::Text("%s", label);
    if (!texture) {
        ImGui::Text("Not ready");
        return;
    }
    ImGui::Image((ImTextureID)texture, size);
}

void Sample::PrepareFrame(uint32_t) {
    const double timeSec = m_Timer.GetTimeStamp() * 0.001 - m_StartTimeSec;
    const bool canRunRoundTrip = CanRunRoundTrip();

    InitializeGeneratedFrames((float)timeSec);

    if (m_RequireVideoRoundTrip && !m_SmokeRoundTripComplete && !canRunRoundTrip && !m_SmokeRoundTripFailed) {
        std::fprintf(stderr, "VIDEO_ROUND_TRIP_FAILED: %s\n", m_VideoStatus.c_str());
        m_SmokeRoundTripFailed = true;
    }

    if (canRunRoundTrip && timeSec - m_LastRoundTripTimeSec >= ROUND_TRIP_INTERVAL_SEC) {
        if (TryRunRoundTrip((float)timeSec))
            m_LastRoundTripTimeSec = timeSec;
    }

    ImGui::NewFrame();
    {
        ImGui::SetNextWindowPos({20.0f, 20.0f}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({900.0f, 520.0f}, ImGuiCond_Once);
        ImGui::Begin("NRI Video Encode / Decode");
        {
            const uint32_t minCodecQp = m_Codec == SampleCodec::AV1 ? 1 : 0;
            const uint32_t effectiveQpI = m_VideoQuality.lossless ? minCodecQp : std::max(m_VideoQuality.qpI, minCodecQp);
            const uint32_t effectiveQpP = m_VideoQuality.lossless ? minCodecQp : std::max(m_VideoQuality.qpP, minCodecQp);
            const uint32_t effectiveQpB = m_VideoQuality.lossless ? minCodecQp : std::max(m_VideoQuality.qpB, minCodecQp);
            ImGui::Text("Codec: %s, format: NV12, size: %ux%u", GetCodecName(m_Codec), m_VideoWidth, m_VideoHeight);
            ImGui::Text("Startup-only: codec, video size, AV1/H.26x frame mode");
            ImGui::Text("CQP: I=%u, P=%u, B=%u%s", effectiveQpI, effectiveQpP, effectiveQpB, m_Codec == SampleCodec::AV1 ? ", AV1 baseQIndex follows below" : "");
            if (m_Codec == SampleCodec::AV1) {
                const uint32_t effectiveAv1BaseQIndex = m_VideoQuality.lossless ? 1 : std::max(m_VideoQuality.av1BaseQIndex, 1u);
                ImGui::Text("AV1: frame=%s, baseQIndex=%u", m_AV1FrameArg.c_str(), effectiveAv1BaseQIndex);
            } else
                ImGui::Text("H.26x frame permutation: %s", m_H26FrameArg.c_str());
            ImGui::TextWrapped("Video: %s", m_VideoStatus.c_str());
            ImGui::TextWrapped("Preview: %s", m_PreviewStatus.c_str());
            ImGui::Text("Encode queue: %s, decode queue: %s", m_VideoEncodeQueue ? "yes" : "no", m_VideoDecodeQueue ? "yes" : "no");
            const bool roundTripSupported = CanRunRoundTrip();
            ImGui::Text("Round trip: %s", roundTripSupported ? "running" : "backend must support NRI video encode/decode");
            if (m_VideoReady && !m_DecodePreviewReady)
                ImGui::Text("Decode preview: waiting for first decoded frame");

            ImGui::Separator();
            ImGui::Text("Live controls");
            ImGui::Checkbox("Lossless / zero quantizers", &m_VideoQuality.lossless);
            if (m_VideoQuality.lossless) {
                if (m_Codec == SampleCodec::AV1)
                    ImGui::TextWrapped("AV1 hardware encode rejects baseQIndex 0 on current backends, so lossless mode uses near-lossless baseQIndex 1.");
                else
                    ImGui::TextWrapped("%s forces CQP 0%s.", GetCodecName(m_Codec), m_Codec == SampleCodec::H264 ? " with transform bypass advertised in the SPS" : "");
            }
            int qpI = (int)m_VideoQuality.qpI;
            int qpP = (int)m_VideoQuality.qpP;
            int qpB = (int)m_VideoQuality.qpB;
            const int qpMin = m_Codec == SampleCodec::AV1 ? 1 : 0;
            const int qpMax = m_Codec == SampleCodec::AV1 ? 255 : 51;
            if (ImGui::SliderInt("QP I / IDR", &qpI, qpMin, qpMax))
                m_VideoQuality.qpI = (uint32_t)qpI;
            if (ImGui::SliderInt("QP P", &qpP, qpMin, qpMax))
                m_VideoQuality.qpP = (uint32_t)qpP;
            if (ImGui::SliderInt("QP B", &qpB, qpMin, qpMax))
                m_VideoQuality.qpB = (uint32_t)qpB;
            if (m_Codec == SampleCodec::AV1) {
                int av1BaseQIndex = (int)m_VideoQuality.av1BaseQIndex;
                if (ImGui::SliderInt("AV1 base Q index", &av1BaseQIndex, 1, 255))
                    m_VideoQuality.av1BaseQIndex = (uint32_t)av1BaseQIndex;
            }
            ImGui::SliderFloat("Pattern motion", &m_PatternMotionSpeed, 0.0f, 4.0f, "%.2f");
            ImGui::SliderFloat("Pattern detail", &m_PatternDetailStrength, 0.0f, 2.5f, "%.2f");
            ImGui::Checkbox("Show artifact diff", &m_ShowDifference);
            if (m_ShowDifference)
                ImGui::SliderFloat("Diff amplification", &m_DiffStrength, 1.0f, 20.0f, "%.1f");

            ImGui::Separator();
            if (ImGui::BeginTable("PreviewPanels", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();
                float width = std::max(200.0f, ImGui::GetContentRegionAvail().x);
                DrawTexturePanel("Generated source", m_SourcePreviewTextureView, {width, width * float(m_VideoHeight) / float(m_VideoWidth)});

                ImGui::TableNextColumn();
                width = std::max(200.0f, ImGui::GetContentRegionAvail().x);
                DrawTexturePanel(m_DecodePreviewReady ? (m_ShowDifference ? "Artifact diff" : "Decoded preview") : "Decoded preview pending", m_DecodePreviewReady ? m_DecodePreviewTextureView : nullptr, {width, width * float(m_VideoHeight) / float(m_VideoWidth)});
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
}

bool Sample::AppShouldClose() {
    return m_RequireVideoRoundTrip && (m_SmokeRoundTripComplete || m_SmokeRoundTripFailed);
}

void Sample::RenderFrame(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % (uint32_t)m_QueuedFrames.size();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    uint32_t recycledSemaphoreIndex = frameIndex % (uint32_t)m_SwapChainTextures.size();
    nri::Fence* swapChainAcquireSemaphore = m_SwapChainTextures[recycledSemaphoreIndex].acquireSemaphore;

    uint32_t currentSwapChainTextureIndex = 0;
    NRI.AcquireNextTexture(*m_SwapChain, *swapChainAcquireSemaphore, currentSwapChainTextureIndex);

    const SwapChainTexture& swapChainTexture = m_SwapChainTextures[currentSwapChainTextureIndex];

    nri::CommandBuffer& commandBuffer = *queuedFrame.commandBuffer;
    NRI.BeginCommandBuffer(commandBuffer, nullptr);
    {
        nri::TextureBarrierDesc textureBarriers = {};
        textureBarriers.texture = swapChainTexture.texture;
        textureBarriers.after = {nri::AccessBits::COLOR_ATTACHMENT, nri::Layout::COLOR_ATTACHMENT};
        textureBarriers.layerNum = 1;
        textureBarriers.mipNum = 1;

        nri::BarrierDesc barrierDesc = {};
        barrierDesc.textureNum = 1;
        barrierDesc.textures = &textureBarriers;
        NRI.CmdBarrier(commandBuffer, barrierDesc);

        nri::AttachmentDesc colorAttachmentDesc = {};
        colorAttachmentDesc.descriptor = swapChainTexture.colorAttachment;
        colorAttachmentDesc.clearValue.color.f = {0.03f, 0.03f, 0.03f, 1.0f};
        colorAttachmentDesc.loadOp = nri::LoadOp::CLEAR;
        colorAttachmentDesc.storeOp = nri::StoreOp::STORE;

        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colorNum = 1;
        renderingDesc.colors = &colorAttachmentDesc;

        const nri::ImguiRenderData imguiRenderData = CmdCopyImguiData(commandBuffer, *m_Streamer);

        NRI.CmdBeginRendering(commandBuffer, renderingDesc);
        CmdDrawImgui(commandBuffer, imguiRenderData, swapChainTexture.attachmentFormat, 1.0f, true);
        NRI.CmdEndRendering(commandBuffer);

        textureBarriers.before = textureBarriers.after;
        textureBarriers.after = {nri::AccessBits::NONE, nri::Layout::PRESENT, nri::StageBits::NONE};
        NRI.CmdBarrier(commandBuffer, barrierDesc);
    }
    NRI.EndCommandBuffer(commandBuffer);

    nri::FenceSubmitDesc textureAcquiredFence = {};
    textureAcquiredFence.fence = swapChainAcquireSemaphore;
    textureAcquiredFence.stages = nri::StageBits::COLOR_ATTACHMENT;

    nri::FenceSubmitDesc renderingFinishedFence = {};
    renderingFinishedFence.fence = swapChainTexture.releaseSemaphore;

    nri::QueueSubmitDesc queueSubmitDesc = {};
    queueSubmitDesc.waitFences = &textureAcquiredFence;
    queueSubmitDesc.waitFenceNum = 1;
    queueSubmitDesc.commandBuffers = &queuedFrame.commandBuffer;
    queueSubmitDesc.commandBufferNum = 1;
    queueSubmitDesc.signalFences = &renderingFinishedFence;
    queueSubmitDesc.signalFenceNum = 1;
    NRI.QueueSubmit(*m_GraphicsQueue, queueSubmitDesc);

    NRI.EndStreamerFrame(*m_Streamer);
    NRI.QueuePresent(*m_SwapChain, *swapChainTexture.releaseSemaphore, 0);

    nri::FenceSubmitDesc signalFence = {};
    signalFence.fence = m_FrameFence;
    signalFence.value = 1 + frameIndex;

    nri::QueueSubmitDesc signalSubmitDesc = {};
    signalSubmitDesc.signalFences = &signalFence;
    signalSubmitDesc.signalFenceNum = 1;
    NRI.QueueSubmit(*m_GraphicsQueue, signalSubmitDesc);
}

SAMPLE_MAIN(Sample, 0);
