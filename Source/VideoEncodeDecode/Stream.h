// © 2021 NVIDIA Corporation

#pragma once

#include "Shared.h"

namespace video_sample {

// Headless encode -> decode stream runs ("--stream" / "--matrix")
struct StreamOptions {
    std::string name = "stream";
    VideoConfig config = {};
    VideoQuality quality = {};
    uint32_t frameNum = 8;
    std::string gop = "P";       // display-order kinds of the frames after the first IDR, repeated: I (IDR), P, p (non-reference P), B (non-reference bi-predicted)
    uint32_t idrPeriod = 0;      // 0 - only the first frame is IDR
    int32_t rateControlChangeFrame = -1; // coding index where "rateControlB" replaces "quality.rateControl"
    RateControlSetting rateControlB = {};
    bool verbose = false;        // log every picture before it is submitted
    bool encodeOnly = false;     // skip the decoder (the stream is still dumped)
    double minPsnr = 30.0;
    std::string dumpDir;         // writes <dumpDir>/<name>/{stream,decoded.yuv,source.yuv,frames.csv}
};

struct RunEnvironment {
    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
    uint32_t adapterIndex = 0;
    bool debugAPI = false;
    bool debugNRI = false;
    bool enhancedBarriers = false; // D3D12
};

enum class StreamStatus : uint8_t {
    PASS,
    FAIL,
    UNSUPPORTED,
};

struct StreamResult {
    StreamStatus status = StreamStatus::FAIL;
    std::string message;
    std::string adapterName;
    uint32_t frameNum = 0;
    uint64_t byteNum = 0;
    double psnrMin = 0.0;
    double psnrAvg = 0.0;
    uint32_t nriErrorNum = 0;
    uint32_t nriWarningNum = 0;
    std::string notes;
};

// Display-order test pattern, "patternWidth/Height" define the content, "width/height" the generated area
void GenerateNv12Pattern(uint8_t* dst, const Nv12BufferLayout& layout, uint32_t width, uint32_t height, uint32_t patternWidth, uint32_t patternHeight, uint32_t frame);

// Frame plans in decode order
uint32_t GetReorderNum(const std::vector<FramePlan>& plans);
std::vector<FramePlan> BuildFramePlans(const VideoConfig& config, const std::string& gop, uint32_t frameNum, uint32_t idrPeriod);

// Query capabilities, derive the coded size and bitstream limits
bool QueryVideoSize(const VideoContext& context, const VideoConfig& config, VideoSize& size, bool& isUnsupported, std::string& status);

StreamResult RunStream(const RunEnvironment& environment, const StreamOptions& options);
// Decode an external AV1 OBU stream (one OBU_FRAME per picture), optionally compared bit-exactly with an NV12 reference of the shown pictures
StreamResult RunDecodeFile(const RunEnvironment& environment, const std::string& path, const std::string& refYuvPath, const std::string& name, const std::string& dumpDir, bool verbose);
std::vector<StreamOptions> MakeMatrixCases(const StreamOptions& base);
const char* GetStreamStatusName(StreamStatus status);

} // namespace video_sample
