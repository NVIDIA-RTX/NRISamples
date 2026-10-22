// © 2021 NVIDIA Corporation

#include "Stream.h"

#include "Decoder.h"
#include "Encoder.h"

#if NRISAMPLES_ENABLE_AV1_FILE_DECODING
// External file decoding is developer-only: the public AV1 helper requires an encode session.
#    include "SharedVideo.h"
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <memory>

namespace video_sample {

//==================================================================================================
// Test pattern
//==================================================================================================

void GenerateNv12Pattern(uint8_t* dst, const Nv12BufferLayout& layout, uint32_t width, uint32_t height, uint32_t patternWidth, uint32_t patternHeight, uint32_t frame) {
    const int32_t pw = (int32_t)patternWidth;
    const int32_t ph = (int32_t)patternHeight;
    const int32_t radius = std::max(ph / 8, 8);
    const int32_t boxX = (int32_t)((frame * 13u) % (uint32_t)pw);
    const int32_t boxY = ph / 3 + (int32_t)((frame * 5u) % (uint32_t)std::max(ph / 3, 1));

    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = dst + uint64_t(y) * layout.yRowPitchBytes;
        for (uint32_t x = 0; x < width; x++) {
            int32_t v = 32 + (int32_t)(((x + frame * 3u) % (uint32_t)pw) * 160u / (uint32_t)pw) + (int32_t)(y * 48u / (uint32_t)ph);
            v += (((x >> 3) ^ (y >> 3)) & 1) ? 12 : -12;
            const int32_t dx = (int32_t)x - boxX;
            const int32_t dy = (int32_t)y - boxY;
            if (dx * dx + dy * dy < radius * radius)
                v = 220 - ((dx * dx + dy * dy) * 60) / (radius * radius);
            const uint32_t hash = (x * 73856093u) ^ (y * 19349663u) ^ (frame * 83492791u);
            v += (int32_t)((hash >> 13) & 7) - 3;
            row[x] = (uint8_t)std::clamp(v, 16, 235);
        }
    }

    for (uint32_t y = 0; y < height / 2; y++) {
        uint8_t* row = dst + layout.uvOffsetBytes + uint64_t(y) * layout.uvRowPitchBytes;
        for (uint32_t x = 0; x < width / 2; x++) {
            const int32_t fx = (int32_t)x * 2;
            const int32_t fy = (int32_t)y * 2;
            int32_t u = 128 + (int32_t)(((uint32_t)fx * 50u) / (uint32_t)pw) - 25;
            int32_t v = 128 + (int32_t)(((uint32_t)fy * 50u) / (uint32_t)ph) - 25;
            const int32_t dx = fx - boxX;
            const int32_t dy = fy - boxY;
            if (dx * dx + dy * dy < radius * radius) {
                u = 90;
                v = 200;
            }
            row[2 * x] = (uint8_t)std::clamp(u, 16, 240);
            row[2 * x + 1] = (uint8_t)std::clamp(v, 16, 240);
        }
    }
}

//==================================================================================================
// Frame plans
//==================================================================================================

struct DpbEntry {
    uint32_t slot = 0;
    int32_t poc = 0;
    uint32_t frameNum = 0;
    uint32_t frameId = 0;
    uint8_t orderHint = 0;
    nri::VideoFrameType type = nri::VideoFrameType::IDR;
};

// SPS max_num_reorder_pics: the most pictures that precede a picture in decode order and follow it in output order
uint32_t GetReorderNum(const std::vector<FramePlan>& plans) {
    uint32_t reorderNum = 0;
    for (size_t i = 0; i < plans.size(); i++) {
        uint32_t num = 0;
        for (size_t j = 0; j < i; j++)
            num += plans[j].displayIndex > plans[i].displayIndex;
        reorderNum = std::max(reorderNum, num);
    }
    return reorderNum;
}

std::vector<FramePlan> BuildFramePlans(const VideoConfig& config, const std::string& gop, uint32_t frameNum, uint32_t idrPeriod) {
    const uint32_t referenceNum = std::clamp(config.referenceNum, 1u, MAX_REFERENCE_NUM);
    const bool isAv1 = config.codec == SampleCodec::AV1;
    const uint32_t maxFrameNum = 256; // H.264 SPS: log2_max_frame_num = 8

    std::vector<FramePlan> plans;
    std::vector<DpbEntry> dpb; // oldest first
    std::vector<uint32_t> pendingB;
    uint32_t lastReferenceFrameNum = 0;
    uint32_t idrDisplayIndex = 0;
    uint32_t idrCount = 0;
    uint32_t codingIndex = 0;

    const char nonReferenceP = isAv1 ? 'p' : 'P';

    auto getKind = [&](uint32_t displayIndex) {
        if (displayIndex == 0 || (idrPeriod && (displayIndex % idrPeriod) == 0))
            return 'I';
        if (gop.empty())
            return 'P';
        char kind = gop[(displayIndex - 1) % gop.size()];
        if (kind == 'B' && isAv1)
            kind = 'p';
        if (kind == 'p' && !isAv1)
            kind = 'P'; // H.264/H.265: only B pictures are non-reference (the backend marks them by the frame type)
        return kind;
    };

    auto emit = [&](uint32_t displayIndex, char kind) {
        FramePlan plan = {};
        plan.codingIndex = codingIndex++;
        plan.displayIndex = displayIndex;
        plan.type = kind == 'I' ? nri::VideoFrameType::IDR : (kind == 'B' ? nri::VideoFrameType::B : nri::VideoFrameType::P);
        plan.isReference = kind == 'I' || kind == 'P';

        if (kind == 'I') {
            dpb.clear();
            idrDisplayIndex = displayIndex;
            plan.idrPictureId = idrCount++ & 0xFFFF;
        }

        plan.poc = (int32_t)(displayIndex - idrDisplayIndex);
        plan.frameNum = kind == 'I' ? 0 : (lastReferenceFrameNum + 1) % maxFrameNum;
        plan.av1OrderHint = (uint8_t)(displayIndex & 0xFF);
        plan.av1RefreshFrameFlags = kind == 'I' ? 0xFF : (plan.isReference ? 0x1 : 0x0);
        if (isAv1)
            plan.frameNum = plan.codingIndex; // frame id

        auto toReference = [&](const DpbEntry& entry, uint8_t listIndex) {
            PlanReference reference = {};
            reference.slot = entry.slot;
            reference.poc = entry.poc;
            reference.frameNum = entry.frameNum;
            reference.frameId = entry.frameId;
            reference.orderHint = entry.orderHint;
            reference.type = entry.type;
            reference.listIndex = listIndex;
            return reference;
        };

        if (kind == 'P' || kind == 'p') {
            // The most recent reference first (initial List0 order)
            const uint32_t count = isAv1 ? 1u : (uint32_t)std::min<size_t>(dpb.size(), referenceNum);
            for (uint32_t i = 0; i < count && i < dpb.size(); i++)
                plan.references[plan.referenceNum++] = toReference(dpb[dpb.size() - 1 - i], 0);
        } else if (kind == 'B') {
            std::vector<DpbEntry> past;
            std::vector<DpbEntry> future;
            for (const DpbEntry& entry : dpb)
                (entry.poc < plan.poc ? past : future).push_back(entry);
            std::sort(past.begin(), past.end(), [](const DpbEntry& a, const DpbEntry& b) { return a.poc > b.poc; });
            std::sort(future.begin(), future.end(), [](const DpbEntry& a, const DpbEntry& b) { return a.poc < b.poc; });

            // The H.265 B picture List0 limit is not exposed by NRI (NVIDIA: 1), H.264 takes the DPB size
            const uint32_t pastMax = config.codec == SampleCodec::H265 ? 1u : std::max(referenceNum, 2u) - 1;
            const uint32_t pastNum = (uint32_t)std::min<size_t>(past.size(), pastMax);
            for (uint32_t i = 0; i < pastNum; i++)
                plan.references[plan.referenceNum++] = toReference(past[i], 0);
            if (!future.empty())
                plan.references[plan.referenceNum++] = toReference(future[0], 1);
        }

        // A free slot, never one of the DPB pictures
        for (uint32_t slot = 0; slot <= referenceNum; slot++) {
            bool isUsed = false;
            for (const DpbEntry& entry : dpb)
                isUsed = isUsed || entry.slot == slot;
            if (!isUsed) {
                plan.setupSlot = slot;
                break;
            }
        }

        // The H.265 RPS of a picture only lists its own references (NRI cannot keep other DPB pictures), a decoder drops the rest
        if (kind == 'B' && config.codec == SampleCodec::H265) {
            dpb.erase(std::remove_if(dpb.begin(), dpb.end(), [&](const DpbEntry& entry) {
                for (uint32_t i = 0; i < plan.referenceNum; i++) {
                    if (plan.references[i].slot == entry.slot)
                        return false;
                }
                return true;
            }),
                dpb.end());
        }

        plan.reverseReferenceOrder = config.reverseReferences && config.codec == SampleCodec::H265 && plan.referenceNum > 1 && (plan.codingIndex & 1);
        plan.reverseDecodeReferenceOrder = config.reverseDecodeReferences && config.codec == SampleCodec::H265 && plan.referenceNum > 1 && (plan.codingIndex & 1);

        if (plan.isReference) {
            DpbEntry entry = {};
            entry.slot = plan.setupSlot;
            entry.poc = plan.poc;
            entry.frameNum = plan.frameNum;
            entry.frameId = plan.codingIndex;
            entry.orderHint = plan.av1OrderHint;
            entry.type = plan.type;
            dpb.push_back(entry);
            if (dpb.size() > referenceNum)
                dpb.erase(dpb.begin());
            lastReferenceFrameNum = plan.frameNum;
        }

        plans.push_back(plan);
    };

    for (uint32_t displayIndex = 0; displayIndex < frameNum; displayIndex++) {
        const char kind = getKind(displayIndex);

        if (kind == 'B') {
            pendingB.push_back(displayIndex);
            continue;
        }

        if (kind == 'I') {
            // B pictures can't reference across an IDR
            for (uint32_t pending : pendingB)
                emit(pending, nonReferenceP);
            pendingB.clear();
            emit(displayIndex, kind);
            continue;
        }

        emit(displayIndex, kind);
        for (uint32_t pending : pendingB)
            emit(pending, 'B');
        pendingB.clear();
    }

    // Trailing B pictures have no future reference
    for (uint32_t pending : pendingB)
        emit(pending, nonReferenceP);

    return plans;
}

//==================================================================================================
// Video size
//==================================================================================================

bool QueryVideoSize(const VideoContext& context, const VideoConfig& config, VideoSize& size, bool& isUnsupported, std::string& status) {
    nri::VideoInterface& video = *context.video;
    isUnsupported = true;

    nri::VideoSessionDesc encodeSessionDesc = {};
    encodeSessionDesc.type = nri::VideoSessionType::ENCODE;
    encodeSessionDesc.codec = GetNriCodec(config.codec);
    encodeSessionDesc.format = nri::Format::NV12_UNORM;
    // NVIDIA D3D12 H.265 encode corrupts the bottom partial CTB row (driver 617.42, 360, 368, 376 and 400 lines at 32x32 CTBs): code multiples of 32 lines and crop
    const uint32_t widthAlignment = config.codedAlignment;
    const uint32_t heightAlignment = (context.graphicsAPI == nri::GraphicsAPI::D3D12 && config.codec == SampleCodec::H265) ? std::max(config.codedAlignment, 32u) : config.codedAlignment;
    encodeSessionDesc.width = (uint32_t)AlignUp(config.videoWidth, widthAlignment); // the backends that echo the extents need the aligned size
    encodeSessionDesc.height = (uint32_t)AlignUp(config.videoHeight, heightAlignment);
    encodeSessionDesc.maxReferenceNum = config.referenceNum;

    nri::VideoSessionDesc decodeSessionDesc = encodeSessionDesc;
    decodeSessionDesc.type = nri::VideoSessionType::DECODE;

    nri::VideoCapabilities encodeCapabilities = {};
    nri::VideoCapabilities decodeCapabilities = {};
    video.GetVideoCapabilities(*context.device, encodeSessionDesc, encodeCapabilities);
    video.GetVideoCapabilities(*context.device, decodeSessionDesc, decodeCapabilities);
    if (!encodeCapabilities.pictureAccessGranularityWidth || !encodeCapabilities.pictureAccessGranularityHeight || !decodeCapabilities.pictureAccessGranularityWidth || !decodeCapabilities.pictureAccessGranularityHeight) {
        status = std::string(GetCodecName(config.codec)) + " encode or decode is unsupported";
        return false;
    }

    const uint32_t codedWidthAlignment = std::max({encodeCapabilities.pictureAccessGranularityWidth, decodeCapabilities.pictureAccessGranularityWidth, widthAlignment});
    const uint32_t codedHeightAlignment = std::max({encodeCapabilities.pictureAccessGranularityHeight, decodeCapabilities.pictureAccessGranularityHeight, heightAlignment});
    const uint32_t codedWidth = (uint32_t)AlignUp(std::max({config.videoWidth, encodeCapabilities.widthMin, decodeCapabilities.widthMin}), codedWidthAlignment);
    const uint32_t codedHeight = (uint32_t)AlignUp(std::max({config.videoHeight, encodeCapabilities.heightMin, decodeCapabilities.heightMin}), codedHeightAlignment);
    if ((encodeCapabilities.widthMax && codedWidth > encodeCapabilities.widthMax) || (decodeCapabilities.widthMax && codedWidth > decodeCapabilities.widthMax) || (encodeCapabilities.heightMax && codedHeight > encodeCapabilities.heightMax) || (decodeCapabilities.heightMax && codedHeight > decodeCapabilities.heightMax)) {
        status = std::string(GetCodecName(config.codec)) + " coded video size " + std::to_string(codedWidth) + "x" + std::to_string(codedHeight) + " exceeds backend capabilities";
        return false;
    }

    encodeSessionDesc.width = codedWidth;
    encodeSessionDesc.height = codedHeight;
    decodeSessionDesc.width = codedWidth;
    decodeSessionDesc.height = codedHeight;
    if (video.GetVideoCapabilities(*context.device, encodeSessionDesc, encodeCapabilities) != nri::Result::SUCCESS || video.GetVideoCapabilities(*context.device, decodeSessionDesc, decodeCapabilities) != nri::Result::SUCCESS) {
        status = std::string(GetCodecName(config.codec)) + " coded video size " + std::to_string(codedWidth) + "x" + std::to_string(codedHeight) + " is unsupported";
        return false;
    }

    size = {};
    size.videoWidth = config.videoWidth;
    size.videoHeight = config.videoHeight;
    size.codedWidth = codedWidth;
    size.codedHeight = codedHeight;
    size.decodeBitstreamSizeAlignment = std::max(decodeCapabilities.bitstreamSizeAlignment, 1u);
    size.bitstreamOffsetAlignment = std::max(encodeCapabilities.bitstreamOffsetAlignment, 1u);
    size.bitstreamSizeAlignment = std::max(encodeCapabilities.bitstreamSizeAlignment, 1u);
    size.bitstreamSize = AlignUp(std::max<uint64_t>(2 * 1024 * 1024, uint64_t(codedWidth) * codedHeight), 1024 * 1024);

    if ((ENCODED_SLICE_OFFSET % size.bitstreamOffsetAlignment) != 0 || ((size.bitstreamSize - ENCODED_SLICE_OFFSET) % size.bitstreamSizeAlignment) != 0) {
        status = "Encode bitstream alignment is not satisfiable";
        return false;
    }

    isUnsupported = false;
    return true;
}

//==================================================================================================
// Stream run
//==================================================================================================

namespace {

struct MessageStats {
    uint32_t errorNum = 0;
    uint32_t warningNum = 0;
    std::vector<std::string> errors;
};

void NRI_CALL MessageCallback(nri::Message messageType, const char* file, uint32_t line, const char* message, void* userArg) {
    MessageStats& stats = *(MessageStats*)userArg;
    const char* name = messageType == nri::Message::ERROR ? "ERROR" : (messageType == nri::Message::WARNING ? "WARNING" : "INFO");

    if (messageType == nri::Message::ERROR) {
        stats.errorNum++;
        if (stats.errors.size() < 4)
            stats.errors.push_back(message);
    } else if (messageType == nri::Message::WARNING)
        stats.warningNum++;

    if (messageType != nri::Message::INFO)
        std::fprintf(stderr, "[NRI %s] %s (%s:%u)\n", name, message, file, line);
}

void NRI_CALL AbortCallback(void*) {
    // Errors are counted, the run continues
}

struct DeviceHolder {
    ~DeviceHolder() {
        if (device)
            nri::nriDestroyDevice(device);
    }

    nri::Device* device = nullptr;
};

double GetPsnr(double squaredErrorSum, uint64_t sampleNum) {
    if (!sampleNum || squaredErrorSum <= 0.0)
        return 99.0;
    return std::min(99.0, 10.0 * std::log10(255.0 * 255.0 * double(sampleNum) / squaredErrorSum));
}

void ComputeSquaredError(const uint8_t* a, const uint8_t* b, uint64_t size, double& sum) {
    uint64_t total = 0;
    for (uint64_t i = 0; i < size; i++) {
        const int32_t d = (int32_t)a[i] - (int32_t)b[i];
        total += uint64_t(d * d);
    }
    sum += double(total);
}

const char* GetStreamExtension(SampleCodec codec) {
    return codec == SampleCodec::H265 ? "h265" : (codec == SampleCodec::AV1 ? "obu" : "h264");
}

bool WriteFile(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    FILE* file = std::fopen(path.string().c_str(), "wb");
    if (!file)
        return false;
    const bool ok = data.empty() || std::fwrite(data.data(), 1, data.size(), file) == data.size();
    std::fclose(file);
    return ok;
}

const char* GetFrameTypeName(nri::VideoFrameType type) {
    switch (type) {
        case nri::VideoFrameType::IDR:
            return "IDR";
        case nri::VideoFrameType::I:
            return "I";
        case nri::VideoFrameType::P:
            return "P";
        case nri::VideoFrameType::B:
        default:
            return "B";
    }
}

} // namespace

const char* GetStreamStatusName(StreamStatus status) {
    switch (status) {
        case StreamStatus::PASS:
            return "PASS";
        case StreamStatus::UNSUPPORTED:
            return "SKIP";
        case StreamStatus::FAIL:
        default:
            return "FAIL";
    }
}

StreamResult RunStream(const RunEnvironment& environment, const StreamOptions& options) {
    StreamResult result = {};
    MessageStats stats;
    VideoConfig config = options.config;
    const bool hasB = options.gop.find('B') != std::string::npos && config.codec != SampleCodec::AV1;
    config.hasBFrames = hasB;
    if (hasB)
        config.referenceNum = std::max(config.referenceNum, 2u);
    if (config.codec == SampleCodec::AV1)
        config.referenceNum = 1; // the AV1 picture layout uses the latest reference only
    config.referenceNum = std::clamp(config.referenceNum, 1u, MAX_REFERENCE_NUM);

    auto finish = [&](StreamStatus status, const std::string& message) {
        result.status = status;
        result.message = message;
        result.nriErrorNum = stats.errorNum;
        result.nriWarningNum = stats.warningNum;
        if (status == StreamStatus::PASS && stats.errorNum) {
            result.status = StreamStatus::FAIL;
            result.message = "NRI reported " + std::to_string(stats.errorNum) + " error(s): " + (stats.errors.empty() ? "" : stats.errors[0]);
        }
        return result;
    };

    // Device
    nri::AdapterDesc adapters[8] = {};
    uint32_t adapterNum = helper::GetCountOf(adapters);
    if (nri::nriEnumerateAdapters(adapters, adapterNum) != nri::Result::SUCCESS || !adapterNum)
        return finish(StreamStatus::UNSUPPORTED, "No graphics adapters");

    // Adapters of the requested API (the list also contains the NONE adapter)
    uint32_t apiAdapterNum = 0;
    for (uint32_t i = 0; i < adapterNum; i++) {
        if (adapters[i].supportedGraphicsAPIs & environment.graphicsAPI)
            adapters[apiAdapterNum++] = adapters[i];
    }
    if (!apiAdapterNum)
        return finish(StreamStatus::UNSUPPORTED, "No adapters support the requested graphics API");
    const nri::AdapterDesc& adapter = adapters[std::min(environment.adapterIndex, apiAdapterNum - 1)];
    result.adapterName = adapter.name;
    if (environment.graphicsAPI == nri::GraphicsAPI::D3D11 || !adapter.queueNum[(uint32_t)nri::QueueType::VIDEO_ENCODE] || !adapter.queueNum[(uint32_t)nri::QueueType::VIDEO_DECODE])
        return finish(StreamStatus::UNSUPPORTED, "The adapter has no NRI video encode and decode queues");

    nri::QueueFamilyDesc queueFamilies[3] = {};
    queueFamilies[0].queueNum = 1;
    queueFamilies[0].queueType = nri::QueueType::GRAPHICS;
    queueFamilies[1].queueNum = 1;
    queueFamilies[1].queueType = nri::QueueType::VIDEO_ENCODE;
    queueFamilies[2].queueNum = 1;
    queueFamilies[2].queueType = nri::QueueType::VIDEO_DECODE;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = environment.graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = environment.debugAPI;
    deviceCreationDesc.enableNRIValidation = environment.debugNRI;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.disableD3D12EnhancedBarriers = environment.graphicsAPI == nri::GraphicsAPI::D3D12 && !environment.enhancedBarriers; // legacy barriers by default
    deviceCreationDesc.adapterDesc = &adapter;
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = helper::GetCountOf(queueFamilies);
    deviceCreationDesc.callbackInterface.MessageCallback = MessageCallback;
    deviceCreationDesc.callbackInterface.AbortExecution = AbortCallback;
    deviceCreationDesc.callbackInterface.userArg = &stats;

    DeviceHolder deviceHolder;
    if (nri::nriCreateDevice(deviceCreationDesc, deviceHolder.device) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to create the device");
    nri::Device& device = *deviceHolder.device;

    NRIInterface NRI = {};
    nri::VideoInterface Video = {};
    if (nri::nriGetInterface(device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) != nri::Result::SUCCESS || nri::nriGetInterface(device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to get NRI interfaces");
    if (nri::nriGetInterface(device, NRI_INTERFACE(nri::VideoInterface), &Video) != nri::Result::SUCCESS || !Video.CreateVideoSession)
        return finish(StreamStatus::UNSUPPORTED, "NRI video interface is unavailable");

    VideoContext context = {};
    context.nri = &NRI;
    context.video = &Video;
    context.device = &device;
    context.graphicsAPI = environment.graphicsAPI;
    if (NRI.GetQueue(device, nri::QueueType::GRAPHICS, 0, context.graphicsQueue) != nri::Result::SUCCESS || NRI.GetQueue(device, nri::QueueType::VIDEO_ENCODE, 0, context.encodeQueue) != nri::Result::SUCCESS || NRI.GetQueue(device, nri::QueueType::VIDEO_DECODE, 0, context.decodeQueue) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to get queues");

    VideoSize size = {};
    bool isUnsupported = false;
    std::string status;
    if (!QueryVideoSize(context, config, size, isUnsupported, status))
        return finish(isUnsupported ? StreamStatus::UNSUPPORTED : StreamStatus::FAIL, status);

    std::vector<FramePlan> plans = BuildFramePlans(config, options.gop, options.frameNum, options.idrPeriod);
    config.reorderNum = GetReorderNum(plans);
    CodecParameters codecParameters = MakeCodecParameters(environment.graphicsAPI, config, size.codedWidth, size.codedHeight);

    if (plans.empty())
        return finish(StreamStatus::FAIL, "Empty frame plan");

    // Objects (the encoder and decoder must be destroyed before the device)
    auto encoder = std::make_unique<Encoder>();
    auto decoder = std::make_unique<Decoder>();
    nri::Buffer* sourceBuffer = nullptr;
    nri::Buffer* readbackBuffer = nullptr;
    auto cleanup = [&]() {
        NRI.DeviceWaitIdle(&device);
        decoder.reset();
        encoder.reset();
        if (sourceBuffer)
            NRI.DestroyBuffer(sourceBuffer);
        if (readbackBuffer)
            NRI.DestroyBuffer(readbackBuffer);
    };

    if (!encoder->Initialize(context, config, size, codecParameters)) {
        const bool isEncoderUnsupported = encoder->IsUnsupported();
        const std::string message = encoder->GetStatus();
        cleanup();
        return finish(isEncoderUnsupported ? StreamStatus::UNSUPPORTED : StreamStatus::FAIL, message);
    }
    if (!options.encodeOnly && !decoder->Initialize(context, config, size, codecParameters)) {
        const bool isDecoderUnsupported = decoder->IsUnsupported();
        const std::string message = decoder->GetStatus();
        cleanup();
        return finish(isDecoderUnsupported ? StreamStatus::UNSUPPORTED : StreamStatus::FAIL, message);
    }

    if (options.verbose) {
        const nri::VideoCapabilities& e = encoder->GetCapabilities();
        std::fprintf(stderr, "[%s] encode caps: maxRef=%u extent=%ux%u-%ux%u granularity=%ux%u bitstreamAlign=%u/%u dpbArray=%d(min %u) feedback=%d resolve=%d resolveQueue=%u metadata=%llu resolved=%llu rangeSize=%d\n", options.name.c_str(), e.maxReferenceNum, e.widthMin, e.heightMin, e.widthMax, e.heightMax, e.pictureAccessGranularityWidth, e.pictureAccessGranularityHeight, e.bitstreamOffsetAlignment, e.bitstreamSizeAlignment, e.dpbTextureArrayRequired, e.dpbTextureArrayMinLayerNum, e.encodeFeedbackSupported, e.encodeFeedbackResolveRequired, (uint32_t)e.resolvedMetadataQueueType, (unsigned long long)e.metadataSize, (unsigned long long)e.resolvedMetadataSize, e.encodeBitstreamRangeSizeSupported);
        if (!options.encodeOnly) {
            const nri::VideoCapabilities& d = decoder->GetCapabilities();
            std::fprintf(stderr, "[%s] decode caps: maxRef=%u extent=%ux%u-%ux%u bitstreamAlign=%u/%u coincide=%d distinct=%d dpbArray=%d(min %u) sources=%u\n", options.name.c_str(), d.maxReferenceNum, d.widthMin, d.heightMin, d.widthMax, d.heightMax, d.bitstreamOffsetAlignment, d.bitstreamSizeAlignment, d.decodeDpbAndOutputCoincide, d.decodeDpbAndOutputDistinct, d.dpbTextureArrayRequired, d.dpbTextureArrayMinLayerNum, (uint32_t)d.decodeBitstreamSourceMask);
        }
    }

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(device);
    const Nv12BufferLayout sourceLayout = MakeNv12BufferLayout(deviceDesc, size.codedWidth, size.codedHeight);
    const Nv12BufferLayout readbackLayout = MakeNv12BufferLayout(deviceDesc, size.videoWidth, size.videoHeight);

    nri::BufferDesc sourceBufferDesc = {};
    sourceBufferDesc.size = sourceLayout.totalSizeBytes;
    nri::BufferDesc readbackBufferDesc = {};
    readbackBufferDesc.size = readbackLayout.totalSizeBytes;
    if (NRI.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_UPLOAD, 0.0f, sourceBufferDesc, sourceBuffer) != nri::Result::SUCCESS || NRI.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_READBACK, 0.0f, readbackBufferDesc, readbackBuffer) != nri::Result::SUCCESS) {
        cleanup();
        return finish(StreamStatus::FAIL, "Failed to create the NV12 buffers");
    }

    // Round trip
    const uint32_t width = size.videoWidth;
    const uint32_t height = size.videoHeight;
    const uint64_t lumaSize = uint64_t(width) * height;
    const uint64_t chromaSize = lumaSize / 2;
    Nv12BufferLayout tightLayout = {};
    tightLayout.yRowPitchBytes = width;
    tightLayout.ySlicePitchBytes = (uint32_t)lumaSize;
    tightLayout.uvOffsetBytes = lumaSize;
    tightLayout.uvRowPitchBytes = width;
    tightLayout.uvSlicePitchBytes = (uint32_t)chromaSize;
    tightLayout.totalSizeBytes = lumaSize + chromaSize;

    std::vector<std::vector<uint8_t>> decodedFrames(options.frameNum);
    std::vector<uint8_t> sourceReference(tightLayout.totalSizeBytes);
    std::vector<uint8_t> streamBytes;
    std::vector<std::string> frameLines;
    std::vector<double> psnrs;
    double psnrSum = 0.0;
    double psnrMin = 99.0;
    uint32_t av1MetadataUnsupportedNum = 0;
    uint32_t feedbackNum = 0;
    uint64_t byteNum = 0;
    std::string failure;
    bool isFailureUnsupported = false;

    for (const FramePlan& plan : plans) {
        // Source
        uint8_t* source = (uint8_t*)NRI.MapBuffer(*sourceBuffer, 0, sourceLayout.totalSizeBytes);
        if (!source) {
            failure = "Failed to map the source buffer";
            break;
        }
        GenerateNv12Pattern(source, sourceLayout, size.codedWidth, size.codedHeight, width, height, plan.displayIndex);
        NRI.UnmapBuffer(*sourceBuffer);

        EncodeRequest encodeRequest = {};
        encodeRequest.nv12Buffer = sourceBuffer;
        encodeRequest.nv12BufferState = {nri::AccessBits::NONE, nri::StageBits::NONE};
        encodeRequest.nv12Layout = &sourceLayout;
        encodeRequest.nv12Width = size.codedWidth;
        encodeRequest.nv12Height = size.codedHeight;
        encodeRequest.quality = options.quality;
        encodeRequest.plan = plan;
        if (options.rateControlChangeFrame >= 0 && plan.codingIndex >= (uint32_t)options.rateControlChangeFrame)
            encodeRequest.quality.rateControl = options.rateControlB;

        if (options.verbose) {
            std::fprintf(stderr, "[%s] picture %u: %s display=%u poc=%d frameNum=%u setupSlot=%u reference=%d refs:", options.name.c_str(), plan.codingIndex, GetFrameTypeName(plan.type), plan.displayIndex, plan.poc, plan.frameNum, plan.setupSlot, plan.isReference ? 1 : 0);
            for (uint32_t i = 0; i < plan.referenceNum; i++)
                std::fprintf(stderr, " [slot=%u poc=%d list=%u]", plan.references[i].slot, plan.references[i].poc, plan.references[i].listIndex);
            std::fprintf(stderr, "\n");
        }

        EncodedFrame encodedFrame = {};
        if (!encoder->Encode(encodeRequest, encodedFrame)) {
            failure = "Frame " + std::to_string(plan.codingIndex) + ": " + encoder->GetStatus();
            isFailureUnsupported = encoder->IsUnsupported();
            break;
        }
        byteNum += encodedFrame.payload.size();
        feedbackNum += encodedFrame.feedbackResolved ? 1 : 0;
        av1MetadataUnsupportedNum += encodedFrame.av1MetadataUnsupported ? 1 : 0;
        streamBytes.insert(streamBytes.end(), encodedFrame.headers.begin(), encodedFrame.headers.end());
        streamBytes.insert(streamBytes.end(), encodedFrame.payload.begin(), encodedFrame.payload.end());

        if (options.encodeOnly) {
            psnrs.push_back(99.0);
            frameLines.push_back(std::to_string(plan.codingIndex) + "," + GetFrameTypeName(plan.type) + "," + std::to_string(plan.displayIndex) + "," + std::to_string(plan.poc) + "," + std::to_string(plan.setupSlot) + "," + std::to_string(encodedFrame.payload.size()));
            continue;
        }

        DecodeRequest decodeRequest = {};
        decodeRequest.frame = &encodedFrame;
        decodeRequest.nv12ReadbackBuffer = readbackBuffer;
        decodeRequest.nv12Layout = &readbackLayout;
        DecodedFrame decodedFrame = {};
        if (!decoder->Decode(decodeRequest, decodedFrame)) {
            failure = "Frame " + std::to_string(plan.codingIndex) + ": " + decoder->GetStatus();
            break;
        }

        // Compare
        const uint8_t* readback = (const uint8_t*)NRI.MapBuffer(*readbackBuffer, 0, readbackLayout.totalSizeBytes);
        if (!readback) {
            failure = "Failed to map the readback buffer";
            break;
        }
        std::vector<uint8_t>& decoded = decodedFrames[plan.displayIndex];
        decoded.resize(tightLayout.totalSizeBytes);
        for (uint32_t y = 0; y < height; y++)
            std::memcpy(decoded.data() + uint64_t(y) * width, readback + uint64_t(y) * readbackLayout.yRowPitchBytes, width);
        for (uint32_t y = 0; y < height / 2; y++)
            std::memcpy(decoded.data() + lumaSize + uint64_t(y) * width, readback + readbackLayout.uvOffsetBytes + uint64_t(y) * readbackLayout.uvRowPitchBytes, width);
        NRI.UnmapBuffer(*readbackBuffer);

        GenerateNv12Pattern(sourceReference.data(), tightLayout, width, height, width, height, plan.displayIndex);
        double lumaError = 0.0;
        double chromaError = 0.0;
        ComputeSquaredError(sourceReference.data(), decoded.data(), lumaSize, lumaError);
        ComputeSquaredError(sourceReference.data() + lumaSize, decoded.data() + lumaSize, chromaSize, chromaError);
        const double psnr = GetPsnr(lumaError, lumaSize);
        psnrSum += psnr;
        psnrMin = std::min(psnrMin, psnr);
        psnrs.push_back(psnr);

        char line[256] = {};
        std::snprintf(line, sizeof(line), "%u,%s,%u,%d,%u,%zu,%llu,%.2f,%.2f", plan.codingIndex, GetFrameTypeName(plan.type), plan.displayIndex, plan.poc, plan.setupSlot, encodedFrame.payload.size(), (unsigned long long)encodedFrame.feedback.averageQp, psnr, GetPsnr(chromaError, chromaSize));
        frameLines.push_back(line);
    }

    const bool wasDistinct = !options.encodeOnly && decoder->IsDistinct();

    // Dump
    if (!options.dumpDir.empty()) {
        const std::filesystem::path directory = std::filesystem::path(options.dumpDir) / options.name;
        std::error_code errorCode;
        std::filesystem::create_directories(directory, errorCode);

        WriteFile(directory / (std::string("stream.") + GetStreamExtension(config.codec)), streamBytes);

        std::vector<uint8_t> decodedBytes;
        std::vector<uint8_t> sourceBytes;
        for (uint32_t i = 0; i < options.frameNum; i++) {
            if (decodedFrames[i].empty() && !options.encodeOnly)
                continue;
            decodedBytes.insert(decodedBytes.end(), decodedFrames[i].begin(), decodedFrames[i].end());
            GenerateNv12Pattern(sourceReference.data(), tightLayout, width, height, width, height, i);
            sourceBytes.insert(sourceBytes.end(), sourceReference.begin(), sourceReference.end());
        }
        char info[512] = {};
        std::snprintf(info, sizeof(info), "codec=%s\nvideo=%ux%u\ncoded=%ux%u\nframes=%u\nencodeOnly=%d\n", GetCodecName(config.codec), size.videoWidth, size.videoHeight, size.codedWidth, size.codedHeight, (uint32_t)plans.size(), options.encodeOnly ? 1 : 0);
        WriteFile(directory / "info.txt", std::vector<uint8_t>(info, info + std::strlen(info)));
        WriteFile(directory / "decoded.yuv", decodedBytes);
        WriteFile(directory / "source.yuv", sourceBytes);

        std::string csv = "coding,type,display,poc,setupSlot,bytes,averageQp,psnrY,psnrUV\n";
        for (const std::string& line : frameLines)
            csv += line + "\n";
        WriteFile(directory / "frames.csv", std::vector<uint8_t>(csv.begin(), csv.end()));
    }

    cleanup();

    result.frameNum = (uint32_t)psnrs.size();
    result.byteNum = byteNum;
    result.psnrMin = psnrs.empty() ? 0.0 : psnrMin;
    result.psnrAvg = psnrs.empty() ? 0.0 : psnrSum / double(psnrs.size());

    char notes[256] = {};
    std::snprintf(notes, sizeof(notes), "coded=%ux%u refs=%u dpb=%s%s feedback=%u/%u%s", size.codedWidth, size.codedHeight, config.referenceNum, wasDistinct ? "distinct" : "coincide", config.av1Info == Av1InfoMode::METADATA && config.codec == SampleCodec::AV1 ? "" : "", feedbackNum, (uint32_t)plans.size(),
        av1MetadataUnsupportedNum ? " av1-metadata=UNSUPPORTED(fallback to payload header)" : "");
    result.notes = notes;

    if (!failure.empty())
        return finish(isFailureUnsupported ? StreamStatus::UNSUPPORTED : StreamStatus::FAIL, failure);
    if (result.psnrMin < options.minPsnr) {
        char message[128] = {};
        std::snprintf(message, sizeof(message), "PSNR %.2f dB is below %.2f dB", result.psnrMin, options.minPsnr);
        return finish(StreamStatus::FAIL, message);
    }

    return finish(StreamStatus::PASS, "");
}

//==================================================================================================
// Matrix
//==================================================================================================

std::vector<StreamOptions> MakeMatrixCases(const StreamOptions& base) {
    std::vector<StreamOptions> cases;

    for (SampleCodec codec : {SampleCodec::H264, SampleCodec::H265, SampleCodec::AV1}) {
        std::string prefix = codec == SampleCodec::H265 ? "h265" : (codec == SampleCodec::AV1 ? "av1" : "h264");

        auto add = [&](const char* name, auto&& configure) {
            StreamOptions options = base;
            options.name = prefix + "-" + name;
            options.config.codec = codec;
            configure(options);
            cases.push_back(options);
        };
        auto setRate = [](StreamOptions& options, RateControlKind kind, uint32_t kbps, uint32_t maxKbps) {
            options.quality.rateControl = {kind, kbps, maxKbps};
            options.minPsnr = 25.0;
        };

        add("idr", [](StreamOptions& o) { o.gop = "I"; });
        add("p", [](StreamOptions& o) { o.gop = "P"; });
        add("nonref", [](StreamOptions& o) { o.gop = "pP"; });
        add("dpb-coincide", [](StreamOptions& o) { o.config.decodeDpb = DecodeDpbMode::COINCIDE; });
        add("dpb-distinct", [](StreamOptions& o) { o.config.decodeDpb = DecodeDpbMode::DISTINCT; });
        add("dpb-array", [](StreamOptions& o) { o.config.dpbLayout = DpbLayout::ARRAY; });
        add("dpb-separate", [](StreamOptions& o) { o.config.dpbLayout = DpbLayout::SEPARATE; });
        add("input-plain", [](StreamOptions& o) { o.config.encodeInputOutputOnly = false; });
        add("cbr", [&](StreamOptions& o) { setRate(o, RateControlKind::CBR, 4000, 0); });
        add("vbr", [&](StreamOptions& o) { setRate(o, RateControlKind::VBR, 3000, 6000); });
        add("rc-change", [&](StreamOptions& o) {
            setRate(o, RateControlKind::CBR, 6000, 0);
            o.rateControlChangeFrame = 4;
            o.rateControlB = {RateControlKind::VBR, 2000, 4000};
        });
        add("rc-change-cqp", [&](StreamOptions& o) {
            o.rateControlChangeFrame = 4;
            o.rateControlB = {RateControlKind::CBR, 3000, 0};
            o.minPsnr = 25.0;
        });
        // Rate control limits: D3D12 enables them by flags, VK by the CBR/VBR structures
        add("cbr-qp-range", [&](StreamOptions& o) {
            setRate(o, RateControlKind::CBR, 4000, 0);
            o.quality.rateControl.qpMin = 30;
            o.quality.rateControl.qpMax = 30;
        });
        add("cbr-max-frame", [&](StreamOptions& o) {
            setRate(o, RateControlKind::CBR, 4000, 0);
            o.quality.rateControl.maxFrameBits = 120000;
        });
        add("vbr-vbv", [&](StreamOptions& o) {
            setRate(o, RateControlKind::VBR, 3000, 6000);
            o.quality.rateControl.vbvMs = 200;
            o.quality.rateControl.vbvInitialMs = 100;
        });
        add("no-feedback", [](StreamOptions& o) { o.config.encodeFeedback = false; });
        add("1080p", [](StreamOptions& o) {
            o.config.videoWidth = 1920;
            o.config.videoHeight = 1080;
            o.frameNum = 4;
        });
        add("4k", [](StreamOptions& o) {
            o.config.videoWidth = 3840;
            o.config.videoHeight = 2160;
            o.frameNum = 3;
        });
        add("long", [](StreamOptions& o) {
            o.frameNum = 40;
            o.idrPeriod = 17;
        });

        if (codec != SampleCodec::AV1) {
            add("refs3", [](StreamOptions& o) {
                o.config.referenceNum = 3;
                o.frameNum = 12;
            });
            add("refs3-nonref", [](StreamOptions& o) {
                o.config.referenceNum = 3;
                o.gop = "PpP";
                o.frameNum = 12;
            });
            add("b", [](StreamOptions& o) {
                o.gop = "BP";
                o.config.referenceNum = 2;
            });
            add("bb", [](StreamOptions& o) {
                o.gop = "BBP";
                o.config.referenceNum = 3;
                o.frameNum = 13;
            });
            add("b-distinct", [](StreamOptions& o) {
                o.gop = "BP";
                o.config.referenceNum = 2;
                o.config.decodeDpb = DecodeDpbMode::DISTINCT;
            });
        }

        if (codec == SampleCodec::H264) {
            add("ids", [](StreamOptions& o) {
                o.config.h264SpsId = 3;
                o.config.h264PpsId = 5;
                o.config.referenceNum = 2;
            });
        }

        if (codec == SampleCodec::H265) {
            add("ids-nolistmod", [](StreamOptions& o) {
                o.config.h265VpsId = 3;
                o.config.h265SpsId = 5;
                o.config.h265PpsId = 7;
                o.config.referenceNum = 2;
                o.config.h265ListsModification = false;
                o.config.reverseReferences = false;
            });
            add("b-nolistmod", [](StreamOptions& o) {
                o.gop = "BP";
                o.config.referenceNum = 2;
                o.config.h265ListsModification = false;
                o.config.reverseReferences = false;
                o.frameNum = 8;
            });
            add("ids", [](StreamOptions& o) {
                o.config.h265VpsId = 3;
                o.config.h265SpsId = 5;
                o.config.h265PpsId = 7;
                o.config.referenceNum = 2;
            });
            // The same without lists_modification_present_flag (no list modification syntax in the slice headers)
            add("refs3-nolistmod", [](StreamOptions& o) {
                o.config.referenceNum = 3;
                o.config.h265ListsModification = false;
                o.config.reverseReferences = false;
                o.frameNum = 12;
            });
            add("bb-nolistmod", [](StreamOptions& o) {
                o.gop = "BBP";
                o.config.referenceNum = 3;
                o.config.h265ListsModification = false;
                o.config.reverseReferences = false;
                o.frameNum = 13;
            });
            // The backend writes its own parameter sets, they are not available through NRI: encode only
            add("default-params", [](StreamOptions& o) {
                o.config.h265DefaultParameters = true;
                o.encodeOnly = true;
            });
            add("4k-default-params", [](StreamOptions& o) {
                o.config.h265DefaultParameters = true;
                o.encodeOnly = true;
                o.config.videoWidth = 3840;
                o.config.videoHeight = 2160;
                o.frameNum = 3;
            });
        }

        if (codec == SampleCodec::AV1) {
            add("metadata", [](StreamOptions& o) { o.config.av1Info = Av1InfoMode::METADATA; });
            add("header", [](StreamOptions& o) { o.config.av1Info = Av1InfoMode::HEADER; });
        }
    }

    return cases;
}


//==================================================================================================
// Decode of an external AV1 stream (OBU file, e.g. "ffmpeg -c:v libaom-av1 -f obu")
//==================================================================================================

#if NRISAMPLES_ENABLE_AV1_FILE_DECODING

namespace {

struct BitReader {
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    bool ok = true;

    uint32_t Bits(uint32_t n) {
        uint64_t v = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t bit = 0;
            if (pos < size * 8)
                bit = (data[pos >> 3] >> (7 - (pos & 7))) & 1;
            else
                ok = false;
            pos++;
            v = (v << 1) | bit;
        }
        return (uint32_t)v;
    }

    uint32_t Uvlc() {
        uint32_t zeros = 0;
        while (ok && !Bits(1)) {
            if (++zeros >= 32)
                return UINT32_MAX;
        }
        return Bits(zeros) + ((1u << zeros) - 1u);
    }
};

struct ObuSpan {
    uint8_t type = 0;
    size_t begin = 0;   // OBU header
    size_t payload = 0; // first payload byte
    size_t end = 0;
};

bool ReadObus(const std::vector<uint8_t>& data, std::vector<ObuSpan>& obus) {
    size_t pos = 0;
    while (pos < data.size()) {
        const uint8_t header = data[pos];
        const bool hasExtension = (header >> 2) & 1;
        const bool hasSize = (header >> 1) & 1;
        if ((header >> 7) || !hasSize)
            return false;

        size_t p = pos + 1 + (hasExtension ? 1 : 0);
        uint64_t size = 0;
        for (uint32_t i = 0; i < 8; i++) {
            if (p >= data.size())
                return false;
            const uint8_t byte = data[p++];
            size |= uint64_t(byte & 0x7F) << (7 * i);
            if (!(byte & 0x80))
                break;
        }
        if (p + size > data.size())
            return false;

        obus.push_back({(uint8_t)((header >> 3) & 15), pos, p, (size_t)(p + size)});
        pos = (size_t)(p + size);
    }

    return true;
}

// Sequence header OBU payload -> neutral description (AV1 5.5)
bool ParseSequenceHeader(const uint8_t* data, size_t size, nri::VideoAV1SequenceDesc& desc) {
    BitReader r = {data, size};
    desc = {};
    auto flag = [&](nri::VideoAV1SequenceBits bit) {
        if (r.Bits(1))
            desc.flags |= bit;
    };

    desc.seqProfile = (uint8_t)r.Bits(3);
    flag(nri::VideoAV1SequenceBits::STILL_PICTURE);
    if (r.Bits(1))
        return false; // reduced_still_picture_header
    if (r.Bits(1)) {  // timing_info_present_flag
        desc.flags |= nri::VideoAV1SequenceBits::TIMING_INFO_PRESENT;
        desc.numUnitsInDisplayTick = r.Bits(32);
        desc.timeScale = r.Bits(32);
        if (r.Bits(1))
            desc.numTicksPerPictureMinus1 = r.Uvlc();
        if (r.Bits(1))
            return false; // decoder_model_info_present_flag
    }
    const bool initialDisplayDelayPresent = r.Bits(1);
    if (initialDisplayDelayPresent)
        desc.flags |= nri::VideoAV1SequenceBits::INITIAL_DISPLAY_DELAY_PRESENT;
    const uint32_t operatingPointNum = r.Bits(5) + 1;
    for (uint32_t i = 0; i < operatingPointNum; i++) {
        r.Bits(12); // operating_point_idc
        const uint32_t level = r.Bits(5);
        if (i == 0)
            desc.level = (uint8_t)level;
        if (level > 7)
            r.Bits(1); // seq_tier
        if (initialDisplayDelayPresent && r.Bits(1))
            r.Bits(4);
    }
    desc.frameWidthBitsMinus1 = (uint8_t)r.Bits(4);
    desc.frameHeightBitsMinus1 = (uint8_t)r.Bits(4);
    desc.maxFrameWidthMinus1 = (uint16_t)r.Bits(desc.frameWidthBitsMinus1 + 1);
    desc.maxFrameHeightMinus1 = (uint16_t)r.Bits(desc.frameHeightBitsMinus1 + 1);
    if (r.Bits(1)) { // frame_id_numbers_present_flag
        desc.flags |= nri::VideoAV1SequenceBits::FRAME_ID_NUMBERS_PRESENT;
        desc.deltaFrameIdLengthMinus2 = (uint8_t)r.Bits(4);
        desc.additionalFrameIdLengthMinus1 = (uint8_t)r.Bits(3);
    }
    flag(nri::VideoAV1SequenceBits::USE_128X128_SUPERBLOCK);
    flag(nri::VideoAV1SequenceBits::ENABLE_FILTER_INTRA);
    flag(nri::VideoAV1SequenceBits::ENABLE_INTRA_EDGE_FILTER);
    flag(nri::VideoAV1SequenceBits::ENABLE_INTERINTRA_COMPOUND);
    flag(nri::VideoAV1SequenceBits::ENABLE_MASKED_COMPOUND);
    flag(nri::VideoAV1SequenceBits::ENABLE_WARPED_MOTION);
    flag(nri::VideoAV1SequenceBits::ENABLE_DUAL_FILTER);
    const bool enableOrderHint = r.Bits(1);
    if (enableOrderHint) {
        desc.flags |= nri::VideoAV1SequenceBits::ENABLE_ORDER_HINT;
        flag(nri::VideoAV1SequenceBits::ENABLE_JNT_COMP);
        flag(nri::VideoAV1SequenceBits::ENABLE_REF_FRAME_MVS);
    }
    desc.seqForceScreenContentTools = r.Bits(1) ? 2 : (uint8_t)r.Bits(1);
    desc.seqForceIntegerMv = 2;
    if (desc.seqForceScreenContentTools > 0)
        desc.seqForceIntegerMv = r.Bits(1) ? 2 : (uint8_t)r.Bits(1);
    if (enableOrderHint)
        desc.orderHintBitsMinus1 = (uint8_t)r.Bits(3);
    flag(nri::VideoAV1SequenceBits::ENABLE_SUPERRES);
    flag(nri::VideoAV1SequenceBits::ENABLE_CDEF);
    flag(nri::VideoAV1SequenceBits::ENABLE_RESTORATION);

    // color_config()
    const bool highBitDepth = r.Bits(1);
    desc.bitDepth = highBitDepth ? 10 : 8;
    if (desc.seqProfile == 2 && highBitDepth && r.Bits(1))
        desc.bitDepth = 12;
    const bool monochrome = desc.seqProfile == 1 ? false : r.Bits(1);
    if (monochrome)
        desc.flags |= nri::VideoAV1SequenceBits::MONO_CHROME;
    uint32_t colorPrimaries = 2, transfer = 2, matrix = 2;
    if (r.Bits(1)) {
        desc.flags |= nri::VideoAV1SequenceBits::COLOR_DESCRIPTION_PRESENT;
        colorPrimaries = r.Bits(8);
        transfer = r.Bits(8);
        matrix = r.Bits(8);
    }
    desc.colorPrimaries = (uint8_t)colorPrimaries;
    desc.transferCharacteristics = (uint8_t)transfer;
    desc.matrixCoefficients = (uint8_t)matrix;
    if (monochrome) {
        flag(nri::VideoAV1SequenceBits::COLOR_RANGE);
        desc.subsamplingX = desc.subsamplingY = 1;
    } else if (colorPrimaries == 1 && transfer == 13 && matrix == 0) {
        desc.flags |= nri::VideoAV1SequenceBits::COLOR_RANGE;
    } else {
        flag(nri::VideoAV1SequenceBits::COLOR_RANGE);
        if (desc.seqProfile == 0)
            desc.subsamplingX = desc.subsamplingY = 1;
        else if (desc.seqProfile == 1)
            desc.subsamplingX = desc.subsamplingY = 0;
        else if (desc.bitDepth == 12) {
            desc.subsamplingX = (uint8_t)r.Bits(1);
            desc.subsamplingY = desc.subsamplingX ? (uint8_t)r.Bits(1) : 0;
        } else {
            desc.subsamplingX = 1;
            desc.subsamplingY = 0;
        }
        if (desc.subsamplingX && desc.subsamplingY)
            desc.chromaSamplePosition = (uint8_t)r.Bits(2);
    }
    if (!monochrome)
        flag(nri::VideoAV1SequenceBits::SEPARATE_UV_DELTA_Q);
    flag(nri::VideoAV1SequenceBits::FILM_GRAIN_PARAMS_PRESENT);

    return r.ok;
}

} // namespace

StreamResult RunDecodeFile(const RunEnvironment& environment, const std::string& path, const std::string& refYuvPath, const std::string& name, const std::string& dumpDir, bool verbose) {
    StreamResult result = {};
    MessageStats stats;

    auto finish = [&](StreamStatus status, const std::string& message) {
        result.status = status;
        result.message = message;
        result.nriErrorNum = stats.errorNum;
        result.nriWarningNum = stats.warningNum;
        if (status == StreamStatus::PASS && stats.errorNum) {
            result.status = StreamStatus::FAIL;
            result.message = "NRI reported " + std::to_string(stats.errorNum) + " error(s): " + (stats.errors.empty() ? "" : stats.errors[0]);
        }
        return result;
    };

    auto readFile = [](const std::string& filePath, std::vector<uint8_t>& bytes) {
        FILE* file = std::fopen(filePath.c_str(), "rb");
        if (!file)
            return false;
        std::fseek(file, 0, SEEK_END);
        const long size = std::ftell(file);
        std::fseek(file, 0, SEEK_SET);
        bytes.resize(size > 0 ? (size_t)size : 0);
        const bool ok = bytes.empty() || std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
        std::fclose(file);
        return ok;
    };

    std::vector<uint8_t> fileBytes;
    std::vector<uint8_t> refBytes;
    const bool hasReference = !refYuvPath.empty();
    if (!readFile(path, fileBytes))
        return finish(StreamStatus::FAIL, "Failed to read " + path);
    if (hasReference && (!readFile(refYuvPath, refBytes) || refBytes.empty()))
        return finish(StreamStatus::FAIL, "Failed to read or empty reference " + refYuvPath);

    std::vector<ObuSpan> obus;
    if (!ReadObus(fileBytes, obus))
        return finish(StreamStatus::FAIL, "Not a low-overhead OBU stream (every OBU needs obu_size)");

    nri::VideoAV1SequenceDesc sequence = {};
    bool hasSequence = false;
    for (const ObuSpan& obu : obus) {
        if (obu.type == 1 && !hasSequence) {
            if (!ParseSequenceHeader(fileBytes.data() + obu.payload, obu.end - obu.payload, sequence))
                return finish(StreamStatus::FAIL, "Unsupported AV1 sequence header");
            hasSequence = true;
        }
    }
    if (!hasSequence)
        return finish(StreamStatus::FAIL, "No AV1 sequence header");

    if (verbose)
        std::fprintf(stderr, "[%s] sequence: flags=0x%X profile=%u bitDepth=%u ss=%u%u size=%ux%u bits=%u/%u orderHintBitsMinus1=%u sct=%u imv=%u level=%u cp=%u tc=%u mc=%u csp=%u\n", name.c_str(), (uint32_t)sequence.flags, sequence.seqProfile, sequence.bitDepth, sequence.subsamplingX, sequence.subsamplingY, sequence.maxFrameWidthMinus1 + 1u, sequence.maxFrameHeightMinus1 + 1u, sequence.frameWidthBitsMinus1, sequence.frameHeightBitsMinus1, sequence.orderHintBitsMinus1, sequence.seqForceScreenContentTools, sequence.seqForceIntegerMv, sequence.level, sequence.colorPrimaries, sequence.transferCharacteristics, sequence.matrixCoefficients, sequence.chromaSamplePosition);

    VideoConfig config = {};
    config.codec = SampleCodec::AV1;
    config.videoWidth = sequence.maxFrameWidthMinus1 + 1u;
    config.videoHeight = sequence.maxFrameHeightMinus1 + 1u;
    config.referenceNum = 8;
    config.decodePictureAtVideoSize = true; // the AV1 frame size is the picture extent (not the texture size)
    config.av1FilmGrain = !!(sequence.flags & nri::VideoAV1SequenceBits::FILM_GRAIN_PARAMS_PRESENT);

    // Device
    nri::AdapterDesc adapters[8] = {};
    uint32_t adapterNum = helper::GetCountOf(adapters);
    if (nri::nriEnumerateAdapters(adapters, adapterNum) != nri::Result::SUCCESS || !adapterNum)
        return finish(StreamStatus::UNSUPPORTED, "No graphics adapters");
    uint32_t apiAdapterNum = 0;
    for (uint32_t i = 0; i < adapterNum; i++) {
        if (adapters[i].supportedGraphicsAPIs & environment.graphicsAPI)
            adapters[apiAdapterNum++] = adapters[i];
    }
    if (!apiAdapterNum)
        return finish(StreamStatus::UNSUPPORTED, "No adapters support the requested graphics API");
    const nri::AdapterDesc& adapter = adapters[std::min(environment.adapterIndex, apiAdapterNum - 1)];
    result.adapterName = adapter.name;
    if (!adapter.queueNum[(uint32_t)nri::QueueType::VIDEO_DECODE])
        return finish(StreamStatus::UNSUPPORTED, "The adapter has no NRI video decode queue");

    nri::QueueFamilyDesc queueFamilies[2] = {};
    queueFamilies[0].queueNum = 1;
    queueFamilies[0].queueType = nri::QueueType::GRAPHICS;
    queueFamilies[1].queueNum = 1;
    queueFamilies[1].queueType = nri::QueueType::VIDEO_DECODE;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = environment.graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = environment.debugAPI;
    deviceCreationDesc.disableD3D12EnhancedBarriers = environment.graphicsAPI == nri::GraphicsAPI::D3D12 && !environment.enhancedBarriers; // legacy barriers by default
    deviceCreationDesc.enableNRIValidation = environment.debugNRI;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &adapter;
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = helper::GetCountOf(queueFamilies);
    deviceCreationDesc.callbackInterface.MessageCallback = MessageCallback;
    deviceCreationDesc.callbackInterface.AbortExecution = AbortCallback;
    deviceCreationDesc.callbackInterface.userArg = &stats;

    DeviceHolder deviceHolder;
    if (nri::nriCreateDevice(deviceCreationDesc, deviceHolder.device) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to create the device");
    nri::Device& device = *deviceHolder.device;

    NRIInterface NRI = {};
    nri::VideoInterface Video = {};
    if (nri::nriGetInterface(device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) != nri::Result::SUCCESS || nri::nriGetInterface(device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to get NRI interfaces");
    if (nri::nriGetInterface(device, NRI_INTERFACE(nri::VideoInterface), &Video) != nri::Result::SUCCESS || !Video.CreateVideoSession)
        return finish(StreamStatus::UNSUPPORTED, "NRI video interface is unavailable");

    VideoContext context = {};
    context.nri = &NRI;
    context.video = &Video;
    context.device = &device;
    context.graphicsAPI = environment.graphicsAPI;
    if (NRI.GetQueue(device, nri::QueueType::GRAPHICS, 0, context.graphicsQueue) != nri::Result::SUCCESS || NRI.GetQueue(device, nri::QueueType::VIDEO_DECODE, 0, context.decodeQueue) != nri::Result::SUCCESS)
        return finish(StreamStatus::FAIL, "Failed to get queues");

    // Decode-only video size
    nri::VideoSessionDesc sessionDesc = {};
    sessionDesc.type = nri::VideoSessionType::DECODE;
    sessionDesc.codec = nri::VideoCodec::AV1;
    sessionDesc.format = nri::Format::NV12_UNORM;
    sessionDesc.width = config.videoWidth;
    sessionDesc.height = config.videoHeight;
    sessionDesc.maxReferenceNum = config.referenceNum;
    sessionDesc.av1FilmGrain = config.av1FilmGrain;
    nri::VideoCapabilities capabilities = {};
    Video.GetVideoCapabilities(device, sessionDesc, capabilities);
    if (!capabilities.pictureAccessGranularityWidth || !capabilities.pictureAccessGranularityHeight)
        return finish(StreamStatus::UNSUPPORTED, config.av1FilmGrain ? "AV1 film grain decode is unsupported" : "AV1 decode is unsupported");
    VideoSize size = {};
    size.videoWidth = config.videoWidth;
    size.videoHeight = config.videoHeight;
    size.codedWidth = (uint32_t)AlignUp(std::max(config.videoWidth, capabilities.widthMin), capabilities.pictureAccessGranularityWidth);
    size.codedHeight = (uint32_t)AlignUp(std::max(config.videoHeight, capabilities.heightMin), capabilities.pictureAccessGranularityHeight);
    size.decodeBitstreamSizeAlignment = std::max(capabilities.bitstreamSizeAlignment, 1u);
    size.bitstreamSize = AlignUp(std::max<uint64_t>(2 * 1024 * 1024, uint64_t(size.codedWidth) * size.codedHeight), 1024 * 1024);

    CodecParameters codecParameters = {};
    codecParameters.av1Sequence = sequence;

    auto decoder = std::make_unique<Decoder>();
    nri::Buffer* readbackBuffer = nullptr;
    auto cleanup = [&]() {
        NRI.DeviceWaitIdle(&device);
        decoder.reset();
        if (readbackBuffer)
            NRI.DestroyBuffer(readbackBuffer);
    };

    if (!decoder->Initialize(context, config, size, codecParameters)) {
        const bool isUnsupported = decoder->IsUnsupported();
        const std::string message = decoder->GetStatus();
        cleanup();
        return finish(isUnsupported ? StreamStatus::UNSUPPORTED : StreamStatus::FAIL, message);
    }

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(device);
    const Nv12BufferLayout readbackLayout = MakeNv12BufferLayout(deviceDesc, size.videoWidth, size.videoHeight);
    nri::BufferDesc readbackBufferDesc = {};
    readbackBufferDesc.size = readbackLayout.totalSizeBytes;
    if (NRI.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_READBACK, 0.0f, readbackBufferDesc, readbackBuffer) != nri::Result::SUCCESS) {
        cleanup();
        return finish(StreamStatus::FAIL, "Failed to create the readback buffer");
    }

    const uint32_t width = size.videoWidth;
    const uint32_t height = size.videoHeight;
    const uint64_t lumaSize = uint64_t(width) * height;
    const uint64_t frameSize = lumaSize * 3 / 2;

    struct RefSlot {
        bool valid = false;
        uint32_t dpbSlot = 0;
        nri::VideoFrameType type = nri::VideoFrameType::IDR;
        uint8_t orderHint = 0;
        uint32_t frameId = 0;
        uint8_t saved[8] = {};
    };
    RefSlot refs[8] = {};
    nri::VideoAV1SavedFrameState savedFrameStates[8] = {}; // state of the frames in the reference slots for headers that inherit it

    std::vector<uint8_t> decodedBytes;
    std::vector<std::string> frameLines;
    double psnrMin = 99.0;
    double psnrSum = 0.0;
    uint32_t outputNum = 0;
    uint32_t codingIndex = 0;
    uint64_t byteNum = 0;
    std::string failure;

    for (const ObuSpan& obu : obus) {
        if (obu.type != 6) {
            if (obu.type == 3 || obu.type == 4 || obu.type == 8) {
                failure = "Unsupported OBU layout (OBU_FRAME_HEADER / OBU_TILE_GROUP): use a single OBU_FRAME per picture";
                break;
            }
            continue;
        }

        EncodedFrame frame = {};
        frame.payload.assign(fileBytes.begin() + obu.begin, fileBytes.begin() + obu.end);
        frame.feedback.encodedBitstreamWrittenBytes = frame.payload.size();
        byteNum += frame.payload.size();

        nri::VideoAV1ReferenceDesc snapshot[8] = {};
        for (uint32_t j = 0; j < 8; j++) {
            snapshot[j].name = nri::VideoAV1ReferenceName::NONE;
            snapshot[j].refFrameIndex = (uint8_t)j;
            snapshot[j].frameType = refs[j].type;
            snapshot[j].orderHint = refs[j].orderHint;
            snapshot[j].frameId = refs[j].frameId;
            snapshot[j].slot = refs[j].dpbSlot;
            snapshot[j].savedOrderHints = refs[j].saved;
        }
        const bool hasReferences = refs[0].valid;

        nri::VideoAV1EncodeDecodeInfoDesc infoDesc = {};
        infoDesc.feedback = &frame.feedback;
        infoDesc.sequence = &sequence;
        infoDesc.encodedPayloadHeader = frame.payload.data();
        infoDesc.encodedPayloadHeaderSize = frame.payload.size();
        infoDesc.references = hasReferences ? snapshot : nullptr;
        infoDesc.referenceNum = hasReferences ? 8 : 0;
        infoDesc.savedFrameStates = savedFrameStates;
        const nri::Result infoResult = nri::video::av1::GetEncodeDecodeInfoFromHeader(infoDesc, frame.av1DecodeInfo);
        if (infoResult != nri::Result::SUCCESS) {
            failure = "Picture " + std::to_string(codingIndex) + ": the AV1 frame header is not supported by the NRI parser";
            break;
        }
        frame.hasAv1DecodeInfo = true;

        const nri::VideoAV1DecodePictureDesc& picture = frame.av1DecodeInfo.picture;
        const bool isKey = picture.frameType == nri::VideoFrameType::IDR;

        FramePlan& plan = frame.plan;
        plan.type = isKey ? nri::VideoFrameType::IDR : nri::VideoFrameType::P;
        plan.codingIndex = codingIndex;
        plan.displayIndex = outputNum;
        plan.isReference = picture.refreshFrameFlags != 0;
        plan.isDisplay = true;

        bool used[MAX_DPB_SLOT_NUM] = {};
        for (const RefSlot& ref : refs)
            used[ref.dpbSlot] = used[ref.dpbSlot] || ref.valid;
        plan.setupSlot = MAX_DPB_SLOT_NUM;
        for (uint32_t slot = 0; slot < MAX_DPB_SLOT_NUM; slot++) {
            if (!used[slot]) {
                plan.setupSlot = slot;
                break;
            }
        }
        if (plan.setupSlot == MAX_DPB_SLOT_NUM) {
            failure = "No free DPB slot";
            break;
        }

        // Distinct DPB slots referenced by the picture
        if (!isKey) {
            for (uint32_t i = 0; i < picture.referenceNum; i++) {
                const uint32_t slot = picture.references[i].slot;
                bool found = false;
                for (uint32_t k = 0; k < plan.referenceNum; k++)
                    found = found || plan.references[k].slot == slot;
                if (!found && plan.referenceNum < MAX_REFERENCE_NUM) {
                    PlanReference& reference = plan.references[plan.referenceNum++];
                    reference.slot = slot;
                    reference.type = picture.references[i].frameType;
                    reference.orderHint = picture.references[i].orderHint;
                    reference.frameId = picture.references[i].frameId;
                }
            }
        }

        // "SavedOrderHints" of this picture (AV1 7.20): OrderHints[] indexed by reference name
        uint8_t savedOrderHints[8] = {};
        savedOrderHints[0] = picture.orderHint;
        if (isKey) {
            for (uint8_t& hint : savedOrderHints)
                hint = picture.orderHint;
        } else {
            for (uint32_t i = 0; i < picture.referenceNum; i++) {
                const uint8_t nameIndex = nri::video::av1::GetReferenceNameIndex(picture.references[i].name);
                if (nameIndex < 7)
                    savedOrderHints[1 + nameIndex] = picture.references[i].orderHint;
            }
        }
        frame.hasAv1SavedOrderHints = true;
        for (const RefSlot& ref : refs) {
            if (ref.valid)
                std::memcpy(frame.av1SavedOrderHints[ref.dpbSlot], ref.saved, sizeof(ref.saved));
        }

        if (verbose) {
            std::fprintf(stderr, "[%s] picture %u: %s orderHint=%u refresh=0x%02X setupSlot=%u refs:", name.c_str(), codingIndex, isKey ? "KEY" : "INTER", picture.orderHint, picture.refreshFrameFlags, plan.setupSlot);
            for (uint32_t i = 0; i < picture.referenceNum; i++)
                std::fprintf(stderr, " [name=%u idx=%u slot=%u hint=%u]", (uint32_t)picture.references[i].name, picture.references[i].refFrameIndex, picture.references[i].slot, picture.references[i].orderHint);
            std::fprintf(stderr, "\n    payload=%zu bitstreamOffset=%llu bitstreamSize=%llu tileNum=%u", frame.payload.size(), (unsigned long long)frame.av1DecodeInfo.bitstreamOffset, (unsigned long long)frame.av1DecodeInfo.bitstreamSize, picture.tileNum);
            for (uint32_t i = 0; i < picture.tileNum && i < 4; i++)
                std::fprintf(stderr, " tile%u=[off=%u size=%u r=%u c=%u]", i, picture.tiles[i].offset, picture.tiles[i].size, picture.tiles[i].row, picture.tiles[i].column);
            std::fprintf(stderr, "\n    flags=0x%X baseQ=%u txMode=%u interp=%u cdefBits=%u frameHeaderOffset=%u lr=[%u %u %u]", (uint32_t)picture.flags, picture.baseQIndex, picture.txMode, picture.interpolationFilter, picture.cdefBits, picture.frameHeaderOffset, picture.loopRestoration ? picture.loopRestoration->frameRestorationType[0] : 9, picture.loopRestoration ? picture.loopRestoration->frameRestorationType[1] : 9, picture.loopRestoration ? picture.loopRestoration->frameRestorationType[2] : 9);
            std::fprintf(stderr, " skipMode=%d [%u %u]\n", !!(picture.flags & nri::VideoAV1PictureBits::SKIP_MODE_PRESENT), (uint32_t)picture.skipModeFrames[0], (uint32_t)picture.skipModeFrames[1]);
        }

        DecodeRequest decodeRequest = {};
        decodeRequest.frame = &frame;
        decodeRequest.nv12ReadbackBuffer = readbackBuffer;
        decodeRequest.nv12Layout = &readbackLayout;
        DecodedFrame decodedFrame = {};
        if (!decoder->Decode(decodeRequest, decodedFrame)) {
            failure = "Picture " + std::to_string(codingIndex) + ": " + decoder->GetStatus();
            break;
        }

        // Shown pictures only (the first frame header bits: show_existing_frame, frame_type, show_frame)
        const bool isShown = !!(picture.flags & nri::VideoAV1PictureBits::SHOW_FRAME);
        if (isShown) {
            const uint8_t* readback = (const uint8_t*)NRI.MapBuffer(*readbackBuffer, 0, readbackLayout.totalSizeBytes);
            if (!readback) {
                failure = "Failed to map the readback buffer";
                break;
            }
            const size_t offset = decodedBytes.size();
            decodedBytes.resize(offset + frameSize);
            for (uint32_t y = 0; y < height; y++)
                std::memcpy(decodedBytes.data() + offset + uint64_t(y) * width, readback + uint64_t(y) * readbackLayout.yRowPitchBytes, width);
            for (uint32_t y = 0; y < height / 2; y++)
                std::memcpy(decodedBytes.data() + offset + lumaSize + uint64_t(y) * width, readback + readbackLayout.uvOffsetBytes + uint64_t(y) * readbackLayout.uvRowPitchBytes, width);
            NRI.UnmapBuffer(*readbackBuffer);

            double psnr = 99.0;
            if (hasReference) {
                if (offset + frameSize > refBytes.size()) {
                    failure = "Reference has fewer frames";
                    break;
                }
                double lumaError = 0.0;
                ComputeSquaredError(refBytes.data() + offset, decodedBytes.data() + offset, lumaSize, lumaError);
                double chromaError = 0.0;
                ComputeSquaredError(refBytes.data() + offset + lumaSize, decodedBytes.data() + offset + lumaSize, lumaSize / 2, chromaError);
                psnr = (lumaError == 0.0 && chromaError == 0.0) ? 99.0 : std::min(98.0, GetPsnr(lumaError + chromaError, lumaSize * 3 / 2));
            }
            psnrMin = std::min(psnrMin, psnr);
            psnrSum += psnr;
            outputNum++;
            char line[128] = {};
            std::snprintf(line, sizeof(line), "%u,%s,%u,%u,%u,%zu,0,%.2f,0", codingIndex, isKey ? "IDR" : "P", plan.displayIndex, picture.orderHint, plan.setupSlot, frame.payload.size(), psnr);
            frameLines.push_back(line);
        }

        // Reference update (AV1 7.20)
        for (uint32_t j = 0; j < 8; j++) {
            if (picture.refreshFrameFlags & (1u << j)) {
                refs[j].valid = true;
                refs[j].dpbSlot = plan.setupSlot;
                refs[j].type = plan.type;
                refs[j].orderHint = picture.orderHint;
                refs[j].frameId = picture.currentFrameId;
                std::memcpy(refs[j].saved, savedOrderHints, sizeof(savedOrderHints));
                savedFrameStates[j] = frame.av1DecodeInfo.savedFrameState;
            }
        }
        codingIndex++;
    }

    cleanup();

    if (!dumpDir.empty()) {
        const std::filesystem::path directory = std::filesystem::path(dumpDir) / name;
        std::error_code errorCode;
        std::filesystem::create_directories(directory, errorCode);
        WriteFile(directory / "stream.obu", fileBytes);
        WriteFile(directory / "decoded.yuv", decodedBytes);
        if (hasReference)
            WriteFile(directory / "source.yuv", refBytes);
        char info[256] = {};
        std::snprintf(info, sizeof(info), "codec=AV1\nvideo=%ux%u\ncoded=%ux%u\nframes=%u\nencodeOnly=0\n", width, height, size.codedWidth, size.codedHeight, outputNum);
        WriteFile(directory / "info.txt", std::vector<uint8_t>(info, info + std::strlen(info)));
        std::string csv = "coding,type,display,poc,setupSlot,bytes,averageQp,psnrY,psnrUV\n";
        for (const std::string& line : frameLines)
            csv += line + "\n";
        WriteFile(directory / "frames.csv", std::vector<uint8_t>(csv.begin(), csv.end()));
    }

    result.frameNum = outputNum;
    result.byteNum = byteNum;
    result.psnrMin = outputNum ? psnrMin : 0.0;
    result.psnrAvg = outputNum ? psnrSum / outputNum : 0.0;
    result.notes = "coded=" + std::to_string(size.codedWidth) + "x" + std::to_string(size.codedHeight) + " refs=8 external-av1" + (hasReference ? " (bit-exact vs reference required)" : " (no reference)");

    if (!failure.empty())
        return finish(StreamStatus::FAIL, failure);
    if (!outputNum)
        return finish(StreamStatus::FAIL, "No picture was decoded");
    if (hasReference && (outputNum * frameSize != refBytes.size() || result.psnrMin < 99.0)) {
        char message[128] = {};
        std::snprintf(message, sizeof(message), "decoded %u frames, reference has %llu; min PSNR vs reference %.2f dB (bit-exact is 99)", outputNum, (unsigned long long)(refBytes.size() / frameSize), result.psnrMin);
        return finish(StreamStatus::FAIL, message);
    }

    return finish(StreamStatus::PASS, "");
}

#else

StreamResult RunDecodeFile(const RunEnvironment&, const std::string&, const std::string&, const std::string&, const std::string&, bool) {
    StreamResult result = {};
    result.status = StreamStatus::UNSUPPORTED;
    result.message = "External AV1 file decoding is disabled; configure with -DNRISAMPLES_ENABLE_AV1_FILE_DECODING=ON to enable this developer feature";
    return result;
}

#endif

} // namespace video_sample
