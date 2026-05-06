// © 2021 NVIDIA Corporation

#if defined(_WIN32)
#    include <d3d12.h>
#endif

#include "NRIFramework.h"

#include "Extensions/NRIVideo.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <utility>

namespace {

constexpr uint32_t VIDEO_WIDTH = 1920;
constexpr uint32_t VIDEO_HEIGHT = 1088;
constexpr double ROUND_TRIP_INTERVAL_SEC = 1.0 / 60.0;
constexpr uint64_t BITSTREAM_SIZE = 2 * 1024 * 1024;
constexpr uint64_t ENCODED_SLICE_OFFSET = 4096;
constexpr uint64_t AV1_HEADER_READBACK_SIZE = 4096;
constexpr uint64_t METADATA_SIZE = 4 * 1024 * 1024;
constexpr uint64_t RESOLVED_METADATA_SIZE = 4096;

static_assert(VIDEO_WIDTH % 4 == 0, "Compute-backed NV12 writer expects width divisible by 4");
static_assert(VIDEO_WIDTH % 16 == 0, "H.264 macroblock width should stay aligned");
static_assert(VIDEO_HEIGHT % 16 == 0, "H.264 macroblock height should stay aligned");

enum PatternOperation : uint32_t {
    OP_GENERATE_PATTERN = 0,
    OP_NV12_TO_PREVIEW = 1,
};

enum class SampleCodec : uint8_t {
    H264,
    H265,
    AV1,
};

static const char* GetCodecName(SampleCodec codec) {
    switch (codec) {
        case SampleCodec::H265:
            return "H.265";
        case SampleCodec::AV1:
            return "AV1";
        case SampleCodec::H264:
        default:
            return "H.264";
    }
}

static nri::VideoCodec GetNriCodec(SampleCodec codec) {
    switch (codec) {
        case SampleCodec::H265:
            return nri::VideoCodec::H265;
        case SampleCodec::AV1:
            return nri::VideoCodec::AV1;
        case SampleCodec::H264:
        default:
            return nri::VideoCodec::H264;
    }
}

static uint64_t GetEncodedPayloadHeaderSkip(SampleCodec codec, uint64_t encodedBitstreamBytes) {
    const uint64_t headerSize = codec == SampleCodec::H264 ? 1 : 0;
    return std::min(headerSize, encodedBitstreamBytes);
}

static nri::VideoAV1SequenceDesc MakeAV1SequenceDesc() {
    nri::VideoAV1SequenceDesc desc = {};
    desc.flags = nri::VideoAV1SequenceBits::ENABLE_ORDER_HINT | nri::VideoAV1SequenceBits::ENABLE_CDEF | nri::VideoAV1SequenceBits::ENABLE_RESTORATION | nri::VideoAV1SequenceBits::COLOR_DESCRIPTION_PRESENT;
    desc.bitDepth = 8;
    desc.subsamplingX = 1;
    desc.subsamplingY = 1;
    desc.maxFrameWidthMinus1 = VIDEO_WIDTH - 1;
    desc.maxFrameHeightMinus1 = VIDEO_HEIGHT - 1;
    desc.frameWidthBitsMinus1 = 15;
    desc.frameHeightBitsMinus1 = 15;
    desc.orderHintBitsMinus1 = 7;
    desc.seqForceIntegerMv = 2;
    desc.seqForceScreenContentTools = 2;
    desc.colorPrimaries = 1;
    desc.transferCharacteristics = 1;
    desc.matrixCoefficients = 1;
    desc.chromaSamplePosition = 1;
    return desc;
}

struct QueuedFrame {
    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
};

struct PatternConstants {
    uint32_t width = VIDEO_WIDTH;
    uint32_t height = VIDEO_HEIGHT;
    uint32_t yOffsetBytes = 0;
    uint32_t yRowPitchBytes = 0;
    uint32_t uvRowPitchBytes = 0;
    uint32_t uvOffsetBytes = 0;
    uint32_t operation = OP_GENERATE_PATTERN;
    float time = 0.0f;
    uint32_t _padding = 0;
    uint32_t _padding1 = 0;
};

struct Nv12BufferLayout {
    uint32_t yRowPitchBytes = VIDEO_WIDTH;
    uint32_t ySlicePitchBytes = VIDEO_WIDTH * VIDEO_HEIGHT;
    uint64_t uvOffsetBytes = uint64_t(VIDEO_WIDTH) * VIDEO_HEIGHT;
    uint32_t uvRowPitchBytes = VIDEO_WIDTH;
    uint32_t uvSlicePitchBytes = VIDEO_WIDTH * VIDEO_HEIGHT / 2;
    uint64_t totalSizeBytes = uint64_t(VIDEO_WIDTH) * VIDEO_HEIGHT * 3 / 2;
};

static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

static Nv12BufferLayout MakeNv12BufferLayout(const nri::DeviceDesc& deviceDesc) {
    const uint32_t rowAlignment = std::max(deviceDesc.memoryAlignment.uploadBufferTextureRow, 1u);
    const uint32_t sliceAlignment = std::max(deviceDesc.memoryAlignment.uploadBufferTextureSlice, 1u);

    Nv12BufferLayout layout = {};
    layout.yRowPitchBytes = (uint32_t)AlignUp(VIDEO_WIDTH, rowAlignment);
    layout.ySlicePitchBytes = (uint32_t)AlignUp(uint64_t(layout.yRowPitchBytes) * VIDEO_HEIGHT, sliceAlignment);
    layout.uvOffsetBytes = layout.ySlicePitchBytes;
    layout.uvRowPitchBytes = (uint32_t)AlignUp(VIDEO_WIDTH, rowAlignment);
    layout.uvSlicePitchBytes = (uint32_t)AlignUp(uint64_t(layout.uvRowPitchBytes) * (VIDEO_HEIGHT / 2), sliceAlignment);
    layout.totalSizeBytes = layout.uvOffsetBytes + layout.uvSlicePitchBytes;
    return layout;
}

template <typename Record>
static bool SubmitOneTime(nri::CoreInterface& core, nri::Queue& queue, Record&& record) {
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    bool ok = core.CreateCommandAllocator(queue, allocator) == nri::Result::SUCCESS && allocator && core.CreateCommandBuffer(*allocator, commandBuffer) == nri::Result::SUCCESS && commandBuffer && core.BeginCommandBuffer(*commandBuffer, nullptr) == nri::Result::SUCCESS;
    if (ok) {
        std::forward<Record>(record)(*commandBuffer);
        ok = core.EndCommandBuffer(*commandBuffer) == nri::Result::SUCCESS;
    }
    if (ok) {
        const nri::CommandBuffer* commandBuffers[] = {commandBuffer};
        nri::QueueSubmitDesc submit = {};
        submit.commandBuffers = commandBuffers;
        submit.commandBufferNum = 1;
        ok = core.QueueSubmit(queue, submit) == nri::Result::SUCCESS && core.QueueWaitIdle(&queue) == nri::Result::SUCCESS;
    }
    if (commandBuffer)
        core.DestroyCommandBuffer(commandBuffer);
    if (allocator)
        core.DestroyCommandAllocator(allocator);
    return ok;
}

template <typename Record>
static bool SubmitOneTime(
    nri::CoreInterface& core, nri::Queue& queue, nri::DescriptorPool* descriptorPool, Record&& record) {
    nri::CommandAllocator* allocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    bool ok = core.CreateCommandAllocator(queue, allocator) == nri::Result::SUCCESS && allocator && core.CreateCommandBuffer(*allocator, commandBuffer) == nri::Result::SUCCESS && commandBuffer && core.BeginCommandBuffer(*commandBuffer, descriptorPool) == nri::Result::SUCCESS;
    if (ok) {
        std::forward<decltype(record)>(record)(*commandBuffer);
        ok = core.EndCommandBuffer(*commandBuffer) == nri::Result::SUCCESS;
    }
    if (ok) {
        const nri::CommandBuffer* commandBuffers[] = {commandBuffer};
        nri::QueueSubmitDesc submit = {};
        submit.commandBuffers = commandBuffers;
        submit.commandBufferNum = 1;
        ok = core.QueueSubmit(queue, submit) == nri::Result::SUCCESS && core.QueueWaitIdle(&queue) == nri::Result::SUCCESS;
    }
    if (commandBuffer)
        core.DestroyCommandBuffer(commandBuffer);
    if (allocator)
        core.DestroyCommandAllocator(allocator);
    return ok;
}

static bool CopyNv12BufferToTexture(nri::CoreInterface& core, nri::Queue& queue, const Nv12BufferLayout& layout, nri::Buffer& src, nri::Texture& dst) {
    return SubmitOneTime(core, queue, [&](nri::CommandBuffer& commandBuffer) {
        nri::BufferBarrierDesc bufferBarrier = {};
        bufferBarrier.buffer = &src;
        bufferBarrier.before = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        bufferBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};

        nri::TextureBarrierDesc textureBarrier = {};
        textureBarrier.texture = &dst;
        textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
        textureBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::Layout::COPY_DESTINATION, nri::StageBits::COPY};
        textureBarrier.mipNum = nri::REMAINING;
        textureBarrier.layerNum = nri::REMAINING;
        textureBarrier.planes = nri::PlaneBits::ALL;

        nri::BarrierDesc barrierDesc = {};
        barrierDesc.buffers = &bufferBarrier;
        barrierDesc.bufferNum = 1;
        barrierDesc.textures = &textureBarrier;
        barrierDesc.textureNum = 1;
        core.CmdBarrier(commandBuffer, barrierDesc);

        nri::TextureRegionDesc lumaRegion = {};
        lumaRegion.width = VIDEO_WIDTH;
        lumaRegion.height = VIDEO_HEIGHT;
        lumaRegion.depth = 1;
        lumaRegion.planes = nri::PlaneBits::PLANE_0;

        nri::TextureDataLayoutDesc lumaLayout = {};
        lumaLayout.rowPitch = layout.yRowPitchBytes;
        lumaLayout.slicePitch = layout.ySlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, dst, lumaRegion, src, lumaLayout);

        nri::TextureRegionDesc chromaRegion = {};
        chromaRegion.width = VIDEO_WIDTH;
        chromaRegion.height = VIDEO_HEIGHT;
        chromaRegion.depth = 1;
        chromaRegion.planes = nri::PlaneBits::PLANE_1;

        nri::TextureDataLayoutDesc chromaLayout = {};
        chromaLayout.offset = layout.uvOffsetBytes;
        chromaLayout.rowPitch = layout.uvRowPitchBytes;
        chromaLayout.slicePitch = layout.uvSlicePitchBytes;
        core.CmdUploadBufferToTexture(commandBuffer, dst, chromaRegion, src, chromaLayout);

        textureBarrier.before = textureBarrier.after;
        textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
        bufferBarrier.before = bufferBarrier.after;
        bufferBarrier.after = {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};
        core.CmdBarrier(commandBuffer, barrierDesc);
    });
}

static nri::Result CreateEncodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_READBACK, priority, bufferDesc, buffer);
}

static nri::Result CreateDecodeBitstreamBuffer(nri::CoreInterface& core, nri::Device& device, float priority, const nri::BufferDesc& bufferDesc, nri::Buffer*& buffer) {
    return core.CreateCommittedBuffer(device, nri::MemoryLocation::HOST_UPLOAD, priority, bufferDesc, buffer);
}

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

private:
    bool InitializeGraphics(nri::GraphicsAPI graphicsAPI);
    bool TryInitializePreviewTextures(nri::GraphicsAPI graphicsAPI);
    void InitializeGeneratedFrames(float timeSec);
    bool CanRunRoundTrip() const;
    void TryInitializeVideo(nri::GraphicsAPI graphicsAPI);
    PatternConstants MakePatternConstants(PatternOperation operation, float timeSec) const;
    bool GeneratePatternWithCompute(const PatternConstants& constants, nri::Descriptor* previewTexture, bool returnSourceBufferToShaderStorage = false);
    bool WriteAnnexBHeadersToUploadBuffer(std::vector<uint8_t>& annexBHeaders);
    bool TrySubmitEncodeAndMetadataReadback(float timeSec);
    bool TryDecodePendingMetadata(float timeSec);
    bool DecodeEncodedBitstream(const nri::VideoEncodeFeedback& feedback, const nri::VideoAV1EncodeDecodeInfo* av1DecodeInfo, float timeSec);
    bool TryRunRoundTrip(float timeSec);
    void DrawTexturePanel(const char* label, nri::Descriptor* texture, const ImVec2& size);

private:
    NRIInterface NRI = {};
    nri::VideoInterface Video = {};
    nri::GraphicsAPI m_GraphicsAPI = nri::GraphicsAPI::NONE;

    nri::Device* m_Device = nullptr;
    nri::Streamer* m_Streamer = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::Queue* m_GraphicsQueue = nullptr;
    nri::Queue* m_VideoEncodeQueue = nullptr;
    nri::Queue* m_VideoDecodeQueue = nullptr;
    nri::Fence* m_FrameFence = nullptr;

    nri::VideoSession* m_EncodeSession = nullptr;
    nri::VideoSession* m_DecodeSession = nullptr;
    nri::VideoSessionParameters* m_EncodeParameters = nullptr;
    nri::VideoSessionParameters* m_DecodeParameters = nullptr;
    nri::Texture* m_EncodeTexture = nullptr;
    nri::Texture* m_ReconstructedTexture = nullptr;
    nri::Texture* m_DecodeTexture = nullptr;
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
    nri::Buffer* m_BitstreamHeaderUploadBuffer = nullptr;
    nri::Buffer* m_BitstreamHeaderReadbackBuffer = nullptr;
    nri::Buffer* m_BitstreamBuffer = nullptr;
    nri::Buffer* m_DecodeBitstreamBuffer = nullptr;
    nri::Buffer* m_MetadataBuffer = nullptr;
    nri::Buffer* m_ResolvedMetadataBuffer = nullptr;
    nri::Buffer* m_ResolvedMetadataReadbackBuffer = nullptr;
    nri::VideoPicture* m_EncodePicture = nullptr;
    nri::VideoPicture* m_ReconstructedPicture = nullptr;
    nri::VideoPicture* m_DecodePicture = nullptr;
    nri::CommandAllocator* m_MetadataReadbackCommandAllocator = nullptr;
    nri::CommandBuffer* m_MetadataReadbackCommandBuffer = nullptr;
    nri::Fence* m_MetadataReadbackFence = nullptr;

    std::vector<QueuedFrame> m_QueuedFrames;
    std::vector<SwapChainTexture> m_SwapChainTextures;

    Nv12BufferLayout m_Nv12Layout = {};
    nri::Format m_SwapChainFormat = nri::Format::UNKNOWN;
    std::string m_VideoStatus = "Initializing video";
    std::string m_PreviewStatus = "Initializing preview";
    std::string m_CodecArg = "H264";
    SampleCodec m_Codec = SampleCodec::H264;
    nri::VideoH264SequenceParameterSetDesc m_H264Sps = {};
    nri::VideoH264PictureParameterSetDesc m_H264Pps = {};
    nri::VideoH265VideoParameterSetDesc m_H265Vps = {};
    nri::VideoH265SequenceParameterSetDesc m_H265Sps = {};
    nri::VideoH265PictureParameterSetDesc m_H265Pps = {};
    nri::VideoAV1SequenceDesc m_AV1Sequence = {};
    double m_StartTimeSec = 0.0;
    double m_LastRoundTripTimeSec = -1.0;
    bool m_VideoReady = false;
    bool m_DecodePreviewReady = false;
    bool m_PreviewTexturesShaderReadable = false;
    bool m_MetadataReadbackPending = false;
    uint64_t m_MetadataReadbackFenceValue = 0;
};

Sample::~Sample() {
    if (NRI.HasCore()) {
        NRI.DeviceWaitIdle(m_Device);

        if (Video.DestroyVideoPicture) {
            if (m_DecodePicture)
                Video.DestroyVideoPicture(*m_DecodePicture);
            if (m_ReconstructedPicture)
                Video.DestroyVideoPicture(*m_ReconstructedPicture);
            if (m_EncodePicture)
                Video.DestroyVideoPicture(*m_EncodePicture);
            if (m_DecodeParameters)
                Video.DestroyVideoSessionParameters(*m_DecodeParameters);
            if (m_EncodeParameters)
                Video.DestroyVideoSessionParameters(*m_EncodeParameters);
            if (m_DecodeSession)
                Video.DestroyVideoSession(*m_DecodeSession);
            if (m_EncodeSession)
                Video.DestroyVideoSession(*m_EncodeSession);
        }

        if (m_MetadataReadbackCommandBuffer)
            NRI.DestroyCommandBuffer(m_MetadataReadbackCommandBuffer);
        if (m_MetadataReadbackCommandAllocator)
            NRI.DestroyCommandAllocator(m_MetadataReadbackCommandAllocator);
        if (m_MetadataReadbackFence)
            NRI.DestroyFence(m_MetadataReadbackFence);
        if (m_ResolvedMetadataReadbackBuffer)
            NRI.DestroyBuffer(m_ResolvedMetadataReadbackBuffer);
        if (m_ResolvedMetadataBuffer)
            NRI.DestroyBuffer(m_ResolvedMetadataBuffer);
        if (m_GenerateDescriptorPool)
            NRI.DestroyDescriptorPool(m_GenerateDescriptorPool);
        if (m_GenerateComputePipeline)
            NRI.DestroyPipeline(m_GenerateComputePipeline);
        if (m_GeneratePipelineLayout)
            NRI.DestroyPipelineLayout(m_GeneratePipelineLayout);
        if (m_UploadBufferView)
            NRI.DestroyDescriptor(m_UploadBufferView);
        if (m_MetadataBuffer)
            NRI.DestroyBuffer(m_MetadataBuffer);
        if (m_DecodeBitstreamBuffer)
            NRI.DestroyBuffer(m_DecodeBitstreamBuffer);
        if (m_BitstreamBuffer)
            NRI.DestroyBuffer(m_BitstreamBuffer);
        if (m_BitstreamHeaderReadbackBuffer)
            NRI.DestroyBuffer(m_BitstreamHeaderReadbackBuffer);
        if (m_BitstreamHeaderUploadBuffer)
            NRI.DestroyBuffer(m_BitstreamHeaderUploadBuffer);
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
        if (m_DecodeTexture)
            NRI.DestroyTexture(m_DecodeTexture);
        if (m_ReconstructedTexture)
            NRI.DestroyTexture(m_ReconstructedTexture);
        if (m_EncodeTexture)
            NRI.DestroyTexture(m_EncodeTexture);

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

    nri::nriDestroyDevice(m_Device);
}

void Sample::InitCmdLine(cmdline::parser& cmdLine) {
    cmdLine.add<std::string>("codec", 0, "video codec: H264, H265, or AV1", false, m_CodecArg, cmdline::oneof<std::string>("H264", "H265", "AV1"));
}

void Sample::ReadCmdLine(cmdline::parser& cmdLine) {
    m_CodecArg = cmdLine.get<std::string>("codec");
    m_Codec = m_CodecArg == "H265" ? SampleCodec::H265 : (m_CodecArg == "AV1" ? SampleCodec::AV1 : SampleCodec::H264);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI, bool) {
    m_GraphicsAPI = graphicsAPI;
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

    nri::DeviceCreationDesc deviceCreationDesc = {};
    nri::QueueFamilyDesc queueFamilies[] = {
        {nullptr, 1, nri::QueueType::GRAPHICS},
        {nullptr, 1, nri::QueueType::VIDEO_ENCODE},
        {nullptr, 1, nri::QueueType::VIDEO_DECODE},
    };

    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableGraphicsAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.enableD3D11CommandBufferEmulation = D3D11_ENABLE_COMMAND_BUFFER_EMULATION;
    deviceCreationDesc.disableD3D12EnhancedBarriers = D3D12_DISABLE_ENHANCED_BARRIERS;
    deviceCreationDesc.vkBindingOffsets = VK_BINDING_OFFSETS;
    deviceCreationDesc.adapterDesc = &adapterDesc[std::min(m_AdapterIndex, adapterDescsNum - 1)];
    deviceCreationDesc.allocationCallbacks = m_AllocationCallbacks;
    deviceCreationDesc.queueFamilies = queueFamilies;
    deviceCreationDesc.queueFamilyNum = helper::GetCountOf(queueFamilies);
    NRI_ABORT_ON_FAILURE(nri::nriCreateDevice(deviceCreationDesc, m_Device));

    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::StreamerInterface), (nri::StreamerInterface*)&NRI));
    NRI_ABORT_ON_FAILURE(nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI));

    m_Nv12Layout = MakeNv12BufferLayout(NRI.GetDeviceDesc(*m_Device));

    nri::StreamerDesc streamerDesc = {};
    streamerDesc.dynamicBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.dynamicBufferDesc = {0, 0, nri::BufferUsageBits::VERTEX_BUFFER | nri::BufferUsageBits::INDEX_BUFFER};
    streamerDesc.constantBufferMemoryLocation = nri::MemoryLocation::HOST_UPLOAD;
    streamerDesc.queuedFrameNum = GetQueuedFrameNum();
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

    m_QueuedFrames.resize(GetQueuedFrameNum());
    for (QueuedFrame& queuedFrame : m_QueuedFrames) {
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandAllocator(*m_GraphicsQueue, queuedFrame.commandAllocator));
        NRI_ABORT_ON_FAILURE(NRI.CreateCommandBuffer(*queuedFrame.commandAllocator, queuedFrame.commandBuffer));
    }

    return true;
}

PatternConstants Sample::MakePatternConstants(PatternOperation operation, float timeSec) const {
    PatternConstants patternConstants = {};
    patternConstants.width = VIDEO_WIDTH;
    patternConstants.height = VIDEO_HEIGHT;
    patternConstants.yRowPitchBytes = m_Nv12Layout.yRowPitchBytes;
    patternConstants.uvRowPitchBytes = m_Nv12Layout.uvRowPitchBytes;
    patternConstants.uvOffsetBytes = (uint32_t)m_Nv12Layout.uvOffsetBytes;
    patternConstants.operation = operation;
    patternConstants.time = timeSec;
    return patternConstants;
}

void Sample::InitializeGeneratedFrames(float timeSec) {
    if (!m_SourcePreviewTexture || !m_SourcePreviewStorage || !m_UploadBuffer || !m_UploadBufferView)
        return;

    PatternConstants patternConstants = MakePatternConstants(OP_GENERATE_PATTERN, timeSec);

    if (m_EncodeTexture) {
        if (!GeneratePatternWithCompute(patternConstants, m_SourcePreviewStorage, true)) {
            m_PreviewStatus = "Failed to generate source pattern via compute";
            return;
        }

        if (!CopyNv12BufferToTexture(NRI, *m_GraphicsQueue, m_Nv12Layout, *m_UploadBuffer, *m_EncodeTexture)) {
            m_PreviewStatus = "Failed to upload NV12 source to video texture";
            return;
        }
        m_PreviewStatus = "Source preview is generated by compute";
        return;
    }

    if (!GeneratePatternWithCompute(patternConstants, m_SourcePreviewStorage, true)) {
        m_PreviewStatus = "Failed to generate source pattern via compute";
        return;
    }

    m_PreviewStatus = "Source preview is generated by compute";
}

bool Sample::GeneratePatternWithCompute(const PatternConstants& constants, nri::Descriptor* previewTexture, bool returnSourceBufferToShaderStorage) {
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
        [this, &constants, dispatchX, dispatchY, previewTexturesShaderReadable, returnSourceBufferToShaderStorage](nri::CommandBuffer& commandBuffer) {
            nri::SetDescriptorSetDesc descriptorSet = {0, m_GenerateDescriptorSet};

            nri::BufferBarrierDesc bufferBarrier = {};
            bufferBarrier.buffer = m_UploadBuffer;
            bufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
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
    previewTextureDesc.width = VIDEO_WIDTH;
    previewTextureDesc.height = VIDEO_HEIGHT;
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

    if (!deviceDesc.adapterDesc.queueNum[(uint32_t)nri::QueueType::VIDEO_ENCODE] || !deviceDesc.adapterDesc.queueNum[(uint32_t)nri::QueueType::VIDEO_DECODE]) {
        m_VideoStatus = "Adapter has no NRI video encode/decode queues";
        return;
    }

    if (nri::nriGetInterface(*m_Device, NRI_INTERFACE(nri::VideoInterface), &Video) != nri::Result::SUCCESS || !Video.CreateVideoSession) {
        m_VideoStatus = "NRI video interface is unavailable";
        return;
    }

    nri::VideoSessionDesc encodeSessionDesc = {};
    encodeSessionDesc.usage = nri::VideoUsage::ENCODE;
    encodeSessionDesc.codec = GetNriCodec(m_Codec);
    encodeSessionDesc.format = nri::Format::NV12_UNORM;
    encodeSessionDesc.width = VIDEO_WIDTH;
    encodeSessionDesc.height = VIDEO_HEIGHT;
    encodeSessionDesc.maxReferenceNum = 1;

    nri::VideoSessionDesc decodeSessionDesc = encodeSessionDesc;
    decodeSessionDesc.usage = nri::VideoUsage::DECODE;
    decodeSessionDesc.maxReferenceNum = 16;

    if (NRI.GetQueue(*m_Device, nri::QueueType::VIDEO_ENCODE, 0, m_VideoEncodeQueue) != nri::Result::SUCCESS || NRI.GetQueue(*m_Device, nri::QueueType::VIDEO_DECODE, 0, m_VideoDecodeQueue) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to get video queues";
        return;
    }

    if (Video.CreateVideoSession(*m_Device, encodeSessionDesc, m_EncodeSession) != nri::Result::SUCCESS) {
        m_VideoStatus = std::string("Failed to create ") + GetCodecName(m_Codec) + " encode session";
        return;
    }

    if (Video.CreateVideoSession(*m_Device, decodeSessionDesc, m_DecodeSession) != nri::Result::SUCCESS) {
        m_VideoStatus = std::string("Failed to create ") + GetCodecName(m_Codec) + " decode session";
        return;
    }

    nri::VideoH264SequenceParameterSetDesc sps = {};
    sps.flags = nri::VideoH264SequenceParameterSetBits::DIRECT_8X8_INFERENCE | nri::VideoH264SequenceParameterSetBits::FRAME_MBS_ONLY;
    sps.profileIdc = 100;
    sps.levelIdc = 42;
    sps.chromaFormatIdc = 1;
    sps.sequenceParameterSetId = 0;
    sps.log2MaxFrameNumMinus4 = 0;
    sps.pictureOrderCountType = 0;
    sps.log2MaxPictureOrderCountLsbMinus4 = 0;
    sps.referenceFrameNum = 1;
    sps.pictureWidthInMbsMinus1 = VIDEO_WIDTH / 16 - 1;
    sps.pictureHeightInMapUnitsMinus1 = VIDEO_HEIGHT / 16 - 1;

    nri::VideoH264PictureParameterSetDesc pps = {};
    pps.flags = nri::VideoH264PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT;
    pps.sequenceParameterSetId = 0;
    pps.pictureParameterSetId = 0;
    pps.refIndexL0DefaultActiveMinus1 = 0;
    pps.refIndexL1DefaultActiveMinus1 = 0;
    m_H264Sps = sps;
    m_H264Pps = pps;

    nri::VideoH264SessionParametersDesc h264Parameters = {};
    h264Parameters.sequenceParameterSets = &sps;
    h264Parameters.sequenceParameterSetNum = 1;
    h264Parameters.pictureParameterSets = &pps;
    h264Parameters.pictureParameterSetNum = 1;
    h264Parameters.maxSequenceParameterSetNum = 1;
    h264Parameters.maxPictureParameterSetNum = 1;

    nri::VideoH265VideoParameterSetDesc vps = {};
    vps.flags = nri::VideoH265VideoParameterSetBits::TEMPORAL_ID_NESTING;
    vps.videoParameterSetId = 0;
    vps.maxSubLayersMinus1 = 0;
    vps.profileTierLevel.flags = nri::VideoH265ProfileTierLevelBits::FRAME_ONLY_CONSTRAINT;
    vps.profileTierLevel.generalProfileIdc = 1;
    vps.profileTierLevel.generalLevelIdc = 90;
    vps.decPicBufMgr.maxDecPicBufferingMinus1[0] = 2;
    vps.decPicBufMgr.maxNumReorderPics[0] = 1;

    nri::VideoH265SequenceParameterSetDesc h265Sps = {};
    h265Sps.flags = nri::VideoH265SequenceParameterSetBits::TEMPORAL_ID_NESTING | nri::VideoH265SequenceParameterSetBits::AMP_ENABLED | nri::VideoH265SequenceParameterSetBits::SAMPLE_ADAPTIVE_OFFSET_ENABLED;
    h265Sps.videoParameterSetId = vps.videoParameterSetId;
    h265Sps.maxSubLayersMinus1 = vps.maxSubLayersMinus1;
    h265Sps.sequenceParameterSetId = 0;
    h265Sps.chromaFormatIdc = 1;
    h265Sps.pictureWidthInLumaSamples = VIDEO_WIDTH;
    h265Sps.pictureHeightInLumaSamples = VIDEO_HEIGHT;
    h265Sps.log2MaxPictureOrderCountLsbMinus4 = 3;
    h265Sps.log2MinLumaCodingBlockSizeMinus3 = 0;
    h265Sps.log2DiffMaxMinLumaCodingBlockSize = 2;
    h265Sps.log2MinLumaTransformBlockSizeMinus2 = 0;
    h265Sps.log2DiffMaxMinLumaTransformBlockSize = 3;
    h265Sps.maxTransformHierarchyDepthInter = 3;
    h265Sps.maxTransformHierarchyDepthIntra = 3;
    h265Sps.profileTierLevel = vps.profileTierLevel;
    h265Sps.decPicBufMgr = vps.decPicBufMgr;

    nri::VideoH265PictureParameterSetDesc h265Pps = {};
    h265Pps.flags = nri::VideoH265PictureParameterSetBits::CABAC_INIT_PRESENT | nri::VideoH265PictureParameterSetBits::TRANSFORM_SKIP_ENABLED | nri::VideoH265PictureParameterSetBits::CU_QP_DELTA_ENABLED | nri::VideoH265PictureParameterSetBits::SLICE_CHROMA_QP_OFFSETS_PRESENT | nri::VideoH265PictureParameterSetBits::DEBLOCKING_FILTER_CONTROL_PRESENT;
    h265Pps.pictureParameterSetId = 0;
    h265Pps.sequenceParameterSetId = h265Sps.sequenceParameterSetId;
    h265Pps.videoParameterSetId = vps.videoParameterSetId;
    m_H265Vps = vps;
    m_H265Sps = h265Sps;
    m_H265Pps = h265Pps;

    nri::VideoH265SessionParametersDesc h265Parameters = {};
    h265Parameters.videoParameterSets = &vps;
    h265Parameters.videoParameterSetNum = 1;
    h265Parameters.sequenceParameterSets = &h265Sps;
    h265Parameters.sequenceParameterSetNum = 1;
    h265Parameters.pictureParameterSets = &h265Pps;
    h265Parameters.pictureParameterSetNum = 1;
    h265Parameters.maxVideoParameterSetNum = 1;
    h265Parameters.maxSequenceParameterSetNum = 1;
    h265Parameters.maxPictureParameterSetNum = 1;

    m_AV1Sequence = MakeAV1SequenceDesc();
    nri::VideoAV1SessionParametersDesc av1Parameters = {};
    av1Parameters.sequence = m_AV1Sequence;

    nri::VideoSessionParametersDesc encodeParametersDesc = {};
    encodeParametersDesc.session = m_EncodeSession;
    encodeParametersDesc.h264Parameters = m_Codec == SampleCodec::H264 ? &h264Parameters : nullptr;
    encodeParametersDesc.h265Parameters = m_Codec == SampleCodec::H265 ? &h265Parameters : nullptr;
    encodeParametersDesc.av1Parameters = m_Codec == SampleCodec::AV1 ? &av1Parameters : nullptr;

    nri::VideoSessionParametersDesc decodeParametersDesc = {};
    decodeParametersDesc.session = m_DecodeSession;
    decodeParametersDesc.h264Parameters = m_Codec == SampleCodec::H264 ? &h264Parameters : nullptr;
    decodeParametersDesc.h265Parameters = m_Codec == SampleCodec::H265 ? &h265Parameters : nullptr;
    decodeParametersDesc.av1Parameters = m_Codec == SampleCodec::AV1 ? &av1Parameters : nullptr;

    if (Video.CreateVideoSessionParameters(*m_Device, encodeParametersDesc, m_EncodeParameters) != nri::Result::SUCCESS) {
        m_VideoStatus = std::string("Failed to create ") + GetCodecName(m_Codec) + " encode parameters";
        return;
    }

    if (Video.CreateVideoSessionParameters(*m_Device, decodeParametersDesc, m_DecodeParameters) != nri::Result::SUCCESS) {
        m_VideoStatus = std::string("Failed to create ") + GetCodecName(m_Codec) + " decode parameters";
        return;
    }

    nri::TextureDesc encodeTextureDesc = {};
    encodeTextureDesc.type = nri::TextureType::TEXTURE_2D;
    encodeTextureDesc.usage = nri::TextureUsageBits::VIDEO_ENCODE;
    encodeTextureDesc.format = nri::Format::NV12_UNORM;
    encodeTextureDesc.width = VIDEO_WIDTH;
    encodeTextureDesc.height = VIDEO_HEIGHT;
    encodeTextureDesc.mipNum = 1;
    encodeTextureDesc.layerNum = 1;
    encodeTextureDesc.videoCodec = GetNriCodec(m_Codec);

    nri::TextureDesc decodeTextureDesc = encodeTextureDesc;
    decodeTextureDesc.usage = nri::TextureUsageBits::VIDEO_DECODE;

    if (NRI.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, encodeTextureDesc, m_EncodeTexture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create NV12 encode texture";
        return;
    }
    NRI.SetDebugName(m_EncodeTexture, "VideoEncodeTexture");

    if (NRI.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, encodeTextureDesc, m_ReconstructedTexture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create NV12 reconstructed texture";
        return;
    }
    NRI.SetDebugName(m_ReconstructedTexture, "VideoReconstructedTexture");

    if (NRI.CreateCommittedTexture(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, decodeTextureDesc, m_DecodeTexture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create NV12 decode texture";
        return;
    }
    NRI.SetDebugName(m_DecodeTexture, "VideoDecodeTexture");

    if (!SubmitOneTime(NRI, *m_GraphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarriers[3] = {};
            textureBarriers[0].texture = m_EncodeTexture;
            textureBarriers[1].texture = m_ReconstructedTexture;
            textureBarriers[2].texture = m_DecodeTexture;

            for (nri::TextureBarrierDesc& textureBarrier : textureBarriers) {
                textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::UNDEFINED, nri::StageBits::ALL};
                textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
                textureBarrier.mipNum = nri::REMAINING;
                textureBarrier.layerNum = nri::REMAINING;
                textureBarrier.planes = nri::PlaneBits::ALL;
            }

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_VideoStatus = "Failed to initialize video texture layouts";
        return;
    }

    if (!m_GenerateComputePipeline) {
        nri::TextureViewDesc sourceStorageTextureViewDesc = {m_SourcePreviewTexture, nri::TextureView::STORAGE_TEXTURE, nri::Format::RGBA8_UNORM};
        nri::TextureViewDesc decodeStorageTextureViewDesc = {m_DecodePreviewTexture, nri::TextureView::STORAGE_TEXTURE, nri::Format::RGBA8_UNORM};

        if (!m_SourcePreviewStorage) {
            if (NRI.CreateTextureView(sourceStorageTextureViewDesc, m_SourcePreviewStorage) != nri::Result::SUCCESS) {
                m_VideoStatus = "Failed to create source preview storage texture view";
                return;
            }
        }

        if (!m_DecodePreviewStorage) {
            if (NRI.CreateTextureView(decodeStorageTextureViewDesc, m_DecodePreviewStorage) != nri::Result::SUCCESS) {
                m_VideoStatus = "Failed to create decode preview storage texture view";
                return;
            }
        }

        nri::BufferDesc uploadBufferDesc = {};
        uploadBufferDesc.size = m_Nv12Layout.totalSizeBytes;
        uploadBufferDesc.usage = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE;

        if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, uploadBufferDesc, m_UploadBuffer) != nri::Result::SUCCESS) {
            m_VideoStatus = "Failed to create NV12 upload buffer";
            return;
        }

        nri::BufferViewDesc uploadBufferViewDesc = {};
        uploadBufferViewDesc.buffer = m_UploadBuffer;
        uploadBufferViewDesc.format = nri::Format::R32_UINT;
        uploadBufferViewDesc.type = nri::BufferView::STORAGE_BUFFER;
        uploadBufferViewDesc.size = uploadBufferDesc.size;

        if (NRI.CreateBufferView(uploadBufferViewDesc, m_UploadBufferView) != nri::Result::SUCCESS) {
            m_VideoStatus = "Failed to create NV12 compute output buffer view";
            return;
        }

        utils::ShaderCodeStorage shaderCodeStorage;
        {
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
                m_VideoStatus = "Failed to create compute pipeline layout for pattern generation";
                return;
            }

            nri::ComputePipelineDesc computePipelineDesc = {};
            computePipelineDesc.pipelineLayout = m_GeneratePipelineLayout;
            computePipelineDesc.shader = utils::LoadShader(deviceDesc.graphicsAPI, "VideoEncodePattern.cs", shaderCodeStorage);
            if (NRI.CreateComputePipeline(*m_Device, computePipelineDesc, m_GenerateComputePipeline) != nri::Result::SUCCESS) {
                m_VideoStatus = "Failed to create pattern generation compute pipeline";
                return;
            }

            nri::DescriptorPoolDesc descriptorPoolDesc = {};
            descriptorPoolDesc.descriptorSetMaxNum = 1;
            descriptorPoolDesc.storageBufferMaxNum = 1;
            descriptorPoolDesc.storageTextureMaxNum = 2;
            descriptorPoolDesc.textureMaxNum = 2;
            if (NRI.CreateDescriptorPool(*m_Device, descriptorPoolDesc, m_GenerateDescriptorPool) != nri::Result::SUCCESS) {
                m_VideoStatus = "Failed to create compute descriptor pool for pattern generation";
                return;
            }

            if (NRI.AllocateDescriptorSets(*m_GenerateDescriptorPool, *m_GeneratePipelineLayout, 0, &m_GenerateDescriptorSet, 1, 0) != nri::Result::SUCCESS) {
                m_VideoStatus = "Failed to allocate compute descriptor set for pattern generation";
                return;
            }

            nri::UpdateDescriptorRangeDesc updateDescriptorRangeDescs[] = {
                {m_GenerateDescriptorSet, 0, 0, &m_UploadBufferView, 1},
                {m_GenerateDescriptorSet, 1, 0, &m_SourcePreviewStorage, 1},
                {m_GenerateDescriptorSet, 2, 0, &m_DecodePreviewStorage, 1},
            };
            NRI.UpdateDescriptorRanges(updateDescriptorRangeDescs, helper::GetCountOf(updateDescriptorRangeDescs));
        }
    }

    nri::BufferDesc bitstreamHeaderUploadBufferDesc = {};
    bitstreamHeaderUploadBufferDesc.size = ENCODED_SLICE_OFFSET;

    nri::BufferDesc bitstreamBufferDesc = {};
    bitstreamBufferDesc.size = BITSTREAM_SIZE;
    bitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc decodeBitstreamBufferDesc = {};
    decodeBitstreamBufferDesc.size = BITSTREAM_SIZE;
    decodeBitstreamBufferDesc.usage = nri::BufferUsageBits::VIDEO_DECODE;

    nri::BufferDesc metadataBufferDesc = {};
    metadataBufferDesc.size = METADATA_SIZE;
    metadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc resolvedMetadataBufferDesc = {};
    resolvedMetadataBufferDesc.size = RESOLVED_METADATA_SIZE;
    resolvedMetadataBufferDesc.usage = nri::BufferUsageBits::VIDEO_ENCODE;

    nri::BufferDesc resolvedMetadataReadbackBufferDesc = {};
    resolvedMetadataReadbackBufferDesc.size = RESOLVED_METADATA_SIZE;
    resolvedMetadataReadbackBufferDesc.usage = nri::BufferUsageBits::NONE;

    nri::BufferDesc bitstreamHeaderReadbackBufferDesc = {};
    bitstreamHeaderReadbackBufferDesc.size = AV1_HEADER_READBACK_SIZE;
    bitstreamHeaderReadbackBufferDesc.usage = nri::BufferUsageBits::NONE;

    if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::HOST_UPLOAD, 0.0f, bitstreamHeaderUploadBufferDesc, m_BitstreamHeaderUploadBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create bitstream header upload buffer";
        return;
    }

    if (CreateEncodeBitstreamBuffer(NRI, *m_Device, 0.0f, bitstreamBufferDesc, m_BitstreamBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create encode bitstream buffer";
        return;
    }

    if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::HOST_READBACK, 0.0f, bitstreamHeaderReadbackBufferDesc, m_BitstreamHeaderReadbackBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create bitstream header readback buffer";
        return;
    }

    if (CreateDecodeBitstreamBuffer(NRI, *m_Device, 0.0f, decodeBitstreamBufferDesc, m_DecodeBitstreamBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create decode bitstream buffer";
        return;
    }

    if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, metadataBufferDesc, m_MetadataBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create encode metadata buffer";
        return;
    }

    if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::DEVICE, 0.0f, resolvedMetadataBufferDesc, m_ResolvedMetadataBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create resolved encode metadata buffer";
        return;
    }

    if (NRI.CreateCommittedBuffer(*m_Device, nri::MemoryLocation::HOST_READBACK, 0.0f, resolvedMetadataReadbackBufferDesc, m_ResolvedMetadataReadbackBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create resolved encode metadata readback buffer";
        return;
    }

    if (NRI.CreateFence(*m_Device, 0, m_MetadataReadbackFence) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create metadata readback fence";
        return;
    }

    if (NRI.CreateCommandAllocator(*m_GraphicsQueue, m_MetadataReadbackCommandAllocator) != nri::Result::SUCCESS || NRI.CreateCommandBuffer(*m_MetadataReadbackCommandAllocator, m_MetadataReadbackCommandBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create metadata readback command buffer";
        return;
    }

    nri::VideoPictureDesc encodePictureDesc = {};
    encodePictureDesc.texture = m_EncodeTexture;
    encodePictureDesc.usage = nri::VideoPictureUsage::ENCODE_INPUT;
    encodePictureDesc.format = nri::Format::NV12_UNORM;
    encodePictureDesc.width = VIDEO_WIDTH;
    encodePictureDesc.height = VIDEO_HEIGHT;

    nri::VideoPictureDesc decodePictureDesc = {};
    decodePictureDesc.texture = m_DecodeTexture;
    decodePictureDesc.usage = nri::VideoPictureUsage::DECODE_OUTPUT;
    decodePictureDesc.format = nri::Format::NV12_UNORM;
    decodePictureDesc.width = VIDEO_WIDTH;
    decodePictureDesc.height = VIDEO_HEIGHT;

    nri::VideoPictureDesc reconstructedPictureDesc = encodePictureDesc;
    reconstructedPictureDesc.texture = m_ReconstructedTexture;
    reconstructedPictureDesc.usage = nri::VideoPictureUsage::ENCODE_REFERENCE;

    if (Video.CreateVideoPicture(*m_Device, encodePictureDesc, m_EncodePicture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create encode picture";
        return;
    }

    if (Video.CreateVideoPicture(*m_Device, reconstructedPictureDesc, m_ReconstructedPicture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create reconstructed picture";
        return;
    }

    if (Video.CreateVideoPicture(*m_Device, decodePictureDesc, m_DecodePicture) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to create decode picture";
        return;
    }

    m_VideoReady = true;
    m_VideoStatus = std::string("NRI video queues and ") + GetCodecName(m_Codec) + " encode/decode objects initialized";
}

bool Sample::WriteAnnexBHeadersToUploadBuffer(std::vector<uint8_t>& annexBHeaders) {
    if (m_Codec == SampleCodec::AV1) {
        void* headerPtr = NRI.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        if (!headerPtr) {
            m_VideoStatus = "Failed to map bitstream header upload buffer";
            return false;
        }
        std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
        NRI.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);
        annexBHeaders.clear();
        return true;
    }

    nri::VideoAnnexBParameterSetsDesc annexBDesc = {};
    annexBDesc.codec = GetNriCodec(m_Codec);
    annexBDesc.h264Sps = &m_H264Sps;
    annexBDesc.h264Pps = &m_H264Pps;
    annexBDesc.h265Vps = &m_H265Vps;
    annexBDesc.h265Sps = &m_H265Sps;
    annexBDesc.h265Pps = &m_H265Pps;

    if (Video.WriteVideoAnnexBParameterSets(annexBDesc) != nri::Result::SUCCESS || annexBDesc.writtenSize == 0 || annexBDesc.writtenSize >= ENCODED_SLICE_OFFSET) {
        m_VideoStatus = std::string("Failed to query ") + GetCodecName(m_Codec) + " Annex-B parameter-set size";
        return false;
    }

    annexBHeaders.resize((size_t)annexBDesc.writtenSize);
    annexBDesc.dst = annexBHeaders.data();
    annexBDesc.dstSize = annexBHeaders.size();
    if (Video.WriteVideoAnnexBParameterSets(annexBDesc) != nri::Result::SUCCESS) {
        m_VideoStatus = std::string("Failed to build ") + GetCodecName(m_Codec) + " Annex-B parameter sets";
        return false;
    }

    void* headerPtr = NRI.MapBuffer(*m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
    if (!headerPtr) {
        m_VideoStatus = "Failed to map bitstream header upload buffer";
        return false;
    }
    std::memset(headerPtr, 0, (size_t)ENCODED_SLICE_OFFSET);
    std::memcpy(headerPtr, annexBHeaders.data(), annexBHeaders.size());
    NRI.UnmapBuffer(*m_BitstreamHeaderUploadBuffer);
    return true;
}

bool Sample::TrySubmitEncodeAndMetadataReadback(float timeSec) {
    if (!CanRunRoundTrip()) {
        m_VideoStatus = std::string(GetCodecName(m_Codec)) + " round trip is not currently supported in this configuration";
        return false;
    }

    PatternConstants patternConstants = MakePatternConstants(OP_GENERATE_PATTERN, timeSec);
    if (!GeneratePatternWithCompute(patternConstants, m_SourcePreviewStorage, true)) {
        m_VideoStatus = "Failed to generate NV12 source pattern via compute";
        return false;
    }

    if (!CopyNv12BufferToTexture(NRI, *m_GraphicsQueue, m_Nv12Layout, *m_UploadBuffer, *m_EncodeTexture)) {
        m_VideoStatus = "Failed to upload NV12 source to video texture";
        return false;
    }

    std::vector<uint8_t> annexBHeaders;
    if (!WriteAnnexBHeadersToUploadBuffer(annexBHeaders))
        return false;

    if (!SubmitOneTime(NRI, *m_GraphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            NRI.CmdZeroBuffer(commandBuffer, *m_BitstreamBuffer, 0, BITSTREAM_SIZE);
            NRI.CmdCopyBuffer(commandBuffer, *m_BitstreamBuffer, 0, *m_BitstreamHeaderUploadBuffer, 0, ENCODED_SLICE_OFFSET);
        })) {
        m_VideoStatus = std::string("Failed to upload ") + GetCodecName(m_Codec) + " Annex-B parameter sets";
        return false;
    }

    nri::VideoEncodePictureDesc pictureDesc = {};
    pictureDesc.frameType = nri::VideoEncodeFrameType::IDR;
    pictureDesc.idrPictureId = 1;

    uint16_t av1MiColumnStarts[] = {0, (uint16_t)(2 * ((VIDEO_WIDTH + 7) >> 3))};
    uint16_t av1MiRowStarts[] = {0, (uint16_t)(2 * ((VIDEO_HEIGHT + 7) >> 3))};
    uint16_t av1WidthInSuperblocksMinus1[] = {(uint16_t)(((VIDEO_WIDTH + 63) / 64) - 1)};
    uint16_t av1HeightInSuperblocksMinus1[] = {(uint16_t)(((VIDEO_HEIGHT + 63) / 64) - 1)};
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
    av1PictureDesc.currentFrameId = 0;
    av1PictureDesc.refreshFrameFlags = 0xFF;
    av1PictureDesc.primaryReferenceName = nri::VideoAV1ReferenceName::NONE;
    av1PictureDesc.flags = nri::VideoAV1PictureBits::ERROR_RESILIENT_MODE | nri::VideoAV1PictureBits::DISABLE_CDF_UPDATE | nri::VideoAV1PictureBits::ALLOW_SCREEN_CONTENT_TOOLS | nri::VideoAV1PictureBits::FORCE_INTEGER_MV | nri::VideoAV1PictureBits::SHOW_FRAME | nri::VideoAV1PictureBits::SHOWABLE_FRAME;
    av1PictureDesc.renderWidthMinus1 = VIDEO_WIDTH - 1;
    av1PictureDesc.renderHeightMinus1 = VIDEO_HEIGHT - 1;
    av1PictureDesc.baseQIndex = 20;
    av1PictureDesc.interpolationFilter = 0;
    av1PictureDesc.txMode = 2;
    av1PictureDesc.cdefDampingMinus3 = 3;
    av1PictureDesc.tileLayout = &av1TileLayout;
    av1PictureDesc.loopFilter = &av1LoopFilter;
    av1PictureDesc.cdef = &av1Cdef;
    av1PictureDesc.loopRestoration = &av1LoopRestoration;
    av1PictureDesc.globalMotion = &av1GlobalMotion;

    nri::VideoEncodeRateControlDesc rateControlDesc = {};
    rateControlDesc.mode = nri::VideoEncodeRateControlMode::CQP;
    rateControlDesc.qpI = 20;
    rateControlDesc.qpP = 22;
    rateControlDesc.qpB = 24;
    rateControlDesc.frameRateNumerator = 30;
    rateControlDesc.frameRateDenominator = 1;

    nri::VideoEncodeDesc encodeDesc = {};
    encodeDesc.session = m_EncodeSession;
    encodeDesc.parameters = m_EncodeParameters;
    encodeDesc.srcPicture = m_EncodePicture;
    encodeDesc.dstBitstream.buffer = m_BitstreamBuffer;
    encodeDesc.dstBitstream.offset = ENCODED_SLICE_OFFSET;
    encodeDesc.dstBitstream.size = BITSTREAM_SIZE - ENCODED_SLICE_OFFSET;
    encodeDesc.bitstreamMetadataSize = ENCODED_SLICE_OFFSET;
    encodeDesc.pictureDesc = &pictureDesc;
    encodeDesc.rateControlDesc = &rateControlDesc;
    encodeDesc.reconstructedPicture = m_ReconstructedPicture;
    encodeDesc.metadata = m_MetadataBuffer;
    encodeDesc.resolvedMetadata = m_ResolvedMetadataBuffer;
    encodeDesc.av1PictureDesc = m_Codec == SampleCodec::AV1 ? &av1PictureDesc : nullptr;

    if (!SubmitOneTime(NRI, *m_VideoEncodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::BufferBarrierDesc bufferBarriers[2] = {};
            bufferBarriers[0].buffer = m_MetadataBuffer;
            bufferBarriers[0].after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[1].buffer = m_ResolvedMetadataBuffer;
            bufferBarriers[1].after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};

            nri::TextureBarrierDesc textureBarriers[2] = {};
            textureBarriers[0].texture = m_EncodeTexture;
            textureBarriers[0].before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[0].after = {nri::AccessBits::VIDEO_ENCODE_READ, nri::Layout::VIDEO_ENCODE_SRC, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[0].mipNum = nri::REMAINING;
            textureBarriers[0].layerNum = nri::REMAINING;
            textureBarriers[0].planes = nri::PlaneBits::ALL;
            textureBarriers[1].texture = m_ReconstructedTexture;
            textureBarriers[1].before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[1].after = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::Layout::VIDEO_ENCODE_DPB, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[1].mipNum = nri::REMAINING;
            textureBarriers[1].layerNum = nri::REMAINING;
            textureBarriers[1].planes = nri::PlaneBits::ALL;

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = bufferBarriers;
            barrierDesc.bufferNum = helper::GetCountOf(bufferBarriers);
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);
            NRI.CmdBarrier(commandBuffer, barrierDesc);
            Video.CmdEncodeVideo(commandBuffer, encodeDesc);
            // D3D12 resolves encode metadata inside CmdEncodeVideo and transitions the raw metadata buffer to encode-read before returning.
            bufferBarriers[0].before = {nri::AccessBits::VIDEO_ENCODE_READ, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[0].after = {};
            bufferBarriers[1].before = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::StageBits::VIDEO_ENCODE};
            bufferBarriers[1].after = {};
            textureBarriers[0].before = {nri::AccessBits::VIDEO_ENCODE_READ, nri::Layout::VIDEO_ENCODE_SRC, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[0].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarriers[1].before = {nri::AccessBits::VIDEO_ENCODE_WRITE, nri::Layout::VIDEO_ENCODE_DPB, nri::StageBits::VIDEO_ENCODE};
            textureBarriers[1].after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            barrierDesc.textures = textureBarriers;
            barrierDesc.textureNum = helper::GetCountOf(textureBarriers);
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_VideoStatus = std::string(GetCodecName(m_Codec)) + " encode submission failed";
        return false;
    }

    if (m_MetadataReadbackPending)
        return true;

    NRI.ResetCommandAllocator(*m_MetadataReadbackCommandAllocator);
    if (NRI.BeginCommandBuffer(*m_MetadataReadbackCommandBuffer, nullptr) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to begin metadata readback command buffer";
        return false;
    }

    nri::BufferBarrierDesc metadataBarriers[4] = {};
    metadataBarriers[0].buffer = m_ResolvedMetadataBuffer;
    metadataBarriers[0].before = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[0].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[1].buffer = m_ResolvedMetadataReadbackBuffer;
    metadataBarriers[1].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[2].buffer = m_BitstreamBuffer;
    metadataBarriers[2].before = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[2].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarriers[3].buffer = m_BitstreamHeaderReadbackBuffer;
    metadataBarriers[3].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

    nri::BarrierDesc metadataBarrierDesc = {};
    metadataBarrierDesc.buffers = metadataBarriers;
    metadataBarrierDesc.bufferNum = helper::GetCountOf(metadataBarriers);
    NRI.CmdBarrier(*m_MetadataReadbackCommandBuffer, metadataBarrierDesc);
    Video.CmdResolveVideoEncodeFeedback(*m_MetadataReadbackCommandBuffer, *m_EncodeSession, *m_ResolvedMetadataBuffer, 0);
    metadataBarriers[0].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarrierDesc.bufferNum = 1;
    NRI.CmdBarrier(*m_MetadataReadbackCommandBuffer, metadataBarrierDesc);
    metadataBarrierDesc.bufferNum = helper::GetCountOf(metadataBarriers);
    NRI.CmdCopyBuffer(*m_MetadataReadbackCommandBuffer, *m_ResolvedMetadataReadbackBuffer, 0, *m_ResolvedMetadataBuffer, 0, RESOLVED_METADATA_SIZE);
    NRI.CmdCopyBuffer(*m_MetadataReadbackCommandBuffer, *m_BitstreamHeaderReadbackBuffer, 0, *m_BitstreamBuffer, ENCODED_SLICE_OFFSET, AV1_HEADER_READBACK_SIZE);
    metadataBarriers[0].before = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarriers[0].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[1].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[1].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[2].before = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
    metadataBarriers[2].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    metadataBarriers[3].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
    metadataBarriers[3].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
    NRI.CmdBarrier(*m_MetadataReadbackCommandBuffer, metadataBarrierDesc);

    if (NRI.EndCommandBuffer(*m_MetadataReadbackCommandBuffer) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to end metadata readback command buffer";
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
    if (NRI.QueueSubmit(*m_GraphicsQueue, submit) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to submit async metadata readback";
        return false;
    }

    m_MetadataReadbackPending = true;
    m_VideoStatus = std::string(GetCodecName(m_Codec)) + " encode submitted; waiting for async metadata readback";
    return true;
}

bool Sample::TryDecodePendingMetadata(float timeSec) {
    if (!m_MetadataReadbackPending)
        return false;

    if (NRI.GetFenceValue(*m_MetadataReadbackFence) < m_MetadataReadbackFenceValue)
        return false;

    m_MetadataReadbackPending = false;

    nri::VideoEncodeFeedback feedback = {};
    const nri::Result feedbackResult = Video.GetVideoEncodeFeedback(*m_EncodeSession, *m_ResolvedMetadataReadbackBuffer, 0, feedback);
    if (feedbackResult != nri::Result::SUCCESS) {
        if (feedbackResult == nri::Result::UNSUPPORTED && m_Codec == SampleCodec::AV1) {
            feedback.encodedBitstreamOffset = 0;
            feedback.encodedBitstreamWrittenBytes = AV1_HEADER_READBACK_SIZE;
            feedback.writtenSubregionNum = 1;
        } else {
            if (feedbackResult == nri::Result::UNSUPPORTED)
                m_VideoStatus = std::string(GetCodecName(m_Codec)) + " encode metadata feedback is unsupported";
            else
                m_VideoStatus = "Failed to read resolved encode metadata";
            return false;
        }
    }

    if (feedback.errorFlags || !feedback.encodedBitstreamWrittenBytes) {
        char message[160] = {};
        std::snprintf(message, sizeof(message), "Encoder returned errorFlags=0x%llX bytes=%llu",
            (unsigned long long)feedback.errorFlags, (unsigned long long)feedback.encodedBitstreamWrittenBytes);
        m_VideoStatus = message;
        return false;
    }

    nri::VideoAV1EncodeDecodeInfo av1DecodeInfo = {};
    if (m_Codec == SampleCodec::AV1) {
        const uint8_t* encodedHeader = (const uint8_t*)NRI.MapBuffer(*m_BitstreamHeaderReadbackBuffer, 0, AV1_HEADER_READBACK_SIZE);
        if (!encodedHeader && feedbackResult == nri::Result::UNSUPPORTED) {
            m_VideoStatus = "Failed to map AV1 encoded header readback";
            return false;
        }

        nri::VideoAV1EncodeDecodeInfoDesc av1InfoDesc = {};
        av1InfoDesc.feedback = &feedback;
        av1InfoDesc.sequence = &m_AV1Sequence;
        av1InfoDesc.encodedPayloadHeader = encodedHeader;
        av1InfoDesc.encodedPayloadHeaderSize = encodedHeader ? std::min<uint64_t>(AV1_HEADER_READBACK_SIZE, feedback.encodedBitstreamWrittenBytes) : 0;
        const nri::Result av1InfoResult = Video.GetVideoEncodeAV1DecodeInfo(*m_EncodeSession, *m_ResolvedMetadataReadbackBuffer, 0, av1InfoDesc, av1DecodeInfo);
        if (encodedHeader)
            NRI.UnmapBuffer(*m_BitstreamHeaderReadbackBuffer);
        if (av1InfoResult != nri::Result::SUCCESS) {
            m_VideoStatus = "Failed to prepare AV1 decode metadata";
            return false;
        }
        feedback.encodedBitstreamWrittenBytes = av1DecodeInfo.bitstreamOffset + av1DecodeInfo.bitstreamSize;
    }

    return DecodeEncodedBitstream(feedback, m_Codec == SampleCodec::AV1 ? &av1DecodeInfo : nullptr, timeSec);
}

bool Sample::DecodeEncodedBitstream(const nri::VideoEncodeFeedback& feedback, const nri::VideoAV1EncodeDecodeInfo* av1DecodeInfo, float timeSec) {
    std::vector<uint8_t> annexBHeaders;
    if (!WriteAnnexBHeadersToUploadBuffer(annexBHeaders))
        return false;

    const uint64_t encodedPayloadSkip = av1DecodeInfo ? av1DecodeInfo->bitstreamOffset : GetEncodedPayloadHeaderSkip(m_Codec, feedback.encodedBitstreamWrittenBytes);
    const uint64_t encodedPayloadBytes = av1DecodeInfo ? av1DecodeInfo->bitstreamSize : feedback.encodedBitstreamWrittenBytes - encodedPayloadSkip;
    const uint64_t decodeSliceOffset = annexBHeaders.size();
    const uint64_t decodeBitstreamSize = AlignUp(decodeSliceOffset + encodedPayloadBytes, 256);
    const uint64_t encodedSourceOffset = ENCODED_SLICE_OFFSET + feedback.encodedBitstreamOffset + encodedPayloadSkip;
    if (feedback.encodedBitstreamOffset > BITSTREAM_SIZE - ENCODED_SLICE_OFFSET || encodedSourceOffset > BITSTREAM_SIZE || encodedPayloadBytes > BITSTREAM_SIZE - encodedSourceOffset || decodeBitstreamSize > BITSTREAM_SIZE) {
        m_VideoStatus = std::string("Encoded ") + GetCodecName(m_Codec) + " bitstream exceeded decode buffer size";
        return false;
    }

    if (!SubmitOneTime(NRI, *m_GraphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            NRI.CmdZeroBuffer(commandBuffer, *m_DecodeBitstreamBuffer, 0, BITSTREAM_SIZE);
            if (!annexBHeaders.empty())
                NRI.CmdCopyBuffer(commandBuffer, *m_DecodeBitstreamBuffer, 0, *m_BitstreamHeaderUploadBuffer, 0, annexBHeaders.size());

            nri::BufferBarrierDesc bufferBarriers[2] = {};
            bufferBarriers[0].buffer = m_BitstreamBuffer;
            bufferBarriers[0].before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            bufferBarriers[0].after = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
            bufferBarriers[1].buffer = m_DecodeBitstreamBuffer;
            bufferBarriers[1].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
            bufferBarriers[1].after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.buffers = bufferBarriers;
            barrierDesc.bufferNum = helper::GetCountOf(bufferBarriers);
            NRI.CmdBarrier(commandBuffer, barrierDesc);

            NRI.CmdCopyBuffer(commandBuffer, *m_DecodeBitstreamBuffer, decodeSliceOffset, *m_BitstreamBuffer, encodedSourceOffset, encodedPayloadBytes);

            bufferBarriers[0].before = {nri::AccessBits::COPY_SOURCE, nri::StageBits::COPY};
            bufferBarriers[0].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
            bufferBarriers[1].before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
            bufferBarriers[1].after = {nri::AccessBits::NONE, nri::StageBits::NONE};
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_VideoStatus = std::string("Failed to build exact ") + GetCodecName(m_Codec) + " decode bitstream";
        return false;
    }

    const uint32_t pictureOffsets[] = {(uint32_t)decodeSliceOffset};

    nri::VideoH264DecodePictureDesc h264DecodePicture = {};
    h264DecodePicture.flags = nri::VideoH264DecodePictureBits::IDR | nri::VideoH264DecodePictureBits::INTRA | nri::VideoH264DecodePictureBits::REFERENCE;
    h264DecodePicture.pictureParameterSetId = m_H264Pps.pictureParameterSetId;
    h264DecodePicture.frameNum = 0;
    h264DecodePicture.idrPictureId = 1;
    h264DecodePicture.topFieldOrderCount = 0;
    h264DecodePicture.bottomFieldOrderCount = 0;
    h264DecodePicture.sliceOffsets = pictureOffsets;
    h264DecodePicture.sliceOffsetNum = helper::GetCountOf(pictureOffsets);

    nri::VideoH265DecodePictureDesc h265DecodePicture = {};
    h265DecodePicture.flags = nri::VideoH265DecodePictureBits::IRAP | nri::VideoH265DecodePictureBits::IDR | nri::VideoH265DecodePictureBits::REFERENCE;
    h265DecodePicture.videoParameterSetId = m_H265Vps.videoParameterSetId;
    h265DecodePicture.sequenceParameterSetId = m_H265Sps.sequenceParameterSetId;
    h265DecodePicture.pictureParameterSetId = m_H265Pps.pictureParameterSetId;
    h265DecodePicture.pictureOrderCount = 0;
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
    }

    nri::VideoDecodeDesc decodeDesc = {};
    decodeDesc.session = m_DecodeSession;
    decodeDesc.parameters = m_DecodeParameters;
    decodeDesc.bitstream.buffer = m_DecodeBitstreamBuffer;
    decodeDesc.bitstream.size = decodeBitstreamSize;
    decodeDesc.dstPicture = m_DecodePicture;
    decodeDesc.dstSlot = 0;
    decodeDesc.h264PictureDesc = m_Codec == SampleCodec::H264 ? &h264DecodePicture : nullptr;
    decodeDesc.h265PictureDesc = m_Codec == SampleCodec::H265 ? &h265DecodePicture : nullptr;
    decodeDesc.av1PictureDesc = av1DecodeInfo ? &av1Info.picture : nullptr;

    nri::VideoDecodePictureStates decodePictureStates = {};
    if (Video.GetVideoDecodePictureStates(*m_DecodePicture, decodePictureStates) != nri::Result::SUCCESS) {
        m_VideoStatus = "Failed to query video decode picture states";
        return false;
    }

    if (!SubmitOneTime(NRI, *m_VideoDecodeQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = m_DecodeTexture;
            textureBarrier.before = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            textureBarrier.after = decodePictureStates.decodeWrite;
            textureBarrier.mipNum = nri::REMAINING;
            textureBarrier.layerNum = nri::REMAINING;
            textureBarrier.planes = nri::PlaneBits::ALL;

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = &textureBarrier;
            barrierDesc.textureNum = 1;
            NRI.CmdBarrier(commandBuffer, barrierDesc);
            Video.CmdDecodeVideo(commandBuffer, decodeDesc);

            if (decodePictureStates.releaseAfterDecode) {
                textureBarrier.before = decodePictureStates.decodeWrite;
                textureBarrier.after = decodePictureStates.afterDecode;
                NRI.CmdBarrier(commandBuffer, barrierDesc);
            }
        })) {
        m_VideoStatus = std::string(GetCodecName(m_Codec)) + " decode submission failed";
        return false;
    }

    if (!SubmitOneTime(NRI, *m_GraphicsQueue, [&](nri::CommandBuffer& commandBuffer) {
            nri::TextureBarrierDesc textureBarrier = {};
            textureBarrier.texture = m_DecodeTexture;
            textureBarrier.before = decodePictureStates.graphicsBefore;
            textureBarrier.after = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
            textureBarrier.mipNum = nri::REMAINING;
            textureBarrier.layerNum = nri::REMAINING;
            textureBarrier.planes = nri::PlaneBits::ALL;

            nri::BarrierDesc barrierDesc = {};
            barrierDesc.textures = &textureBarrier;
            barrierDesc.textureNum = 1;
            NRI.CmdBarrier(commandBuffer, barrierDesc);

            nri::BufferBarrierDesc nv12BufferBarrier = {};
            nv12BufferBarrier.buffer = m_UploadBuffer;
            nv12BufferBarrier.before = {nri::AccessBits::NONE, nri::StageBits::NONE};
            nv12BufferBarrier.after = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};

            nri::BarrierDesc copyBarrierDesc = {};
            copyBarrierDesc.buffers = &nv12BufferBarrier;
            copyBarrierDesc.bufferNum = 1;
            NRI.CmdBarrier(commandBuffer, copyBarrierDesc);

            nri::TextureRegionDesc lumaRegion = {};
            lumaRegion.width = VIDEO_WIDTH;
            lumaRegion.height = VIDEO_HEIGHT;
            lumaRegion.depth = 1;
            lumaRegion.planes = nri::PlaneBits::PLANE_0;

            nri::TextureDataLayoutDesc lumaLayout = {};
            lumaLayout.rowPitch = m_Nv12Layout.yRowPitchBytes;
            lumaLayout.slicePitch = m_Nv12Layout.ySlicePitchBytes;
            NRI.CmdReadbackTextureToBuffer(commandBuffer, *m_UploadBuffer, lumaLayout, *m_DecodeTexture, lumaRegion);

            nri::TextureRegionDesc chromaRegion = {};
            chromaRegion.width = VIDEO_WIDTH;
            chromaRegion.height = VIDEO_HEIGHT;
            chromaRegion.depth = 1;
            chromaRegion.planes = nri::PlaneBits::PLANE_1;

            nri::TextureDataLayoutDesc chromaLayout = {};
            chromaLayout.offset = m_Nv12Layout.uvOffsetBytes;
            chromaLayout.rowPitch = m_Nv12Layout.uvRowPitchBytes;
            chromaLayout.slicePitch = m_Nv12Layout.uvSlicePitchBytes;
            NRI.CmdReadbackTextureToBuffer(commandBuffer, *m_UploadBuffer, chromaLayout, *m_DecodeTexture, chromaRegion);

            nv12BufferBarrier.before = {nri::AccessBits::COPY_DESTINATION, nri::StageBits::COPY};
            nv12BufferBarrier.after = {nri::AccessBits::NONE, nri::StageBits::NONE};
            NRI.CmdBarrier(commandBuffer, copyBarrierDesc);

            textureBarrier.before = {nri::AccessBits::COPY_SOURCE, nri::Layout::COPY_SOURCE, nri::StageBits::COPY};
            textureBarrier.after = {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
            NRI.CmdBarrier(commandBuffer, barrierDesc);
        })) {
        m_VideoStatus = "Failed to copy decoded NV12 for preview";
        return false;
    }

    PatternConstants patternConstants = MakePatternConstants(OP_NV12_TO_PREVIEW, timeSec);
    if (!GeneratePatternWithCompute(patternConstants, m_DecodePreviewStorage, true)) {
        m_VideoStatus = "Failed to convert decoded NV12 to preview texture";
        return false;
    }

    m_DecodePreviewReady = true;
    char message[128] = {};
    std::snprintf(message, sizeof(message), "%s encode/decode round trip complete, encoded %llu bytes", GetCodecName(m_Codec), (unsigned long long)feedback.encodedBitstreamWrittenBytes);
    m_VideoStatus = message;
    return true;
}

bool Sample::TryRunRoundTrip(float timeSec) {
    if (m_MetadataReadbackPending)
        return TryDecodePendingMetadata(timeSec);

    return TrySubmitEncodeAndMetadataReadback(timeSec);
}

bool Sample::CanRunRoundTrip() const {
    return m_VideoReady && m_GraphicsQueue && m_VideoEncodeQueue && m_VideoDecodeQueue && m_UploadBuffer && m_UploadBufferView && m_SourcePreviewStorage && m_DecodePreviewStorage && m_GeneratePipelineLayout && m_GenerateComputePipeline && m_GenerateDescriptorSet;
}

void Sample::LatencySleep(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
    const QueuedFrame& queuedFrame = m_QueuedFrames[queuedFrameIndex];

    NRI.Wait(*m_FrameFence, frameIndex >= GetQueuedFrameNum() ? 1 + frameIndex - GetQueuedFrameNum() : 0);
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
            ImGui::Text("Codec: %s, format: NV12, size: %ux%u", GetCodecName(m_Codec), VIDEO_WIDTH, VIDEO_HEIGHT);
            ImGui::TextWrapped("Video: %s", m_VideoStatus.c_str());
            ImGui::TextWrapped("Preview: %s", m_PreviewStatus.c_str());
            ImGui::Text("Encode queue: %s, decode queue: %s", m_VideoEncodeQueue ? "yes" : "no", m_VideoDecodeQueue ? "yes" : "no");
            const bool roundTripSupported = CanRunRoundTrip();
            ImGui::Text("Round trip: %s", roundTripSupported ? "running" : "backend must support NRI video encode/decode");
            if (m_VideoReady && !m_DecodePreviewReady)
                ImGui::Text("Decode preview: waiting for first decoded frame");

            ImGui::Separator();
            if (ImGui::BeginTable("PreviewPanels", 2, ImGuiTableFlags_SizingStretchSame)) {
                ImGui::TableNextColumn();
                float width = std::max(200.0f, ImGui::GetContentRegionAvail().x);
                DrawTexturePanel("Generated source", m_SourcePreviewTextureView, {width, width * float(VIDEO_HEIGHT) / float(VIDEO_WIDTH)});

                ImGui::TableNextColumn();
                width = std::max(200.0f, ImGui::GetContentRegionAvail().x);
                DrawTexturePanel(m_DecodePreviewReady ? "Decoded preview" : "Decoded preview pending", m_DecodePreviewReady ? m_DecodePreviewTextureView : nullptr, {width, width * float(VIDEO_HEIGHT) / float(VIDEO_WIDTH)});
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
    ImGui::EndFrame();
    ImGui::Render();
}

void Sample::RenderFrame(uint32_t frameIndex) {
    uint32_t queuedFrameIndex = frameIndex % GetQueuedFrameNum();
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

        nri::RenderingDesc renderingDesc = {};
        renderingDesc.colorNum = 1;
        renderingDesc.colors = &colorAttachmentDesc;

        CmdCopyImguiData(commandBuffer, *m_Streamer);

        NRI.CmdBeginRendering(commandBuffer, renderingDesc);
        CmdDrawImgui(commandBuffer, swapChainTexture.attachmentFormat, 1.0f, true);
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
    NRI.QueuePresent(*m_SwapChain, *swapChainTexture.releaseSemaphore);

    nri::FenceSubmitDesc signalFence = {};
    signalFence.fence = m_FrameFence;
    signalFence.value = 1 + frameIndex;

    nri::QueueSubmitDesc signalSubmitDesc = {};
    signalSubmitDesc.signalFences = &signalFence;
    signalSubmitDesc.signalFenceNum = 1;
    NRI.QueueSubmit(*m_GraphicsQueue, signalSubmitDesc);
}

SAMPLE_MAIN(Sample, 0);
