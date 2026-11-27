# NRI SAMPLES

[![Status](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml)

Samples demonstrating [NVIDIA Render Interface (NRI)](https://github.com/NVIDIA-RTX/NRI), a cross-platform rendering hardware interface (RHI), covering graphics, compute, ray tracing, resource copies, streaming, low latency, video encoding/decoding, and other features.

## HOW TO BUILD

Prerequisites:

- *CMake* 3.30+ and a C/C++ compiler
- *Windows:* *Windows SDK* and [*Vulkan SDK*](https://vulkan.lunarg.com/sdk/home)
- *Linux:* *Vulkan SDK* and *xorg-dev*
- *macOS:* *Xcode* 26+ with command line tools, *Ninja*, and the *Vulkan SDK* (provides *DXC* for shader compilation and *Vulkan* through *MoltenVK*). Install the Metal toolchain with `xcodebuild -downloadComponent MetalToolchain`. Install *Metal Shader Converter* to run the samples on *Metal*

Notes:

- Run `Scripts/<platform>/1-Deploy.<ext>`, then `Scripts/<platform>/2-Build.<ext>` for your platform
- Scripts resolve the project root from their own location and can be launched from any working directory. Build output is placed in `_Build` and `_Bin` at the project root
- To clean generated files, run `Scripts/<platform>/4-Clean.<ext>`. Cleanup also removes downloaded `_Data`
- To use the static MSVC runtime, pass `-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` when configuring CMake
- On macOS, source the Vulkan SDK's `setup-env.sh` before building. *Metal Shader Converter* enables the *HLSL to DXIL to Metal shader* workflow automatically when installed and makes *Metal* the default API; otherwise, *Vulkan* is used

*CMake* options:

- `NRIF_USE_WAYLAND` - use Wayland instead of X11 on Linux

## HOW TO RUN

Run executables from `_Bin` with the working directory set to the project root so they can load resources from `_Data` and shaders from `_Shaders`. To run all samples sequentially, use `Scripts/Windows/3-Test samples.bat`, `bash "Scripts/Linux/3-Test samples.sh"`, or `bash "Scripts/MacOS/3-Test samples.sh"`. These scripts set the working directory automatically. The Linux runner uses Vulkan and WebGPU; the macOS runner also uses Metal.

## SAMPLES

- [AsyncCompute](Source/AsyncCompute.cpp) - demonstrates parallel execution of graphics and compute workloads
- [BindlessSceneViewer](Source/BindlessSceneViewer.cpp) - bindless GPU-driven rendering test
- [Buffers](Source/Buffers.c) - buffer creation, copying and readback
- [Clear](Source/Clear.cpp) - minimal example of rendering using framebuffer clears only
- [ClearStorage](Source/ClearStorage.c) - clear storage testing
- [CopyTests](Source/CopyTests.cpp) - validates core copy commands and synchronous host texture copies
- [DeviceInfo](Source/DeviceInfo.c) - queries and prints information about GPU adapters in the system
- [DescriptorPoolIndexing](Source/DescriptorPoolIndexing.cpp) - HLSL dynamic resources through a directly indexed descriptor pool
- [DescriptorHeapIndexing](Source/DescriptorHeapIndexing.cpp) - HLSL dynamic resources through a directly indexed descriptor heap
- [DescriptorManagement](Source/DescriptorManagement.cpp) - descriptor copying, pool switching, update-after-set and pool recycling testing
- [DedicatedQueues](Source/DedicatedQueues.cpp) - dedicated copy queue, synchronization and copy-queue timestamp testing
- [GraphicsPipelineStates](Source/GraphicsPipelineStates.cpp) - dynamic graphics state, geometry, tessellation and less common rasterization testing
- [GraphicsVertexInput](Source/GraphicsVertexInput.cpp) - vertex input, geometry and tessellation draw testing
- [InputAttachment](Source/InputAttachment.cpp) - "dynamic rendering local read" demonstration (reading on-chip rendering results)
- [IndirectCommands](Source/IndirectCommands.cpp) - indirect graphics and compute command testing
- [LowLatency](Source/LowLatency.cpp) - low latency demonstration
- [MemoryAliasing](Source/MemoryAliasing.cpp) - overlapping resource memory testing
- [MeshShader](Source/MeshShader.cpp) - direct and indirect mesh shader dispatch testing
- [Multisample](Source/Multisample.cpp) - multisample rendering testing
- [MultiThreading](Source/MultiThreading.cpp) - shows advantages of multi-threaded command buffer recording
- [Multiview](Source/Multiview.cpp) - multiview demonstration in _LAYER_BASED_ mode (VK and D3D12 compatible)
- [Queries](Source/Queries.cpp) - timestamp, occlusion and calibrated timestamp testing
- [RayTracingAdvanced](Source/RayTracingAdvanced.cpp) - procedural geometry, update, clone and compaction testing
- [RayTracingBoxes](Source/RayTracingBoxes.cpp) - a more advanced ray tracing example with many BLASes in TLAS
- [RayTracingTriangle](Source/RayTracingTriangle.cpp) - simple triangle rendering through ray tracing
- [Readback](Source/Readback.cpp) - getting data from the GPU back to the CPU
- [Resize](Source/Resize.cpp) - demonstrates window resize
- [ResolveSampleZero](Source/ResolveSampleZero.cpp) - sample-zero resolve testing for color and depth-stencil textures
- [Resources](Source/Resources.c) - resource allocation testing
- [SceneViewer](Source/SceneViewer.cpp) - loading & rendering of meshes with materials (also tests programmable sample locations, shading rate and pipeline statistics)
- [Streamer](Source/Streamer.cpp) - buffer and texture streaming testing
- [TextureTypes](Source/TextureTypes.cpp) - 1D, 2D array, cube and 3D texture/view testing
- [Triangle](Source/Triangle.cpp) - simple textured triangle rendering (also multiview demonstration in _FLEXIBLE_ mode)
- [VideoEncodeDecode](Source/VideoEncodeDecode.cpp) - H.264/H.265/AV1 NV12 video encode/decode round trip
- [Wrapper](Source/Wrapper.cpp) - shows how to wrap native D3D11/D3D12/VK objects into *NRI* entities

VideoEncodeDecode notes:

- Select `--codec=H264`, `H265`, or `AV1` (default: `H264`). The animated preview supports `--h26Frame=IDR|P|B` for H.264/H.265 and `--av1Frame=IDR|P` for AV1; both default to `IDR`.
- Use `--videoWidth` and `--videoHeight` for the NV12 video size (default: 1920x1080). Each dimension falls back to `--width` or `--height` if supplied; explicit video dimensions take precedence. Width must be divisible by 4 and height must be even.
- CQP is the default rate control mode. Set `--qpI`, `--qpP`, and `--qpB` for frame quantizers, and `--av1BaseQIndex` for AV1's base quantizer. The sample accepts 0..51 for H.264/H.265 and 1..255 for AV1. Lower values favor quality. `--lossless` uses zero quantizers for H.264/H.265 and near-lossless quantizer 1 for AV1.
- Use `--stream` for a headless encode/decode sequence, `--matrix` for the built-in cases, or `--listCases` to list them. Stream options include `--frames`, `--gop`, `--rc=cqp|cbr|vbr`, and `--dump` for output files.
- External AV1 OBU decoding with `--decodeFile` requires `-DNRISAMPLES_ENABLE_AV1_FILE_DECODING=ON` and matching NRI sources for its private parser. Optional `--refYuv` checks decoded output against an NV12 reference. This developer feature is off by default; previews, round-trip streams, and matrix cases use public NRI APIs.
