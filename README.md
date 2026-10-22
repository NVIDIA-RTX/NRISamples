# NRI Samples

[![Status](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml)

This is the test bench for [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIA-RTX/NRI).

## Build instructions

Scripts resolve the project root from their own location and can be launched from any working directory. Build output remains in `_Build` and `_Bin` at the project root.

To clean generated files, run `Scripts/Windows/4-Clean.bat`, `bash Scripts/Linux/4-Clean.sh` or `bash Scripts/MacOS/4-Clean.sh`. Cleanup also removes downloaded `_Data` and runs the matching cleanup scripts in NRIFramework and NRI.

### Windows

- Install **WindowsSDK** and **VulkanSDK**
- Clone project and init submodules
- Generate and build project using **cmake**
  - To build the binary with static MSVC runtime, add `-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"` parameter

Or by running scripts only:
- Run `Scripts/Windows/1-Deploy.bat`
- Run `Scripts/Windows/2-Build.bat`

### Linux

- Install **VulkanSDK**, **xorg-dev**,
- Clone project and init submodules
- Generate and build project using **cmake**

Or by running scripts only:
- Run `bash Scripts/Linux/1-Deploy.sh`
- Run `bash Scripts/Linux/2-Build.sh`

### macOS

- Install **Xcode** 26+ with command line tools and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- Install **CMake** 3.30+ and **Ninja**
- Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac) (provides *DXC*, required to compile shaders, and *Vulkan* through *MoltenVK*)
- Source the Vulkan SDK's `setup-env.sh`
- Install **Metal Shader Converter** to run the samples on *Metal* (HLSL → DXIL → Metal, enabled automatically if installed, *Metal* becomes the default API; otherwise *Vulkan* is used)
- Clone project and init submodules
- Generate and build project using **cmake**

Or by running scripts only:
- Run `bash Scripts/MacOS/1-Deploy.sh`
- Run `bash Scripts/MacOS/2-Build.sh`

### CMake options

- `DISABLE_SHADER_COMPILATION` - disable compilation of shaders (shaders can be built on other platform)
- `NRIF_USE_WAYLAND` - use Wayland instead of X11 on Linux

## How to run

The executables from `_Bin` directory load resources from `_Data`, therefore the samples need to be run with the working directory set to the project root folder. To run all samples sequentially, use `Scripts/Windows/3-Test samples.bat`, `bash "Scripts/Linux/3-Test samples.sh"` or `bash "Scripts/MacOS/3-Test samples.sh"`. These scripts set the working directory to the project root. The Linux and macOS runners select backends enabled in `_Build/CMakeCache.txt`.

## Samples

- AsyncCompute - demonstrates parallel execution of graphic and compute workloads
- BindlessSceneViewer - bindless GPU-driven rendering test
- Buffers - various buffer-related stuff
- Clear - minimal example of rendering using framebuffer clears only
- ClearStorage - clear storage testing
- CopyTests - validates core copy commands and synchronous host texture copies
- DeviceInfo - queries and prints out information about device groups in the system
- DescriptorPoolIndexing - HLSL dynamic resources through a directly indexed descriptor pool
- DescriptorHeapIndexing - HLSL dynamic resources through a directly indexed descriptor heap
- DescriptorManagement - descriptor copying, pool switching, update-after-set and pool recycling testing
- DedicatedQueues - dedicated copy queue, synchronization and copy-queue timestamp testing
- GraphicsPipelineStates - dynamic graphics state, geometry, tessellation and less common rasterization testing
- GraphicsVertexInput - vertex input, geometry and tessellation draw testing
- InputAttachment - "dynamic rendering local read" demonstration (reading on-chip rendering results)
- IndirectCommands - indirect graphics and compute command testing
- LowLatency - low latency demonstration
- MemoryAliasing - overlapping resource memory testing
- MeshShader - direct and indirect mesh shader dispatch testing
- Multisample - multisample rendering testing
- MultiThreading - shows advantages of multi-threaded command buffer recording
- Multiview - multiview demonstration in _LAYER_BASED_ mode (VK and D3D12 compatible)
- Queries - timestamp, occlusion and calibrated timestamp testing
- RayTracingAdvanced - procedural geometry, update, clone and compaction testing
- RayTracingBoxes - a more advanced ray tracing example with many BLASes in TLAS
- RayTracingTriangle - simple triangle rendering through ray tracing
- Readback - getting data from the GPU back to the CPU
- Resize - demonstrates window resize
- Resources - various resources allocation related stuff
- SceneViewer - loading & rendering of meshes with materials (also tests programmable sample locations, shading rate and pipeline statistics)
- Streamer - buffer and texture streaming testing
- TextureTypes - 1D, 2D array, cube and 3D texture/view testing
- Triangle - simple textured triangle rendering (also multiview demonstration in _FLEXIBLE_ mode)
- VideoEncodeDecode - H.264/H.265/AV1 NV12 video encode/decode round trip; built only when the selected NRI source provides the NRIVideo extension
- Wrapper - shows how to wrap native D3D11/D3D12/VK objects into *NRI* entities

### VideoEncodeDecode AV1 notes

`VideoEncodeDecode --codec=AV1 --av1Frame=P` encodes a static generated source image as an IDR reference followed by a P frame. The source is intentionally frozen for this permutation so the left generated image and right decoded image can be compared directly while validating AV1 reference/DPB handling. It is not intended to demonstrate animated AV1 P-frame motion.

The framework `--width` and `--height` arguments control the sample output window size only. Use `--videoWidth` and `--videoHeight` to select the NV12 encode/decode surface size. The default video size is 1920x1080.

The sample uses CQP rate control by default. Use `--qpI`, `--qpP`, and `--qpB` to change the per-frame-type quantizers, and `--av1BaseQIndex` to change AV1's frame base quantizer. H.264/H.265 QP values must be in the 0..51 range; AV1 quantizer values use the 0..255 range. Lower values generally produce higher quality and larger bitstreams.

External AV1 file decoding (`--decodeFile` and `--refYuv`) is an optional developer feature. Enable it with `-DNRISAMPLES_ENABLE_AV1_FILE_DECODING=ON` when configuring CMake. It uses NRI's private AV1 parser and requires a matching NRI source checkout. The feature is off by default; the visual sample, round-trip streams and test matrix use only public NRI APIs.
