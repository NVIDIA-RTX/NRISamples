# NRI Samples

[![Status](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIA-RTX/NRISamples/actions/workflows/build.yml)

This is the test bench for [*NRI (NVIDIA Rendering Interface)*](https://github.com/NVIDIA-RTX/NRI).

## Build instructions

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

- Install Xcode command line tools, CMake 3.30+, Ninja and the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac)
- Source the Vulkan SDK's `setup-env.sh`
- Run `bash Scripts/MacOS/1-Deploy.sh`, then `bash Scripts/MacOS/2-Build.sh`

Scripts resolve the project root from their own location and can be launched from any working directory. Build output remains in `_Build` and `_Bin` at the project root.

To clean generated files, run `Scripts/Windows/4-Clean.bat`, `bash Scripts/Linux/4-Clean.sh` or `bash Scripts/MacOS/4-Clean.sh`. Cleanup also removes downloaded `_Data` and runs the matching cleanup scripts in NRIFramework and NRI.

### CMake options

- `DISABLE_SHADER_COMPILATION` - disable compilation of shaders (shaders can be built on other platform)
- `NRIF_USE_WAYLAND` - use Wayland instead of X11 on Linux

## How to run

The executables from `_Bin` directory load resources from `_Data`, therefore the samples need to be run with the working directory set to the project root folder. But the simplest way to run ALL samples sequentially is to click on `Scripts/Windows/3-Test samples.bat`, which sets the working directory to the project root.

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
- Wrapper - shows how to wrap native D3D11/D3D12/VK objects into *NRI* entities

