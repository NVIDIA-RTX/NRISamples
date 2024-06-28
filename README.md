# NRI Samples

## Build instructions

### Windows
- Install **WindowsSDK** and **VulkanSDK**
- Clone project and init submodules
- Generate and build project using **cmake**

Or by running scripts only:
- Run ``1-Deploy.bat``
- Run ``2-Build.bat``

### Linux (x86-64)
- Install **VulkanSDK**, **xorg-dev**, **libwayland-dev**, **libxkbcommon-dev**
- Clone project and init submodules
- Generate and build project using **cmake**

Or by running scripts only:
- Run `./1-Deploy.sh`
- RUn `./2-Build.sh`

### Linux (aarch64)
- Install **libx11-dev**, **libxrandr-dev**
- Clone project and init submodules
- Generate and build project using **cmake**

### CMake options
`-DUSE_MINIMAL_DATA=ON` - download minimal resource package (90MB)

`-DDISABLE_SHADER_COMPILATION=ON` - disable compilation of shaders (shaders can be built on other platform)

`-DDXC_CUSTOM_PATH=my/path/to/dxc` - custom path to **dxc**

`-DUSE_DXC_FROM_PACKMAN_ON_AARCH64=OFF` - use default path for **dxc**

## How to run
The executables load resources from `_Data`, therefore please run the samples with working directory set to
the project root folder. The executables can be found in `_Bin`.

## How to add new sample
Create a new cpp file in `Source/` and add `add_sample(YourFileName)` in `CMakeLists.txt`.

CMake script scans and adds all shaders from `Source/Shaders` to a project called `SampleShaders`, therefore
you will need to reconfigure CMake project after adding new shaders.
