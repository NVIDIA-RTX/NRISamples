#!/bin/bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

DIR_BIN=""
ADAPTER=0
FAILED=0

for directory in _Bin/Release _Bin/Debug _Bin; do
    if [[ -x "$directory/DeviceInfo" ]]; then
        DIR_BIN="$directory"
        break
    fi
done

if [[ -z "$DIR_BIN" ]]; then
    echo "The project is not compiled!"
    exit 1
fi

API_NAMES=(METAL VULKAN WGPU)
API_OPTIONS=(METAL VK WGPU)
APIS=()
for i in "${!API_NAMES[@]}"; do
    if [[ -f _Build/CMakeCache.txt ]] && ! grep -Eq "^NRI_ENABLE_${API_OPTIONS[$i]}_SUPPORT:BOOL=(ON|TRUE|1)$" _Build/CMakeCache.txt; then
        continue
    fi
    if [[ "${API_NAMES[$i]}" == METAL && -f _Build/CMakeCache.txt ]] && ! grep -Eq '^NRI_ENABLE_METAL_SHADER_CONVERTER:BOOL=(ON|TRUE|1)$' _Build/CMakeCache.txt; then
        echo "Skipping METAL: samples require NRI_ENABLE_METAL_SHADER_CONVERTER."
        continue
    fi
    APIS+=("${API_NAMES[$i]}")
done

if [[ ${#APIS[@]} -eq 0 ]]; then
    echo "No supported graphics APIs are enabled!"
    exit 1
fi

echo "Running samples from '$DIR_BIN'..."
echo

run_sample() {
    if "$@"; then
        echo "=> OK"
    else
        echo "=> FAILED!"
        FAILED=1
    fi
    echo
}

test_sample() {
    local sample="$1"
    local api

    # Wrapper is built only when the Vulkan SDK is available.
    if [[ "$sample" == Wrapper && ! -x "$DIR_BIN/$sample" ]]; then
        echo "Wrapper => SKIPPED (not built)"
        echo
        return
    fi

    for api in "${APIS[@]}"; do
        echo "$sample [$api]"
        run_sample "$DIR_BIN/$sample" "--api=$api" --timeLimit=3 --debugAPI --debugNRI "--adapter=$ADAPTER"
    done
}

# API independent samples
run_sample "$DIR_BIN/DeviceInfo"

# API dependent samples
test_sample AsyncCompute
test_sample BindlessSceneViewer
test_sample Buffers
test_sample Clear
test_sample ClearStorage
test_sample CopyTests
test_sample DescriptorPoolIndexing
test_sample DescriptorHeapIndexing
test_sample DescriptorManagement
test_sample DedicatedQueues
test_sample GraphicsPipelineStates
test_sample GraphicsVertexInput
test_sample InputAttachment
test_sample IndirectCommands
test_sample LowLatency
test_sample MemoryAliasing
test_sample MeshShader
test_sample Multisample
test_sample MultiThreading
test_sample Multiview
test_sample Queries
test_sample RayTracingAdvanced
test_sample RayTracingBoxes
test_sample RayTracingTriangle
test_sample Readback
test_sample Resize
test_sample ResolveSampleZero
test_sample Resources
test_sample SceneViewer
test_sample Streamer
test_sample TextureTypes
test_sample Triangle
test_sample Wrapper

exit "$FAILED"
