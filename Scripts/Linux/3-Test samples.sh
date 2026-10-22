#!/bin/bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

DIR_BIN=""
ADAPTER=0
FAIL=0

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

APIS=(VULKAN WGPU)

echo "Running samples from '$DIR_BIN'..."
echo

run_sample() {
    if "$@"; then
        echo "=> OK"
    elif [[ $? -eq 2 ]]; then
        echo "=> SKIP"
    else
        echo "=> FAIL"
        FAIL=1
    fi
    echo
}

test_sample() {
    local sample="$1"
    shift
    local api

    # Wrapper is built only when the Vulkan SDK is available.
    if [[ "$sample" == Wrapper && ! -x "$DIR_BIN/$sample" ]]; then
        echo "Wrapper => SKIP (not built)"
        echo
        return
    fi

    for api in "${APIS[@]}"; do
        echo "$sample [$api]"
        run_sample "$DIR_BIN/$sample" "--api=$api" --timeLimit=3 --debugAPI --debugNRI "--adapter=$ADAPTER" "$@"
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
test_sample VideoEncodeDecode --stream --frames=2 --videoWidth=640 --videoHeight=360
test_sample Wrapper

exit "$FAIL"
