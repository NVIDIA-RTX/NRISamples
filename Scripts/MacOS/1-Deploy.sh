#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

git submodule update --init --recursive

# Source the Vulkan SDK's setup-env.sh before deploying.
cmake -S . -B _Build -G "Ninja Multi-Config" "$@"
