#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

rm -rf "build"
rm -rf "_Bin"
rm -rf "_Build"
rm -rf "_Data"
rm -rf "_Shaders"

bash "External/NRIFramework/Scripts/Linux/4-Clean.sh"
