#!/bin/sh

git submodule update --init --recursive

chmod +x "External/Packman/packman.sh"
chmod +x "External/NRIFramework/External/Packman/packman.sh"
chmod +x "External/NRIFramework/External/NRI/External/Packman/packman.sh"
chmod +x "1-Deploy.sh"
chmod +x "2-Build.sh"
chmod +x "4-Clean.sh"

mkdir -p "_Build"

cd "_Build"
cmake ..
cd ..
