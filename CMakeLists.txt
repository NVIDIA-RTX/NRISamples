cmake_minimum_required(VERSION 3.30)

include(FetchContent)

# GitHub actions?
if("$ENV{GITHUB_ACTIONS}" STREQUAL "")
    set(GITHUB_CI FALSE CACHE BOOL "GitHub Actions runner" FORCE)
else()
    set(GITHUB_CI TRUE CACHE BOOL "GitHub Actions runner" FORCE)
    message(STATUS "Detected environment: GitHub Actions...")
endif()

# Options
set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")

# Cached
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")

get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if(IS_MULTI_CONFIG)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")
endif()

# Create project
project(NRISamples LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJC)
endif()

# Globals?
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)

# Download dependencies
set(DEPS)

if(NOT GITHUB_CI)
    fetchcontent_declare(
        assets
        DOWNLOAD_EXTRACT_TIMESTAMP 1
        DOWNLOAD_NO_PROGRESS 1
        SOURCE_SUBDIR "ignore CMakeLists.txt"
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_Data"
        URL https://github.com/NVIDIA-RTX/NRISamples/releases/download/assets/_Data.7z
    )
    list(APPEND DEPS assets)
endif()

if(DEPS)
    message("Downloading dependencies:")
    message(STATUS "${DEPS} ...")

    fetchcontent_makeavailable(${DEPS})
endif()

# Compile options
if(NOT IOS AND NOT((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")))
    set(SIMD -mssse3)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS ${SIMD} -Wextra -Wno-missing-field-initializers)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS ${SIMD} -Wextra -Wno-missing-field-initializers)
elseif(MSVC)
    set(COMPILE_OPTIONS /W4 /WX
        /wd4324 # padding was added at the end of a structure because you specified an alignment specifier
    )
else()
    message(WARNING "Unknown compiler!")
endif()

# Compile definitions
if(WIN32)
    set(COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# External/NRIFramework
set(NRI_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
set(SHADERMAKE_FIND_SLANG ON)

add_subdirectory("External/NRIFramework")

# Move dependencies into "External"
function(fix_folders _DIRECTORY _FOLDER)
    get_property(_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)

    foreach(_TARGET IN LISTS _TARGETS)
        get_property(_FOLDER_ORIGINAL TARGET ${_TARGET} PROPERTY FOLDER)
        set_target_properties(${_TARGET} PROPERTIES FOLDER "${_FOLDER}/${_FOLDER_ORIGINAL}")
    endforeach()

    get_property(SUBDIRS DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)

    foreach(SUBDIR IN LISTS SUBDIRS)
        fix_folders("${SUBDIR}" "${_FOLDER}")
    endforeach()
endfunction()

fix_folders("External/NRIFramework" "External")

# Shaders
file(GLOB_RECURSE SHADERS "Shaders/*.hlsl" "Shaders/*.hlsli")
set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

get_target_property(ML_SOURCE_DIR MathLib SOURCE_DIR)
get_target_property(NRI_SOURCE_DIR NRI SOURCE_DIR)

if(NOT IOS) # iOS: shaders come from a macOS build
    set(SHADERMAKE_GENERAL_ARGS
        --project "NRISamples"
        --compactProgress
        --binary
        --flatten
        --stripReflection
        --WX
        --sRegShift 0 --bRegShift 32 --uRegShift 64 --tRegShift 128 # TODO: must be in sync with "VK_BINDING_OFFSETS"
        --sourceDir "Shaders"
        --ignoreConfigDir
        --vulkanVersion 1.2 # only for WGPU
        -c "Shaders/Shaders.cfg"
        -o "${SHADER_OUTPUT_PATH}"
        -I "Shaders"
        -I "${ML_SOURCE_DIR}"
        -I "${NRI_SOURCE_DIR}/Include"
    )

    set(SHADERMAKE_COMMANDS "")

    if(NRI_ENABLE_D3D11_SUPPORT)
        set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p DXBC --compiler "${SHADERMAKE_FXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    endif()

    if(NRI_ENABLE_D3D12_SUPPORT)
        set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p DXIL --compiler "${SHADERMAKE_DXC_PATH}" ${SHADERMAKE_GENERAL_ARGS})
    elseif(NRI_ENABLE_METAL_SHADER_CONVERTER)
        set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p DXIL --compiler "${SHADERMAKE_DXC_PATH}" ${SHADERMAKE_GENERAL_ARGS} -D NRI_METAL_DXIL)
    endif()

    if(NRI_ENABLE_VK_SUPPORT)
        set(SHADERMAKE_COMMANDS ${SHADERMAKE_COMMANDS} COMMAND ${SHADERMAKE_PATH} -p SPIRV --compiler "${SHADERMAKE_DXC_VK_PATH}" ${SHADERMAKE_GENERAL_ARGS} --compilerAlias "SLANG=${SHADERMAKE_SLANG_PATH}")
    endif()

    add_custom_target(${PROJECT_NAME}Shaders ALL
        ${SHADERMAKE_COMMANDS}
        DEPENDS ShaderMake
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
        SOURCES ${SHADERS}
    )

    set_target_properties(${PROJECT_NAME}Shaders PROPERTIES FOLDER ${PROJECT_NAME})
endif()

# Samples
set(NRISAMPLES_IOS_BUNDLE_ID_PREFIX "com.example.nrisamples" CACHE STRING "iOS: bundle identifier prefix, must be unique for your Apple ID")
set(NRISAMPLES_IOS_DEVELOPMENT_TEAM "" CACHE STRING "iOS: Apple team ID used for signing")
set(NRISAMPLES_IOS_MSC_DYLIB "/usr/local/lib_iOS/libmetalirconverter.dylib" CACHE FILEPATH "iOS: Metal Shader Converter library to embed into the app")
set(IOS_SAMPLES Clear Triangle) # samples built as iOS apps

# iOS: files copied into the app. Shaders are generated by a macOS build ("_Shaders"), the assets come from "_Data"
set(IOS_SHADERS_Triangle Triangle.vs Triangle.fs)
set(IOS_TEXTURES_Triangle wood.dds)

function(add_sample NAME EXT)
    if(IOS AND NOT "${NAME}" IN_LIST IOS_SAMPLES)
        return()
    endif()

    if(IOS)
        add_executable(${NAME} MACOSX_BUNDLE "Source/${NAME}.${EXT}")
    else()
        add_executable(${NAME} "Source/${NAME}.${EXT}")
    endif()
    source_group("" FILES "Source/${NAME}.${EXT}")
    if(NOT IOS)
        add_dependencies(${NAME} ${PROJECT_NAME}Shaders)
    endif()

    target_compile_definitions(${NAME} PRIVATE ${COMPILE_DEFINITIONS} PROJECT_NAME=${NAME})
    target_compile_options(${NAME} PRIVATE ${COMPILE_OPTIONS})
    target_link_libraries(${NAME} PRIVATE NRIFramework NRI)

    set_target_properties(${NAME} PROPERTIES FOLDER ${PROJECT_NAME})
    set_target_properties(${NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # iOS: an app bundle
    if(IOS)
        set_target_properties(${NAME} PROPERTIES
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/iOS/Info.plist"
            XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${NRISAMPLES_IOS_BUNDLE_ID_PREFIX}.${NAME}"
            XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${NRISAMPLES_IOS_DEVELOPMENT_TEAM}"
            XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
            XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
            XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks"
        )

        # NRI links the Metal Shader Converter dynamically, so the library must travel with the app
        if(NRI_ENABLE_METAL_SHADER_CONVERTER)
            set_target_properties(${NAME} PROPERTIES
                XCODE_EMBED_FRAMEWORKS "${NRISAMPLES_IOS_MSC_DYLIB}"
                XCODE_EMBED_FRAMEWORKS_CODE_SIGN_ON_COPY ON
            )
        endif()

        foreach(SHADER IN LISTS IOS_SHADERS_${NAME})
            set(FILE "${SHADER_OUTPUT_PATH}/${SHADER}.dxil")
            if(NOT EXISTS "${FILE}")
                message(FATAL_ERROR "iOS: '${FILE}' not found, build the samples for macOS first (it generates the shaders)")
            endif()

            target_sources(${NAME} PRIVATE "${FILE}")
            set_source_files_properties("${FILE}" PROPERTIES MACOSX_PACKAGE_LOCATION "_Shaders")
        endforeach()

        foreach(TEXTURE IN LISTS IOS_TEXTURES_${NAME})
            set(FILE "${CMAKE_CURRENT_SOURCE_DIR}/_Data/Textures/${TEXTURE}")
            if(NOT EXISTS "${FILE}")
                message(FATAL_ERROR "iOS: '${FILE}' not found, the assets are downloaded by a macOS configure")
            endif()

            target_sources(${NAME} PRIVATE "${FILE}")
            set_source_files_properties("${FILE}" PROPERTIES MACOSX_PACKAGE_LOCATION "_Data/Textures")
        endforeach()
    endif()

    # Copy arguments for Visual Studio Smart Command Line Arguments extension
    if(WIN32 AND MSVC)
        configure_file(.args "${CMAKE_BINARY_DIR}/${NAME}.args.json" COPYONLY)
    endif()
endfunction()

add_sample(AsyncCompute cpp)
add_sample(Buffers c)
add_sample(BindlessSceneViewer cpp)
add_sample(Clear cpp)
add_sample(ClearStorage c)
add_sample(CopyTests cpp)
add_sample(DescriptorPoolIndexing cpp)
add_sample(DescriptorHeapIndexing cpp)
add_sample(DescriptorManagement cpp)
add_sample(DedicatedQueues cpp)
add_sample(DeviceInfo c)
add_sample(GraphicsPipelineStates cpp)
add_sample(GraphicsVertexInput cpp)
add_sample(InputAttachment cpp)
add_sample(IndirectCommands cpp)
add_sample(LowLatency cpp)
add_sample(MemoryAliasing cpp)
add_sample(MeshShader cpp)
add_sample(Multisample cpp)
add_sample(MultiThreading cpp)
add_sample(Multiview cpp)
add_sample(Queries cpp)
add_sample(RayTracingAdvanced cpp)
add_sample(RayTracingTriangle cpp)
add_sample(RayTracingBoxes cpp)
add_sample(Readback cpp)
add_sample(Resize cpp)
add_sample(ResolveSampleZero cpp)
add_sample(Resources c)
add_sample(SceneViewer cpp)
add_sample(Streamer cpp)
add_sample(TextureTypes cpp)
add_sample(Triangle cpp)

if(NOT IOS)
    add_sample(VideoEncodeDecode cpp)
    target_sources(VideoEncodeDecode PRIVATE
        Source/VideoEncodeDecode/Decoder.cpp
        Source/VideoEncodeDecode/Decoder.h
        Source/VideoEncodeDecode/Encoder.cpp
        Source/VideoEncodeDecode/Encoder.h
        Source/VideoEncodeDecode/Shared.h
        Source/VideoEncodeDecode/Stream.cpp
        Source/VideoEncodeDecode/Stream.h
    )
    option(NRISAMPLES_ENABLE_AV1_FILE_DECODING "Enable developer AV1 file decoding using NRI's private parser" OFF)
    target_compile_definitions(VideoEncodeDecode PRIVATE NRISAMPLES_ENABLE_AV1_FILE_DECODING=$<BOOL:${NRISAMPLES_ENABLE_AV1_FILE_DECODING}>)
    if(NRISAMPLES_ENABLE_AV1_FILE_DECODING)
        target_include_directories(VideoEncodeDecode PRIVATE "${NRI_SOURCE_DIR}/Source/Shared")
    endif()

    if(WIN32)
        target_link_libraries(VideoEncodeDecode PRIVATE d3d12)
    endif()

endif()

# Wrapper depends on Vulkan SDK availability
if(DEFINED ENV{VULKAN_SDK} AND NOT IOS)
    add_sample(Wrapper cpp)
    target_include_directories(Wrapper PRIVATE "$ENV{VULKAN_SDK}/include")

    if(WIN32)
        target_link_libraries(Wrapper PRIVATE d3d11 d3d12)
    endif()
endif()
