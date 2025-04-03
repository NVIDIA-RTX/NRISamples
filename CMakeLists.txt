cmake_minimum_required(VERSION 3.15)

# Options
option(DISABLE_SHADER_COMPILATION "Disable shader compilation" OFF)
option(USE_MINIMAL_DATA "Use minimal '_Data'(90MB)" ON)
set(SHADER_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/_Shaders")

# Cached
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/_Bin" CACHE STRING "")
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "")

# Create project
project(NRISamples LANGUAGES C CXX)

# Globals?
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)

# Download dependencies using Packman
if(WIN32)
    set(PACKMAN_EXT ".cmd")
else()
    set(PACKMAN_EXT ".sh")
endif()

if((CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    if(WIN32)
        set(PACKMAN_PLATFORM "windows-x86_64")
    else()
        set(PACKMAN_PLATFORM "linux-x86_64")
    endif()
elseif((CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64") OR(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
    set(PACKMAN_PLATFORM "linux-aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: '${CMAKE_SYSTEM_PROCESSOR}'")
endif()

message("Packman platform: ${PACKMAN_PLATFORM}")

if(USE_MINIMAL_DATA)
    set(NRI_DATA_VERSION 2.3_minimal)
    message("Using minimal _Data.")
else()
    set(NRI_DATA_VERSION 2.3)
endif()

execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/External/Packman/packman${PACKMAN_EXT}" pull "${CMAKE_CURRENT_SOURCE_DIR}/Dependencies.xml" -p ${PACKMAN_PLATFORM} -t nri_data_version=${NRI_DATA_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE PACKMAN_RESULT)

if(NOT "${PACKMAN_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "Packman failed(code = ${PACKMAN_RESULT})")
endif()

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILE_OPTIONS -msse4.1 -Wextra)
elseif(MSVC)
    set(COMPILE_OPTIONS /W4 /WX /wd4324)
else()
    message(WARNING "Unknown compiler!")
endif()

# Compile definitions
if(WIN32)
    set(COMPILE_DEFINITIONS WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS _UNICODE UNICODE)
endif()

# External/NRIFramework
set(NRIF_SHADERS_PATH "${SHADER_OUTPUT_PATH}" CACHE STRING "")
add_subdirectory("External/NRIFramework")

# Move dependencies into "External"
function(fix_folders _DIRECTORY _FOLDER)
    get_property(_TARGETS DIRECTORY "${_DIRECTORY}" PROPERTY BUILDSYSTEM_TARGETS)

    foreach(_TARGET IN LISTS _TARGETS)
        get_property(_FOLDER_ORIGINAL TARGET ${_TARGET} PROPERTY FOLDER)
        set_property(TARGET ${_TARGET} PROPERTY FOLDER "${_FOLDER}/${_FOLDER_ORIGINAL}")
    endforeach()

    get_property(SUBDIRS DIRECTORY "${_DIRECTORY}" PROPERTY SUBDIRECTORIES)

    foreach(SUBDIR IN LISTS SUBDIRS)
        fix_folders("${SUBDIR}" "${_FOLDER}")
    endforeach()
endfunction()

fix_folders("External/NRIFramework" "External")

# Shaders
if(NOT DISABLE_SHADER_COMPILATION)
    file(GLOB_RECURSE SHADERS "Shaders/*.hlsl" "Shaders/*.hlsli")
    set_source_files_properties(${SHADERS} PROPERTIES VS_TOOL_OVERRIDE "None")

    get_target_property(ML_SOURCE_DIR MathLib SOURCE_DIR)
    get_target_property(NRI_SOURCE_DIR NRI SOURCE_DIR)

    set(SHADERMAKE_GENERAL_ARGS
        --useAPI --binary --flatten --stripReflection --WX
        --sRegShift 100 --tRegShift 200 --bRegShift 300 --uRegShift 400
        --sourceDir "Shaders"
        -c Shaders.cfg
        -o "${SHADER_OUTPUT_PATH}"
        -I "Shaders"
        -I "${ML_SOURCE_DIR}"
        -I "${NRI_SOURCE_DIR}/Include"
    )

    if(WIN32)
        add_custom_target(${PROJECT_NAME}Shaders ALL
            COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXBC --compiler "${FXC_PATH}"
            COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p DXIL --compiler "${DXC_PATH}"
            COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}"
            DEPENDS ShaderMake
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM
            SOURCES ${SHADERS}
        )
    else()
        add_custom_target(${PROJECT_NAME}Shaders ALL
            COMMAND ShaderMake ${SHADERMAKE_GENERAL_ARGS} -p SPIRV --compiler "${DXC_SPIRV_PATH}"
            DEPENDS ShaderMake
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM
            SOURCES ${SHADERS}
        )
    endif()

    set_property(TARGET ${PROJECT_NAME}Shaders PROPERTY FOLDER ${PROJECT_NAME})
endif()

# Samples
function(add_sample NAME EXT)
    add_executable(${NAME} "Source/${NAME}.${EXT}")
    source_group("" FILES "Source/${NAME}.${EXT}")

    if(NOT DISABLE_SHADER_COMPILATION)
        add_dependencies(${NAME} ${PROJECT_NAME}Shaders)
    endif()

    target_compile_definitions(${NAME} PRIVATE ${COMPILE_DEFINITIONS} PROJECT_NAME=${NAME})
    target_compile_options(${NAME} PRIVATE ${COMPILE_OPTIONS})
    target_link_libraries(${NAME} PRIVATE NRIFramework NRI)

    set_property(TARGET ${NAME} PROPERTY FOLDER ${PROJECT_NAME})
    set_property(TARGET ${NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

    # Copy arguments for Visual Studio Smart Command Line Arguments extension
    if(WIN32 AND MSVC)
        configure_file(.args "${CMAKE_BINARY_DIR}/${NAME}.args.json" COPYONLY)
    endif()
endfunction()

add_sample(AsyncCompute cpp)
add_sample(Clear cpp)
add_sample(CTest c)
add_sample(DeviceInfo cpp)
add_sample(LowLatency cpp)
add_sample(MultiThreading cpp)
add_sample(Multiview cpp)
add_sample(RayTracingTriangle cpp)
add_sample(RayTracingBoxes cpp)
add_sample(Readback cpp)
add_sample(SceneViewer cpp)
add_sample(BindlessSceneViewer cpp)
add_sample(Triangle cpp)
add_sample(Wrapper cpp)
add_sample(Resize cpp)

# Samples - special
if(WIN32)
    target_link_libraries(Wrapper PRIVATE d3d11 d3d12)
endif()

target_include_directories(Wrapper PRIVATE "$ENV{VULKAN_SDK}/include")
