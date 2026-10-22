@echo off
setlocal
for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"

pushd "%ROOT%" || exit /B 1

set DIR_DATA=_Data
set DIR_BIN=_Bin\Release
set ADAPTER=0
set FAIL=0

if not exist "%DIR_BIN%" (
    set DIR_BIN=_Bin\Debug
)

if not exist "%DIR_BIN%" (
    echo The project is not compiled!
    pause
    popd
    exit /b 1
)
echo Running samples from '%DIR_BIN%'...
echo.

:: API independent samples

"%DIR_BIN%\DeviceInfo.exe"
call :ReportResult %ERRORLEVEL%

:: API dependent samples

call :TestSample AsyncCompute
call :TestSample BindlessSceneViewer
call :TestSample Buffers
call :TestSample Clear
call :TestSample ClearStorage
call :TestSample CopyTests
call :TestSample DescriptorPoolIndexing
call :TestSample DescriptorHeapIndexing
call :TestSample DescriptorManagement
call :TestSample DedicatedQueues
call :TestSample GraphicsPipelineStates
call :TestSample GraphicsVertexInput
call :TestSample InputAttachment
call :TestSample IndirectCommands
call :TestSample LowLatency
call :TestSample MemoryAliasing
call :TestSample MeshShader
call :TestSample Multisample
call :TestSample MultiThreading
call :TestSample Multiview
call :TestSample Queries
call :TestSample RayTracingAdvanced
call :TestSample RayTracingBoxes
call :TestSample RayTracingTriangle
call :TestSample Readback
call :TestSample Resize
call :TestSample ResolveSampleZero
call :TestSample Resources
call :TestSample SceneViewer
call :TestSample Streamer
call :TestSample TextureTypes
call :TestSample Triangle
call :TestSample VideoEncodeDecode "--stream --frames=2 --videoWidth=640 --videoHeight=360"
call :TestSample Wrapper

pause

popd
exit /b %FAIL%

::========================================================================================
:TestSample

echo %1 [D3D11]
"%DIR_BIN%\%1.exe" --api=D3D11 --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER% %~2
call :ReportResult %ERRORLEVEL%

echo %1 [D3D12]
"%DIR_BIN%\%1.exe" --api=D3D12 --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER% %~2
call :ReportResult %ERRORLEVEL%

echo %1 [VULKAN]
"%DIR_BIN%\%1.exe" --api=VULKAN --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER% %~2
call :ReportResult %ERRORLEVEL%

echo %1 [WGPU]
"%DIR_BIN%\%1.exe" --api=WGPU --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER% %~2
call :ReportResult %ERRORLEVEL%

exit /b

::========================================================================================
:ReportResult

if "%~1"=="0" (
    echo =^> OK
) else if "%~1"=="2" (
    echo =^> SKIP
) else (
    echo =^> FAIL
    set FAIL=1
)
echo.
exit /b 0
