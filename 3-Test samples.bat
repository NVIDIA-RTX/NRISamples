@echo off

set DIR_DATA=_Data
set DIR_BIN=_Bin\Release

if not exist "%DIR_BIN%" (
    set DIR_BIN=_Bin\Debug
)

if not exist "%DIR_BIN%" (
    echo The project is not compiled!
    pause
    exit /b
)
echo Running samples from '%DIR_BIN%'...
echo.

:: API independent samples
"%DIR_BIN%\DeviceInfo.exe"
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

:: API dependent samples
call :TestSample AsyncCompute 100
call :TestSample BindlessSceneViewer 1000
call :TestSample Clear 1000
call :TestSample LowLatency 100
call :TestSample MultiThreading 50
call :TestSample Multiview 1000
call :TestSample Multisample 1000
call :TestSample RayTracingBoxes 1000
call :TestSample RayTracingTriangle 1000
call :TestSample Readback 1000
call :TestSample Resize 25000
call :TestSample SceneViewer 1000
call :TestSample Triangle 1000
call :TestSample Wrapper 1000

pause

exit /b

::========================================================================================
:TestSample

echo %1 [D3D11]
"%DIR_BIN%\%1.exe" --api=D3D11 --frameNum=%2 --debugAPI --debugNRI
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

echo %1 [D3D12]
"%DIR_BIN%\%1.exe" --api=D3D12 --frameNum=%2 --debugAPI --debugNRI
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

echo %1 [VULKAN]
"%DIR_BIN%\%1.exe" --api=VULKAN --frameNum=%2 --debugAPI --debugNRI
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

exit /b
