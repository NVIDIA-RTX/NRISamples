@echo off

set DIR_DATA=_Data
set DIR_BIN=_Bin\Release
set ADAPTER=0

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
call :TestSample AsyncCompute
call :TestSample BindlessSceneViewer
call :TestSample Buffers
call :TestSample LowLatency
call :TestSample MultiThreading
call :TestSample Multiview
call :TestSample Multisample
call :TestSample RayTracingBoxes
call :TestSample RayTracingTriangle
call :TestSample Readback
call :TestSample Resize
call :TestSample SceneViewer
call :TestSample Triangle
call :TestSample Wrapper

pause

exit /b

::========================================================================================
:TestSample

echo %1 [D3D11]
"%DIR_BIN%\%1.exe" --api=D3D11 --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER%
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

echo %1 [D3D12]
"%DIR_BIN%\%1.exe" --api=D3D12 --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER%
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

echo %1 [VULKAN]
"%DIR_BIN%\%1.exe" --api=VULKAN --timeLimit=3 --debugAPI --debugNRI --adapter=%ADAPTER%
if %ERRORLEVEL% equ 0 (
    echo =^> OK
) else (
    echo =^> FAILED!
)
echo.

exit /b
