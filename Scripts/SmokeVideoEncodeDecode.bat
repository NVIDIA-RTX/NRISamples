@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "DURATION_SEC=6"
set "WINDOW_WIDTH=3840"
set "WINDOW_HEIGHT=2160"
set "VIDEO_WIDTH=1280"
set "VIDEO_HEIGHT=720"
set "APIS=D3D12"
set "CASE_FILTER="
set "NO_BUILD=0"
set "INCLUDE_DEBUG=0"
set "INCLUDE_VULKAN=0"

call :ParseArgs %*

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "EXE_PATH=%REPO_ROOT%\_Bin\Release\VideoEncodeDecode.exe"

set "TS=%DATE%_%TIME%"
set "TS=%TS:/=%"
set "TS=%TS:.=%"
set "TS=%TS::=%"
set "TS=%TS:,=%"
set "TS=%TS: =0%"
set "TIMESTAMP=%TS%"
set "OUT_DIR=%REPO_ROOT%\_Smoke\VideoEncodeDecode_%TIMESTAMP%"
set "LOG_PATH=%OUT_DIR%\VideoEncodeDecodeSmoke.log"
mkdir "%OUT_DIR%" >nul 2>nul

if "%NO_BUILD%"=="0" (
    call :WriteLog "Building VideoEncodeDecode Release target..."
    cmake --build "%REPO_ROOT%\_Build" --config Release --target VideoEncodeDecode > "%OUT_DIR%\build.tmp.log" 2> "%OUT_DIR%\build.tmp.err"
    set "BUILD_EXIT=!ERRORLEVEL!"
    type "%OUT_DIR%\build.tmp.log" >> "%LOG_PATH%" 2>nul
    type "%OUT_DIR%\build.tmp.err" >> "%LOG_PATH%" 2>nul
    del "%OUT_DIR%\build.tmp.log" "%OUT_DIR%\build.tmp.err" >nul 2>nul
    if not "!BUILD_EXIT!"=="0" (
        call :WriteLog "BUILD FAILED: exit code !BUILD_EXIT!"
        exit /b !BUILD_EXIT!
    )
)

if not exist "%EXE_PATH%" (
    call :WriteLog "Executable not found: %EXE_PATH%"
    exit /b 1
)

if "%INCLUDE_VULKAN%"=="1" call :AddApi VULKAN

set "CASE_COUNT=0"
set "PASS_COUNT=0"
set "FAIL_COUNT=0"

call :WriteLog "VideoEncodeDecode smoke started"
echo %DATE% %TIME% Output directory: %OUT_DIR%
>> "%LOG_PATH%" echo %DATE% %TIME% Output directory: %OUT_DIR%
call :WriteLog "Duration per run: %DURATION_SEC% seconds"
call :WriteLog "Window: %WINDOW_WIDTH%x%WINDOW_HEIGHT%, video: %VIDEO_WIDTH%x%VIDEO_HEIGHT%"
echo %DATE% %TIME% APIs: %APIS%
>> "%LOG_PATH%" echo %DATE% %TIME% APIs: %APIS%

for %%A in (%APIS%) do (
    for %%C in (H264 H265) do (
        for %%F in (IDR P B) do (
            call :MaybeRunCase "%%A-%%C-%%F-cqp" %%A %%C h26Frame %%F cqp 0
            call :MaybeRunCase "%%A-%%C-%%F-lossless" %%A %%C h26Frame %%F lossless 1
        )
    )

    for %%F in (IDR P) do (
        call :MaybeRunCase "%%A-AV1-%%F-cqp" %%A AV1 av1Frame %%F cqp 0
        call :MaybeRunCase "%%A-AV1-%%F-near-lossless" %%A AV1 av1Frame %%F near-lossless 1
    )
)

if not "%CASE_FILTER%"=="" (
    call :WriteLog "Case filter: %CASE_FILTER%"
    call :WriteLog "Filtered cases: %CASE_COUNT%"
    if "%CASE_COUNT%"=="0" (
        call :WriteLog "No cases matched filter"
        exit /b 1
    )
)

call :WriteLog ""
call :WriteLog "VideoEncodeDecode smoke finished: %PASS_COUNT% passed, %FAIL_COUNT% failed"
call :WriteLog "Log: %LOG_PATH%"

if not "%FAIL_COUNT%"=="0" exit /b 1
exit /b 0

::========================================================================================
:ParseArgs
if "%~1"=="" exit /b 0
if /I "%~1"=="--videoWidth" set "VIDEO_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "%~1"=="--videoHeight" set "VIDEO_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
set "ARG=%~1"

if /I "!ARG!"=="-NoBuild" set "NO_BUILD=1" & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--no-build" set "NO_BUILD=1" & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-IncludeVulkan" set "INCLUDE_VULKAN=1" & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--include-vulkan" set "INCLUDE_VULKAN=1" & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-IncludeDebug" set "INCLUDE_DEBUG=1" & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--include-debug" set "INCLUDE_DEBUG=1" & shift /1 & goto :ParseArgs

if /I "!ARG!"=="-DurationSec" set "DURATION_SEC=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--duration" set "DURATION_SEC=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--durationSec" set "DURATION_SEC=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-WindowWidth" set "WINDOW_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--width" set "WINDOW_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--windowWidth" set "WINDOW_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-WindowHeight" set "WINDOW_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--height" set "WINDOW_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--windowHeight" set "WINDOW_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-VideoWidth" set "VIDEO_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--videoWidth" set "VIDEO_WIDTH=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-VideoHeight" set "VIDEO_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--videoHeight" set "VIDEO_HEIGHT=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-Apis" set "APIS=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--apis" set "APIS=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="-CaseFilter" set "CASE_FILTER=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--filter" set "CASE_FILTER=%~2" & shift /1 & shift /1 & goto :ParseArgs
if /I "!ARG!"=="--caseFilter" set "CASE_FILTER=%~2" & shift /1 & shift /1 & goto :ParseArgs

call :ParseEquals "%ARG%"
shift /1
goto :ParseArgs

::========================================================================================
:ParseEquals
set "ARG=%~1"
for /f "tokens=1,* delims==" %%K in ("!ARG!") do (
    set "KEY=%%K"
    set "VALUE=%%L"
)
if /I "!KEY!"=="--duration" set "DURATION_SEC=!VALUE!"
if /I "!KEY!"=="--durationSec" set "DURATION_SEC=!VALUE!"
if /I "!KEY!"=="-DurationSec" set "DURATION_SEC=!VALUE!"
if /I "!KEY!"=="--width" set "WINDOW_WIDTH=!VALUE!"
if /I "!KEY!"=="--windowWidth" set "WINDOW_WIDTH=!VALUE!"
if /I "!KEY!"=="-WindowWidth" set "WINDOW_WIDTH=!VALUE!"
if /I "!KEY!"=="--height" set "WINDOW_HEIGHT=!VALUE!"
if /I "!KEY!"=="--windowHeight" set "WINDOW_HEIGHT=!VALUE!"
if /I "!KEY!"=="-WindowHeight" set "WINDOW_HEIGHT=!VALUE!"
if /I "!KEY!"=="--videoWidth" set "VIDEO_WIDTH=!VALUE!"
if /I "!KEY!"=="-VideoWidth" set "VIDEO_WIDTH=!VALUE!"
if /I "!KEY!"=="--videoHeight" set "VIDEO_HEIGHT=!VALUE!"
if /I "!KEY!"=="-VideoHeight" set "VIDEO_HEIGHT=!VALUE!"
if /I "!KEY!"=="--apis" set "APIS=!VALUE!"
if /I "!KEY!"=="-Apis" set "APIS=!VALUE!"
if /I "!KEY!"=="--filter" set "CASE_FILTER=!VALUE!"
if /I "!KEY!"=="--caseFilter" set "CASE_FILTER=!VALUE!"
if /I "!KEY!"=="-CaseFilter" set "CASE_FILTER=!VALUE!"
exit /b 0

::========================================================================================
:AddApi
echo %APIS% | findstr /I /C:"%~1" >nul
if errorlevel 1 set "APIS=%APIS% %~1"
exit /b 0

::========================================================================================
:WriteLog
echo %DATE% %TIME% %~1
>> "%LOG_PATH%" echo %DATE% %TIME% %~1
exit /b 0

::========================================================================================
:MaybeRunCase
set "CASE_NAME=%~1"
if not "%CASE_FILTER%"=="" (
    echo %CASE_NAME% | findstr /I /R /C:"%CASE_FILTER%" >nul
    if errorlevel 1 exit /b 0
)

call :RunCase %*
exit /b 0

::========================================================================================
:RunCase
set "CASE_NAME=%~1"
set "API=%~2"
set "CODEC=%~3"
set "FRAME_ARG_NAME=%~4"
set "FRAME_ARG_VALUE=%~5"
set "SUFFIX=%~6"
set "LOSSLESS=%~7"

set /a CASE_COUNT+=1
set "CASE_OUT=%OUT_DIR%\%CASE_NAME%.stdout.tmp"
set "CASE_ERR=%OUT_DIR%\%CASE_NAME%.stderr.tmp"

set "CASE_ARGS=--api=%API% --width=%WINDOW_WIDTH% --height=%WINDOW_HEIGHT% --videoWidth=%VIDEO_WIDTH% --videoHeight=%VIDEO_HEIGHT% --codec=%CODEC% --timeLimit=%DURATION_SEC% --alwaysActive --requireVideoRoundTrip --%FRAME_ARG_NAME%=%FRAME_ARG_VALUE%"
if "%LOSSLESS%"=="1" set "CASE_ARGS=%CASE_ARGS% --lossless"
if "%INCLUDE_DEBUG%"=="1" set "CASE_ARGS=%CASE_ARGS% --debugAPI --debugNRI"

call :WriteLog ""
call :WriteLog "[%CASE_COUNT%] START %CASE_NAME%"
call :WriteLog "Command: ""%EXE_PATH%"" %CASE_ARGS%"

pushd "%REPO_ROOT%" >nul
"%EXE_PATH%" %CASE_ARGS% > "%CASE_OUT%" 2> "%CASE_ERR%"
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul

call :WriteLog "ExitCode: %EXIT_CODE%"
>> "%LOG_PATH%" echo ----- stdout: %CASE_NAME% -----
type "%CASE_OUT%" >> "%LOG_PATH%" 2>nul
>> "%LOG_PATH%" echo ----- stderr: %CASE_NAME% -----
type "%CASE_ERR%" >> "%LOG_PATH%" 2>nul

set "CASE_FAILED=0"
if not "%EXIT_CODE%"=="0" set "CASE_FAILED=1"
findstr /R /C:"\<ERROR\>" /C:"\<FAILED\>" "%CASE_OUT%" "%CASE_ERR%" >nul 2>nul
if not errorlevel 1 set "CASE_FAILED=1"
findstr /C:"VIDEO_ROUND_TRIP_OK" "%CASE_OUT%" >nul 2>nul
if errorlevel 1 set "CASE_FAILED=1"

del "%CASE_OUT%" "%CASE_ERR%" >nul 2>nul

if "%CASE_FAILED%"=="1" (
    set /a FAIL_COUNT+=1
    call :WriteLog "FAILED %CASE_NAME%"
) else (
    set /a PASS_COUNT+=1
    call :WriteLog "OK %CASE_NAME%"
)

exit /b 0
