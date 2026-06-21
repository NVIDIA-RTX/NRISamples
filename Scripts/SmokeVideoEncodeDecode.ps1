param(
    [double] $DurationSec = 6.0,
    [int] $WindowWidth = 3840,
    [int] $WindowHeight = 2160,
    [int] $VideoWidth = 1280,
    [int] $VideoHeight = 720,
    [string[]] $Apis = @("D3D12"),
    [string] $CaseFilter = "",
    [int] $ScreenshotDelayMs = 2500,
    [switch] $NoBuild,
    [switch] $NoScreenshots,
    [switch] $IncludeVulkan,
    [switch] $IncludeDebug
)

$ErrorActionPreference = "Stop"

if ($DurationSec -lt 2.0) {
    Write-Warning "DurationSec is below 2 seconds; this is only useful for script debugging, not smoke coverage."
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ExePath = Join-Path $RepoRoot "_Bin\Release\VideoEncodeDecode.exe"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $RepoRoot "_Smoke\VideoEncodeDecode_$Timestamp"
$LogPath = Join-Path $OutDir "VideoEncodeDecodeSmoke.log"

if ($IncludeVulkan -and -not ($Apis -contains "VULKAN")) {
    $Apis += "VULKAN"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Write-Log {
    param([string] $Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

if (-not $NoBuild) {
    Write-Log "Building VideoEncodeDecode Release target..."
    $buildLog = Join-Path $OutDir "build.tmp.log"
    $build = Start-Process -FilePath "cmake" -ArgumentList @("--build", "_Build", "--config", "Release", "--target", "VideoEncodeDecode") -WorkingDirectory $RepoRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $buildLog -RedirectStandardError "$buildLog.err"
    Get-Content $buildLog, "$buildLog.err" -ErrorAction SilentlyContinue | Add-Content $LogPath
    Remove-Item $buildLog, "$buildLog.err" -ErrorAction SilentlyContinue
    if ($build.ExitCode -ne 0) {
        Write-Log "BUILD FAILED: exit code $($build.ExitCode)"
        exit $build.ExitCode
    }
}

if (-not (Test-Path $ExePath)) {
    Write-Log "Executable not found: $ExePath"
    exit 1
}

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class Win32WindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
}
"@

if (-not $NoScreenshots) {
    Add-Type -AssemblyName System.Drawing
}

function Wait-MainWindow {
    param(
        [System.Diagnostics.Process] $Process,
        [int] $TimeoutMs = 8000
    )

    $deadline = [Environment]::TickCount + $TimeoutMs
    while (-not $Process.HasExited -and [Environment]::TickCount -lt $deadline) {
        $Process.Refresh()
        if ($Process.MainWindowHandle -ne [IntPtr]::Zero) {
            return $Process.MainWindowHandle
        }
        Start-Sleep -Milliseconds 100
    }

    return [IntPtr]::Zero
}

function Capture-Window {
    param(
        [IntPtr] $Handle,
        [string] $Path
    )

    if ($NoScreenshots -or $Handle -eq [IntPtr]::Zero) {
        return $false
    }

    Start-Sleep -Milliseconds $ScreenshotDelayMs

    $rect = New-Object Win32WindowCapture+RECT
    if (-not [Win32WindowCapture]::GetWindowRect($Handle, [ref] $rect)) {
        return $false
    }

    $width = $rect.Right - $rect.Left
    $height = $rect.Bottom - $rect.Top
    if ($width -le 0 -or $height -le 0) {
        return $false
    }

    $bitmap = New-Object System.Drawing.Bitmap $width, $height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size)
        $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }

    return $true
}

function New-Case {
    param(
        [string] $Name,
        [string] $Api,
        [string] $Codec,
        [string] $FrameArgName,
        [string] $FrameArgValue,
        [bool] $Lossless
    )

    $args = @(
        "--api=$Api",
        "--width=$WindowWidth",
        "--height=$WindowHeight",
        "--videoWidth=$VideoWidth",
        "--videoHeight=$VideoHeight",
        "--codec=$Codec",
        "--timeLimit=$DurationSec",
        "--alwaysActive"
    )

    if ($FrameArgName) {
        $args += "--$FrameArgName=$FrameArgValue"
    }
    if ($Lossless) {
        $args += "--lossless"
    }
    if ($IncludeDebug) {
        $args += "--debugAPI"
        $args += "--debugNRI"
    }

    [pscustomobject]@{
        Name = $Name
        Args = $args
    }
}

$cases = New-Object System.Collections.Generic.List[object]
foreach ($api in $Apis) {
    foreach ($codec in @("H264", "H265")) {
        $h26Frames = @("IDR", "P", "B")
        if ($api -eq "VULKAN" -and $codec -eq "H265") {
            $h26Frames = @("IDR", "P")
        }
        foreach ($frame in $h26Frames) {
            foreach ($lossless in @($false, $true)) {
                $suffix = if ($lossless) { "lossless" } else { "cqp" }
                $cases.Add((New-Case -Name "$api-$codec-$frame-$suffix" -Api $api -Codec $codec -FrameArgName "h26Frame" -FrameArgValue $frame -Lossless $lossless))
            }
        }
    }

    foreach ($frame in @("IDR", "P")) {
        foreach ($lossless in @($false, $true)) {
            $suffix = if ($lossless) { "near-lossless" } else { "cqp" }
            $cases.Add((New-Case -Name "$api-AV1-$frame-$suffix" -Api $api -Codec "AV1" -FrameArgName "av1Frame" -FrameArgValue $frame -Lossless $lossless))
        }
    }
}

Write-Log "VideoEncodeDecode smoke started"
Write-Log "Output directory: $OutDir"
Write-Log "Duration per run: $DurationSec seconds"
Write-Log "Window: ${WindowWidth}x${WindowHeight}, video: ${VideoWidth}x${VideoHeight}"
Write-Log "APIs: $($Apis -join ', ')"
Write-Log "Cases: $($cases.Count)"

if ($CaseFilter) {
    $cases = @($cases | Where-Object { $_.Name -match $CaseFilter })
    Write-Log "Case filter: $CaseFilter"
    Write-Log "Filtered cases: $($cases.Count)"
    if ($cases.Count -eq 0) {
        Write-Log "No cases matched filter"
        exit 1
    }
}

$failures = 0
$index = 0
foreach ($case in $cases) {
    $index++
    $caseOut = Join-Path $OutDir "$($case.Name).stdout.tmp"
    $caseErr = Join-Path $OutDir "$($case.Name).stderr.tmp"
    $screenshot = Join-Path $OutDir "$($case.Name).png"

    Write-Log ""
    Write-Log "[$index/$($cases.Count)] START $($case.Name)"
    Write-Log "Command: `"$ExePath`" $($case.Args -join ' ')"

    $process = Start-Process -FilePath $ExePath -ArgumentList $case.Args -WorkingDirectory $RepoRoot -PassThru -RedirectStandardOutput $caseOut -RedirectStandardError $caseErr
    $handle = Wait-MainWindow -Process $process

    if ($handle -ne [IntPtr]::Zero) {
        Write-Log "Window handle: $handle"
        [Win32WindowCapture]::SetForegroundWindow($handle) | Out-Null
        if (Capture-Window -Handle $handle -Path $screenshot) {
            Write-Log "Screenshot: $screenshot"
        } elseif (-not $NoScreenshots) {
            Write-Log "Screenshot capture failed"
        }
    } else {
        Write-Log "Window handle not found before timeout"
    }

    $maxWaitSec = [Math]::Ceiling($DurationSec + 12)
    $timedOut = $false
    if (-not $process.WaitForExit($maxWaitSec * 1000)) {
        Write-Log "TIMEOUT: killing process after $maxWaitSec seconds"
        $timedOut = $true
        $process.Kill()
        $process.WaitForExit()
    }
    $process.Refresh()

    $exitCode = $process.ExitCode
    if ($null -eq $exitCode -or "$exitCode" -eq "") {
        $exitCode = 0
    }
    Write-Log "ExitCode: $exitCode"
    Add-Content -Path $LogPath -Value "----- stdout: $($case.Name) -----"
    Get-Content $caseOut -ErrorAction SilentlyContinue | Add-Content $LogPath
    Add-Content -Path $LogPath -Value "----- stderr: $($case.Name) -----"
    Get-Content $caseErr -ErrorAction SilentlyContinue | Add-Content $LogPath
    $caseText = (Get-Content $caseOut, $caseErr -ErrorAction SilentlyContinue) -join "`n"
    Remove-Item $caseOut, $caseErr -ErrorAction SilentlyContinue

    if ($timedOut -or $exitCode -ne 0 -or $caseText -match "(?m)\b(ERROR|FAILED)\b") {
        $failures++
        Write-Log "FAILED $($case.Name)"
    } else {
        Write-Log "OK $($case.Name)"
    }
}

Write-Log ""
Write-Log "VideoEncodeDecode smoke finished: $($cases.Count - $failures) passed, $failures failed"
Write-Log "Log: $LogPath"
if ($failures -ne 0) {
    exit 1
}
