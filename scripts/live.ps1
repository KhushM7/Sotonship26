# Live fall detection demo on the webcam: the skeleton on black (never the camera image)
# with the alert banner (OK / CHECK / FALL / ON FLOOR), using the frozen model.
#
#   .\scripts\live.ps1                  # webcam 0
#   .\scripts\live.ps1 -Camera 1        # another camera (.\scripts\live.ps1 -List shows them)
#   .\scripts\live.ps1 -Lite            # lighter pose model, if under 10 fps
#   .\scripts\live.ps1 -Record          # also save the session for the wall (FALL_DATA or data\recordings)
#   .\scripts\live.ps1 -Video clip.mp4  # replay a video file instead of the webcam
#
# Keys in the window: q or Esc quits, r resets the alert state. Ctrl+C here also stops it.
param(
    [int]$Camera = 0,
    [string]$Run = "e9_ens_gru",
    [double]$TLow = 0.6,  # the demo setting (thresholds.yaml has 0.3: more CHECKs)
    [string]$Video,
    [switch]$Lite,
    [switch]$Record,
    [switch]$List
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# -Record saves to $env:FALL_DATA\recordings when it is set, else data\recordings here
$env:PYTHONUNBUFFERED = "1"

$python = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "No .venv: see 'Run the live webcam demo' in the README" }
$pose = if ($Lite) { "lite" } else { "full" }  # downloaded by 09_live.py on first use
if (-not (Test-Path "$root\models\$Run\thresholds.yaml")) {
    throw "models\$Run has no thresholds.yaml"
}

if ($List) {
    & $python scripts\09_live.py --list-cameras
    exit $LASTEXITCODE
}

$liveArgs = @("--run", $Run, "--pose", $pose, "--t-low", $TLow)
if ($Video) { $liveArgs += @("--video", $Video) } else { $liveArgs += @("--camera", $Camera) }
if ($Record) { $liveArgs += "--record" }

Write-Host "Keys: q / Esc quit, r reset alerts"
& $python scripts\09_live.py @liveArgs
exit $LASTEXITCODE
