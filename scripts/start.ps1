# Start fallsense as deployed: wall, presenter, log and phones at the address in
# configs/server.local.yaml (https://fall-safe.bid through the Cloudflare tunnel).
#
#   .\scripts\start.ps1           # start
#   .\scripts\start.ps1 -Build    # rebuild the web app first (after pulling changes)
#
# Stop with Ctrl+C so the tunnel closes with the server.
param([switch]$Build)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $env:FALL_DATA) { $env:FALL_DATA = "C:\fall-data" }
if (-not $env:FALLSENSE_BACKEND) { $env:FALLSENSE_BACKEND = (Resolve-Path "$root\..\Sotonship26").Path }
$env:PYTHONUNBUFFERED = "1"

if ($Build -or -not (Test-Path "$root\frontend\dist\index.html")) {
    Push-Location "$root\frontend"
    try { npm install --no-fund --no-audit; npm run build } finally { Pop-Location }
}

$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($busy) { throw "Port 8000 is already in use (another fallsense server?). Stop it first." }

& "$root\.venv\Scripts\python.exe" scripts\11_serve.py
