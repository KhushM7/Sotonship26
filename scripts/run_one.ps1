# Full pipeline for one run: train -> calibrate -> evaluate (val). Idempotent: rerun to resume.
#   pwsh -NoProfile -File scripts\run_one.ps1 -Run e5_gru_s1 -TrainArgs "--model gru --seed 1 ..." [-Threads 2] [-Workers 3]
param(
    [Parameter(Mandatory)][string]$Run,
    [string]$TrainArgs = "",
    [int]$Threads = 2,
    [int]$Workers = 3
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Logs = Join-Path $Repo "logs"
$ModelDir = Join-Path $Repo "models\$Run"
$Logits = Join-Path $ModelDir "val_logits.npy"
$Thresholds = Join-Path $ModelDir "thresholds.yaml"
$EvalJson = Join-Path $Repo "reports\eval\eval_${Run}_val.json"
New-Item -ItemType Directory -Force $Logs | Out-Null

# OneDrive can hold a lock on a file for a moment: retry a write once after 5 s
function Write-Retry([string]$Path, [string]$Text) {
    try { Set-Content -Path $Path -Value $Text -NoNewline }
    catch { Start-Sleep -Seconds 5; Set-Content -Path $Path -Value $Text -NoNewline }
}

function Set-Threads([int]$N) {
    $env:OMP_NUM_THREADS = "$N"
    $env:MKL_NUM_THREADS = "$N"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:NUMEXPR_NUM_THREADS = "1"
    $env:PYTHONUNBUFFERED = "1"
}

function Invoke-Step([string]$Step, [string]$PyArgs) {
    Write-Retry (Join-Path $Logs "$Run.stage") $Step
    $log = Join-Path $Logs "$Run.$Step.log"
    # cmd merges stdout and stderr into one plain-text log
    cmd /c "`"$Py`" $PyArgs > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Retry (Join-Path $Logs "$Run.failed") "$Step exit $code"
        Write-Retry (Join-Path $Logs "$Run.stage") "failed"
        exit $code
    }
}

function Test-Newer([string]$A, [string]$B) {
    (Test-Path $A) -and (Test-Path $B) -and ((Get-Item $A).LastWriteTimeUtc -gt (Get-Item $B).LastWriteTimeUtc)
}

Remove-Item (Join-Path $Logs "$Run.failed"), (Join-Path $Logs "$Run.done") -ErrorAction SilentlyContinue

# 1. train
if (-not (Test-Path $Logits)) {
    if (Test-Path $ModelDir) {
        $partial = Join-Path $Repo "models\_partial"
        New-Item -ItemType Directory -Force $partial | Out-Null
        Move-Item $ModelDir (Join-Path $partial "${Run}_$(Get-Date -Format yyyyMMdd-HHmmss)")
    }
    if (-not $TrainArgs) {
        Write-Retry (Join-Path $Logs "$Run.failed") "train exit -1 (no val_logits.npy and no train args)"
        exit 1
    }
    Set-Threads $Threads
    Invoke-Step "train" "scripts\06_train.py $TrainArgs --threads $Threads --workers $Workers --out models\$Run"
}

# calibration and evaluation are mostly single-threaded bootstrap work
Set-Threads 1

# 2. calibrate
if (-not (Test-Newer $Thresholds $Logits)) {
    Invoke-Step "calib" "scripts\07_calibrate.py --run $Run"
}

# 3. evaluate
if (-not (Test-Newer $EvalJson $Thresholds)) {
    Invoke-Step "eval" "scripts\08_evaluate.py --model $Run --split val"
}

Write-Retry (Join-Path $Logs "$Run.stage") "done"
Write-Retry (Join-Path $Logs "$Run.done") (Get-Date -Format o)
exit 0
