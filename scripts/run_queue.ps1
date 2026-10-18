# Keep the training slots full from a queue file; exits when every line is .done or .failed.
#   pwsh -NoProfile -File scripts\run_queue.ps1 -Queue logs\queue.txt
# Queue lines: <run>|<train args>[|<threads>|<workers>]. Empty train args = calibrate/evaluate only.
# Re-read every tick: logs\maxjobs.txt (training slots) and logs\launch.txt ("<threads> <workers>",
# the default for new launches). Both can be edited while the runner is going.
param(
    [string]$Queue = "logs\queue.txt",
    [int]$GapSeconds = 60,
    [int]$TickSeconds = 15
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Logs = Join-Path $Repo "logs"
$Pwsh = (Get-Process -Id $PID).Path
$Retriable = "MemoryError|PermissionError|WinError 32|WinError 5\b|being used by another process"

$running = @{}  # run -> @{ Proc; Train }
$fails = @{}    # run -> failure count
$lastTrainStart = [datetime]::MinValue

function Read-Int([string]$Path, [int]$Default) {
    try { [int](Get-Content $Path -Raw).Trim() } catch { $Default }
}

function Read-Queue {
    Get-Content (Join-Path $Repo $Queue) | Where-Object { $_.Trim() -and -not $_.StartsWith("#") } | ForEach-Object {
        $f = $_.Split("|")
        [pscustomobject]@{
            Run = $f[0].Trim()
            Args = if ($f.Count -gt 1) { $f[1].Trim() } else { "" }
            Threads = if ($f.Count -gt 2 -and $f[2].Trim()) { [int]$f[2] } else { $null }
            Workers = if ($f.Count -gt 3 -and $f[3].Trim()) { [int]$f[3] } else { $null }
        }
    }
}

function Get-Stage([string]$Run) {
    try { (Get-Content (Join-Path $Logs "$Run.stage") -Raw).Trim() } catch { "" }
}

function Get-Epoch([string]$Run) {
    $log = Join-Path $Logs "$Run.train.log"
    if (-not (Test-Path $log)) { return "start" }
    $m = Select-String -Path $log -Pattern "^epoch\s+(\d+)" | Select-Object -Last 1
    if ($m) { "ep $($m.Matches[0].Groups[1].Value)" } else { "start" }
}

while ($true) {
    $items = @(Read-Queue)

    # reap finished jobs
    foreach ($run in @($running.Keys)) {
        $p = $running[$run].Proc
        if (-not $p.HasExited) { continue }
        $running.Remove($run)
        $failedFile = Join-Path $Logs "$run.failed"
        if (Test-Path $failedFile) {
            $fails[$run] = 1 + [int]$fails[$run]
            $step = ((Get-Content $failedFile -Raw) -split " ")[0]
            $log = Join-Path $Logs "$run.$step.log"
            $ours = (Test-Path $log) -and (Select-String -Path $log -Pattern $Retriable -Quiet)
            if ($ours -and $fails[$run] -lt 2) {
                Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) retry $run ($(Get-Content $failedFile -Raw))"
                Remove-Item $failedFile
            } else {
                Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) failed $run ($(Get-Content $failedFile -Raw))"
            }
        } elseif (-not (Test-Path (Join-Path $Logs "$run.done"))) {
            # exited without a marker (killed, or crashed in the launcher)
            Set-Content (Join-Path $Logs "$run.failed") "launcher exit $($p.ExitCode)"
            $fails[$run] = 1 + [int]$fails[$run]
            Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) failed $run (launcher exit $($p.ExitCode))"
        } else {
            Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) done $run"
        }
    }

    $nTrain = @($running.Values | Where-Object { $_.Train -and (Get-Stage $_.Run) -in @("", "train") }).Count
    $maxJobs = Read-Int (Join-Path $Logs "maxjobs.txt") 6
    $tw = try { (Get-Content (Join-Path $Logs "launch.txt") -Raw).Trim() -split "\s+" } catch { @() }

    foreach ($it in $items) {
        $run = $it.Run
        if ($running.ContainsKey($run)) { continue }
        if (Test-Path (Join-Path $Logs "$run.done")) { continue }
        if (Test-Path (Join-Path $Logs "$run.failed")) { continue }
        $hasLogits = Test-Path (Join-Path $Repo "models\$run\val_logits.npy")
        $needsSlot = -not $hasLogits
        if ($needsSlot) {
            if ($nTrain -ge $maxJobs) { continue }
            if (((Get-Date) - $lastTrainStart).TotalSeconds -lt $GapSeconds) { continue }
        }
        $threads = if ($it.Threads) { $it.Threads } elseif ($tw.Count -ge 2) { [int]$tw[0] } else { 2 }
        $workers = if ($it.Workers) { $it.Workers } elseif ($tw.Count -ge 2) { [int]$tw[1] } else { 3 }
        Remove-Item (Join-Path $Logs "$run.stage") -ErrorAction SilentlyContinue
        $argList = @("-NoProfile", "-File", "scripts\run_one.ps1", "-Run", $run,
            "-TrainArgs", "`"$($it.Args)`"", "-Threads", "$threads", "-Workers", "$workers")
        $p = Start-Process $Pwsh -ArgumentList $argList -WorkingDirectory $Repo -NoNewWindow -PassThru
        $running[$run] = @{ Proc = $p; Train = $needsSlot; Run = $run }
        Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) start $run pid $($p.Id) threads $threads workers $workers$(if (-not $needsSlot) { ' (calib/eval only)' })"
        if ($needsSlot) { $nTrain++; $lastTrainStart = Get-Date }
    }

    $runningTxt = ($running.Keys | Sort-Object | ForEach-Object {
        $s = Get-Stage $_
        if ($s -in @("", "train")) { "$_(train $(Get-Epoch $_))" } else { "$_($s)" }
    }) -join " "
    $done = ($items | Where-Object { Test-Path (Join-Path $Logs "$($_.Run).done") } | ForEach-Object Run) -join " "
    $failed = ($items | Where-Object { Test-Path (Join-Path $Logs "$($_.Run).failed") } | ForEach-Object Run) -join " "
    $waiting = ($items | Where-Object {
        -not $running.ContainsKey($_.Run) -and -not (Test-Path (Join-Path $Logs "$($_.Run).done")) -and -not (Test-Path (Join-Path $Logs "$($_.Run).failed"))
    } | ForEach-Object Run) -join " "
    $line = "$(Get-Date -Format s) maxjobs $maxJobs | running: $runningTxt | waiting: $waiting | done: $done | failed: $failed"
    try { Set-Content (Join-Path $Logs "queue_status.txt") $line } catch { }

    if ($running.Count -eq 0 -and -not $waiting) {
        Add-Content (Join-Path $Logs "queue_events.log") "$(Get-Date -Format s) queue empty, exiting"
        Write-Output $line
        exit 0
    }
    Start-Sleep -Seconds $TickSeconds
}
