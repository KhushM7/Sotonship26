# PostToolUse hook: after Claude writes or edits a .py file, format it, lint it and
# type-check it with the project's settings (pyproject.toml). Exit code 2 sends the
# problems back to Claude so it fixes them before moving on.

$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$file = $payload.tool_input.file_path
if (-not $file -or -not $file.EndsWith(".py") -or -not (Test-Path $file)) { exit 0 }

$repo = $env:CLAUDE_PROJECT_DIR
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
if (-not $env:FALL_DATA) { $env:FALL_DATA = "C:\fall-data" }  # config.py reads it at import

Push-Location $repo
try {
    & $py -m ruff format --quiet $file | Out-Null
    $problems = @()
    $lint = & $py -m ruff check --output-format concise $file 2>&1
    if ($LASTEXITCODE -ne 0) { $problems += "ruff check:"; $problems += $lint }
    $types = & $py -m mypy --no-error-summary $file 2>&1
    if ($LASTEXITCODE -ne 0) { $problems += "mypy:"; $problems += $types }
}
finally {
    Pop-Location
}

if ($problems.Count -gt 0) {
    [Console]::Error.WriteLine("Python checks failed for ${file}:`n" + ($problems -join "`n"))
    exit 2
}
exit 0
