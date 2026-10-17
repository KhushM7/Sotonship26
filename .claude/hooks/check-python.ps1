# PostToolUse hook: after Claude writes or edits a .py file, format and lint it with
# the project's ruff settings (pyproject.toml). Exit code 2 sends any lint problems
# back to Claude so it fixes them before moving on.

$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$file = $payload.tool_input.file_path
if (-not $file -or -not $file.EndsWith(".py") -or -not (Test-Path $file)) { exit 0 }

$repo = $env:CLAUDE_PROJECT_DIR
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Push-Location $repo
try {
    & $py -m ruff format --quiet $file | Out-Null
    $lint = & $py -m ruff check --output-format concise $file 2>&1
    $failed = $LASTEXITCODE -ne 0
}
finally {
    Pop-Location
}

if ($failed) {
    [Console]::Error.WriteLine("ruff check failed for ${file}:`n" + ($lint -join "`n"))
    exit 2
}
exit 0
