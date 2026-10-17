# PreToolUse hook: stops Claude from running `git commit` or `git push`
# through the Bash or PowerShell tools. Exit code 2 blocks the call and
# sends the stderr message back to Claude.

$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$command = $payload.tool_input.command
if (-not $command) { exit 0 }

# `git` (or git.exe), any options such as `-C path` or `-c k=v`, then commit/push.
if ($command -match '(?i)\bgit(\.exe)?\b[^\r\n|;&]*?\s(commit|push)\b') {
    [Console]::Error.WriteLine("Blocked by .claude/hooks/block-git-commit-push.ps1: Claude may not run 'git $($Matches[2])'. Ask the user to run it themselves.")
    exit 2
}
exit 0
