# Bridge script for Claude Code -> BBX (PowerShell version for Windows)
# Usage: bridge.ps1 [event_name]

param(
    [Parameter(Mandatory=$true)]
    [string]$EventName
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $ScriptDir))

# Determine the command to run BBX
$BbxCommand = Get-Command bbx -ErrorAction SilentlyContinue
$CliPath = Join-Path $ProjectRoot "cli.py"

if ($BbxCommand) {
    $Cmd = "bbx"
} elseif (Test-Path $CliPath) {
    $Cmd = "python `"$CliPath`""
} else {
    Write-Error "Could not find 'bbx' command or 'cli.py'"
    exit 1
}

# Run the hook command
# The JSON payload is passed via stdin
if ($BbxCommand) {
    $Input | bbx hook $EventName
} else {
    $Input | python "$CliPath" hook $EventName
}
