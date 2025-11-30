# Simplified Bridge script for Claude Code -> BBX (PowerShell)
# Usage: bridge-simple.ps1 [event_name]

param(
    [Parameter(Mandatory=$true)]
    [string]$EventName
)

# Direct approach - assumes you're running from project root or BBX is in PATH
# Modify the path below to match your installation

# Option 1: If bbx is in PATH
# $Input | bbx hook $EventName

# Option 2: Direct python call (recommended for Windows)
$Input | python cli.py hook $EventName

# Option 3: Use absolute path
# $Input | python "C:\path\to\bbx\cli.py" hook $EventName
