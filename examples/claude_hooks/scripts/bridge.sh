#!/bin/bash
# Bridge script for Claude Code -> BBX
# Usage: bridge.sh [event_name]

# This script pipes stdin (JSON payload from Claude) to 'bbx hook'.

# Determine the command to run BBX
# If 'bbx' is in PATH, use it.
# Otherwise, try to find cli.py in the project root.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

if command -v bbx &> /dev/null; then
    CMD="bbx"
elif [ -f "$PROJECT_ROOT/cli.py" ]; then
    # Assume python is available
    CMD="python $PROJECT_ROOT/cli.py"
else
    echo "Error: Could not find 'bbx' command or 'cli.py'" >&2
    exit 1
fi

# Run the hook command
# We pass the event name as the first argument
# The JSON payload is passed via stdin
$CMD hook "$1"
