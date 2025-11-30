# Claude Code Hooks - Setup Guide

## Quick Start

### 1. Verify BBX Installation

```bash
# Check if bbx command is available
bbx --version

# Or use python directly
python cli.py --version
```

### 2. Test Hook Integration

**Windows (PowerShell):**
```powershell
Get-Content examples\claude_hooks\test_deny.json | python cli.py hook PreToolUse --workflow examples\claude_hooks\workflows\PreToolUse.yaml
```

**Linux/macOS:**
```bash
cat examples/claude_hooks/test_deny.json | python cli.py hook PreToolUse --workflow examples/claude_hooks/workflows/PreToolUse.yaml
```

Expected output:
```json
{"decision": "deny", "reason": "Rm tool is disabled by BBX policy", ...}
```

### 3. Configure Claude Code

Create or edit `~/.claude/settings.json` (or `.claude/settings.json` in your project):

**Windows:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "powershell.exe -File C:\\absolute\\path\\to\\examples\\claude_hooks\\scripts\\bridge.ps1 PreToolUse"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "powershell.exe -File C:\\absolute\\path\\to\\examples\\claude_hooks\\scripts\\bridge.ps1 PostToolUse"
          }
        ]
      }
    ]
  }
}
```

**Linux/macOS:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/examples/claude_hooks/scripts/bridge.sh PreToolUse"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/examples/claude_hooks/scripts/bridge.sh PostToolUse"
          }
        ]
      }
    ]
  }
}
```

### 4. Make Bridge Script Executable (Linux/macOS only)

```bash
chmod +x examples/claude_hooks/scripts/bridge.sh
```

## Creating Custom Hook Workflows

1. Create a new YAML file in `examples/claude_hooks/workflows/` or `hooks/`
2. Name it after the hook event (e.g., `PostToolUse.yaml`, `UserPromptSubmit.yaml`)
3. Use the PreToolUse.yaml as a template

Example structure:
```yaml
name: My Custom Hook
id: my_custom_hook
description: Description of what this hook does

inputs:
  event:
    description: The hook event data

steps:
  - id: process_event
    adapter: python
    method: script
    inputs:
      script: |
        import json
        inputs = variables.get('inputs', {})
        event = inputs.get('event', {})
        
        # Your custom logic here
        result = {
            "decision": "allow",  # or "deny"
            "reason": "Your reason here"
        }
        
        print(json.dumps(result))
```

## Available Hook Events

According to Claude Code documentation:

### Decision Hooks (return allow/deny)
- `PreToolUse` - Before a tool is used
- `UserPromptSubmit` - Before user prompt is submitted

### Observability Hooks (no decision required)
- `PostToolUse` - After a tool is used
- `UserPromptResponse` - After Claude responds

## Hook Response Format

Decision hooks must return:
```json
{
  "decision": "allow" | "deny",
  "reason": "Human-readable explanation",
  "hookSpecificOutput": {
    // Optional hook-specific fields
  }
}
```

## Troubleshooting

### Hook not triggering
1. Check Claude Code logs
2. Verify bridge script path is absolute
3. Ensure bridge script is executable (Linux/macOS)
4. Test hook command manually

### Wrong decision returned
1. Check workflow logic
2. Verify JSON output in workflow
3. Add logging steps to debug

### Error in workflow execution
1. Check BBX logs with `-v` flag
2. Verify workflow syntax
3. Test workflow independently: `bbx run examples/claude_hooks/workflows/PreToolUse.yaml`

## Examples

Test payloads are provided:
- `test_payload.json` - Tests Rm tool (should deny)
- `test_allow.json` - Tests Read tool (should allow)

Test them manually:
```bash
cat test_payload.json | python cli.py hook PreToolUse --workflow examples/claude_hooks/workflows/PreToolUse.yaml
```
