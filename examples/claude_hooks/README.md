# BBX Claude Hooks Integration

BBX can integrate with Claude Code hooks to enforce policies, log actions, and extend agent capabilities.

## Overview

Claude Code supports hooks that execute before/after tool calls. BBX provides a bridge to process these hooks through workflows.

## Setup

See **[SETUP.md](SETUP.md)** for detailed installation instructions.

### Quick Configuration

Add to `~/.claude/settings.json` (or `.claude/settings.json` in your project):

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

**Windows (PowerShell):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "powershell.exe -File C:\\path\\to\\examples\\claude_hooks\\scripts\\bridge.ps1 PreToolUse"
          }
        ]
      }
    ]
  }
}
```

## Workflows

The bridge script calls `bbx hook <event_name>`. BBX looks for a workflow named `<event_name>.yaml` in `examples/claude_hooks/workflows/` (or `hooks/` in your current directory).

You can customize the workflows to implement your own logic.

### Example: PreToolUse.yaml

This workflow checks the tool name and denies usage of the `Rm` tool:

```yaml
name: PreToolUse Hook
steps:
  - id: check_tool
    adapter: python
    method: run
    args:
      code: |
        if inputs['event']['tool_name'] == 'Rm':
            return {"decision": "deny", "reason": "Policy violation"}
        return {"decision": "allow"}

  - id: hook_response
    adapter: transform
    method: map
    args:
      data: "${check_tool.output}"
```

## Available Hook Events

| Hook | Type | Description |
|------|------|-------------|
| `PreToolUse` | Decision | Before a tool is used (can deny) |
| `PostToolUse` | Observability | After a tool is used |
| `UserPromptSubmit` | Decision | Before user prompt is submitted |
| `UserPromptResponse` | Observability | After Claude responds |

## Hook Response Format

Decision hooks must return:

```json
{
  "decision": "allow",
  "reason": "Optional explanation"
}
```

Or to deny:

```json
{
  "decision": "deny",
  "reason": "Why the action was blocked"
}
```

## Files

```
examples/claude_hooks/
├── README.md           # This file
├── SETUP.md            # Detailed setup guide
├── scripts/
│   ├── bridge.sh       # Linux/macOS bridge
│   └── bridge.ps1      # Windows PowerShell bridge
├── workflows/
│   └── PreToolUse.yaml # Example hook workflow
├── test_deny.json      # Test payload (should deny)
└── test_allow.json     # Test payload (should allow)
```

## Testing

```bash
# Test deny scenario
cat test_deny.json | python cli.py hook PreToolUse --workflow workflows/PreToolUse.yaml

# Expected: {"decision": "deny", ...}
```

---

**See also:** [SETUP.md](SETUP.md) for complete installation instructions.
