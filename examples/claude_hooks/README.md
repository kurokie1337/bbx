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

## Workflows

The `bridge.sh` script calls `bbx hook <event_name>`. The `ClaudeHooksAdapter` in BBX will look for a workflow named `<event_name>.yaml` in `examples/claude_hooks/workflows/` (or `hooks/` in your current directory).

You can customize the workflows to implement your own logic.

### Example: PreToolUse.yaml

This workflow checks the tool name and denies usage of the `Rm` tool.

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
