# BBX Troubleshooting Guide

Common issues and solutions.

---

## Installation Issues

### Python version error

```
Error: Python 3.8+ required
```

**Solution:**
```bash
python --version  # Check version
# Install Python 3.9+ from python.org
```

### Missing dependencies

```
ModuleNotFoundError: No module named 'click'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

## Docker Issues

### Docker daemon not running

```
Error: Cannot connect to the Docker daemon
```

**Solutions:**

1. **Windows/Mac:** Start Docker Desktop
2. **Linux:**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```
3. **Permission issues (Linux):**
   ```bash
   sudo usermod -aG docker $USER
   # Logout and login again
   ```

### Docker pull timeout

```
Error: Failed to pull image alpine:latest
```

**Solutions:**
```bash
# Pre-pull images
docker pull alpine:latest
docker pull python:3.11

# Check network
ping docker.io
```

---

## Workflow Execution Issues

### Step output not found

```
Error: Cannot resolve ${steps.my_step.output}
```

**Causes:**
1. Step hasn't run yet - add `depends_on`
2. Typo in step name
3. Step failed

**Solution:**
```yaml
steps:
  my_step:
    use: http.get
    args:
      url: "https://api.example.com"

  next_step:
    use: logger.info
    args:
      message: "${steps.my_step.output}"
    depends_on: [my_step]  # Ensure my_step runs first
```

### Adapter not found

```
Error: Adapter 'xxx' not found
```

**Solution:** Check available adapters:
```bash
python cli.py adapters
```

Common adapters:
- `logger` (not `log`)
- `system` (not `shell`)
- `http` (not `request`)

---

## MCP Server Issues

### MCP tools not appearing in Claude Code

**Solution:**
1. Verify MCP server is configured:
   ```bash
   cat ~/.claude/mcp.json
   ```
2. Restart Claude Code session:
   ```bash
   # Exit current session
   # Run: claude
   ```
3. Check connection:
   ```bash
   claude mcp list
   ```

### MCP server connection error

```
Error: Failed to connect to MCP server
```

**Solutions:**
1. Check Python path in mcp.json
2. Verify PYTHONPATH includes project root
3. Test server manually:
   ```bash
   python -m blackbox.mcp.server
   ```

---

## A2A Protocol Issues

### Agent not responding

```
Error: Connection refused to http://localhost:8001
```

**Solutions:**
1. Start the agent:
   ```bash
   python examples/a2a/demo_agents.py analyst
   ```
2. Check port is not in use:
   ```bash
   netstat -an | grep 8001
   ```

### Agent Card not found

```
Error: 404 at /.well-known/agent-card.json
```

**Solution:** Verify agent is A2A compliant. BBX agents serve card at:
```
http://localhost:PORT/.well-known/agent-card.json
```

---

## State Issues

### State not persisting

**Solution:** Ensure workspace is active:
```bash
python cli.py workspace info
# If no workspace, create one:
python cli.py workspace create my-project
python cli.py workspace set ~/.bbx/workspaces/my_project
```

State is stored in workspace `state/` directory.

---

## Getting Help

1. Check logs with verbose mode:
   ```bash
   python cli.py run workflow.bbx -v
   ```

2. Validate workflow syntax:
   ```bash
   python cli.py validate workflow.bbx
   ```

3. File an issue: https://github.com/kurokie1337/bbx/issues

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk**
