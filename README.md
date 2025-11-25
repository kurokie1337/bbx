# BBX - Operating System for AI Agents

> **The first operating system designed specifically for AI agents.**
> Run workflows, manage processes, persist state - just like Linux, but for AI.

[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP%20Tools-40-brightgreen.svg)](#mcp-server-40-tools)
[![A2A](https://img.shields.io/badge/A2A-v0.3-orange.svg)](#a2a-protocol)

```
BBX - Operating System for AI Agents
Copyright 2025 Ilya Makarov
First Release: November 26, 2025
```

---

## Why BBX?

When AI agents (Claude, GPT, etc.) work on complex tasks, they need infrastructure:

| Problem | Linux Solution | BBX Solution |
|---------|---------------|--------------|
| Isolated environments | `/home/user` | Workspaces |
| Long-running tasks | `command &` | `bbx run --background` |
| Monitor processes | `ps`, `kill` | `bbx ps`, `bbx kill` |
| Persistent memory | `env`, config files | `bbx state` |
| Spawn child processes | `fork()` | Nested workflows |
| Automation | Shell scripts | `.bbx` workflows |

**BBX provides this infrastructure through 40 MCP tools that any AI agent can use.**

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/kurokie1337/bbx.git
cd bbx
pip install -r requirements.txt

# Run your first workflow
python cli.py run examples/hello_world.bbx

# Or create a workspace
python cli.py workspace create my-project
python cli.py workspace set ~/.bbx/workspaces/my_project
python cli.py run  # Runs main.bbx
```

---

## Core Features

### 1. Workspaces
Isolated environments for each project:
```bash
python cli.py workspace create ai-assistant
# Creates: main.bbx, config.yaml, state/, logs/, workflows/
```

### 2. Background Execution
Run workflows without blocking:
```bash
python cli.py run deploy.bbx --background
python cli.py ps                    # List running
python cli.py logs <id> --follow    # Stream logs
python cli.py kill <id>             # Stop execution
```

### 3. Persistent State
Memory that survives between sessions:
```bash
python cli.py state set counter 0
python cli.py state get counter
python cli.py state list
```

### 4. Nested Workflows
Workflows can spawn other workflows:
```yaml
steps:
  deploy:
    use: workflow.run
    args:
      path: workflows/deploy.bbx
      background: true
```

---

## Workflow Format (BBX v6.0)

```yaml
id: example_workflow
name: Example Workflow
version: "1.0.0"

inputs:
  message:
    type: string
    default: "Hello"

steps:
  greet:
    use: logger.info
    args:
      message: "${inputs.message} from BBX!"

  fetch_data:
    use: http.get
    args:
      url: "https://api.example.com/data"

  process:
    use: logger.info
    args:
      message: "Got: ${steps.fetch_data.output}"
    depends_on: [fetch_data]
```

**Key concepts:**
- Steps run in parallel by default
- Use `depends_on` for sequential execution
- Variables: `${inputs.*}`, `${steps.*}`, `${env.*}`

---

## Built-in Adapters (14)

| Adapter | Description | Methods |
|---------|-------------|---------|
| `logger` | Logging | info, warn, error, debug |
| `system` | Shell commands | exec |
| `http` | HTTP requests | get, post, put, delete |
| `file` | File operations | read, write, copy, delete, exists, list, mkdir, glob |
| `string` | String manipulation | split, join, replace, regex, encode, decode, hash |
| `state` | Persistent state | get, set, delete, list, increment, append |
| `workflow` | Nested workflows | run, wait, status, kill |
| `a2a` | Agent-to-Agent | discover, call, status, cancel, wait |
| `docker` | Containers | run, build, push |
| `python` | Python execution | exec, eval |
| `process` | Process management | start, stop, status |
| `transform` | Data transformation | merge, filter, map |
| `database` | SQL operations | query, migrate |
| `storage` | Key-value storage | get, set, list |

---

## MCP Server (40 Tools)

BBX exposes 40 MCP tools for AI agents like Claude Code.

### Setup for Claude Code

```bash
# Add BBX as MCP server
claude mcp add bbx -- python -m blackbox.mcp.server

# Restart Claude Code, then verify
claude mcp list
# Should show: bbx - Connected
```

### Available Tools

**Workspace Management:**
- `bbx_workspace_create` - Create isolated environment
- `bbx_workspace_set` - Set active workspace
- `bbx_workspace_list` - List all workspaces

**Process Management:**
- `bbx_run` - Execute workflow
- `bbx_run_background` - Run in background
- `bbx_ps` - List executions
- `bbx_kill` - Kill execution
- `bbx_wait` - Wait for completion
- `bbx_logs` - Get execution logs

**State Management:**
- `bbx_state_get` - Get value
- `bbx_state_set` - Set value
- `bbx_state_list` - List all keys
- `bbx_state_delete` - Delete key
- `bbx_state_increment` - Atomic increment
- `bbx_state_append` - Append to list

**And 25+ more tools for workflows, MCP integration, versioning, etc.**

---

## A2A Protocol

BBX implements Google's Agent2Agent (A2A) Protocol v0.3 for multi-agent communication.

```yaml
# Call another A2A agent from workflow
steps:
  analyze:
    use: a2a.call
    args:
      agent: http://analyst:8001
      skill: analyze_text
      input:
        text: "${inputs.data}"
      wait: true

  report:
    use: a2a.call
    args:
      agent: http://writer:8002
      skill: write_report
      input:
        data: "${steps.analyze.output}"
    depends_on: [analyze]
```

### A2A CLI Commands
```bash
python cli.py a2a serve --port 8000    # Start A2A server
python cli.py a2a discover <url>       # Discover agent
python cli.py a2a call <url> <skill>   # Call skill
```

---

## BBX-Only Coding

Write workflows without Python using file and string adapters:

```yaml
id: process_files
steps:
  read:
    use: file.read
    args:
      path: input.txt
      lines: true

  transform:
    use: string.json_encode
    args:
      value:
        lines: ${steps.read.content}
        count: ${steps.read.count}
    depends_on: [read]

  save:
    use: file.write
    args:
      path: output.json
      content: ${steps.transform.result}
    depends_on: [transform]
```

---

## Project Structure

```
bbx/
├── blackbox/           # Core package
│   ├── core/           # Runtime, adapters, parsers
│   ├── mcp/            # MCP server & client
│   ├── a2a/            # A2A protocol implementation
│   └── cli/            # CLI helpers
├── examples/           # Example workflows
├── docs/               # Documentation
├── tests/              # Test suite
├── cli.py              # CLI entry point
└── requirements.txt    # Dependencies
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Quick start guide |
| [docs/BBX_SPEC_v6.md](docs/BBX_SPEC_v6.md) | Workflow format specification |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |

---

## CLI Reference

```bash
# Workflows
python cli.py run [file.bbx]           # Run workflow
python cli.py run --background         # Background execution
python cli.py validate file.bbx        # Validate syntax

# Workspaces
python cli.py workspace create <name>
python cli.py workspace set <path>
python cli.py workspace info

# State
python cli.py state get <key>
python cli.py state set <key> <value>
python cli.py state list

# Process Management
python cli.py ps                       # List executions
python cli.py logs <id>                # View logs
python cli.py kill <id>                # Kill execution

# A2A
python cli.py a2a serve --port 8000
python cli.py a2a discover <url>
python cli.py a2a call <url> <skill>

# Tools
python cli.py adapters                 # List adapters
python cli.py schema                   # Generate JSON schema
```

---

## Requirements

- Python 3.9+
- Docker (optional, for container operations)

```bash
pip install -r requirements.txt
```

---

## License

BBX is licensed under the **Business Source License 1.1**.

### What This Means

**You CAN:**
- Use BBX for development, testing, and evaluation
- Study and learn from the source code
- Fork and modify for non-production use
- Contribute improvements back to the project

**You CANNOT (without a commercial license):**
- Offer BBX as a hosted/SaaS service to third parties
- Use BBX in production for commercial purposes
- Sell BBX-based products commercially

**On November 5, 2028**, BBX automatically converts to **Apache License 2.0**.

### Commercial Use

For production/commercial use, please contact us to discuss licensing terms.

### Why BSL?

1. **Protect Innovation** - Prevent unfair competition from large cloud providers
2. **Fund Development** - Support full-time development of BBX
3. **Ensure Openness** - Guarantee BBX becomes fully open source in 3 years

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

## Author

**Ilya Makarov**
Krasnoyarsk, Siberia, Russia

*BBX - Operating System for AI Agents*
*First conceived and developed: November 2025*
