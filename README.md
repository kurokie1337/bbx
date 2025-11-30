# BBX — Operating System for AI Agents

> **The first operating system designed specifically for AI agents.**
> Run workflows, manage processes, persist state, and orchestrate intelligence — just like Linux, but for the age of AI.

[![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![Architecture](https://img.shields.io/badge/Architecture-4--Level-purple.svg)](#-4-level-architecture)
[![Status](https://img.shields.io/badge/Status-Testing-orange.svg)](#-project-status)
[![MCP](https://img.shields.io/badge/MCP%20Tools-40+-brightgreen.svg)](#-mcp-server-integration)
[![A2A](https://img.shields.io/badge/A2A-v0.3-blue.svg)](#-agent-to-agent-protocol-a2a)

```
BBX - Operating System for AI Agents
Copyright 2025 Ilya Makarov
First Release: November 26, 2025
```

---

## ⚠️ Project Status

> **CURRENT STATUS: TESTING & DEVELOPMENT**
> This project is currently in active development. While the core architecture is stable and **patent-pending**, we are refining the system for production release.
>
> **Use with caution in critical environments.** We are building the future, and the paint is still wet.

---

## 🌌 The Vision

When AI agents (Claude, GPT, etc.) work on complex tasks, they need more than just a chat interface. They need **infrastructure**.

Linux gave humans:
*   **Filesystems** for storage
*   **Processes** for execution
*   **Shells** for automation
*   **Kernels** for resource management

**BBX gives AI agents:**
*   **Workspaces** for isolated environments (`/home/agent`)
*   **State** for persistent memory (Key-Value Store)
*   **AgentRing** for high-performance batch operations
*   **ContextTiering** for infinite memory management
*   **A2A Protocol** for multi-agent collaboration

---

## 💎 The Core Idea: BBX Workflow Format

At the heart of BBX is the **`.bbx` file format** — a **declarative, YAML-based workflow language** for AI agents.

### Why This Matters
Traditional AI agents write **imperative code** (Python, JavaScript). This creates:
*   **Security risks**: Agents have full system access.
*   **Debugging nightmares**: Code is hard to trace and reproduce.
*   **No isolation**: One bad agent can break everything.

**BBX solves this** by making agents write **declarative workflows** instead:
```yaml
# hello_world.bbx
id: hello_world
name: "My First Workflow"
version: 1.0.0

steps:
  greet:
    use: logger.info
    args:
      message: "Hello from BBX!"
```

This is **not code** — it's a **data structure**. BBX safely executes it through the runtime.

### BBX Format Philosophy
1.  **Declarative over Imperative**: Agents describe *what* to do, not *how*.
2.  **Adapter-Based**: All actions go through safe, audited adapters.
3.  **DAG Execution**: Steps run in parallel when possible, sequentially when needed.
4.  **Human-Readable**: YAML format is easy to read, debug, and version-control.
5.  **Reproducible**: Same workflow = same result, every time.

### BBX Workflow Anatomy
```yaml
# Metadata
id: my_workflow
name: "Descriptive Name"
version: 1.0.0
description: "What this workflow does"

# Inputs (optional)
inputs:
  api_key:
    type: string
    required: true
  max_results:
    type: integer
    default: 10

# Steps (the core)
steps:
  fetch_data:
    use: http.get
    args:
      url: "https://api.example.com/data"
      headers:
        Authorization: "Bearer ${inputs.api_key}"
    
  process:
    use: python.run
    args:
      code: |
        data = steps.fetch_data.output.json()
        return [item['title'] for item in data[:${inputs.max_results}]]
    depends_on: [fetch_data]
    
  save:
    use: file.write
    args:
      path: "results.txt"
      content: "${steps.process.output}"
    depends_on: [process]
```

**Key Features:**
*   **Variable Interpolation**: `${inputs.api_key}`, `${steps.fetch_data.output}`
*   **Dependencies**: `depends_on` creates a DAG (Directed Acyclic Graph).
*   **Type Safety**: Inputs have types, defaults, and validation.
*   **Adapters**: `http.get`, `python.run`, `file.write` — all sandboxed.

---

## 🏗️ 4-Level Architecture

BBX is built on a robust 4-level architecture, mimicking a traditional OS but optimized for AI:

### Level 1: Kernel (Runtime)
The core Python engine (`blackbox.core`) that executes code, manages threads, and handles low-level I/O. It is the "CPU" of the agent system.
*   **Async Execution**: Native `asyncio` support for high concurrency.
*   **DAG Engine**: Directed Acyclic Graph execution for complex dependencies.
*   **Sandboxing**: Secure execution environment.
*   **State Isolation**: Each execution has its own context.

### Level 2: BBX Base (Standard Library)
A rich set of **15+ Adapters** that provide standardized interfaces for agents. Instead of writing raw code, agents use these safe, atomic tools:
*   `http`: Network requests (GET, POST, PUT, DELETE).
*   `file`: Safe filesystem access (read, write, append, list).
*   `docker`: Container management (run, exec, pull, build).
*   `system`: OS command execution (with timeout, sandboxing).
*   `state`: Persistent memory (get, set, delete, increment, append).
*   `process`: Background process control (spawn, monitor, kill).
*   `a2a`: Agent communication (call other agents via A2A protocol).
*   `python`: Code execution (sandboxed Python interpreter).
*   `database`: SQL operations (PostgreSQL, MySQL, SQLite).
*   `logger`, `string`, `transform`, `workflow`, `storage`, `os_abstraction`.

### Level 3: OS Layer (System Services)
The "User Space" for agents, managed via the **CLI**:
*   **Workspaces**: Isolated project environments (like `/home/user`).
    *   `bbx workspace create`, `bbx workspace set`, `bbx workspace info`
*   **Process Manager**: Background execution and monitoring (like `systemd`).
    *   `bbx run --background`, `bbx ps`, `bbx kill`, `bbx logs`, `bbx wait`
*   **State Management**: Persistent key-value memory (like `env` vars but persistent).
    *   `bbx state set`, `bbx state get`, `bbx state list`, `bbx state delete`
*   **System Health**: Docker integration and self-checks.
    *   `bbx system`, `bbx adapters`
*   **Version Control**: Workflow version management.
    *   `bbx version-ctrl create`, `bbx version-ctrl list`, `bbx version-ctrl rollback`

### Level 4: Agent System (Intelligence)
The high-level cognitive layer:
*   **A2A Protocol**: Standardized Agent-to-Agent communication (v0.3).
*   **MCP Integration**: Full Model Context Protocol support (Server & Client).
*   **AI Models**: Local model management for workflow generation.
    *   `bbx model download qwen-1.8b`, `bbx generate "task description"`
*   **Tool Learning**: Auto-learn CLI tools by parsing `--help`.
    *   `bbx learn tool kubectl`, `bbx learned_tools`

---

## 🔧 BBX Kernel (Bare-Metal OS)

BBX now includes a **bare-metal operating system kernel** written in Rust, designed to run AI agents directly on hardware without a host OS.

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      USER SPACE (Agents)                        │
│                   (BBX Workflows, A2A Protocol)                 │
├─────────────────────────────────────────────────────────────────┤
│                    SYSCALL INTERFACE                            │
│         (spawn, io_submit, state_get, agent_call, etc.)        │
├─────────────────────────────────────────────────────────────────┤
│                      BBX KERNEL                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  SCHEDULER  │  │   MEMORY    │  │       I/O RING          │ │
│  │  (DAG-based)│  │  (Tiered)   │  │  (io_uring-inspired)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                       HARDWARE (x86_64)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components
| Component | Description |
|-----------|-------------|
| **Scheduler** | Priority-based (REALTIME > HIGH > NORMAL > LOW), DAG-aware |
| **Memory** | Tiered memory (HOT → WARM → COOL → COLD), frame allocator, paging |
| **I/O Ring** | io_uring-inspired batch submission/completion |
| **Syscalls** | Agent-specific: `agent_call`, `workflow_run`, `state_get/set` |
| **Drivers** | Serial (UART 16550), Timer (PIT), Keyboard (PS/2) |

### Building the Kernel
```bash
cd kernel
cargo build --release
cargo bootimage --release

# Run in QEMU
qemu-system-x86_64 -drive format=raw,file=target/x86_64-unknown-none/release/bootimage-bbx-kernel.bin -serial stdio
```

📖 **[Full Kernel Documentation](kernel/README.md)**

---

## 🚀 Key Innovations (The "Secret Sauce")

BBX 2.0 introduces revolutionary concepts adapted from modern Linux kernel development:

### 1. AgentRing (inspired by `io_uring`)
*   **Problem**: Agents make thousands of small API calls (HTTP, File), creating massive overhead.
*   **Solution**: A ring-buffer architecture for **batch operation submission**. Agents submit 100 operations in one call, and the kernel processes them asynchronously.
*   **Result**: 10-100x reduction in overhead and latency.

### 2. BBX Hooks (inspired by `eBPF`)
*   **Problem**: Modifying workflows for logging, security, or metrics requires rewriting code.
*   **Solution**: Dynamic code injection. "Attach" hooks to any step of a workflow (pre-execution, post-execution) to enforce security policies or gather metrics without touching the workflow logic.
*   **Use Case**: "Block all network calls to non-internal IPs" — enforced by a hook, not the agent.

### 3. ContextTiering (inspired by `MGLRU`)
*   **Problem**: LLM context windows are finite and expensive.
*   **Solution**: Multi-Generation LRU memory management.
    *   **Hot (Gen 0)**: Immediate RAM (current task).
    *   **Warm (Gen 1)**: Compressed RAM (recent history).
    *   **Cool (Gen 2)**: NVMe/Disk (session history).
    *   **Cold (Gen 3)**: Vector DB (long-term archive).
*   **Result**: Infinite effective memory with optimal performance.

### 4. StateSnapshots (inspired by `XFS Reflink`)
*   **Problem**: Forking an agent or rolling back state is slow and data-heavy.
*   **Solution**: Copy-on-Write (CoW) state management. Creating a snapshot is instant (O(1)).
*   **Result**: Instant "Save Game" and "Load Game" for agents.

---

## � Ecosystem & Connectivity

### MCP Server Integration
BBX exposes **40+ MCP tools** for AI agents (like Claude Code, Cursor, Windsurf).
Add BBX as an MCP server to give your agent full OS capabilities.

**Tool Categories:**
*   **Core**: `bbx_run`, `bbx_validate`, `bbx_generate`
*   **Workspace**: `bbx_workspace_create`, `bbx_workspace_set`
*   **Process**: `bbx_ps`, `bbx_kill`, `bbx_logs`, `bbx_run_background`
*   **State**: `bbx_state_set`, `bbx_state_get`, `bbx_state_list`
*   **MCP Client**: `bbx_mcp_call`, `bbx_mcp_discover` (Connect to *other* MCP servers!)
*   **Versioning**: `bbx_version_create`, `bbx_version_rollback`

### Agent-to-Agent Protocol (A2A)
BBX implements the **Google Agent2Agent Protocol v0.3**.
*   **Agent Card**: `/.well-known/agent-card.json` for capability discovery.
*   **Task Lifecycle**: Pending -> In Progress -> Completed/Failed.
*   **Artifacts**: Exchange files, code, and data between agents.

---

## 🛠️ Quick Start

```bash
# 1. Clone and install
git clone https://github.com/kurokie1337/bbx.git
cd bbx
pip install -r requirements.txt

# 2. Create a Workspace (Your Project Home)
python cli.py workspace create my-project
python cli.py workspace set ~/.bbx/workspaces/my_project

# 3. Generate a Workflow with AI
python cli.py generate "Scrape hacker news and save top 5 titles to a file"

# 4. Run it
python cli.py run generated.bbx

# 5. Check system
python cli.py system
```

---

## 📚 Documentation

### 🚀 For Humans
*   **[Getting Started](docs/guides/GETTING_STARTED.md)**: Step-by-step installation and first workflow.
*   **[Troubleshooting](docs/guides/TROUBLESHOOTING.md)**: Common issues and solutions.

### 🤖 For Agents (AI)
*   **[Agent Guide](docs/guides/AGENT_GUIDE.md)**: **<-- READ THIS FIRST IF YOU ARE AN AGENT.** Context injection and operating manual.

### 📖 Reference
*   **[Workflow Format](docs/reference/WORKFLOW_FORMAT.md)**: Complete specification for `.bbx` files.

### 🖥️ BBX Console (Web UI)
*   **[BBX Console](bbx-console/README.md)**: Web-based management console for monitoring workflows, agents, and system state.
*   **[UI Specification](docs/BBX_CONSOLE_UI_SPEC.md)**: Detailed UI/UX design specification.

### 🔬 Research & Internal
*   **[Manifesto](docs/research/MANIFESTO.md)**: The vision behind BBX 2.0.
*   **[Roadmap](docs/research/ROADMAP.md)**: Future development plans.
*   **[Architecture Report](docs/research/ARCHITECTURE_REPORT.md)**: Deep dive into system design.
*   **[Technical Spec](docs/internal/TECHNICAL_SPEC.md)**: Implementation details.

### 📦 SDKs
*   **[Python SDK](sdks/python/bbx-sdk/README.md)**: Official Python client library.
*   **[Node.js SDK](sdks/nodejs/README.md)**: Official TypeScript/JavaScript SDK.

---

## ⚖️ License & Usage

BBX is licensed under the **Business Source License 1.1**.

### 🟢 Personal & Non-Commercial Use
**You are free to use BBX for:**
*   Personal projects
*   Learning and research
*   Non-profit development
*   Testing and evaluation

### 🔴 Commercial Use
**You must obtain a license for:**
*   Using BBX in a commercial production environment
*   Building commercial products on top of BBX
*   Offering BBX as a service

> *Note: We do not publicly list pricing or contact emails. If you are a business interested in licensing, you know how to find us.*

**On November 5, 2028**, BBX automatically converts to **Apache License 2.0**.

---

## Author

**Ilya Makarov**
Krasnoyarsk, Siberia, Russia

*BBX - Operating System for AI Agents*
*First conceived and developed: November 2025*
