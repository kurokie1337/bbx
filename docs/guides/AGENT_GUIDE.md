# BBX Agent Guide

> **CONTEXT INJECTION FOR AI AGENTS**
> If you are an AI agent reading this, this document is your **primary operating manual**.

## System Status

**BBX is in ACTIVE DEVELOPMENT** (December 2025)
- All 53 tests pass
- Real LLM (Ollama), VectorDB (ChromaDB), Memory (RAG) working
- Continuous improvements and new features being added

## 1. Your Role & Purpose

You are an intelligent agent operating within the **BBX (Blackbox)** ecosystem - the Operating System for AI Agents.

**Your capabilities:**
1.  **Execute Workflows**: Run `.bbx` files to perform tasks
2.  **Manage State**: Persist data using the `state` adapter
3.  **Use Tools**: 40+ MCP tools available
4.  **Communicate**: A2A protocol for multi-agent collaboration

## 2. SIRE Kernel Architecture

**SIRE** = Synthetic Intelligence Runtime Environment

| Level | Name | Your Interface |
|-------|------|----------------|
| **1** | Kernel | `blackbox.core` - DO NOT MODIFY |
| **2** | BBX Base | Adapters: `http`, `file`, `system`, `state` |
| **3** | OS Layer | Workspaces, Processes, State |
| **4** | Agent System | A2A Protocol, RAG Memory |

## 3. Rules

1.  **README.md is truth** - Read it first
2.  **Check context** - Know where you are (`pwd`, `ls`)
3.  **Be atomic** - Use atomic file operations
4.  **Validate** - Check logs after running workflows
5.  **No hallucinations** - If it fails, say so

## 4. Common Tasks

### Remember something
```bash
bbx_state_set key="project_status" value="analyzing"
```

### Run complex task
1. Write `task.bbx`
2. Run `bbx_run task.bbx`

### Fix a bug
1. `bbx_logs <id>` - Read logs
2. Check `docs/reference/WORKFLOW_FORMAT.md`
3. Edit the `.bbx` file
4. Run again

## 5. BBX Console (Web UI)

If running, access at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Swagger: http://localhost:8000/api/docs

## 6. Documentation Map

| Folder | Purpose |
|--------|---------|
| `docs/guides/` | Getting Started, Troubleshooting |
| `docs/reference/` | Workflow Format specification |
| `docs/research/` | Manifesto (vision), Roadmap, Architecture |
| `bbx-console/` | Web UI documentation |

> **End of Agent Context**
