# BBX A2A Multi-Agent Examples

Examples of using the A2A protocol to build multi-agent systems.

## Quick Start

### 1. Start Demo Agents

```bash
# Terminal 1: Analyst Agent (port 8001)
python -m examples.a2a.demo_agents analyst

# Terminal 2: Writer Agent (port 8002)
python -m examples.a2a.demo_agents writer

# Terminal 3: Orchestrator Agent (port 8000)
python -m blackbox.a2a.server --port 8000
```

### 2. Test with curl

```bash
# Get Agent Card
curl http://localhost:8001/.well-known/agent-card.json

# Create task
curl -X POST http://localhost:8001/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{"skillId": "analyze_text", "input": {"text": "Hello world"}}'

# Check status
curl http://localhost:8001/a2a/tasks/{task_id}
```

### 3. Run Workflow

```bash
# Simple agent call
bbx run examples/a2a/simple_call.bbx

# Multi-agent pipeline
bbx run examples/a2a/agent_pipeline.bbx

# Parallel agent calls
bbx run examples/a2a/parallel_agents.bbx
```

## Example Workflows

### simple_call.bbx
Simple single agent call for text analysis.

### agent_pipeline.bbx
Sequential pipeline: data -> analysis -> report.

### parallel_agents.bbx
Parallel calls to multiple agents with result aggregation.

### self_orchestration.bbx
Orchestrator agent that decides which agents to call.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     BBX Orchestrator Agent                       │
│                      (localhost:8000)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    BBX Workflow Engine                     │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  steps:                                              │ │  │
│  │  │    analyze:                                          │ │  │
│  │  │      use: a2a.call                                   │ │  │
│  │  │      args:                                           │ │  │
│  │  │        agent: http://localhost:8001                  │ │  │
│  │  │        skill: analyze_text                           │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ A2A Protocol
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Analyst Agent │  │ Writer Agent  │  │ Checker Agent │
│ localhost:8001│  │ localhost:8002│  │ localhost:8003│
│               │  │               │  │               │
│ Skills:       │  │ Skills:       │  │ Skills:       │
│ - analyze_text│  │ - write_report│  │ - check_facts │
│ - extract_data│  │ - summarize   │  │ - validate    │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Creating Your Own Agent

### 1. Minimal Agent

```python
from blackbox.a2a import create_a2a_app

app = create_a2a_app(
    name="my-agent",
    url="http://localhost:8001",
    description="My custom agent"
)

# Run
import uvicorn
uvicorn.run(app, port=8001)
```

### 2. Agent with Custom Skills

See `demo_agents.py` for a complete example.

### 3. Using in Workflow

```yaml
steps:
  call_my_agent:
    use: a2a.call
    args:
      agent: http://localhost:8001
      skill: my_skill
      input:
        data: "some data"
```
