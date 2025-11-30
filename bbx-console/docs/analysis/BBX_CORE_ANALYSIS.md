# BBX Core Analysis

## Overview

BBX (Blackbox) - полноценная Operating System for AI Agents. Анализ ключевых компонентов.

---

## 1. Runtime Engine (`blackbox/core/runtime.py`)

### Основные функции

```python
async def run_file(
    file_path: str,
    event_bus: Optional[EventBus] = None,
    use_cache: bool = True,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]
```

### Ключевые особенности

1. **Dual Execution Mode**:
   - Sequential execution (простые workflows)
   - DAG parallel execution (workflows с зависимостями)
   - Автоматический выбор: `should_use_dag(steps)`

2. **Input Handling**:
   - Default inputs из workflow definition
   - Runtime inputs (переданные в `run_file`)
   - Merge с приоритетом runtime inputs
   - Доступ через `${inputs.key}`

3. **Step Execution Features**:
   - `when` condition - условное выполнение
   - `timeout` - таймаут в ms
   - `retry` + `retry_delay` + `retry_backoff`
   - `pre_check` - проверка перед выполнением
   - `fallback` - резервный action при ошибке
   - `on_failure` - действия при неудаче

4. **Context Resolution**:
   - `${steps.id.output}` - результат предыдущего шага
   - `${inputs.key}` - входные параметры
   - `${state.key}` - workspace state

### Integration Points for Console

```python
# BBX Bridge должен:
- Получать EventBus события (STEP_START, STEP_END, STEP_ERROR)
- Передавать inputs из UI
- Отслеживать results для каждого шага
- Поддерживать cancel операцию
```

---

## 2. DAG Engine (`blackbox/core/dag.py`)

### WorkflowDAG Class

```python
class WorkflowDAG:
    def __init__(self, steps: List[Dict[str, Any]])
    def get_execution_levels(self) -> List[List[str]]
    def get_step(self, step_id: str) -> Dict[str, Any]
    def get_dependencies(self, step_id: str) -> Set[str]
```

### Ключевые особенности

1. **Level-based Execution**:
   - Steps группируются в levels по зависимостям
   - Steps в одном level выполняются параллельно
   - Levels выполняются последовательно

2. **Dependency Tracking**:
   - `depends_on` - явные зависимости
   - `parallel` flag - разрешение параллельного выполнения
   - Cycle detection via DFS

3. **Visualization Data**:
   ```python
   levels = dag.get_execution_levels()
   # [[step1, step2], [step3], [step4, step5]]
   # Level 0: step1, step2 параллельно
   # Level 1: step3 после level 0
   # Level 2: step4, step5 параллельно после level 1
   ```

### Integration Points for Console

```yaml
WorkflowDAG visualization:
  nodes:
    - id: step_id
    - label: step name
    - status: pending | running | completed | failed
    - level: 0, 1, 2...

  edges:
    - source: dependency_id
    - target: step_id
```

---

## 3. Context Tiering (`blackbox/core/v2/context_tiering.py`)

### MGLRU Architecture

4 поколения памяти (как Linux MGLRU):

| Tier | Storage | Compression | Access Time | Max Age |
|------|---------|-------------|-------------|---------|
| HOT  | RAM     | None        | Instant     | 5 min   |
| WARM | RAM     | zlib        | Fast        | 1 hour  |
| COOL | Disk    | zlib        | Medium      | 1 day   |
| COLD | Archive | zlib        | Slow        | Unlimited |

### Key Classes

```python
class ContextTiering:
    async def get(key: str) -> Any           # Auto-promote from cold
    async def set(key: str, value: Any)      # Goes to HOT
    async def pin(key: str)                  # Prevent demotion
    async def unpin(key: str)                # Allow demotion
    def get_stats() -> Dict                  # Hit rate, sizes, etc.

class RefaultTracker:
    def get_distance(key: str) -> int        # Generations since HOT
    def get_access_frequency(key: str) -> float
```

### Background Aging Loop

```python
async def _aging_loop():
    # Runs every aging_interval (30s default)
    # Demotes items based on:
    # - Last access time
    # - Refault distance
    # - Pinned status
```

### Integration Points for Console

```yaml
Memory Explorer API needs:
  GET /memory/tiers:
    - tier: HOT/WARM/COOL/COLD
    - items: [key, value_preview, access_count, is_pinned]
    - size_bytes, item_count

  Actions:
    - pin/unpin
    - manual promote/demote
    - delete
    - clear tier

  Real-time events:
    - memory:item:promoted
    - memory:item:demoted
    - memory:stats:update
```

---

## 4. AgentRing (`blackbox/core/v2/ring.py`)

### io_uring-inspired Architecture

```
          ┌─────────────────┐
          │ Submission Queue│ (PriorityQueue, max 4096)
          │     (SQ)        │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  Worker Pool    │ (4-32 workers)
          │   Processing    │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │ Completion Queue│ (Queue, max 4096)
          │     (CQ)        │
          └─────────────────┘
```

### Priority Levels

```python
class OperationPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3  # Processed first
```

### Key APIs

```python
class AgentRing:
    async def submit(op: Operation) -> str
    async def submit_batch(ops: List[Operation]) -> List[str]
    async def wait_completion(op_id: str) -> Completion
    async def wait_batch(op_ids: List[str]) -> List[Completion]
    async def cancel(op_id: str) -> bool
    def get_stats() -> ExtendedRingStats
```

### Statistics Available

```python
@dataclass
class ExtendedRingStats:
    operations_submitted: int
    operations_completed: int
    operations_failed: int
    operations_cancelled: int
    operations_timeout: int
    pending_count: int
    processing_count: int
    active_workers: int
    worker_pool_size: int
    submission_queue_size: int
    completion_queue_size: int
    throughput_ops_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    worker_utilization: float
```

### Integration Points for Console

```yaml
Ring Monitor needs:
  Queues:
    - SQ size, drain rate
    - CQ size, drain rate
    - Items in each queue

  Workers:
    - active_workers / max_workers
    - utilization %

  Priorities:
    - Items per priority level
    - Wait time per priority

  Real-time:
    - ring:item:submitted
    - ring:item:started
    - ring:item:completed
```

---

## 5. A2A Client (`blackbox/a2a/client.py`)

### Agent-to-Agent Protocol

```python
class A2AClient:
    # Discovery
    async def discover(agent_url: str) -> AgentCard
    async def discover_skill(skill_id: str) -> Optional[tuple]

    # Task Management
    async def create_task(agent_url, skill_id, input) -> Dict
    async def get_task(agent_url, task_id) -> Dict
    async def cancel_task(agent_url, task_id) -> Dict
    async def wait_for_task(agent_url, task_id) -> Dict
    async def stream_task(agent_url, task_id) -> AsyncGenerator

    # JSON-RPC
    async def rpc_call(agent_url, method, params) -> Any
```

### Agent Card Structure

```json
{
  "name": "Agent Name",
  "description": "...",
  "url": "http://localhost:9000",
  "version": "1.0.0",
  "protocolVersion": "0.3",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "analyze",
      "name": "Analyze Data",
      "description": "...",
      "inputSchema": { ... }
    }
  ],
  "authentication": { "schemes": ["none"] }
}
```

### Task Lifecycle

```
submitted → working → completed
              ↓         ↓
         input-required failed
                        ↓
                    cancelled
```

### Integration Points for Console

```yaml
A2A Playground needs:
  - Agent Card viewer
  - Task sender (select skill, fill input, send)
  - Task timeline (submitted → working → completed)
  - SSE streaming display
  - JSON-RPC interface
```

---

## 6. Agent SDK Adapter (`blackbox/core/adapters/agent_sdk.py`)

### Claude Agent SDK Integration

```python
class AgentSDKAdapter(MCPAdapter):
    async def query(prompt, system_prompt, max_turns, allowed_tools, ...)
    async def invoke_subagent(name, prompt, context, timeout)
    async def list_subagents(path=".claude/agents")
    async def parallel_query(queries, max_parallel)
    async def query_with_memory(prompt, memory_key, include_history)
```

### A2A Wrapper

```python
class AgentA2AWrapper:
    def get_agent_card() -> Dict          # A2A Agent Card
    async def handle_task(skill_id, input_data) -> Dict
    def create_fastapi_app()              # A2A HTTP server
```

### Integration Points for Console

```yaml
Agent Monitor needs:
  - List subagents from .claude/agents/
  - Show agent profiles (frontmatter + instructions)
  - Track agent status (idle, working, queued)
  - Show metrics (tasks completed, avg duration)

Task Manager AI needs:
  - query() for AI decomposition
  - invoke_subagent() for specific agent tasks
  - parallel_query() for batch operations
```

---

## 7. MCP Server (`blackbox/mcp/server.py`)

### MCP Protocol Integration

```python
def create_server() -> Server:
    server = Server("bbx-workflow-engine")

    @server.list_tools()
    async def list_tools() -> list[Tool]

    @server.call_tool()
    async def call_tool(name, arguments) -> Sequence[TextContent]
```

### Available Tools (via TOOL_HANDLERS)

- `bbx_run` - Run workflow
- `bbx_validate` - Validate workflow
- `bbx_info` - Workflow info
- `bbx_list_workflows` - List workflows
- `bbx_mcp_discover` - Discover MCP servers
- `bbx_mcp_call` - Call MCP tool
- `bbx_ps` - List executions
- `bbx_state_get/set/list` - State management
- `bbx_workspace_*` - Workspace management
- ... (50+ tools)

### Integration Points for Console

```yaml
MCP Tools page needs:
  - List all tools with schemas
  - Tool invoker (form from inputSchema)
  - External MCP servers list
  - Connection status
```

---

## Summary of Integration Requirements

### Events to Capture

1. **Workflow Events**:
   - `WORKFLOW_START`, `WORKFLOW_END`
   - `STEP_START`, `STEP_END`, `STEP_ERROR`

2. **Memory Events**:
   - Promotion/Demotion
   - Access (for hit rate)

3. **Ring Events**:
   - Submit/Complete
   - Worker status changes

4. **A2A Events**:
   - Task created/updated/completed
   - Message received

### Data Polling Requirements

| Component | Poll Interval | Data |
|-----------|---------------|------|
| Ring Stats | 1s | throughput, latency, queues |
| Memory Stats | 5s | hit rate, tier sizes |
| Agent Status | 2s | idle/working/queued |
| Workflow Status | real-time | via WebSocket |

### BBX Bridge Implementation Checklist

- [ ] Initialize all BBX components (ContextTiering, AgentRing)
- [ ] Start background tasks (aging loop, workers)
- [ ] Expose get/set methods for each component
- [ ] Hook into EventBus for real-time updates
- [ ] Forward events to WebSocket manager
- [ ] Handle graceful shutdown
