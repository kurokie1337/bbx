# BBX Console Implementation Work Plan

## Phase Overview

| Phase | Description | Key Deliverables |
|-------|-------------|------------------|
| 0 | Thinking | Analysis docs, architecture decisions |
| 1 | Foundation | Project scaffolding, Docker, DB |
| 2 | BBX Bridge | Core integration, WebSocket |
| 3 | Workflow Manager | DAG visualization, editor |
| 4 | Monitors | Agent, Memory, Ring monitors |
| 5 | Task Manager | AI decomposition, task board |
| 6 | Polish | A2A, MCP, tests, documentation |

---

## Phase 1: Foundation

### 1.1 Backend Setup

**Files to create**:
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings
│   │   └── security.py      # Auth (stub)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       └── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py       # DB connection
│   │   └── models/
│   │       └── __init__.py
│   └── ws/
│       ├── __init__.py
│       └── manager.py       # WebSocket manager
├── requirements.txt
├── Dockerfile
└── pytest.ini
```

**requirements.txt**:
```
fastapi>=0.109.0
uvicorn>=0.27.0
websockets>=12.0
sqlalchemy>=2.0.0
aiosqlite>=0.19.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6
httpx>=0.26.0
pyyaml>=6.0
```

### 1.2 Frontend Setup

**Commands**:
```bash
cd bbx-console/frontend
npm create vite@latest . -- --template react-ts
npm install @tanstack/react-query zustand @xyflow/react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Files to create**:
```
frontend/
├── src/
│   ├── components/
│   │   └── ui/              # shadcn components
│   ├── pages/
│   │   └── Dashboard.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts
│   ├── stores/
│   │   └── appStore.ts
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   └── main.tsx
├── index.html
├── package.json
├── tsconfig.json
├── tailwind.config.js
├── vite.config.ts
└── Dockerfile
```

### 1.3 Docker Setup

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ../:/app/bbx:ro           # BBX Core mount
      - ./backend:/app/backend
    environment:
      - BBX_PATH=/app/bbx
      - DATABASE_URL=sqlite:///./data/bbx_console.db
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - VITE_API_URL=http://localhost:8000

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=bbx
      - POSTGRES_PASSWORD=bbx
      - POSTGRES_DB=bbx_console
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### 1.4 Database Schema

**Initial migration**:
```sql
-- executions
CREATE TABLE executions (
    id VARCHAR PRIMARY KEY,
    workflow_id VARCHAR NOT NULL,
    workflow_name VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'pending',
    inputs JSON,
    results JSON,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- step_logs
CREATE TABLE step_logs (
    id VARCHAR PRIMARY KEY,
    execution_id VARCHAR NOT NULL REFERENCES executions(id),
    step_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'pending',
    output JSON,
    error TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    retry_count INTEGER DEFAULT 0
);

-- tasks
CREATE TABLE tasks (
    id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    description TEXT,
    status VARCHAR NOT NULL DEFAULT 'pending',
    priority VARCHAR NOT NULL DEFAULT 'medium',
    parent_id VARCHAR REFERENCES tasks(id),
    assigned_agent VARCHAR,
    execution_id VARCHAR REFERENCES executions(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    metadata JSON
);

-- agent_metrics
CREATE TABLE agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    tasks_completed INTEGER DEFAULT 0,
    tasks_failed INTEGER DEFAULT 0,
    avg_duration_ms FLOAT DEFAULT 0
);

-- Indexes
CREATE INDEX idx_executions_workflow ON executions(workflow_id);
CREATE INDEX idx_executions_status ON executions(status);
CREATE INDEX idx_step_logs_execution ON step_logs(execution_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_parent ON tasks(parent_id);
```

---

## Phase 2: BBX Bridge + WebSocket

### 2.1 BBX Bridge

**backend/app/bbx/bridge.py**:
```python
class BBXBridge:
    """Bridge between Console and BBX Core"""

    def __init__(self):
        self.context_tiering: ContextTiering
        self.agent_ring: AgentRing
        self.event_bus: EventBus

    async def initialize(self):
        """Initialize BBX components"""

    async def shutdown(self):
        """Cleanup BBX components"""

    # Workflow methods
    async def list_workflows(self) -> List[WorkflowInfo]
    async def get_workflow(self, id: str) -> WorkflowDetail
    async def run_workflow(self, id: str, inputs: Dict) -> str
    async def cancel_execution(self, exec_id: str) -> bool

    # Agent methods
    async def list_agents(self) -> List[AgentInfo]
    async def get_agent(self, id: str) -> AgentDetail
    async def get_agent_metrics(self, id: str) -> AgentMetrics

    # Memory methods
    async def get_memory_stats(self) -> MemoryStats
    async def get_tier_items(self, tier: str) -> List[MemoryItem]
    async def pin_item(self, key: str) -> bool
    async def unpin_item(self, key: str) -> bool

    # Ring methods
    async def get_ring_stats(self) -> RingStats
    async def get_queue_items(self, queue: str) -> List[QueueItem]
```

### 2.2 WebSocket Manager

**backend/app/ws/manager.py**:
```python
class WebSocketManager:
    """Manage WebSocket connections and subscriptions"""

    def __init__(self):
        self.connections: Dict[str, WebSocket]
        self.subscriptions: Dict[str, Set[str]]

    async def connect(self, ws: WebSocket) -> str
    async def disconnect(self, conn_id: str)
    async def subscribe(self, conn_id: str, channel: str)
    async def unsubscribe(self, conn_id: str, channel: str)
    async def broadcast(self, channel: str, event: str, data: Any)
```

### 2.3 Event Bridge

```python
# Connect BBX EventBus to WebSocket
async def setup_event_bridge(bbx: BBXBridge, ws_manager: WebSocketManager):
    @bbx.event_bus.on(EventType.STEP_START)
    async def on_step_start(event):
        await ws_manager.broadcast(
            f"execution:{event.data['workflow_id']}",
            "step:started",
            event.data
        )
```

---

## Phase 3: Workflow Manager + DAG

### 3.1 API Routes

```python
# backend/app/api/routes/workflows.py

@router.get("/")
async def list_workflows() -> List[WorkflowListItem]

@router.get("/{id}")
async def get_workflow(id: str) -> WorkflowDetail

@router.post("/validate")
async def validate_workflow(content: str) -> ValidationResult

@router.post("/{id}/run")
async def run_workflow(id: str, inputs: Dict) -> ExecutionInfo

# backend/app/api/routes/executions.py

@router.get("/")
async def list_executions() -> List[ExecutionState]

@router.get("/{id}")
async def get_execution(id: str) -> ExecutionState

@router.post("/{id}/cancel")
async def cancel_execution(id: str) -> bool
```

### 3.2 Frontend Components

```
frontend/src/components/workflow/
├── WorkflowList.tsx
├── WorkflowDetail.tsx
├── WorkflowEditor.tsx
├── DAGVisualization.tsx
├── ExecutionTimeline.tsx
├── StepNode.tsx
└── InputForm.tsx
```

### 3.3 DAG Visualization

```typescript
// DAGVisualization.tsx
import ReactFlow from '@xyflow/react';

const DAGVisualization = ({workflow, execution}) => {
  const nodes = useMemo(() => buildNodes(workflow, execution), [workflow, execution]);
  const edges = useMemo(() => buildEdges(workflow), [workflow]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
    >
      <MiniMap />
      <Controls />
      <Background />
    </ReactFlow>
  );
};
```

---

## Phase 4: Monitors

### 4.1 Agent Monitor

**Features**:
- Agent list with status indicators
- Agent detail panel
- Task history
- Metrics charts

**Components**:
```
frontend/src/components/agents/
├── AgentList.tsx
├── AgentCard.tsx
├── AgentDetail.tsx
├── AgentMetrics.tsx
└── TaskHistory.tsx
```

### 4.2 Memory Explorer

**Features**:
- Tier visualization (HOT/WARM/COOL/COLD)
- Item list per tier
- Pin/unpin actions
- Stats dashboard

**Components**:
```
frontend/src/components/memory/
├── MemoryDashboard.tsx
├── TierVisualization.tsx
├── TierPanel.tsx
├── MemoryItem.tsx
└── MemoryStats.tsx
```

### 4.3 Ring Monitor

**Features**:
- Queue visualization
- Worker status
- Throughput graph
- Latency percentiles

**Components**:
```
frontend/src/components/ring/
├── RingDashboard.tsx
├── QueueVisualization.tsx
├── WorkerStatus.tsx
├── ThroughputChart.tsx
└── LatencyChart.tsx
```

---

## Phase 5: Task Manager

### 5.1 Task Board

**Features**:
- Kanban-style board
- Drag-and-drop
- Task detail modal
- Subtask tree

**Components**:
```
frontend/src/components/tasks/
├── TaskBoard.tsx
├── TaskColumn.tsx
├── TaskCard.tsx
├── TaskDetail.tsx
├── SubtaskTree.tsx
└── CreateTaskModal.tsx
```

### 5.2 AI Decomposition

**Service**:
```python
# backend/app/services/task_decomposer.py

class TaskDecomposer:
    async def decompose(self, task_description: str) -> DecompositionResult:
        """Use architect agent to decompose task"""

        result = await self.agent_adapter.invoke_subagent(
            name="architect",
            prompt=f"Break down this task into subtasks: {task_description}"
        )

        # Parse result into subtasks
        subtasks = self.parse_subtasks(result)

        # Assign agents
        for subtask in subtasks:
            subtask.assigned_agent = self.recommend_agent(subtask)

        return DecompositionResult(
            original_task=task_description,
            subtasks=subtasks,
            suggested_workflow=self.generate_workflow(subtasks)
        )
```

### 5.3 Workflow Generation

```python
def generate_workflow(subtasks: List[Subtask]) -> str:
    """Generate BBX workflow from subtasks"""

    steps = []
    for i, subtask in enumerate(subtasks):
        step = {
            "id": f"step_{i+1}",
            "mcp": "agent",
            "method": "subagent",
            "inputs": {
                "name": subtask.assigned_agent,
                "prompt": subtask.description
            }
        }
        if subtask.depends_on:
            step["depends_on"] = subtask.depends_on

        steps.append(step)

    workflow = {
        "bbx": "6.0",
        "workflow": {
            "id": "generated",
            "name": "Generated Workflow",
            "steps": steps
        }
    }

    return yaml.dump(workflow)
```

---

## Phase 6: Polish

### 6.1 A2A Playground

**Features**:
- Agent discovery
- Task sender
- Response viewer
- SSE streaming

### 6.2 MCP Tools

**Features**:
- Tool list
- Tool invoker
- Result viewer

### 6.3 Testing

**Backend tests**:
```
backend/tests/
├── test_api/
│   ├── test_workflows.py
│   ├── test_executions.py
│   └── test_agents.py
├── test_bbx/
│   └── test_bridge.py
└── test_ws/
    └── test_manager.py
```

**Frontend tests**:
```
frontend/src/__tests__/
├── components/
│   ├── DAGVisualization.test.tsx
│   └── TaskBoard.test.tsx
└── hooks/
    └── useWebSocket.test.ts
```

### 6.4 Documentation

```
docs/
├── getting-started.md
├── configuration.md
├── api-reference.md
├── deployment.md
└── troubleshooting.md
```

---

## Checkpoints

After each phase, verify:

- [ ] All planned files created
- [ ] Tests passing
- [ ] Docker compose working
- [ ] Features demo-able
- [ ] Documentation updated

## Success Criteria

Console is complete when:

1. Can list and run workflows
2. DAG visualization works in real-time
3. Agent status visible
4. Memory tiers browsable
5. Ring stats displayed
6. Tasks can be created and decomposed
7. All tests passing
8. Documentation complete
