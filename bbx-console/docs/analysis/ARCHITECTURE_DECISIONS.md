# Architecture Decisions Record (ADR)

## ADR-001: Backend Framework

### Decision
Use **FastAPI** for the backend.

### Context
Need a Python backend that:
- Integrates directly with BBX Core (Python)
- Supports WebSocket
- Has async support
- Provides automatic API documentation

### Options Considered
1. **FastAPI** - Modern, async, type hints, automatic OpenAPI
2. **Flask** - Simpler, but no native async
3. **Django** - Too heavy for this use case
4. **Node.js** - Would require FFI/subprocess for BBX

### Rationale
- FastAPI is native Python - direct import of BBX modules
- Built-in WebSocket support
- Automatic OpenAPI/Swagger docs
- Excellent async/await support matches BBX's async architecture
- Type hints enable better IDE support

---

## ADR-002: Frontend Framework

### Decision
Use **React + TypeScript + Vite**.

### Context
Need a modern frontend for:
- Real-time dashboard with WebSocket updates
- Complex UI (DAG visualization, editors)
- Good developer experience

### Options Considered
1. **React + Vite** - Fast, ecosystem, flexibility
2. **Vue 3** - Good, but smaller ecosystem for visualization libs
3. **Svelte** - Newer, less library support
4. **Next.js** - SSR not needed, adds complexity

### Rationale
- React has best ecosystem for visualization (React Flow, Cytoscape)
- Vite provides fast development experience
- TypeScript for type safety in complex data structures
- Large community, easy to find solutions

---

## ADR-003: WebSocket Protocol

### Decision
Use **native WebSocket with JSON messages** + subscription model.

### Context
Need real-time updates for:
- Workflow execution status
- Agent status changes
- Memory tier movements
- Ring queue updates

### Options Considered
1. **Native WebSocket** - Simple, no dependencies
2. **Socket.IO** - More features, but heavier
3. **Server-Sent Events (SSE)** - One-way only
4. **GraphQL Subscriptions** - Overkill

### Protocol Design

```typescript
// Client → Server
{
  "type": "subscribe",
  "channel": "workflow:execution:abc123"
}

{
  "type": "unsubscribe",
  "channel": "workflow:execution:abc123"
}

{
  "type": "action",
  "action": "cancel_execution",
  "data": {"executionId": "abc123"}
}

// Server → Client
{
  "type": "event",
  "channel": "workflow:execution:abc123",
  "event": "step:completed",
  "data": {...}
}
```

---

## ADR-004: State Management

### Decision
Use **Zustand** for global state + **TanStack Query** for server state.

### Context
Need to manage:
- UI state (panels, selected items)
- Server data (workflows, agents, executions)
- Real-time updates

### Options Considered
1. **Zustand + TanStack Query** - Simple, effective
2. **Redux + RTK Query** - More boilerplate
3. **Recoil** - Less mature
4. **MobX** - Different paradigm

### Rationale
- Zustand is minimal, easy to learn
- TanStack Query handles caching, refetching, optimistic updates
- Good separation: Zustand for UI, TanStack for server data
- Both work well with TypeScript

---

## ADR-005: DAG Visualization Library

### Decision
Use **React Flow** for DAG visualization.

### Context
Need interactive workflow visualization:
- Nodes represent steps
- Edges represent dependencies
- Real-time status updates
- Pan, zoom, select

### Options Considered
1. **React Flow** - Purpose-built for flow diagrams
2. **Cytoscape.js** - Powerful but complex
3. **D3.js** - Low-level, lots of work
4. **vis.js** - Good but less React-native

### Rationale
- React Flow is specifically designed for workflows
- Built-in node/edge customization
- Mini-map, controls, background grid
- Good performance with virtualization
- Active community

---

## ADR-006: Database

### Decision
Use **SQLite** for development, **PostgreSQL** for production.

### Context
Need to persist:
- Task history
- Execution logs
- User preferences
- Agent metrics

### Options Considered
1. **SQLite/PostgreSQL** - Proven, reliable
2. **MongoDB** - JSON native but less ACID
3. **File-based** - Too simple

### Schema Design

```sql
-- Executions
CREATE TABLE executions (
  id UUID PRIMARY KEY,
  workflow_id VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  inputs JSONB,
  results JSONB,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  duration_ms INTEGER
);

-- Step Logs
CREATE TABLE step_logs (
  id UUID PRIMARY KEY,
  execution_id UUID REFERENCES executions(id),
  step_id VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  output JSONB,
  error TEXT,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  retry_count INTEGER DEFAULT 0
);

-- Tasks
CREATE TABLE tasks (
  id UUID PRIMARY KEY,
  title VARCHAR NOT NULL,
  description TEXT,
  status VARCHAR NOT NULL,
  priority VARCHAR NOT NULL,
  parent_id UUID REFERENCES tasks(id),
  assigned_agent VARCHAR,
  execution_id UUID REFERENCES executions(id),
  created_at TIMESTAMP,
  completed_at TIMESTAMP,
  duration_ms INTEGER
);

-- Agent Metrics
CREATE TABLE agent_metrics (
  id SERIAL PRIMARY KEY,
  agent_id VARCHAR NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  tasks_completed INTEGER,
  tasks_failed INTEGER,
  avg_duration_ms FLOAT
);
```

---

## ADR-007: Code Editor

### Decision
Use **Monaco Editor** for YAML editing.

### Context
Need code editor for:
- Workflow YAML editing
- Syntax highlighting
- Autocomplete
- Error markers

### Options Considered
1. **Monaco Editor** - VSCode's editor, full-featured
2. **CodeMirror 6** - Lighter, extensible
3. **Ace Editor** - Older, still good

### Rationale
- Monaco is familiar (VSCode users)
- Excellent YAML support
- Custom language server possible
- Built-in diff viewer

---

## ADR-008: Styling

### Decision
Use **Tailwind CSS** + **shadcn/ui** components.

### Context
Need consistent styling:
- Dashboard with many components
- Dark theme for developer tool
- Responsive layout

### Options Considered
1. **Tailwind + shadcn/ui** - Utility-first, copy-paste components
2. **MUI** - Heavy, Google-style
3. **Chakra UI** - Good but opinions differ from ours
4. **Plain CSS** - Too much work

### Rationale
- Tailwind enables rapid prototyping
- shadcn/ui provides accessible, customizable components
- Not a dependency, code lives in project
- Easy dark mode support

---

## ADR-009: Authentication

### Decision
**Optional authentication** with JWT for multi-user mode.

### Context
Console may be used:
- Locally (single user, no auth needed)
- Team mode (multiple users, auth required)

### Design

```python
# Settings
auth_enabled: bool = False
auth_secret: str = "..."

# Middleware
if settings.auth_enabled:
    # Verify JWT token
else:
    # Allow all requests
```

For Phase 1, auth is disabled. Implement in Phase 6 if needed.

---

## ADR-010: Project Structure

### Decision

```
bbx-console/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── workflows.py
│   │   │   │   ├── executions.py
│   │   │   │   ├── agents.py
│   │   │   │   ├── memory.py
│   │   │   │   ├── ring.py
│   │   │   │   ├── tasks.py
│   │   │   │   ├── a2a.py
│   │   │   │   └── mcp.py
│   │   │   └── schemas/
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── bbx/
│   │   │   └── bridge.py         # BBX Core integration
│   │   ├── db/
│   │   │   ├── models/
│   │   │   └── session.py
│   │   ├── services/
│   │   │   └── task_decomposer.py
│   │   ├── ws/
│   │   │   └── manager.py        # WebSocket manager
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/               # shadcn components
│   │   │   ├── workflow/
│   │   │   ├── agents/
│   │   │   ├── memory/
│   │   │   ├── ring/
│   │   │   ├── tasks/
│   │   │   └── common/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── stores/
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── docs/
```

### Rationale
- Clear separation backend/frontend
- API routes mirror features
- BBX bridge isolated for easy updates
- Services layer for business logic
- Component organization by feature
