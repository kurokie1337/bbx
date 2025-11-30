# BBX Agents Analysis

## Agent Configuration Format

Agents defined in `.claude/agents/*.md` files with YAML frontmatter.

---

## 1. Architect Agent

**File**: `.claude/agents/architect.md`

```yaml
---
name: architect
description: Use PROACTIVELY for system design, architecture planning, and breaking down complex tasks into steps
tools: Read, Grep, Glob, Task
model: sonnet
---
```

### Responsibilities
- Analyze requirements
- Design architecture
- Create implementation plans
- Identify risks

### Output Format
```markdown
## Analysis
[What you understand about the task]

## Proposed Architecture
[High-level design with diagrams if helpful]

## Implementation Plan
1. Step 1: [Description] - Files: [list]
2. Step 2: [Description] - Files: [list]

## Risks & Considerations
- Risk 1: [description] - Mitigation: [strategy]
```

### Usage in Task Manager
- Primary agent for **AI Decomposition**
- Breaks complex tasks into subtasks
- Assigns agents to subtasks
- Generates workflow structure

---

## 2. Coder Agent

**File**: `.claude/agents/coder.md`

```yaml
---
name: coder
description: Use for writing implementation code, fixing bugs, and refactoring
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---
```

### Responsibilities
- Write implementation code
- Fix bugs
- Refactor existing code
- Follow project patterns

### Characteristics
- Has write access (Write, Edit)
- Can run commands (Bash)
- Most tools available

### Usage in Task Manager
- Executes implementation subtasks
- Writes actual code
- Applies fixes

---

## 3. Reviewer Agent

**File**: `.claude/agents/reviewer.md`

```yaml
---
name: reviewer
description: Use for code review, security audit, and quality checks
tools: Read, Grep, Glob
model: sonnet
---
```

### Responsibilities
- Code review
- Security audit
- Quality checks
- Best practice validation

### Characteristics
- Read-only access
- Cannot modify files
- Analysis focused

### Output Format
```markdown
## Review Summary
[Overall assessment]

## Issues Found
1. [Issue] - Severity: [Critical/High/Medium/Low]
   - Location: [file:line]
   - Recommendation: [fix]

## Security Concerns
[Any security issues]

## Suggestions
[Improvement ideas]
```

### Usage in Task Manager
- Reviews code after implementation
- Validates quality
- Approves or requests changes

---

## 4. Tester Agent

**File**: `.claude/agents/tester.md`

```yaml
---
name: tester
description: Use PROACTIVELY to write tests and validate implementations
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---
```

### Responsibilities
- Write tests
- Validate implementations
- Run test suites
- Report coverage

### Characteristics
- Full write access for test files
- Can run tests via Bash
- Proactive usage encouraged

### Usage in Task Manager
- Creates tests for new features
- Validates implementations
- Reports test results

---

## Agent Invocation Methods

### 1. Via Agent SDK Adapter

```python
# Direct query
result = await adapter.query(
    prompt="Design auth system",
    system_prompt="You are an architect..."
)

# Subagent invocation
result = await adapter.invoke_subagent(
    name="architect",
    prompt="Design database schema"
)

# Parallel queries
results = await adapter.parallel_query([
    {"prompt": "Review file1.py"},
    {"prompt": "Review file2.py"}
])
```

### 2. Via BBX Workflow

```yaml
steps:
  design:
    mcp: agent
    method: subagent
    inputs:
      name: architect
      prompt: "Design the feature"

  implement:
    mcp: agent
    method: subagent
    inputs:
      name: coder
      prompt: "Implement ${steps.design.output}"
    depends_on: [design]

  review:
    mcp: agent
    method: subagent
    inputs:
      name: reviewer
      prompt: "Review the implementation"
    depends_on: [implement]
```

### 3. Via A2A Protocol

```python
# If agent runs as A2A server
client = A2AClient()
card = await client.discover("http://localhost:9000")
task = await client.create_task(
    agent_url="http://localhost:9000",
    skill_id="query",
    input={"prompt": "..."}
)
```

---

## Agent Status Model

```python
class AgentStatus(Enum):
    IDLE = "idle"           # Ready, no active task
    WORKING = "working"     # Processing a task
    QUEUED = "queued"       # Has pending tasks
    ERROR = "error"         # Last task failed
```

### Status Tracking

```python
@dataclass
class AgentInfo:
    id: str                    # architect, coder, etc.
    name: str                  # Display name
    description: str           # From frontmatter
    status: AgentStatus
    current_task: Optional[str]
    tools: List[str]
    model: str

    # Metrics
    tasks_completed: int
    tasks_failed: int
    avg_duration_ms: float
    success_rate: float
    last_active_at: datetime
```

---

## Agent Workflow Patterns

### 1. Sequential Chain

```
architect → coder → reviewer → tester
```

Best for: Feature development with clear phases

### 2. Parallel Review

```
         ┌→ reviewer
coder ───┤
         └→ tester
```

Best for: Quick validation of implementation

### 3. Iterative Refinement

```
architect → coder → reviewer ──┐
              ↑                │
              └────────────────┘
              (if issues found)
```

Best for: Quality-critical features

### 4. AI Decomposition

```
User Task
    ↓
architect (decompose)
    ↓
┌───────────────────────────────┐
│ Subtask 1 → architect         │
│ Subtask 2 → coder             │ (parallel where possible)
│ Subtask 3 → coder             │
│ Subtask 4 → tester            │
└───────────────────────────────┘
    ↓
Task Complete
```

---

## Console Integration Requirements

### Agent List View

```typescript
interface AgentListItem {
  id: string;
  name: string;
  description: string;
  status: 'idle' | 'working' | 'queued' | 'error';
  currentTask?: string;
  metrics: {
    tasksCompleted: number;
    tasksFailed: number;
    avgDurationMs: number;
    successRate: number;
  };
}
```

### Agent Detail View

```typescript
interface AgentDetail extends AgentListItem {
  tools: string[];
  model: string;
  systemPrompt: string;   // Parsed from .md content
  recentTasks: TaskHistory[];
  metricsHistory: MetricPoint[];
}
```

### Agent Status WebSocket Events

```typescript
// Subscribe to agent status updates
ws.subscribe('agent:status:changed', (data) => {
  // { agentId, oldStatus, newStatus, taskId? }
});

ws.subscribe('agent:task:started', (data) => {
  // { agentId, taskId, prompt }
});

ws.subscribe('agent:task:completed', (data) => {
  // { agentId, taskId, duration, success }
});
```

---

## API Endpoints

```yaml
# List all agents
GET /api/agents
Response: AgentListItem[]

# Get agent details
GET /api/agents/{id}
Response: AgentDetail

# Get agent metrics
GET /api/agents/{id}/metrics
Query: ?period=1h|24h|7d
Response: MetricsData

# Get agent task history
GET /api/agents/{id}/tasks
Query: ?limit=20&offset=0
Response: TaskHistory[]

# Get overall stats
GET /api/agents/stats
Response: {
  totalTasks: number,
  averageSuccessRate: number,
  busyAgents: number,
  queuedTasks: number
}
```
