# BBX Workflows Analysis

## BBX v6 Workflow Format

### Basic Structure

```yaml
# workflow.bbx
bbx: "6.0"

workflow:
  id: my-workflow
  name: My Workflow
  description: Description of what this workflow does

  inputs:
    param1:
      type: string
      required: true
    param2:
      type: string
      default: "value"

  steps:
    - id: step1
      mcp: adapter_name
      method: method_name
      inputs:
        key: value

    - id: step2
      mcp: adapter_name
      method: method_name
      inputs:
        key: ${steps.step1.output}
      depends_on: [step1]
```

---

## Step Configuration

### Required Fields

```yaml
- id: unique_step_id          # Required, string
  mcp: adapter_type           # Required, string
  method: method_name         # Required, string
```

### Optional Fields

```yaml
  inputs: {}                  # Input parameters

  # Dependencies
  depends_on: [step1, step2]  # Array of step IDs
  parallel: true              # Can run in parallel

  # Conditions
  when: "${inputs.flag} == true"

  # Timing
  timeout: 30000              # Milliseconds

  # Retry
  retry: 3                    # Retry count
  retry_delay: "2s"           # Delay between retries
  retry_backoff: 2            # Backoff multiplier

  # Error Handling
  fallback:
    adapter: fallback_adapter
    action: fallback_method
    params: {}

  on_failure:
    - adapter: notify
      action: send
      params:
        message: "Step failed"

  pre_check:
    adapter: checker
    action: validate
    params: {}
    on_failure:
      - adapter: setup
        action: prepare
```

---

## Expression Syntax

### Variable Access

```yaml
# Input parameters
${inputs.param1}

# Previous step output
${steps.step1.output}
${steps.step1.output.nested.field}

# Workspace state
${state.key}
```

### SafeExpr Evaluation

For `when` conditions:

```yaml
when: "${inputs.count} > 5"
when: "${steps.check.output.valid} == true"
when: "steps.step1.status == 'success'"
```

Supported operators: `==`, `!=`, `>`, `<`, `>=`, `<=`, `and`, `or`, `not`

---

## DAG Execution

### Execution Levels

```yaml
steps:
  # Level 0 (parallel)
  - id: a
    mcp: http
    method: get
    inputs: {url: "..."}

  - id: b
    mcp: http
    method: get
    inputs: {url: "..."}

  # Level 1 (after a and b)
  - id: c
    mcp: process
    method: merge
    inputs:
      data1: ${steps.a.output}
      data2: ${steps.b.output}
    depends_on: [a, b]

  # Level 2 (after c)
  - id: d
    mcp: file
    method: write
    inputs:
      path: result.json
      content: ${steps.c.output}
    depends_on: [c]
```

### Visualization Data

```json
{
  "levels": [
    ["a", "b"],     // Level 0: parallel
    ["c"],          // Level 1: after both
    ["d"]           // Level 2: after c
  ],
  "nodes": {
    "a": {"id": "a", "mcp": "http", "method": "get"},
    "b": {"id": "b", "mcp": "http", "method": "get"},
    "c": {"id": "c", "mcp": "process", "method": "merge"},
    "d": {"id": "d", "mcp": "file", "method": "write"}
  },
  "edges": [
    {"from": "a", "to": "c"},
    {"from": "b", "to": "c"},
    {"from": "c", "to": "d"}
  ]
}
```

---

## Workflow Execution States

### Step States

```python
class StepStatus(Enum):
    PENDING = "pending"       # Not started
    WAITING = "waiting"       # Waiting for dependencies
    RUNNING = "running"       # Currently executing
    SUCCESS = "success"       # Completed successfully
    FAILED = "failed"         # Execution failed
    SKIPPED = "skipped"       # Skipped due to 'when' condition
    TIMEOUT = "timeout"       # Timed out
    CANCELLED = "cancelled"   # Cancelled by user
```

### Workflow States

```python
class WorkflowStatus(Enum):
    PENDING = "pending"       # Not started
    RUNNING = "running"       # Steps executing
    COMPLETED = "completed"   # All steps done
    FAILED = "failed"         # One or more steps failed
    CANCELLED = "cancelled"   # Cancelled by user
```

---

## Event Types

From `blackbox/core/events.py`:

```python
class EventType(Enum):
    WORKFLOW_START = "workflow:start"
    WORKFLOW_END = "workflow:end"
    STEP_START = "step:start"
    STEP_END = "step:end"
    STEP_ERROR = "step:error"
```

### Event Payloads

```python
# WORKFLOW_START
{"id": "workflow_id"}

# WORKFLOW_END
{"id": "workflow_id", "results": {...}}

# STEP_START
{"step_id": "step1", "workflow_id": "workflow_id"}

# STEP_END
{"step_id": "step1", "output": {...}}

# STEP_ERROR
{"step_id": "step1", "error": "error message"}
```

---

## Console Workflow Manager Requirements

### 1. Workflow List

```typescript
interface WorkflowListItem {
  id: string;
  name: string;
  description?: string;
  filePath: string;
  stepCount: number;
  lastRun?: {
    status: WorkflowStatus;
    startedAt: Date;
    duration: number;
  };
}
```

### 2. Workflow Detail

```typescript
interface WorkflowDetail extends WorkflowListItem {
  inputs: InputDefinition[];
  steps: StepDefinition[];
  dag: DAGVisualization;
}

interface InputDefinition {
  name: string;
  type: string;
  required: boolean;
  default?: any;
}

interface StepDefinition {
  id: string;
  mcp: string;
  method: string;
  inputs: Record<string, any>;
  dependsOn: string[];
  timeout?: number;
  retry?: number;
}
```

### 3. DAG Visualization

```typescript
interface DAGVisualization {
  nodes: DAGNode[];
  edges: DAGEdge[];
  levels: string[][];
}

interface DAGNode {
  id: string;
  label: string;
  status: StepStatus;
  level: number;
  position: {x: number; y: number};
  data: {
    mcp: string;
    method: string;
    duration?: number;
    error?: string;
  };
}

interface DAGEdge {
  source: string;
  target: string;
  animated?: boolean;  // true if target is running
}
```

### 4. Execution View

```typescript
interface ExecutionState {
  workflowId: string;
  executionId: string;
  status: WorkflowStatus;
  startedAt: Date;
  completedAt?: Date;
  steps: {
    [stepId: string]: {
      status: StepStatus;
      startedAt?: Date;
      completedAt?: Date;
      duration?: number;
      output?: any;
      error?: string;
      retryCount?: number;
    };
  };
  currentLevel: number;
  progress: number;  // 0-100%
}
```

---

## API Endpoints

```yaml
# List workflows
GET /api/workflows
Response: WorkflowListItem[]

# Get workflow
GET /api/workflows/{id}
Response: WorkflowDetail

# Validate workflow
POST /api/workflows/validate
Body: {content: string}
Response: {valid: boolean, errors?: string[]}

# Run workflow
POST /api/workflows/{id}/run
Body: {inputs: Record<string, any>}
Response: {executionId: string}

# Get execution status
GET /api/executions/{id}
Response: ExecutionState

# Cancel execution
POST /api/executions/{id}/cancel
Response: {success: boolean}

# List executions
GET /api/executions
Query: ?workflowId=xxx&status=running&limit=20
Response: ExecutionState[]

# Get execution logs
GET /api/executions/{id}/logs
Response: LogEntry[]
```

---

## WebSocket Events

```typescript
// Subscribe to execution updates
ws.subscribe('workflow:execution:started', (data) => {
  // {executionId, workflowId, inputs}
});

ws.subscribe('workflow:step:started', (data) => {
  // {executionId, stepId, level}
});

ws.subscribe('workflow:step:completed', (data) => {
  // {executionId, stepId, duration, output}
});

ws.subscribe('workflow:step:failed', (data) => {
  // {executionId, stepId, error, retryCount}
});

ws.subscribe('workflow:execution:completed', (data) => {
  // {executionId, status, duration, results}
});
```

---

## Workflow YAML Editor Features

### Monaco Editor Configuration

```typescript
const monacoConfig = {
  language: 'yaml',
  theme: 'vs-dark',
  minimap: {enabled: false},
  lineNumbers: 'on',
  formatOnPaste: true,
  formatOnType: true,
  automaticLayout: true,
};
```

### Custom Validation

```typescript
// Validate BBX syntax
function validateBBX(content: string): ValidationResult {
  // 1. YAML syntax
  // 2. Required fields (id, mcp, method)
  // 3. Dependency cycles
  // 4. Unknown adapters
  // 5. Expression syntax
}
```

### Autocomplete Providers

```typescript
// Adapter names
const adapters = ['http', 'file', 'agent', 'shell', 'python', ...];

// Methods per adapter
const adapterMethods = {
  http: ['get', 'post', 'put', 'delete'],
  file: ['read', 'write', 'append', 'delete'],
  agent: ['query', 'subagent', 'parallel'],
  // ...
};

// Expression templates
const expressions = [
  '${inputs.}',
  '${steps..output}',
  '${state.}',
];
```

---

## Visual DAG Editor Features

### Node Types

- **Start Node**: Entry point
- **Step Node**: Regular step with adapter/method
- **End Node**: Exit point
- **Parallel Group**: Visual grouping of parallel steps

### Interactions

- Drag nodes to reposition
- Click node to select/edit
- Drag from node to create edge
- Double-click to edit step details
- Delete key to remove selected

### Real-time Sync

```typescript
// Editor changes → YAML update
onNodeMove(nodeId, position) {
  // Update visual layout only (not workflow logic)
  updateLayoutMetadata(nodeId, position);
}

onEdgeCreate(source, target) {
  // Update depends_on in YAML
  addDependency(target, source);
}

// YAML changes → Editor update
onYamlChange(content) {
  const workflow = parseYaml(content);
  updateDAGVisualization(workflow);
}
```
