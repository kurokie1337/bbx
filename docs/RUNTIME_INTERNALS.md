# Blackbox Runtime Internals

> **Technical deep-dive into the workflow execution engine**

## 🏗️ Architecture Overview

Blackbox follows a **layered modular architecture**:

```
┌─────────────────────────────────────┐
│     CLI / API / Library             │  Entry Points
├─────────────────────────────────────┤
│     Workflow Runtime (runtime.py)   │  Execution Engine
├─────────────────────────────────────┤
│     Context + Registry + Events     │  Core Services
├─────────────────────────────────────┤
│     MCP Adapters (Plugins)          │  Integration Layer
└─────────────────────────────────────┘
```

---

## ⚙️ Execution Engine (`runtime.py`)

### Entry Point: `run_file()`

```python
async def run_file(file_path: str, event_bus: Optional[EventBus] = None) -> Dict[str, Any]:
    """Execute a BBX workflow file"""
```

**Execution Flow:**

```
1. LOAD → Parse YAML file
2. INIT → Create context, registry, event bus
3. REGISTER → Register MCP adapters
4. ITERATE → For each step:
   ├─ EMIT → STEP_START event
   ├─ RESOLVE → Resolve ${...} variables in inputs
   ├─ CHECK → Evaluate 'when' condition
   ├─ EXECUTE → Call MCP adapter method
   ├─ STORE → Save output to context
   └─ EMIT → STEP_END event
5. RETURN → Result dictionary
```

### Step Resolution

```python
# Variable resolution
resolved_inputs = {
    k: context.resolve(v) if isinstance(v, str) else v
    for k, v in inputs.items()
}

# Condition evaluation
if when_condition:
    resolved_condition = context.resolve(when_condition)
    if not SafeExpr.evaluate(resolved_condition, context.variables):
        # Skip step
        continue
```

---

## 🗂️ Context Management (`context.py`)

### WorkflowContext Class

Manages workflow state and variable resolution:

```python
class WorkflowContext:
    def __init__(self):
        self.variables = {
            "step": {},    # Step outputs
            "env": {},     # Environment variables
            "input": {}    # Workflow inputs
        }
    
    def resolve(self, template: str) -> str:
        """Resolve ${type.path.to.value} templates"""
        
    def set_step_output(self, step_id: str, data: Dict[str, Any]):
        """Store step execution result"""
```

### Variable Resolution Algorithm

```python
def resolve(self, template: str) -> str:
    # Pattern: ${type.path.to.value}
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        path = match.group(1).split('.')
        # Walk the context tree
        value = self.variables
        for key in path:
            value = value.get(key)
        return str(value)
    
    return re.sub(pattern, replacer, template)
```

**Example:**

```python
context.variables = {
    "step": {
        "fetch": {
            "status": "success",
            "data": {"price": 42}
        }
    }
}

# Resolution
context.resolve("Price: ${step.fetch.data.price}")
# Result: "Price: 42"
```

---

## 🔌 MCP Registry (`registry.py`)

### Adapter Management

```python
class MCPRegistry:
    def __init__(self):
        self._adapters: Dict[str, MCPAdapter] = {}
    
    def register(self, name: str, adapter: MCPAdapter):
        """Register a new MCP adapter"""
        self._adapters[name] = adapter
    
    def get_adapter(self, name: str) -> Optional[MCPAdapter]:
        """Retrieve adapter by name"""
        return self._adapters.get(name)
```

**Usage:**

```python
registry = MCPRegistry()
registry.register("http", LocalHttpAdapter())
registry.register("sql", SQLAdapter())

adapter = registry.get_adapter("http")
result = await adapter.execute("get", {"url": "..."})
```

---

## 🎭 Safe Expression Parser (`expressions.py`)

### Design Goals

1. **Security** - No arbitrary code execution
2. **Type Safety** - Runtime type checking
3. **Simplicity** - Limited but sufficient operators

### Evaluation Algorithm

```python
class SafeExpr:
    @staticmethod
    def evaluate(expr: str, context: Dict[str, Any]) -> bool:
        # 1. Handle logical operators (and, or, not)
        if ' and ' in expr:
            return all(evaluate(part) for part in expr.split(' and '))
        
        if ' or ' in expr:
            return any(evaluate(part) for part in expr.split(' or '))
        
        if expr.startswith('not '):
            return not evaluate(expr[4:])
        
        # 2. Handle comparison operators (==, !=, >, <, >=, <=)
        for op in ['==', '!=', '>', '<', '>=', '<=']:
            if op in expr:
                lhs, rhs = expr.split(op, 1)
                return compare(lhs, op, rhs)
        
        # 3. Return boolean value
        return resolve_value(expr)
```

### Value Resolution

```python
def _resolve_value(value: str, context: Dict[str, Any]) -> Any:
    value = value.strip()
    
    # String literal
    if value.startswith("'") or value.startswith('"'):
        return value[1:-1]
    
    # Boolean literal
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Number literal
    try:
        return float(value) if '.' in value else int(value)
    except ValueError:
        pass
    
    # Variable access (dot notation)
    if '.' in value:
        parts = value.split('.')
        current = context
        for part in parts:
            current = current.get(part)
        return current
    
    # Simple variable
    return context.get(value)
```

**Example:**

```python
context = {
    "step": {"fetch": {"price": 100}},
    "threshold": 50
}

# Expression evaluation
SafeExpr.evaluate("step.fetch.price > threshold", context)
# Result: True (100 > 50)
```

---

## 📡 Event System (`events.py`)

### Event Types

```python
class EventType(Enum):
    WORKFLOW_START = "workflow.start"
    WORKFLOW_END = "workflow.end"
    STEP_START = "step.start"
    STEP_END = "step.end"
    STEP_ERROR = "step.error"
```

### Event Bus

```python
class EventBus:
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
    
    async def emit(self, event: Event):
        """Emit event to all listeners"""
        for listener in self._listeners.get(event.type, []):
            await listener(event)
    
    def on(self, event_type: EventType, handler: Callable):
        """Register event listener"""
        self._listeners.setdefault(event_type, []).append(handler)
```

**Usage:**

```python
event_bus = EventBus()

# Register listener
async def on_step_start(event: Event):
    print(f"Starting step: {event.data['step_id']}")

event_bus.on(EventType.STEP_START, on_step_start)

# Emit event
await event_bus.emit(Event(
    EventType.STEP_START,
    {"step_id": "fetch_data"}
))
```

---

## 🔧 MCP Adapter Protocol

### Base Interface

```python
class MCPAdapter:
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute adapter method.
        
        Args:
            method: Method name (e.g., "get", "post")
            inputs: Method arguments
            
        Returns:
            Serializable result
        """
        raise NotImplementedError
```

### HTTP Adapter Implementation

```python
class LocalHttpAdapter(MCPAdapter):
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        url = inputs.get("url")
        headers = inputs.get("headers", {})
        
        if method == "get":
            response = await self.client.get(url, headers=headers)
        elif method == "post":
            json_data = inputs.get("json")
            response = await self.client.post(url, headers=headers, json=json_data)
        # ... other methods
        
        return {
            "status": response.status_code,
            "data": response.json() if response.headers.get("content-type") == "application/json" else response.text
        }
```

---

## 🚀 Performance Optimizations

### 1. Async Execution

All I/O operations use `async/await`:

```python
# Workflow execution
result = await run_file("workflow.bbx")

# Adapter methods
output = await adapter.execute(method, inputs)
```

### 2. Lazy Loading

Adapters initialized only when needed:

```python
# Registry only stores references
registry.register("sql", SQLAdapter)  # Not instantiated yet

# Adapter created on first use
adapter = registry.get_adapter("sql")
```

### 3. Connection Pooling

HTTP adapter reuses connections:

```python
class LocalHttpAdapter:
    def __init__(self):
        # Shared client with connection pool
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100)
        )
```

---

## 🐛 Error Handling

### Error Propagation

```python
try:
    output = await adapter.execute(method, inputs)
    results[step_id] = {"status": "success", "output": output}
    
except Exception as e:
    # Log error
    traceback.print_exc()
    
    # Store error in context
    error_data = {"status": "error", "error": str(e)}
    results[step_id] = error_data
    context.set_step_output(step_id, error_data)
    
    # Emit error event
    await event_bus.emit(Event(EventType.STEP_ERROR, {
        "step_id": step_id,
        "error": str(e)
    }))
```

### Retry Logic

```python
retry_count = step.get("retry", 0)
for attempt in range(retry_count + 1):
    try:
        output = await adapter.execute(method, inputs)
        break  # Success
    except Exception as e:
        if attempt == retry_count:
            raise  # Final attempt failed
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

---

## 📊 Execution Trace Example

```
🚀 Starting: Crypto Price Monitor
  ▶️  Executing fetch_price (http.get)
  ✅ fetch_price completed
  ▶️  Executing log_price (logger.info)
  ✅ log_price completed
  ⏭️  Skipping send_alert (condition false)
  
Results:
  fetch_price: success
  log_price: success
  send_alert: skipped
```

---

## 🔮 Future Enhancements

### 1. Parallel Execution

DAG-based dependency resolution:

```yaml
steps:
  - id: fetch_a
    parallel: true
  - id: fetch_b
    parallel: true
  - id: combine
    depends_on: [fetch_a, fetch_b]
```

### 2. Caching

Step output caching:

```yaml
- id: expensive_query
  cache:
    ttl: 3600  # 1 hour
    key: "query:${input.id}"
```

### 3. Observability

Distributed tracing integration:

```python
with tracer.start_as_current_span("workflow.execute"):
    result = await run_file("workflow.bbx")
```

---

## 📚 See Also

- [BBX Specification](BBX_SPEC.md) - File format details
- [Agent Guide](AGENT_GUIDE.md) - For AI agents
- [MCP Development](MCP_DEVELOPMENT.md) - Creating adapters
