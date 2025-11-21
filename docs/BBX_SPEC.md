# BBX Format Specification (v5.0)

> **Complete technical specification for `.bbx` workflow files**

## 📋 Overview

BBX (Blackbox Format) is a declarative YAML-based language for defining workflow automation. It uses MCP (Model Context Protocol) adapters as execution primitives.

---

## 🏗️ File Structure

```yaml
bbx_version: "5.0"              # Required: Format version
type: workflow                  # Required: Document type

workflow:
  id: "unique_identifier"       # Required: Unique workflow ID
  name: "Human Readable Name"   # Required: Display name
  version: "1.0.0"              # Optional: Semantic version
  description: "..."            # Optional: Workflow description
  
  inputs:                       # Optional: Expected input parameters
    - name: user_id
      type: string
      required: true
      
  steps:                        # Required: Execution steps
    - id: "step1"               # Step definition
      mcp: "adapter_name"
      method: "method_name"
      inputs: {...}
      outputs: "variable_name"
```

---

## 🔧 Step Definition

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique step identifier within workflow |
| `mcp` | string | MCP adapter name (e.g., "http", "sql") |
| `method` | string | Adapter method to invoke |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `inputs` | object | `{}` | Arguments passed to the method |
| `outputs` | string | `"result"` | Variable name to save output |
| `when` | string | `null` | Conditional execution expression |
| `retry` | integer | `0` | Number of retry attempts on failure |
| `timeout` | integer | `30000` | Timeout in milliseconds |

---

## 🎯 Variable Resolution

### Syntax: `${type.path.to.value}`

Variables use the template syntax `${...}` and support dot notation for nested access.

### Variable Types

#### 1. Step Outputs (`step`)

Access data from previous steps:

```yaml
steps:
  - id: "fetch_user"
    mcp: "http"
    method: "get"
    inputs:
      url: "https://api.example.com/users/123"
    outputs: "user"
    
  - id: "greet"
    mcp: "logger"
    method: "info"
    inputs:
      message: "Hello, ${step.fetch_user.user.name}!"
```

**Resolution Path:**
- `step` → Access step outputs
- `fetch_user` → Step ID
- `user` → Output variable name
- `name` → Nested field in output

#### 2. Environment Variables (`env`)

```yaml
inputs:
  api_key: "${env.API_TOKEN}"
  db_url: "${env.DATABASE_URL}"
```

#### 3. Workflow Inputs (`input`)

```yaml
workflow:
  inputs:
    - name: user_id
      type: string
      
  steps:
    - id: "fetch"
      inputs:
        id: "${input.user_id}"
```

---

## ⚡ Conditional Execution

### `when` Clause

Execute steps conditionally using boolean expressions:

```yaml
- id: "send_alert"
  when: "${step.check_price.price} > 50000"
  mcp: "telegram"
  method: "send"
```

### Supported Operators

#### Comparison Operators
- `==` - Equality
- `!=` - Inequality
- `>` - Greater than
- `<` - Less than
- `>=` - Greater than or equal
- `<=` - Less than or equal

#### Logical Operators
- `and` - Logical AND
- `or` - Logical OR
- `not` - Logical NOT

#### Grouping
- `(...)` - Parentheses for precedence

### Examples

```yaml
# Simple equality
when: "${step.auth.status} == 'success'"

# Numeric comparison
when: "${step.fetch.count} >= 100"

# Logical operators
when: "${step.a.valid} and ${step.b.valid}"

# Complex expression
when: "(${step.price.btc} > 50000) or (${step.alerts.enabled} == true)"

# Negation
when: "not ${step.check.failed}"
```

### Expression Rules

1. **String literals** must use quotes: `'value'` or `"value"`
2. **Boolean literals**: `true`, `false` (case-insensitive)
3. **Number literals**: `123`, `45.67`
4. **Variable access**: Must use `${...}` syntax
5. **Type safety**: Comparing incompatible types raises error

---

## 🔌 MCP Adapter Interface

### Adapter Contract

All MCP adapters must implement:

```python
class MCPAdapter:
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute an adapter method.
        
        Args:
            method: Method name (e.g., "get", "post")
            inputs: Method arguments
            
        Returns:
            Method result (serializable)
        """
        pass
```

### Built-in Adapters

#### HTTP Adapter

```yaml
- id: "api_call"
  mcp: "http"
  method: "get"  # or "post", "put", "delete"
  inputs:
    url: "https://api.example.com/data"
    headers:
      Authorization: "Bearer ${env.TOKEN}"
    params:
      limit: 10
```

**Methods:**
- `get(url, headers?, params?)`
- `post(url, headers?, json?, data?)`
- `put(url, headers?, json?)`
- `delete(url, headers?)`

---

## 📊 Execution Flow

### Step Lifecycle

```
1. PARSE → Load and validate YAML
2. RESOLVE → Resolve variables in inputs
3. CHECK → Evaluate 'when' condition
4. EXECUTE → Call MCP adapter
5. STORE → Save outputs to context
6. RETRY → On failure, retry if configured
```

### Context Structure

```python
{
  "step": {
    "step_id_1": {
      "status": "success",
      "output": {...},
      "variable_name": {...}  # If 'outputs' is specified
    },
    "step_id_2": {
      "status": "error",
      "error": "Error message"
    }
  },
  "env": {
    "API_KEY": "...",
    # ... environment variables
  },
  "input": {
    "user_id": "123",
    # ... workflow inputs
  }
}
```

---

## 🛡️ Error Handling

### Step Status Values

- `success` - Step completed successfully
- `error` - Step failed (exception occurred)
- `skipped` - Step skipped (condition false)

### Error Information

When a step fails, the context stores:

```yaml
step:
  failed_step:
    status: "error"
    error: "Connection timeout after 30000ms"
```

### Error Handling Pattern

```yaml
steps:
  - id: "risky_operation"
    mcp: "http"
    method: "get"
    retry: 3
    
  - id: "handle_success"
    when: "${step.risky_operation.status} == 'success'"
    mcp: "logger"
    method: "info"
    inputs:
      message: "Operation succeeded"
      
  - id: "handle_error"
    when: "${step.risky_operation.status} == 'error'"
    mcp: "logger"
    method: "error"
    inputs:
      message: "Operation failed: ${step.risky_operation.error}"
```

---

## 🔒 Security Considerations

### Safe Expression Evaluation

BBX uses a **safe expression parser** that:
- ✅ **Prevents code injection** - No `eval()` or arbitrary code execution
- ✅ **Type-safe** - Runtime type checking
- ✅ **Sandboxed** - Only accesses provided context
- ❌ **No function calls** - Cannot execute arbitrary Python functions
- ❌ **No imports** - Cannot load external modules

### Best Practices

1. **Never hardcode secrets** - Use environment variables
2. **Validate inputs** - Check workflow inputs before use
3. **Set timeouts** - Prevent infinite execution
4. **Use HTTPS** - Secure API calls
5. **Limit retries** - Avoid resource exhaustion

---

## 📚 Complete Example

```yaml
bbx_version: "5.0"
type: workflow

workflow:
  id: "crypto_price_monitor"
  name: "Crypto Price Alert System"
  version: "1.0.0"
  description: "Monitor Bitcoin price and send Telegram alerts"
  
  inputs:
    - name: threshold
      type: number
      default: 50000
      
  steps:
    # Fetch current BTC price
    - id: "fetch_price"
      mcp: "http"
      method: "get"
      inputs:
        url: "https://api.coingecko.com/api/v3/simple/price"
        params:
          ids: "bitcoin"
          vs_currencies: "usd"
      outputs: "price_data"
      retry: 3
      timeout: 10000
      
    # Log the price
    - id: "log_price"
      when: "${step.fetch_price.status} == 'success'"
      mcp: "logger"
      method: "info"
      inputs:
        message: "Current BTC: $${step.fetch_price.price_data.bitcoin.usd}"
        
    # Send alert if threshold exceeded
    - id: "send_alert"
      when: "${step.fetch_price.price_data.bitcoin.usd} > ${input.threshold}"
      mcp: "telegram"
      method: "send_message"
      inputs:
        chat_id: "${env.TELEGRAM_CHAT_ID}"
        text: "🚀 Bitcoin price alert!\nCurrent: $${step.fetch_price.price_data.bitcoin.usd}\nThreshold: $${input.threshold}"
      retry: 2
      
    # Handle errors
    - id: "log_error"
      when: "${step.fetch_price.status} == 'error'"
      mcp: "logger"
      method: "error"
      inputs:
        message: "Failed to fetch price: ${step.fetch_price.error}"
```

---

## 🔄 Version History

### v5.0 (Current)
- **Safe expression parser** - Replaces unsafe `eval()`
- **Enhanced error handling** - Better error messages
- **Type safety** - Runtime type checking

### v4.0
- Initial public release
- Basic MCP adapter system
- HTTP adapter support

---

## 📖 See Also

- [Agent Guide](AGENT_GUIDE.md) - For AI agents using BBX
- [Runtime Internals](RUNTIME_INTERNALS.md) - How the engine works
- [MCP Development](MCP_DEVELOPMENT.md) - Creating custom adapters
