# Blackbox Agent Guide

> **Complete reference for AI agents working with Blackbox workflows**

## 🎯 Quick Start

Blackbox is a workflow automation engine that executes declarative `.bbx` files.

**Core Concept:** Write YAML workflows, not Python code.

---

## 📝 BBX Format (v6.0)

### Basic Structure

```yaml
workflow:
  id: unique_id
  name: Human Readable Name
  version: "6.0"
  description: Optional workflow description

  steps:
    - id: step1
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com
```

### Step Anatomy

```yaml
- id: fetch_data              # Unique step identifier
  mcp: bbx.http               # MCP adapter to use (v6.0 uses bbx. prefix)
  method: get                 # Adapter method
  inputs:                     # Method arguments
    url: https://api.example.com
  timeout: 30000              # Timeout in ms
  retry: 3                    # Retry on failure
  depends_on: []              # Dependencies (optional)
```

---

## 🔧 Variable Resolution

### Syntax: `${steps.step_id.field}` or `${env.VARIABLE}` or `${inputs.param}`

| Type | Example | Description |
|------|---------|-------------|
| `steps` | `${steps.fetch.output}` | Output from previous step |
| `env` | `${env.API_KEY}` | Environment variable |
| `inputs` | `${inputs.user_id}` | Workflow input parameter |

### Examples

```yaml
# Access step output
message: "Hello ${steps.fetch_user.output.name}"

# Access nested data
price: ${steps.get_price.output.bitcoin.usd}

# Environment variable
token: ${env.API_TOKEN}

# Workflow inputs
user_id: ${inputs.user_id}
```

---

## 🎭 Conditional Execution

### `when` Clause

```yaml
- id: send_alert
  when: "${steps.check_price.output} > 50000"
  mcp: bbx.telegram
  method: send_message
  inputs:
    chat_id: ${env.TELEGRAM_CHAT_ID}
    text: "Price alert!"
```

### Supported Operators

- **Comparison:** `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical:** `and`, `or`, `not`
- **Literals:** `'string'`, `123`, `true`, `false`

### Examples

```yaml
# Simple equality
when: "${steps.fetch.status} == 'success'"

# Numeric comparison
when: "${steps.count.output} >= 100"

# Logical operators
when: "${steps.a.output.valid} and ${steps.b.output.valid}"

# Complex condition
when: "(${steps.price.output.btc} > 50000) or (${steps.price.output.eth} > 3000)"
```

---

## 📦 MCP Adapters

### HTTP Adapter

```yaml
- id: api_call
  mcp: bbx.http
  method: get
  inputs:
    url: https://api.example.com/data
    headers:
      Authorization: Bearer ${env.API_TOKEN}
    params:
      limit: 10
```

**Available Methods:**
- `get` - HTTP GET request
- `post` - HTTP POST request
- `put` - HTTP PUT request
- `delete` - HTTP DELETE request
- `patch` - HTTP PATCH request
- `download` - Download file

---

## 🚀 Running Workflows

### CLI

```bash
# Run workflow file
python cli.py run my-flow.bbx

# Run with inputs
python cli.py run my-flow.bbx --input key=value
```

### Python

```python
from blackbox.core import run_file
import asyncio

result = asyncio.run(run_file("workflow.bbx"))
print(result)
```

### Server

```bash
# Start server
uvicorn blackbox.server.app:app --port 8000

# Execute via HTTP
curl -X POST http://localhost:8000/api/execute/my-flow \
  -H "Content-Type: application/json" \
  -d '{"workflow_id": "my-flow", "inputs": {}}'
```

---

## 🏗️ Common Patterns

### 1. API Orchestration

```yaml
steps:
  - id: "auth"
    mcp: "http"
    method: "post"
    inputs:
      url: "https://api.example.com/auth"
      json: {user: "admin", pass: "${env.PASSWORD}"}
    outputs: "token"
    
  - id: "fetch_data"
    mcp: "http"
    method: "get"
    inputs:
      url: "https://api.example.com/data"
      headers:
        Authorization: "Bearer ${step.auth.token}"
    when: "${step.auth.status} == 'success'"
```

### 2. Error Handling

```yaml
steps:
  - id: "risky_operation"
    mcp: "http"
    method: "get"
    inputs: {...}
    retry: 3
    
  - id: "handle_success"
    when: "${step.risky_operation.status} == 'success'"
    mcp: "http"
    method: "post"
    inputs: {...}
    
  - id: "handle_error"
    when: "${step.risky_operation.status} == 'error'"
    mcp: "logger"
    method: "error"
    inputs:
      message: "Operation failed: ${step.risky_operation.error}"
```

### 3. Data Transformation

```yaml
steps:
  - id: "fetch"
    mcp: "http"
    method: "get"
    inputs: {url: "..."}
    outputs: "raw_data"
    
  - id: "extract"
    mcp: "transform"
    method: "jsonpath"
    inputs:
      data: "${step.fetch.raw_data}"
      path: "$.items[*].name"
    outputs: "names"
```

---

## ⚡ Best Practices

### 1. **Use Descriptive IDs**
```yaml
# ✅ Good
- id: "fetch_user_profile"
- id: "validate_email"

# ❌ Bad
- id: "step1"
- id: "do_thing"
```

### 2. **Handle Errors**
Always add error handling steps:
```yaml
- id: "log_error"
  when: "${step.main_task.status} == 'error'"
```

### 3. **Use Environment Variables**
Never hardcode secrets:
```yaml
# ✅ Good
api_key: "${env.API_KEY}"

# ❌ Bad
api_key: "sk-1234567890abcdef"
```

### 4. **Add retry/timeout**
For network calls:
```yaml
retry: 3
timeout: 30000
```

---

## 🐛 Troubleshooting

### Common Errors

**1. Variable Not Found**
```
ExpressionError: Variable 'step.fetch.data' not found
```
**Fix:** Check that the step ID and output key exist.

**2. MCP Not Registered**
```
ValueError: Unknown MCP type: telegram
```
**Fix:** Ensure the adapter is registered:
```python
registry.register("telegram", TelegramAdapter())
```

**3. Condition Syntax Error**
```
ExpressionError: Unknown value: 'success'
```
**Fix:** Use quotes for string literals:
```yaml
when: "${step.fetch.status} == 'success'"  # ✅
when: "${step.fetch.status} == success"     # ❌
```

---

## 📚 Examples

### Complete Workflow

```yaml
bbx_version: "5.0"
type: workflow

workflow:
  id: "price_monitor"
  name: "Crypto Price Monitor"
  version: "1.0.0"
  
  steps:
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
      
    - id: "check_threshold"
      when: "${step.fetch_price.status} == 'success'"
      mcp: "logic"
      method: "evaluate"
      inputs:
        condition: "${step.fetch_price.price_data.bitcoin.usd} > 50000"
      outputs: "alert_needed"
      
    - id: "send_alert"
      when: "${step.check_threshold.alert_needed} == true"
      mcp: "telegram"
      method: "send_message"
      inputs:
        chat_id: "${env.TELEGRAM_CHAT_ID}"
        text: "🚀 BTC > $50k! Current: $${step.fetch_price.price_data.bitcoin.usd}"
```

---

## 🔐 Security

- **Never commit `.env` files**
- **Use environment variables for secrets**
- **Validate all external inputs**
- **Set timeouts on network calls**
- **Use HTTPS for API calls**

---

## 📖 Reference

- [BBX Specification](BBX_SPEC.md)
- [Runtime Internals](RUNTIME_INTERNALS.md)
- [MCP Development](MCP_DEVELOPMENT.md)
- [Deployment Guide](DEPLOYMENT.md)

---

**Questions?** Check examples in `workflows/` directory.
