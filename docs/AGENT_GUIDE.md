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
workflow:
  id: api_orchestration
  name: API Orchestration Example
  version: "6.0"

  steps:
    - id: auth
      mcp: bbx.http
      method: post
      inputs:
        url: https://api.example.com/auth
        json:
          user: admin
          pass: ${env.PASSWORD}

    - id: fetch_data
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/data
        headers:
          Authorization: Bearer ${steps.auth.output.token}
      depends_on: [auth]
      when: "${steps.auth.output.success} == true"
```

### 2. Error Handling

```yaml
workflow:
  id: error_handling
  name: Error Handling Example
  version: "6.0"

  steps:
    - id: risky_operation
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/risky
      retry: 3
      timeout: 5000

    - id: handle_success
      when: "${steps.risky_operation.status} == 'success'"
      mcp: bbx.logger
      method: info
      inputs:
        message: "Operation succeeded"
      depends_on: [risky_operation]

    - id: handle_error
      when: "${steps.risky_operation.status} == 'error'"
      mcp: bbx.logger
      method: error
      inputs:
        message: "Operation failed: ${steps.risky_operation.error}"
      depends_on: [risky_operation]
```

### 3. Data Transformation

```yaml
workflow:
  id: data_transformation
  name: Data Transformation Example
  version: "6.0"

  steps:
    - id: fetch
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/data

    - id: extract
      mcp: bbx.transform
      method: map
      inputs:
        data: ${steps.fetch.output}
        function: "lambda x: x['name']"
      depends_on: [fetch]
```

---

## ⚡ Best Practices

### 1. **Use Descriptive IDs**
```yaml
# ✅ Good
- id: fetch_user_profile
- id: validate_email

# ❌ Bad
- id: step1
- id: do_thing
```

### 2. **Handle Errors**
Always add error handling steps:
```yaml
- id: log_error
  when: "${steps.main_task.status} == 'error'"
  mcp: bbx.logger
  method: error
```

### 3. **Use Environment Variables**
Never hardcode secrets:
```yaml
# ✅ Good
api_key: ${env.API_KEY}

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
ExpressionError: Variable 'steps.fetch.output' not found
```
**Fix:** Check that the step ID and output path exist. In v6.0, use `steps.` prefix.

**2. MCP Not Registered**
```
ValueError: Unknown MCP type: bbx.telegram
```
**Fix:** Ensure the adapter is available. All built-in adapters use `bbx.` prefix in v6.0.

**3. Condition Syntax Error**
```
ExpressionError: Unknown value: 'success'
```
**Fix:** Use quotes for string literals:
```yaml
when: "${steps.fetch.status} == 'success'"  # ✅
when: "${steps.fetch.status} == success"     # ❌
```

---

## 📚 Examples

### Complete Workflow

```yaml
workflow:
  id: price_monitor
  name: Crypto Price Monitor
  version: "6.0"
  description: Monitor Bitcoin price and send alerts

  steps:
    - id: fetch_price
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.coingecko.com/api/v3/simple/price
        params:
          ids: bitcoin
          vs_currencies: usd
      retry: 3
      timeout: 10000

    - id: log_price
      mcp: bbx.logger
      method: info
      inputs:
        message: "Current BTC: $${steps.fetch_price.output.bitcoin.usd}"
      depends_on: [fetch_price]

    - id: send_alert
      when: "${steps.fetch_price.output.bitcoin.usd} > 50000"
      mcp: bbx.telegram
      method: send_message
      inputs:
        chat_id: ${env.TELEGRAM_CHAT_ID}
        text: "🚀 BTC > $50k! Current: $${steps.fetch_price.output.bitcoin.usd}"
      depends_on: [fetch_price]
```

---

## 🔐 Security

- **Never commit `.env` files**
- **Use environment variables for secrets**
- **Validate all external inputs**
- **Set timeouts on network calls**
- **Use HTTPS for API calls**

---

## 📖 See Also

- [BBX v6.0 Specification](BBX_SPEC_v6.md) - Complete workflow format reference
- [Universal Adapter Guide](UNIVERSAL_ADAPTER.md) - Zero-code integrations
- [Runtime Internals](RUNTIME_INTERNALS.md) - How the engine works
- [MCP Development](MCP_DEVELOPMENT.md) - Creating custom adapters
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Getting Started](GETTING_STARTED.md) - Quick start guide for beginners

---

**Questions?** Check the [Documentation Index](INDEX.md) or examples in `workflows/` directory.
