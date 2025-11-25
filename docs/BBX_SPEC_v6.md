# BBX Format Specification v6.0

> **Complete technical specification for `.bbx` workflow files - BBX v6.0**

## 📋 Overview

BBX (Blackbox Format) is a declarative YAML-based language for defining workflow automation. Version 6.0 introduces simplified syntax, better adapter naming, and improved variable resolution.

**Version:** 6.0
**Status:** Current
**Release Date:** November 26, 2025

---

## 🔄 What's New in v6.0

### Key Changes from v5.0

1. **Removed top-level `bbx_version` and `type` fields**
   - Cleaner syntax, version now in workflow metadata

2. **Adapter naming uses `bbx.` prefix**
   - Old: `mcp: "http"`
   - New: `mcp: bbx.http`

3. **Variable syntax changed**
   - Old: `${step.step_id.field}`
   - New: `${steps.step_id.output.field}`

4. **Simplified step structure**
   - Removed `outputs` field (implicit in `output`)
   - Better `depends_on` support

5. **Enhanced built-in adapters**
   - 28+ production-ready adapters with `bbx.` prefix
   - Universal Adapter for Docker-based tools

---

## 🏗️ File Structure

### Basic Workflow

```yaml
workflow:
  id: unique_identifier       # Required: Unique workflow ID
  name: Human Readable Name   # Required: Display name
  version: "6.0"              # Required: Format version
  description: Optional text  # Optional: Workflow description

  inputs:                     # Optional: Input parameters
    api_key:
      type: string
      required: true
      default: ""

  steps:                      # Required: Execution steps
    - id: step1              # Step definition
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com
      timeout: 30000
      retry: 3
```

---

## 📦 Workflow Metadata

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique workflow identifier (lowercase, underscores) |
| `name` | string | Human-readable workflow name |
| `version` | string | BBX format version (must be "6.0") |
| `steps` | array | List of execution steps (at least one required) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | `""` | Workflow description |
| `inputs` | object | `{}` | Expected input parameters |
| `outputs` | object | `{}` | Workflow output mapping |
| `tags` | array | `[]` | Metadata tags for categorization |

### Example

```yaml
workflow:
  id: data_pipeline
  name: Data Processing Pipeline
  version: "6.0"
  description: Fetch, transform, and store data from external API

  tags:
    - data
    - etl
    - production

  inputs:
    api_url:
      type: string
      required: true
      description: API endpoint URL

    batch_size:
      type: integer
      required: false
      default: 100
      description: Number of records per batch
```

---

## 🔧 Step Definition

### Required Step Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique step identifier within workflow |
| `mcp` | string | MCP adapter name (e.g., bbx.http, bbx.docker) |
| `method` | string | Adapter method to invoke |

### Optional Step Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `inputs` | object | `{}` | Arguments passed to the method |
| `when` | string | `null` | Conditional execution expression |
| `depends_on` | array | `[]` | List of step IDs this step depends on |
| `retry` | integer | `0` | Number of retry attempts on failure |
| `retry_delay` | integer | `1000` | Initial retry delay in milliseconds |
| `retry_backoff` | float | `2.0` | Retry delay multiplier (exponential backoff) |
| `timeout` | integer | `30000` | Timeout in milliseconds |
| `parallel` | boolean | `false` | Allow parallel execution with other steps |

### Full Example

```yaml
- id: fetch_data
  mcp: bbx.http
  method: get
  inputs:
    url: https://api.example.com/data
    headers:
      Authorization: Bearer ${env.API_TOKEN}
    params:
      limit: 100
      offset: ${inputs.offset}
  when: "${inputs.enabled} == true"
  depends_on: [validate_credentials]
  retry: 3
  retry_delay: 1000
  retry_backoff: 2.0
  timeout: 30000
  parallel: false
```

---

## 🎯 Variable Resolution

### Syntax: `${source.path.to.value}`

BBX v6.0 supports three variable sources:

### 1. Step Outputs (`steps`)

Access data from previous step executions:

```yaml
steps:
  - id: fetch_user
    mcp: bbx.http
    method: get
    inputs:
      url: https://api.example.com/users/123

  - id: greet
    mcp: bbx.logger
    method: info
    inputs:
      message: "Hello, ${steps.fetch_user.output.name}!"
    depends_on: [fetch_user]
```

**Resolution Path:**
- `steps` → Access step results
- `fetch_user` → Step ID
- `output` → Step's output data
- `name` → Nested field in output

**Special Fields:**
- `${steps.step_id.status}` → Step status (success, error, skipped)
- `${steps.step_id.error}` → Error message (if failed)
- `${steps.step_id.output}` → Full output data

### 2. Environment Variables (`env`)

Access system environment variables:

```yaml
inputs:
  api_key: ${env.API_TOKEN}
  db_url: ${env.DATABASE_URL}
  debug: ${env.DEBUG}
```

**Best Practice:** Always use environment variables for secrets.

### 3. Workflow Inputs (`inputs`)

Access parameters passed to workflow:

```yaml
workflow:
  inputs:
    user_id:
      type: string
      required: true

  steps:
    - id: fetch
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/users/${inputs.user_id}
```

### Variable Resolution Rules

1. **Variables are resolved at runtime** before step execution
2. **Nested access supported** via dot notation (up to 10 levels)
3. **Type preservation** - numbers, booleans, strings, objects, arrays
4. **No circular references** - depends_on prevents circular dependencies
5. **Error handling** - undefined variables cause step failure with clear error message

### Complex Example

```yaml
workflow:
  id: complex_variables
  version: "6.0"

  inputs:
    environment:
      type: string
      default: production

  steps:
    - id: get_config
      mcp: bbx.http
      method: get
      inputs:
        url: https://config.example.com/${inputs.environment}

    - id: deploy
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: deployment.yaml
        namespace: ${steps.get_config.output.namespace}
        replicas: ${steps.get_config.output.replicas}
        image: myapp:${steps.get_config.output.version}
      depends_on: [get_config]

    - id: notify
      mcp: bbx.telegram
      method: send_message
      inputs:
        chat_id: ${env.TELEGRAM_CHAT_ID}
        text: |
          Deployment complete!
          Environment: ${inputs.environment}
          Namespace: ${steps.get_config.output.namespace}
          Replicas: ${steps.deploy.output.actual_replicas}
      depends_on: [deploy]
```

---

## ⚡ Conditional Execution

### `when` Clause

Execute steps conditionally using boolean expressions:

```yaml
- id: send_alert
  when: "${steps.check_price.output.price} > 50000"
  mcp: bbx.telegram
  method: send_message
  inputs:
    chat_id: ${env.TELEGRAM_CHAT_ID}
    text: "Price alert!"
  depends_on: [check_price]
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

### Condition Examples

```yaml
# Simple equality
when: "${steps.auth.status} == 'success'"

# Numeric comparison
when: "${steps.fetch.output.count} >= 100"

# Boolean check
when: "${steps.validate.output.is_valid} == true"

# Logical operators
when: "${steps.a.output.valid} and ${steps.b.output.valid}"

# Complex expression
when: "(${steps.price.output.btc} > 50000) or (${inputs.force_alert} == true)"

# Negation
when: "not ${steps.check.output.failed}"

# String comparison
when: "${inputs.environment} == 'production'"
```

### Expression Rules

1. **String literals** must use quotes: `'value'` or `"value"`
2. **Boolean literals**: `true`, `false` (case-insensitive)
3. **Number literals**: `123`, `45.67`
4. **Variable access**: Must use `${...}` syntax
5. **Type safety**: Comparing incompatible types causes error
6. **Short-circuit evaluation**: `and`, `or` use short-circuit logic

---

## 🔌 Built-in Adapters

BBX v6.0 includes 28+ production-ready adapters. All use the `bbx.` prefix.

### Cloud Providers

#### bbx.aws
```yaml
- id: launch_ec2
  mcp: bbx.aws
  method: ec2_launch
  inputs:
    image_id: ami-12345
    instance_type: t3.micro
    key_name: my-keypair
```

**Methods:** `ec2_launch`, `ec2_terminate`, `s3_upload`, `s3_download`, `s3_list`, `lambda_invoke`

#### bbx.gcp
```yaml
- id: create_vm
  mcp: bbx.gcp
  method: compute_create
  inputs:
    name: my-instance
    zone: us-central1-a
    machine_type: e2-micro
```

**Methods:** `compute_create`, `compute_delete`, `storage_upload`, `storage_download`, `cloud_function_deploy`

#### bbx.azure
```yaml
- id: create_vm
  mcp: bbx.azure
  method: vm_create
  inputs:
    name: my-vm
    resource_group: my-rg
    size: Standard_B1s
```

**Methods:** `vm_create`, `vm_delete`, `storage_create_account`, `storage_upload_blob`

### Infrastructure as Code

#### bbx.docker
```yaml
- id: build_image
  mcp: bbx.docker
  method: build
  inputs:
    path: .
    tag: myapp:latest
    dockerfile: Dockerfile
```

**Methods:** `build`, `run`, `stop`, `push`, `pull`, `logs`, `exec`

#### bbx.kubernetes
```yaml
- id: deploy
  mcp: bbx.kubernetes
  method: apply
  inputs:
    file: deployment.yaml
    namespace: production
```

**Methods:** `apply`, `delete`, `get`, `scale`, `rollout`, `logs`, `exec`, `port_forward`, `helm_install`, `helm_upgrade`

#### bbx.terraform
```yaml
- id: provision
  mcp: bbx.terraform
  method: apply
  inputs:
    working_dir: ./terraform
    var_file: prod.tfvars
    auto_approve: true
```

**Methods:** `init`, `plan`, `apply`, `destroy`, `output`, `validate`, `fmt`, `show`

#### bbx.ansible
```yaml
- id: configure
  mcp: bbx.ansible
  method: playbook
  inputs:
    playbook: site.yml
    inventory: hosts.ini
```

**Methods:** `playbook`, `adhoc`, `galaxy_install`, `inventory_list`, `vault_encrypt`, `vault_decrypt`

### Utilities

#### bbx.http
```yaml
- id: api_call
  mcp: bbx.http
  method: post
  inputs:
    url: https://api.example.com/webhook
    headers:
      Authorization: Bearer ${env.API_TOKEN}
    json:
      event: deployment_complete
```

**Methods:** `get`, `post`, `put`, `delete`, `patch`, `download`

#### bbx.logger
```yaml
- id: log_event
  mcp: bbx.logger
  method: info
  inputs:
    message: "Processing complete: ${steps.process.output.count} items"
```

**Methods:** `debug`, `info`, `warning`, `error`, `critical`

#### bbx.transform
```yaml
- id: process_data
  mcp: bbx.transform
  method: map
  inputs:
    data: ${steps.fetch.output}
    function: "lambda x: x['value'] * 2"
```

**Methods:** `map`, `filter`, `reduce`, `merge`, `flatten`, `group_by`, `sort`

#### bbx.flow
```yaml
- id: run_subflow
  mcp: bbx.flow
  method: run
  inputs:
    path: deploy_service.bbx
    inputs:
      service_name: api
      version: "1.2.0"
```

**Methods:** `run` - Execute another workflow as subflow

### Specialized Adapters

- **bbx.process** - Process management with health checks
- **bbx.storage** - S3-compatible storage operations
- **bbx.queue** - Message queue (SQS, RabbitMQ, Kafka)
- **bbx.database** - Database operations (PostgreSQL, MySQL, MongoDB)
- **bbx.ai** - AI/ML integrations (OpenAI, Anthropic)
- **bbx.browser** - Browser automation (Playwright)
- **bbx.telegram** - Telegram bot integration
- **bbx.mobile** - Mobile app testing
- **bbx.mcp_bridge** - Universal MCP protocol bridge
- **bbx.sandbox** - Secure code execution sandbox
- **bbx.system** - System commands execution
- **bbx.wasm** - WebAssembly execution

---

## 🔄 Execution Flow

### Step Lifecycle

```
1. PARSE → Load and validate YAML
2. VALIDATE → Check workflow structure
3. BUILD DAG → Analyze dependencies
4. FOR EACH STEP:
   ├─ WAIT → Wait for depends_on steps
   ├─ RESOLVE → Resolve ${...} variables
   ├─ CHECK → Evaluate 'when' condition
   ├─ EXECUTE → Call adapter method
   ├─ STORE → Save output
   └─ RETRY → On failure, retry if configured
5. COMPLETE → Return results
```

### Dependency Resolution

BBX automatically determines execution order based on `depends_on`:

```yaml
workflow:
  steps:
    # These run in parallel (no dependencies)
    - id: fetch_users
      mcp: bbx.http
      method: get
      inputs: {url: "https://api.example.com/users"}

    - id: fetch_products
      mcp: bbx.http
      method: get
      inputs: {url: "https://api.example.com/products"}

    # Waits for both fetch_users and fetch_products
    - id: merge_data
      mcp: bbx.transform
      method: merge
      inputs:
        datasets:
          - ${steps.fetch_users.output}
          - ${steps.fetch_products.output}
      depends_on: [fetch_users, fetch_products]
```

**Execution Order:**
1. `fetch_users` and `fetch_products` run in parallel
2. `merge_data` waits for both to complete
3. If either fails, `merge_data` is skipped

### Context Structure

During execution, BBX maintains a context:

```python
{
  "steps": {
    "step_id_1": {
      "status": "success",
      "output": {...},
      "error": null
    },
    "step_id_2": {
      "status": "error",
      "output": null,
      "error": "Connection timeout"
    }
  },
  "env": {
    "API_KEY": "...",
    "DATABASE_URL": "..."
  },
  "inputs": {
    "user_id": "123",
    "environment": "production"
  }
}
```

---

## 🛡️ Error Handling

### Step Status Values

- `success` - Step completed successfully
- `error` - Step failed (exception occurred)
- `skipped` - Step skipped (condition false or dependency failed)

### Retry Mechanism

```yaml
- id: flaky_api_call
  mcp: bbx.http
  method: get
  inputs:
    url: https://api.example.com/flaky
  retry: 3              # Retry 3 times on failure
  retry_delay: 1000     # Start with 1 second delay
  retry_backoff: 2.0    # Double delay each time (1s, 2s, 4s)
  timeout: 5000         # 5 second timeout per attempt
```

**Retry Logic:**
1. Attempt 1: Execute → Fail → Wait 1000ms
2. Attempt 2: Execute → Fail → Wait 2000ms (1000 * 2.0)
3. Attempt 3: Execute → Fail → Wait 4000ms (2000 * 2.0)
4. Attempt 4: Execute → Fail → Mark as error

### Error Handling Pattern

```yaml
workflow:
  steps:
    - id: risky_operation
      mcp: bbx.http
      method: post
      inputs: {url: "https://api.example.com/risky"}
      retry: 3

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

    - id: cleanup
      mcp: bbx.system
      method: execute
      inputs:
        command: "cleanup.sh"
      depends_on: [risky_operation]
```

---

## 🔒 Security Best Practices

### 1. Never Hardcode Secrets

```yaml
# ❌ BAD
inputs:
  api_key: "sk-1234567890abcdef"

# ✅ GOOD
inputs:
  api_key: ${env.API_KEY}
```

### 2. Validate Inputs

```yaml
workflow:
  inputs:
    email:
      type: string
      required: true
      pattern: "^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"
```

### 3. Set Timeouts

```yaml
- id: external_api
  mcp: bbx.http
  method: get
  timeout: 10000  # Always set timeout for external calls
```

### 4. Use HTTPS

```yaml
inputs:
  url: https://api.example.com  # ✅ Always HTTPS
  # url: http://api.example.com  # ❌ Never HTTP for sensitive data
```

### 5. Limit Retries

```yaml
retry: 3  # ✅ Reasonable
# retry: 100  # ❌ Excessive, can cause resource exhaustion
```

---

## 📚 Complete Examples

### 1. Simple HTTP API Call

```yaml
workflow:
  id: simple_api_call
  name: Simple API Call Example
  version: "6.0"

  steps:
    - id: fetch_data
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/data
```

### 2. Multi-Step with Dependencies

```yaml
workflow:
  id: multi_step_pipeline
  name: Multi-Step Data Pipeline
  version: "6.0"

  steps:
    - id: fetch
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/data

    - id: transform
      mcp: bbx.transform
      method: map
      inputs:
        data: ${steps.fetch.output}
        function: "lambda x: {'id': x['id'], 'name': x['name'].upper()}"
      depends_on: [fetch]

    - id: store
      mcp: bbx.database
      method: insert
      inputs:
        table: processed_data
        data: ${steps.transform.output}
      depends_on: [transform]

    - id: notify
      mcp: bbx.telegram
      method: send_message
      inputs:
        chat_id: ${env.TELEGRAM_CHAT_ID}
        text: "Processed ${steps.transform.output.length} records"
      depends_on: [store]
```

### 3. Conditional Execution with Error Handling

```yaml
workflow:
  id: conditional_workflow
  name: Conditional Execution Example
  version: "6.0"

  inputs:
    environment:
      type: string
      required: true

  steps:
    - id: validate
      mcp: bbx.system
      method: execute
      inputs:
        command: "validate-env.sh ${inputs.environment}"

    - id: deploy_prod
      when: "${inputs.environment} == 'production' and ${steps.validate.status} == 'success'"
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: prod-deployment.yaml
      depends_on: [validate]

    - id: deploy_dev
      when: "${inputs.environment} != 'production' and ${steps.validate.status} == 'success'"
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: dev-deployment.yaml
      depends_on: [validate]

    - id: rollback
      when: "${steps.validate.status} == 'error'"
      mcp: bbx.logger
      method: error
      inputs:
        message: "Validation failed, deployment aborted"
      depends_on: [validate]
```

### 4. Parallel Execution

```yaml
workflow:
  id: parallel_processing
  name: Parallel Data Processing
  version: "6.0"

  steps:
    # These three run in parallel
    - id: fetch_users
      mcp: bbx.http
      method: get
      inputs: {url: "https://api.example.com/users"}

    - id: fetch_orders
      mcp: bbx.http
      method: get
      inputs: {url: "https://api.example.com/orders"}

    - id: fetch_products
      mcp: bbx.http
      method: get
      inputs: {url: "https://api.example.com/products"}

    # Waits for all three
    - id: merge
      mcp: bbx.transform
      method: merge
      inputs:
        datasets:
          - ${steps.fetch_users.output}
          - ${steps.fetch_orders.output}
          - ${steps.fetch_products.output}
      depends_on: [fetch_users, fetch_orders, fetch_products]
```

### 5. Workflow Composition (Subflows)

```yaml
# main.bbx
workflow:
  id: main_workflow
  name: Main Workflow
  version: "6.0"

  steps:
    - id: deploy_database
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service: database
          replicas: 3

    - id: deploy_api
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service: api
          replicas: 5
      depends_on: [deploy_database]

    - id: deploy_frontend
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service: frontend
          replicas: 2
      depends_on: [deploy_api]
```

---

## 🔄 Migration from v5.0

### Syntax Changes

| v5.0 | v6.0 | Note |
|------|------|------|
| `bbx_version: "5.0"` | *(removed)* | Version now in workflow metadata |
| `type: workflow` | *(removed)* | Inferred from structure |
| `mcp: "http"` | `mcp: bbx.http` | All adapters use `bbx.` prefix |
| `outputs: "data"` | *(removed)* | Output always in `output` field |
| `${step.id.field}` | `${steps.id.output.field}` | New variable syntax |
| `when: "${step.a.status}"` | `when: "${steps.a.status}"` | Use `steps` plural |

### Migration Example

**v5.0:**
```yaml
bbx_version: "5.0"
type: workflow

workflow:
  id: "example"
  steps:
    - id: "fetch"
      mcp: "http"
      method: "get"
      inputs:
        url: "https://api.example.com"
      outputs: "data"

    - id: "log"
      mcp: "logger"
      method: "info"
      inputs:
        message: "${step.fetch.data}"
```

**v6.0:**
```yaml
workflow:
  id: example
  name: Example Workflow
  version: "6.0"

  steps:
    - id: fetch
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com

    - id: log
      mcp: bbx.logger
      method: info
      inputs:
        message: ${steps.fetch.output}
      depends_on: [fetch]
```

---

## 📖 See Also

- [Agent Guide](AGENT_GUIDE.md) - For AI agents using BBX
- [Universal Adapter Guide](UNIVERSAL_ADAPTER.md) - Zero-code Docker-based adapters
- [Runtime Internals](RUNTIME_INTERNALS.md) - How the engine works
- [MCP Development](MCP_DEVELOPMENT.md) - Creating custom adapters
- [Getting Started](GETTING_STARTED.md) - Quick start guide
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues

---

**BBX v6.0 Specification**
**Release Date:** November 26, 2025
**Status:** Current
**Copyright:** 2025 Ilya Makarov
**License:** BSL 1.1 (converts to Apache 2.0 on 2028-11-05)
