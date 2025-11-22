# BBX API Reference

## Core Components

### UniversalAdapterV2

**Universal Docker-based adapter for executing any containerized tool.**

#### Constructor

```python
from blackbox.core.universal_v2 import UniversalAdapterV2

adapter = UniversalAdapterV2(definition)
```

**Parameters:**
- `definition` (dict): Adapter configuration

**Definition Schema:**
```yaml
id: string              # Unique identifier
uses: string            # Docker image (docker://image:tag)
cmd: list[string]       # Command to execute
env: dict              # Environment variables (supports Jinja2)
volumes: dict          # Volume mappings {host: container}
working_dir: string    # Working directory in container
timeout: int           # Timeout in seconds
resources:             # Resource limits
  cpu: string         # CPU limit (e.g., "0.5")
  memory: string      # Memory limit (e.g., "512m")
output_parser:         # Output parser config
  type: string        # Parser type (json, yaml, etc.)
```

#### Methods

##### `execute(method, inputs)`

Execute the adapter with given inputs.

```python
result = await adapter.execute(method='run', inputs={'key': 'value'})
```

**Parameters:**
- `method` (str): Execution method (usually 'run')
- `inputs` (dict): Input variables for template rendering

**Returns:**
```python
{
  "success": bool,
  "data": str,           # Command output
  "error": str,          # Error message (if failed)
  "error_type": str,     # Error type
  "metadata": {
    "exit_code": int,
    "stdout": str,
    "stderr": str
  }
}
```

---

### Runtime

**Workflow execution engine.**

#### `run_file()`

Execute a BBX workflow file.

```python
from blackbox.core.runtime import run_file

results = await run_file(
    file_path='workflow.bbx',
    event_bus=None,
    use_cache=True,
    inputs={'key': 'value'}
)
```

**Parameters:**
- `file_path` (str): Path to .bbx workflow file
- `event_bus` (EventBus, optional): Event bus for tracking
- `use_cache` (bool): Enable workflow parsing cache
- `inputs` (dict): Workflow input variables

**Returns:**
```python
{
  "step_id": {
    "status": "success",
    "output": {...}
  },
  ...
}
```

---

### Package Manager

**Manage reusable workflow components.**

```python
from blackbox.core.packages import PackageManager

pm = PackageManager()
```

#### Methods

##### `list_packages()`
```python
packages = pm.list_packages()
# Returns: ['terraform', 'aws', 'kubectl', ...]
```

##### `load_package(name)`
```python
definition = pm.load_package('terraform')
# Returns: Package definition dict
```

##### `install_package(name, definition)`
```python
pm.install_package('custom-tool', {...})
```

---

### Health Checks

**System health monitoring.**

```python
from blackbox.core.health import HealthChecker

# Liveness check
health = HealthChecker.liveness()
# Returns: {"status": "alive", "message": "BBX system is running"}

# Readiness check
readiness = HealthChecker.readiness()
# Returns: {"ready": bool, "checks": {...}}

# Metrics
metrics = HealthChecker.metrics()
# Returns: {"bbx_workflows_total": 0, ...}
```

---

## CLI Commands

### Workflow Execution

```bash
# Run a workflow
python -m blackbox.cli.main run workflow.bbx

# Run with inputs
python -m blackbox.cli.main run workflow.bbx --input key=value

# Enable debug output
python -m blackbox.cli.main run workflow.bbx --verbose
```

### Package Management

```bash
# List all packages
python -m blackbox.cli.main package list

# Validate a package
python -m blackbox.cli.main package validate terraform

# Install a package
python -m blackbox.cli.main package install custom-tool.yaml
```

### System Commands

```bash
# System health check
python -m blackbox.cli.main system check

# Version info
python -m blackbox.cli.main version

# Help
python -m blackbox.cli.main --help
```

---

## Workflow Syntax

### Basic Workflow

```yaml
name: My Workflow
version: 1.0

steps:
  - id: step1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Hello World"]
```

### With Dependencies

```yaml
steps:
  - id: build
    mcp: universal
    method: run
    inputs:
      uses: docker://node:18
      cmd: ["npm", "run", "build"]
  
  - id: deploy
    depends_on: [build]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["deploy.sh"]
```

### With Environment Variables

```yaml
steps:
  - id: api_call
    mcp: universal
    method: run
    inputs:
      uses: docker://curlimages/curl:latest
      cmd: ["curl", "-H", "Authorization: Bearer $TOKEN", "https://api.com"]
      env:
        TOKEN: "{{ inputs.auth_token }}"
```

### With Conditionals

```yaml
steps:
  - id: check
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["test", "-f", "/config/prod.yml"]
  
  - id: production
    condition: "{{ steps.check.status == 'success' }}"
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["deploy", "--env", "prod"]
```

### With Timeout

```yaml
steps:
  - id: long_task
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["long-running-script.sh"]
      timeout: 300  # 5 minutes
```

---

## Auth Providers

### GitHub Auth

```yaml
steps:
  - id: clone_repo
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine/git:latest
      cmd: ["git", "clone", "https://github.com/user/repo.git"]
      auth:
        type: github
        token: "{{ inputs.github_token }}"
```

### AWS Auth

```yaml
steps:
  - id: s3_upload
    mcp: universal
    method: run
    inputs:
      uses: docker://amazon/aws-cli:latest
      cmd: ["aws", "s3", "cp", "file.txt", "s3://bucket/"]
      auth:
        type: aws
        access_key_id: "{{ inputs.aws_key }}"
        secret_access_key: "{{ inputs.aws_secret }}"
```

---

## Error Handling

### Success Response
```python
{
  "success": True,
  "data": "command output",
  "metadata": {
    "exit_code": 0,
    "stdout": "...",
    "stderr": ""
  }
}
```

### Error Response
```python
{
  "success": False,
  "data": None,
  "error": "Command failed with exit code 1",
  "error_type": "execution_error",
  "metadata": {
    "exit_code": 1,
    "stdout": "",
    "stderr": "error message"
  }
}
```

### Error Types
- `execution_error` - Command execution failed
- `timeout_error` - Command timed out
- `validation_error` - Invalid configuration
- `docker_error` - Docker-related error

---

## Template Variables

### Available in Jinja2 Templates

```yaml
# Access inputs
cmd: ["echo", "{{ inputs.message }}"]

# Access environment
env:
  PATH: "{{ env.PATH }}"

# Access step outputs (in depends_on steps)
cmd: ["echo", "{{ steps.step1.output.data }}"]

# Conditionals
condition: "{{ inputs.deploy == true }}"
```

---

## Examples

### Multi-Cloud Deployment

```yaml
name: Multi-Cloud Deploy
version: 1.0

steps:
  - id: aws_deploy
    mcp: universal
    method: run
    inputs:
      uses: docker://amazon/aws-cli:latest
      cmd: ["aws", "ec2", "describe-instances"]
  
  - id: gcp_deploy
    mcp: universal
    method: run
    inputs:
      uses: docker://google/cloud-sdk:latest
      cmd: ["gcloud", "compute", "instances", "list"]
  
  - id: notify
    depends_on: [aws_deploy, gcp_deploy]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Deployment complete"]
```

### CI/CD Pipeline

```yaml
name: CI/CD Pipeline
version: 1.0

steps:
  - id: test
    mcp: universal
    method: run
    inputs:
      uses: docker://node:18
      cmd: ["npm", "test"]
  
  - id: build
    depends_on: [test]
    mcp: universal
    method: run
    inputs:
      uses: docker://node:18
      cmd: ["npm", "run", "build"]
  
  - id: deploy
    depends_on: [build]
    condition: "{{ inputs.environment == 'production' }}"
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["deploy.sh", "production"]
```

---

*For more examples, see the `examples/` directory.*

---

## 📚 See Also

- **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Complete workflow format reference
- **[Universal Adapter Guide](UNIVERSAL_ADAPTER.md)** - Execute any CLI tool without code
- **[Getting Started](GETTING_STARTED.md)** - Beginner's guide to BBX
- **[MCP Development](MCP_DEVELOPMENT.md)** - Create custom adapters
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Documentation Index](INDEX.md)** - Complete documentation navigation

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0
