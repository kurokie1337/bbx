# BBX Universal Adapter - Complete Guide

**The Universal Adapter architecture enables execution of any containerized CLI tool without writing Python code.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Philosophy](#-philosophy)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Package Management](#-package-management)
- [Creating YAML Definitions](#-creating-yaml-definitions)
- [Authentication Providers](#-authentication-providers)
- [Advanced Features](#-advanced-features)
- [Security Scanning](#-security-scanning)
- [Private Registries](#-private-registries)
- [Output Parsing](#-output-parsing)
- [Multi-Step Workflows](#-multi-step-workflows)
- [Performance Optimization](#-performance-optimization)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Examples](#-examples)
- [See Also](#-see-also)

---

## 🎯 Overview

The **Universal Adapter** (`bbx.universal`) is a revolutionary approach to workflow automation that replaces thousands of lines of Python boilerplate with simple, declarative YAML configurations.

### What is Universal Adapter?

Universal Adapter is a Docker-based execution engine that:
- Executes any CLI tool inside a container
- Supports authentication providers (AWS, GCP, Azure, Kubernetes)
- Parses structured output (JSON, YAML, XML)
- Manages volumes, environment variables, and resource limits
- Provides real-time streaming and timeout enforcement
- Eliminates the need to write adapter code

### Impact Metrics

Replacing traditional Python adapters with YAML definitions:

| Adapter | Old (Python) | New (YAML) | Reduction |
|---------|--------------|------------|-----------|
| Terraform | 371 lines | 15 lines | **96%** |
| Kubernetes | 591 lines | 12 lines | **98%** |
| AWS CLI | 474 lines | 11 lines | **98%** |
| **Total** | **1,436 lines** | **38 lines** | **97.4%** |

---

## 🔥 Philosophy

**"Zero-Code Adapters"** - The Universal Adapter embodies the principle that configuration should be declarative, not imperative.

### Core Principles

1. **Declarative Over Imperative** - Define what you want, not how to get it
2. **Reusability** - One definition, unlimited uses
3. **Type Safety** - Pydantic validation for all inputs
4. **Security First** - Sandboxed execution, no code injection
5. **Developer Experience** - Simple YAML beats complex Python

### Traditional Approach (Python)

```python
# terraform_adapter.py (371 lines)
class TerraformAdapter(BaseAdapter):
    def __init__(self):
        # 50 lines of initialization

    async def plan(self, working_dir, var_file=None, ...):
        # 80 lines of command building
        # 40 lines of volume mounting
        # 30 lines of auth handling
        # 50 lines of output parsing
        # 40 lines of error handling

    async def apply(self, ...):
        # Another 120+ lines
```

### Universal Adapter Approach (YAML)

```yaml
# terraform.yaml (15 lines)
id: terraform
uses: docker://hashicorp/terraform:latest
auth:
  type: aws_credentials
cmd:
  - terraform
  - {{ inputs.command }}
  - {% if inputs.var_file %}-var-file={{ inputs.var_file }}{% endif %}
working_dir: {{ inputs.working_dir }}
output_parser:
  type: text
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   BBX Workflow Engine                   │
├─────────────────────────────────────────────────────────┤
│                  Universal Adapter V2                   │
│  ┌───────────┬──────────────┬─────────────────────┐   │
│  │ Template  │ Auth         │ Docker Execution    │   │
│  │ Engine    │ Providers    │ Engine              │   │
│  │ (Jinja2)  │ (AWS/GCP/K8s)│ (Volume/Env/Limits) │   │
│  └───────────┴──────────────┴─────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                   Package Manager                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ blackbox/library/                                │  │
│  │  ├── terraform.yaml                              │  │
│  │  ├── kubectl.yaml                                │  │
│  │  ├── aws_cli.yaml                                │  │
│  │  └── ... (28+ definitions)                       │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                     Docker Engine                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Container Lifecycle: Pull → Run → Stream → Stop  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Execution Flow

1. **Load Definition** - From library or inline in workflow
2. **Render Template** - Jinja2 processes `{{ inputs.x }}` variables
3. **Auth Provider** - Inject credentials (env vars, volumes)
4. **Docker Execution** - Pull image, create container, execute command
5. **Stream Output** - Real-time stdout/stderr streaming
6. **Parse Output** - JSON/YAML/XML parsing with JMESPath queries
7. **Return Result** - Structured response with metadata

---

## 🚀 Quick Start

### Method 1: Dynamic Mode (Inline Definition)

Create a workflow with inline Universal Adapter configuration:

```yaml
workflow:
  id: terraform_example
  name: Terraform Plan
  version: "6.0"

  steps:
    - id: plan
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:latest
        auth:
          type: aws_credentials
        cmd:
          - terraform
          - plan
          - -out=tfplan
        working_dir: ./terraform
        timeout: 300
        output_parser:
          type: text
```

Run it:
```bash
python cli.py run terraform_example.bbx
```

### Method 2: Library Reference Mode

Use pre-built definitions from `blackbox/library/`:

```yaml
workflow:
  id: kubectl_example
  name: Kubernetes Deployment
  version: "6.0"

  steps:
    - id: deploy
      mcp: bbx.universal
      method: run
      definition: blackbox/library/k8s_apply.yaml
      inputs:
        file: deployment.yaml
        namespace: production
```

### Method 3: External Definition

Reference a custom definition file:

```yaml
workflow:
  id: custom_tool
  steps:
    - id: run_tool
      mcp: bbx.universal
      method: run
      definition: ./custom_definitions/mytool.yaml
      inputs:
        action: deploy
        config: config.yaml
```

---

## 📦 Package Management

The Package Manager provides installation, validation, and discovery of Universal Adapter definitions.

### CLI Commands

#### List Available Packages
```bash
python cli.py package list

# Output:
# Available packages in blackbox/library/:
#   - aws_cli (AWS CLI operations)
#   - gcloud (Google Cloud SDK)
#   - kubectl (Kubernetes client)
#   - terraform (Terraform IaC)
#   - ansible (Ansible automation)
#   - helm (Helm chart manager)
#   - ... (28+ total)
```

#### View Package Info
```bash
python cli.py package info terraform

# Output:
# Package: terraform
# ID: terraform
# Image: docker://hashicorp/terraform:latest
# Auth: aws_credentials
# Description: Terraform infrastructure as code tool
```

#### Install Package
```bash
# Install from file
python cli.py package install ./custom_definitions/mytool.yaml

# Validate installation
python cli.py package validate
```

#### Validate All Packages
```bash
python cli.py package validate

# Output:
# ✅ terraform.yaml - Valid
# ✅ kubectl.yaml - Valid
# ❌ broken.yaml - Missing required field: uses
```

### Programmatic Access

```python
from blackbox.core.package_manager import PackageManager

pm = PackageManager()

# List all packages
packages = pm.list_packages()
# Returns: ['terraform', 'aws_cli', 'kubectl', ...]

# Load package definition
definition = pm.load_package('terraform')
# Returns: {'id': 'terraform', 'uses': 'docker://...', ...}

# Install custom package
pm.install_package('custom-tool', {
    'id': 'custom-tool',
    'uses': 'docker://myorg/tool:latest',
    'cmd': ['tool', '{{ inputs.action }}']
})

# Reload packages
pm.reload_packages()
```

---

## 📖 Creating YAML Definitions

### Schema Reference

#### Required Fields

```yaml
id: string              # Unique identifier
uses: string            # Docker image (docker://image:tag)
cmd: list[string]       # Command template (Jinja2 syntax)
```

#### Optional Fields

```yaml
auth:                   # Authentication provider
  type: string          # Provider type (aws_credentials, kubeconfig, etc.)

env:                    # Environment variables (supports Jinja2)
  VAR_NAME: "value"

volumes:                # Volume mappings
  /host/path: /container/path

working_dir: string     # Working directory in container

timeout: integer        # Timeout in seconds (default: 300)

resources:              # Resource limits
  cpu: string          # CPU limit (e.g., "0.5", "2")
  memory: string       # Memory limit (e.g., "512m", "2g")

output_parser:          # Output parser config
  type: string         # Parser type: json, yaml, xml, text
  query: string        # JMESPath query (for JSON output)
```

### Basic Template

```yaml
id: my_tool
uses: docker://namespace/image:tag
cmd:
  - tool
  - {{ inputs.param }}
  - {% if inputs.optional %}--flag={{ inputs.optional }}{% endif %}
```

### Advanced Template with All Features

```yaml
id: advanced_tool
uses: docker://myorg/tool:latest

auth:
  type: aws_credentials  # Auto-inject AWS credentials

env:
  API_KEY: "{{ secrets.API_KEY }}"
  ENV: "{{ inputs.environment }}"
  DEBUG: "{% if inputs.debug %}true{% else %}false{% endif %}"

volumes:
  ./config: /app/config
  ./data: /app/data

working_dir: /app

timeout: 600  # 10 minutes

resources:
  cpu: "2"
  memory: "4g"

cmd:
  - tool
  - {{ inputs.command }}
  - --config=/app/config/{{ inputs.config_file }}
  - {% if inputs.verbose %}--verbose{% endif %}
  - {% for item in inputs.items %}
  - --item={{ item }}
  - {% endfor %}

output_parser:
  type: json
  query: "results[?status=='success'].{id: id, name: name}"
```

### Jinja2 Template Syntax

#### Variable Interpolation
```yaml
cmd:
  - echo
  - "{{ inputs.message }}"  # Simple variable
```

#### Conditionals
```yaml
cmd:
  - terraform
  - {{ inputs.action }}
  - {% if inputs.auto_approve %}--auto-approve{% endif %}
```

#### Loops
```yaml
cmd:
  - aws
  - s3
  - cp
  - {% for file in inputs.files %}
  - {{ file }}
  - {% endfor %}
  - s3://bucket/
```

#### Filters
```yaml
env:
  UPPERCASE: "{{ inputs.name | upper }}"
  LOWERCASE: "{{ inputs.name | lower }}"
  DEFAULT: "{{ inputs.optional | default('fallback') }}"
```

---

## 🔐 Authentication Providers

Universal Adapter supports multiple authentication providers that automatically inject credentials.

### Kubernetes (kubeconfig)

Automatically mounts `~/.kube/config`:

```yaml
id: kubectl
uses: docker://bitnami/kubectl:latest
auth:
  type: kubeconfig
cmd:
  - kubectl
  - {{ inputs.command }}
  - -n
  - {{ inputs.namespace }}
```

**Usage in Workflow:**
```yaml
- id: get_pods
  mcp: bbx.universal
  method: run
  definition: blackbox/library/kubectl.yaml
  inputs:
    command: get pods
    namespace: production
```

### AWS Credentials

Injects `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` and mounts `~/.aws`:

```yaml
id: aws_cli
uses: docker://amazon/aws-cli:latest
auth:
  type: aws_credentials
cmd:
  - aws
  - {{ inputs.service }}
  - {{ inputs.command }}
  - --output=json
output_parser:
  type: json
```

**Usage in Workflow:**
```yaml
- id: list_buckets
  mcp: bbx.universal
  method: run
  definition: blackbox/library/aws_cli.yaml
  inputs:
    service: s3
    command: ls
```

### GCP Credentials

Mounts service account JSON and sets `GOOGLE_APPLICATION_CREDENTIALS`:

```yaml
id: gcloud
uses: docker://google/cloud-sdk:latest
auth:
  type: gcp_credentials
cmd:
  - gcloud
  - {{ inputs.service }}
  - {{ inputs.command }}
  - --project={{ inputs.project }}
  - --format=json
output_parser:
  type: json
```

**Usage in Workflow:**
```yaml
- id: list_instances
  mcp: bbx.universal
  method: run
  definition: blackbox/library/gcloud.yaml
  inputs:
    service: compute instances
    command: list
    project: my-project-id
```

### Azure Credentials

Injects Azure credentials and mounts `~/.azure`:

```yaml
id: az_cli
uses: docker://mcr.microsoft.com/azure-cli:latest
auth:
  type: azure_credentials
cmd:
  - az
  - {{ inputs.service }}
  - {{ inputs.command }}
  - --output=json
output_parser:
  type: json
```

### GitHub Token

Injects `GITHUB_TOKEN` environment variable:

```yaml
id: gh_cli
uses: docker://ghcr.io/cli/cli:latest
auth:
  type: github_token
  token: "{{ env.GITHUB_TOKEN }}"
cmd:
  - gh
  - {{ inputs.command }}
```

### Custom Auth Provider

Create your own authentication logic:

```yaml
id: custom_auth
uses: docker://custom/tool:latest
auth:
  type: custom
  env:
    API_KEY: "{{ secrets.API_KEY }}"
    API_SECRET: "{{ secrets.API_SECRET }}"
  volumes:
    ~/.custom/config: /root/.config
```

---

## 🚀 Advanced Features

### Resource Limits

Control CPU and memory allocation:

```yaml
id: resource_limited
uses: docker://alpine:latest
resources:
  cpu: "0.5"      # 50% of one CPU core
  memory: "256m"  # 256 megabytes
cmd:
  - heavy-computation
```

### Timeout Configuration

Set execution timeouts to prevent hanging:

```yaml
id: long_running
uses: docker://tool:latest
timeout: 1800  # 30 minutes
cmd:
  - long-process
```

### Volume Mounting

Mount host directories into container:

```yaml
id: with_volumes
uses: docker://node:18
volumes:
  ./src: /app/src        # Source code
  ./config: /app/config  # Configuration
  ./output: /app/output  # Results
working_dir: /app
cmd:
  - npm
  - run
  - build
```

### Environment Variables

Pass configuration via environment:

```yaml
id: with_env
uses: docker://alpine:latest
env:
  API_URL: "{{ inputs.api_url }}"
  DEBUG: "{% if inputs.debug %}1{% else %}0{% endif %}"
  DATABASE_URL: "{{ secrets.DATABASE_URL }}"
cmd:
  - script.sh
```

---

## 🔍 Security Scanning

BBX includes integrated container image security scanning.

### CLI Commands

```bash
# Scan an image
python cli.py security-scan alpine:latest

# Scan with severity filter
python cli.py security-scan --severity CRITICAL python:3.11

# Scan multiple images
python cli.py security-scan hashicorp/terraform:latest
python cli.py security-scan bitnami/kubectl:latest
```

### Example Output

```
🔍 Scanning image: python:3.11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Scan complete

📊 Summary:
   CRITICAL: 0
   HIGH: 2
   MEDIUM: 5
   LOW: 12

⚠️  HIGH severity vulnerabilities found:
   - CVE-2023-12345: OpenSSL buffer overflow
   - CVE-2023-67890: libxml2 parsing issue

💡 Recommendation: Update to python:3.11.5 or later
```

### Programmatic Scanning

```python
from blackbox.core.security import SecurityScanner

scanner = SecurityScanner()
results = scanner.scan_image('python:3.11')

if results['critical'] > 0 or results['high'] > 0:
    print("❌ Image has critical vulnerabilities")
else:
    print("✅ Image is safe to use")
```

---

## 🔒 Private Registries

### GitHub Container Registry (ghcr.io)

```bash
# Login
export GITHUB_TOKEN=ghp_xxxxxxxxxxxx
python cli.py registry-login ghcr.io --token $GITHUB_TOKEN

# Use in workflow
- id: use_private_image
  mcp: bbx.universal
  method: run
  inputs:
    uses: docker://ghcr.io/myorg/private-tool:latest
    cmd: ["tool", "action"]
```

### AWS ECR

```bash
# Login (uses AWS CLI credentials)
python -c "from blackbox.core.registry_auth import PrivateRegistryAuth; PrivateRegistryAuth().login_ecr('us-east-1')"

# Use in workflow
- id: use_ecr_image
  mcp: bbx.universal
  method: run
  inputs:
    uses: docker://123456789.dkr.ecr.us-east-1.amazonaws.com/my-tool:latest
    auth:
      type: aws_credentials
    cmd: ["tool"]
```

### Google Container Registry (gcr.io)

```bash
# Login (uses gcloud credentials)
gcloud auth login
python -c "from blackbox.core.registry_auth import PrivateRegistryAuth; PrivateRegistryAuth().login_gcr('my-project-id')"

# Use in workflow
- id: use_gcr_image
  mcp: bbx.universal
  method: run
  inputs:
    uses: docker://gcr.io/my-project/tool:latest
    auth:
      type: gcp_credentials
    cmd: ["tool"]
```

---

## 📊 Output Parsing

### JSON Output

```yaml
id: json_parser
uses: docker://amazon/aws-cli:latest
cmd:
  - aws
  - s3
  - ls
  - --output=json
output_parser:
  type: json
  query: "Buckets[?contains(Name, 'prod')].Name"
```

**Response:**
```python
{
  "success": True,
  "data": ["prod-bucket-1", "prod-bucket-2"],
  "metadata": {...}
}
```

### YAML Output

```yaml
id: yaml_parser
uses: docker://bitnami/kubectl:latest
cmd:
  - kubectl
  - get
  - pods
  - -o
  - yaml
output_parser:
  type: yaml
```

### XML Output

```yaml
id: xml_parser
uses: docker://custom/tool:latest
cmd:
  - tool
  - --output=xml
output_parser:
  type: xml
```

### Text Output (No Parsing)

```yaml
id: text_output
uses: docker://alpine:latest
cmd:
  - echo
  - "Hello World"
output_parser:
  type: text
```

**Response:**
```python
{
  "success": True,
  "data": "Hello World\n",
  "metadata": {
    "exit_code": 0,
    "stdout": "Hello World\n",
    "stderr": ""
  }
}
```

---

## 🔄 Multi-Step Workflows

Execute multiple commands in sequence within a single container:

```yaml
workflow:
  id: multi_step_terraform
  name: Terraform Multi-Step Deployment
  version: "6.0"

  steps:
    - id: terraform_deploy
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:latest
        auth:
          type: aws_credentials
        working_dir: /workspace
        volumes:
          ./terraform: /workspace
        steps:
          - name: "Initialize Terraform"
            cmd: ["terraform", "init"]
            continue_on_error: false

          - name: "Validate Configuration"
            cmd: ["terraform", "validate"]
            continue_on_error: false

          - name: "Plan Infrastructure"
            cmd: ["terraform", "plan", "-out=tfplan"]
            continue_on_error: false

          - name: "Apply Infrastructure"
            cmd: ["terraform", "apply", "tfplan"]
            continue_on_error: false

        resources:
          cpu: "2"
          memory: "2g"
```

### Multi-Step CI/CD Pipeline

```yaml
workflow:
  id: ci_cd_pipeline
  name: Complete CI/CD Pipeline
  version: "6.0"

  steps:
    - id: build_and_test
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://node:18
        working_dir: /app
        volumes:
          ./src: /app
        steps:
          - name: "Install Dependencies"
            cmd: ["npm", "ci"]

          - name: "Lint Code"
            cmd: ["npm", "run", "lint"]
            continue_on_error: false

          - name: "Run Tests"
            cmd: ["npm", "test"]
            continue_on_error: false

          - name: "Build Application"
            cmd: ["npm", "run", "build"]

        resources:
          cpu: "4"
          memory: "4g"
```

---

## ⚡ Performance Optimization

### 1. Pre-pull Docker Images

Avoid download delays during execution:

```bash
# Pre-pull common images
docker pull hashicorp/terraform:latest
docker pull bitnami/kubectl:latest
docker pull amazon/aws-cli:latest
docker pull google/cloud-sdk:latest
```

### 2. Use Specific Tags

Instead of `:latest`, use specific versions:

```yaml
# ❌ Slower (pulls latest every time)
uses: docker://python:latest

# ✅ Faster (cached after first pull)
uses: docker://python:3.11-alpine
```

### 3. Enable Package Caching

Package Manager automatically caches installed packages:

```python
from blackbox.core.package_manager import PackageManager

pm = PackageManager()
# First load: Reads from disk
definition = pm.load_package('terraform')

# Subsequent loads: Returns from cache
definition = pm.load_package('terraform')  # Instant
```

### 4. Use Alpine-based Images

Alpine images are significantly smaller:

```yaml
# ❌ Larger image (800MB)
uses: docker://python:3.11

# ✅ Smaller image (50MB)
uses: docker://python:3.11-alpine
```

### 5. Optimize Resource Limits

Set appropriate CPU/memory limits:

```yaml
resources:
  cpu: "1"      # 1 CPU core (not 4)
  memory: "512m"  # 512MB (not 4GB)
```

---

## 📚 API Reference

### UniversalAdapterV2

**Constructor:**
```python
from blackbox.core.universal_v2 import UniversalAdapterV2

adapter = UniversalAdapterV2(definition)
```

**Parameters:**
- `definition` (dict): Adapter configuration

**Methods:**

#### `execute(method, inputs)`

Execute the adapter with given inputs.

```python
result = await adapter.execute(method='run', inputs={'key': 'value'})
```

**Returns:**
```python
{
  "success": bool,
  "data": str,           # Command output (or parsed data)
  "error": str,          # Error message (if failed)
  "error_type": str,     # Error type
  "metadata": {
    "exit_code": int,
    "stdout": str,
    "stderr": str
  }
}
```

### Package Manager

```python
from blackbox.core.package_manager import PackageManager

pm = PackageManager()
```

**Methods:**

#### `list_packages() -> List[str]`
Returns list of available package names.

#### `load_package(name: str) -> Dict`
Loads and returns package definition.

#### `install_package(name: str, definition: Dict)`
Installs a new package.

#### `validate_all() -> List[Dict]`
Validates all installed packages.

#### `reload_packages()`
Clears cache and reloads from disk.

---

## ❓ Troubleshooting

### "No Docker image specified"
**Problem:** Missing `uses` field in definition.
**Solution:** Ensure `uses: docker://image:tag` is present.

```yaml
# ❌ Missing uses
id: my_tool
cmd: ["tool"]

# ✅ Correct
id: my_tool
uses: docker://tool:latest
cmd: ["tool"]
```

### "Missing input for template"
**Problem:** Jinja2 template references undefined variable.
**Solution:** Provide all required inputs.

```yaml
# Definition expects inputs.file
cmd: ["kubectl", "apply", "-f", "{{ inputs.file }}"]

# Workflow must provide it
inputs:
  file: deployment.yaml
```

### "Failed to parse JSON output"
**Problem:** CLI outputs text, not JSON.
**Solution:** Add `--output=json` flag or change parser type.

```yaml
# ❌ Outputs text
cmd: ["kubectl", "get", "pods"]
output_parser:
  type: json

# ✅ Outputs JSON
cmd: ["kubectl", "get", "pods", "-o", "json"]
output_parser:
  type: json
```

### "Subprocess hanging / timeout"
**Problem:** Large image download during execution.
**Solution:** Pre-pull images before running workflow.

```bash
# Pre-pull before workflow execution
docker pull hashicorp/terraform:1.6
```

### "Permission denied"
**Problem:** Docker daemon not accessible.
**Solution:** Check Docker daemon is running and user has permissions.

```bash
# Check Docker status
docker ps

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
```

### "Authentication failed"
**Problem:** Missing or invalid credentials.
**Solution:** Set environment variables or mount credential files.

```bash
# AWS
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=yyy

# GCP
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Kubernetes
# Ensure ~/.kube/config exists and is valid
```

---

## 💡 Examples

### Example 1: Terraform Deployment

```yaml
workflow:
  id: terraform_deploy
  name: Terraform AWS Deployment
  version: "6.0"

  steps:
    - id: terraform_apply
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://hashicorp/terraform:1.6
        auth:
          type: aws_credentials
        working_dir: /workspace
        volumes:
          ./terraform: /workspace
        steps:
          - name: "Initialize"
            cmd: ["terraform", "init"]
          - name: "Plan"
            cmd: ["terraform", "plan", "-out=tfplan"]
          - name: "Apply"
            cmd: ["terraform", "apply", "tfplan"]
        timeout: 1800
```

### Example 2: Kubernetes Deployment

```yaml
workflow:
  id: k8s_deploy
  name: Kubernetes Deployment
  version: "6.0"

  steps:
    - id: deploy
      mcp: bbx.universal
      method: run
      definition: blackbox/library/k8s_apply.yaml
      inputs:
        file: deployment.yaml
        namespace: production

    - id: verify
      mcp: bbx.universal
      method: run
      definition: blackbox/library/kubectl.yaml
      inputs:
        command: get pods
        namespace: production
      depends_on: [deploy]
```

### Example 3: Multi-Cloud CLI

```yaml
workflow:
  id: multi_cloud
  name: Multi-Cloud Operations
  version: "6.0"

  steps:
    # AWS
    - id: aws_list_buckets
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://amazon/aws-cli:latest
        auth:
          type: aws_credentials
        cmd: ["aws", "s3", "ls", "--output=json"]
        output_parser:
          type: json

    # GCP (parallel with AWS)
    - id: gcp_list_instances
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://google/cloud-sdk:latest
        auth:
          type: gcp_credentials
        cmd: ["gcloud", "compute", "instances", "list", "--format=json"]
        output_parser:
          type: json

    # Azure (parallel with AWS and GCP)
    - id: azure_list_vms
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://mcr.microsoft.com/azure-cli:latest
        auth:
          type: azure_credentials
        cmd: ["az", "vm", "list", "--output=json"]
        output_parser:
          type: json
```

### Example 4: Database Migration

```yaml
workflow:
  id: db_migration
  name: PostgreSQL Migration
  version: "6.0"

  steps:
    - id: run_migrations
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://postgres:15
        env:
          PGHOST: "{{ inputs.db_host }}"
          PGPORT: "5432"
          PGDATABASE: "{{ inputs.db_name }}"
          PGUSER: "{{ secrets.DB_USER }}"
          PGPASSWORD: "{{ secrets.DB_PASSWORD }}"
        volumes:
          ./migrations: /migrations
        cmd:
          - psql
          - -f
          - /migrations/schema.sql
```

### Example 5: Custom Tool with All Features

```yaml
workflow:
  id: advanced_example
  name: Advanced Universal Adapter Demo
  version: "6.0"

  steps:
    - id: complex_operation
      mcp: bbx.universal
      method: run
      inputs:
        uses: docker://custom/tool:latest

        auth:
          type: aws_credentials

        env:
          API_KEY: "{{ secrets.API_KEY }}"
          ENVIRONMENT: "{{ inputs.environment }}"
          DEBUG: "{% if inputs.debug %}true{% else %}false{% endif %}"

        volumes:
          ./config: /app/config
          ./data: /app/data
          ./output: /app/output

        working_dir: /app

        timeout: 600

        resources:
          cpu: "2"
          memory: "4g"

        cmd:
          - tool
          - process
          - --config=/app/config/{{ inputs.config_file }}
          - --input=/app/data/{{ inputs.input_file }}
          - --output=/app/output/result.json
          - {% if inputs.verbose %}--verbose{% endif %}

        output_parser:
          type: json
          query: "results[?status=='success']"
```

---

## 📖 See Also

- **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Complete workflow format reference
- **[Getting Started](GETTING_STARTED.md)** - Quick start guide for beginners
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[MCP Adapter Development](MCP_DEVELOPMENT.md)** - Create custom adapters
- **[Library Reference](../blackbox/library/README.md)** - Available package definitions
- **[Architecture Guide](ARCHITECTURE.md)** - System design and internals
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0

**BBX Universal Adapter - Zero Python. Infinite Possibilities.** 🚀
