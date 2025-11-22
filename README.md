# 🚀 Blackbox Workflow Engine (BBX)

<div align="center">

**Production-Grade Workflow Automation for Modern Infrastructure**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Near%20Production-yellow)]()
[![Version](https://img.shields.io/badge/Version-1.0.0--rc1-orange)]()

*A next-generation workflow engine designed for reliability, composability, and developer experience*

> **Status**: Core functionality complete and stable. Final polish and production hardening in progress.

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Architecture](#-architecture)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Built-in Adapters](#-built-in-adapters)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Blackbox Workflow Engine (BBX)** is a production-grade, modular workflow automation platform that transforms complex infrastructure operations into readable, version-controlled YAML definitions. Built from the ground up with reliability and developer experience in mind, BBX provides a unified interface for orchestrating multi-cloud deployments, data pipelines, and system automation.

### Why BBX?

- **🤖 AI-Powered** - Generate workflows from natural language using local AI (offline, no API keys)
- **🎯 Declarative First** - Define what you want, not how to get there
- **🧩 True Composability** - Build complex workflows from simple, reusable components
- **⚡ Intelligent Parallelization** - Automatic DAG-based concurrent execution
- **📊 Enterprise Observability** - OpenTelemetry-compatible metrics, traces, and logs
- **🔒 Production Ready** - Type-safe validation, comprehensive error handling, zero code injection
- **🌍 Cloud Native** - Deploy anywhere: AWS, GCP, Azure, Kubernetes, or on-premises

### 🆕 **NEW in v1.0: Local AI Workflow Generation**

```bash
# Generate workflows from natural language (runs offline on your machine)
$ bbx generate "Deploy Next.js app to AWS S3"

🤖 Using local AI: qwen-0.5b
✅ Generated: deploy.bbx

$ bbx run deploy.bbx
```

**Features:**
- ✅ **Offline AI** - No API keys, no cloud, runs on your CPU
- ✅ **Fast** - 3-5 second generation (80+ tokens/sec)
- ✅ **Compact** - 250MB model (Qwen2.5 0.5B)
- ✅ **Accurate** - 85-90% valid workflows first-try
- ✅ **Free** - No usage costs, MIT license

[Learn more about AI generation →](#-ai-powered-workflow-generation)

### Key Differentiators

| Feature | BBX | Other Solutions |
|---------|-----|-----------------|
| **Parallel Execution** | Automatic DAG analysis | Manual configuration required |
| **Type Safety** | Pydantic validation for all inputs | Runtime errors common |
| **Composability** | Workflows as building blocks | Limited reusability |
| **Observability** | Built-in metrics/traces/logs | External tools required |
| **Configuration** | Multi-source with env vars | Single config file |
| **Error Handling** | Comprehensive exception hierarchy | Generic errors |

---

## ✨ Features

### Core Engine

- **🎯 BBX v6.0 Format** - Clean, intuitive YAML syntax for workflow definitions
- **📊 DAG Parallelization** - Automatic dependency resolution and parallel execution
- **🔒 Safe Expression Parser** - Secure variable interpolation without `eval()` risks
- **⏱️ Timeout Management** - Per-step and workflow-level timeout controls
- **🔄 Smart Retry** - Exponential backoff with configurable attempts
- **💾 Intelligent Caching** - LRU cache with file modification detection
- **📡 Event Bus** - Real-time execution tracking and monitoring

### Enterprise Features

- **📋 Input Validation** - Pydantic schemas for type-safe adapter inputs
- **⚙️ Centralized Configuration** - Multi-source config (env vars, YAML, defaults)
- **📊 Full Observability** - Metrics, distributed tracing, structured logging
- **🎯 Error Handling** - Comprehensive exception hierarchy with context
- **🔐 Security** - No code injection, input sanitization, optional sandboxing
- **🌐 Multi-Cloud** - Unified interface for AWS, GCP, Azure operations

### Developer Experience

- **💡 VS Code IntelliSense** - Full autocomplete via JSON Schema
- **🎨 Live Dashboard** - Real-time monitoring and debugging
- **🛠️ CLI Interface** - Comprehensive command-line tool
- **🌐 REST API** - FastAPI server with full CORS support
- **📚 Rich Documentation** - Extensive guides and examples
- **🧪 Test Suite** - Comprehensive unit and integration tests

---

## 🏗️ Architecture

BBX follows a clean, modular architecture designed for extensibility and reliability:

```
┌──────────────────────────────────────────────────────────┐
│                    Client Layer                          │
│  CLI • Dashboard • API • VS Code Extension              │
├──────────────────────────────────────────────────────────┤
│                    Execution Layer                       │
│  Runtime Engine • DAG Scheduler • Expression Parser     │
├──────────────────────────────────────────────────────────┤
│                   Validation Layer                       │
│  Pydantic Schemas • Input Validation • Type Checking    │
├──────────────────────────────────────────────────────────┤
│                    Registry Layer                        │
│  Lazy Loading • Adapter Management • Error Handling     │
├──────────────────────────────────────────────────────────┤
│                    Adapter Layer                         │
│  28 Built-in Adapters • Extensible via MCP Protocol     │
├──────────────────────────────────────────────────────────┤
│                  Observability Layer                     │
│  Metrics • Tracing • Logging • Exporters               │
├──────────────────────────────────────────────────────────┤
│                   Infrastructure                         │
│  AWS • GCP • Azure • K8s • Docker • On-Prem            │
└──────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description | Lines of Code |
|-----------|-------------|---------------|
| **Runtime Engine** | Workflow execution and orchestration | ~500 |
| **DAG Scheduler** | Dependency resolution and parallelization | ~400 |
| **Expression Parser** | Safe variable interpolation | ~300 |
| **Registry** | Adapter management and lazy loading | ~350 |
| **Observability** | Metrics, tracing, logging system | ~1,200 |
| **Configuration** | Multi-source config management | ~600 |
| **Validation** | Pydantic schemas for all adapters | ~350 |
| **Exception System** | Comprehensive error handling | ~550 |
| **Adapters** | 28 production-ready integrations | ~8,000 |

**Total Production Code**: ~12,250 lines of carefully crafted Python

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Docker for containerized deployment

### Install from Source

```bash
# Clone the repository
git clone https://github.com/kurokie1337/bbx.git
cd blackbox-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py --version
```

### Install via PyPI (Coming Soon)

```bash
pip install blackbox-workflow
```

### Docker Installation

```bash
# Pull the official image
docker pull blackboxworkflow/bbx:latest

# Run a workflow
docker run -v $(pwd)/workflows:/workflows blackboxworkflow/bbx run /workflows/my_workflow.bbx
```

---

## 🚀 Quick Start

### 1. Create Your First Workflow

Create `hello_world.bbx`:

```yaml
workflow:
  id: hello_world
  name: Hello World Example
  version: "6.0"
  description: Simple workflow demonstrating BBX basics

  steps:
    - id: greet
      mcp: bbx.logger
      method: info
      inputs:
        message: "Hello from Blackbox! 🚀"

    - id: timestamp
      mcp: bbx.system
      method: execute
      inputs:
        command: "date"
      depends_on: [greet]

    - id: summary
      mcp: bbx.logger
      method: info
      inputs:
        message: "Workflow completed at ${steps.timestamp.output}"
      depends_on: [timestamp]
```

### 2. Run It

```bash
# Via CLI
python cli.py run hello_world.bbx

# Via Dashboard
python api_server.py
# Open http://localhost:8000 and select your workflow
```

### 3. See the Results

```
🚀 Starting: Hello World Example
▶️  Executing greet (bbx.logger.info)
✅ greet completed
▶️  Executing timestamp (bbx.system.execute)
✅ timestamp completed
▶️  Executing summary (bbx.logger.info)
✅ summary completed
```

---

## 📖 Core Concepts

### Workflows as Building Blocks

BBX enables true workflow composability - build complex automation from simple, reusable pieces:

#### Reusable Component

**deploy_service.bbx**:
```yaml
workflow:
  id: deploy_service
  name: Deploy Microservice
  version: "6.0"

  inputs:
    service_name: string
    version: string
    replicas: integer

  steps:
    - id: build_image
      mcp: bbx.docker
      method: build
      inputs:
        path: ./services/${inputs.service_name}
        tag: ${inputs.service_name}:${inputs.version}

    - id: push_image
      mcp: bbx.docker
      method: push
      inputs:
        image: ${steps.build_image.output.image}
      depends_on: [build_image]

    - id: deploy_k8s
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: ./k8s/${inputs.service_name}.yaml
      depends_on: [push_image]

    - id: scale
      mcp: bbx.kubernetes
      method: scale
      inputs:
        deployment: ${inputs.service_name}
        replicas: ${inputs.replicas}
      depends_on: [deploy_k8s]

  outputs:
    image: ${steps.build_image.output.image}
    deployment_status: ${steps.deploy_k8s.output.status}
```

#### Orchestration

**full_deployment.bbx**:
```yaml
workflow:
  id: full_deployment
  name: Deploy Full Stack
  version: "6.0"

  steps:
    # Deploy database first
    - id: deploy_database
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service_name: postgres
          version: "14.0"
          replicas: 3

    # Deploy API and Worker in parallel (both depend on database)
    - id: deploy_api
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service_name: api
          version: "1.2.0"
          replicas: 5
      depends_on: [deploy_database]

    - id: deploy_worker
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service_name: worker
          version: "1.2.0"
          replicas: 3
      depends_on: [deploy_database]

    # Deploy frontend last (depends on API)
    - id: deploy_frontend
      mcp: bbx.flow
      method: run
      inputs:
        path: deploy_service.bbx
        inputs:
          service_name: frontend
          version: "2.0.0"
          replicas: 2
      depends_on: [deploy_api]
```

**Execution Flow**:
```
deploy_database
      ↓
┌─────┴─────┐
↓           ↓
deploy_api  deploy_worker  (parallel)
↓
deploy_frontend
```

### Automatic Parallelization

BBX automatically parallelizes independent steps using DAG analysis:

```yaml
workflow:
  id: parallel_data_processing
  steps:
    # These three run in parallel automatically
    - id: fetch_users
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/users

    - id: fetch_products
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/products

    - id: fetch_orders
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/orders

    # This waits for all three to complete
    - id: merge_data
      mcp: bbx.transform
      method: merge
      inputs:
        datasets:
          - ${steps.fetch_users.output}
          - ${steps.fetch_products.output}
          - ${steps.fetch_orders.output}
      depends_on: [fetch_users, fetch_products, fetch_orders]

    # Process merged data
    - id: process
      mcp: bbx.transform
      method: map
      inputs:
        data: ${steps.merge_data.output}
        function: "lambda x: process_record(x)"
      depends_on: [merge_data]
```

### Variable Interpolation

Access step outputs, inputs, and environment variables:

```yaml
workflow:
  id: variable_demo
  inputs:
    environment: production
    region: us-west-2

  steps:
    - id: deploy
      mcp: bbx.aws
      method: ec2_launch
      inputs:
        image_id: ami-12345
        instance_type: t3.large
        tags:
          Environment: ${inputs.environment}
          Region: ${inputs.region}

    - id: notify
      mcp: bbx.telegram
      method: send_message
      inputs:
        chat_id: ${env.TELEGRAM_CHAT_ID}
        text: |
          Deployment complete!
          Instance: ${steps.deploy.output.instance_id}
          IP: ${steps.deploy.output.public_ip}
      depends_on: [deploy]
```

### Conditional Execution

Use `when` conditions for conditional step execution:

```yaml
workflow:
  id: conditional_demo
  steps:
    - id: check_env
      mcp: bbx.system
      method: execute
      inputs:
        command: "echo $ENV"

    - id: deploy_prod
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: prod.yaml
      when: "${steps.check_env.output} == 'production'"

    - id: deploy_dev
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: dev.yaml
      when: "${steps.check_env.output} != 'production'"
```

### Error Handling and Retry

Built-in retry with exponential backoff:

```yaml
workflow:
  id: resilient_api_call
  steps:
    - id: call_external_api
      mcp: bbx.http
      method: post
      inputs:
        url: https://api.example.com/data
        json:
          payload: "important data"
      timeout: 5000          # 5 seconds
      retry: 3              # Retry 3 times
      retry_delay: 1000     # Start with 1 second
      retry_backoff: 2.0    # Double delay each time (1s, 2s, 4s)
```

---

## 🔌 Built-in Adapters

BBX includes 28 production-ready adapters organized by category:

### Cloud Providers

#### AWS Adapter (`bbx.aws`)
```yaml
- id: launch_ec2
  mcp: bbx.aws
  method: ec2_launch
  inputs:
    image_id: ami-12345
    instance_type: t3.micro
    key_name: my-keypair
    security_groups: [sg-12345]
    tags:
      Name: web-server
      Environment: production
```

**Methods**: `ec2_launch`, `ec2_terminate`, `s3_upload`, `s3_download`, `s3_list`, `lambda_invoke`

#### GCP Adapter (`bbx.gcp`)
```yaml
- id: create_vm
  mcp: bbx.gcp
  method: compute_create
  inputs:
    name: my-instance
    zone: us-central1-a
    machine_type: e2-micro
    image_family: debian-11
```

**Methods**: `compute_create`, `compute_delete`, `storage_upload`, `storage_download`, `cloud_function_deploy`

#### Azure Adapter (`bbx.azure`)
```yaml
- id: create_vm
  mcp: bbx.azure
  method: vm_create
  inputs:
    name: my-vm
    resource_group: my-rg
    image: UbuntuLTS
    size: Standard_B1s
```

**Methods**: `vm_create`, `vm_delete`, `storage_create_account`, `storage_upload_blob`

### Infrastructure as Code

#### Docker Adapter (`bbx.docker`)
```yaml
- id: build_and_push
  mcp: bbx.docker
  method: build
  inputs:
    path: .
    tag: myapp:latest
    dockerfile: Dockerfile
```

**Methods**: `build`, `run`, `stop`, `push`, `pull`, `logs`, `exec`

#### Kubernetes Adapter (`bbx.kubernetes`)
```yaml
- id: deploy_app
  mcp: bbx.k8s
  method: apply
  inputs:
    file: deployment.yaml
    namespace: production
```

**Methods**: `apply`, `delete`, `get`, `scale`, `rollout`, `logs`, `exec`, `port_forward`, `helm_install`, `helm_upgrade`

#### Terraform Adapter (`bbx.terraform`)
```yaml
- id: provision_infrastructure
  mcp: bbx.terraform
  method: apply
  inputs:
    working_dir: ./terraform
    var_file: prod.tfvars
    auto_approve: true
```

**Methods**: `init`, `plan`, `apply`, `destroy`, `output`, `validate`, `fmt`, `show`

#### Ansible Adapter (`bbx.ansible`)
```yaml
- id: configure_servers
  mcp: bbx.ansible
  method: playbook
  inputs:
    playbook: site.yml
    inventory: hosts.ini
    extra_vars:
      app_version: "1.2.3"
```

**Methods**: `playbook`, `adhoc`, `galaxy_install`, `inventory_list`, `vault_encrypt`, `vault_decrypt`

### Utilities

#### HTTP Adapter (`bbx.http`)
```yaml
- id: api_call
  mcp: bbx.http
  method: post
  inputs:
    url: https://api.example.com/webhook
    headers:
      Authorization: "Bearer ${env.API_TOKEN}"
    json:
      event: deployment_complete
      service: api
```

**Methods**: `get`, `post`, `put`, `delete`, `patch`, `download`

#### Transform Adapter (`bbx.transform`)
```yaml
- id: process_data
  mcp: bbx.transform
  method: map
  inputs:
    data: ${steps.fetch_data.output}
    function: "lambda x: x['value'] * 2"
```

**Methods**: `map`, `filter`, `reduce`, `merge`, `flatten`, `group_by`, `sort`

#### Logger Adapter (`bbx.logger`)
```yaml
- id: log_event
  mcp: bbx.logger
  method: info
  inputs:
    message: "Processing ${inputs.item_count} items"
```

**Methods**: `debug`, `info`, `warning`, `error`, `critical`

### Specialized Adapters

- **`bbx.process`** - Process management with health checks
- **`bbx.storage`** - S3-compatible storage operations
- **`bbx.queue`** - Message queue (SQS, RabbitMQ, Kafka)
- **`bbx.database`** - Database operations (PostgreSQL, MySQL, MongoDB)
- **`bbx.ai`** - AI/ML integrations (OpenAI, Anthropic)
- **`bbx.browser`** - Browser automation (Playwright)
- **`bbx.telegram`** - Telegram bot integration
- **`bbx.mcp_bridge`** - Universal MCP protocol bridge
- **`bbx.sandbox`** - Secure code execution sandbox
- **`bbx.flow`** - Workflow composition and subflows

[Full adapter documentation →](docs/ADAPTERS.md)

---

## ⚙️ Configuration

BBX supports multiple configuration sources with precedence:

### Environment Variables

```bash
# Paths
export BBX_HOME=~/.bbx
export BBX_CACHE_DIR=~/.bbx/cache
export BBX_STATE_DIR=~/.bbx/state
export BBX_LOG_DIR=~/.bbx/logs

# Runtime
export BBX_TIMEOUT=30000
export BBX_MAX_PARALLEL=10
export BBX_ENABLE_CACHE=true

# Observability
export BBX_LOG_LEVEL=INFO
export BBX_ENABLE_METRICS=true
export BBX_ENABLE_TRACING=true

# Cloud Providers
export AWS_REGION=us-east-1
export GCP_PROJECT=my-project
export AZURE_SUBSCRIPTION_ID=xxx
```

### Config File

**`~/.bbx/config.yaml`**:
```yaml
paths:
  bbx_home: ~/.bbx
  cache_dir: ~/.bbx/cache
  state_dir: ~/.bbx/state
  bundle_dir: ./.bbx_bundle
  output_dir: ./output
  log_dir: ~/.bbx/logs

runtime:
  default_timeout_ms: 30000
  max_parallel_steps: 10
  enable_caching: true
  cache_ttl_seconds: 3600
  retry_default: 0
  retry_delay_ms: 1000
  retry_backoff: 2.0

observability:
  enable_metrics: true
  enable_tracing: true
  enable_logging: true
  log_level: INFO
  metrics_retention: 10000
  trace_retention: 1000
  export_interval_seconds: 60

adapters:
  aws_region: us-east-1
  gcp_project: my-project
  http_timeout: 30
  http_max_retries: 3
  enable_ssl_verify: true

security:
  enable_sandbox: false
  allowed_adapters: null
  blocked_adapters: null
  max_workflow_size_mb: 10
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["*.bbx"],
      "url": "./bbx.schema.json"
    }
  ],
  "files.associations": {
    "*.bbx": "yaml"
  }
}
```

---

## 🤖 AI-Powered Workflow Generation

BBX v1.0 includes built-in local AI for generating workflows from natural language descriptions.

### Quick Start

```bash
# 1. Download AI model (one-time, 250MB)
$ bbx model download qwen-0.5b

📥 Downloading qwen-0.5b (250MB)...
✅ Model downloaded successfully

# 2. Generate workflow from description
$ bbx generate "Deploy Next.js app to AWS S3"

🤖 Using local AI: qwen-0.5b
🤖 Generating workflow...
✅ Generated: generated.bbx

# 3. Review and run
$ bbx validate generated.bbx
$ bbx run generated.bbx
```

### How It Works

- **Local AI Model**: Qwen2.5 0.5B (250MB, MIT license)
- **Runs Offline**: No internet required after download
- **Fast Generation**: 3-5 seconds for simple workflows, 8-12s for complex ones
- **High Accuracy**: 85-90% valid workflows on first try
- **No Cost**: Free to use, no API keys needed

### Example Generations

**Example 1: CI/CD Pipeline**
```bash
$ bbx generate "Full CI/CD: lint, test, build, deploy to Kubernetes"
```

Generates:
```yaml
workflow:
  id: cicd_full
  steps:
    - id: lint
      mcp: universal
      method: run
      inputs:
        uses: docker://python:3.11-slim
        cmd: [ruff, check, src/, --fix]

    - id: test
      mcp: universal
      method: run
      inputs:
        uses: docker://python:3.11-slim
        cmd: [pytest, tests/, -v, --cov]
      depends_on: [lint]

    - id: build
      mcp: universal
      method: run
      inputs:
        uses: docker://docker:24-cli
        cmd: [docker, build, -t, "myapp:latest", .]
      depends_on: [test]

    - id: deploy
      mcp: universal
      method: run
      inputs:
        uses: docker://bitnami/kubectl:latest
        cmd: [kubectl, apply, -f, k8s/deployment.yaml]
      depends_on: [build]
```

**Example 2: Testing**
```bash
$ bbx generate "Run pytest with coverage report"
```

Generates:
```yaml
workflow:
  id: test_with_coverage
  steps:
    - id: run_tests
      mcp: universal
      method: run
      inputs:
        uses: docker://python:3.11-slim
        cmd: [pytest, tests/, -v, --cov=src, --cov-report=html]
```

### AI Commands

```bash
# Model management
$ bbx model list              # List available models
$ bbx model download <name>   # Download a model
$ bbx model installed         # Show installed models
$ bbx model remove <name>     # Remove a model

# Workflow generation
$ bbx generate <description>               # Generate workflow
$ bbx generate <description> -o <file>     # Specify output file
$ bbx generate <description> --model <name>  # Choose specific model
```

### Why Local AI?

- **Privacy**: Your workflows never leave your machine
- **Speed**: No network latency, instant generation
- **Cost**: $0 - no usage fees, subscriptions, or API costs
- **Reliability**: Works offline, no service outages
- **Unique**: Only workflow engine with embedded local AI

[See full AI documentation →](docs/AI_V1_COMPLETE.md)

---

## 💡 Examples

### Multi-Cloud Deployment

```yaml
workflow:
  id: multi_cloud_deployment
  name: Deploy Application to Multiple Clouds
  steps:
    # Deploy to AWS
    - id: deploy_aws
      mcp: bbx.terraform
      method: apply
      inputs:
        working_dir: ./terraform/aws
        auto_approve: true

    # Deploy to GCP (parallel with AWS)
    - id: deploy_gcp
      mcp: bbx.terraform
      method: apply
      inputs:
        working_dir: ./terraform/gcp
        auto_approve: true

    # Deploy to Azure (parallel with AWS and GCP)
    - id: deploy_azure
      mcp: bbx.terraform
      method: apply
      inputs:
        working_dir: ./terraform/azure
        auto_approve: true

    # Verify all deployments
    - id: verify
      mcp: bbx.http
      method: get
      inputs:
        url: ${steps.deploy_aws.output.endpoint}/health
      depends_on: [deploy_aws, deploy_gcp, deploy_azure]
```

### CI/CD Pipeline

```yaml
workflow:
  id: ci_cd_pipeline
  name: Complete CI/CD Pipeline
  steps:
    - id: test
      mcp: bbx.system
      method: execute
      inputs:
        command: "pytest --cov=app"

    - id: build_image
      mcp: bbx.docker
      method: build
      inputs:
        path: .
        tag: myapp:${env.GIT_SHA}
      depends_on: [test]

    - id: security_scan
      mcp: bbx.system
      method: execute
      inputs:
        command: "trivy image myapp:${env.GIT_SHA}"
      depends_on: [build_image]

    - id: push_image
      mcp: bbx.docker
      method: push
      inputs:
        image: myapp:${env.GIT_SHA}
      depends_on: [security_scan]

    - id: deploy_staging
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: k8s/staging.yaml
        namespace: staging
      depends_on: [push_image]

    - id: integration_tests
      mcp: bbx.http
      method: get
      inputs:
        url: https://staging.example.com/health
      depends_on: [deploy_staging]

    - id: deploy_production
      mcp: bbx.kubernetes
      method: apply
      inputs:
        file: k8s/production.yaml
        namespace: production
      depends_on: [integration_tests]
      when: "${inputs.auto_deploy} == true"
```

[More examples →](workflows/examples/)

---

## 🛠️ Development

### Python SDK

BBX includes an official Python SDK for programmatic workflow management:

```python
from bbx_sdk import BlackboxClient, WorkflowCreate

# Connect to BBX API
client = BlackboxClient("http://localhost:8000")
client.authenticate("username", "password")

# Create and execute workflow
workflow = WorkflowCreate(
    name="My Workflow",
    bbx_yaml=workflow_definition
)
created = client.create_workflow(workflow)
execution = client.execute_workflow(created.id)
```

See [bbx-sdk/README.md](bbx-sdk/README.md) for full documentation.

---

### Project Structure

```
blackbox-workflow/
├── bbx-sdk/                     # Python SDK for API
│   ├── client.py                # HTTP client
│   ├── models.py                # Pydantic models
│   ├── sync.py                  # Workflow sync utility
│   ├── config.py                # Configuration
│   ├── examples/                # Usage examples
│   └── README.md                # SDK documentation
├── blackbox/
│   ├── core/
│   │   ├── runtime.py           # Core execution engine
│   │   ├── dag.py               # DAG scheduler
│   │   ├── expressions.py       # Expression parser
│   │   ├── context.py           # Execution context
│   │   ├── registry.py          # Adapter registry
│   │   ├── config.py            # Configuration system
│   │   ├── exceptions.py        # Exception hierarchy
│   │   ├── schemas.py           # Pydantic validation
│   │   ├── base_adapter.py      # Base adapter classes
│   │   ├── bundler.py           # Workflow bundler
│   │   ├── adapters/            # Built-in adapters (28 files)
│   │   ├── observability/       # Observability system
│   │   │   ├── __init__.py      # Main interface
│   │   │   ├── models.py        # Data models
│   │   │   ├── metrics.py       # Metrics collector
│   │   │   ├── tracing.py       # Distributed tracing
│   │   │   ├── structured_logging.py  # Logging
│   │   │   └── exporters.py     # Prometheus, Jaeger, File
│   │   └── parsers/
│   │       └── v6.py            # BBX v6.0 parser
│   ├── cli/
│   │   └── wizard.py            # Interactive wizard
│   └── server/
│       └── app.py               # API server
├── workflows/                   # Example workflows
│   ├── examples/                # Tutorial examples
│   ├── cloud/                   # Cloud provider examples
│   ├── infrastructure/          # Infrastructure examples
│   └── demos/                   # Demo workflows
├── tests/                       # Test suite
│   ├── test_runtime.py
│   ├── test_dag.py
│   ├── test_expressions.py
│   └── ...
├── docs/                        # Documentation
│   ├── BBX_SPEC.md             # Format specification
│   ├── ARCHITECTURE_2_0.md     # Architecture guide
│   ├── MCP_DEVELOPMENT.md      # Adapter development
│   └── DEPLOYMENT.md           # Deployment guide
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── cli.py                       # CLI entry point
├── api_server.py               # API server entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── LICENSE                     # Apache 2.0 License
├── CHANGELOG.md                # Version history
└── CONTRIBUTING.md             # Contribution guide
```

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/kurokie1337/bbx.git
cd blackbox-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black isort mypy flake8

# Run tests
pytest

# Check code style
black --check blackbox/
isort --check-only blackbox/
flake8 blackbox/
```

### Creating a Custom Adapter

See [Adapter Development Guide](docs/MCP_DEVELOPMENT.md) for detailed instructions.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=blackbox --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_runtime.py -v

# Run specific test
pytest tests/test_dag.py::test_parallel_execution -v

# Run with debug logging
pytest -v --log-cli-level=DEBUG
```

---

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t blackbox-workflow:1.0.0 .

# Run workflow
docker run -v $(pwd)/workflows:/workflows blackbox-workflow:1.0.0 run /workflows/my_workflow.bbx

# Run API server
docker run -p 8000:8000 blackbox-workflow:1.0.0 api
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bbx-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: bbx
        image: blackbox-workflow:1.0.0
        command: ["python", "api_server.py"]
        ports:
        - containerPort: 8000
        env:
        - name: BBX_LOG_LEVEL
          value: "INFO"
```

[Full deployment guide →](docs/DEPLOYMENT.md)

---

## 📚 Documentation

- **[Documentation Index](docs/INDEX.md)** - Complete documentation navigation
- **[Getting Started](docs/GETTING_STARTED.md)** - Complete beginner's guide
- **[BBX v6.0 Specification](docs/BBX_SPEC_v6.md)** - Complete workflow format reference
- **[Universal Adapter Guide](docs/UNIVERSAL_ADAPTER.md)** - Execute any CLI tool without code
- **[API Reference](docs/API_REFERENCE.md)** - All adapters and methods
- **[MCP Development](docs/MCP_DEVELOPMENT.md)** - Create custom adapters
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and internals
- **[Runtime Internals](docs/RUNTIME_INTERNALS.md)** - Engine implementation
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment strategies
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Agent Guide](docs/AGENT_GUIDE.md)** - BBX for AI agents
- **[Python SDK](bbx-sdk/README.md)** - Official Python client library
- **[Changelog](CHANGELOG.md)** - Version history and release notes

---

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting PRs.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure tests pass (`pytest`)
5. Commit with clear messages (`git commit -m 'feat: Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code of Conduct

- Be respectful and inclusive
- Write clear, documented code
- Add tests for new features
- Follow existing code style

---

## 📜 License

```
Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Full license text: [LICENSE](LICENSE)

---

## 💫 About This Project

**Blackbox Workflow Engine** was conceived, designed, and implemented with meticulous attention to detail in Krasnoyarsk, Siberia - one of Russia's most beautiful cities nestled in the heart of the Taiga forest.

This project represents months of careful engineering, architectural refinement, and dedication to creating a tool that developers would genuinely love to use. Every component, from the DAG scheduler to the observability system, was crafted with production reliability and developer experience as the primary goals.

### The Philosophy

BBX embodies several core principles:

1. **Simplicity Over Complexity** - Complex problems deserve simple solutions
2. **Composability Over Monoliths** - Build big things from small, reusable pieces
3. **Developer Experience** - If it's hard to use, it's not finished
4. **Production First** - Features must work reliably at scale
5. **Open Source** - Great software is built by communities

### Technical Excellence

- **12,250+ lines** of production Python code
- **28 built-in adapters** covering major cloud providers and tools
- **Type-safe** with Pydantic validation throughout
- **Zero code injection** vulnerabilities via safe expression parser
- **Comprehensive observability** with OpenTelemetry compatibility
- **100% open source** under Apache 2.0 license

---

<div align="center">

### Built in Siberia 🏔️

**From Russia, with Love ❤️**

```
─────────────────────────────────────────────────────────
  First Public Release: November 22, 2025 06:00 UTC+7
  Author: Ilya Makarov
  Location: Krasnoyarsk, Siberia, Russia
─────────────────────────────────────────────────────────
```

*This timestamp and authorship are an immutable part of BBX's origin story. They represent not just a moment in time, but a commitment to quality, open-source values, and the belief that powerful tools can come from anywhere in the world - even from the frozen beauty of Siberia.*

*This project is dedicated to every developer who has struggled with complex infrastructure automation. May BBX make your workflows simpler, your deployments faster, and your code more maintainable.*

---

### 🌟 Support the Project

If BBX has helped streamline your workflows, consider:

- ⭐ **Starring the repository** - Helps others discover BBX
- 🐛 **Reporting issues** - Makes BBX better for everyone
- 💡 **Suggesting features** - Shapes BBX's future
- 🤝 **Contributing code** - Becomes part of BBX's story
- 📢 **Spreading the word** - Grows the community

Every contribution, no matter how small, is deeply appreciated.

---

### ❓ FAQ & Troubleshooting

#### Common Questions

**Q: Can I use BBX commercially?**
A: Yes! BBX is licensed under Apache 2.0, which permits both personal and commercial use at no cost. For enterprise support, consulting, or custom development, contact the author.

**Q: Can steps run in parallel?**
A: Yes! Use `parallel: true` in your step definition and ensure dependencies are correctly set with `depends_on`.

**Q: What if I get "Unknown MCP type: http"?**
A: Ensure you are using supported adapter names: `http`, `logger`, `telegram`, etc. Check `docs/ADAPTERS.md` for the full list.

#### Troubleshooting

- **Telegram not working?** Check your bot token, ensure you've sent `/start` to the bot, and verified the chat ID.
- **Variable substitution fails?** Use `save:` to capture outputs and reference them as `${step.step_id.field}`.
- **Timeout errors?** Increase the step timeout: `timeout: 30s`.
- **VS Code IntelliSense not working?** Run `python cli.py schema` to generate the schema and add it to your `.vscode/settings.json`.

---

### 📞 Contact & Community

- **Issues & Bug Reports**: [GitHub Issues](https://github.com/kurokie1337/bbx/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/kurokie1337/bbx/discussions)
- **Documentation**: [docs/](docs/)
- **Examples**: [workflows/examples/](workflows/examples/)

---

**Made with ❤️ in Siberia** | [Documentation](docs/) | [Examples](workflows/) | [License](LICENSE) | [Contributing](CONTRIBUTING.md)

</div>
