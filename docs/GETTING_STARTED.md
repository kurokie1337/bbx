# Getting Started with BBX

**Complete beginner's guide to Blackbox Workflow Engine**

Welcome to BBX! This guide will help you go from zero to running your first workflow in under 10 minutes.

---

## 📋 Table of Contents

- [What is BBX?](#what-is-bbx)
- [Installation](#installation)
- [Your First Workflow](#your-first-workflow)
- [Understanding BBX Basics](#understanding-bbx-basics)
- [Common Patterns](#common-patterns)
- [Next Steps](#next-steps)

---

## What is BBX?

**BBX (Blackbox Workflow Engine)** is a production-grade workflow automation platform that lets you orchestrate complex infrastructure operations using simple YAML files.

### Key Features

- **Declarative Workflows** - Write what you want, not how to do it
- **Automatic Parallelization** - BBX analyzes dependencies and runs steps in parallel
- **Built-in Adapters** - 28+ integrations for cloud providers, Kubernetes, Docker, etc.
- **Universal Adapter** - Execute any CLI tool without writing code
- **Type Safety** - Full validation with helpful error messages
- **Production Ready** - Used in production environments

### When to Use BBX

✅ **Perfect for:**
- Multi-cloud deployments (AWS, GCP, Azure)
- CI/CD pipelines
- Infrastructure automation (Terraform, Ansible, Kubernetes)
- Data processing workflows
- System automation tasks

❌ **Not designed for:**
- Real-time event processing (use message queues)
- High-frequency trading (use specialized platforms)
- Simple bash scripts (just use bash)

---

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** package manager
- **(Optional) Docker** for Universal Adapter features

### Install BBX

```bash
# Clone the repository
git clone https://github.com/kurokie1337/bbx.git
cd bbx

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py --version
```

### Verify Installation

```bash
# Run system check
python cli.py system check

# Expected output:
# ✅ Python version: 3.11.0
# ✅ Dependencies: OK
# ✅ BBX Core: OK
# ✅ Adapters loaded: 28
```

---

## Your First Workflow

Let's create a simple workflow that demonstrates BBX basics.

### Step 1: Create a Workflow File

Create `hello_world.bbx`:

```yaml
workflow:
  id: hello_world
  name: Hello World Example
  version: "6.0"
  description: My first BBX workflow

  steps:
    - id: greet
      mcp: bbx.logger
      method: info
      inputs:
        message: "Hello from BBX! 🚀"

    - id: get_date
      mcp: bbx.system
      method: execute
      inputs:
        command: "date"
      depends_on: [greet]

    - id: summary
      mcp: bbx.logger
      method: info
      inputs:
        message: "Workflow completed at ${steps.get_date.output}"
      depends_on: [get_date]
```

### Step 2: Run It

```bash
python cli.py run hello_world.bbx
```

### Step 3: See the Results

```
🚀 Starting: Hello World Example
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️  Executing greet (bbx.logger.info)
    Message: Hello from BBX! 🚀
✅ greet completed

▶️  Executing get_date (bbx.system.execute)
    Command: date
✅ get_date completed
    Output: Fri Nov 22 06:00:00 UTC 2025

▶️  Executing summary (bbx.logger.info)
    Message: Workflow completed at Fri Nov 22 06:00:00 UTC 2025
✅ summary completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Workflow completed successfully
   Duration: 0.5s
   Steps: 3/3 successful
```

**Congratulations!** 🎉 You just ran your first BBX workflow!

---

## Understanding BBX Basics

### Workflow Structure

Every BBX workflow has this basic structure:

```yaml
workflow:
  id: unique_identifier        # Required: Unique ID
  name: Human-Readable Name    # Required: Display name
  version: "6.0"               # Required: BBX format version
  description: What it does    # Optional: Documentation

  inputs:                      # Optional: Workflow parameters
    param_name: type

  steps:                       # Required: List of steps
    - id: step1                # Step definition
      mcp: bbx.adapter_name
      method: method_name
      inputs:
        key: value

  outputs:                     # Optional: Workflow outputs
    result: ${steps.step1.output}
```

### Steps

A **step** is a single unit of work. Each step:

1. Has a unique `id`
2. Calls an adapter (`mcp`)
3. Executes a method
4. Receives inputs
5. Produces outputs

```yaml
- id: send_message           # Unique step ID
  mcp: bbx.telegram          # Adapter to use
  method: send_message       # Method to call
  inputs:                    # Method inputs
    chat_id: "123456"
    text: "Hello!"
```

### Dependencies

Use `depends_on` to control execution order:

```yaml
- id: build
  mcp: bbx.docker
  method: build
  inputs:
    path: .

- id: test
  mcp: bbx.system
  method: execute
  inputs:
    command: pytest
  depends_on: [build]        # Runs AFTER build

- id: deploy
  mcp: bbx.kubernetes
  method: apply
  inputs:
    file: deployment.yaml
  depends_on: [test]         # Runs AFTER test
```

**Execution flow:**
```
build → test → deploy
```

### Parallel Execution

Steps without dependencies run in parallel automatically:

```yaml
steps:
  # These three run simultaneously
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

  # This waits for all three
  - id: process
    mcp: bbx.transform
    method: merge
    inputs:
      datasets:
        - ${steps.fetch_users.output}
        - ${steps.fetch_products.output}
        - ${steps.fetch_orders.output}
    depends_on: [fetch_users, fetch_products, fetch_orders]
```

**Execution flow:**
```
fetch_users   ┐
fetch_products├─→ process
fetch_orders  ┘
```

### Variable Interpolation

Access data from other steps, inputs, or environment:

```yaml
workflow:
  inputs:
    environment: production
    region: us-west-2

  steps:
    - id: deploy
      mcp: bbx.aws
      method: ec2_launch
      inputs:
        instance_type: t3.large
        tags:
          Environment: ${inputs.environment}    # From workflow inputs
          Region: ${inputs.region}

    - id: notify
      mcp: bbx.telegram
      method: send_message
      inputs:
        chat_id: ${env.TELEGRAM_CHAT_ID}       # From environment
        text: |
          Deployed instance: ${steps.deploy.output.instance_id}
          Public IP: ${steps.deploy.output.public_ip}
      depends_on: [deploy]
```

### Conditional Execution

Use `when` to run steps conditionally:

```yaml
- id: check_env
  mcp: bbx.system
  method: execute
  inputs:
    command: "echo $ENVIRONMENT"

- id: deploy_production
  mcp: bbx.kubernetes
  method: apply
  inputs:
    file: prod.yaml
  when: "${steps.check_env.output} == 'production'"

- id: deploy_staging
  mcp: bbx.kubernetes
  method: apply
  inputs:
    file: staging.yaml
  when: "${steps.check_env.output} != 'production'"
```

---

## Common Patterns

### Pattern 1: HTTP API Call

```yaml
workflow:
  id: api_call
  steps:
    - id: fetch_data
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.github.com/repos/kurokie1337/bbx
        headers:
          Accept: application/json

    - id: log_result
      mcp: bbx.logger
      method: info
      inputs:
        message: "Stars: ${steps.fetch_data.output.stargazers_count}"
      depends_on: [fetch_data]
```

### Pattern 2: Execute Shell Commands

```yaml
workflow:
  id: run_commands
  steps:
    - id: list_files
      mcp: bbx.system
      method: execute
      inputs:
        command: "ls -la"

    - id: check_disk
      mcp: bbx.system
      method: execute
      inputs:
        command: "df -h"
```

### Pattern 3: Docker Build and Push

```yaml
workflow:
  id: docker_workflow
  steps:
    - id: build
      mcp: bbx.docker
      method: build
      inputs:
        path: .
        tag: myapp:latest

    - id: push
      mcp: bbx.docker
      method: push
      inputs:
        image: ${steps.build.output.image}
      depends_on: [build]
```

### Pattern 4: Multi-Cloud Deployment

```yaml
workflow:
  id: multi_cloud
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

    # Verify both deployments
    - id: verify
      mcp: bbx.logger
      method: info
      inputs:
        message: "Both clouds deployed successfully"
      depends_on: [deploy_aws, deploy_gcp]
```

### Pattern 5: Error Handling with Retry

```yaml
workflow:
  id: resilient_workflow
  steps:
    - id: flaky_api
      mcp: bbx.http
      method: get
      inputs:
        url: https://api.example.com/data
      timeout: 5000        # 5 seconds
      retry: 3            # Retry 3 times
      retry_delay: 1000   # Start with 1 second delay
      retry_backoff: 2.0  # Double delay each time (1s, 2s, 4s)
```

---

## Next Steps

### Learn More About BBX

1. **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Complete workflow format reference
2. **[Universal Adapter Guide](UNIVERSAL_ADAPTER.md)** - Execute any CLI tool without code
3. **[Built-in Adapters](API_REFERENCE.md)** - 28+ ready-to-use integrations
4. **[Agent Guide](AGENT_GUIDE.md)** - Use BBX with AI agents

### Try More Examples

Explore the `examples/` directory:

```bash
# Simple examples
python cli.py run examples/poc_simple_alpine.bbx
python cli.py run examples/poc_universal_kubectl.bbx

# Real-world examples
python cli.py run examples/pipeline_cicd.bbx
python cli.py run examples/universal_multi_step.bbx
```

### Use the Dashboard

Start the web dashboard for visual workflow management:

```bash
python api_server.py

# Open browser to http://localhost:8000
```

### Create Your Own Adapter

See [MCP Development Guide](MCP_DEVELOPMENT.md) to create custom adapters.

### Join the Community

- **Issues**: [GitHub Issues](https://github.com/kurokie1337/bbx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kurokie1337/bbx/discussions)
- **Documentation**: [docs/](.)

---

## Quick Reference

### Essential CLI Commands

```bash
# Run a workflow
python cli.py run workflow.bbx

# Run with inputs
python cli.py run workflow.bbx --input key=value

# Validate workflow syntax
python cli.py validate workflow.bbx

# List available adapters
python cli.py adapters list

# System health check
python cli.py system check

# Start API server
python api_server.py
```

### Essential Adapters

| Adapter | Purpose | Example |
|---------|---------|---------|
| `bbx.logger` | Logging | `method: info` |
| `bbx.system` | Shell commands | `method: execute` |
| `bbx.http` | HTTP requests | `method: get` |
| `bbx.docker` | Docker operations | `method: build` |
| `bbx.kubernetes` | K8s operations | `method: apply` |
| `bbx.terraform` | Terraform IaC | `method: apply` |
| `bbx.aws` | AWS operations | `method: ec2_launch` |
| `bbx.universal` | Any CLI tool | `method: run` |

### Variable Syntax

```yaml
${inputs.name}              # Workflow input
${env.VAR_NAME}             # Environment variable
${steps.step_id.output}     # Step output
${steps.step_id.output.key} # Step output field
```

### Common Workflow Metadata

```yaml
workflow:
  id: my_workflow           # Required
  name: My Workflow         # Required
  version: "6.0"            # Required
  description: What it does # Optional

  inputs:                   # Optional
    param1: type
    param2: type

  outputs:                  # Optional
    result: ${steps.x.output}
```

---

## Troubleshooting

### "Unknown MCP type: xxx"

**Problem:** Adapter name is incorrect.

**Solution:** Check available adapters:
```bash
python cli.py adapters list
```

### "Missing required input: xxx"

**Problem:** Adapter requires an input you didn't provide.

**Solution:** Check adapter documentation or use `--help`:
```bash
python cli.py adapters info bbx.http
```

### "Workflow validation failed"

**Problem:** YAML syntax error or missing required fields.

**Solution:** Validate workflow:
```bash
python cli.py validate workflow.bbx
```

### "Timeout error"

**Problem:** Step took too long to execute.

**Solution:** Increase timeout:
```yaml
- id: long_step
  timeout: 300000  # 5 minutes in milliseconds
```

---

## Summary

You've learned:

✅ How to install BBX
✅ How to create and run workflows
✅ Basic workflow structure
✅ Dependencies and parallel execution
✅ Variable interpolation
✅ Common patterns
✅ Essential CLI commands

### What's Next?

1. **Practice**: Modify the examples to fit your use case
2. **Explore**: Try different adapters from the [API Reference](API_REFERENCE.md)
3. **Build**: Create workflows for your infrastructure
4. **Share**: Contribute back to the community

---

**Welcome to the BBX community!** 🚀

If you have questions, check the [Troubleshooting Guide](TROUBLESHOOTING.md) or [open an issue](https://github.com/kurokie1337/bbx/issues).

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0
