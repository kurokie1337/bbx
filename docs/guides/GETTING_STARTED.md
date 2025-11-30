# Getting Started with BBX

**Detailed guide to Blackbox Workflow Engine**

> For a quick overview, see the main **[README](../../README.md)**.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/kurokie1337/bbx.git
cd bbx

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python cli.py --version
python cli.py system
```

---

## Your First Workflow

Create `hello.bbx`:

```yaml
id: hello_world
name: Hello World
version: "1.0.0"

steps:
  greet:
    use: logger.info
    args:
      message: "Hello from BBX!"

  get_time:
    use: system.exec
    args:
      command: "date"
    depends_on: [greet]
```

Run it:

```bash
python cli.py run hello.bbx
```

---

## Core Concepts

### Steps and Dependencies

Steps run in parallel unless you specify dependencies:

```yaml
steps:
  step_a:
    use: logger.info
    args:
      message: "A runs first"

  step_b:
    use: logger.info
    args:
      message: "B runs after A"
    depends_on: [step_a]

  step_c:
    use: logger.info
    args:
      message: "C runs parallel with A"
```

### Variable Interpolation

Reference outputs from other steps:

```yaml
steps:
  fetch:
    use: http.get
    args:
      url: "https://api.example.com/data"

  process:
    use: logger.info
    args:
      message: "Got: ${steps.fetch.output}"
    depends_on: [fetch]
```

### Inputs

Define workflow inputs with defaults:

```yaml
id: deploy
name: Deploy App

inputs:
  environment:
    type: string
    default: "staging"
  version:
    type: string
    required: true

steps:
  deploy:
    use: logger.info
    args:
      message: "Deploying ${inputs.version} to ${inputs.environment}"
```

Run with inputs:

```bash
python cli.py run deploy.bbx -i version=1.2.3 -i environment=production
```

---

## Available Adapters

| Adapter | Description | Example |
|---------|-------------|---------|
| `logger` | Logging | `logger.info`, `logger.error` |
| `system` | Shell commands | `system.exec` |
| `http` | HTTP requests | `http.get`, `http.post` |
| `file` | File operations | `file.read`, `file.write` |
| `string` | String manipulation | `string.split`, `string.replace` |
| `state` | Persistent state | `state.get`, `state.set` |
| `workflow` | Nested workflows | `workflow.run` |
| `a2a` | Agent-to-Agent | `a2a.call`, `a2a.discover` |

Full list: `python cli.py adapters`

---

## Workspaces

Create isolated project environments:

```bash
# Create workspace
python cli.py workspace create my-project

# Set as active
python cli.py workspace set ~/.bbx/workspaces/my_project

# Run main.bbx from workspace
python cli.py run
```

---

## Background Execution

Run long workflows in background:

```bash
# Start background
python cli.py run deploy.bbx --background

# Check status
python cli.py ps

# View logs
python cli.py logs <exec_id>

# Kill if needed
python cli.py kill <exec_id>
```

---

## Next Steps

- **[Workflow Format](../reference/WORKFLOW_FORMAT.md)** - Full BBX workflow specification
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Agent Guide](AGENT_GUIDE.md)** - Guide for AI agents working with BBX
- **[Examples](../../examples/)** - More workflow examples

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk**
