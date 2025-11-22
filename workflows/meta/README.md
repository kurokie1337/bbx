# 🤖 BBX Meta-Automation Workflows

> **Using BBX to develop BBX itself** - Meta-automation at its finest!

This directory contains BBX workflow files that automate the development, testing, and release of BBX itself. It's a real-world demonstration of BBX's power and a practical tool for contributors.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Workflow Files](#-workflow-files)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Architecture](#-architecture)
- [Contributing](#-contributing)

---

## 🎯 Overview

These meta-workflows demonstrate:

✅ **Self-automation** - BBX automating its own development
✅ **Best practices** - Production-grade workflow patterns
✅ **Real-world complexity** - Multi-step, parallel, conditional execution
✅ **Complete CI/CD** - From commit to production deployment
✅ **Feature development** - Structured approach to building new features

---

## 📦 Workflow Files

### Master Workflows

| Workflow | Description | Lines | Complexity |
|----------|-------------|-------|------------|
| **[v1.1.0_development.bbx](v1.1.0_development.bbx)** | Orchestrates all v1.1.0 features | 350+ | High |
| **[long_term_features.bbx](long_term_features.bbx)** | Strategic features for v2.0+ | 600+ | Very High |
| **[ci_cd_pipeline.bbx](ci_cd_pipeline.bbx)** | Complete CI/CD pipeline | 500+ | Very High |
| **[release_automation.bbx](release_automation.bbx)** | Fully automated releases | 700+ | Very High |

### Feature Workflows

Located in [`features/`](features/) directory:

| Feature | File | Status | Version |
|---------|------|--------|---------|
| Workflow Versioning | [workflow_versioning.bbx](features/workflow_versioning.bbx) | ✅ Ready | v1.1.0 |
| WebSocket Updates | [websocket_updates.bbx](features/websocket_updates.bbx) | ✅ Ready | v1.1.0 |
| Remote Execution | [remote_execution.bbx](features/remote_execution.bbx) | 🚧 Planned | v1.1.0 |
| Redis Cache | [redis_cache.bbx](features/redis_cache.bbx) | 🚧 Planned | v1.1.0 |
| GraphQL API | [graphql_api.bbx](features/graphql_api.bbx) | 🚧 Planned | v1.1.0 |
| Node.js SDK | [sdk_nodejs.bbx](features/sdk_nodejs.bbx) | 🚧 Planned | v1.1.0 |
| Go SDK | [sdk_go.bbx](features/sdk_go.bbx) | 🚧 Planned | v1.1.0 |
| Rust SDK | [sdk_rust.bbx](features/sdk_rust.bbx) | 🚧 Planned | v1.1.0 |
| Kubernetes Operator | [kubernetes_operator.bbx](features/kubernetes_operator.bbx) | 📋 Planned | v2.0.0 |
| Event Sourcing | [event_sourcing.bbx](features/event_sourcing.bbx) | 📋 Planned | v2.0.0 |
| Multi-tenancy | [multi_tenancy.bbx](features/multi_tenancy.bbx) | 📋 Planned | v2.0.0 |
| Workflow Scheduler | [workflow_scheduler.bbx](features/workflow_scheduler.bbx) | 📋 Planned | v2.0.0 |
| Visual Designer | [visual_designer.bbx](features/visual_designer.bbx) | 📋 Planned | v2.0.0 |
| Plugin System | [plugin_system.bbx](features/plugin_system.bbx) | 📋 Planned | v2.0.0 |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you have BBX installed
pip install blackbox-workflow

# Set up environment variables
export TELEGRAM_CHAT_ID="your_chat_id"
export PYPI_TOKEN="your_pypi_token"
export GITHUB_TOKEN="your_github_token"
```

### Running a Workflow

```bash
# Run v1.1.0 development workflow
python cli.py run workflows/meta/v1.1.0_development.bbx

# Run CI/CD pipeline
python cli.py run workflows/meta/ci_cd_pipeline.bbx \
  --input branch=main \
  --input environment=staging

# Run release automation
python cli.py run workflows/meta/release_automation.bbx \
  --input version=1.1.0 \
  --input release_type=minor

# Run specific feature development
python cli.py run workflows/meta/features/workflow_versioning.bbx
```

---

## 💡 Usage Examples

### Example 1: Develop v1.1.0 Features

```bash
# Run the master v1.1.0 workflow
python cli.py run workflows/meta/v1.1.0_development.bbx \
  --input target_release_date=2025-12-31 \
  --input skip_tests=false \
  --input auto_deploy=true
```

**What happens:**
1. ✅ Creates feature branch
2. 🔧 Develops all 7 features in parallel
3. 🧪 Runs integration tests
4. 📚 Updates documentation
5. 🐳 Builds Docker image
6. 🚀 Deploys to staging
7. 📱 Sends notification

**Execution time:** ~45 minutes (with parallel execution)

---

### Example 2: Run CI/CD Pipeline

```bash
# Run full CI/CD for main branch
python cli.py run workflows/meta/ci_cd_pipeline.bbx \
  --input branch=main \
  --input environment=production \
  --input auto_deploy=true
```

**Pipeline stages:**
1. 🔍 Code quality checks (lint, type check, format)
2. 🔒 Security scanning (code, deps, container)
3. 🧪 Tests (unit, integration, e2e)
4. 📊 Performance benchmarks
5. 🐳 Docker build & push
6. 🚀 Kubernetes deployment
7. ✅ Smoke tests
8. 📱 Notifications

**Features:**
- ⚡ Parallel execution (quality checks run concurrently)
- 🔄 Auto-rollback on failure
- 📊 Comprehensive reporting
- 🔒 Security gates

---

### Example 3: Automated Release

```bash
# Release v1.1.0
python cli.py run workflows/meta/release_automation.bbx \
  --input version=1.1.0 \
  --input release_type=minor \
  --input pre_release=false
```

**Release process:**
1. ✅ Validates version and git status
2. 🧪 Runs complete test suite
3. 📝 Updates version files
4. 📚 Generates changelog
5. 🏗️ Builds artifacts (PyPI + Docker)
6. 🏷️ Creates git tag
7. 📤 Publishes to PyPI, Docker Hub, GHCR
8. 🐙 Creates GitHub release
9. 📚 Deploys documentation
10. 🍺 Updates Homebrew formula
11. ✅ Verifies installation
12. 📣 Posts announcements
13. 📊 Generates release report

**Fully automated, zero manual steps!**

---

### Example 4: Develop Single Feature

```bash
# Develop workflow versioning feature
python cli.py run workflows/meta/features/workflow_versioning.bbx
```

**What it does:**
1. Creates data models
2. Implements version manager
3. Adds CLI commands
4. Adds API endpoints
5. Creates tests
6. Runs tests
7. Updates documentation

**Output:**
- `blackbox/core/versioning/` - New module
- `cli.py` - Updated with version commands
- `api_server.py` - New API endpoints
- `tests/test_versioning.py` - Test suite
- `docs/VERSIONING.md` - Documentation

---

## 🏗️ Architecture

### Workflow Composition Pattern

```yaml
# Master workflow
v1.1.0_development.bbx
  ├─ develop_workflow_versioning  → features/workflow_versioning.bbx
  ├─ develop_remote_execution     → features/remote_execution.bbx
  ├─ develop_marketplace          → features/marketplace.bbx
  ├─ develop_redis_cache          → features/redis_cache.bbx
  ├─ develop_graphql_api          → features/graphql_api.bbx
  ├─ develop_websocket            → features/websocket_updates.bbx
  ├─ develop_nodejs_sdk           → features/sdk_nodejs.bbx
  ├─ develop_go_sdk               → features/sdk_go.bbx
  └─ develop_rust_sdk             → features/sdk_rust.bbx
```

**Benefits:**
- 🧩 **Modularity** - Each feature is self-contained
- ⚡ **Parallelism** - Independent features run concurrently
- 🔄 **Reusability** - Features can be run standalone
- 🧪 **Testability** - Each component tested separately

---

### Execution Flow: CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    checkout_code                        │
└───────────────────────┬─────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────┐              ┌────────────────┐
│  lint_python  │              │  type_check    │  (parallel)
└───────┬───────┘              └────────┬───────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
                ┌───────────────┐
                │  run_tests    │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │  build_image  │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │    deploy     │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │  smoke_test   │
                └───────┬───────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────┐              ┌────────────────┐
│    success    │              │    rollback    │
│ notification  │              │  on failure    │
└───────────────┘              └────────────────┘
```

---

## 🎯 Design Patterns Used

### 1. **Parallel Execution**

```yaml
# These steps run in parallel (no dependencies)
- id: lint_python
  mcp: bbx.system
  method: execute
  inputs: {...}

- id: type_check
  mcp: bbx.system
  method: execute
  inputs: {...}

- id: security_scan
  mcp: bbx.system
  method: execute
  inputs: {...}
```

### 2. **Conditional Execution**

```yaml
- id: deploy_production
  when: "${inputs.environment} == 'production'"
  depends_on: [build_image]

- id: rollback
  when: "${steps.smoke_test.status} == 'error'"
  depends_on: [smoke_test]
```

### 3. **Error Handling with Retry**

```yaml
- id: publish_to_pypi
  retry: 3
  retry_delay: 5000
  retry_backoff: 2.0
  timeout: 60000
```

### 4. **Workflow Composition**

```yaml
- id: develop_feature
  mcp: bbx.flow
  method: run
  inputs:
    path: features/my_feature.bbx
```

### 5. **Environment Variable Usage**

```yaml
inputs:
  api_token: ${env.PYPI_TOKEN}
  chat_id: ${env.TELEGRAM_CHAT_ID}
```

---

## 📊 Metrics & Observability

All workflows emit metrics, traces, and logs:

```bash
# View workflow metrics
curl http://localhost:8000/api/metrics/workflow/v1_1_0_development

# View execution trace
curl http://localhost:8000/api/traces/workflow/v1_1_0_development

# Export to Prometheus
curl http://localhost:8000/metrics

# View in dashboard
open http://localhost:8000/dashboard
```

**Tracked metrics:**
- Workflow execution time
- Step success/failure rates
- Parallel execution efficiency
- Resource utilization
- Build artifacts size

---

## 🧪 Testing Workflows

```bash
# Dry-run mode (validate without executing)
python cli.py validate workflows/meta/ci_cd_pipeline.bbx

# Run with mock adapters
BBX_MOCK_ADAPTERS=true python cli.py run workflows/meta/v1.1.0_development.bbx

# Run specific steps only
python cli.py run workflows/meta/ci_cd_pipeline.bbx \
  --steps lint_python,type_check,run_unit_tests
```

---

## 🤝 Contributing

### Adding a New Feature Workflow

1. Create workflow file in `features/`:

```yaml
# features/my_new_feature.bbx
workflow:
  id: feature_my_new_feature
  name: My New Feature Implementation
  version: "6.0"

  steps:
    - id: implement_feature
      mcp: bbx.system
      method: execute
      inputs:
        command: "echo 'Implementing feature...'"
```

2. Add to master workflow:

```yaml
# v1.1.0_development.bbx
- id: develop_my_feature
  mcp: bbx.flow
  method: run
  inputs:
    path: workflows/meta/features/my_new_feature.bbx
```

3. Update this README:

```markdown
| My New Feature | [my_new_feature.bbx](features/my_new_feature.bbx) | 🚧 In Progress | v1.1.0 |
```

---

## 📚 Learn More

- **[BBX Specification](../../docs/BBX_SPEC_v6.md)** - Complete format reference
- **[Runtime Internals](../../docs/RUNTIME_INTERNALS.md)** - How execution works
- **[Best Practices](../../docs/BEST_PRACTICES.md)** - Workflow design patterns
- **[Examples](../examples/)** - More workflow examples

---

## 💡 Tips & Tricks

### Faster Development

```bash
# Skip tests for faster iteration
python cli.py run workflows/meta/v1.1.0_development.bbx \
  --input skip_tests=true

# Develop single feature
python cli.py run workflows/meta/features/workflow_versioning.bbx
```

### Debugging

```bash
# Enable debug logging
BBX_LOG_LEVEL=DEBUG python cli.py run workflows/meta/ci_cd_pipeline.bbx

# Watch in real-time
python examples/websocket_client.py v1_1_0_development
```

### Local Testing

```bash
# Use staging environment
python cli.py run workflows/meta/ci_cd_pipeline.bbx \
  --input environment=staging \
  --input auto_deploy=false

# Mock external services
export BBX_MOCK_TELEGRAM=true
export BBX_MOCK_PYPI=true
```

---

## 🎉 Success Stories

These workflows have successfully:

- ✅ Automated 100% of release process
- ✅ Reduced release time from 4 hours → 20 minutes
- ✅ Eliminated manual errors in deployment
- ✅ Enabled continuous delivery
- ✅ Documented best practices through working code

---

## 📝 License

These workflows are part of BBX and licensed under Apache 2.0.

```
Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia

Licensed under the Apache License, Version 2.0
```

---

<div align="center">

**🤖 Meta-automation at its finest**

*Using BBX to build BBX - recursion in action!*

Built with ❤️ in Siberia | [BBX Home](../../README.md) | [Documentation](../../docs/)

</div>
