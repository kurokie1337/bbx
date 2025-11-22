# 🏗️ Blackbox 2.0 Architecture: The "System-as-Code" Vision

This document outlines the architectural evolution of Blackbox to address Enterprise requirements (Persistence, Security, Scheduling, Scalability) using the **BBX Language** itself.

The core philosophy remains: **"Everything is a Workflow"**.

---

## 1. Persistence: The `storage` Adapter 💾

Instead of hiding the database, we expose it. Workflows can explicitly manage their state.

### Feature: Key-Value Store (Simple)
For storing last run times, simple flags, or cross-workflow state.

```yaml
steps:
  get_last_id:
    use: storage.kv.get
    args:
      key: "last_processed_id"
      default: 0
    outputs: last_id

  process_new_items:
    use: http.get
    args:
      url: "https://api.example.com/items?since=${get_last_id.output}"
    outputs: new_items

  save_new_cursor:
    use: storage.kv.set
    args:
      key: "last_processed_id"
      value: "${process_new_items.output.latest_id}"
    when: "${process_new_items.output.count} > 0"
```

### Feature: SQL Access (Advanced)
For complex data reporting or logging.

```yaml
steps:
  log_to_db:
    use: storage.sql.execute
    args:
      query: "INSERT INTO audit_log (user, action) VALUES (?, ?)"
      params: ["${inputs.user}", "login"]
```

---

## 2. Security: Secrets & Policies 🛡️

### Feature: Secrets Management
Hardcoded tokens are banned. We introduce a `secrets` namespace in variable resolution.

**Old Way (Insecure):**
```yaml
args:
  token: "sk-1234567890"
```

**New Way (Secure):**
```yaml
args:
  token: "${secrets.OPENAI_API_KEY}"
```
*The engine resolves `${secrets.*}` from a secure vault (Env vars, HashiCorp Vault, or encrypted local file).*

### Feature: Policy Workflows (Middleware)
Security rules are defined as workflows that run *before* the main workflow.

**`policies/no_shell.bbx`**:
```yaml
id: policy_no_shell
steps:
  scan_steps:
    use: logic.validate
    args:
      target: "${inputs.workflow_ast}"
      rule: "step.use != 'system.shell'"
      error_message: "Shell commands are forbidden!"
```

---

## 3. Scheduling: Declarative Triggers ⏰

We move scheduling from external tools (crontab) into the workflow definition.

### Feature: `triggers` Block
Every `.bbx` file can define when it should run.

```yaml
id: nightly_backup
name: Nightly Database Backup

# 🆕 New Section
triggers:
  - type: cron
    schedule: "0 3 * * *"  # Every day at 3:00 AM
    timezone: "UTC"
  - type: webhook
    path: "/webhooks/backup-now"

steps:
  ...
```

*The Engine's Scheduler component scans all loaded workflows and registers these triggers automatically.*

---

## 4. Scalability: The `queue` Adapter 🚀

To handle 1000+ concurrent jobs, we decouple **Trigger** from **Execution**.

### Feature: Async Execution
Instead of `flow.run` (synchronous), we use `queue.push`.

**Frontend Workflow (API Receiver):**
```yaml
id: handle_webhook
steps:
  enqueue_processing:
    use: queue.push
    args:
      queue: "image_processing"
      workflow: "process_image.bbx"
      inputs:
        image_url: "${inputs.url}"
    # Returns immediately, API responds 200 OK
```

**Worker Process:**
*   Listens on `image_processing` queue.
*   Picks up job.
*   Executes `process_image.bbx`.

### Feature: Distributed Workers
*   **API Node**: Accepts HTTP, pushes to Redis.
*   **Worker Node 1**: Consumes from Redis, runs `system.shell`.
*   **Worker Node 2**: Consumes from Redis, runs `browser.open`.

---

## Summary of New Adapters

| Adapter | Methods | Purpose |
| :--- | :--- | :--- |
| `storage` | `kv.get`, `kv.set`, `sql.execute` | Persistence & State |
| `secrets` | *(Variable Resolution)* | Secure Token Access |
| `queue` | `push`, `pop`, `status` | Async/Distributed Work |
| `scheduler` | *(System Component)* | Cron & Event Triggers |

This architecture transforms Blackbox from a simple script runner into a **Distributed Operating System for Workflows**.

---

## 📖 See Also

- **[BBX v6.0 Specification](BBX_SPEC_v6.md)** - Complete workflow format reference
- **[Runtime Internals](RUNTIME_INTERNALS.md)** - Detailed engine implementation
- **[MCP Development](MCP_DEVELOPMENT.md)** - Custom adapter development
- **[Universal Adapter Guide](UNIVERSAL_ADAPTER.md)** - Zero-code adapters
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment
- **[Documentation Index](INDEX.md)** - Complete documentation navigation

---

**Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia**
Licensed under the Apache License, Version 2.0
