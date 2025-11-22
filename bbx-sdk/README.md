# BBX Python SDK

Official Python client library for [Blackbox Workflow Engine](https://github.com/kurokie1337/bbx) API.

## Features

- **Full API Coverage** - Complete workflow and execution management
- **Type Safety** - Pydantic models for all requests and responses
- **Authentication** - JWT token-based authentication with auto-refresh
- **Workflow Sync** - Sync local `.bbx` files with remote server
- **Error Handling** - Comprehensive exception handling
- **Logging** - Built-in logging support

## Installation

```bash
# From source
cd bbx-sdk
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from bbx_sdk import BlackboxClient

# Initialize client
client = BlackboxClient("http://localhost:8000")

# Authenticate
client.authenticate("username", "password")

# List workflows
workflows = client.list_workflows()
for workflow in workflows.items:
    print(f"{workflow.name}: {workflow.status}")

# Create a new workflow
from bbx_sdk import WorkflowCreate

new_workflow = WorkflowCreate(
    name="My Workflow",
    description="Automated deployment pipeline",
    bbx_yaml="""
workflow:
  id: my_workflow
  name: My Workflow
  version: "6.0"
  steps:
    - id: hello
      mcp: bbx.logger
      method: info
      inputs:
        message: "Hello from BBX SDK!"
"""
)

created = client.create_workflow(new_workflow)
print(f"Created workflow: {created.id}")

# Execute workflow
execution = client.execute_workflow(created.id)
print(f"Execution started: {execution.id}")

# Check execution status
status = client.get_execution_status(execution.id)
print(f"Status: {status.status}")
```

### Workflow Synchronization

Sync local `.bbx` files with the remote server:

```python
from bbx_sdk import BlackboxClient, WorkflowSyncer

client = BlackboxClient("http://localhost:8000")
client.authenticate()

syncer = WorkflowSyncer(client)

# Sync single file
syncer.sync_file_to_blackbox("./my_workflow.bbx")

# Sync entire directory
syncer.sync_directory("./workflows")

# Download workflow from server
syncer.download_workflow("workflow-id-123", "./downloaded")
```

## Configuration

Create `.env` file in your project root:

```env
BLACKBOX_API_URL=http://localhost:8000
BLACKBOX_USERNAME=your_username
BLACKBOX_PASSWORD=your_password
BLACKBOX_API_TIMEOUT=30
LOG_LEVEL=INFO
```

Or configure programmatically:

```python
from bbx_sdk import BlackboxClient

client = BlackboxClient(
    base_url="https://api.example.com",
    timeout=60
)
```

## API Reference

### BlackboxClient

#### Methods

- `authenticate(username, password)` - Authenticate and get access token
- `create_workflow(workflow)` - Create new workflow
- `get_workflow(workflow_id)` - Get workflow by ID
- `list_workflows(skip=0, limit=100)` - List workflows with pagination
- `update_workflow(workflow_id, update_data)` - Update existing workflow
- `delete_workflow(workflow_id)` - Delete workflow
- `execute_workflow(workflow_id, trigger_data={})` - Execute workflow
- `get_execution_status(execution_id)` - Get execution status
- `health_check()` - Check API health

### WorkflowSyncer

#### Methods

- `load_local_workflow(file_path)` - Load and parse local .bbx file
- `sync_file_to_blackbox(file_path, update_existing=True)` - Sync single file
- `sync_directory(directory)` - Sync all .bbx files in directory
- `download_workflow(workflow_id, output_dir)` - Download workflow to file

### Models

#### WorkflowCreate
```python
WorkflowCreate(
    name: str,
    description: Optional[str] = None,
    bbx_yaml: str,
    license_type: LicenseType = LicenseType.FREE,
    max_holders: int = 1000,
    signature: Optional[str] = None
)
```

#### WorkflowUpdate
```python
WorkflowUpdate(
    name: Optional[str] = None,
    description: Optional[str] = None,
    bbx_yaml: Optional[str] = None,
    status: Optional[WorkflowStatus] = None
)
```

#### WorkflowResponse
```python
WorkflowResponse(
    id: str,
    user_id: str,
    name: str,
    description: Optional[str],
    bbx_yaml: str,
    status: WorkflowStatus,
    version: int,
    created_at: datetime,
    updated_at: datetime
)
```

#### ExecutionCreate
```python
ExecutionCreate(
    workflow_id: UUID,
    trigger_data: Dict[str, Any] = {}
)
```

#### ExecutionResponse
```python
ExecutionResponse(
    id: str,
    workflow_id: str,
    status: str,
    started_at: datetime,
    current_step: Optional[str],
    outputs: Dict[str, Any]
)
```

## Error Handling

```python
import httpx
from bbx_sdk import BlackboxClient

client = BlackboxClient()

try:
    client.authenticate("user", "pass")
except httpx.HTTPStatusError as e:
    print(f"Authentication failed: {e.response.status_code}")
    print(f"Details: {e.response.text}")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Examples

See [examples/](examples/) directory for complete examples:

- `basic_usage.py` - Basic workflow management
- `sync_workflows.py` - Workflow synchronization
- `execute_and_monitor.py` - Workflow execution and monitoring

## Requirements

- Python 3.8+
- httpx
- pydantic
- pydantic-settings
- PyYAML

## License

Copyright 2025 Ilya Makarov, Krasnoyarsk, Siberia, Russia

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.

## Contributing

This SDK is part of the main BBX project. See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/kurokie1337/bbx/issues)
- **Documentation**: [Main Documentation](../docs/)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)

---

**Built with ❤️ in Siberia** | Part of [Blackbox Workflow Engine](https://github.com/kurokie1337/bbx)
