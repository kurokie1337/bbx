# Remote Workflow Execution

Execute BBX workflows on remote instances.

## Usage

\`\`\`yaml
- id: remote_deploy
  mcp: bbx.remote
  method: exec
  inputs:
    remote_url: https://bbx-remote.example.com
    api_key: {}
    workflow_path: deploy.bbx
    inputs:
      environment: production
\`\`\`

## Python Client

\`\`\`python
from blackbox.remote.client import RemoteExecutor

client = RemoteExecutor("https://bbx.example.com", api_key="xxx")
result = client.execute_workflow("my_workflow.bbx", {"param": "value"})
\`\`\`
