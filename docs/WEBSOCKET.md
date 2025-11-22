# WebSocket Real-time Updates

BBX supports real-time workflow execution updates via WebSocket.

## Features

- Live workflow execution status
- Step-by-step progress streaming
- Logs and metrics streaming
- Multi-client broadcasting
- Automatic reconnection

## WebSocket Endpoints

### Workflow Updates
\`\`\`
ws://localhost:8000/ws/workflows/{workflow_id}
\`\`\`

### Global System Updates
\`\`\`
ws://localhost:8000/ws/global
\`\`\`

## Event Types

- \`workflow_start\` - Workflow execution started
- \`workflow_complete\` - Workflow execution completed
- \`step_start\` - Step execution started
- \`step_complete\` - Step execution completed
- \`step_progress\` - Step progress update
- \`log\` - Log message
- \`metric\` - Metric update

## Python Client Example

\`\`\`python
import asyncio
import websockets
import json

async def watch_workflow(workflow_id):
    uri = f"ws://localhost:8000/ws/workflows/{workflow_id}"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"Event: {data['event']}")

asyncio.run(watch_workflow("my_workflow"))
\`\`\`

## JavaScript Client Example

\`\`\`javascript
const ws = new WebSocket('ws://localhost:8000/ws/workflows/my_workflow');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event:', data.event);
};
\`\`\`

## React Component

See \`dashboard/src/components/WorkflowLiveView.jsx\` for full React integration.
