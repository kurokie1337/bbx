# @bbx/sdk

Official BBX SDK for Node.js and TypeScript.

## Installation

\`\`\`bash
npm install @bbx/sdk
\`\`\`

## Usage

\`\`\`typescript
import { BBXClient, WorkflowBuilder } from '@bbx/sdk';

// Connect to BBX
const client = new BBXClient('http://localhost:8000');

// Execute workflow
const execution = await client.executeWorkflow('my_workflow.bbx', {
  param1: 'value1'
});

console.log(execution.status);

// Build workflow programmatically
const workflow = new WorkflowBuilder('my_workflow', 'My Workflow')
  .addStep({
    id: 'step1',
    mcp: 'bbx.http',
    method: 'get',
    inputs: { url: 'https://api.example.com' }
  })
  .build();
\`\`\`

## License

BSL-1.1 (converts to Apache 2.0 on 2028-11-05)
