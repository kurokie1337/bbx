// Blackbox Node.js SDK
// 
// Official Node.js client for Blackbox workflow engine

const axios = require('axios');

class BlackboxClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.client = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        this.token = null;
    }

    /**
     * Authenticate with the Blackbox API
     */
    async authenticate(username, password) {
        const response = await this.client.post('/auth/login', {
            username,
            password
        });

        this.token = response.data.access_token;
        this.client.defaults.headers['Authorization'] = `Bearer ${this.token}`;

        return response.data;
    }

    /**
     * Create a new workflow
     */
    async createWorkflow(workflow) {
        const response = await this.client.post('/api/workflows', {
            name: workflow.name,
            description: workflow.description,
            bbx_yaml: workflow.bbxYaml
        });

        return response.data;
    }

    /**
     * Execute a workflow
     */
    async executeWorkflow(workflowId, inputs = {}) {
        const response = await this.client.post(`/api/execute/${workflowId}`, {
            inputs
        });

        return response.data;
    }

    /**
     * Get workflow status
     */
    async getWorkflowStatus(executionId) {
        const response = await this.client.get(`/api/executions/${executionId}`);
        return response.data;
    }

    /**
     * List workflows
     */
    async listWorkflows(page = 1, size = 10) {
        const response = await this.client.get('/api/workflows', {
            params: { page, size }
        });

        return response.data;
    }

    /**
     * Delete workflow
     */
    async deleteWorkflow(workflowId) {
        await this.client.delete(`/api/workflows/${workflowId}`);
        return { success: true };
    }

    /**
     * Subscribe to workflow execution events (WebSocket)
     */
    subscribeToExecution(executionId, callbacks) {
        const ws = new WebSocket(`${this.baseURL.replace('http', 'ws')}/ws/execution/${executionId}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'STEP_START' && callbacks.onStepStart) {
                callbacks.onStepStart(data);
            } else if (data.type === 'STEP_END' && callbacks.onStepEnd) {
                callbacks.onStepEnd(data);
            } else if (data.type === 'STEP_ERROR' && callbacks.onStepError) {
                callbacks.onStepError(data);
            } else if (data.type === 'WORKFLOW_END' && callbacks.onWorkflowEnd) {
                callbacks.onWorkflowEnd(data);
            }
        };

        ws.onerror = (error) => {
            if (callbacks.onError) {
                callbacks.onError(error);
            }
        };

        return ws;
    }
}

module.exports = { BlackboxClient };

// Example usage:
/*
const { BlackboxClient } = require('@blackbox/client');

const client = new BlackboxClient('https://api.blackbox.dev');

async function main() {
  // Authenticate
  await client.authenticate('user', 'password');
  
  // Create workflow
  const workflow = await client.createWorkflow({
    name: 'My Workflow',
    description: 'Test workflow',
    bbxYaml: `
      id: test_workflow
      steps:
        fetch_data:
          use: http.get
          args:
            url: https://api.example.com/data
    `
  });
  
  // Execute workflow
  const execution = await client.executeWorkflow(workflow.id, {
    input_param: 'value'
  });
  
  // Monitor execution
  client.subscribeToExecution(execution.id, {
    onStepStart: (data) => console.log('Step started:', data.step_id),
    onStepEnd: (data) => console.log('Step completed:', data.step_id),
    onWorkflowEnd: (data) => console.log('Workflow complete:', data.results)
  });
}

main().catch(console.error);
*/
