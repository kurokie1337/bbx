/**
 * BBX SDK Client
 */
import axios, { AxiosInstance } from 'axios';

export interface WorkflowInputs {
  [key: string]: any;
}

export interface WorkflowExecution {
  id: string;
  workflow_id: string;
  status: 'running' | 'completed' | 'failed';
  outputs?: any;
}

export class BBXClient {
  private client: AxiosInstance;

  constructor(baseURL: string, apiKey?: string) {
    this.client = axios.create({
      baseURL,
      headers: apiKey ? { 'Authorization': `Bearer {}` } : {}
    });
  }

  async executeWorkflow(
    workflowPath: string,
    inputs?: WorkflowInputs
  ): Promise<WorkflowExecution> {
    const response = await this.client.post('/api/workflows/execute', {
      workflow_path: workflowPath,
      inputs: inputs || {}
    });
    return response.data;
  }

  async getExecution(executionId: string): Promise<WorkflowExecution> {
    const response = await this.client.get(`/api/executions/{}`);
    return response.data;
  }

  async listWorkflows(): Promise<any[]> {
    const response = await this.client.get('/api/workflows');
    return response.data.workflows;
  }
}
