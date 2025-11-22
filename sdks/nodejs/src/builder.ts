/**
 * Type-safe workflow builder
 */
export interface Step {
  id: string;
  mcp: string;
  method: string;
  inputs?: Record<string, any>;
  depends_on?: string[];
}

export class WorkflowBuilder {
  private workflow: any = {
    workflow: {
      id: '',
      name: '',
      version: '6.0',
      steps: []
    }
  };

  constructor(id: string, name: string) {
    this.workflow.workflow.id = id;
    this.workflow.workflow.name = name;
  }

  addStep(step: Step): this {
    this.workflow.workflow.steps.push(step);
    return this;
  }

  build(): any {
    return this.workflow;
  }

  toYAML(): string {
    // Simple YAML generation
    return JSON.stringify(this.workflow, null, 2);
  }
}
