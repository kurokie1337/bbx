import { create } from 'zustand'

export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'paused'
export type ExecutionStatus = 'idle' | 'running' | 'completed' | 'failed' | 'paused'

export interface WorkflowStep {
  id: string
  name: string
  adapter: string
  status: StepStatus
  duration?: number
  output?: string
  error?: string
  progress?: number
  dependencies?: string[]
}

export interface WorkflowExecution {
  id: string
  workflowId: string
  workflowName: string
  workflowPath?: string
  status: ExecutionStatus
  steps: WorkflowStep[]
  startedAt: Date
  completedAt?: Date
  progress: number
  currentStep?: string
  logs: ExecutionLog[]
}

export interface ExecutionLog {
  id: string
  timestamp: Date
  level: 'info' | 'warn' | 'error' | 'debug'
  step?: string
  message: string
}

interface WorkflowState {
  executions: WorkflowExecution[]
  activeExecutionId: string | null

  // Actions
  startExecution: (workflowId: string, workflowName: string, steps: Omit<WorkflowStep, 'status' | 'duration'>[]) => string
  updateExecution: (id: string, updates: Partial<WorkflowExecution>) => void
  updateStep: (executionId: string, stepId: string, updates: Partial<WorkflowStep>) => void
  addLog: (executionId: string, log: Omit<ExecutionLog, 'id' | 'timestamp'>) => void
  setActiveExecution: (id: string | null) => void
  pauseExecution: (id: string) => void
  resumeExecution: (id: string) => void
  stopExecution: (id: string) => void
  clearExecutions: () => void
}

export const useWorkflowStore = create<WorkflowState>((set) => ({
  executions: [],
  activeExecutionId: null,

  startExecution: (workflowId, workflowName, steps) => {
    const id = `exec-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
    const execution: WorkflowExecution = {
      id,
      workflowId,
      workflowName,
      status: 'running',
      steps: steps.map(s => ({ ...s, status: 'pending' as StepStatus })),
      startedAt: new Date(),
      progress: 0,
      logs: [{
        id: `log-${Date.now()}`,
        timestamp: new Date(),
        level: 'info',
        message: `Starting workflow: ${workflowName}`
      }]
    }

    set(state => ({
      executions: [execution, ...state.executions],
      activeExecutionId: id
    }))

    return id
  },

  updateExecution: (id, updates) => set(state => ({
    executions: state.executions.map(exec =>
      exec.id === id ? { ...exec, ...updates } : exec
    )
  })),

  updateStep: (executionId, stepId, updates) => set(state => ({
    executions: state.executions.map(exec => {
      if (exec.id !== executionId) return exec

      const newSteps = exec.steps.map(step =>
        step.id === stepId ? { ...step, ...updates } : step
      )

      // Calculate progress
      const completedSteps = newSteps.filter(s =>
        s.status === 'completed' || s.status === 'skipped'
      ).length
      const progress = (completedSteps / newSteps.length) * 100

      return { ...exec, steps: newSteps, progress }
    })
  })),

  addLog: (executionId, log) => set(state => ({
    executions: state.executions.map(exec =>
      exec.id === executionId
        ? {
            ...exec,
            logs: [...exec.logs, {
              ...log,
              id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`,
              timestamp: new Date()
            }]
          }
        : exec
    )
  })),

  setActiveExecution: (id) => set({ activeExecutionId: id }),

  pauseExecution: (id) => set(state => ({
    executions: state.executions.map(exec =>
      exec.id === id && exec.status === 'running'
        ? { ...exec, status: 'paused' as ExecutionStatus }
        : exec
    )
  })),

  resumeExecution: (id) => set(state => ({
    executions: state.executions.map(exec =>
      exec.id === id && exec.status === 'paused'
        ? { ...exec, status: 'running' as ExecutionStatus }
        : exec
    )
  })),

  stopExecution: (id) => set(state => ({
    executions: state.executions.map(exec =>
      exec.id === id && (exec.status === 'running' || exec.status === 'paused')
        ? {
            ...exec,
            status: 'failed' as ExecutionStatus,
            completedAt: new Date(),
            logs: [...exec.logs, {
              id: `log-${Date.now()}`,
              timestamp: new Date(),
              level: 'warn' as const,
              message: 'Execution stopped by user'
            }]
          }
        : exec
    )
  })),

  clearExecutions: () => set({ executions: [], activeExecutionId: null }),
}))
