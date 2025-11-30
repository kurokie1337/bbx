// API Types

export interface Workflow {
  id: string
  name: string
  description?: string
  file_path: string
  step_count: number
  last_run?: {
    execution_id: string
    status: string
    started_at: string
    duration_ms?: number
  }
}

export interface WorkflowDetail extends Workflow {
  bbx_version: string
  inputs: WorkflowInput[]
  steps: WorkflowStep[]
  dag: DAGVisualization
}

export interface WorkflowInput {
  name: string
  type: string
  required: boolean
  default?: any
  description?: string
}

export interface WorkflowStep {
  id: string
  mcp: string
  method: string
  inputs: Record<string, any>
  depends_on: string[]
  timeout?: number
  retry?: number
  when?: string
}

export interface DAGVisualization {
  nodes: DAGNode[]
  edges: DAGEdge[]
  levels: string[][]
}

export interface DAGNode {
  id: string
  label: string
  level: number
  mcp: string
  method: string
  position?: { x: number; y: number }
}

export interface DAGEdge {
  source: string
  target: string
}

// Execution Types

export interface Execution {
  id: string
  workflow_id: string
  workflow_name?: string
  status: ExecutionStatus
  inputs: Record<string, any>
  results: Record<string, any>
  error?: string
  started_at?: string
  completed_at?: string
  duration_ms?: number
  steps: Record<string, StepState>
  current_level: number
  progress: number
}

export type ExecutionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface StepState {
  step_id: string
  status: StepStatus
  started_at?: string
  completed_at?: string
  duration_ms?: number
  output?: any
  error?: string
  retry_count: number
}

export type StepStatus = 'pending' | 'waiting' | 'running' | 'success' | 'failed' | 'skipped' | 'timeout' | 'cancelled'

// Agent Types

export interface Agent {
  id: string
  name: string
  description: string
  status: AgentStatus
  current_task?: string
  tools: string[]
  model: string
  metrics: AgentMetrics
}

export type AgentStatus = 'idle' | 'working' | 'queued' | 'error'

export interface AgentMetrics {
  tasks_completed: number
  tasks_failed: number
  avg_duration_ms: number
  success_rate: number
}

export interface AgentDetail extends Agent {
  system_prompt: string
  file_path: string
  recent_tasks: TaskHistory[]
}

export interface TaskHistory {
  id: string
  prompt: string
  status: string
  started_at: string
  duration_ms?: number
}

// Memory Types

export interface MemoryStats {
  generations: MemoryTier[]
  total_items: number
  total_size_bytes: number
  promotions: number
  demotions: number
  cache_hits: number
  cache_misses: number
  hit_rate: number
}

export interface MemoryTier {
  tier: 'HOT' | 'WARM' | 'COOL' | 'COLD'
  items: number
  size_bytes: number
  max_size_bytes: number
  utilization: number
}

export interface MemoryItem {
  key: string
  value_preview: string
  tier: string
  access_count: number
  is_pinned: boolean
  last_accessed: string
}

// Ring Types

export interface RingStats {
  operations_submitted: number
  operations_completed: number
  operations_failed: number
  operations_cancelled: number
  operations_timeout: number
  pending_count: number
  processing_count: number
  active_workers: number
  worker_pool_size: number
  submission_queue_size: number
  completion_queue_size: number
  throughput_ops_sec: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  worker_utilization: number
}

// Task Types

export interface Task {
  id: string
  title: string
  description?: string
  status: TaskStatus
  priority: TaskPriority
  parent_id?: string
  assigned_agent?: string
  execution_id?: string
  created_at: string
  completed_at?: string
  duration_ms?: number
  metadata: Record<string, any>
  subtasks: Task[]
}

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
export type TaskPriority = 'low' | 'medium' | 'high' | 'critical'

export interface TaskBoard {
  columns: TaskColumn[]
  total_tasks: number
}

export interface TaskColumn {
  status: string
  title: string
  tasks: Task[]
  count: number
}

export interface DecomposeResult {
  original_task: string
  subtasks: DecomposedSubtask[]
  suggested_workflow: string
  confidence: number
}

export interface DecomposedSubtask {
  title: string
  description: string
  assigned_agent: string
  depends_on: string[]
  priority: string
}

// WebSocket Types

export interface WSMessage {
  type: string
  channel?: string
  event?: string
  data?: any
  timestamp?: string
}
