import axios from 'axios'
import type {
  Workflow,
  WorkflowDetail,
  Execution,
  Agent,
  AgentDetail,
  MemoryStats,
  RingStats,
  Task,
  TaskBoard,
  DecomposeResult,
} from '@/types'

export const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Workflows

export async function getWorkflows(): Promise<Workflow[]> {
  const { data } = await api.get('/workflows/')
  return data
}

export async function getWorkflow(id: string, filePath?: string): Promise<WorkflowDetail> {
  const params = filePath ? { file_path: filePath } : {}
  const { data } = await api.get(`/workflows/${id}`, { params })
  return data
}

export async function validateWorkflow(content: string) {
  const { data } = await api.post('/workflows/validate', { content })
  return data
}

export async function runWorkflow(id: string, inputs: Record<string, any>, filePath?: string) {
  const params = filePath ? { file_path: filePath } : {}
  const { data } = await api.post(`/workflows/${id}/run`, { inputs }, { params })
  return data
}

// Executions

export async function getExecutions(params?: {
  workflow_id?: string
  status?: string
  limit?: number
  offset?: number
}): Promise<Execution[]> {
  const { data } = await api.get('/executions/', { params })
  return data
}

export async function getExecution(id: string): Promise<Execution> {
  const { data } = await api.get(`/executions/${id}`)
  return data
}

export async function cancelExecution(id: string) {
  const { data } = await api.post(`/executions/${id}/cancel`)
  return data
}

// Agents

export async function getAgents(): Promise<Agent[]> {
  const { data } = await api.get('/agents/')
  return data
}

export async function getAgent(id: string): Promise<AgentDetail> {
  const { data } = await api.get(`/agents/${id}`)
  return data
}

export async function getAgentStats() {
  const { data } = await api.get('/agents/stats')
  return data
}

// Memory

export async function getMemoryStats(): Promise<MemoryStats> {
  const { data } = await api.get('/memory/stats')
  return data
}

export async function getMemoryTier(tier: string) {
  const { data } = await api.get(`/memory/tiers/${tier}`)
  return data
}

export async function pinMemoryItem(key: string) {
  const { data } = await api.post(`/memory/items/${key}/pin`)
  return data
}

export async function unpinMemoryItem(key: string) {
  const { data } = await api.post(`/memory/items/${key}/unpin`)
  return data
}

// Ring

export async function getRingStats(): Promise<RingStats> {
  const { data } = await api.get('/ring/stats')
  return data
}

export async function getRingWorkers() {
  const { data } = await api.get('/ring/workers')
  return data
}

// Tasks

export async function getTasks(params?: {
  status?: string
  priority?: string
  parent_id?: string
  limit?: number
  offset?: number
}): Promise<Task[]> {
  const { data } = await api.get('/tasks/', { params })
  return data
}

export async function getTaskBoard(): Promise<TaskBoard> {
  const { data } = await api.get('/tasks/board')
  return data
}

export async function getTask(id: string): Promise<Task> {
  const { data } = await api.get(`/tasks/${id}`)
  return data
}

export async function createTask(task: {
  title: string
  description?: string
  priority?: string
  parent_id?: string
  assigned_agent?: string
}): Promise<Task> {
  const { data } = await api.post('/tasks/', task)
  return data
}

export async function updateTask(id: string, updates: Partial<Task>): Promise<Task> {
  const { data } = await api.patch(`/tasks/${id}`, updates)
  return data
}

export async function deleteTask(id: string) {
  const { data } = await api.delete(`/tasks/${id}`)
  return data
}

export async function decomposeTask(description: string, context?: string): Promise<DecomposeResult> {
  const { data } = await api.post('/tasks/decompose', { description, context })
  return data
}

// A2A

export async function discoverAgent(url: string) {
  const { data } = await api.post('/a2a/discover', { url })
  return data
}

export async function createA2ATask(agentUrl: string, skillId: string, input: Record<string, any>) {
  const { data } = await api.post('/a2a/tasks', { agent_url: agentUrl, skill_id: skillId, input })
  return data
}

// MCP

export async function getMCPServers() {
  const { data } = await api.get('/mcp/servers')
  return data
}

export async function getMCPTools(server: string) {
  const { data } = await api.get(`/mcp/servers/${server}/tools`)
  return data
}

export async function callMCPTool(server: string, tool: string, args: Record<string, any>) {
  const { data } = await api.post('/mcp/call', { server, tool, arguments: args })
  return data
}

// Health

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getWSStats() {
  const { data } = await api.get('/ws/stats')
  return data
}

// LLM Chat - Direct interaction with Ollama

export interface ChatRequest {
  prompt: string
  model?: string
}

export interface ChatResponse {
  success: boolean
  response?: string
  error?: string
  model: string
  tokens?: number
}

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat/', request)
  return data
}
