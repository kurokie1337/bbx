import { create } from 'zustand'

export type AgentStatus = 'idle' | 'working' | 'active' | 'completed' | 'error' | 'queued'

export interface Agent {
  id: string
  name: string
  status: AgentStatus
  progress?: number
  currentTask?: string
  duration?: number
  queuePosition?: number
  error?: string
}

interface AgentsState {
  agents: Agent[]
  setAgents: (agents: Agent[]) => void
  updateAgent: (id: string, updates: Partial<Agent>) => void
  setAgentStatus: (id: string, status: AgentStatus, extra?: Partial<Agent>) => void
  resetAllAgents: () => void
}

const defaultAgents: Agent[] = [
  { id: 'architect', name: 'architect', status: 'idle' },
  { id: 'coder', name: 'coder', status: 'idle' },
  { id: 'reviewer', name: 'reviewer', status: 'idle' },
  { id: 'tester', name: 'tester', status: 'idle' },
]

export const useAgentsStore = create<AgentsState>((set) => ({
  agents: defaultAgents,

  setAgents: (agents) => set({ agents }),

  updateAgent: (id, updates) => set((state) => ({
    agents: state.agents.map(agent =>
      agent.id === id ? { ...agent, ...updates } : agent
    )
  })),

  setAgentStatus: (id, status, extra = {}) => set((state) => ({
    agents: state.agents.map(agent =>
      agent.id === id
        ? {
            ...agent,
            status,
            ...extra,
            // Clear fields based on status
            ...(status === 'idle' ? { progress: undefined, currentTask: undefined, error: undefined } : {}),
            ...(status === 'completed' ? { progress: 100 } : {}),
          }
        : agent
    )
  })),

  resetAllAgents: () => set({ agents: defaultAgents }),
}))
