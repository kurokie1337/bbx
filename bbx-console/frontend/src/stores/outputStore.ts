import { create } from 'zustand'

export type LogLevel = 'info' | 'success' | 'error' | 'warning' | 'system'
export type LogType = 'message' | 'file_add' | 'file_mod' | 'file_del' | 'transition' | 'code'

export interface OutputLine {
  id: string
  timestamp: Date
  agent?: string
  message: string
  level: LogLevel
  type: LogType
  code?: string
}

interface OutputState {
  lines: OutputLine[]
  isRunning: boolean

  // Actions
  addLine: (line: Omit<OutputLine, 'id' | 'timestamp'>) => void
  addTransition: (from: string, to: string) => void
  addCodeBlock: (agent: string, message: string, code: string) => void
  setRunning: (running: boolean) => void
  clearOutput: () => void
}

export const useOutputStore = create<OutputState>((set) => ({
  lines: [],
  isRunning: false,

  addLine: (line) => set((state) => ({
    lines: [...state.lines, {
      ...line,
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
    }]
  })),

  addTransition: (from, to) => set((state) => ({
    lines: [...state.lines, {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
      message: `${from} → ${to}`,
      level: 'system',
      type: 'transition',
    }]
  })),

  addCodeBlock: (agent, message, code) => set((state) => ({
    lines: [...state.lines, {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
      agent,
      message,
      level: 'info',
      type: 'code',
      code,
    }]
  })),

  setRunning: (running) => set({ isRunning: running }),

  clearOutput: () => set({ lines: [] }),
}))
