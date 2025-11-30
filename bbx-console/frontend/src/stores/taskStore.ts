import { create } from 'zustand'

export type TaskStatus = 'idle' | 'running' | 'completed' | 'error'

export interface TaskHistoryItem {
  id: string
  task: string
  status: TaskStatus
  duration?: number
  startedAt: Date
  completedAt?: Date
  error?: string
}

interface TaskState {
  // Current task
  currentTask: string
  taskStatus: TaskStatus
  taskDuration: number
  taskError: string | null

  // History
  history: TaskHistoryItem[]
  commandHistory: string[]
  historyIndex: number

  // Actions
  setCurrentTask: (task: string) => void
  startTask: () => void
  completeTask: (duration: number) => void
  failTask: (error: string) => void
  stopTask: () => void
  clearTask: () => void

  // History navigation
  navigateHistoryUp: () => void
  navigateHistoryDown: () => void
  addToHistory: (item: TaskHistoryItem) => void
}

export const useTaskStore = create<TaskState>((set, get) => ({
  // Initial state
  currentTask: '',
  taskStatus: 'idle',
  taskDuration: 0,
  taskError: null,
  history: [],
  commandHistory: [],
  historyIndex: -1,

  // Actions
  setCurrentTask: (task) => set({ currentTask: task }),

  startTask: () => {
    const { currentTask, commandHistory } = get()
    if (!currentTask.trim()) return

    const newHistory = [currentTask, ...commandHistory.filter(c => c !== currentTask)].slice(0, 50)

    set({
      taskStatus: 'running',
      taskDuration: 0,
      taskError: null,
      commandHistory: newHistory,
      historyIndex: -1,
    })
  },

  completeTask: (duration) => {
    const { currentTask, history } = get()
    const historyItem: TaskHistoryItem = {
      id: Date.now().toString(),
      task: currentTask,
      status: 'completed',
      duration,
      startedAt: new Date(Date.now() - duration),
      completedAt: new Date(),
    }

    set({
      taskStatus: 'completed',
      taskDuration: duration,
      history: [historyItem, ...history].slice(0, 100),
    })
  },

  failTask: (error) => {
    const { currentTask, history } = get()
    const historyItem: TaskHistoryItem = {
      id: Date.now().toString(),
      task: currentTask,
      status: 'error',
      startedAt: new Date(),
      completedAt: new Date(),
      error,
    }

    set({
      taskStatus: 'error',
      taskError: error,
      history: [historyItem, ...history].slice(0, 100),
    })
  },

  stopTask: () => set({ taskStatus: 'idle', taskDuration: 0, taskError: null }),

  clearTask: () => set({ currentTask: '', taskStatus: 'idle', taskDuration: 0, taskError: null }),

  navigateHistoryUp: () => {
    const { commandHistory, historyIndex } = get()
    if (historyIndex < commandHistory.length - 1) {
      const newIndex = historyIndex + 1
      set({
        historyIndex: newIndex,
        currentTask: commandHistory[newIndex]
      })
    }
  },

  navigateHistoryDown: () => {
    const { commandHistory, historyIndex } = get()
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1
      set({
        historyIndex: newIndex,
        currentTask: commandHistory[newIndex]
      })
    } else if (historyIndex === 0) {
      set({ historyIndex: -1, currentTask: '' })
    }
  },

  addToHistory: (item) => set((state) => ({
    history: [item, ...state.history].slice(0, 100)
  })),
}))
