import { create } from 'zustand'
import type { WSMessage } from '@/types'

interface AppState {
  // WebSocket
  ws: WebSocket | null
  wsConnected: boolean
  wsConnectionId: string | null

  // Subscriptions
  subscriptions: Set<string>

  // UI State
  sidebarCollapsed: boolean
  selectedWorkflow: string | null
  selectedExecution: string | null

  // Actions
  connectWebSocket: () => void
  disconnectWebSocket: () => void
  subscribe: (channel: string) => void
  unsubscribe: (channel: string) => void
  sendMessage: (message: WSMessage) => void

  setSidebarCollapsed: (collapsed: boolean) => void
  setSelectedWorkflow: (id: string | null) => void
  setSelectedExecution: (id: string | null) => void
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  ws: null,
  wsConnected: false,
  wsConnectionId: null,
  subscriptions: new Set(),
  sidebarCollapsed: false,
  selectedWorkflow: null,
  selectedExecution: null,

  // WebSocket actions
  connectWebSocket: () => {
    const { ws } = get()
    if (ws?.readyState === WebSocket.OPEN) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    const socket = new WebSocket(wsUrl)

    socket.onopen = () => {
      set({ wsConnected: true })
      console.log('WebSocket connected')
    }

    socket.onclose = () => {
      set({ wsConnected: false, wsConnectionId: null })
      console.log('WebSocket disconnected')

      // Reconnect after 3 seconds
      setTimeout(() => {
        get().connectWebSocket()
      }, 3000)
    }

    socket.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    socket.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)
        handleMessage(message, set)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    set({ ws: socket })
  },

  disconnectWebSocket: () => {
    const { ws } = get()
    if (ws) {
      ws.close()
      set({ ws: null, wsConnected: false })
    }
  },

  subscribe: (channel: string) => {
    const { ws, subscriptions } = get()
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(JSON.stringify({ type: 'subscribe', channel }))
    subscriptions.add(channel)
    set({ subscriptions: new Set(subscriptions) })
  },

  unsubscribe: (channel: string) => {
    const { ws, subscriptions } = get()
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(JSON.stringify({ type: 'unsubscribe', channel }))
    subscriptions.delete(channel)
    set({ subscriptions: new Set(subscriptions) })
  },

  sendMessage: (message: WSMessage) => {
    const { ws } = get()
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    ws.send(JSON.stringify(message))
  },

  // UI actions
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setSelectedWorkflow: (id) => set({ selectedWorkflow: id }),
  setSelectedExecution: (id) => set({ selectedExecution: id }),
}))

function handleMessage(message: WSMessage, set: any) {
  switch (message.type) {
    case 'connected':
      set({ wsConnectionId: message.data?.connectionId })
      break

    case 'subscribed':
      console.log(`Subscribed to ${message.channel}`)
      break

    case 'unsubscribed':
      console.log(`Unsubscribed from ${message.channel}`)
      break

    case 'event':
      // Handle different event types
      // These would trigger query invalidation or state updates
      console.log(`Event: ${message.event}`, message.data)
      break

    case 'pong':
      // Heartbeat response
      break

    default:
      console.log('Unknown message type:', message.type)
  }
}
