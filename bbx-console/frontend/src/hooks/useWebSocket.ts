import { useEffect, useCallback } from 'react'
import { useAppStore } from '@/stores/appStore'

export function useWebSocket(channel?: string) {
  const {
    wsConnected,
    subscribe,
    unsubscribe,
    sendMessage,
  } = useAppStore()

  // Subscribe to channel on mount
  useEffect(() => {
    if (channel && wsConnected) {
      subscribe(channel)
      return () => unsubscribe(channel)
    }
  }, [channel, wsConnected, subscribe, unsubscribe])

  return {
    connected: wsConnected,
    subscribe,
    unsubscribe,
    sendMessage,
  }
}

export function useExecutionSubscription(executionId?: string) {
  const channel = executionId ? `execution:${executionId}` : undefined
  return useWebSocket(channel)
}

export function useWorkflowSubscription() {
  return useWebSocket('executions')
}
