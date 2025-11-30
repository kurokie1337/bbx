import { useEffect } from 'react'
import { useUIStore } from '@/stores/uiStore'
import { useOutputStore } from '@/stores/outputStore'
import { useTaskStore } from '@/stores/taskStore'

export function useKeyboardShortcuts() {
  const { toggleSidePanel, openPopup, closePopup, activePopup, sidePanelOpen } = useUIStore()
  const { clearOutput } = useOutputStore()
  const { taskStatus, stopTask } = useTaskStore()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMeta = e.metaKey || e.ctrlKey

      // Don't intercept when typing in inputs (except for specific shortcuts)
      const isInput = ['INPUT', 'TEXTAREA'].includes((e.target as Element).tagName)

      // Toggle side panel (Explorer) - Cmd+K
      if (isMeta && e.key === 'k') {
        e.preventDefault()
        toggleSidePanel()
        return
      }

      // Close popup - Escape (always works)
      if (e.key === 'Escape') {
        if (activePopup) {
          e.preventDefault()
          closePopup()
          return
        }
        // If running, stop task
        if (taskStatus === 'running') {
          e.preventDefault()
          stopTask()
          return
        }
      }

      // Skip other shortcuts when in input (unless meta key)
      if (isInput && !isMeta) return

      // Stop task - Cmd+.
      if (isMeta && e.key === '.') {
        e.preventDefault()
        if (taskStatus === 'running') {
          stopTask()
        }
        return
      }

      // Clear output - Cmd+L
      if (isMeta && e.key === 'l') {
        e.preventDefault()
        clearOutput()
        return
      }

      // Memory popup - Cmd+M
      if (isMeta && e.key === 'm') {
        e.preventDefault()
        openPopup('memory')
        return
      }

      // Agents popup - Cmd+A (override select all)
      if (isMeta && e.key === 'a' && !isInput) {
        e.preventDefault()
        openPopup('agents')
        return
      }

      // Ring popup - Cmd+R (override refresh)
      if (isMeta && e.key === 'r') {
        e.preventDefault()
        openPopup('ring')
        return
      }

      // History popup - Cmd+H
      if (isMeta && e.key === 'h') {
        e.preventDefault()
        openPopup('history')
        return
      }

      // Settings popup - Cmd+,
      if (isMeta && e.key === ',') {
        e.preventDefault()
        openPopup('settings')
        return
      }

      // Close popup - Cmd+W
      if (isMeta && e.key === 'w') {
        if (activePopup) {
          e.preventDefault()
          closePopup()
          return
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    toggleSidePanel,
    openPopup,
    closePopup,
    clearOutput,
    stopTask,
    taskStatus,
    activePopup,
    sidePanelOpen
  ])
}
