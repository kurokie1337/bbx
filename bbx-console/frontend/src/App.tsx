import { useEffect } from 'react'

// Layout
import { Header } from '@/components/layout/Header'
import { StatusBar } from '@/components/layout/StatusBar'

// Core components
import { CommandInput } from '@/components/core/CommandInput'
import { AgentsPanel } from '@/components/core/AgentsPanel'
import { LiveOutput } from '@/components/core/LiveOutput'
import { LeftPanel } from '@/components/core/LeftPanel'
import { SidePanel } from '@/components/core/SidePanel'

// Sandbox components
import { VirtualDesktop } from '@/components/sandbox/VirtualDesktop'

// Popups
import { MemoryPopup } from '@/components/popups/MemoryPopup'
import { RingPopup } from '@/components/popups/RingPopup'
import { HistoryPopup } from '@/components/popups/HistoryPopup'
import { AgentsPopup } from '@/components/popups/AgentsPopup'
import { SettingsPopup } from '@/components/popups/SettingsPopup'
import { LogsPopup } from '@/components/popups/LogsPopup'
import { StatePopup } from '@/components/popups/StatePopup'

// Stores and hooks
import { useUIStore, type ViewMode } from '@/stores/uiStore'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'

function App() {
  const { activePopup, viewMode, setViewMode, setConnectionStatus } = useUIStore()

  // Initialize keyboard shortcuts
  useKeyboardShortcuts()

  // WebSocket connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    // Use same host as frontend (Vite proxy will forward to backend)
    const wsUrl = `${protocol}//${window.location.host}/ws`

    let ws: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout

    const connect = () => {
      setConnectionStatus('connecting')

      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setConnectionStatus('connected')
        console.log('WebSocket connected')
      }

      ws.onclose = () => {
        setConnectionStatus('disconnected')
        console.log('WebSocket disconnected, reconnecting in 3s...')

        // Reconnect after 3 seconds
        reconnectTimeout = setTimeout(connect, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleWSMessage(message)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
    }

    connect()

    return () => {
      if (ws) ws.close()
      clearTimeout(reconnectTimeout)
    }
  }, [setConnectionStatus])

  const handleWSMessage = (message: any) => {
    // Handle different message types
    switch (message.type) {
      case 'connected':
        console.log('Connection ID:', message.data?.connectionId)
        break
      case 'event':
        console.log('Event:', message.event, message.data)
        break
      default:
        break
    }
  }

  // Render active popup
  const renderPopup = () => {
    switch (activePopup) {
      case 'memory': return <MemoryPopup />
      case 'ring': return <RingPopup />
      case 'history': return <HistoryPopup />
      case 'agents': return <AgentsPopup />
      case 'settings': return <SettingsPopup />
      case 'logs': return <LogsPopup />
      case 'state': return <StatePopup />
      default: return null
    }
  }

  // Mode selector component
  const renderModeSelector = () => (
    <div className="flex items-center gap-1 px-2">
      {(['console', 'sandbox', 'desktop'] as ViewMode[]).map((mode) => (
        <button
          key={mode}
          onClick={() => setViewMode(mode)}
          className={`px-3 py-1 text-[10px] rounded-lg cursor-pointer transition-fast ${
            viewMode === mode ? 'glass-button-accent' : 'glass-hover'
          }`}
          style={{
            color: viewMode === mode ? 'var(--accent-light)' : 'var(--text-muted)',
          }}
        >
          {mode === 'console' && '⌨️'}
          {mode === 'sandbox' && '🔀'}
          {mode === 'desktop' && '🖥️'}
          <span className="ml-1.5 capitalize">{mode}</span>
        </button>
      ))}
    </div>
  )

  // Render content based on view mode
  const renderContent = () => {
    switch (viewMode) {
      case 'desktop':
        return <VirtualDesktop />

      case 'sandbox':
        // BBX OS - симуляция рабочей среды
        return (
          <div className="flex-1 flex overflow-hidden">
            <LeftPanel />
            <main className="flex-1 flex flex-col overflow-hidden">
              <VirtualDesktop />
            </main>
            <SidePanel />
          </div>
        )

      case 'console':
      default:
        return (
          <div className="flex-1 flex overflow-hidden">
            <LeftPanel />
            <main className="flex-1 flex flex-col overflow-hidden px-8 py-6">
              <CommandInput />
              <AgentsPanel />
              <LiveOutput />
            </main>
            <SidePanel />
          </div>
        )
    }
  }

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden"
      style={{ background: 'var(--bg-primary)' }}
    >
      {/* Header with Mode Selector */}
      <div className="flex items-center justify-between h-[52px] glass border-b border-glass">
        <Header />
        {renderModeSelector()}
      </div>

      {/* Content based on view mode */}
      {renderContent()}

      {/* Status Bar - only in console/sandbox mode */}
      {viewMode !== 'desktop' && <StatusBar />}

      {/* Active Popup (modal) */}
      {renderPopup()}
    </div>
  )
}

export default App
