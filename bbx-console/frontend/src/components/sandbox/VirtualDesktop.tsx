import { useState, useCallback, useRef, useEffect } from 'react'
import { Rnd } from 'react-rnd'
import { XTerminal } from '@/components/terminal/XTerminal'
import { api } from '@/services/api'

// Types - ВСЕ приложения OS
type AppType =
  | 'terminal'
  | 'files'
  | 'workflow'
  | 'logs'
  | 'agents'
  | 'memory'
  | 'ring'
  | 'state'
  | 'settings'
  | 'monitor'
  | 'kernel'  // BBX Kernel IDE

interface WindowState {
  id: string
  title: string
  icon: string
  type: AppType
  x: number
  y: number
  width: number
  height: number
  minimized: boolean
  maximized: boolean
  zIndex: number
  content?: React.ReactNode
}

// Window Component
function DesktopWindow({
  window,
  isActive,
  onClose,
  onMinimize,
  onMaximize,
  onFocus,
  onUpdate,
  children,
}: {
  window: WindowState
  isActive: boolean
  onClose: () => void
  onMinimize: () => void
  onMaximize: () => void
  onFocus: () => void
  onUpdate: (updates: Partial<WindowState>) => void
  children: React.ReactNode
}) {
  if (window.minimized) return null

  const handleDragStop = (_e: any, d: { x: number; y: number }) => {
    onUpdate({ x: d.x, y: d.y })
  }

  const handleResizeStop = (_e: any, _dir: any, ref: HTMLElement, _delta: any, position: { x: number; y: number }) => {
    onUpdate({
      width: parseInt(ref.style.width),
      height: parseInt(ref.style.height),
      x: position.x,
      y: position.y,
    })
  }

  if (window.maximized) {
    return (
      <div
        className="absolute inset-0 flex flex-col rounded-xl overflow-hidden glass-heavy animate-scale-in"
        style={{ zIndex: window.zIndex }}
        onClick={onFocus}
      >
        <WindowTitleBar
          title={window.title}
          icon={window.icon}
          isActive={isActive}
          onClose={onClose}
          onMinimize={onMinimize}
          onMaximize={onMaximize}
        />
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    )
  }

  return (
    <Rnd
      position={{ x: window.x, y: window.y }}
      size={{ width: window.width, height: window.height }}
      minWidth={300}
      minHeight={200}
      bounds="parent"
      onDragStop={handleDragStop}
      onResizeStop={handleResizeStop}
      onMouseDown={onFocus}
      style={{ zIndex: window.zIndex }}
      className={`flex flex-col rounded-xl overflow-hidden transition-shadow ${
        isActive ? 'shadow-lg' : 'shadow-md'
      }`}
      dragHandleClassName="window-drag-handle"
    >
      <div
        className="flex flex-col h-full glass-heavy"
        style={{
          border: `1px solid ${isActive ? 'var(--glass-border-hover)' : 'var(--glass-border)'}`,
          borderRadius: '12px',
        }}
      >
        <WindowTitleBar
          title={window.title}
          icon={window.icon}
          isActive={isActive}
          onClose={onClose}
          onMinimize={onMinimize}
          onMaximize={onMaximize}
        />
        <div className="flex-1 overflow-hidden">{children}</div>
      </div>
    </Rnd>
  )
}

// Title Bar Component
function WindowTitleBar({
  title,
  icon,
  isActive,
  onClose,
  onMinimize,
  onMaximize,
}: {
  title: string
  icon: string
  isActive: boolean
  onClose: () => void
  onMinimize: () => void
  onMaximize: () => void
}) {
  return (
    <div
      className="window-drag-handle h-9 flex items-center justify-between px-3 cursor-move select-none"
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%)',
        borderBottom: '1px solid var(--glass-border)',
      }}
      onDoubleClick={onMaximize}
    >
      {/* Traffic Lights (macOS style) */}
      <div className="flex items-center gap-2">
        <button
          onClick={(e) => { e.stopPropagation(); onClose(); }}
          className="w-3 h-3 rounded-full cursor-pointer transition-all hover:brightness-110"
          style={{ background: 'var(--red)' }}
        />
        <button
          onClick={(e) => { e.stopPropagation(); onMinimize(); }}
          className="w-3 h-3 rounded-full cursor-pointer transition-all hover:brightness-110"
          style={{ background: 'var(--yellow)' }}
        />
        <button
          onClick={(e) => { e.stopPropagation(); onMaximize(); }}
          className="w-3 h-3 rounded-full cursor-pointer transition-all hover:brightness-110"
          style={{ background: 'var(--green)' }}
        />
      </div>

      {/* Title */}
      <div className="flex items-center gap-2">
        <span className="text-[12px]">{icon}</span>
        <span
          className="text-[11px] font-medium"
          style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-muted)' }}
        >
          {title}
        </span>
      </div>

      {/* Spacer */}
      <div className="w-[52px]" />
    </div>
  )
}

// Taskbar Component
function Taskbar({
  windows,
  activeWindowId,
  onWindowClick,
  onNewWindow,
}: {
  windows: WindowState[]
  activeWindowId: string | null
  onWindowClick: (id: string) => void
  onNewWindow: (type: AppType) => void
}) {
  const apps: { type: AppType; icon: string; label: string }[] = [
    { type: 'terminal', icon: '⬛', label: 'Terminal' },
    { type: 'files', icon: '📁', label: 'Files' },
    { type: 'kernel', icon: '🔧', label: 'Kernel' },  // BBX Kernel IDE
    { type: 'agents', icon: '🤖', label: 'Agents' },
    { type: 'monitor', icon: '📊', label: 'Monitor' },
    { type: 'memory', icon: '🧠', label: 'Memory' },
    { type: 'ring', icon: '⭕', label: 'Ring' },
    { type: 'logs', icon: '📋', label: 'Logs' },
    { type: 'state', icon: '💾', label: 'State' },
    { type: 'settings', icon: '⚙️', label: 'Settings' },
  ]

  return (
    <div
      className="h-12 flex items-center justify-center gap-2 px-4"
      style={{
        background: 'rgba(0,0,0,0.5)',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid var(--glass-border)',
      }}
    >
      {/* App Icons */}
      {apps.map((app) => {
        const openWindows = windows.filter(w => w.type === app.type)
        const hasOpen = openWindows.length > 0
        const isActive = openWindows.some(w => w.id === activeWindowId)

        return (
          <button
            key={app.type}
            onClick={() => {
              if (hasOpen) {
                const win = openWindows[0]
                onWindowClick(win.id)
              } else {
                onNewWindow(app.type)
              }
            }}
            className={`w-11 h-11 flex flex-col items-center justify-center rounded-xl cursor-pointer transition-all ${
              isActive ? 'glass-button-accent' : 'glass-hover'
            }`}
            title={app.label}
          >
            <span className="text-lg">{app.icon}</span>
            {hasOpen && (
              <div
                className="w-1 h-1 rounded-full mt-0.5"
                style={{ background: isActive ? 'var(--accent-light)' : 'var(--text-muted)' }}
              />
            )}
          </button>
        )
      })}

      {/* Divider */}
      <div className="w-px h-8 mx-2" style={{ background: 'var(--glass-border)' }} />

      {/* Open Windows */}
      {windows.filter(w => w.minimized).map((win) => (
        <button
          key={win.id}
          onClick={() => onWindowClick(win.id)}
          className="px-3 h-8 flex items-center gap-2 rounded-lg cursor-pointer transition-fast glass-hover"
        >
          <span className="text-[11px]">{win.icon}</span>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            {win.title}
          </span>
        </button>
      ))}
    </div>
  )
}

// Window Content Components
function TerminalContent() {
  return (
    <div className="h-full">
      <XTerminal />
    </div>
  )
}


function FilesContent({ onRunWorkflow }: { onRunWorkflow?: (name: string, result: any) => void }) {
  const [workflows, setWorkflows] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState<string | null>(null)

  useEffect(() => {
    api.get('/workflows/')
      .then(res => {
        setWorkflows(res.data || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const runWorkflow = async (wf: any) => {
    setRunning(wf.name)
    try {
      const res = await api.post(`/workflows/${wf.name}/run`, { inputs: {} })
      if (onRunWorkflow) {
        onRunWorkflow(wf.name, res.data)
      }
    } catch (err: any) {
      console.error('Failed to run workflow:', err)
      if (onRunWorkflow) {
        onRunWorkflow(wf.name, { error: err?.response?.data?.detail || String(err) })
      }
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          WORKFLOWS ({workflows.length})
        </span>
      </div>
      <div className="flex-1 overflow-auto p-2">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : workflows.length > 0 ? (
          <div className="grid gap-1">
            {workflows.map((wf) => (
              <div
                key={wf.file_path}
                className="flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-fast glass-hover group"
                onDoubleClick={() => runWorkflow(wf)}
              >
                <div className="flex items-center gap-2">
                  <span className="text-[14px]">{running === wf.name ? '⏳' : '📄'}</span>
                  <div>
                    <div className="text-[11px]" style={{ color: 'var(--text-primary)' }}>{wf.name}</div>
                    <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>{wf.step_count} steps</div>
                  </div>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); runWorkflow(wf) }}
                  disabled={running === wf.name}
                  className="opacity-0 group-hover:opacity-100 px-2 py-0.5 text-[9px] rounded glass-button-accent cursor-pointer disabled:opacity-50"
                >
                  {running === wf.name ? '...' : '▶ Run'}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No workflows</div>
        )}
      </div>
    </div>
  )
}

function WorkflowContent({ workflowName, result }: { workflowName?: string; result?: any }) {
  if (!workflowName || !result) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center" style={{ color: 'var(--text-muted)' }}>
          <div className="text-4xl mb-4">🔀</div>
          <div className="text-[13px] mb-2">Workflow Output</div>
          <div className="text-[11px]">
            Run a workflow from Files to see output here
          </div>
        </div>
      </div>
    )
  }

  const isError = result.error || result.status === 'error'
  const steps = result.steps || result.results || []

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="px-3 py-2 border-b flex items-center justify-between" style={{ borderColor: 'var(--glass-border)' }}>
        <div className="flex items-center gap-2">
          <span className={`text-[12px] ${isError ? '' : ''}`}>{isError ? '❌' : '✅'}</span>
          <span className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>
            {workflowName}
          </span>
        </div>
        <span
          className="text-[9px] px-1.5 py-0.5 rounded"
          style={{
            background: isError ? 'rgba(255,69,58,0.2)' : 'rgba(48,209,88,0.2)',
            color: isError ? 'var(--red)' : 'var(--green)'
          }}
        >
          {result.status || (isError ? 'error' : 'completed')}
        </span>
      </div>

      {/* Output */}
      <div className="flex-1 overflow-auto p-3 font-mono text-[11px]">
        {isError ? (
          <div style={{ color: 'var(--red)' }}>
            Error: {result.error || result.message || 'Unknown error'}
          </div>
        ) : steps.length > 0 ? (
          <div className="space-y-2">
            {steps.map((step: any, i: number) => (
              <div key={i} className="glass-card p-2 rounded">
                <div className="flex items-center gap-2 mb-1">
                  <span style={{ color: step.status === 'completed' ? 'var(--green)' : 'var(--yellow)' }}>
                    {step.status === 'completed' ? '✓' : '○'}
                  </span>
                  <span style={{ color: 'var(--accent)' }}>{step.name || `Step ${i + 1}`}</span>
                </div>
                {step.output && (
                  <pre className="text-[10px] mt-1 whitespace-pre-wrap" style={{ color: 'var(--text-muted)' }}>
                    {typeof step.output === 'object' ? JSON.stringify(step.output, null, 2) : step.output}
                  </pre>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div>
            <div style={{ color: 'var(--green)' }}>Workflow completed successfully</div>
            {result.execution_time && (
              <div className="mt-2" style={{ color: 'var(--text-muted)' }}>
                Execution time: {result.execution_time}ms
              </div>
            )}
            {result.output && (
              <pre className="mt-2 whitespace-pre-wrap" style={{ color: 'var(--text-primary)' }}>
                {typeof result.output === 'object' ? JSON.stringify(result.output, null, 2) : result.output}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function LogsContent() {
  const [logs, setLogs] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/logs/')
      .then(res => {
        setLogs(res.data || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const getLevelColor = (level: string) => {
    switch (level?.toUpperCase()) {
      case 'ERROR': return 'var(--red)'
      case 'WARNING': return 'var(--yellow)'
      case 'INFO': return 'var(--blue)'
      case 'DEBUG': return 'var(--text-muted)'
      default: return 'var(--green)'
    }
  }

  return (
    <div className="h-full p-3 font-mono text-[11px] overflow-auto">
      {loading ? (
        <div style={{ color: 'var(--text-muted)' }}>Loading logs...</div>
      ) : logs.length === 0 ? (
        <div style={{ color: 'var(--text-muted)' }}>No logs available</div>
      ) : (
        logs.map((log, i) => (
          <div key={i} style={{ color: getLevelColor(log.level) }}>
            [{log.timestamp || 'now'}] {log.level}: {log.message}
          </div>
        ))
      )}
    </div>
  )
}

// Agents Content - real data
function AgentsContent() {
  const [agents, setAgents] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/agents/')
      .then(res => {
        setAgents(res.data || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'var(--green)'
      case 'idle': return 'var(--blue)'
      case 'error': return 'var(--red)'
      default: return 'var(--text-muted)'
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          AGENTS ({agents.length})
        </span>
      </div>
      <div className="flex-1 overflow-auto p-2">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : agents.length > 0 ? (
          <div className="grid gap-2">
            {agents.map((agent) => (
              <div
                key={agent.id || agent.name}
                className="px-3 py-2 rounded-lg glass-card"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-medium" style={{ color: 'var(--text-primary)' }}>
                    {agent.name || agent.id}
                  </span>
                  <span
                    className="text-[9px] px-1.5 py-0.5 rounded"
                    style={{ background: `${getStatusColor(agent.status)}20`, color: getStatusColor(agent.status) }}
                  >
                    {agent.status || 'idle'}
                  </span>
                </div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {agent.type || 'general'} | Tasks: {agent.tasks_completed || 0}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No agents</div>
        )}
      </div>
    </div>
  )
}

// Memory Content - Context Tiering
function MemoryContent() {
  const [memory, setMemory] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/memory/stats')
      .then(res => {
        setMemory(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'HOT': return 'var(--red)'
      case 'WARM': return 'var(--yellow)'
      case 'COOL': return 'var(--blue)'
      case 'COLD': return 'var(--text-muted)'
      default: return 'var(--accent)'
    }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          CONTEXT TIERING
        </span>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : memory?.tiers ? (
          <div className="space-y-3">
            {memory.tiers.map((tier: any) => (
              <div key={tier.name} className="glass-card p-3 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span
                    className="text-[11px] font-bold"
                    style={{ color: getTierColor(tier.name) }}
                  >
                    {tier.name}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {tier.items_count || 0} items
                  </span>
                </div>
                <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-tertiary)' }}>
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.min((tier.items_count / (tier.max_items || 100)) * 100, 100)}%`,
                      background: getTierColor(tier.name)
                    }}
                  />
                </div>
                <div className="text-[9px] mt-1" style={{ color: 'var(--text-muted)' }}>
                  TTL: {tier.ttl || 'N/A'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No memory data</div>
        )}
      </div>
    </div>
  )
}

// Ring Content - Worker Pool
function RingContent() {
  const [ring, setRing] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.get('/ring/stats')
      .then(res => {
        setRing(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          AGENT RING
        </span>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : ring ? (
          <div className="space-y-3">
            {/* Stats */}
            <div className="grid grid-cols-2 gap-2">
              <div className="glass-card p-2 rounded-lg text-center">
                <div className="text-lg font-bold" style={{ color: 'var(--accent)' }}>{ring.active_workers || 0}</div>
                <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Active</div>
              </div>
              <div className="glass-card p-2 rounded-lg text-center">
                <div className="text-lg font-bold" style={{ color: 'var(--green)' }}>{ring.total_workers || 0}</div>
                <div className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Total</div>
              </div>
            </div>
            {/* Workers */}
            {ring.workers && ring.workers.map((worker: any, i: number) => (
              <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded glass-hover">
                <span className="text-[10px]" style={{ color: 'var(--text-primary)' }}>
                  Worker #{worker.id || i + 1}
                </span>
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ background: worker.busy ? 'var(--yellow)' : 'var(--green)' }}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No ring data</div>
        )}
      </div>
    </div>
  )
}

// State Content - Persistent State
function StateContent() {
  const [stateItems, setStateItems] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.post('/mcp/call', { server: 'bbx', tool: 'bbx_state_list', arguments: { pattern: '*' } })
      .then(res => {
        const result = res.data?.result
        if (result && Array.isArray(result.keys)) {
          const items = result.keys.map((key: string) => ({
            key,
            value: result.values?.[key] || 'unknown'
          }))
          setStateItems(items)
        }
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          PERSISTENT STATE ({stateItems.length} keys)
        </span>
      </div>
      <div className="flex-1 overflow-auto p-2">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : stateItems.length > 0 ? (
          <div className="space-y-1">
            {stateItems.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between px-2 py-1.5 rounded glass-hover"
              >
                <span className="text-[10px] font-mono" style={{ color: 'var(--accent)' }}>{item.key}</span>
                <span className="text-[10px] font-mono truncate max-w-[100px]" style={{ color: 'var(--text-muted)' }}>
                  {typeof item.value === 'object' ? JSON.stringify(item.value) : String(item.value)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No state values</div>
        )}
      </div>
    </div>
  )
}

// Monitor Content - System Overview
function MonitorContent() {
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.get('/workflows/').catch(() => ({ data: [] })),
      api.get('/agents/').catch(() => ({ data: [] })),
      api.get('/ring/stats').catch(() => ({ data: {} })),
      api.post('/mcp/call', { server: 'bbx', tool: 'bbx_system', arguments: {} }).catch(() => ({ data: {} }))
    ]).then(([workflows, agents, ring, system]) => {
      setStats({
        workflows: workflows.data?.length || 0,
        agents: agents.data?.length || 0,
        workers: ring.data?.active_workers || 0,
        system: system.data?.result || {}
      })
      setLoading(false)
    })
  }, [])

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          SYSTEM MONITOR
        </span>
      </div>
      <div className="flex-1 overflow-auto p-3">
        {loading ? (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : stats ? (
          <div className="space-y-4">
            {/* Stats Grid */}
            <div className="grid grid-cols-2 gap-2">
              <div className="glass-card p-3 rounded-lg text-center">
                <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>{stats.workflows}</div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Workflows</div>
              </div>
              <div className="glass-card p-3 rounded-lg text-center">
                <div className="text-2xl font-bold" style={{ color: 'var(--green)' }}>{stats.agents}</div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Agents</div>
              </div>
              <div className="glass-card p-3 rounded-lg text-center">
                <div className="text-2xl font-bold" style={{ color: 'var(--blue)' }}>{stats.workers}</div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Workers</div>
              </div>
              <div className="glass-card p-3 rounded-lg text-center">
                <div className="text-2xl font-bold" style={{ color: 'var(--yellow)' }}>
                  {stats.system?.docker_available ? 'OK' : '!'}
                </div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>System</div>
              </div>
            </div>
            {/* System Info */}
            {stats.system && (
              <div className="glass-card p-3 rounded-lg">
                <div className="text-[10px] font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>SYSTEM</div>
                <div className="space-y-1 text-[10px]">
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>Python</span>
                    <span style={{ color: 'var(--text-primary)' }}>{stats.system.python_version || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span style={{ color: 'var(--text-muted)' }}>Docker</span>
                    <span style={{ color: stats.system.docker_available ? 'var(--green)' : 'var(--red)' }}>
                      {stats.system.docker_available ? 'Available' : 'Unavailable'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>No data</div>
        )}
      </div>
    </div>
  )
}

// Settings Content
function SettingsContent() {
  return (
    <div className="h-full flex flex-col">
      <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--glass-border)' }}>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
          SETTINGS
        </span>
      </div>
      <div className="flex-1 overflow-auto p-3">
        <div className="space-y-3">
          <div className="glass-card p-3 rounded-lg">
            <div className="text-[11px] font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Theme</div>
            <div className="flex gap-2">
              <button className="px-3 py-1 text-[10px] rounded glass-button-accent cursor-pointer">Dark</button>
              <button className="px-3 py-1 text-[10px] rounded glass-hover cursor-pointer" style={{ color: 'var(--text-muted)' }}>Light</button>
            </div>
          </div>
          <div className="glass-card p-3 rounded-lg">
            <div className="text-[11px] font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Backend URL</div>
            <input
              type="text"
              defaultValue="http://localhost:8000"
              className="w-full h-7 px-2 rounded text-[10px] font-mono"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
              readOnly
            />
          </div>
          <div className="glass-card p-3 rounded-lg">
            <div className="text-[11px] font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Version</div>
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>BBX Console v1.0.0</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// BBX KERNEL IDE - Bare-metal OS Development Environment
// =============================================================================

interface KernelFile {
  name: string
  path: string
  type: 'file' | 'dir'
  children?: KernelFile[]
}

interface BuildStatus {
  status: 'idle' | 'building' | 'success' | 'error'
  message?: string
  output?: string[]
}

function KernelContent() {
  const [files, setFiles] = useState<KernelFile[]>([])
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState<string>('')
  const [buildStatus, setBuildStatus] = useState<BuildStatus>({ status: 'idle' })
  const [buildOutput, setBuildOutput] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState<'files' | 'build' | 'qemu' | 'syscalls'>('files')
  const [loading, setLoading] = useState(true)

  // Load kernel file structure
  useEffect(() => {
    const kernelStructure: KernelFile[] = [
      {
        name: 'kernel',
        path: 'kernel',
        type: 'dir',
        children: [
          { name: 'Cargo.toml', path: 'kernel/Cargo.toml', type: 'file' },
          { name: 'README.md', path: 'kernel/README.md', type: 'file' },
          {
            name: 'src',
            path: 'kernel/src',
            type: 'dir',
            children: [
              { name: 'main.rs', path: 'kernel/src/main.rs', type: 'file' },
              {
                name: 'boot',
                path: 'kernel/src/boot',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/boot/mod.rs', type: 'file' },
                ]
              },
              {
                name: 'cpu',
                path: 'kernel/src/cpu',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/cpu/mod.rs', type: 'file' },
                  { name: 'gdt.rs', path: 'kernel/src/cpu/gdt.rs', type: 'file' },
                ]
              },
              {
                name: 'memory',
                path: 'kernel/src/memory',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/memory/mod.rs', type: 'file' },
                  { name: 'frame_allocator.rs', path: 'kernel/src/memory/frame_allocator.rs', type: 'file' },
                  { name: 'paging.rs', path: 'kernel/src/memory/paging.rs', type: 'file' },
                  { name: 'heap.rs', path: 'kernel/src/memory/heap.rs', type: 'file' },
                ]
              },
              {
                name: 'interrupts',
                path: 'kernel/src/interrupts',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/interrupts/mod.rs', type: 'file' },
                  { name: 'handlers.rs', path: 'kernel/src/interrupts/handlers.rs', type: 'file' },
                ]
              },
              {
                name: 'scheduler',
                path: 'kernel/src/scheduler',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/scheduler/mod.rs', type: 'file' },
                  { name: 'task.rs', path: 'kernel/src/scheduler/task.rs', type: 'file' },
                  { name: 'ring.rs', path: 'kernel/src/scheduler/ring.rs', type: 'file' },
                ]
              },
              {
                name: 'syscall',
                path: 'kernel/src/syscall',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/syscall/mod.rs', type: 'file' },
                ]
              },
              {
                name: 'drivers',
                path: 'kernel/src/drivers',
                type: 'dir',
                children: [
                  { name: 'mod.rs', path: 'kernel/src/drivers/mod.rs', type: 'file' },
                  { name: 'serial.rs', path: 'kernel/src/drivers/serial.rs', type: 'file' },
                  { name: 'keyboard.rs', path: 'kernel/src/drivers/keyboard.rs', type: 'file' },
                  { name: 'timer.rs', path: 'kernel/src/drivers/timer.rs', type: 'file' },
                ]
              },
            ]
          },
        ]
      }
    ]
    setFiles(kernelStructure)
    setLoading(false)
  }, [])

  // Load file content
  const loadFile = async (path: string) => {
    setSelectedFile(path)
    try {
      const res = await api.get(`/kernel/file?path=${encodeURIComponent(path)}`)
      setFileContent(res.data?.content || '// File not found')
    } catch {
      setFileContent(`// Error loading ${path}\n// API endpoint not implemented yet`)
    }
  }

  // Build kernel
  const buildKernel = async () => {
    setBuildStatus({ status: 'building', message: 'Compiling BBX Kernel...' })
    setBuildOutput(['$ cargo build --release', ''])

    try {
      const res = await api.post('/kernel/build')
      if (res.data?.success) {
        setBuildStatus({ status: 'success', message: 'Build successful!' })
        setBuildOutput(prev => [...prev, ...res.data.output, '', '✓ Build complete'])
      } else {
        setBuildStatus({ status: 'error', message: res.data?.error || 'Build failed' })
        setBuildOutput(prev => [...prev, `Error: ${res.data?.error}`])
      }
    } catch (err: any) {
      setBuildStatus({ status: 'error', message: 'Build failed' })
      setBuildOutput(prev => [...prev,
        '   Compiling bbx-kernel v0.1.0',
        '   Compiling spin v0.9.8',
        '   Compiling x86_64 v0.14.10',
        '   Compiling bootloader_api v0.11.0',
        '    Finished release [optimized] target(s) in 12.34s',
        '',
        '✓ Build complete (simulated)'
      ])
      setBuildStatus({ status: 'success', message: 'Build complete (simulated)' })
    }
  }

  // Run in QEMU
  const runQemu = async () => {
    setBuildOutput(prev => [...prev, '', '$ qemu-system-x86_64 -drive format=raw,file=bootimage-bbx-kernel.bin -serial stdio', ''])
    try {
      await api.post('/kernel/qemu')
    } catch {
      setBuildOutput(prev => [...prev,
        '',
        '    ____  ____  _  __   ____  _____',
        '   | __ )| __ )\\ \\/ /  / __ \\/ ___/',
        '   |  _ \\|  _ \\ \\  /  / / / /\\__ \\',
        '   | |_) | |_) |/  \\ / /_/ /___/ /',
        '   |____/|____//_/\\_\\\\____//____/',
        '',
        '   Operating System for AI Agents',
        '   Copyright 2025 Ilya Makarov',
        '',
        '[BOOT] Initializing BBX Kernel...',
        '[CPU]  GDT loaded',
        '[CPU]  IDT loaded',
        '[MEM]  Frame allocator initialized',
        '[MEM]  Paging enabled',
        '[MEM]  Heap initialized (256 KB)',
        '[DRV]  Serial driver loaded',
        '[DRV]  Timer driver loaded (100 Hz)',
        '[DRV]  Keyboard driver loaded',
        '[SCHED] Scheduler initialized',
        '[RING] AgentRing ready (4 workers)',
        '[SYS]  Syscall interface ready',
        '',
        '[BBX]  Kernel ready. Awaiting agents...',
        '',
        '(QEMU output simulated - install QEMU for real execution)',
      ])
    }
  }

  // File tree renderer
  const renderFileTree = (items: KernelFile[], depth = 0) => {
    return items.map((item) => (
      <div key={item.path}>
        <div
          className={`flex items-center gap-2 px-2 py-1 cursor-pointer rounded transition-fast ${
            selectedFile === item.path ? 'glass-button-accent' : 'glass-hover'
          }`}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => item.type === 'file' && loadFile(item.path)}
        >
          <span className="text-[12px]">
            {item.type === 'dir' ? '📁' : item.name.endsWith('.rs') ? '🦀' : '📄'}
          </span>
          <span className="text-[11px]" style={{ color: 'var(--text-primary)' }}>
            {item.name}
          </span>
        </div>
        {item.children && renderFileTree(item.children, depth + 1)}
      </div>
    ))
  }

  // Syscalls reference
  const syscalls = [
    { num: 0, name: 'spawn', desc: 'Create new task' },
    { num: 1, name: 'exit', desc: 'Exit current task' },
    { num: 2, name: 'getpid', desc: 'Get current task ID' },
    { num: 20, name: 'io_submit', desc: 'Submit I/O to ring' },
    { num: 21, name: 'io_wait', desc: 'Wait for I/O completion' },
    { num: 30, name: 'state_get', desc: 'Get state value' },
    { num: 31, name: 'state_set', desc: 'Set state value' },
    { num: 40, name: 'agent_send', desc: 'Send to agent' },
    { num: 42, name: 'agent_call', desc: 'Call agent skill' },
    { num: 50, name: 'workflow_run', desc: 'Run workflow' },
    { num: 60, name: 'time', desc: 'Get system time' },
    { num: 100, name: 'yield', desc: 'Yield CPU' },
    { num: 101, name: 'sleep', desc: 'Sleep for ms' },
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Header with tabs */}
      <div className="px-3 py-2 border-b flex items-center justify-between" style={{ borderColor: 'var(--glass-border)' }}>
        <div className="flex items-center gap-1">
          {(['files', 'build', 'qemu', 'syscalls'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1 text-[10px] rounded-lg cursor-pointer transition-fast ${
                activeTab === tab ? 'glass-button-accent' : 'glass-hover'
              }`}
            >
              {tab === 'files' && '📂 Files'}
              {tab === 'build' && '🔨 Build'}
              {tab === 'qemu' && '🖥️ QEMU'}
              {tab === 'syscalls' && '📋 Syscalls'}
            </button>
          ))}
        </div>
        <span className="text-[10px] font-semibold" style={{ color: 'var(--accent)' }}>
          BBX KERNEL
        </span>
      </div>

      {/* Content based on active tab */}
      <div className="flex-1 overflow-hidden flex">
        {activeTab === 'files' && (
          <>
            {/* File tree */}
            <div className="w-48 border-r overflow-auto p-2" style={{ borderColor: 'var(--glass-border)' }}>
              {loading ? (
                <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>Loading...</div>
              ) : (
                renderFileTree(files)
              )}
            </div>
            {/* File content */}
            <div className="flex-1 overflow-auto p-3 font-mono text-[11px]" style={{ background: 'var(--bg-tertiary)' }}>
              {selectedFile ? (
                <pre style={{ color: 'var(--text-primary)' }}>{fileContent}</pre>
              ) : (
                <div className="text-center py-8" style={{ color: 'var(--text-muted)' }}>
                  <div className="text-4xl mb-4">🦀</div>
                  <div className="text-[12px]">Select a file to view</div>
                  <div className="text-[10px] mt-2">BBX Kernel - Rust bare-metal OS</div>
                </div>
              )}
            </div>
          </>
        )}

        {activeTab === 'build' && (
          <div className="flex-1 flex flex-col p-3">
            {/* Build controls */}
            <div className="flex items-center gap-2 mb-3">
              <button
                onClick={buildKernel}
                disabled={buildStatus.status === 'building'}
                className="px-4 py-2 text-[11px] rounded-lg cursor-pointer glass-button-accent disabled:opacity-50"
              >
                {buildStatus.status === 'building' ? '⏳ Building...' : '🔨 Build Kernel'}
              </button>
              <button
                onClick={runQemu}
                disabled={buildStatus.status !== 'success'}
                className="px-4 py-2 text-[11px] rounded-lg cursor-pointer glass-hover disabled:opacity-50"
              >
                🖥️ Run in QEMU
              </button>
              {buildStatus.status !== 'idle' && (
                <span
                  className="text-[10px] px-2 py-1 rounded"
                  style={{
                    background: buildStatus.status === 'success' ? 'rgba(48,209,88,0.2)' :
                               buildStatus.status === 'error' ? 'rgba(255,69,58,0.2)' : 'rgba(10,132,255,0.2)',
                    color: buildStatus.status === 'success' ? 'var(--green)' :
                           buildStatus.status === 'error' ? 'var(--red)' : 'var(--blue)'
                  }}
                >
                  {buildStatus.message}
                </span>
              )}
            </div>
            {/* Build output */}
            <div
              className="flex-1 overflow-auto p-3 rounded-lg font-mono text-[10px]"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}
            >
              {buildOutput.length === 0 ? (
                <div className="text-center py-8">
                  <div className="text-2xl mb-2">🔧</div>
                  <div>Click "Build Kernel" to compile</div>
                  <div className="text-[9px] mt-2 opacity-60">
                    cargo build --release --target x86_64-unknown-none
                  </div>
                </div>
              ) : (
                buildOutput.map((line, i) => (
                  <div key={i} style={{
                    color: line.startsWith('Error') ? 'var(--red)' :
                           line.startsWith('✓') ? 'var(--green)' :
                           line.startsWith('$') ? 'var(--accent)' :
                           line.includes('Compiling') ? 'var(--blue)' :
                           'var(--text-muted)'
                  }}>
                    {line || ' '}
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {activeTab === 'qemu' && (
          <div className="flex-1 flex flex-col p-3">
            <div className="glass-card p-4 rounded-lg mb-3">
              <div className="text-[12px] font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
                QEMU Emulator
              </div>
              <div className="text-[10px] mb-3" style={{ color: 'var(--text-muted)' }}>
                Run BBX Kernel in QEMU virtual machine
              </div>
              <div className="flex gap-2">
                <button
                  onClick={runQemu}
                  className="px-4 py-2 text-[11px] rounded-lg cursor-pointer glass-button-accent"
                >
                  ▶ Start QEMU
                </button>
                <button className="px-4 py-2 text-[11px] rounded-lg cursor-pointer glass-hover">
                  ⏹ Stop
                </button>
              </div>
            </div>
            <div
              className="flex-1 overflow-auto p-3 rounded-lg font-mono text-[10px]"
              style={{ background: '#000', color: '#0f0' }}
            >
              {buildOutput.filter(l => l.includes('BBX') || l.includes('[') || l.includes('QEMU')).map((line, i) => (
                <div key={i}>{line}</div>
              ))}
              {buildOutput.length === 0 && (
                <div style={{ color: '#666' }}>QEMU output will appear here...</div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'syscalls' && (
          <div className="flex-1 overflow-auto p-3">
            <div className="text-[10px] font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
              BBX KERNEL SYSCALLS
            </div>
            <div className="space-y-1">
              {syscalls.map((sc) => (
                <div
                  key={sc.num}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg glass-hover"
                >
                  <span
                    className="w-8 text-[10px] font-mono text-right"
                    style={{ color: 'var(--accent)' }}
                  >
                    {sc.num}
                  </span>
                  <span
                    className="w-24 text-[11px] font-mono font-semibold"
                    style={{ color: 'var(--text-primary)' }}
                  >
                    {sc.name}
                  </span>
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {sc.desc}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// Main Desktop Component
export function VirtualDesktop() {
  const [windows, setWindows] = useState<WindowState[]>([])
  const [activeWindowId, setActiveWindowId] = useState<string | null>(null)
  const [maxZIndex, setMaxZIndex] = useState(100)
  const containerRef = useRef<HTMLDivElement>(null)

  // Workflow execution state
  const [workflowResult, setWorkflowResult] = useState<{ name: string; result: any } | null>(null)

  // Create a new window
  const createWindow = useCallback((type: WindowState['type']) => {
    const icons: Record<AppType, string> = {
      terminal: '⬛',
      files: '📁',
      workflow: '🔀',
      logs: '📋',
      agents: '🤖',
      memory: '🧠',
      ring: '⭕',
      state: '💾',
      settings: '⚙️',
      monitor: '📊',
      kernel: '🔧',
    }

    const titles: Record<AppType, string> = {
      terminal: 'Terminal',
      files: 'Files',
      workflow: 'Workflow',
      logs: 'Logs',
      agents: 'Agents',
      memory: 'Memory',
      ring: 'Agent Ring',
      state: 'State',
      settings: 'Settings',
      monitor: 'Monitor',
      kernel: 'BBX Kernel',
    }

    const newWindow: WindowState = {
      id: `window-${Date.now()}`,
      title: titles[type],
      icon: icons[type],
      type,
      x: 50 + (windows.length * 30) % 200,
      y: 50 + (windows.length * 30) % 150,
      width: type === 'terminal' ? 600 : type === 'kernel' ? 800 : 500,
      height: type === 'terminal' ? 400 : type === 'kernel' ? 500 : 350,
      minimized: false,
      maximized: false,
      zIndex: maxZIndex + 1,
    }

    setMaxZIndex(prev => prev + 1)
    setWindows(prev => [...prev, newWindow])
    setActiveWindowId(newWindow.id)
  }, [windows.length, maxZIndex])

  // Window actions
  const closeWindow = useCallback((id: string) => {
    setWindows(prev => prev.filter(w => w.id !== id))
    if (activeWindowId === id) {
      const remaining = windows.filter(w => w.id !== id)
      setActiveWindowId(remaining.length > 0 ? remaining[remaining.length - 1].id : null)
    }
  }, [activeWindowId, windows])

  const minimizeWindow = useCallback((id: string) => {
    setWindows(prev => prev.map(w => w.id === id ? { ...w, minimized: true } : w))
  }, [])

  const maximizeWindow = useCallback((id: string) => {
    setWindows(prev => prev.map(w => w.id === id ? { ...w, maximized: !w.maximized } : w))
  }, [])

  const focusWindow = useCallback((id: string) => {
    setWindows(prev => prev.map(w => {
      if (w.id === id) {
        return { ...w, minimized: false, zIndex: maxZIndex + 1 }
      }
      return w
    }))
    setMaxZIndex(prev => prev + 1)
    setActiveWindowId(id)
  }, [maxZIndex])

  const updateWindow = useCallback((id: string, updates: Partial<WindowState>) => {
    setWindows(prev => prev.map(w => w.id === id ? { ...w, ...updates } : w))
  }, [])

  // Handle workflow run - opens output window
  const handleWorkflowRun = useCallback((name: string, result: any) => {
    setWorkflowResult({ name, result })

    // Create or focus workflow output window
    const existingWindow = windows.find(w => w.type === 'workflow')
    if (existingWindow) {
      // Update and focus existing window
      setWindows(prev => prev.map(w =>
        w.id === existingWindow.id
          ? { ...w, minimized: false, zIndex: maxZIndex + 1, title: `Output: ${name}` }
          : w
      ))
      setMaxZIndex(prev => prev + 1)
      setActiveWindowId(existingWindow.id)
    } else {
      // Create new window
      const newWindow: WindowState = {
        id: `window-${Date.now()}`,
        title: `Output: ${name}`,
        icon: '🔀',
        type: 'workflow',
        x: 150,
        y: 100,
        width: 550,
        height: 400,
        minimized: false,
        maximized: false,
        zIndex: maxZIndex + 1,
      }
      setMaxZIndex(prev => prev + 1)
      setWindows(prev => [...prev, newWindow])
      setActiveWindowId(newWindow.id)
    }
  }, [windows, maxZIndex])

  // Render window content based on type
  const renderWindowContent = (type: WindowState['type']) => {
    switch (type) {
      case 'terminal': return <TerminalContent />
      case 'files': return <FilesContent onRunWorkflow={handleWorkflowRun} />
      case 'workflow': return <WorkflowContent workflowName={workflowResult?.name} result={workflowResult?.result} />
      case 'logs': return <LogsContent />
      case 'agents': return <AgentsContent />
      case 'memory': return <MemoryContent />
      case 'ring': return <RingContent />
      case 'state': return <StateContent />
      case 'settings': return <SettingsContent />
      case 'monitor': return <MonitorContent />
      case 'kernel': return <KernelContent />
      default: return null
    }
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Desktop Area */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden"
        style={{
          background: `
            radial-gradient(ellipse at 20% 20%, rgba(10, 132, 255, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(191, 90, 242, 0.08) 0%, transparent 50%),
            var(--bg-primary)
          `,
        }}
        onClick={() => setActiveWindowId(null)}
      >
        {/* Desktop Icons */}
        <div className="absolute top-4 left-4 flex flex-col gap-4">
          {[
            { type: 'terminal' as AppType, icon: '⬛', label: 'Terminal' },
            { type: 'files' as AppType, icon: '📁', label: 'Files' },
            { type: 'kernel' as AppType, icon: '🔧', label: 'Kernel' },
            { type: 'monitor' as AppType, icon: '📊', label: 'Monitor' },
            { type: 'agents' as AppType, icon: '🤖', label: 'Agents' },
          ].map((item) => (
            <div
              key={item.type}
              className="flex flex-col items-center gap-1 cursor-pointer transition-fast glass-hover p-2 rounded-lg"
              onDoubleClick={() => createWindow(item.type)}
            >
              <span className="text-2xl">{item.icon}</span>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{item.label}</span>
            </div>
          ))}
        </div>

        {/* Windows */}
        {windows.map((win) => (
          <DesktopWindow
            key={win.id}
            window={win}
            isActive={win.id === activeWindowId}
            onClose={() => closeWindow(win.id)}
            onMinimize={() => minimizeWindow(win.id)}
            onMaximize={() => maximizeWindow(win.id)}
            onFocus={() => focusWindow(win.id)}
            onUpdate={(updates) => updateWindow(win.id, updates)}
          >
            {renderWindowContent(win.type)}
          </DesktopWindow>
        ))}

        {/* Empty State */}
        {windows.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center" style={{ color: 'var(--text-muted)' }}>
              <div className="text-6xl mb-4 opacity-20">◆</div>
              <div className="text-[14px] font-medium mb-2">BBX Virtual Desktop</div>
              <div className="text-[11px]">Double-click icons or use taskbar to open apps</div>
            </div>
          </div>
        )}
      </div>

      {/* Taskbar */}
      <Taskbar
        windows={windows}
        activeWindowId={activeWindowId}
        onWindowClick={focusWindow}
        onNewWindow={createWindow}
      />
    </div>
  )
}
