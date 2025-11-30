import { useState, useEffect } from 'react'
import { useUIStore } from '@/stores/uiStore'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface MCPServer {
  name: string
  status: 'connected' | 'disconnected' | 'error'
  tools_count?: number
}

interface Adapter {
  name: string
  type: string
  status: 'active' | 'inactive'
}

interface Workspace {
  name: string
  path: string
  active: boolean
}

interface Process {
  id: string
  workflow: string
  status: 'running' | 'completed' | 'failed'
  started: string
}

export function LeftPanel() {
  const { leftPanelOpen, toggleLeftPanel, openPopup } = useUIStore()

  const [mcpServers, setMcpServers] = useState<MCPServer[]>([])
  const [adapters, setAdapters] = useState<Adapter[]>([])
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [processes, setProcesses] = useState<Process[]>([])
  const [expandedSections, setExpandedSections] = useState({
    processes: true,
    mcp: true,
    adapters: false,
    workspaces: false
  })

  // Fetch data
  useEffect(() => {
    // MCP servers - API returns array of {name, status, tools_count}
    api.get('/mcp/servers').then((res: AxiosResponse) => {
      const servers = (res.data || []).map((server: any) => ({
        name: server.name,
        status: server.status || 'disconnected',
        tools_count: server.tools_count || 0
      }))
      setMcpServers(servers)
    }).catch((err) => {
      console.error('Failed to fetch MCP servers:', err)
      setMcpServers([])
    })

    // Adapters - use MCP call to get adapters list
    api.post('/mcp/call', { server: 'bbx', tool: 'bbx_adapters', arguments: {} })
      .then((res: AxiosResponse) => {
        // Parse adapter info from result
        const result = res.data?.result || ''
        const lines = result.split('\n').filter((l: string) => l.includes(':'))
        const adapterList: Adapter[] = []
        let currentType = 'core'
        for (const line of lines) {
          if (line.includes('Core:')) currentType = 'core'
          else if (line.includes('Optional:')) currentType = 'opt'
          else if (line.includes('MCP:')) currentType = 'mcp'
          else if (line.trim().startsWith('-')) {
            const name = line.trim().replace('-', '').trim().split(' ')[0]
            if (name) adapterList.push({ name, type: currentType, status: 'active' })
          }
        }
        if (adapterList.length > 0) setAdapters(adapterList.slice(0, 10))
      })
      .catch((err) => {
        console.error('Failed to fetch adapters:', err)
        setAdapters([])
      })

    // Workspaces - fetch from real API
    api.get('/workspaces/').then((res: AxiosResponse) => {
      const data = res.data || []
      const wsList = Array.isArray(data) ? data : []
      setWorkspaces(wsList.map((ws: any) => ({
        name: ws.name || 'unknown',
        path: ws.path || '',
        active: ws.active || false
      })))
    }).catch((err) => {
      console.error('Failed to fetch workspaces:', err)
      setWorkspaces([])
    })

    // Processes - poll for running executions (API returns array)
    const fetchProcesses = () => {
      api.get('/executions/').then((res: AxiosResponse) => {
        const executions = res.data || []
        const running = executions.map((exec: any) => ({
          id: (exec.id || '').slice(0, 8),
          workflow: exec.workflow_id || exec.workflow_name || 'unknown',
          status: exec.status,
          started: exec.started_at
        }))
        setProcesses(running.filter((p: Process) => p.status === 'running'))
      }).catch(() => {
        setProcesses([])
      })
    }
    fetchProcesses()
    const interval = setInterval(fetchProcesses, 5000)
    return () => clearInterval(interval)
  }, [])

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
      case 'active':
      case 'running':
        return 'var(--green)'
      case 'disconnected':
      case 'inactive':
        return 'var(--text-muted)'
      case 'error':
      case 'failed':
        return 'var(--red)'
      default:
        return 'var(--text-muted)'
    }
  }

  if (!leftPanelOpen) {
    return (
      <div
        className="w-10 flex-shrink-0 border-r flex flex-col items-center py-4 cursor-pointer glass-hover transition-fast"
        style={{ borderColor: 'var(--glass-border)' }}
        onClick={toggleLeftPanel}
        title="Open System"
      >
        <span style={{ color: 'var(--text-muted)', writingMode: 'vertical-rl', fontSize: '10px', letterSpacing: '2px', fontWeight: 500 }}>
          SYSTEM
        </span>
      </div>
    )
  }

  return (
    <div
      className="w-[220px] flex-shrink-0 border-r flex flex-col overflow-hidden glass"
      style={{ borderColor: 'var(--glass-border)' }}
    >
      {/* Header */}
      <div
        className="h-11 flex items-center justify-between px-4 border-b flex-shrink-0"
        style={{
          borderColor: 'var(--glass-border)',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)'
        }}
      >
        <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          System
        </span>
        <button
          onClick={toggleLeftPanel}
          className="w-6 h-6 flex items-center justify-center rounded-md glass-hover transition-fast cursor-pointer"
          style={{ color: 'var(--text-muted)' }}
        >
          &times;
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Processes (ps) */}
        <Section
          title="PROCESSES"
          badge={processes.length > 0 ? `${processes.length}` : undefined}
          badgeColor="var(--blue)"
          expanded={expandedSections.processes}
          onToggle={() => toggleSection('processes')}
        >
          {processes.length > 0 ? processes.map(proc => (
            <div
              key={proc.id}
              className="flex items-center justify-between px-3 py-1 hover:bg-[var(--bg-secondary)] transition-fast cursor-pointer group"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="animate-pulse w-1.5 h-1.5 rounded-full" style={{ background: 'var(--blue)' }} />
                <span className="text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>
                  {proc.id}
                </span>
              </div>
              <button
                className="text-[10px] px-1 opacity-0 group-hover:opacity-100 transition-fast"
                style={{ color: 'var(--red)' }}
                title="Kill process"
              >
                &#10005;
              </button>
            </div>
          )) : (
            <div className="px-3 py-1.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
              No running processes
            </div>
          )}
        </Section>

        {/* MCP Servers */}
        <Section
          title="MCP"
          badge={`${mcpServers.filter(s => s.status === 'connected').length}/${mcpServers.length}`}
          expanded={expandedSections.mcp}
          onToggle={() => toggleSection('mcp')}
        >
          {mcpServers.map(server => (
            <div
              key={server.name}
              className="flex items-center justify-between px-3 py-1 hover:bg-[var(--bg-secondary)] transition-fast cursor-pointer"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: getStatusColor(server.status) }}
                />
                <span className="text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>
                  {server.name}
                </span>
              </div>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                {server.tools_count}
              </span>
            </div>
          ))}
        </Section>

        {/* Adapters */}
        <Section
          title="ADAPTERS"
          badge={`${adapters.filter(a => a.status === 'active').length}`}
          expanded={expandedSections.adapters}
          onToggle={() => toggleSection('adapters')}
        >
          {adapters.map(adapter => (
            <div
              key={adapter.name}
              className="flex items-center justify-between px-3 py-1 hover:bg-[var(--bg-secondary)] transition-fast cursor-pointer"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: getStatusColor(adapter.status) }}
                />
                <span className="text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>
                  {adapter.name}
                </span>
              </div>
              <span
                className="text-[9px] px-1 rounded"
                style={{ color: 'var(--text-muted)', background: 'var(--bg-tertiary)' }}
              >
                {adapter.type}
              </span>
            </div>
          ))}
        </Section>

        {/* Workspaces */}
        <Section
          title="WORKSPACES"
          badge={workspaces.find(w => w.active)?.name}
          expanded={expandedSections.workspaces}
          onToggle={() => toggleSection('workspaces')}
        >
          {workspaces.map(ws => (
            <div
              key={ws.name}
              className="flex items-center justify-between px-3 py-1 hover:bg-[var(--bg-secondary)] transition-fast cursor-pointer"
              style={{ background: ws.active ? 'var(--accent-alpha-10)' : 'transparent' }}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[10px]" style={{ color: ws.active ? 'var(--accent)' : 'var(--text-muted)' }}>
                  {ws.active ? '&#9679;' : '&#9675;'}
                </span>
                <span className="text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>
                  {ws.name}
                </span>
              </div>
            </div>
          ))}
        </Section>
      </div>

      {/* Quick Actions */}
      <div className="border-t p-2 flex gap-1" style={{ borderColor: 'var(--border)' }}>
        <QuickAction icon="&#128203;" label="logs" onClick={() => openPopup('logs')} />
        <QuickAction icon="&#128190;" label="state" onClick={() => openPopup('state')} />
        <QuickAction icon="&#9881;" label="settings" onClick={() => openPopup('settings')} />
      </div>
    </div>
  )
}

function Section({
  title,
  badge,
  badgeColor,
  expanded,
  onToggle,
  children
}: {
  title: string
  badge?: string
  badgeColor?: string
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border-b" style={{ borderColor: 'var(--border)' }}>
      <div
        className="flex items-center justify-between px-3 py-1.5 cursor-pointer hover:bg-[var(--bg-secondary)] transition-fast"
        onClick={onToggle}
      >
        <div className="flex items-center gap-1.5">
          <span
            className="text-[9px] transition-transform"
            style={{
              color: 'var(--text-muted)',
              transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)'
            }}
          >
            &#9654;
          </span>
          <span className="text-[10px] font-semibold tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {title}
          </span>
        </div>
        {badge && (
          <span
            className="text-[9px] px-1 rounded"
            style={{ color: badgeColor || 'var(--text-muted)', background: badgeColor ? `${badgeColor}20` : 'var(--bg-tertiary)' }}
          >
            {badge}
          </span>
        )}
      </div>
      {expanded && <div className="pb-1">{children}</div>}
    </div>
  )
}

function QuickAction({ icon, label, onClick }: { icon: string; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex-1 flex flex-col items-center py-1 rounded hover:bg-[var(--bg-secondary)] transition-fast cursor-pointer"
      title={label}
    >
      <span className="text-[11px]" style={{ color: 'var(--text-muted)' }} dangerouslySetInnerHTML={{ __html: icon }} />
      <span className="text-[8px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </button>
  )
}
