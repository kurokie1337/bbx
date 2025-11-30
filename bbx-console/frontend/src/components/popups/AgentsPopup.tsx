import { useEffect, useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface AgentDetail {
  id: string
  name: string
  description: string
  tools: string[]
  model: string
  status: string
  metrics: {
    tasks_completed: number
    tasks_failed: number
    avg_duration_ms: number
    success_rate: number
  }
}

export function AgentsPopup() {
  const [agents, setAgents] = useState<AgentDetail[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedAgent, setSelectedAgent] = useState<AgentDetail | null>(null)

  useEffect(() => {
    api.get('/agents/')
      .then((res: AxiosResponse) => {
        setAgents(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'working': return 'var(--blue)'
      case 'idle': return 'var(--text-muted)'
      case 'completed': return 'var(--green)'
      case 'error': return 'var(--red)'
      default: return 'var(--text-muted)'
    }
  }

  return (
    <PopupWrapper
      title="AGENTS"
      footer={
        <div className="text-[11px]">
          {agents.length} agents available
        </div>
      }
    >
      {loading ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Loading...
        </div>
      ) : selectedAgent ? (
        // Agent detail view
        <div>
          <button
            onClick={() => setSelectedAgent(null)}
            className="flex items-center gap-2 text-xs mb-4 cursor-pointer"
            style={{ color: 'var(--accent)' }}
          >
            \u2190 Back to list
          </button>

          <div
            className="rounded-lg border p-4"
            style={{
              background: 'var(--bg-secondary)',
              borderColor: 'var(--border)'
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {selectedAgent.name}
                </h3>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {selectedAgent.description || 'No description'}
                </p>
              </div>
              <div
                className="flex items-center gap-2 px-2 py-1 rounded"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ background: getStatusColor(selectedAgent.status) }}
                />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {selectedAgent.status}
                </span>
              </div>
            </div>

            {/* Model */}
            <div className="mb-4">
              <div className="text-[10px] font-semibold mb-1" style={{ color: 'var(--text-muted)' }}>
                MODEL
              </div>
              <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                {selectedAgent.model}
              </span>
            </div>

            {/* Tools */}
            <div className="mb-4">
              <div className="text-[10px] font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>
                TOOLS ({selectedAgent.tools.length})
              </div>
              <div className="flex flex-wrap gap-2">
                {selectedAgent.tools.map(tool => (
                  <span
                    key={tool}
                    className="px-2 py-1 rounded text-xs"
                    style={{
                      background: 'var(--bg-tertiary)',
                      color: 'var(--text-secondary)'
                    }}
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>

            {/* Metrics */}
            <div>
              <div className="text-[10px] font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>
                METRICS
              </div>
              <div className="grid grid-cols-4 gap-3">
                <MetricBox label="Completed" value={selectedAgent.metrics.tasks_completed} />
                <MetricBox label="Failed" value={selectedAgent.metrics.tasks_failed} />
                <MetricBox label="Avg Time" value={`${selectedAgent.metrics.avg_duration_ms}ms`} />
                <MetricBox label="Success" value={`${(selectedAgent.metrics.success_rate * 100).toFixed(0)}%`} />
              </div>
            </div>
          </div>
        </div>
      ) : (
        // Agents list
        <div className="space-y-2">
          {agents.map(agent => (
            <div
              key={agent.id}
              className="flex items-center justify-between px-4 py-3 rounded-lg cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)'
              }}
              onClick={() => setSelectedAgent(agent)}
            >
              <div className="flex items-center gap-3">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ background: getStatusColor(agent.status) }}
                />
                <div>
                  <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                    {agent.name}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {agent.description || 'No description'}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4 text-xs">
                <span style={{ color: 'var(--text-muted)' }}>
                  {agent.tools.length} tools
                </span>
                <span style={{ color: 'var(--text-muted)' }}>
                  {agent.model}
                </span>
                <span style={{ color: 'var(--text-muted)' }}>\u203A</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </PopupWrapper>
  )
}

function MetricBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      className="rounded p-2 text-center"
      style={{ background: 'var(--bg-tertiary)' }}
    >
      <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
        {value}
      </div>
      <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {label}
      </div>
    </div>
  )
}
