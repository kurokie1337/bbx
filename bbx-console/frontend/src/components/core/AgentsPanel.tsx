import { useAgentsStore, Agent, AgentStatus } from '@/stores/agentsStore'
import { useUIStore } from '@/stores/uiStore'

export function AgentsPanel() {
  const { agents } = useAgentsStore()
  const { openPopup } = useUIStore()

  const allIdle = agents.every(a => a.status === 'idle')

  // Collapsed state when all idle
  if (allIdle) {
    return (
      <div
        className="rounded-xl px-4 py-3 mb-4 cursor-pointer transition-fast glass glass-hover"
        style={{
          borderColor: 'var(--glass-border)'
        }}
        onClick={() => openPopup('agents')}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {agents.map(agent => (
              <div key={agent.id} className="flex items-center gap-2">
                <span
                  className="w-2 h-2 rounded-full border"
                  style={{ borderColor: 'var(--text-ghost)', background: 'transparent' }}
                />
                <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {agent.name}
                </span>
              </div>
            ))}
          </div>
          <span
            className="text-[10px] px-2 py-0.5 rounded-md"
            style={{ color: 'var(--text-muted)', background: 'var(--glass-bg)' }}
          >
            all idle
          </span>
        </div>
      </div>
    )
  }

  // Expanded state
  return (
    <div
      className="rounded-xl p-4 mb-4 glass"
      style={{
        borderColor: 'var(--glass-border)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 16px rgba(0,0,0,0.2)'
      }}
    >
      <div
        className="text-[10px] font-semibold tracking-widest uppercase mb-3"
        style={{ color: 'var(--text-muted)' }}
      >
        Agents
      </div>

      <div className="flex flex-col gap-1.5">
        {agents.map(agent => (
          <AgentRow key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  )
}

function AgentRow({ agent }: { agent: Agent }) {
  const { openPopup } = useUIStore()

  return (
    <div
      className="flex items-center gap-3 h-7 px-2 -mx-2 rounded-lg cursor-pointer transition-fast hover:bg-[var(--glass-bg-hover)]"
      onClick={() => openPopup('agents')}
    >
      {/* Status indicator */}
      <AgentStatusIcon status={agent.status} />

      {/* Agent name */}
      <span
        className="text-[12px] w-20 flex-shrink-0"
        style={{ color: 'var(--text-primary)' }}
      >
        {agent.name}
      </span>

      {/* Progress bar (only when working) */}
      {agent.status === 'working' && (
        <div
          className="w-32 h-1 rounded-full overflow-hidden flex-shrink-0 progress-glow"
          style={{ background: 'var(--bg-tertiary)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{
              width: `${agent.progress || 0}%`,
              background: 'linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%)',
              boxShadow: '0 0 8px var(--accent-glow)'
            }}
          />
        </div>
      )}

      {/* Status text */}
      <span
        className="flex-1 text-[11px] truncate"
        style={{ color: getStatusTextColor(agent) }}
      >
        {getStatusText(agent)}
      </span>
    </div>
  )
}

function AgentStatusIcon({ status }: { status: AgentStatus }) {
  switch (status) {
    case 'idle':
      return (
        <span
          className="w-2 h-2 rounded-full border flex-shrink-0"
          style={{ borderColor: 'var(--text-ghost)' }}
        />
      )
    case 'working':
      return (
        <span
          className="w-2 h-2 flex-shrink-0 animate-spin status-dot"
          style={{ color: 'var(--accent)' }}
        >
          &#9680;
        </span>
      )
    case 'active':
      return (
        <span
          className="w-2 h-2 rounded-full flex-shrink-0 status-dot"
          style={{ background: 'var(--green)' }}
        />
      )
    case 'completed':
      return (
        <span
          className="w-2 h-2 flex-shrink-0 status-dot"
          style={{ color: 'var(--green)' }}
        >
          &#10003;
        </span>
      )
    case 'error':
      return (
        <span
          className="w-2 h-2 flex-shrink-0 status-dot"
          style={{ color: 'var(--red)' }}
        >
          &#10007;
        </span>
      )
    case 'queued':
      return (
        <span
          className="w-2 h-2 flex-shrink-0 status-dot"
          style={{ color: 'var(--yellow)' }}
        >
          &#9719;
        </span>
      )
    default:
      return null
  }
}

function getStatusText(agent: Agent): string {
  switch (agent.status) {
    case 'working':
      return agent.currentTask ? `"${agent.currentTask}"` : 'working...'
    case 'idle':
      return 'waiting'
    case 'completed':
      return agent.duration ? `done in ${formatDuration(agent.duration)}` : 'done'
    case 'error':
      return agent.error ? `error: ${agent.error}` : 'error'
    case 'queued':
      return agent.queuePosition ? `queued (${agent.queuePosition})` : 'queued'
    default:
      return ''
  }
}

function getStatusTextColor(agent: Agent): string {
  switch (agent.status) {
    case 'working':
      return 'var(--text-secondary)'
    case 'completed':
      return 'var(--green-muted)'
    case 'error':
      return 'var(--red-muted)'
    case 'queued':
      return 'var(--yellow-muted)'
    default:
      return 'var(--text-muted)'
  }
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}
