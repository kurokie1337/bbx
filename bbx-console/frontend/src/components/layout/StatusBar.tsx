import { useUIStore } from '@/stores/uiStore'
import { useTaskStore } from '@/stores/taskStore'
import { useAgentsStore, type AgentStatus } from '@/stores/agentsStore'
import { useEffect, useState } from 'react'
import { api } from '@/services/api'

interface Stats {
  memory: { hot: number; total: number }
  ring: { queued: number; active: number }
  tasks: { running: number; completed: number }
  processes: { running: number }
}

// Get color based on agent status
const getAgentStatusColor = (status: AgentStatus): string => {
  switch (status) {
    case 'working':
    case 'active':
      return 'var(--blue)'
    case 'completed':
      return 'var(--green)'
    case 'error':
      return 'var(--red)'
    case 'queued':
      return 'var(--yellow)'
    case 'idle':
    default:
      return 'var(--text-ghost)'
  }
}

export function StatusBar() {
  const { openPopup } = useUIStore()
  const { taskStatus, taskDuration } = useTaskStore()
  const { agents } = useAgentsStore()
  const [stats, setStats] = useState<Stats>({
    memory: { hot: 0, total: 0 },
    ring: { queued: 0, active: 0 },
    tasks: { running: 0, completed: 0 },
    processes: { running: 0 }
  })
  const [elapsed, setElapsed] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)

  // Fetch stats periodically
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [memoryRes, ringRes] = await Promise.all([
          api.get('/memory/stats').catch(() => ({ data: null })),
          api.get('/ring/stats').catch(() => ({ data: null }))
        ])

        const memoryData = memoryRes.data
        const ringData = ringRes.data

        setStats({
          memory: {
            hot: memoryData?.generations?.find((g: any) => g.tier === 'HOT')?.items || 0,
            total: memoryData?.total_items || 0
          },
          ring: {
            queued: ringData?.pending_count || 0,
            active: ringData?.active_workers || 0
          },
          tasks: { running: ringData?.processing_count || 0, completed: ringData?.operations_completed || 0 },
          processes: { running: ringData?.active_workers || 0 }
        })
      } catch {
        // Ignore
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 3000)
    return () => clearInterval(interval)
  }, [])

  // Timer for running tasks
  useEffect(() => {
    if (taskStatus === 'running') {
      setStartTime(Date.now())
      const timer = setInterval(() => {
        setElapsed(Date.now() - (startTime || Date.now()))
      }, 100)
      return () => clearInterval(timer)
    } else {
      setStartTime(null)
    }
  }, [taskStatus])

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <footer className="h-9 flex items-center justify-between glass border-t border-glass">
      {/* Left - Quick access: ring, tasks, memory */}
      <div className="flex items-center h-full">
        {/* Ring */}
        <StatusItem
          icon={
            <div className="relative">
              <div
                className="w-2 h-2 rounded-full status-dot"
                style={{ background: stats.ring.queued > 0 ? 'var(--yellow)' : 'var(--text-muted)' }}
              />
            </div>
          }
          label="ring"
          value={`${stats.ring.active}/${stats.ring.queued}`}
          active={stats.ring.queued > 0}
          onClick={() => openPopup('ring')}
        />

        {/* Tasks */}
        <StatusItem
          icon={
            <div className="relative">
              <div
                className="w-2 h-2 rounded-sm status-dot"
                style={{ background: stats.tasks.running > 0 ? 'var(--blue)' : 'var(--text-muted)' }}
              />
            </div>
          }
          label="tasks"
          value={stats.tasks.running > 0 ? `${stats.tasks.running} running` : 'idle'}
          active={stats.tasks.running > 0}
          onClick={() => openPopup('history')}
        />

        {/* Memory */}
        <StatusItem
          icon={
            <div className="relative">
              <div
                className="w-2 h-2 rounded-full status-dot"
                style={{ background: stats.memory.hot > 0 ? 'var(--red)' : 'var(--text-muted)' }}
              />
            </div>
          }
          label="memory"
          value={`${stats.memory.hot} hot`}
          active={stats.memory.hot > 0}
          onClick={() => openPopup('memory')}
        />
      </div>

      {/* Center - Agents with glow dots */}
      <div
        className="flex items-center gap-3 px-4 h-full cursor-pointer transition-fast glass-hover rounded-lg mx-2"
        onClick={() => openPopup('agents')}
      >
        <span className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
          agents
        </span>
        <div className="flex items-center gap-1.5">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`w-2 h-2 rounded-full transition-fast ${agent.status !== 'idle' ? 'status-dot' : ''} ${agent.status === 'working' ? 'animate-pulse' : ''}`}
              style={{ background: getAgentStatusColor(agent.status) }}
              title={`${agent.name}: ${agent.status}${agent.currentTask ? ` - ${agent.currentTask}` : ''}`}
            />
          ))}
        </div>
      </div>

      {/* Right - Timer & settings */}
      <div className="flex items-center h-full">
        {/* Timer */}
        <div
          className="flex items-center gap-2 px-4 h-full border-l transition-fast"
          style={{ borderColor: 'var(--glass-border)' }}
        >
          {taskStatus === 'running' ? (
            <>
              <div className="w-2 h-2 rounded-full animate-pulse status-dot" style={{ background: 'var(--blue)' }} />
              <span className="text-[11px] font-mono" style={{ color: 'var(--accent-light)' }}>
                {formatTime(elapsed)}
              </span>
            </>
          ) : taskStatus === 'completed' ? (
            <>
              <div className="w-2 h-2 rounded-full status-dot" style={{ background: 'var(--green)' }} />
              <span className="text-[11px] font-mono" style={{ color: 'var(--green)' }}>
                {formatTime(taskDuration)}
              </span>
            </>
          ) : (
            <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>ready</span>
          )}
        </div>

        {/* Logs */}
        <div
          className="flex items-center px-3 h-full border-l cursor-pointer transition-fast glass-hover"
          style={{ borderColor: 'var(--glass-border)' }}
          onClick={() => openPopup('logs')}
          title="Logs"
        >
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>&#128203;</span>
        </div>

        {/* Settings */}
        <div
          className="flex items-center px-3 h-full border-l cursor-pointer transition-fast glass-hover"
          style={{ borderColor: 'var(--glass-border)' }}
          onClick={() => openPopup('settings')}
          title="Settings"
        >
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>&#9881;</span>
        </div>
      </div>
    </footer>
  )
}

function StatusItem({
  icon,
  label,
  value,
  active,
  onClick
}: {
  icon: React.ReactNode
  label: string
  value: string
  active?: boolean
  onClick: () => void
}) {
  return (
    <div
      onClick={onClick}
      className="flex items-center gap-2 px-4 h-full border-r cursor-pointer transition-fast glass-hover"
      style={{ borderColor: 'var(--glass-border)' }}
    >
      {icon}
      <span className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
        {label}
      </span>
      <span
        className="text-[11px] font-medium"
        style={{ color: active ? 'var(--text-primary)' : 'var(--text-muted)' }}
      >
        {value}
      </span>
    </div>
  )
}
