import { useEffect, useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

type ViewMode = 'overview' | 'worker' | 'queue'

interface WorkerDetail {
  id: number
  status: 'active' | 'idle'
  current_task?: string
  tasks_completed: number
  avg_latency_ms: number
  uptime: string
}

interface QueueOperation {
  id: string
  type: string
  priority: 'HIGH' | 'NORMAL' | 'LOW'
  status: 'pending' | 'processing' | 'completed' | 'failed'
  submitted_at: string
  duration_ms?: number
}

interface RingStats {
  operations_submitted: number
  operations_completed: number
  operations_failed: number
  pending_count: number
  processing_count: number
  active_workers: number
  worker_pool_size: number
  throughput_ops_sec: number
  avg_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  worker_utilization: number
}

export function RingPopup() {
  const [stats, setStats] = useState<RingStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState<ViewMode>('overview')
  const [selectedWorker, setSelectedWorker] = useState<WorkerDetail | null>(null)
  const [queueOps, setQueueOps] = useState<QueueOperation[]>([])

  useEffect(() => {
    api.get('/ring/stats')
      .then((res: AxiosResponse) => {
        setStats(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const handleWorkerClick = (workerId: number, isActive: boolean) => {
    setSelectedWorker({
      id: workerId,
      status: isActive ? 'active' : 'idle',
      current_task: isActive ? `task_${workerId}_exec` : undefined,
      tasks_completed: Math.floor(Math.random() * 50) + 10,
      avg_latency_ms: Math.floor(Math.random() * 100) + 20,
      uptime: `${Math.floor(Math.random() * 60)}m`
    })
    setViewMode('worker')
  }

  const handleQueueClick = (queueType: 'submission' | 'completion') => {
    setQueueOps([
      { id: 'op_001', type: 'agent.execute', priority: 'HIGH', status: queueType === 'submission' ? 'pending' : 'completed', submitted_at: '2s ago', duration_ms: 145 },
      { id: 'op_002', type: 'workflow.step', priority: 'NORMAL', status: queueType === 'submission' ? 'pending' : 'completed', submitted_at: '5s ago', duration_ms: 89 },
      { id: 'op_003', type: 'mcp.call', priority: 'NORMAL', status: queueType === 'submission' ? 'processing' : 'completed', submitted_at: '8s ago', duration_ms: 234 },
      { id: 'op_004', type: 'agent.execute', priority: 'LOW', status: queueType === 'submission' ? 'pending' : 'failed', submitted_at: '12s ago', duration_ms: 0 },
    ])
    setViewMode('queue')
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'HIGH': return 'var(--yellow)'
      case 'NORMAL': return 'var(--text-muted)'
      case 'LOW': return 'var(--blue)'
      default: return 'var(--text-muted)'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'var(--yellow)'
      case 'processing': return 'var(--blue)'
      case 'completed': return 'var(--green)'
      case 'failed': return 'var(--red)'
      default: return 'var(--text-muted)'
    }
  }

  // Worker detail view
  if (viewMode === 'worker' && selectedWorker) {
    return (
      <PopupWrapper
        title={`WORKER ${selectedWorker.id}`}
        footer={
          <div className="text-[11px]">
            Uptime: {selectedWorker.uptime} | Tasks: {selectedWorker.tasks_completed}
          </div>
        }
      >
        <button
          onClick={() => { setViewMode('overview'); setSelectedWorker(null) }}
          className="flex items-center gap-1 text-xs mb-4 cursor-pointer hover:underline"
          style={{ color: 'var(--accent)' }}
        >
          <span>&#8592;</span> Back to overview
        </button>

        <div className="space-y-4">
          {/* Status */}
          <div
            className="rounded-lg border p-4"
            style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>STATUS</span>
              <span
                className="text-xs px-2 py-0.5 rounded"
                style={{
                  background: selectedWorker.status === 'active' ? 'var(--blue)' : 'var(--bg-tertiary)',
                  color: selectedWorker.status === 'active' ? 'var(--bg-primary)' : 'var(--text-muted)'
                }}
              >
                {selectedWorker.status}
              </span>
            </div>
            {selectedWorker.current_task && (
              <div className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>
                {selectedWorker.current_task}
              </div>
            )}
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-3">
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: 'var(--green)' }}>
                {selectedWorker.tasks_completed}
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Completed</div>
            </div>
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                {selectedWorker.avg_latency_ms}ms
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Avg Latency</div>
            </div>
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                {selectedWorker.uptime}
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Uptime</div>
            </div>
          </div>
        </div>
      </PopupWrapper>
    )
  }

  // Queue detail view
  if (viewMode === 'queue') {
    return (
      <PopupWrapper
        title="QUEUE OPERATIONS"
        footer={
          <div className="text-[11px]">
            {queueOps.length} operations
          </div>
        }
      >
        <button
          onClick={() => setViewMode('overview')}
          className="flex items-center gap-1 text-xs mb-4 cursor-pointer hover:underline"
          style={{ color: 'var(--accent)' }}
        >
          <span>&#8592;</span> Back to overview
        </button>

        <div
          className="rounded-lg border overflow-hidden"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
        >
          {queueOps.map((op, i) => (
            <div
              key={op.id}
              className="flex items-center justify-between px-3 py-2"
              style={{ borderBottom: i < queueOps.length - 1 ? '1px solid var(--border)' : 'none' }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: getStatusColor(op.status) }}
                />
                <span className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
                  {op.id}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {op.type}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span
                  className="text-[9px] px-1.5 py-0.5 rounded"
                  style={{ background: `${getPriorityColor(op.priority)}20`, color: getPriorityColor(op.priority) }}
                >
                  {op.priority}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  {op.submitted_at}
                </span>
                {op.duration_ms && op.duration_ms > 0 && (
                  <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                    {op.duration_ms}ms
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </PopupWrapper>
    )
  }

  // Overview
  return (
    <PopupWrapper
      title="AGENT RING"
      footer={stats && (
        <div className="flex items-center gap-4 text-[11px]">
          <span>Throughput: {stats.throughput_ops_sec.toFixed(1)} ops/s</span>
          <span>|</span>
          <span>Latency p50: {stats.p50_latency_ms.toFixed(0)}ms</span>
          <span>|</span>
          <span>Utilization: {stats.worker_utilization.toFixed(0)}%</span>
        </div>
      )}
    >
      {loading ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Loading...
        </div>
      ) : stats ? (
        <div className="space-y-6">
          {/* Queues */}
          <div className="grid grid-cols-2 gap-4">
            {/* Submission Queue */}
            <div
              className="rounded-lg border p-4 cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
              style={{
                background: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
              onClick={() => handleQueueClick('submission')}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  SUBMISSION QUEUE (SQ)
                </span>
                <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
              </div>
              <div className="space-y-2">
                {stats.pending_count > 0 ? (
                  Array.from({ length: Math.min(stats.pending_count, 3) }).map((_, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span style={{ color: 'var(--text-muted)' }}>&#9656;</span>
                      <span style={{ color: 'var(--text-primary)' }}>task_{i + 1}</span>
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px]"
                        style={{
                          background: i === 0 ? 'var(--yellow)' : 'var(--bg-tertiary)',
                          color: i === 0 ? 'var(--bg-primary)' : 'var(--text-muted)'
                        }}
                      >
                        {i === 0 ? 'HIGH' : 'NORMAL'}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Queue empty
                  </div>
                )}
              </div>
            </div>

            {/* Completion Queue */}
            <div
              className="rounded-lg border p-4 cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
              style={{
                background: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
              onClick={() => handleQueueClick('completion')}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                  COMPLETION QUEUE (CQ)
                </span>
                <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
              </div>
              <div className="space-y-2">
                {stats.operations_completed > 0 ? (
                  Array.from({ length: Math.min(2, stats.operations_completed) }).map((_, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span style={{ color: 'var(--green)' }}>&#10003;</span>
                      <span style={{ color: 'var(--text-primary)' }}>task_{stats.operations_completed - i}</span>
                      <span style={{ color: 'var(--text-muted)' }}>done</span>
                    </div>
                  ))
                ) : (
                  <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    No completed tasks
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Workers */}
          <div
            className="rounded-lg border p-4"
            style={{
              background: 'var(--bg-secondary)',
              borderColor: 'var(--border)'
            }}
          >
            <div className="text-xs font-semibold mb-3" style={{ color: 'var(--text-muted)' }}>
              WORKERS
            </div>
            <div className="space-y-2">
              {Array.from({ length: Math.min(stats.worker_pool_size, 4) }).map((_, i) => {
                const isActive = i < stats.active_workers
                const progress = isActive ? Math.floor(Math.random() * 80 + 20) : 0

                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 text-xs cursor-pointer hover:bg-[var(--bg-tertiary)] -mx-2 px-2 py-1 rounded transition-fast"
                    onClick={() => handleWorkerClick(i + 1, isActive)}
                  >
                    <span style={{ color: 'var(--text-muted)' }}>[{i + 1}]</span>

                    {isActive ? (
                      <>
                        <div
                          className="flex-1 h-2 rounded overflow-hidden"
                          style={{ background: 'var(--bg-tertiary)' }}
                        >
                          <div
                            className="h-full rounded"
                            style={{
                              width: `${progress}%`,
                              background: 'var(--accent)'
                            }}
                          />
                        </div>
                        <span style={{ color: 'var(--text-muted)' }}>
                          task_{i + 1} (agent)
                        </span>
                      </>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>idle</span>
                    )}
                    <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-4 gap-4">
            <StatBox label="Submitted" value={stats.operations_submitted} />
            <StatBox label="Completed" value={stats.operations_completed} color="var(--green)" />
            <StatBox label="Failed" value={stats.operations_failed} color="var(--red)" />
            <StatBox label="Pending" value={stats.pending_count} color="var(--yellow)" />
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Failed to load ring stats
        </div>
      )}
    </PopupWrapper>
  )
}

function StatBox({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div
      className="rounded-lg border p-3 text-center"
      style={{
        background: 'var(--bg-secondary)',
        borderColor: 'var(--border)'
      }}
    >
      <div
        className="text-xl font-semibold"
        style={{ color: color || 'var(--text-primary)' }}
      >
        {value}
      </div>
      <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
        {label}
      </div>
    </div>
  )
}
