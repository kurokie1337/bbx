import { useState, useEffect, useRef } from 'react'
import { api } from '@/services/api'

/**
 * BBX OS - Real-time Kernel Visualization
 *
 * Connects to backend APIs:
 * - /api/memory/stats - ContextTiering data
 * - /api/ring/stats - AgentRing data
 * - /api/executions - Active tasks
 * - /api/kernel/syscalls - Syscall reference
 */

interface MemoryTier {
  tier: string
  items: number
  size_bytes: number
  max_size_bytes: number
  utilization: number
}

interface MemoryStats {
  generations: MemoryTier[]
  total_items: number
  total_size_bytes: number
  promotions: number
  demotions: number
  cache_hits: number
  cache_misses: number
  hit_rate: number
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
  worker_utilization: number
}

interface Execution {
  id: string
  workflow_id: string
  workflow_name?: string
  status: string
  started_at?: string
}

interface Syscall {
  num: number
  name: string
  category: string
  desc: string
}

export function VirtualDesktop() {
  // Boot state
  const [bootIdx, setBootIdx] = useState(0)
  const [booted, setBooted] = useState(false)
  const [uptime, setUptime] = useState(0)

  // Real data from backend
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null)
  const [ringStats, setRingStats] = useState<RingStats | null>(null)
  const [executions, setExecutions] = useState<Execution[]>([])
  const [syscalls, setSyscalls] = useState<Syscall[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [logs, setLogs] = useState<string[]>([])

  const logsRef = useRef<HTMLDivElement>(null)

  const bootMsgs = [
    '[BOOT] BBX Kernel v0.1.0',
    '[CPU]  GDT loaded',
    '[CPU]  IDT loaded',
    '[MEM]  Frame allocator: OK',
    '[MEM]  Heap: 256KB',
    '[MEM]  ContextTiering: ON',
    '[DRV]  Serial: COM1',
    '[DRV]  Timer: 100Hz',
    '[SCHED] DAG scheduler ready',
    '[RING] AgentRing ready',
    '[SYS]  Syscalls: 20',
    '',
    '  ____  ____  _  __',
    ' | __ )| __ )\\  \\/ /',
    ' |  _ \\|  _ \\ \\  /',
    ' | |_) | |_) |/  \\',
    ' |____/|____//_/\\_\\',
    '',
    ' OS for AI Agents',
    '',
    '[BBX] Connecting to backend...',
  ]

  // Boot sequence - fast boot (10ms per line)
  useEffect(() => {
    if (bootIdx >= bootMsgs.length) {
      setBooted(true)
      return
    }
    const t = setTimeout(() => setBootIdx(i => i + 1), 10)
    return () => clearTimeout(t)
  }, [bootIdx])

  // Fetch data from backend after boot
  useEffect(() => {
    if (!booted) return

    const addLog = (msg: string) => {
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`].slice(-30))
    }

    // Fetch memory stats
    const fetchMemory = async () => {
      try {
        const response = await api.get<MemoryStats>('/memory/stats')
        setMemoryStats(response.data)
        addLog('[MEM] Stats updated')
      } catch (e) {
        addLog('[MEM] Fetch failed')
      }
    }

    // Fetch ring stats
    const fetchRing = async () => {
      try {
        const response = await api.get<RingStats>('/ring/stats')
        setRingStats(response.data)
        addLog('[RING] Stats updated')
      } catch (e) {
        addLog('[RING] Fetch failed')
      }
    }

    // Fetch executions (tasks)
    const fetchExecutions = async () => {
      try {
        const response = await api.get<Execution[]>('/executions/')
        setExecutions(response.data)
        if (response.data.length > 0) {
          addLog(`[SCHED] ${response.data.length} active tasks`)
        }
      } catch (e) {
        addLog('[SCHED] Fetch failed')
      }
    }

    // Fetch syscalls reference
    const fetchSyscalls = async () => {
      try {
        const response = await api.get<{ syscalls: Syscall[] }>('/kernel/syscalls')
        setSyscalls(response.data.syscalls || [])
        addLog(`[SYS] ${response.data.syscalls?.length || 0} syscalls loaded`)
      } catch (e) {
        addLog('[SYS] Syscalls fetch failed')
      }
    }

    // Initial fetch
    addLog('[BBX] Initializing...')
    fetchMemory()
    fetchRing()
    fetchExecutions()
    fetchSyscalls()

    // Uptime counter
    const uptimeInt = setInterval(() => setUptime(u => u + 1), 1000)

    // Poll for updates
    const memInt = setInterval(fetchMemory, 2000)
    const ringInt = setInterval(fetchRing, 1000)
    const execInt = setInterval(fetchExecutions, 1500)

    return () => {
      clearInterval(uptimeInt)
      clearInterval(memInt)
      clearInterval(ringInt)
      clearInterval(execInt)
    }
  }, [booted])

  // Autoscroll logs
  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight
  }, [logs, bootIdx])

  const fmt = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`
  const fmtBytes = (b: number) => b > 1048576 ? `${(b / 1048576).toFixed(1)}MB` : b > 1024 ? `${(b / 1024).toFixed(1)}KB` : `${b}B`

  const tierColor: Record<string, string> = {
    HOT: '#f44',
    WARM: '#fa0',
    COOL: '#0af',
    COLD: '#666',
  }

  const statusColor: Record<string, string> = {
    running: '#4f4',
    completed: '#0af',
    failed: '#f44',
    pending: '#fa0',
  }

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#080808', color: '#ccc', fontFamily: 'monospace', fontSize: '11px' }}>

      {/* Top bar */}
      <div style={{ height: 28, background: '#111', borderBottom: '1px solid #222', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px' }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <span style={{ color: '#0af', fontWeight: 'bold' }}>BBX OS</span>
          <span style={{ color: booted ? '#4f4' : '#fa0' }}>{booted ? '● LIVE' : '◐ BOOT'}</span>
        </div>
        <div style={{ color: '#666' }}>
          {booted && `UP ${fmt(uptime)} | TASKS ${executions.length} | RING ${ringStats?.pending_count || 0}`}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* Left - Logs */}
        <div style={{ width: 300, borderRight: '1px solid #222', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '6px 10px', background: '#0a0a0a', borderBottom: '1px solid #222', color: '#666', fontSize: 10 }}>KERNEL LOG</div>
          <div ref={logsRef} style={{ flex: 1, overflow: 'auto', padding: 8 }}>
            {bootMsgs.slice(0, bootIdx).map((m, i) => (
              <div key={i} style={{ color: m.includes('BBX') ? '#4f4' : m.includes('___') ? '#0af' : '#777' }}>{m || '\u00A0'}</div>
            ))}
            {booted && logs.map((l, i) => <div key={`l${i}`} style={{ color: '#666' }}>{l}</div>)}
            {!booted && <span style={{ color: '#0af' }}>█</span>}
          </div>
        </div>

        {/* Center */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>

          {/* Executions (Tasks) */}
          <div style={{ flex: 1, borderBottom: '1px solid #222', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '6px 10px', background: '#0a0a0a', borderBottom: '1px solid #222', color: '#666', fontSize: 10 }}>
              SCHEDULER - Active Executions
            </div>
            <div style={{ flex: 1, padding: 10, overflow: 'auto' }}>
              {!booted ? <div style={{ color: '#444' }}>Booting...</div> : executions.length === 0 ? (
                <div style={{ color: '#444' }}>No active tasks - run a workflow</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {executions.map(exec => (
                    <div key={exec.id} style={{
                      padding: '8px 12px',
                      background: exec.status === 'running' ? '#1a2a1a' : '#151515',
                      border: `1px solid ${statusColor[exec.status] || '#333'}`,
                      borderRadius: 3
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ color: '#aaa' }}>{exec.workflow_name || exec.workflow_id}</span>
                        <span style={{ color: statusColor[exec.status] || '#666', fontSize: 10 }}>{exec.status.toUpperCase()}</span>
                      </div>
                      <div style={{ color: '#555', fontSize: 9, marginTop: 4 }}>ID: {exec.id.slice(0, 8)}...</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Memory Tiers */}
          <div style={{ height: 130, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '6px 10px', background: '#0a0a0a', borderBottom: '1px solid #222', color: '#666', fontSize: 10, display: 'flex', justifyContent: 'space-between' }}>
              <span>CONTEXT TIERING (MGLRU)</span>
              {memoryStats && <span>Hit Rate: {(memoryStats.hit_rate * 100).toFixed(1)}%</span>}
            </div>
            <div style={{ flex: 1, padding: 10, display: 'flex', gap: 16, alignItems: 'center' }}>
              {memoryStats?.generations?.map(tier => (
                <div key={tier.tier} style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: tierColor[tier.tier] || '#888' }}>{tier.tier}</span>
                    <span style={{ color: '#666' }}>{tier.items} items</span>
                  </div>
                  <div style={{ height: 8, background: '#1a1a1a', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min(100, tier.utilization * 100)}%`,
                      background: tierColor[tier.tier] || '#888',
                      opacity: 0.7,
                      transition: 'width 0.3s'
                    }} />
                  </div>
                  <div style={{ color: '#555', fontSize: 9, marginTop: 2 }}>{fmtBytes(tier.size_bytes)}</div>
                </div>
              )) || (
                  <div style={{ color: '#444' }}>Loading memory stats...</div>
                )}
            </div>
          </div>
        </div>

        {/* Right - Ring + Syscalls */}
        <div style={{ width: 240, borderLeft: '1px solid #222', display: 'flex', flexDirection: 'column' }}>

          {/* Ring Stats */}
          <div style={{ flex: 1, borderBottom: '1px solid #222', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '6px 10px', background: '#0a0a0a', borderBottom: '1px solid #222', color: '#666', fontSize: 10 }}>AGENT RING (io_uring)</div>
            <div style={{ flex: 1, padding: 8, overflow: 'auto' }}>
              {ringStats ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#888' }}>Submitted</span>
                    <span style={{ color: '#4f4' }}>{ringStats.operations_submitted}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#888' }}>Completed</span>
                    <span style={{ color: '#0af' }}>{ringStats.operations_completed}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#888' }}>Failed</span>
                    <span style={{ color: '#f44' }}>{ringStats.operations_failed}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#888' }}>Pending</span>
                    <span style={{ color: '#fa0' }}>{ringStats.pending_count}</span>
                  </div>
                  <div style={{ borderTop: '1px solid #222', paddingTop: 8, marginTop: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#666' }}>Workers</span>
                      <span style={{ color: '#888' }}>{ringStats.active_workers}/{ringStats.worker_pool_size}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#666' }}>Throughput</span>
                      <span style={{ color: '#888' }}>{ringStats.throughput_ops_sec.toFixed(1)} ops/s</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: '#666' }}>Latency</span>
                      <span style={{ color: '#888' }}>{ringStats.avg_latency_ms.toFixed(0)}ms avg</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#444' }}>Loading ring stats...</div>
              )}
            </div>
          </div>

          {/* Syscalls Reference */}
          <div style={{ height: 180, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '6px 10px', background: '#0a0a0a', borderBottom: '1px solid #222', color: '#666', fontSize: 10 }}>
              SYSCALLS ({syscalls.length})
            </div>
            <div style={{ flex: 1, padding: 8, overflow: 'auto' }}>
              {(syscalls || []).slice(0, 15).map(sc => (
                <div key={sc.num} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                  <span style={{ color: '#666', fontSize: 9 }}>{sc.num}</span>
                  <span style={{ color: '#888', fontSize: 10, flex: 1, marginLeft: 8 }}>{sc.name}</span>
                  <span style={{ color: '#555', fontSize: 9 }}>{sc.category}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom */}
      <div style={{ height: 22, background: '#111', borderTop: '1px solid #222', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 12px', fontSize: 10, color: '#555' }}>
        <span>x86_64</span>
        <span>{memoryStats ? fmtBytes(memoryStats.total_size_bytes) : '...'}</span>
        <span>{ringStats ? `${ringStats.active_workers} Workers` : '...'}</span>
        <span>io_uring</span>
        <span>DAG Scheduler</span>
      </div>
    </div>
  )
}
