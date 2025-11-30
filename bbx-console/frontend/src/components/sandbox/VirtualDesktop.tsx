import { useState, useEffect, useRef } from 'react'

/**
 * BBX OS - Bare-Metal Operating System for AI Agents
 *
 * ЭТО НЕ имитация Windows/macOS/Linux!
 * Это визуализация BBX Kernel - нашей bare-metal OS.
 *
 * Архитектура BBX OS:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                      USER SPACE (Agents)                        │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                    SYSCALL INTERFACE                            │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                      BBX KERNEL                                 │
 * │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
 * │  │  SCHEDULER  │  │   MEMORY    │  │       AGENT RING        │ │
 * │  │  (DAG-based)│  │  (Tiered)   │  │  (io_uring-inspired)    │ │
 * │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
 * ├─────────────────────────────────────────────────────────────────┤
 * │                       HARDWARE (x86_64)                         │
 * └─────────────────────────────────────────────────────────────────┘
 */

// =============================================================================
// TYPES
// =============================================================================

interface Task {
  id: number
  name: string
  priority: 'REALTIME' | 'HIGH' | 'NORMAL' | 'LOW'
  state: 'RUNNING' | 'READY' | 'BLOCKED' | 'TERMINATED'
  cpu_time: number
}

interface MemoryTier {
  name: 'HOT' | 'WARM' | 'COOL' | 'COLD'
  items: number
  max_items: number
  size_bytes: number
}

interface RingOp {
  id: number
  type: 'READ' | 'WRITE' | 'NETWORK' | 'AGENT_CALL' | 'TIMER'
  priority: 'HIGH' | 'NORMAL' | 'LOW'
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED'
}

interface KernelLog {
  timestamp: number
  module: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  message: string
}

interface Syscall {
  num: number
  name: string
  task_id: number
  timestamp: number
}

// =============================================================================
// BBX OS VISUALIZATION
// =============================================================================

export function VirtualDesktop() {
  // Kernel state
  const [booted, setBooted] = useState(false)
  const [bootProgress, setBootProgress] = useState(0)
  const [bootMessages, setBootMessages] = useState<string[]>([])
  const [kernelLogs, setKernelLogs] = useState<KernelLog[]>([])
  const [uptime, setUptime] = useState(0)

  // Scheduler state
  const [tasks, setTasks] = useState<Task[]>([])
  const [currentTask, setCurrentTask] = useState<number | null>(null)

  // Memory state
  const [memoryTiers, setMemoryTiers] = useState<MemoryTier[]>([
    { name: 'HOT', items: 0, max_items: 100, size_bytes: 0 },
    { name: 'WARM', items: 0, max_items: 500, size_bytes: 0 },
    { name: 'COOL', items: 0, max_items: 2000, size_bytes: 0 },
    { name: 'COLD', items: 0, max_items: 10000, size_bytes: 0 },
  ])

  // AgentRing state
  const [ringOps, setRingOps] = useState<RingOp[]>([])
  const [ringStats, setRingStats] = useState({ submitted: 0, completed: 0, pending: 0 })

  // Recent syscalls
  const [syscalls, setSyscalls] = useState<Syscall[]>([])

  const logRef = useRef<HTMLDivElement>(null)

  // Boot sequence
  useEffect(() => {
    const bootSequence = [
      '[BOOT] BBX Kernel v0.1.0 initializing...',
      '[CPU]  Detecting x86_64 CPU features...',
      '[CPU]  SSE, SSE2, AVX supported',
      '[CPU]  Loading GDT...',
      '[CPU]  Loading IDT...',
      '[MEM]  Initializing frame allocator...',
      '[MEM]  Physical memory: 256 MB available',
      '[MEM]  Setting up paging...',
      '[MEM]  Initializing heap (256 KB)...',
      '[MEM]  ContextTiering enabled (HOT/WARM/COOL/COLD)',
      '[DRV]  Serial driver loaded (COM1: 115200 baud)',
      '[DRV]  Timer driver loaded (PIT: 100 Hz)',
      '[DRV]  Keyboard driver loaded (PS/2)',
      '[SCHED] Initializing DAG-based scheduler...',
      '[SCHED] Priority queues: REALTIME > HIGH > NORMAL > LOW',
      '[RING] Initializing AgentRing (io_uring-inspired)...',
      '[RING] Submission queue: 256 entries',
      '[RING] Completion queue: 256 entries',
      '[RING] Workers: 4 threads',
      '[SYS]  Registering syscall handlers...',
      '[SYS]  20 syscalls available',
      '[A2A]  Agent-to-Agent protocol ready',
      '',
      '    ____  ____  _  __   ____  _____',
      '   | __ )| __ )\\  \\/ /  / __ \\/ ___/',
      '   |  _ \\|  _ \\ \\  /  / / / /\\__ \\',
      '   | |_) | |_) |/  \\ / /_/ /___/ /',
      '   |____/|____//_/\\_\\\\____//____/',
      '',
      '   Operating System for AI Agents',
      '   Copyright 2025 Ilya Makarov',
      '',
      '[BBX]  Kernel ready. Awaiting agents...',
    ]

    let i = 0
    const bootInterval = setInterval(() => {
      if (i < bootSequence.length) {
        setBootMessages(prev => [...prev, bootSequence[i]])
        setBootProgress(Math.round((i / bootSequence.length) * 100))
        i++
      } else {
        clearInterval(bootInterval)
        setBooted(true)
        setBootProgress(100)
      }
    }, 80)

    return () => clearInterval(bootInterval)
  }, [])

  // Uptime counter
  useEffect(() => {
    if (!booted) return
    const interval = setInterval(() => {
      setUptime(prev => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [booted])

  // Simulate kernel activity after boot
  useEffect(() => {
    if (!booted) return

    // Simulate tasks
    const taskInterval = setInterval(() => {
      setTasks(prev => {
        const newTasks = [...prev]
        // Random task state changes
        if (newTasks.length > 0 && Math.random() > 0.7) {
          const idx = Math.floor(Math.random() * newTasks.length)
          newTasks[idx] = {
            ...newTasks[idx],
            cpu_time: newTasks[idx].cpu_time + Math.floor(Math.random() * 10),
            state: Math.random() > 0.3 ? 'RUNNING' : 'READY',
          }
        }
        // Occasionally add new task
        if (newTasks.length < 8 && Math.random() > 0.9) {
          newTasks.push({
            id: Date.now(),
            name: ['agent_worker', 'workflow_exec', 'io_handler', 'state_sync'][Math.floor(Math.random() * 4)],
            priority: ['REALTIME', 'HIGH', 'NORMAL', 'LOW'][Math.floor(Math.random() * 4)] as Task['priority'],
            state: 'READY',
            cpu_time: 0,
          })
        }

        // Update current task based on new tasks
        if (newTasks.length > 0) {
          setCurrentTask(newTasks[Math.floor(Math.random() * newTasks.length)]?.id || null)
        }

        return newTasks
      })
    }, 500)

    // Simulate memory activity
    const memInterval = setInterval(() => {
      setMemoryTiers(prev => prev.map(tier => ({
        ...tier,
        items: Math.min(tier.max_items, Math.max(0, tier.items + Math.floor(Math.random() * 10) - 3)),
        size_bytes: tier.items * (tier.name === 'HOT' ? 1024 : tier.name === 'WARM' ? 512 : tier.name === 'COOL' ? 256 : 128),
      })))
    }, 1000)

    // Simulate ring operations
    const ringInterval = setInterval(() => {
      setRingOps(prev => {
        let ops = [...prev]
        // Complete some ops
        ops = ops.map(op => op.status === 'PROCESSING' && Math.random() > 0.5
          ? { ...op, status: 'COMPLETED' as const }
          : op.status === 'PENDING' && Math.random() > 0.7
          ? { ...op, status: 'PROCESSING' as const }
          : op
        )
        // Remove completed
        ops = ops.filter(op => op.status !== 'COMPLETED' || Math.random() > 0.3)
        // Add new ops
        if (ops.length < 10 && Math.random() > 0.6) {
          ops.push({
            id: Date.now(),
            type: ['READ', 'WRITE', 'NETWORK', 'AGENT_CALL', 'TIMER'][Math.floor(Math.random() * 5)] as RingOp['type'],
            priority: ['HIGH', 'NORMAL', 'LOW'][Math.floor(Math.random() * 3)] as RingOp['priority'],
            status: 'PENDING',
          })
        }

        // Update ring stats
        setRingStats({
          submitted: Math.floor(Math.random() * 100) + 50,
          completed: Math.floor(Math.random() * 80) + 40,
          pending: ops.filter(op => op.status === 'PENDING').length,
        })

        return ops.slice(-12)
      })
    }, 300)

    // Simulate syscalls
    const syscallInterval = setInterval(() => {
      if (Math.random() > 0.5) {
        const syscallNames = ['spawn', 'io_submit', 'io_wait', 'state_get', 'state_set', 'agent_call', 'yield', 'sleep']
        setSyscalls(prev => [...prev, {
          num: Math.floor(Math.random() * 100),
          name: syscallNames[Math.floor(Math.random() * syscallNames.length)],
          task_id: Math.floor(Math.random() * 1000),
          timestamp: Date.now(),
        }].slice(-20))
      }
    }, 200)

    // Kernel logs
    const logInterval = setInterval(() => {
      if (Math.random() > 0.7) {
        const modules = ['SCHED', 'MEM', 'RING', 'SYS', 'A2A']
        const messages = [
          'Task scheduled',
          'Memory tier promotion',
          'I/O operation completed',
          'Syscall handled',
          'Agent message received',
          'Context switch',
          'Page fault handled',
          'Timer interrupt',
        ]
        const level: KernelLog['level'] = Math.random() > 0.9 ? 'WARN' : 'INFO'
        setKernelLogs(prev => [...prev, {
          timestamp: Date.now(),
          module: modules[Math.floor(Math.random() * modules.length)],
          level,
          message: messages[Math.floor(Math.random() * messages.length)],
        }].slice(-50))
      }
    }, 500)

    return () => {
      clearInterval(taskInterval)
      clearInterval(memInterval)
      clearInterval(ringInterval)
      clearInterval(syscallInterval)
      clearInterval(logInterval)
    }
  }, [booted]) // Only depend on booted - removed tasks/ringOps to prevent infinite loop

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [bootMessages, kernelLogs])

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = seconds % 60
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'HOT': return '#ff453a'
      case 'WARM': return '#ff9f0a'
      case 'COOL': return '#0a84ff'
      case 'COLD': return '#8e8e93'
      default: return '#fff'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'REALTIME': return '#ff453a'
      case 'HIGH': return '#ff9f0a'
      case 'NORMAL': return '#30d158'
      case 'LOW': return '#8e8e93'
      default: return '#fff'
    }
  }

  // =============================================================================
  // RENDER
  // =============================================================================

  return (
    <div
      className="flex-1 flex flex-col overflow-hidden font-mono text-[11px]"
      style={{ background: '#0a0a0a', color: '#e0e0e0' }}
    >
      {/* Top Bar - System Info */}
      <div
        className="h-8 flex items-center justify-between px-4 border-b"
        style={{ borderColor: '#222', background: '#111' }}
      >
        <div className="flex items-center gap-4">
          <span style={{ color: '#0a84ff' }}>BBX OS</span>
          <span style={{ color: '#666' }}>|</span>
          <span style={{ color: booted ? '#30d158' : '#ff9f0a' }}>
            {booted ? '● RUNNING' : '◐ BOOTING'}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span style={{ color: '#666' }}>Uptime: {formatUptime(uptime)}</span>
          <span style={{ color: '#666' }}>|</span>
          <span style={{ color: '#666' }}>Tasks: {tasks.length}</span>
          <span style={{ color: '#666' }}>|</span>
          <span style={{ color: '#666' }}>Ring: {ringOps.length} ops</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - Kernel Log */}
        <div className="w-[400px] flex flex-col border-r" style={{ borderColor: '#222' }}>
          <div className="px-3 py-2 border-b" style={{ borderColor: '#222', background: '#111' }}>
            <span style={{ color: '#666' }}>KERNEL LOG</span>
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-auto p-2"
            style={{ background: '#0a0a0a' }}
          >
            {bootMessages.map((msg, i) => (
              <div
                key={i}
                style={{
                  color: msg.startsWith('[') ? (
                    msg.includes('ERROR') ? '#ff453a' :
                    msg.includes('WARN') ? '#ff9f0a' :
                    msg.includes('[BBX]') ? '#30d158' :
                    msg.includes('____') ? '#0a84ff' :
                    '#888'
                  ) : '#666'
                }}
              >
                {msg || ' '}
              </div>
            ))}
            {booted && kernelLogs.map((log, i) => (
              <div key={i} style={{ color: log.level === 'WARN' ? '#ff9f0a' : '#666' }}>
                [{log.module}] {log.message}
              </div>
            ))}
            {booted && <span className="animate-pulse">█</span>}
          </div>
        </div>

        {/* Center Panel - Scheduler + Memory */}
        <div className="flex-1 flex flex-col">
          {/* Scheduler */}
          <div className="flex-1 flex flex-col border-b" style={{ borderColor: '#222' }}>
            <div className="px-3 py-2 border-b flex items-center justify-between" style={{ borderColor: '#222', background: '#111' }}>
              <span style={{ color: '#666' }}>SCHEDULER (DAG-based, Priority Queues)</span>
              <span style={{ color: '#666' }}>Current: {currentTask ? `Task #${currentTask}` : 'idle'}</span>
            </div>
            <div className="flex-1 p-3 overflow-auto">
              {!booted ? (
                <div className="text-center py-8" style={{ color: '#444' }}>
                  Waiting for kernel boot...
                </div>
              ) : tasks.length === 0 ? (
                <div className="text-center py-8" style={{ color: '#444' }}>
                  No tasks scheduled. Kernel idle.
                </div>
              ) : (
                <div className="space-y-1">
                  {['REALTIME', 'HIGH', 'NORMAL', 'LOW'].map(priority => {
                    const priorityTasks = tasks.filter(t => t.priority === priority)
                    if (priorityTasks.length === 0) return null
                    return (
                      <div key={priority}>
                        <div className="text-[9px] mb-1" style={{ color: getPriorityColor(priority) }}>
                          {priority} ({priorityTasks.length})
                        </div>
                        <div className="flex flex-wrap gap-1 mb-2">
                          {priorityTasks.map(task => (
                            <div
                              key={task.id}
                              className="px-2 py-1 rounded text-[10px]"
                              style={{
                                background: task.id === currentTask ? '#1a3a1a' : '#1a1a1a',
                                border: `1px solid ${task.id === currentTask ? '#30d158' : '#333'}`,
                                color: task.state === 'RUNNING' ? '#30d158' : '#888',
                              }}
                            >
                              {task.name} ({task.cpu_time}ms)
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Memory Tiers */}
          <div className="h-[140px] flex flex-col">
            <div className="px-3 py-2 border-b" style={{ borderColor: '#222', background: '#111' }}>
              <span style={{ color: '#666' }}>CONTEXT TIERING (MGLRU-inspired)</span>
            </div>
            <div className="flex-1 p-3 flex gap-3">
              {memoryTiers.map(tier => (
                <div key={tier.name} className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span style={{ color: getTierColor(tier.name) }}>{tier.name}</span>
                    <span style={{ color: '#666' }}>{tier.items}/{tier.max_items}</span>
                  </div>
                  <div
                    className="h-3 rounded overflow-hidden"
                    style={{ background: '#1a1a1a' }}
                  >
                    <div
                      className="h-full rounded transition-all"
                      style={{
                        width: `${(tier.items / tier.max_items) * 100}%`,
                        background: getTierColor(tier.name),
                        opacity: 0.7,
                      }}
                    />
                  </div>
                  <div className="text-[9px] mt-1" style={{ color: '#444' }}>
                    {(tier.size_bytes / 1024).toFixed(1)} KB
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Panel - AgentRing + Syscalls */}
        <div className="w-[280px] flex flex-col border-l" style={{ borderColor: '#222' }}>
          {/* AgentRing */}
          <div className="flex-1 flex flex-col border-b" style={{ borderColor: '#222' }}>
            <div className="px-3 py-2 border-b" style={{ borderColor: '#222', background: '#111' }}>
              <span style={{ color: '#666' }}>AGENT RING (io_uring-inspired)</span>
            </div>
            <div className="flex-1 p-2 overflow-auto">
              <div className="flex justify-between mb-2 text-[9px]" style={{ color: '#666' }}>
                <span>SQ: {ringStats.submitted}</span>
                <span>CQ: {ringStats.completed}</span>
                <span>Pending: {ringStats.pending}</span>
              </div>
              {ringOps.length === 0 ? (
                <div className="text-center py-4" style={{ color: '#444' }}>
                  Ring idle
                </div>
              ) : (
                <div className="space-y-1">
                  {ringOps.map(op => (
                    <div
                      key={op.id}
                      className="flex items-center justify-between px-2 py-1 rounded"
                      style={{ background: '#111' }}
                    >
                      <span style={{ color: '#888' }}>{op.type}</span>
                      <span style={{
                        color: op.status === 'COMPLETED' ? '#30d158' :
                               op.status === 'PROCESSING' ? '#0a84ff' : '#666'
                      }}>
                        {op.status === 'PROCESSING' ? '◐' : op.status === 'COMPLETED' ? '✓' : '○'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Recent Syscalls */}
          <div className="h-[180px] flex flex-col">
            <div className="px-3 py-2 border-b" style={{ borderColor: '#222', background: '#111' }}>
              <span style={{ color: '#666' }}>SYSCALLS</span>
            </div>
            <div className="flex-1 p-2 overflow-auto">
              {syscalls.length === 0 ? (
                <div className="text-center py-4" style={{ color: '#444' }}>
                  No syscalls
                </div>
              ) : (
                <div className="space-y-0.5">
                  {syscalls.slice(-10).map((sc, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-[9px]"
                      style={{ color: '#666' }}
                    >
                      <span style={{ color: '#0a84ff' }}>{sc.num}</span>
                      <span style={{ color: '#888' }}>{sc.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar - Boot Progress or Stats */}
      <div
        className="h-6 flex items-center justify-between px-4 border-t"
        style={{ borderColor: '#222', background: '#111' }}
      >
        {!booted ? (
          <>
            <span style={{ color: '#666' }}>Booting BBX Kernel...</span>
            <div className="flex items-center gap-2">
              <div className="w-32 h-1.5 rounded overflow-hidden" style={{ background: '#222' }}>
                <div
                  className="h-full rounded transition-all"
                  style={{ width: `${bootProgress}%`, background: '#0a84ff' }}
                />
              </div>
              <span style={{ color: '#666' }}>{bootProgress}%</span>
            </div>
          </>
        ) : (
          <>
            <span style={{ color: '#30d158' }}>BBX Kernel Ready</span>
            <div className="flex items-center gap-4" style={{ color: '#666' }}>
              <span>x86_64</span>
              <span>256 MB RAM</span>
              <span>4 Workers</span>
              <span>20 Syscalls</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
