import { useState, useEffect, useRef, useCallback } from 'react'
import { useOutputStore, OutputLine } from '@/stores/outputStore'
import { useUIStore } from '@/stores/uiStore'
import { useWorkflowStore, StepStatus } from '@/stores/workflowStore'
import { XTerminal } from '@/components/terminal/XTerminal'
import { api } from '@/services/api'

type SandboxMode = 'logs' | 'terminal' | 'workflow' | 'split'

interface WorkflowFile {
  name: string
  path: string
  steps?: number
}

export function WorkflowSandbox() {
  const { lines, isRunning, clearOutput, addLine } = useOutputStore()
  const { autoScrollOutput, toggleAutoScroll } = useUIStore()
  const {
    executions,
    activeExecutionId,
    setActiveExecution,
    startExecution,
    updateExecution,
    updateStep,
    addLog,
    pauseExecution,
    resumeExecution,
    stopExecution,
    clearExecutions
  } = useWorkflowStore()

  const [mode, setMode] = useState<SandboxMode>('workflow')
  const [splitRatio, setSplitRatio] = useState(50)
  const [workflows, setWorkflows] = useState<WorkflowFile[]>([])
  const [loadingWorkflows, setLoadingWorkflows] = useState(false)

  const containerRef = useRef<HTMLDivElement>(null)
  const logsRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    if (autoScrollOutput && logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [lines, autoScrollOutput])

  // Load workflows list
  useEffect(() => {
    setLoadingWorkflows(true)
    api.get('/workflows/')
      .then(res => {
        const data = res.data || []
        // Map API response to WorkflowFile format
        setWorkflows(data.map((wf: any) => ({
          name: wf.name || wf.id,
          path: wf.file_path || wf.path,
          steps: wf.step_count || 0
        })))
      })
      .catch((err) => {
        console.error('Failed to load workflows:', err)
        setWorkflows([])
      })
      .finally(() => setLoadingWorkflows(false))
  }, [])

  // Run workflow (real execution via BBX MCP)
  const runWorkflow = useCallback(async (workflow: WorkflowFile) => {
    // Create execution with placeholder steps (will be populated from workflow info)
    const executionId = startExecution(workflow.path, workflow.name, [
      { id: 'init', name: 'Initialize', adapter: 'system' },
      { id: 'execute', name: 'Execute Steps', adapter: 'bbx' },
      { id: 'cleanup', name: 'Cleanup', adapter: 'system' },
    ])

    addLog(executionId, { level: 'info', message: `Loading workflow: ${workflow.path}` })

    try {
      // Get workflow info first
      const infoRes = await api.post('/mcp/call', {
        server: 'bbx',
        tool: 'bbx_info',
        arguments: { workflow_file: workflow.path }
      })

      if (infoRes.data) {
        addLog(executionId, { level: 'info', message: `Workflow has ${infoRes.data.steps?.length || 0} steps` })
      }

      // Update first step to running
      updateStep(executionId, 'init', { status: 'running' })

      // Run the workflow
      addLog(executionId, { level: 'info', message: 'Starting execution...' })

      const runRes = await api.post('/mcp/call', {
        server: 'bbx',
        tool: 'bbx_run',
        arguments: {
          workflow_file: workflow.path,
          dry_run: false
        }
      })

      // Mark init as complete
      updateStep(executionId, 'init', { status: 'completed', duration: 500 })
      updateStep(executionId, 'execute', { status: 'running' })

      // Process result
      if (runRes.data?.success || runRes.data?.status === 'completed') {
        updateStep(executionId, 'execute', { status: 'completed', duration: 1500 })
        updateStep(executionId, 'cleanup', { status: 'running' })

        setTimeout(() => {
          updateStep(executionId, 'cleanup', { status: 'completed', duration: 200 })
          updateExecution(executionId, {
            status: 'completed',
            completedAt: new Date()
          })
          addLog(executionId, { level: 'info', message: 'Workflow completed successfully!' })
          addLine({ agent: 'system', message: `Workflow ${workflow.name} completed!`, level: 'success', type: 'message' })
        }, 500)
      } else {
        throw new Error(runRes.data?.error || 'Execution failed')
      }
    } catch (error: any) {
      addLog(executionId, { level: 'error', message: error.message || 'Execution failed' })
      updateExecution(executionId, {
        status: 'failed',
        completedAt: new Date()
      })
      addLine({ agent: 'system', message: `Workflow ${workflow.name} failed: ${error.message}`, level: 'error', type: 'message' })
    }
  }, [startExecution, updateExecution, updateStep, addLog, addLine])

  // Demo workflow simulation (offline mode)
  const runDemoWorkflow = useCallback(() => {
    const executionId = startExecution('demo_workflow', 'Demo Pipeline', [
      { id: 'fetch', name: 'Fetch Data', adapter: 'http', dependencies: [] },
      { id: 'validate', name: 'Validate', adapter: 'python', dependencies: ['fetch'] },
      { id: 'transform', name: 'Transform', adapter: 'python', dependencies: ['validate'] },
      { id: 'store', name: 'Store Results', adapter: 'file', dependencies: ['transform'] },
      { id: 'notify', name: 'Send Notification', adapter: 'shell', dependencies: ['store'] },
    ])

    // Simulate step-by-step execution
    const stepIds = ['fetch', 'validate', 'transform', 'store', 'notify']
    let currentIndex = 0

    const runNextStep = () => {
      const execution = useWorkflowStore.getState().executions.find(e => e.id === executionId)
      if (!execution || execution.status !== 'running') return

      if (currentIndex > 0) {
        updateStep(executionId, stepIds[currentIndex - 1], {
          status: 'completed',
          duration: Math.random() * 2000 + 500
        })
        addLog(executionId, {
          level: 'info',
          step: stepIds[currentIndex - 1],
          message: `Step "${stepIds[currentIndex - 1]}" completed`
        })
      }

      if (currentIndex < stepIds.length) {
        updateStep(executionId, stepIds[currentIndex], { status: 'running', progress: 0 })
        updateExecution(executionId, { currentStep: stepIds[currentIndex] })
        addLog(executionId, {
          level: 'info',
          step: stepIds[currentIndex],
          message: `Running step "${stepIds[currentIndex]}"...`
        })

        currentIndex++
        setTimeout(runNextStep, 1500)
      } else {
        updateExecution(executionId, {
          status: 'completed',
          completedAt: new Date(),
          currentStep: undefined
        })
        addLog(executionId, { level: 'info', message: 'Workflow completed successfully!' })
      }
    }

    runNextStep()
  }, [startExecution, updateExecution, updateStep, addLog])

  const activeExecution = executions.find(e => e.id === activeExecutionId) || executions[0]

  const getStepStatusColor = (status: StepStatus) => {
    switch (status) {
      case 'completed': return 'var(--green)'
      case 'running': return 'var(--blue)'
      case 'failed': return 'var(--red)'
      case 'skipped': return 'var(--yellow)'
      case 'paused': return 'var(--orange)'
      default: return 'var(--text-ghost)'
    }
  }

  const getStepStatusIcon = (status: StepStatus) => {
    switch (status) {
      case 'completed': return '✓'
      case 'running': return '↻'
      case 'failed': return '✗'
      case 'skipped': return '⊘'
      case 'paused': return '⏸'
      default: return '○'
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const renderModeSelector = () => (
    <div className="flex items-center gap-1">
      {(['workflow', 'logs', 'terminal', 'split'] as SandboxMode[]).map((m) => (
        <button
          key={m}
          onClick={() => setMode(m)}
          className={`px-2 py-0.5 text-[10px] rounded-md transition-fast cursor-pointer ${
            mode === m ? 'glass-button-accent' : 'glass-hover'
          }`}
          style={{
            color: mode === m ? 'var(--accent-light)' : 'var(--text-muted)',
            background: mode === m ? 'var(--accent-alpha-20)' : 'transparent'
          }}
        >
          {m === 'workflow' && '⚡'}
          {m === 'logs' && '📋'}
          {m === 'terminal' && '▪'}
          {m === 'split' && '⊞'}
          <span className="ml-1">{m}</span>
        </button>
      ))}
    </div>
  )

  const renderLogs = () => (
    <div
      ref={logsRef}
      className="flex-1 overflow-y-auto px-4 py-3 text-[11px] leading-relaxed"
      style={{ fontFamily: 'var(--font-mono)' }}
    >
      {activeExecution ? (
        <>
          {activeExecution.logs.map(log => (
            <div key={log.id} className="flex py-0.5">
              <span className="w-[70px] flex-shrink-0 opacity-60" style={{ color: 'var(--text-muted)' }}>
                {formatTime(log.timestamp)}
              </span>
              <span
                className="w-[50px] flex-shrink-0"
                style={{
                  color: log.level === 'error' ? 'var(--red)' :
                         log.level === 'warn' ? 'var(--yellow)' :
                         log.level === 'debug' ? 'var(--purple)' :
                         'var(--text-muted)'
                }}
              >
                [{log.level}]
              </span>
              {log.step && (
                <span className="w-[80px] flex-shrink-0" style={{ color: 'var(--accent)' }}>
                  [{log.step}]
                </span>
              )}
              <span className="flex-1 break-words" style={{ color: 'var(--text-primary)' }}>
                {log.message}
              </span>
            </div>
          ))}
        </>
      ) : lines.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-full gap-2" style={{ color: 'var(--text-muted)' }}>
          <span className="text-2xl opacity-20">○</span>
          <span className="text-[11px]">No output yet. Run a workflow to see results.</span>
        </div>
      ) : (
        <>
          {lines.map(line => renderLogLine(line))}
          {isRunning && (
            <span className="animate-pulse" style={{ color: 'var(--accent-light)' }}>█</span>
          )}
        </>
      )}
    </div>
  )

  const renderLogLine = (line: OutputLine) => {
    if (line.type === 'transition') {
      return (
        <div key={line.id} className="flex items-center gap-3 my-2">
          <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{line.message}</span>
          <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
        </div>
      )
    }

    return (
      <div key={line.id} className="flex py-0.5">
        <span className="w-[70px] flex-shrink-0 opacity-60" style={{ color: 'var(--text-muted)' }}>
          {formatTime(line.timestamp)}
        </span>
        {line.agent && (
          <span className="w-[100px] flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
            [{line.agent}]
          </span>
        )}
        <span className="flex-1 break-words" style={{ color: 'var(--text-primary)' }}>
          {line.message}
        </span>
      </div>
    )
  }

  const renderWorkflowViz = () => {
    if (!activeExecution) {
      return (
        <div className="flex-1 flex">
          {/* Workflow List */}
          <div className="w-64 border-r flex flex-col" style={{ borderColor: 'var(--glass-border)' }}>
            <div className="px-4 py-3 border-b" style={{ borderColor: 'var(--glass-border)' }}>
              <div className="text-[10px] font-semibold tracking-wider uppercase" style={{ color: 'var(--text-muted)' }}>
                Workflows
              </div>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {loadingWorkflows ? (
                <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>
                  Loading...
                </div>
              ) : workflows.length > 0 ? (
                workflows.map(wf => (
                  <div
                    key={wf.path}
                    className="flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-fast glass-hover mb-1"
                    onClick={() => runWorkflow(wf)}
                  >
                    <div className="flex items-center gap-2">
                      <span style={{ color: 'var(--accent)' }}>&#x1F4C4;</span>
                      <span className="text-[12px]" style={{ color: 'var(--text-primary)' }}>{wf.name}</span>
                    </div>
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      ▶
                    </span>
                  </div>
                ))
              ) : (
                <div className="text-[11px] text-center py-4" style={{ color: 'var(--text-muted)' }}>
                  No workflows found
                </div>
              )}
            </div>
            <div className="p-3 border-t" style={{ borderColor: 'var(--glass-border)' }}>
              <button
                onClick={runDemoWorkflow}
                className="w-full px-3 py-2 text-[11px] rounded-lg cursor-pointer transition-fast glass-button-accent"
              >
                ▶ Run Demo Workflow
              </button>
            </div>
          </div>

          {/* Empty State */}
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center" style={{ color: 'var(--text-muted)' }}>
              <div className="text-4xl mb-4 opacity-20">&#x1F500;</div>
              <div className="text-[13px] mb-2">Select a workflow to run</div>
              <div className="text-[11px]">
                Or click "Run Demo Workflow" to see a simulation
              </div>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="flex-1 flex">
        {/* Executions List */}
        <div className="w-48 border-r flex flex-col" style={{ borderColor: 'var(--glass-border)' }}>
          <div className="px-3 py-2 border-b text-[10px] font-semibold tracking-wider uppercase" style={{ borderColor: 'var(--glass-border)', color: 'var(--text-muted)' }}>
            Executions
          </div>
          <div className="flex-1 overflow-y-auto p-1">
            {executions.map(exec => (
              <div
                key={exec.id}
                onClick={() => setActiveExecution(exec.id)}
                className={`px-2 py-1.5 rounded-lg cursor-pointer transition-fast mb-1 ${
                  exec.id === activeExecutionId ? 'glass-button-accent' : 'glass-hover'
                }`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${exec.status === 'running' ? 'animate-pulse status-dot' : ''}`}
                    style={{
                      background: exec.status === 'completed' ? 'var(--green)' :
                                  exec.status === 'failed' ? 'var(--red)' :
                                  exec.status === 'paused' ? 'var(--yellow)' :
                                  'var(--blue)'
                    }}
                  />
                  <span className="text-[11px] truncate" style={{ color: 'var(--text-primary)' }}>
                    {exec.workflowName}
                  </span>
                </div>
                <div className="text-[9px] pl-4" style={{ color: 'var(--text-muted)' }}>
                  {Math.round(exec.progress)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Execution Detail */}
        <div className="flex-1 overflow-auto p-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div
                className={`w-3 h-3 rounded-full ${activeExecution.status === 'running' ? 'animate-pulse status-dot' : ''}`}
                style={{
                  background: activeExecution.status === 'completed' ? 'var(--green)' :
                              activeExecution.status === 'failed' ? 'var(--red)' :
                              activeExecution.status === 'paused' ? 'var(--yellow)' :
                              'var(--blue)'
                }}
              />
              <span className="text-[13px] font-medium" style={{ color: 'var(--text-primary)' }}>
                {activeExecution.workflowName}
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded" style={{ background: 'var(--glass-bg)', color: 'var(--text-muted)' }}>
                {activeExecution.id.slice(0, 12)}
              </span>
            </div>

            <div className="flex items-center gap-2">
              {activeExecution.status === 'running' && (
                <>
                  <button
                    onClick={() => pauseExecution(activeExecution.id)}
                    className="px-2 py-1 text-[10px] rounded glass-button cursor-pointer"
                  >
                    ⏸ Pause
                  </button>
                  <button
                    onClick={() => stopExecution(activeExecution.id)}
                    className="px-2 py-1 text-[10px] rounded glass-button cursor-pointer"
                    style={{ color: 'var(--red)' }}
                  >
                    ⏹ Stop
                  </button>
                </>
              )}
              {activeExecution.status === 'paused' && (
                <>
                  <button
                    onClick={() => resumeExecution(activeExecution.id)}
                    className="px-2 py-1 text-[10px] rounded glass-button cursor-pointer"
                    style={{ color: 'var(--green)' }}
                  >
                    ▶ Resume
                  </button>
                  <button
                    onClick={() => stopExecution(activeExecution.id)}
                    className="px-2 py-1 text-[10px] rounded glass-button cursor-pointer"
                    style={{ color: 'var(--red)' }}
                  >
                    ⏹ Stop
                  </button>
                </>
              )}
              {(activeExecution.status === 'completed' || activeExecution.status === 'failed') && (
                <button
                  onClick={runDemoWorkflow}
                  className="px-2 py-1 text-[10px] rounded glass-button cursor-pointer"
                >
                  ↻ Rerun
                </button>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Progress</span>
              <span className="text-[10px] font-mono" style={{ color: 'var(--text-primary)' }}>
                {Math.round(activeExecution.progress)}%
              </span>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden progress-glow" style={{ background: 'var(--bg-tertiary)' }}>
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${activeExecution.progress}%`,
                  background: activeExecution.status === 'failed' ? 'var(--red)' :
                              activeExecution.status === 'paused' ? 'var(--yellow)' :
                              'linear-gradient(90deg, var(--accent) 0%, var(--accent-light) 100%)',
                  boxShadow: '0 0 8px var(--accent-glow)'
                }}
              />
            </div>
          </div>

          {/* Steps */}
          <div className="space-y-2">
            {activeExecution.steps.map((step) => (
              <div
                key={step.id}
                className="flex items-center gap-3 p-3 rounded-lg transition-fast glass-hover"
                style={{
                  background: step.status === 'running' ? 'var(--accent-alpha-10)' : 'transparent',
                  borderLeft: `2px solid ${getStepStatusColor(step.status)}`
                }}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center text-[12px] font-medium ${step.status === 'running' ? 'animate-pulse' : ''}`}
                  style={{
                    background: step.status === 'running' ? 'var(--accent-alpha-20)' : 'var(--glass-bg)',
                    color: getStepStatusColor(step.status)
                  }}
                >
                  {getStepStatusIcon(step.status)}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-[12px]" style={{ color: 'var(--text-primary)' }}>{step.name}</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>
                      {step.adapter}
                    </span>
                  </div>
                  {step.status === 'running' && (
                    <div className="text-[10px] mt-0.5" style={{ color: 'var(--accent-light)' }}>
                      Running...
                    </div>
                  )}
                  {step.duration && (
                    <div className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                      {(step.duration / 1000).toFixed(2)}s
                    </div>
                  )}
                  {step.error && (
                    <div className="text-[10px] mt-0.5" style={{ color: 'var(--red)' }}>
                      {step.error}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderTerminal = () => (
    <div className="flex-1">
      <XTerminal />
    </div>
  )

  const renderSplitView = () => (
    <div className="flex-1 flex flex-col">
      <div style={{ height: `${splitRatio}%`, borderColor: 'var(--glass-border)' }} className="border-b">
        {renderWorkflowViz()}
      </div>
      <div
        className="h-1 cursor-row-resize glass-hover"
        onMouseDown={(e) => {
          const startY = e.clientY
          const startRatio = splitRatio
          const onMouseMove = (e: MouseEvent) => {
            const delta = e.clientY - startY
            const containerHeight = containerRef.current?.clientHeight || 1
            const newRatio = startRatio + (delta / containerHeight) * 100
            setSplitRatio(Math.min(80, Math.max(20, newRatio)))
          }
          const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove)
            document.removeEventListener('mouseup', onMouseUp)
          }
          document.addEventListener('mousemove', onMouseMove)
          document.addEventListener('mouseup', onMouseUp)
        }}
      />
      <div style={{ height: `${100 - splitRatio}%` }}>
        {renderLogs()}
      </div>
    </div>
  )

  return (
    <div
      ref={containerRef}
      className="flex-1 flex flex-col rounded-xl overflow-hidden min-h-[200px] glass"
      style={{
        borderColor: 'var(--glass-border)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 16px rgba(0,0,0,0.3)'
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2.5 border-b"
        style={{
          borderColor: 'var(--glass-border)',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%)'
        }}
      >
        <div className="flex items-center gap-4">
          <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
            Sandbox
          </span>
          {renderModeSelector()}
        </div>

        <div className="flex items-center gap-2">
          {executions.length > 0 && (
            <select
              value={activeExecutionId || ''}
              onChange={(e) => setActiveExecution(e.target.value || null)}
              className="text-[10px] px-2 py-0.5 rounded-md bg-transparent border cursor-pointer"
              style={{ borderColor: 'var(--glass-border)', color: 'var(--text-muted)' }}
            >
              {executions.map(exec => (
                <option key={exec.id} value={exec.id}>
                  {exec.workflowName} ({exec.id.slice(0, 8)})
                </option>
              ))}
            </select>
          )}
          <button
            onClick={() => { clearOutput(); clearExecutions(); }}
            className="text-[10px] px-2 py-0.5 rounded-md cursor-pointer transition-fast glass-hover"
            style={{ color: 'var(--text-muted)' }}
          >
            Clear
          </button>
          <button
            onClick={toggleAutoScroll}
            className="text-[10px] px-2 py-0.5 rounded-md cursor-pointer transition-fast"
            style={{
              color: autoScrollOutput ? 'var(--accent-light)' : 'var(--text-muted)',
              background: autoScrollOutput ? 'var(--accent-alpha-10)' : 'transparent'
            }}
          >
            Auto-scroll
          </button>
        </div>
      </div>

      {/* Content */}
      {mode === 'logs' && renderLogs()}
      {mode === 'terminal' && renderTerminal()}
      {mode === 'workflow' && renderWorkflowViz()}
      {mode === 'split' && renderSplitView()}
    </div>
  )
}
