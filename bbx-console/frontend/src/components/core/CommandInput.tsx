import { useRef, useEffect, useCallback } from 'react'
import { useTaskStore } from '@/stores/taskStore'
import { useOutputStore } from '@/stores/outputStore'
import { useAgentsStore } from '@/stores/agentsStore'
import { runWorkflow, getExecution, getWorkflows, chat } from '@/services/api'

export function CommandInput() {
  const inputRef = useRef<HTMLInputElement>(null)
  const startTimeRef = useRef<number>(0)
  const executionIdRef = useRef<string | null>(null)
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  const {
    currentTask,
    taskStatus,
    taskDuration,
    setCurrentTask,
    startTask,
    completeTask,
    failTask,
    stopTask,
    navigateHistoryUp,
    navigateHistoryDown
  } = useTaskStore()

  const { addLine, setRunning, clearOutput } = useOutputStore()
  const { setAgentStatus, resetAllAgents } = useAgentsStore()

  // Focus on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  const pollExecution = useCallback(async (executionId: string) => {
    try {
      const execution = await getExecution(executionId)

      // Process step results
      if (execution.results) {
        for (const [stepId, result] of Object.entries(execution.results)) {
          const stepResult = result as { status: string; output?: { message?: string; level?: string } }
          if (stepResult.status === 'success' && stepResult.output) {
            const output = stepResult.output
            addLine({
              agent: 'workflow',
              message: output.message || `Step ${stepId} completed`,
              level: output.level === 'error' ? 'error' : output.level === 'warn' ? 'warning' : 'success',
              type: 'message'
            })
          }
        }
      }

      if (execution.status === 'completed') {
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }

        const duration = Date.now() - startTimeRef.current
        completeTask(duration)
        setRunning(false)
        setAgentStatus('coder', 'completed', { duration })

        addLine({
          agent: 'system',
          message: `Workflow completed in ${(duration / 1000).toFixed(2)}s`,
          level: 'success',
          type: 'message'
        })
      } else if (execution.status === 'failed') {
        if (pollingRef.current) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
        }

        failTask(execution.error || 'Unknown error')
        setRunning(false)
        resetAllAgents()

        addLine({
          agent: 'system',
          message: `Workflow failed: ${execution.error}`,
          level: 'error',
          type: 'message'
        })
      }
    } catch (error) {
      console.error('Error polling execution:', error)
    }
  }, [addLine, completeTask, failTask, setRunning, setAgentStatus, resetAllAgents])

  const handleRun = async () => {
    if (!currentTask.trim() || taskStatus === 'running') return

    startTask()
    setRunning(true)
    clearOutput()
    resetAllAgents()
    startTimeRef.current = Date.now()

    addLine({
      agent: 'system',
      message: `Starting: "${currentTask}"`,
      level: 'system',
      type: 'message'
    })

    try {
      // Check if input matches a workflow name
      const workflows = await getWorkflows()
      const matchingWorkflow = workflows.find(
        w => w.id.toLowerCase() === currentTask.toLowerCase() ||
             w.name.toLowerCase().includes(currentTask.toLowerCase())
      )

      if (matchingWorkflow) {
        // Run the matching workflow
        setAgentStatus('coder', 'working', { currentTask: `Running workflow: ${matchingWorkflow.name}` })

        addLine({
          agent: 'system',
          message: `Found workflow: ${matchingWorkflow.name}`,
          level: 'info',
          type: 'message'
        })

        const response = await runWorkflow(matchingWorkflow.id, {})
        executionIdRef.current = response.execution_id

        addLine({
          agent: 'system',
          message: `Execution started: ${response.execution_id.slice(0, 8)}...`,
          level: 'info',
          type: 'message'
        })

        // Poll for execution status
        pollingRef.current = setInterval(() => {
          if (executionIdRef.current) {
            pollExecution(executionIdRef.current)
          }
        }, 500)

      } else {
        // No matching workflow - send to LLM as chat prompt
        addLine({
          agent: 'system',
          message: `Sending to LLM: "${currentTask}"`,
          level: 'info',
          type: 'message'
        })

        setAgentStatus('coder', 'working', { currentTask: 'Processing with Ollama...' })

        try {
          const response = await chat({ prompt: currentTask })

          if (response.success && response.response) {
            addLine({
              agent: 'llm',
              message: response.response,
              level: 'success',
              type: 'message'
            })

            addLine({
              agent: 'system',
              message: `Model: ${response.model} | Tokens: ${response.tokens || 'N/A'}`,
              level: 'info',
              type: 'message'
            })

            const duration = Date.now() - startTimeRef.current
            completeTask(duration)
            setAgentStatus('coder', 'completed', { duration })
          } else {
            addLine({
              agent: 'system',
              message: `LLM Error: ${response.error || 'Unknown error'}`,
              level: 'error',
              type: 'message'
            })
            failTask(response.error || 'LLM error')
            resetAllAgents()
          }
        } catch (llmError) {
          const errorMsg = llmError instanceof Error ? llmError.message : 'LLM request failed'
          addLine({
            agent: 'system',
            message: `Error: ${errorMsg}`,
            level: 'error',
            type: 'message'
          })
          failTask(errorMsg)
          resetAllAgents()
        }

        setRunning(false)
      }

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      failTask(errorMsg)
      setRunning(false)
      resetAllAgents()

      addLine({
        agent: 'system',
        message: `Error: ${errorMsg}`,
        level: 'error',
        type: 'message'
      })
    }
  }

  const handleStop = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }

    stopTask()
    setRunning(false)
    resetAllAgents()

    addLine({
      agent: 'system',
      message: 'Task cancelled by user',
      level: 'warning',
      type: 'message'
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (taskStatus === 'idle' || taskStatus === 'completed' || taskStatus === 'error') {
        handleRun()
      }
    } else if (e.key === 'Escape') {
      if (taskStatus === 'running') {
        handleStop()
      } else {
        setCurrentTask('')
      }
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      navigateHistoryUp()
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      navigateHistoryDown()
    }
  }

  const renderPrompt = () => {
    switch (taskStatus) {
      case 'running':
        return <span className="animate-spin text-[var(--blue)]">&#9680;</span>
      case 'completed':
        return <span className="text-[var(--green)]">&#10003;</span>
      case 'error':
        return <span className="text-[var(--red)]">&#10007;</span>
      default:
        return <span className="text-[var(--accent)]">&gt;</span>
    }
  }

  const renderAction = () => {
    switch (taskStatus) {
      case 'running':
        return (
          <button
            onClick={handleStop}
            className="px-4 py-1.5 text-[11px] font-medium rounded-lg cursor-pointer transition-fast glow-red"
            style={{
              background: 'var(--red-muted)',
              color: 'var(--red)',
              border: '1px solid rgba(255, 69, 58, 0.3)'
            }}
          >
            &#9632; Stop
          </button>
        )
      case 'completed':
        return (
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono" style={{ color: 'var(--green)' }}>
              {formatDuration(taskDuration)}
            </span>
            <button
              onClick={handleRun}
              className="px-4 py-1.5 text-[11px] font-medium rounded-lg cursor-pointer transition-fast glass-button"
            >
              &#8634; Rerun
            </button>
          </div>
        )
      case 'error':
        return (
          <div className="flex items-center gap-2">
            <span className="text-[11px]" style={{ color: 'var(--red)' }}>Error</span>
            <button
              onClick={handleRun}
              className="px-4 py-1.5 text-[11px] font-medium rounded-lg cursor-pointer transition-fast"
              style={{
                background: 'var(--orange)',
                color: 'var(--bg-primary)',
                boxShadow: '0 0 12px var(--orange-glow)'
              }}
            >
              &#8634; Retry
            </button>
          </div>
        )
      default:
        return (
          <button
            onClick={handleRun}
            disabled={!currentTask.trim()}
            className="px-4 py-1.5 text-[11px] font-medium rounded-lg cursor-pointer transition-fast disabled:opacity-40 disabled:cursor-not-allowed"
            style={{
              background: currentTask.trim() ? 'var(--accent-alpha-20)' : 'var(--glass-bg)',
              color: currentTask.trim() ? 'var(--accent-light)' : 'var(--text-muted)',
              border: currentTask.trim() ? '1px solid var(--accent-alpha-30)' : '1px solid var(--glass-border)',
              boxShadow: currentTask.trim() ? '0 0 16px var(--accent-glow)' : 'none'
            }}
          >
            &#9166; Run
          </button>
        )
    }
  }

  const getBorderColor = () => {
    switch (taskStatus) {
      case 'running': return 'var(--blue)'
      case 'completed': return 'var(--green)'
      case 'error': return 'var(--red)'
      default: return 'var(--border)'
    }
  }

  return (
    <div
      className="flex items-center h-12 rounded-xl mb-4 transition-fast glass focus-within:border-[var(--accent)] focus-within:shadow-[0_0_16px_var(--accent-glow)]"
      style={{
        borderColor: getBorderColor(),
        boxShadow: taskStatus === 'running'
          ? '0 0 20px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,0.05)'
          : 'inset 0 1px 0 rgba(255,255,255,0.05)'
      }}
    >
      {/* Prompt symbol */}
      <div className="px-4 flex-shrink-0 text-base">
        {renderPrompt()}
      </div>

      {/* Input field */}
      <input
        ref={inputRef}
        type="text"
        value={currentTask}
        onChange={(e) => setCurrentTask(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything or enter workflow name (e.g., 'What is AI?' or 'hello_world')"
        disabled={taskStatus === 'running'}
        className="flex-1 bg-transparent outline-none text-[13px] disabled:opacity-70 disabled:cursor-not-allowed border-0"
        style={{
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)'
        }}
      />

      {/* Action button */}
      <div className="px-3 flex items-center gap-2">
        {renderAction()}
      </div>
    </div>
  )
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}
