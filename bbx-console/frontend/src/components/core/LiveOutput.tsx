import { useRef, useEffect } from 'react'
import { useOutputStore, OutputLine } from '@/stores/outputStore'
import { useUIStore } from '@/stores/uiStore'

export function LiveOutput() {
  const { lines, isRunning } = useOutputStore()
  const { autoScrollOutput, toggleAutoScroll } = useUIStore()
  const { clearOutput } = useOutputStore()
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScrollOutput && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [lines, autoScrollOutput])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getAgentColor = (agent?: string): string => {
    switch (agent) {
      case 'architect': return 'var(--purple)'
      case 'coder': return 'var(--blue)'
      case 'reviewer': return 'var(--orange)'
      case 'tester': return 'var(--green)'
      case 'system': return 'var(--text-muted)'
      default: return 'var(--text-secondary)'
    }
  }

  const getMessagePrefix = (line: OutputLine): { symbol: string; color: string } | null => {
    switch (line.type) {
      case 'file_add':
        return { symbol: '+ ', color: 'var(--green)' }
      case 'file_mod':
        return { symbol: '~ ', color: 'var(--blue)' }
      case 'file_del':
        return { symbol: '- ', color: 'var(--red)' }
      default:
        break
    }

    switch (line.level) {
      case 'success':
        return { symbol: '\u2713 ', color: 'var(--green)' }
      case 'error':
        return { symbol: '\u2717 ', color: 'var(--red)' }
      case 'warning':
        return { symbol: '\u26A0 ', color: 'var(--yellow)' }
      default:
        return null
    }
  }

  const renderLine = (line: OutputLine) => {
    // Transition divider
    if (line.type === 'transition') {
      return (
        <div key={line.id} className="flex items-center gap-3 my-2">
          <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
          <span className="text-[10px] whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>
            {line.message}
          </span>
          <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
        </div>
      )
    }

    // Code block
    if (line.type === 'code' && line.code) {
      return (
        <div key={line.id}>
          <div className="flex py-0.5">
            <span
              className="w-[70px] flex-shrink-0 opacity-60"
              style={{ color: 'var(--text-muted)' }}
            >
              {formatTime(line.timestamp)}
            </span>
            <span
              className="w-[100px] flex-shrink-0"
              style={{ color: getAgentColor(line.agent) }}
            >
              [{line.agent}]
            </span>
            <span style={{ color: 'var(--text-primary)' }}>
              {line.message}
            </span>
          </div>
          <div
            className="ml-[170px] rounded px-3 py-2 my-1"
            style={{ background: 'var(--bg-secondary)' }}
          >
            <pre className="text-[11px] whitespace-pre-wrap">
              {line.code.split('\n').map((codeLine, i) => (
                <div key={i} className="flex">
                  <span style={{ color: 'var(--border)' }}>| </span>
                  <span style={{ color: 'var(--text-primary)' }}>{codeLine}</span>
                </div>
              ))}
            </pre>
          </div>
        </div>
      )
    }

    // Regular log line
    const prefix = getMessagePrefix(line)

    return (
      <div key={line.id} className="flex py-0.5">
        <span
          className="w-[70px] flex-shrink-0 opacity-60"
          style={{ color: 'var(--text-muted)' }}
        >
          {formatTime(line.timestamp)}
        </span>

        {line.agent && (
          <span
            className="w-[100px] flex-shrink-0"
            style={{ color: getAgentColor(line.agent) }}
          >
            [{line.agent}]
          </span>
        )}

        <span
          className="flex-1 break-words"
          style={{ color: 'var(--text-primary)' }}
        >
          {prefix && (
            <span style={{ color: prefix.color }}>{prefix.symbol}</span>
          )}
          {line.message}
        </span>
      </div>
    )
  }

  return (
    <div
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
        <span
          className="text-[10px] font-semibold tracking-widest uppercase"
          style={{ color: 'var(--text-muted)' }}
        >
          Output
        </span>

        <div className="flex items-center gap-2">
          <button
            onClick={clearOutput}
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
              background: autoScrollOutput ? 'var(--accent-alpha-10)' : 'transparent',
              border: autoScrollOutput ? '1px solid var(--accent-alpha-20)' : '1px solid transparent'
            }}
          >
            Auto-scroll
          </button>
        </div>
      </div>

      {/* Output area */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto px-4 py-3 text-[11px] leading-relaxed"
        style={{ fontFamily: 'var(--font-mono)' }}
      >
        {lines.length === 0 ? (
          <div
            className="flex flex-col items-center justify-center h-full gap-2"
            style={{ color: 'var(--text-muted)' }}
          >
            <span className="text-2xl opacity-20">&#9675;</span>
            <span className="text-[11px]">No output yet. Run a task to see results.</span>
          </div>
        ) : (
          <>
            {lines.map(line => renderLine(line))}
            {isRunning && (
              <span className="animate-pulse" style={{ color: 'var(--accent-light)' }}>
                &#9608;
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
