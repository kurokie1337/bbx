import { useEffect, useState, useRef } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface LogEntry {
  timestamp: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  source: string
  message: string
}

export function LogsPopup() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [filter, setFilter] = useState<string>('all')
  const [autoScroll, setAutoScroll] = useState(true)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const fetchLogs = () => {
    api.get('/logs/', { params: { limit: 100 } })
      .then((res: AxiosResponse) => {
        if (res.data?.entries) {
          setLogs(res.data.entries)
        }
      })
      .catch((err) => {
        console.error('Failed to fetch logs:', err)
      })
  }

  useEffect(() => {
    // Fetch initial logs
    fetchLogs()

    // Poll for new logs every 2 seconds
    const interval = setInterval(fetchLogs, 2000)

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'var(--red)'
      case 'WARN': return 'var(--yellow)'
      case 'INFO': return 'var(--blue)'
      case 'DEBUG': return 'var(--text-muted)'
      default: return 'var(--text-muted)'
    }
  }

  const filteredLogs = filter === 'all'
    ? logs
    : logs.filter(log => log.level === filter)

  return (
    <PopupWrapper
      title="LOGS"
      footer={
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center gap-2">
            {['all', 'ERROR', 'WARN', 'INFO', 'DEBUG'].map(level => (
              <button
                key={level}
                onClick={() => setFilter(level)}
                className="text-[10px] px-2 py-0.5 rounded cursor-pointer transition-fast"
                style={{
                  background: filter === level ? 'var(--accent)' : 'var(--bg-tertiary)',
                  color: filter === level ? 'var(--bg-primary)' : 'var(--text-muted)'
                }}
              >
                {level}
              </button>
            ))}
          </div>
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className="text-[10px] px-2 py-0.5 rounded cursor-pointer"
            style={{
              background: autoScroll ? 'var(--green)' : 'var(--bg-tertiary)',
              color: autoScroll ? 'var(--bg-primary)' : 'var(--text-muted)'
            }}
          >
            {autoScroll ? 'auto-scroll on' : 'auto-scroll off'}
          </button>
        </div>
      }
    >
      <div
        className="rounded-lg border overflow-hidden font-mono text-xs"
        style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)', maxHeight: '300px', overflowY: 'auto' }}
      >
        {filteredLogs.length === 0 ? (
          <div className="p-4 text-center" style={{ color: 'var(--text-muted)' }}>
            No logs to display
          </div>
        ) : (
          <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
            {filteredLogs.map((log, i) => (
              <div
                key={i}
                className="flex items-start gap-2 px-3 py-1.5 hover:bg-[var(--bg-tertiary)] transition-fast"
              >
                <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                  {log.timestamp}
                </span>
                <span
                  className="text-[10px] w-12 flex-shrink-0 font-semibold"
                  style={{ color: getLevelColor(log.level) }}
                >
                  {log.level}
                </span>
                <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--accent)' }}>
                  [{log.source}]
                </span>
                <span className="text-[11px] break-all" style={{ color: 'var(--text-primary)' }}>
                  {log.message}
                </span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </PopupWrapper>
  )
}
