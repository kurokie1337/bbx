import { useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { useTaskStore, TaskHistoryItem } from '@/stores/taskStore'

export function HistoryPopup() {
  const { history, setCurrentTask } = useTaskStore()
  const [selectedItem, setSelectedItem] = useState<TaskHistoryItem | null>(null)

  const formatDuration = (ms?: number) => {
    if (!ms) return '-'
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const formatRelativeTime = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - new Date(date).getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 1) return 'just now'
    if (minutes < 60) return `${minutes} min ago`
    if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`
    if (days === 1) return 'yesterday'
    return `${days} days ago`
  }

  const groupByDay = (items: TaskHistoryItem[]) => {
    const groups: { [key: string]: TaskHistoryItem[] } = {}

    items.forEach(item => {
      const date = new Date(item.startedAt)
      const today = new Date()
      const yesterday = new Date(today)
      yesterday.setDate(yesterday.getDate() - 1)

      let key: string
      if (date.toDateString() === today.toDateString()) {
        key = 'TODAY'
      } else if (date.toDateString() === yesterday.toDateString()) {
        key = 'YESTERDAY'
      } else {
        key = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase()
      }

      if (!groups[key]) groups[key] = []
      groups[key].push(item)
    })

    return groups
  }

  const groups = groupByDay(history)

  // Detail view
  if (selectedItem) {
    return (
      <PopupWrapper
        title="TASK DETAILS"
        footer={
          <div className="flex gap-2">
            <button
              onClick={() => {
                setCurrentTask(selectedItem.task)
                setSelectedItem(null)
              }}
              className="px-3 py-1 text-xs rounded cursor-pointer"
              style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
            >
              Rerun
            </button>
          </div>
        }
      >
        {/* Back button */}
        <button
          onClick={() => setSelectedItem(null)}
          className="flex items-center gap-1 text-xs mb-4 cursor-pointer hover:underline"
          style={{ color: 'var(--accent)' }}
        >
          <span>&#8592;</span> Back to list
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
                  background: selectedItem.status === 'completed' ? 'var(--green)' : 'var(--red)',
                  color: 'var(--bg-primary)'
                }}
              >
                {selectedItem.status}
              </span>
            </div>
            <div className="text-sm" style={{ color: 'var(--text-primary)' }}>
              "{selectedItem.task}"
            </div>
          </div>

          {/* Timing */}
          <div className="grid grid-cols-2 gap-3">
            <div
              className="rounded-lg border p-3"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>DURATION</div>
              <div className="text-lg font-mono" style={{ color: 'var(--text-primary)' }}>
                {formatDuration(selectedItem.duration)}
              </div>
            </div>
            <div
              className="rounded-lg border p-3"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>STARTED</div>
              <div className="text-sm" style={{ color: 'var(--text-primary)' }}>
                {formatRelativeTime(selectedItem.startedAt)}
              </div>
            </div>
          </div>

          {/* Error if any */}
          {selectedItem.error && (
            <div
              className="rounded-lg border p-3"
              style={{ background: 'var(--red-alpha-10)', borderColor: 'var(--red)' }}
            >
              <div className="text-[10px] mb-1" style={{ color: 'var(--red)' }}>ERROR</div>
              <div className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
                {selectedItem.error}
              </div>
            </div>
          )}
        </div>
      </PopupWrapper>
    )
  }

  // List view
  return (
    <PopupWrapper
      title="HISTORY"
      footer={
        <div className="text-[11px]">
          {history.length} tasks | Click to view details
        </div>
      }
    >
      {history.length === 0 ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          No history yet
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(groups).map(([day, items]) => (
            <div key={day}>
              <div
                className="text-[10px] font-semibold tracking-wider mb-2"
                style={{ color: 'var(--text-muted)' }}
              >
                {day}
              </div>
              <div
                className="rounded-lg border overflow-hidden"
                style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
              >
                {items.map((item, i) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between px-3 py-2 cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
                    style={{ borderBottom: i < items.length - 1 ? '1px solid var(--border)' : 'none' }}
                    onClick={() => setSelectedItem(item)}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="text-xs"
                        style={{ color: item.status === 'completed' ? 'var(--green)' : 'var(--red)' }}
                      >
                        {item.status === 'completed' ? '\u2713' : '\u2717'}
                      </span>
                      <span className="text-xs truncate max-w-[280px]" style={{ color: 'var(--text-primary)' }}>
                        {item.task}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-[11px]">
                      <span style={{ color: 'var(--text-muted)' }}>
                        {formatDuration(item.duration)}
                      </span>
                      <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </PopupWrapper>
  )
}
