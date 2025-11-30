import { useState, useEffect, useRef, useMemo } from 'react'
import { useUIStore } from '@/stores/uiStore'
import { useTaskStore } from '@/stores/taskStore'
import { useOutputStore } from '@/stores/outputStore'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface Command {
  id: string
  name: string
  description: string
  shortcut?: string
  action: () => void
}

interface Workflow {
  id: string
  name: string
  file_path: string
}

interface RecentItem {
  id: string
  task: string
  time: string
}

export function CommandPalette() {
  const { commandPaletteOpen, closeCommandPalette, openPopup } = useUIStore()
  const { history, setCurrentTask } = useTaskStore()
  const { clearOutput } = useOutputStore()

  const [search, setSearch] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  // Fetch workflows
  useEffect(() => {
    if (commandPaletteOpen) {
      api.get('/workflows/').then((res: AxiosResponse) => {
        setWorkflows(res.data.slice(0, 5))
      }).catch(() => {})
    }
  }, [commandPaletteOpen])

  // Focus input when opened
  useEffect(() => {
    if (commandPaletteOpen) {
      setSearch('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [commandPaletteOpen])

  // Commands list
  const commands: Command[] = useMemo(() => [
    {
      id: 'memory',
      name: 'memory',
      description: 'View memory tiers',
      shortcut: '\u2318M',
      action: () => { closeCommandPalette(); openPopup('memory') }
    },
    {
      id: 'agents',
      name: 'agents',
      description: 'Agent details',
      shortcut: '\u2318A',
      action: () => { closeCommandPalette(); openPopup('agents') }
    },
    {
      id: 'ring',
      name: 'ring',
      description: 'Queue status',
      shortcut: '\u2318R',
      action: () => { closeCommandPalette(); openPopup('ring') }
    },
    {
      id: 'history',
      name: 'history',
      description: 'Past runs',
      shortcut: '\u2318H',
      action: () => { closeCommandPalette(); openPopup('history') }
    },
    {
      id: 'settings',
      name: 'settings',
      description: 'Configuration',
      shortcut: '\u2318,',
      action: () => { closeCommandPalette(); openPopup('settings') }
    },
    {
      id: 'clear',
      name: 'clear',
      description: 'Clear output',
      shortcut: '\u2318L',
      action: () => { closeCommandPalette(); clearOutput() }
    },
  ], [closeCommandPalette, openPopup, clearOutput])

  // Recent items
  const recentItems: RecentItem[] = useMemo(() =>
    history.slice(0, 5).map(item => ({
      id: item.id,
      task: item.task,
      time: formatRelativeTime(item.completedAt || item.startedAt)
    }))
  , [history])

  // Filter results
  const filteredWorkflows = useMemo(() =>
    workflows.filter(w =>
      w.name.toLowerCase().includes(search.toLowerCase())
    )
  , [workflows, search])

  const filteredRecent = useMemo(() =>
    recentItems.filter(r =>
      r.task.toLowerCase().includes(search.toLowerCase())
    )
  , [recentItems, search])

  const filteredCommands = useMemo(() =>
    commands.filter(c =>
      c.name.toLowerCase().includes(search.toLowerCase()) ||
      c.description.toLowerCase().includes(search.toLowerCase())
    )
  , [commands, search])

  // All items flat list for keyboard navigation
  const allItems = useMemo(() => [
    ...filteredWorkflows.map(w => ({ type: 'workflow' as const, item: w })),
    ...filteredRecent.map(r => ({ type: 'recent' as const, item: r })),
    ...filteredCommands.map(c => ({ type: 'command' as const, item: c })),
  ], [filteredWorkflows, filteredRecent, filteredCommands])

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex(prev => Math.min(prev + 1, allItems.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex(prev => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const selected = allItems[selectedIndex]
      if (selected) {
        handleSelect(selected)
      }
    } else if (e.key === 'Escape') {
      closeCommandPalette()
    }
  }

  const handleSelect = (item: typeof allItems[number]) => {
    switch (item.type) {
      case 'workflow':
        closeCommandPalette()
        // TODO: Run workflow
        break
      case 'recent':
        closeCommandPalette()
        setCurrentTask((item.item as RecentItem).task)
        break
      case 'command':
        (item.item as Command).action()
        break
    }
  }

  if (!commandPaletteOpen) return null

  let itemIndex = 0

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end items-start pt-16 pr-6 animate-fade-in"
      style={{ background: 'rgba(0, 0, 0, 0.4)', backdropFilter: 'blur(2px)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeCommandPalette()
      }}
    >
      <div
        className="w-[360px] max-h-[calc(100vh-120px)] rounded-lg border overflow-hidden animate-slide-in flex flex-col"
        style={{
          background: 'var(--bg-primary)',
          borderColor: 'var(--border)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
        }}
      >
        {/* Search input */}
        <div className="p-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <div className="flex items-center gap-2">
            <span style={{ color: 'var(--accent)' }}>&gt;</span>
            <input
              ref={inputRef}
              type="text"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setSelectedIndex(0)
              }}
              onKeyDown={handleKeyDown}
              placeholder="Type a command or search..."
              className="flex-1 bg-transparent outline-none text-[15px]"
              style={{
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)'
              }}
            />
          </div>
        </div>

        {/* Results */}
        <div className="flex-1 overflow-y-auto p-2">
          {/* Workflows */}
          {filteredWorkflows.length > 0 && (
            <div className="mb-4">
              <div
                className="text-[10px] font-semibold tracking-wider px-3 py-2"
                style={{ color: 'var(--text-muted)' }}
              >
                WORKFLOWS
              </div>
              {filteredWorkflows.map(workflow => {
                const currentIndex = itemIndex++
                return (
                  <ResultItem
                    key={workflow.id}
                    icon="\u25B8"
                    text={workflow.name}
                    hint="\u23CE to run"
                    selected={selectedIndex === currentIndex}
                    onClick={() => handleSelect({ type: 'workflow', item: workflow })}
                  />
                )
              })}
            </div>
          )}

          {/* Recent */}
          {filteredRecent.length > 0 && (
            <div className="mb-4">
              <div
                className="text-[10px] font-semibold tracking-wider px-3 py-2"
                style={{ color: 'var(--text-muted)' }}
              >
                RECENT
              </div>
              {filteredRecent.map(item => {
                const currentIndex = itemIndex++
                return (
                  <ResultItem
                    key={item.id}
                    icon="\u25B8"
                    text={`"${item.task}"`}
                    hint={item.time}
                    selected={selectedIndex === currentIndex}
                    onClick={() => handleSelect({ type: 'recent', item })}
                  />
                )
              })}
            </div>
          )}

          {/* Commands */}
          {filteredCommands.length > 0 && (
            <div className="mb-4">
              <div
                className="text-[10px] font-semibold tracking-wider px-3 py-2"
                style={{ color: 'var(--text-muted)' }}
              >
                COMMANDS
              </div>
              {filteredCommands.map(cmd => {
                const currentIndex = itemIndex++
                return (
                  <ResultItem
                    key={cmd.id}
                    icon="\u25B8"
                    text={cmd.name}
                    description={cmd.description}
                    hint={cmd.shortcut}
                    selected={selectedIndex === currentIndex}
                    onClick={cmd.action}
                  />
                )
              })}
            </div>
          )}

          {allItems.length === 0 && (
            <div
              className="flex items-center justify-center py-8"
              style={{ color: 'var(--text-muted)' }}
            >
              No results found
            </div>
          )}
        </div>

        {/* Keyboard hints */}
        <div
          className="flex items-center gap-4 px-4 py-2 border-t"
          style={{ borderColor: 'var(--border)' }}
        >
          <KeyHint keys={['\u2191', '\u2193']} text="navigate" />
          <KeyHint keys={['\u23CE']} text="select" />
          <KeyHint keys={['esc']} text="close" />
        </div>
      </div>
    </div>
  )
}

function ResultItem({
  icon,
  text,
  description,
  hint,
  selected,
  onClick
}: {
  icon: string
  text: string
  description?: string
  hint?: string
  selected: boolean
  onClick: () => void
}) {
  return (
    <div
      className="flex items-center justify-between px-3 py-2 rounded-md cursor-pointer transition-fast"
      style={{
        background: selected ? 'var(--accent-alpha-10)' : 'transparent',
        border: selected ? '1px solid var(--accent-alpha-30)' : '1px solid transparent'
      }}
      onClick={onClick}
    >
      <div className="flex items-center gap-2">
        <span style={{ color: 'var(--text-muted)' }}>{icon}</span>
        <span className="text-[13px]" style={{ color: 'var(--text-primary)' }}>
          {text}
        </span>
        {description && (
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            {description}
          </span>
        )}
      </div>
      {hint && (
        <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {hint}
        </span>
      )}
    </div>
  )
}

function KeyHint({ keys, text }: { keys: string[]; text: string }) {
  return (
    <div className="flex items-center gap-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>
      {keys.map((key, i) => (
        <span
          key={i}
          className="px-1.5 py-0.5 rounded"
          style={{ background: 'var(--bg-secondary)' }}
        >
          {key}
        </span>
      ))}
      <span className="ml-1">{text}</span>
    </div>
  )
}

function formatRelativeTime(date: Date): string {
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
