import { useState, useEffect, useMemo } from 'react'
import { useUIStore } from '@/stores/uiStore'
import { useTaskStore } from '@/stores/taskStore'
import { useOutputStore } from '@/stores/outputStore'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface Workflow {
  id: string
  name: string
  file_path: string
  description?: string
}

interface RecentItem {
  id: string
  task: string
  time: string
}

interface Command {
  id: string
  name: string
  description: string
  shortcut?: string
  action: () => void
}

export function SidePanel() {
  const { sidePanelOpen, toggleSidePanel, openPopup } = useUIStore()
  const { history, setCurrentTask } = useTaskStore()
  const { clearOutput } = useOutputStore()

  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [expandedSections, setExpandedSections] = useState({
    workflows: true,
    recent: true,
    commands: false
  })

  // Fetch workflows
  useEffect(() => {
    api.get('/workflows/').then((res: AxiosResponse) => {
      setWorkflows(res.data)
    }).catch(() => {})
  }, [])

  // Commands
  const commands: Command[] = useMemo(() => [
    {
      id: 'memory',
      name: 'Memory',
      description: 'View memory tiers',
      shortcut: '\u2318M',
      action: () => openPopup('memory')
    },
    {
      id: 'agents',
      name: 'Agents',
      description: 'Agent details',
      shortcut: '\u2318A',
      action: () => openPopup('agents')
    },
    {
      id: 'ring',
      name: 'Ring',
      description: 'Queue status',
      shortcut: '\u2318R',
      action: () => openPopup('ring')
    },
    {
      id: 'logs',
      name: 'Logs',
      description: 'Real-time logs',
      action: () => openPopup('logs')
    },
    {
      id: 'state',
      name: 'State',
      description: 'Persistent state',
      action: () => openPopup('state')
    },
    {
      id: 'history',
      name: 'History',
      description: 'Past runs',
      shortcut: '\u2318H',
      action: () => openPopup('history')
    },
    {
      id: 'settings',
      name: 'Settings',
      description: 'Configuration',
      shortcut: '\u2318,',
      action: () => openPopup('settings')
    },
    {
      id: 'clear',
      name: 'Clear',
      description: 'Clear output',
      shortcut: '\u2318L',
      action: () => clearOutput()
    },
  ], [openPopup, clearOutput])

  // Recent items
  const recentItems: RecentItem[] = useMemo(() =>
    history.slice(0, 8).map(item => ({
      id: item.id,
      task: item.task,
      time: formatRelativeTime(item.completedAt || item.startedAt)
    }))
  , [history])

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  const handleWorkflowClick = (workflow: Workflow) => {
    setCurrentTask(workflow.id)
  }

  const handleRecentClick = (item: RecentItem) => {
    setCurrentTask(item.task)
  }

  if (!sidePanelOpen) {
    return (
      <div
        className="w-10 flex-shrink-0 border-l flex flex-col items-center py-4 cursor-pointer glass-hover transition-fast"
        style={{ borderColor: 'var(--glass-border)' }}
        onClick={toggleSidePanel}
        title="Open Explorer (Cmd+K)"
      >
        <span style={{ color: 'var(--text-muted)', writingMode: 'vertical-rl', transform: 'rotate(180deg)', fontSize: '10px', letterSpacing: '2px', fontWeight: 500 }}>
          EXPLORER
        </span>
      </div>
    )
  }

  return (
    <div
      className="w-[260px] flex-shrink-0 border-l flex flex-col overflow-hidden glass"
      style={{ borderColor: 'var(--glass-border)' }}
    >
      {/* Header */}
      <div
        className="h-11 flex items-center justify-between px-4 border-b flex-shrink-0"
        style={{
          borderColor: 'var(--glass-border)',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%)'
        }}
      >
        <span className="text-[10px] font-semibold tracking-widest uppercase" style={{ color: 'var(--text-muted)' }}>
          Explorer
        </span>
        <button
          onClick={toggleSidePanel}
          className="w-6 h-6 flex items-center justify-center rounded-md glass-hover transition-fast cursor-pointer"
          style={{ color: 'var(--text-muted)' }}
          title="Close panel"
        >
          &times;
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Workflows Section */}
        <Section
          title="WORKFLOWS"
          count={workflows.length}
          expanded={expandedSections.workflows}
          onToggle={() => toggleSection('workflows')}
        >
          {workflows.map(workflow => (
            <SectionItem
              key={workflow.id}
              icon="\u25B8"
              text={workflow.name}
              hint={workflow.description}
              onClick={() => handleWorkflowClick(workflow)}
            />
          ))}
          {workflows.length === 0 && (
            <div className="px-4 py-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              No workflows found
            </div>
          )}
        </Section>

        {/* Recent Section */}
        <Section
          title="RECENT"
          count={recentItems.length}
          expanded={expandedSections.recent}
          onToggle={() => toggleSection('recent')}
        >
          {recentItems.map(item => (
            <SectionItem
              key={item.id}
              icon="\u25CB"
              text={item.task}
              hint={item.time}
              onClick={() => handleRecentClick(item)}
            />
          ))}
          {recentItems.length === 0 && (
            <div className="px-4 py-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              No recent tasks
            </div>
          )}
        </Section>

        {/* Commands Section */}
        <Section
          title="COMMANDS"
          count={commands.length}
          expanded={expandedSections.commands}
          onToggle={() => toggleSection('commands')}
        >
          {commands.map(cmd => (
            <SectionItem
              key={cmd.id}
              icon="\u25B8"
              text={cmd.name}
              hint={cmd.shortcut}
              onClick={cmd.action}
            />
          ))}
        </Section>
      </div>

    </div>
  )
}

function Section({
  title,
  count,
  expanded,
  onToggle,
  children
}: {
  title: string
  count: number
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div className="border-b" style={{ borderColor: 'var(--border)' }}>
      <div
        className="flex items-center justify-between px-3 py-1.5 cursor-pointer hover:bg-[var(--bg-secondary)] transition-fast"
        onClick={onToggle}
      >
        <div className="flex items-center gap-1.5">
          <span
            className="text-[9px] transition-transform"
            style={{
              color: 'var(--text-muted)',
              transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)'
            }}
          >
            &#9654;
          </span>
          <span className="text-[10px] font-semibold tracking-wider" style={{ color: 'var(--text-muted)' }}>
            {title}
          </span>
        </div>
        <span
          className="text-[9px] px-1 rounded"
          style={{ color: 'var(--text-muted)', background: 'var(--bg-tertiary)' }}
        >
          {count}
        </span>
      </div>
      {expanded && (
        <div className="pb-1">
          {children}
        </div>
      )}
    </div>
  )
}

function SectionItem({
  icon,
  text,
  hint,
  onClick
}: {
  icon: string
  text: string
  hint?: string
  onClick: () => void
}) {
  return (
    <div
      className="flex items-center justify-between px-3 py-1 cursor-pointer hover:bg-[var(--bg-secondary)] transition-fast group"
      onClick={onClick}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span
          className="text-[10px] flex-shrink-0 status-dot"
          style={{ color: 'var(--accent)' }}
        >
          {icon}
        </span>
        <span
          className="text-[11px] truncate group-hover:text-[var(--text-primary)]"
          style={{ color: 'var(--text-secondary)' }}
          title={text}
        >
          {text}
        </span>
      </div>
      {hint && (
        <span
          className="text-[9px] flex-shrink-0 ml-2 px-1 rounded"
          style={{ color: 'var(--text-muted)', background: 'var(--bg-tertiary)' }}
        >
          {hint}
        </span>
      )}
    </div>
  )
}

function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - new Date(date).getTime()
  const minutes = Math.floor(diff / 60000)
  const hours = Math.floor(diff / 3600000)
  const days = Math.floor(diff / 86400000)

  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  if (hours < 24) return `${hours}h`
  if (days === 1) return '1d'
  return `${days}d`
}
