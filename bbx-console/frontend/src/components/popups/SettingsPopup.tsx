import { useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { useUIStore } from '@/stores/uiStore'

export function SettingsPopup() {
  const { autoScrollOutput, toggleAutoScroll } = useUIStore()
  const [apiUrl, setApiUrl] = useState(window.location.origin)

  return (
    <PopupWrapper
      title="SETTINGS"
      footer={
        <div className="text-[11px]">
          BBX Console v2.0
        </div>
      }
    >
      <div className="space-y-6">
        {/* General */}
        <Section title="GENERAL">
          <SettingRow
            label="Auto-scroll output"
            description="Automatically scroll to new output lines"
          >
            <Toggle checked={autoScrollOutput} onChange={toggleAutoScroll} />
          </SettingRow>
        </Section>

        {/* Connection */}
        <Section title="CONNECTION">
          <SettingRow
            label="API URL"
            description="Backend server URL"
          >
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-64 px-3 py-1.5 rounded border text-sm"
              style={{
                background: 'var(--bg-tertiary)',
                borderColor: 'var(--border)',
                color: 'var(--text-primary)'
              }}
            />
          </SettingRow>
        </Section>

        {/* Keyboard Shortcuts */}
        <Section title="KEYBOARD SHORTCUTS">
          <div className="grid grid-cols-2 gap-2">
            <ShortcutRow shortcut="\u2318K" description="Open command palette" />
            <ShortcutRow shortcut="\u2318Enter" description="Run task" />
            <ShortcutRow shortcut="\u2318." description="Stop current task" />
            <ShortcutRow shortcut="Escape" description="Close popup / Clear" />
            <ShortcutRow shortcut="\u2318M" description="Open memory" />
            <ShortcutRow shortcut="\u2318A" description="Open agents" />
            <ShortcutRow shortcut="\u2318R" description="Open ring" />
            <ShortcutRow shortcut="\u2318H" description="Open history" />
            <ShortcutRow shortcut="\u2318L" description="Clear output" />
            <ShortcutRow shortcut="\u2318," description="Open settings" />
          </div>
        </Section>

        {/* About */}
        <Section title="ABOUT">
          <div className="text-xs space-y-1" style={{ color: 'var(--text-muted)' }}>
            <p>BBX Console - Terminal UI for AI Agent Orchestration</p>
            <p>Built with React, TypeScript, and Tailwind CSS</p>
            <p>Backend: FastAPI + WebSocket</p>
          </div>
        </Section>
      </div>
    </PopupWrapper>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div
        className="text-[10px] font-semibold tracking-wider mb-3"
        style={{ color: 'var(--text-muted)' }}
      >
        {title}
      </div>
      {children}
    </div>
  )
}

function SettingRow({
  label,
  description,
  children
}: {
  label: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <div
      className="flex items-center justify-between py-3 border-b"
      style={{ borderColor: 'var(--border)' }}
    >
      <div>
        <div className="text-sm" style={{ color: 'var(--text-primary)' }}>
          {label}
        </div>
        {description && (
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {description}
          </div>
        )}
      </div>
      {children}
    </div>
  )
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      onClick={onChange}
      className="w-10 h-5 rounded-full relative transition-all cursor-pointer"
      style={{
        background: checked ? 'var(--accent)' : 'var(--bg-tertiary)'
      }}
    >
      <div
        className="w-4 h-4 rounded-full absolute top-0.5 transition-all"
        style={{
          background: 'var(--text-primary)',
          left: checked ? '22px' : '2px'
        }}
      />
    </button>
  )
}

function ShortcutRow({ shortcut, description }: { shortcut: string; description: string }) {
  return (
    <div className="flex items-center gap-3 py-1">
      <span
        className="px-2 py-0.5 rounded text-xs min-w-[60px] text-center"
        style={{
          background: 'var(--bg-tertiary)',
          color: 'var(--text-secondary)'
        }}
      >
        {shortcut}
      </span>
      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
        {description}
      </span>
    </div>
  )
}
