import { useUIStore } from '@/stores/uiStore'

export function Header() {
  const { connectionStatus } = useUIStore()

  const getStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'var(--green)'
      case 'disconnected': return 'var(--red)'
      case 'connecting': return 'var(--yellow)'
    }
  }

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'connected'
      case 'disconnected': return 'offline'
      case 'connecting': return 'connecting...'
    }
  }

  return (
    <header className="flex-1 flex items-center justify-between px-6">
      {/* Left section - Logo */}
      <div className="flex items-center gap-3">
        {/* Logo with glow */}
        <div className="relative">
          <span
            className="text-xl font-bold"
            style={{
              color: 'var(--accent)',
              textShadow: '0 0 20px var(--accent-glow)'
            }}
          >
            &#9670;
          </span>
        </div>

        {/* Brand */}
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-semibold tracking-wide"
            style={{ color: 'var(--text-primary)' }}
          >
            BBX
          </span>
          <span
            className="badge badge-accent text-[9px]"
          >
            console
          </span>
        </div>
      </div>

      {/* Right section - Connection status */}
      <div className="flex items-center gap-2">
        <div
          className={`w-2 h-2 rounded-full status-dot ${connectionStatus === 'connecting' ? 'animate-pulse' : ''}`}
          style={{ background: getStatusColor() }}
        />
        <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {getStatusText()}
        </span>
      </div>
    </header>
  )
}
