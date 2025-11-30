import { ReactNode, useEffect } from 'react'
import { useUIStore } from '@/stores/uiStore'

interface PopupWrapperProps {
  title: string
  children: ReactNode
  footer?: ReactNode
}

export function PopupWrapper({ title, children, footer }: PopupWrapperProps) {
  const { closePopup } = useUIStore()

  // Handle Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closePopup()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [closePopup])

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center animate-fade-in"
      style={{
        background: 'rgba(0, 0, 0, 0.6)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)'
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closePopup()
      }}
    >
      <div
        className="w-[700px] max-w-[90vw] max-h-[80vh] rounded-2xl overflow-hidden animate-scale-in flex flex-col glass-heavy"
        style={{
          boxShadow: '0 24px 48px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05)'
        }}
      >
        {/* Header with gradient border */}
        <div
          className="flex items-center justify-between px-6 py-4"
          style={{
            borderBottom: '1px solid var(--glass-border)',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.03) 0%, transparent 100%)'
          }}
        >
          <span
            className="text-[13px] font-semibold tracking-wide"
            style={{ color: 'var(--text-primary)' }}
          >
            {title}
          </span>
          <button
            onClick={closePopup}
            className="w-7 h-7 flex items-center justify-center rounded-lg cursor-pointer transition-fast glass-hover"
            style={{ color: 'var(--text-muted)' }}
          >
            <span className="text-lg">&times;</span>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {children}
        </div>

        {/* Footer with gradient border */}
        {footer && (
          <div
            className="px-6 py-4"
            style={{
              borderTop: '1px solid var(--glass-border)',
              color: 'var(--text-muted)',
              background: 'linear-gradient(0deg, rgba(255,255,255,0.02) 0%, transparent 100%)'
            }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  )
}
