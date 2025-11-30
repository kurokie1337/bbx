import { useEffect, useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface MemoryItem {
  key: string
  size: number
  access_count: number
  last_accessed: string
}

interface MemoryTier {
  tier: string
  items: number
  size_bytes: number
  max_size_bytes: number
  utilization: number
}

interface MemoryStats {
  generations: MemoryTier[]
  total_items: number
  total_size_bytes: number
  hit_rate: number
  promotions: number
  demotions: number
}

export function MemoryPopup() {
  const [stats, setStats] = useState<MemoryStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTier, setSelectedTier] = useState<MemoryTier | null>(null)
  const [tierItems, setTierItems] = useState<MemoryItem[]>([])

  useEffect(() => {
    api.get('/memory/stats')
      .then((res: AxiosResponse) => {
        setStats(res.data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const handleTierClick = (tier: MemoryTier) => {
    setSelectedTier(tier)
    // Fetch real tier items from API
    api.get(`/memory/tiers/${tier.tier.toLowerCase()}`)
      .then((res: AxiosResponse) => {
        if (res.data?.items && Array.isArray(res.data.items)) {
          setTierItems(res.data.items)
        } else {
          setTierItems([])
        }
      })
      .catch((err) => {
        console.error(`Failed to fetch ${tier.tier} tier items:`, err)
        setTierItems([])
      })
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'HOT': return 'var(--red)'
      case 'WARM': return 'var(--orange)'
      case 'COOL': return 'var(--blue)'
      case 'COLD': return 'var(--text-muted)'
      default: return 'var(--text-muted)'
    }
  }

  // Detail view for selected tier
  if (selectedTier) {
    return (
      <PopupWrapper
        title={`${selectedTier.tier} TIER`}
        footer={
          <div className="flex items-center gap-4 text-[11px]">
            <span>{selectedTier.items} items</span>
            <span>|</span>
            <span>{formatBytes(selectedTier.size_bytes)} used</span>
          </div>
        }
      >
        {/* Back button */}
        <button
          onClick={() => setSelectedTier(null)}
          className="flex items-center gap-1 text-xs mb-4 cursor-pointer hover:underline"
          style={{ color: 'var(--accent)' }}
        >
          <span>&#8592;</span> Back to overview
        </button>

        <div className="space-y-4">
          {/* Tier stats */}
          <div className="grid grid-cols-3 gap-3">
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: getTierColor(selectedTier.tier) }}>
                {selectedTier.items}
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Items</div>
            </div>
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                {formatBytes(selectedTier.size_bytes)}
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Used</div>
            </div>
            <div
              className="rounded-lg border p-3 text-center"
              style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
            >
              <div className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
                {(selectedTier.utilization * 100).toFixed(1)}%
              </div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Utilization</div>
            </div>
          </div>

          {/* Items list */}
          <div
            className="rounded-lg border overflow-hidden"
            style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <div className="px-3 py-2 border-b" style={{ borderColor: 'var(--border)' }}>
              <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>
                ITEMS IN {selectedTier.tier}
              </span>
            </div>
            {tierItems.length > 0 ? tierItems.map((item, i) => (
              <div
                key={item.key}
                className="flex items-center justify-between px-3 py-2"
                style={{ borderBottom: i < tierItems.length - 1 ? '1px solid var(--border)' : 'none' }}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-[10px]" style={{ color: getTierColor(selectedTier.tier) }}>&#9679;</span>
                  <span className="text-xs truncate font-mono" style={{ color: 'var(--text-primary)' }}>
                    {item.key}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <span>{formatBytes(item.size)}</span>
                  <span>{item.access_count}x</span>
                  <span>{item.last_accessed}</span>
                </div>
              </div>
            )) : (
              <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--text-muted)' }}>
                No items in this tier
              </div>
            )}
          </div>
        </div>
      </PopupWrapper>
    )
  }

  // List view
  return (
    <PopupWrapper
      title="MEMORY"
      footer={stats && (
        <div className="flex items-center gap-4 text-[11px]">
          <span>Stats: {stats.total_items} items</span>
          <span>|</span>
          <span>Hit rate: {(stats.hit_rate * 100).toFixed(0)}%</span>
          <span>|</span>
          <span>Size: {formatBytes(stats.total_size_bytes)}</span>
        </div>
      )}
    >
      {loading ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Loading...
        </div>
      ) : stats ? (
        <div className="space-y-4">
          {stats.generations.map(tier => (
            <div
              key={tier.tier}
              className="rounded-lg border p-4 cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
              style={{
                background: 'var(--bg-secondary)',
                borderColor: 'var(--border)'
              }}
              onClick={() => handleTierClick(tier)}
            >
              <div className="flex items-center justify-between mb-3">
                <span
                  className="text-sm font-semibold"
                  style={{ color: getTierColor(tier.tier) }}
                >
                  {tier.tier} ({tier.items} items)
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {formatBytes(tier.size_bytes)} / {formatBytes(tier.max_size_bytes)}
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
                </div>
              </div>

              {/* Progress bar */}
              <div
                className="h-2 rounded-full overflow-hidden"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(tier.utilization * 100, 100)}%`,
                    background: getTierColor(tier.tier)
                  }}
                />
              </div>

              <div className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
                {(tier.utilization * 100).toFixed(2)}% utilized
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Failed to load memory stats
        </div>
      )}
    </PopupWrapper>
  )
}
