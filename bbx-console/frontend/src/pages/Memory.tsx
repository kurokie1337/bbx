import { useQuery } from '@tanstack/react-query'
import { getMemoryStats } from '@/services/api'
import { cn, formatBytes } from '@/lib/utils'
import { Database, Flame, Sun, Snowflake, Archive } from 'lucide-react'

const tierIcons = {
  HOT: Flame,
  WARM: Sun,
  COOL: Snowflake,
  COLD: Archive,
}

const tierColors = {
  HOT: 'text-red-500 bg-red-500/20',
  WARM: 'text-orange-500 bg-orange-500/20',
  COOL: 'text-blue-500 bg-blue-500/20',
  COLD: 'text-cyan-500 bg-cyan-500/20',
}

export function Memory() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['memory-stats'],
    queryFn: getMemoryStats,
    refetchInterval: 5000,
  })

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-muted-foreground">Loading memory stats...</div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Memory (Context Tiering)</h1>
        <div className="text-sm text-muted-foreground">
          MGLRU-inspired multi-generation memory management
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">Total Items</div>
          <div className="text-2xl font-bold">{stats?.total_items || 0}</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">Total Size</div>
          <div className="text-2xl font-bold">{formatBytes(stats?.total_size_bytes || 0)}</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">Hit Rate</div>
          <div className="text-2xl font-bold text-green-500">
            {((stats?.hit_rate || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm text-muted-foreground">Promotions / Demotions</div>
          <div className="text-2xl font-bold">
            <span className="text-green-500">{stats?.promotions || 0}</span>
            {' / '}
            <span className="text-yellow-500">{stats?.demotions || 0}</span>
          </div>
        </div>
      </div>

      {/* Tier Visualization */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-6">Memory Tiers</h2>
        <div className="space-y-6">
          {stats?.generations?.map((tier) => {
            const Icon = tierIcons[tier.tier as keyof typeof tierIcons]
            const colorClass = tierColors[tier.tier as keyof typeof tierColors]

            return (
              <div key={tier.tier} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={cn('p-2 rounded-lg', colorClass.split(' ')[1])}>
                      <Icon className={cn('h-5 w-5', colorClass.split(' ')[0])} />
                    </div>
                    <div>
                      <div className="font-semibold">{tier.tier}</div>
                      <div className="text-sm text-muted-foreground">
                        {tier.tier === 'HOT' && 'In-memory, uncompressed'}
                        {tier.tier === 'WARM' && 'In-memory, compressed'}
                        {tier.tier === 'COOL' && 'On disk, compressed'}
                        {tier.tier === 'COLD' && 'Archive storage'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{tier.items} items</div>
                    <div className="text-sm text-muted-foreground">
                      {formatBytes(tier.size_bytes)} / {formatBytes(tier.max_size_bytes)}
                    </div>
                  </div>
                </div>
                <div className="h-4 rounded-full bg-muted overflow-hidden">
                  <div
                    className={cn(
                      'h-full transition-all rounded-full',
                      tier.tier === 'HOT' && 'bg-red-500',
                      tier.tier === 'WARM' && 'bg-orange-500',
                      tier.tier === 'COOL' && 'bg-blue-500',
                      tier.tier === 'COLD' && 'bg-cyan-500',
                    )}
                    style={{ width: `${Math.max(tier.utilization * 100, 1)}%` }}
                  />
                </div>
                <div className="text-sm text-muted-foreground text-right">
                  {(tier.utilization * 100).toFixed(2)}% utilized
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Flow Diagram */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">Data Flow</h2>
        <div className="flex items-center justify-center gap-4">
          {['HOT', 'WARM', 'COOL', 'COLD'].map((tier, idx) => {
            const Icon = tierIcons[tier as keyof typeof tierIcons]
            const colorClass = tierColors[tier as keyof typeof tierColors]

            return (
              <div key={tier} className="flex items-center gap-4">
                <div className={cn('p-4 rounded-lg', colorClass.split(' ')[1])}>
                  <Icon className={cn('h-8 w-8', colorClass.split(' ')[0])} />
                  <div className="text-center mt-2 text-sm font-semibold">{tier}</div>
                </div>
                {idx < 3 && (
                  <div className="flex flex-col items-center gap-1">
                    <span className="text-xs text-muted-foreground">demote</span>
                    <div className="text-muted-foreground">→</div>
                    <span className="text-xs text-green-500">promote</span>
                    <div className="text-green-500">←</div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
