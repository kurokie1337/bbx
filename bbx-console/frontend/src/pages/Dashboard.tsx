import { useQuery } from '@tanstack/react-query'
import { getHealth, getAgentStats, getMemoryStats, getRingStats } from '@/services/api'
import { cn, formatBytes, formatDuration } from '@/lib/utils'
import {
  Activity,
  Users,
  Database,
  CircleDot,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
} from 'lucide-react'

export function Dashboard() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 5000,
  })

  const { data: agentStats } = useQuery({
    queryKey: ['agent-stats'],
    queryFn: getAgentStats,
    refetchInterval: 5000,
  })

  const { data: memoryStats } = useQuery({
    queryKey: ['memory-stats'],
    queryFn: getMemoryStats,
    refetchInterval: 5000,
  })

  const { data: ringStats } = useQuery({
    queryKey: ['ring-stats'],
    queryFn: getRingStats,
    refetchInterval: 1000,
  })

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* System Health */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">System Status</div>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="mt-2 flex items-center gap-2">
            {health?.status === 'healthy' ? (
              <>
                <CheckCircle className="h-5 w-5 text-green-500" />
                <span className="text-lg font-semibold text-green-500">Healthy</span>
              </>
            ) : (
              <>
                <XCircle className="h-5 w-5 text-red-500" />
                <span className="text-lg font-semibold text-red-500">Unhealthy</span>
              </>
            )}
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Uptime: {health?.uptime_seconds ? formatDuration(health.uptime_seconds * 1000) : '-'}
          </div>
        </div>

        {/* Agents */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">Agents</div>
            <Users className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="mt-2 text-2xl font-bold">
            {agentStats?.busy_agents || 0} / {agentStats?.total_agents || 0}
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            {agentStats?.queued_tasks || 0} tasks queued
          </div>
        </div>

        {/* Memory */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">Memory</div>
            <Database className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="mt-2 text-2xl font-bold">
            {memoryStats?.total_items || 0} items
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Hit rate: {((memoryStats?.hit_rate || 0) * 100).toFixed(1)}%
          </div>
        </div>

        {/* Ring */}
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">Ring</div>
            <CircleDot className="h-4 w-4 text-muted-foreground" />
          </div>
          <div className="mt-2 text-2xl font-bold">
            {ringStats?.throughput_ops_sec?.toFixed(1) || 0} ops/s
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            Latency: {ringStats?.avg_latency_ms?.toFixed(0) || 0}ms
          </div>
        </div>
      </div>

      {/* Memory Tiers */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="text-lg font-semibold mb-4">Memory Tiers</h2>
        <div className="grid gap-4 md:grid-cols-4">
          {memoryStats?.generations?.map((tier) => (
            <div key={tier.tier} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className={cn(
                  tier.tier === 'HOT' && 'text-red-400',
                  tier.tier === 'WARM' && 'text-orange-400',
                  tier.tier === 'COOL' && 'text-blue-400',
                  tier.tier === 'COLD' && 'text-cyan-400',
                )}>
                  {tier.tier}
                </span>
                <span className="text-muted-foreground">{tier.items} items</span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    'h-full transition-all',
                    tier.tier === 'HOT' && 'bg-red-500',
                    tier.tier === 'WARM' && 'bg-orange-500',
                    tier.tier === 'COOL' && 'bg-blue-500',
                    tier.tier === 'COLD' && 'bg-cyan-500',
                  )}
                  style={{ width: `${Math.min(tier.utilization * 100, 100)}%` }}
                />
              </div>
              <div className="text-xs text-muted-foreground">
                {formatBytes(tier.size_bytes)} / {formatBytes(tier.max_size_bytes)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Ring Stats */}
      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="text-lg font-semibold mb-4">Ring Performance</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">Operations</div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold">{ringStats?.operations_completed || 0}</span>
              <span className="text-sm text-green-500">completed</span>
            </div>
            <div className="flex gap-4 text-sm text-muted-foreground">
              <span>{ringStats?.operations_failed || 0} failed</span>
              <span>{ringStats?.pending_count || 0} pending</span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">Latency (ms)</div>
            <div className="flex items-baseline gap-4">
              <div>
                <span className="text-xl font-bold">{ringStats?.p50_latency_ms?.toFixed(0) || 0}</span>
                <span className="text-xs text-muted-foreground ml-1">p50</span>
              </div>
              <div>
                <span className="text-xl font-bold">{ringStats?.p95_latency_ms?.toFixed(0) || 0}</span>
                <span className="text-xs text-muted-foreground ml-1">p95</span>
              </div>
              <div>
                <span className="text-xl font-bold">{ringStats?.p99_latency_ms?.toFixed(0) || 0}</span>
                <span className="text-xs text-muted-foreground ml-1">p99</span>
              </div>
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-sm text-muted-foreground">Workers</div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold">{ringStats?.active_workers || 0}</span>
              <span className="text-sm text-muted-foreground">/ {ringStats?.worker_pool_size || 0}</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {ringStats?.worker_utilization?.toFixed(1) || 0}% utilization
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
