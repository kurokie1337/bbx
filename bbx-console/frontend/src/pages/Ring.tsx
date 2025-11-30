import { useQuery } from '@tanstack/react-query'
import { getRingStats } from '@/services/api'
import { formatDuration } from '@/lib/utils'
import { CircleDot, Activity, Clock, Users, Inbox, CheckSquare } from 'lucide-react'

export function Ring() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['ring-stats'],
    queryFn: getRingStats,
    refetchInterval: 1000,
  })

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-muted-foreground">Loading ring stats...</div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">AgentRing</h1>
        <div className="text-sm text-muted-foreground">
          io_uring-inspired batch operation system
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Activity className="h-4 w-4" />
            Throughput
          </div>
          <div className="text-2xl font-bold">
            {stats?.throughput_ops_sec?.toFixed(1) || 0}
          </div>
          <div className="text-xs text-muted-foreground">ops/sec</div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="h-4 w-4" />
            Avg Latency
          </div>
          <div className="text-2xl font-bold">
            {stats?.avg_latency_ms?.toFixed(0) || 0}
          </div>
          <div className="text-xs text-muted-foreground">ms</div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Inbox className="h-4 w-4" />
            Pending
          </div>
          <div className="text-2xl font-bold text-yellow-500">
            {stats?.pending_count || 0}
          </div>
          <div className="text-xs text-muted-foreground">operations</div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <CircleDot className="h-4 w-4" />
            Processing
          </div>
          <div className="text-2xl font-bold text-blue-500">
            {stats?.processing_count || 0}
          </div>
          <div className="text-xs text-muted-foreground">operations</div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <CheckSquare className="h-4 w-4" />
            Completed
          </div>
          <div className="text-2xl font-bold text-green-500">
            {stats?.operations_completed || 0}
          </div>
          <div className="text-xs text-muted-foreground">total</div>
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4" />
            Workers
          </div>
          <div className="text-2xl font-bold">
            {stats?.active_workers || 0} / {stats?.worker_pool_size || 0}
          </div>
          <div className="text-xs text-muted-foreground">
            {stats?.worker_utilization?.toFixed(1)}% util
          </div>
        </div>
      </div>

      {/* Ring Visualization */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Submission Queue */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">Submission Queue (SQ)</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Queue Size</span>
              <span className="font-mono">{stats?.submission_queue_size || 0}</span>
            </div>
            <div className="h-8 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all"
                style={{
                  width: `${Math.min((stats?.submission_queue_size || 0) / 100 * 100, 100)}%`
                }}
              />
            </div>
            <div className="text-sm text-muted-foreground">
              Operations waiting to be processed
            </div>
          </div>
        </div>

        {/* Completion Queue */}
        <div className="rounded-lg border border-border bg-card p-6">
          <h2 className="text-lg font-semibold mb-4">Completion Queue (CQ)</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Queue Size</span>
              <span className="font-mono">{stats?.completion_queue_size || 0}</span>
            </div>
            <div className="h-8 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all"
                style={{
                  width: `${Math.min((stats?.completion_queue_size || 0) / 100 * 100, 100)}%`
                }}
              />
            </div>
            <div className="text-sm text-muted-foreground">
              Completed operations awaiting processing
            </div>
          </div>
        </div>
      </div>

      {/* Latency Percentiles */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">Latency Distribution</h2>
        <div className="flex items-end justify-around h-48 gap-8">
          {[
            { label: 'p50', value: stats?.p50_latency_ms || 0, color: 'bg-green-500' },
            { label: 'p95', value: stats?.p95_latency_ms || 0, color: 'bg-yellow-500' },
            { label: 'p99', value: stats?.p99_latency_ms || 0, color: 'bg-red-500' },
          ].map(({ label, value, color }) => {
            const maxValue = Math.max(stats?.p99_latency_ms || 100, 100)
            const height = (value / maxValue) * 100

            return (
              <div key={label} className="flex flex-col items-center gap-2">
                <span className="text-sm font-mono">{value.toFixed(0)}ms</span>
                <div className="w-16 bg-muted rounded-t-lg overflow-hidden" style={{ height: '150px' }}>
                  <div
                    className={`w-full ${color} transition-all`}
                    style={{ height: `${height}%`, marginTop: `${100 - height}%` }}
                  />
                </div>
                <span className="text-sm text-muted-foreground">{label}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Operation Stats */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">Operation Statistics</h2>
        <div className="grid gap-4 md:grid-cols-5">
          <div className="text-center p-4 rounded-lg bg-muted/50">
            <div className="text-2xl font-bold">{stats?.operations_submitted || 0}</div>
            <div className="text-sm text-muted-foreground">Submitted</div>
          </div>
          <div className="text-center p-4 rounded-lg bg-green-500/20">
            <div className="text-2xl font-bold text-green-500">
              {stats?.operations_completed || 0}
            </div>
            <div className="text-sm text-muted-foreground">Completed</div>
          </div>
          <div className="text-center p-4 rounded-lg bg-red-500/20">
            <div className="text-2xl font-bold text-red-500">
              {stats?.operations_failed || 0}
            </div>
            <div className="text-sm text-muted-foreground">Failed</div>
          </div>
          <div className="text-center p-4 rounded-lg bg-orange-500/20">
            <div className="text-2xl font-bold text-orange-500">
              {stats?.operations_timeout || 0}
            </div>
            <div className="text-sm text-muted-foreground">Timeout</div>
          </div>
          <div className="text-center p-4 rounded-lg bg-gray-500/20">
            <div className="text-2xl font-bold text-gray-500">
              {stats?.operations_cancelled || 0}
            </div>
            <div className="text-sm text-muted-foreground">Cancelled</div>
          </div>
        </div>
      </div>
    </div>
  )
}
