import { useQuery } from '@tanstack/react-query'
import { getAgents, getAgent } from '@/services/api'
import { cn, getStatusColor, getStatusBgColor } from '@/lib/utils'
import { useState } from 'react'
import { Users, Wrench, Brain, Activity } from 'lucide-react'

export function Agents() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)

  const { data: agents, isLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: getAgents,
    refetchInterval: 2000,
  })

  const { data: agentDetail } = useQuery({
    queryKey: ['agent', selectedAgent],
    queryFn: () => getAgent(selectedAgent!),
    enabled: !!selectedAgent,
  })

  return (
    <div className="p-6 h-full flex gap-6">
      {/* Agent List */}
      <div className="w-96 space-y-4">
        <h1 className="text-3xl font-bold">Agents</h1>

        {isLoading ? (
          <div className="text-muted-foreground">Loading agents...</div>
        ) : agents?.length === 0 ? (
          <div className="text-center py-12">
            <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h2 className="text-xl font-semibold mb-2">No agents found</h2>
            <p className="text-muted-foreground">
              Add agents in .claude/agents/
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {agents?.map((agent) => (
              <button
                key={agent.id}
                onClick={() => setSelectedAgent(agent.id)}
                className={cn(
                  'w-full text-left rounded-lg border border-border bg-card p-4 hover:border-primary transition-colors',
                  selectedAgent === agent.id && 'border-primary'
                )}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-semibold">{agent.name}</h3>
                    <p className="text-sm text-muted-foreground line-clamp-1">
                      {agent.description}
                    </p>
                  </div>
                  <span className={cn(
                    'px-2 py-0.5 rounded-full text-xs',
                    getStatusColor(agent.status),
                    getStatusBgColor(agent.status),
                  )}>
                    {agent.status}
                  </span>
                </div>

                <div className="mt-3 flex items-center justify-between text-sm text-muted-foreground">
                  <div className="flex items-center gap-4">
                    <span>{agent.metrics.tasks_completed} completed</span>
                    <span>{(agent.metrics.success_rate * 100).toFixed(0)}% success</span>
                  </div>
                  <span>{agent.model}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Agent Detail */}
      <div className="flex-1 rounded-lg border border-border bg-card p-6">
        {selectedAgent && agentDetail ? (
          <div className="space-y-6">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-2xl font-bold">{agentDetail.name}</h2>
                <p className="text-muted-foreground">{agentDetail.description}</p>
              </div>
              <span className={cn(
                'px-3 py-1 rounded-full',
                getStatusColor(agentDetail.status),
                getStatusBgColor(agentDetail.status),
              )}>
                {agentDetail.status}
              </span>
            </div>

            {/* Tools */}
            <div>
              <h3 className="flex items-center gap-2 font-semibold mb-2">
                <Wrench className="h-4 w-4" />
                Tools
              </h3>
              <div className="flex flex-wrap gap-2">
                {agentDetail.tools.map((tool) => (
                  <span
                    key={tool}
                    className="px-2 py-1 bg-muted rounded-md text-sm"
                  >
                    {tool}
                  </span>
                ))}
              </div>
            </div>

            {/* Model */}
            <div>
              <h3 className="flex items-center gap-2 font-semibold mb-2">
                <Brain className="h-4 w-4" />
                Model
              </h3>
              <span className="px-3 py-1 bg-primary/20 text-primary rounded-md">
                {agentDetail.model}
              </span>
            </div>

            {/* Metrics */}
            <div>
              <h3 className="flex items-center gap-2 font-semibold mb-2">
                <Activity className="h-4 w-4" />
                Metrics
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">{agentDetail.metrics.tasks_completed}</div>
                  <div className="text-sm text-muted-foreground">Tasks Completed</div>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">{agentDetail.metrics.tasks_failed}</div>
                  <div className="text-sm text-muted-foreground">Tasks Failed</div>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">
                    {(agentDetail.metrics.success_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-muted-foreground">Success Rate</div>
                </div>
                <div className="p-3 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">
                    {agentDetail.metrics.avg_duration_ms.toFixed(0)}ms
                  </div>
                  <div className="text-sm text-muted-foreground">Avg Duration</div>
                </div>
              </div>
            </div>

            {/* System Prompt */}
            <div>
              <h3 className="font-semibold mb-2">System Prompt</h3>
              <pre className="p-4 bg-muted/50 rounded-lg text-sm overflow-auto max-h-64 whitespace-pre-wrap">
                {agentDetail.system_prompt}
              </pre>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            Select an agent to view details
          </div>
        )}
      </div>
    </div>
  )
}
