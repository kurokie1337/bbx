import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { discoverAgent, createA2ATask } from '@/services/api'
import { Network, Search, Send, ExternalLink } from 'lucide-react'

export function A2A() {
  const [agentUrl, setAgentUrl] = useState('')
  const [discoveredAgent, setDiscoveredAgent] = useState<any>(null)
  const [selectedSkill, setSelectedSkill] = useState<string>('')
  const [taskInput, setTaskInput] = useState('')

  const discoverMutation = useMutation({
    mutationFn: (url: string) => discoverAgent(url),
    onSuccess: (data) => {
      setDiscoveredAgent(data)
      if (data.skills?.length > 0) {
        setSelectedSkill(data.skills[0].id)
      }
    },
  })

  const taskMutation = useMutation({
    mutationFn: () => createA2ATask(agentUrl, selectedSkill, { prompt: taskInput }),
  })

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">A2A Playground</h1>
        <div className="text-sm text-muted-foreground">
          Agent-to-Agent Protocol Testing
        </div>
      </div>

      {/* Discovery Section */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h2 className="text-lg font-semibold mb-4">Discover Agent</h2>
        <div className="flex gap-4">
          <input
            type="text"
            value={agentUrl}
            onChange={(e) => setAgentUrl(e.target.value)}
            placeholder="https://agent.example.com"
            className="flex-1 px-3 py-2 bg-background border border-border rounded-md"
          />
          <button
            onClick={() => discoverMutation.mutate(agentUrl)}
            disabled={!agentUrl || discoverMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
          >
            <Search className="h-4 w-4" />
            {discoverMutation.isPending ? 'Discovering...' : 'Discover'}
          </button>
        </div>

        {discoverMutation.error && (
          <div className="mt-4 p-4 bg-red-500/20 text-red-500 rounded-lg">
            Failed to discover agent: {(discoverMutation.error as Error).message}
          </div>
        )}
      </div>

      {/* Agent Card Display */}
      {discoveredAgent && (
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold">{discoveredAgent.name}</h2>
              <p className="text-muted-foreground">{discoveredAgent.description}</p>
            </div>
            <a
              href={discoveredAgent.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-primary hover:underline"
            >
              <ExternalLink className="h-4 w-4" />
              {discoveredAgent.url}
            </a>
          </div>

          {/* Skills */}
          <div className="mb-6">
            <h3 className="font-semibold mb-2">Skills</h3>
            <div className="grid gap-2 md:grid-cols-2">
              {discoveredAgent.skills?.map((skill: any) => (
                <button
                  key={skill.id}
                  onClick={() => setSelectedSkill(skill.id)}
                  className={`p-3 text-left rounded-lg border transition-colors ${
                    selectedSkill === skill.id
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  }`}
                >
                  <div className="font-medium">{skill.name}</div>
                  <div className="text-sm text-muted-foreground">{skill.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Capabilities */}
          <div className="mb-6">
            <h3 className="font-semibold mb-2">Capabilities</h3>
            <div className="flex gap-4">
              <span className={`px-2 py-1 rounded text-sm ${
                discoveredAgent.capabilities?.streaming ? 'bg-green-500/20 text-green-500' : 'bg-muted'
              }`}>
                Streaming: {discoveredAgent.capabilities?.streaming ? 'Yes' : 'No'}
              </span>
              <span className={`px-2 py-1 rounded text-sm ${
                discoveredAgent.capabilities?.pushNotifications ? 'bg-green-500/20 text-green-500' : 'bg-muted'
              }`}>
                Push Notifications: {discoveredAgent.capabilities?.pushNotifications ? 'Yes' : 'No'}
              </span>
            </div>
          </div>

          {/* Task Input */}
          <div className="space-y-4">
            <h3 className="font-semibold">Send Task</h3>
            <div className="flex gap-4">
              <input
                type="text"
                value={taskInput}
                onChange={(e) => setTaskInput(e.target.value)}
                placeholder="Enter your prompt..."
                className="flex-1 px-3 py-2 bg-background border border-border rounded-md"
              />
              <button
                onClick={() => taskMutation.mutate()}
                disabled={!taskInput || !selectedSkill || taskMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
                {taskMutation.isPending ? 'Sending...' : 'Send'}
              </button>
            </div>

            {taskMutation.data && (
              <div className="p-4 bg-muted/50 rounded-lg">
                <h4 className="font-semibold mb-2">Response:</h4>
                <pre className="text-sm overflow-auto">
                  {JSON.stringify(taskMutation.data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!discoveredAgent && !discoverMutation.isPending && (
        <div className="text-center py-12">
          <Network className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">No agent discovered</h2>
          <p className="text-muted-foreground">
            Enter an agent URL above to discover its capabilities
          </p>
        </div>
      )}
    </div>
  )
}
