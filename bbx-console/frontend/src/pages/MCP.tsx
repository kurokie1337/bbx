import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { getMCPServers, getMCPTools, callMCPTool } from '@/services/api'
import { Wrench, Play, Server, CheckCircle, XCircle } from 'lucide-react'

export function MCP() {
  const [selectedServer, setSelectedServer] = useState<string>('bbx')
  const [selectedTool, setSelectedTool] = useState<string>('')
  const [toolArgs, setToolArgs] = useState<string>('{}')

  const { data: servers } = useQuery({
    queryKey: ['mcp-servers'],
    queryFn: getMCPServers,
  })

  const { data: tools } = useQuery({
    queryKey: ['mcp-tools', selectedServer],
    queryFn: () => getMCPTools(selectedServer),
    enabled: !!selectedServer,
  })

  const callMutation = useMutation({
    mutationFn: () => {
      const args = JSON.parse(toolArgs)
      return callMCPTool(selectedServer, selectedTool, args)
    },
  })

  return (
    <div className="p-6 h-full flex gap-6">
      {/* Tools List */}
      <div className="w-96 space-y-4">
        <h1 className="text-3xl font-bold">MCP Tools</h1>

        {/* Server Selection */}
        <div className="space-y-2">
          <label className="text-sm text-muted-foreground">Server</label>
          <select
            value={selectedServer}
            onChange={(e) => {
              setSelectedServer(e.target.value)
              setSelectedTool('')
            }}
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
          >
            {servers?.map((server: any) => (
              <option key={server.name} value={server.name}>
                {server.name} ({server.tools_count} tools)
              </option>
            ))}
          </select>
        </div>

        {/* Tools List */}
        <div className="space-y-2 max-h-[calc(100vh-300px)] overflow-auto">
          {tools?.map((tool: any) => (
            <button
              key={tool.name}
              onClick={() => setSelectedTool(tool.name)}
              className={`w-full text-left p-3 rounded-lg border transition-colors ${
                selectedTool === tool.name
                  ? 'border-primary bg-primary/10'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <Wrench className="h-4 w-4 text-muted-foreground" />
                <span className="font-mono text-sm">{tool.name}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                {tool.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Tool Detail & Invoker */}
      <div className="flex-1 rounded-lg border border-border bg-card p-6">
        {selectedTool ? (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold font-mono">{selectedTool}</h2>
              <p className="text-muted-foreground mt-1">
                {tools?.find((t: any) => t.name === selectedTool)?.description}
              </p>
            </div>

            {/* Input Schema */}
            <div>
              <h3 className="font-semibold mb-2">Input Schema</h3>
              <pre className="p-4 bg-muted/50 rounded-lg text-sm overflow-auto max-h-48">
                {JSON.stringify(
                  tools?.find((t: any) => t.name === selectedTool)?.input_schema,
                  null,
                  2
                )}
              </pre>
            </div>

            {/* Arguments Input */}
            <div>
              <h3 className="font-semibold mb-2">Arguments (JSON)</h3>
              <textarea
                value={toolArgs}
                onChange={(e) => setToolArgs(e.target.value)}
                className="w-full h-32 px-3 py-2 bg-background border border-border rounded-md font-mono text-sm"
                placeholder="{}"
              />
            </div>

            {/* Invoke Button */}
            <button
              onClick={() => callMutation.mutate()}
              disabled={callMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {callMutation.isPending ? 'Invoking...' : 'Invoke Tool'}
            </button>

            {/* Result */}
            {callMutation.data && (
              <div>
                <h3 className="font-semibold mb-2 flex items-center gap-2">
                  Result
                  {callMutation.data.success ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                </h3>
                <pre className="p-4 bg-muted/50 rounded-lg text-sm overflow-auto max-h-64 whitespace-pre-wrap">
                  {callMutation.data.success
                    ? typeof callMutation.data.result === 'string'
                      ? callMutation.data.result
                      : JSON.stringify(callMutation.data.result, null, 2)
                    : callMutation.data.error}
                </pre>
              </div>
            )}

            {callMutation.error && (
              <div className="p-4 bg-red-500/20 text-red-500 rounded-lg">
                Error: {(callMutation.error as Error).message}
              </div>
            )}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Wrench className="h-12 w-12 mx-auto mb-4" />
              <p>Select a tool to view details and invoke</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
