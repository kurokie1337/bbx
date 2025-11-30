import { useQuery, useMutation } from '@tanstack/react-query'
import { useParams, useSearchParams } from 'react-router-dom'
import { useCallback, useMemo, useState } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { getWorkflow, runWorkflow } from '@/services/api'
import { cn, getStatusColor, getStatusBgColor } from '@/lib/utils'
import { Play, Settings } from 'lucide-react'
import type { WorkflowDetail as WorkflowDetailType } from '@/types'

export function WorkflowDetail() {
  const { id } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const filePath = searchParams.get('file_path')

  const [inputs, setInputs] = useState<Record<string, any>>({})

  const { data: workflow, isLoading } = useQuery({
    queryKey: ['workflow', id, filePath],
    queryFn: () => getWorkflow(id!, filePath || undefined),
    enabled: !!id,
  })

  const runMutation = useMutation({
    mutationFn: () => runWorkflow(id!, inputs, filePath || undefined),
  })

  // Build React Flow nodes and edges
  const { nodes: initialNodes, edges: initialEdges } = useMemo(() => {
    if (!workflow?.dag) return { nodes: [], edges: [] }

    const levelSpacing = 200
    const nodeSpacing = 180

    const nodes: Node[] = workflow.dag.nodes.map((node, index) => {
      const levelNodes = workflow.dag.levels[node.level] || []
      const indexInLevel = levelNodes.indexOf(node.id)
      const levelWidth = levelNodes.length * nodeSpacing
      const startX = -(levelWidth / 2) + (nodeSpacing / 2)

      return {
        id: node.id,
        type: 'default',
        position: {
          x: startX + (indexInLevel * nodeSpacing),
          y: node.level * levelSpacing,
        },
        data: {
          label: (
            <div className="px-3 py-2">
              <div className="font-semibold">{node.id}</div>
              <div className="text-xs text-muted-foreground">
                {node.mcp}.{node.method}
              </div>
            </div>
          ),
        },
        style: {
          background: 'hsl(var(--card))',
          border: '1px solid hsl(var(--border))',
          borderRadius: '8px',
          color: 'hsl(var(--foreground))',
        },
      }
    })

    const edges: Edge[] = workflow.dag.edges.map((edge) => ({
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      style: { stroke: 'hsl(var(--muted-foreground))' },
      animated: false,
    }))

    return { nodes, edges }
  }, [workflow])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-muted-foreground">Loading workflow...</div>
      </div>
    )
  }

  if (!workflow) {
    return (
      <div className="p-6">
        <div className="text-red-500">Workflow not found</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{workflow.name}</h1>
            <p className="text-muted-foreground">{workflow.description}</p>
          </div>
          <div className="flex items-center gap-2">
            <button className="flex items-center gap-2 px-3 py-2 border border-border rounded-md hover:bg-accent">
              <Settings className="h-4 w-4" />
              Configure
            </button>
            <button
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {runMutation.isPending ? 'Running...' : 'Run'}
            </button>
          </div>
        </div>

        {/* Inputs */}
        {workflow.inputs.length > 0 && (
          <div className="mt-4 p-4 bg-muted/50 rounded-lg">
            <h3 className="font-semibold mb-2">Inputs</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {workflow.inputs.map((input) => (
                <div key={input.name}>
                  <label className="text-sm text-muted-foreground">
                    {input.name}
                    {input.required && <span className="text-red-500 ml-1">*</span>}
                  </label>
                  <input
                    type="text"
                    value={inputs[input.name] || input.default || ''}
                    onChange={(e) => setInputs({ ...inputs, [input.name]: e.target.value })}
                    placeholder={input.description || `Enter ${input.name}`}
                    className="mt-1 w-full px-3 py-2 bg-background border border-border rounded-md"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* DAG Visualization */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background />
          <Controls />
          <MiniMap
            nodeColor={() => 'hsl(var(--primary))'}
            maskColor="hsl(var(--background) / 0.8)"
          />
        </ReactFlow>
      </div>

      {/* Steps list */}
      <div className="p-4 border-t border-border max-h-64 overflow-auto">
        <h3 className="font-semibold mb-2">Steps ({workflow.steps.length})</h3>
        <div className="space-y-2">
          {workflow.steps.map((step) => (
            <div
              key={step.id}
              className="flex items-center justify-between p-2 bg-muted/50 rounded-md"
            >
              <div>
                <span className="font-mono text-sm">{step.id}</span>
                <span className="text-muted-foreground text-sm ml-2">
                  {step.mcp}.{step.method}
                </span>
              </div>
              {step.depends_on.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  depends on: {step.depends_on.join(', ')}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
