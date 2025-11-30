import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getWorkflows } from '@/services/api'
import { cn, formatDuration, formatRelativeTime, getStatusColor, getStatusBgColor } from '@/lib/utils'
import { Workflow, Play, Clock, FileCode } from 'lucide-react'

export function Workflows() {
  const { data: workflows, isLoading } = useQuery({
    queryKey: ['workflows'],
    queryFn: getWorkflows,
  })

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Workflows</h1>
        <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90">
          <FileCode className="h-4 w-4" />
          New Workflow
        </button>
      </div>

      {isLoading ? (
        <div className="text-muted-foreground">Loading workflows...</div>
      ) : workflows?.length === 0 ? (
        <div className="text-center py-12">
          <Workflow className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h2 className="text-xl font-semibold mb-2">No workflows found</h2>
          <p className="text-muted-foreground">
            Create your first workflow to get started
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {workflows?.map((workflow) => (
            <Link
              key={workflow.id}
              to={`/workflows/${workflow.id}?file_path=${encodeURIComponent(workflow.file_path)}`}
              className="rounded-lg border border-border bg-card p-4 hover:border-primary transition-colors"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{workflow.name}</h3>
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {workflow.description || 'No description'}
                  </p>
                </div>
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <span>{workflow.step_count}</span>
                  <span>steps</span>
                </div>
              </div>

              {workflow.last_run && (
                <div className="mt-4 flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span className={cn(
                      'px-2 py-0.5 rounded-full text-xs',
                      getStatusColor(workflow.last_run.status),
                      getStatusBgColor(workflow.last_run.status),
                    )}>
                      {workflow.last_run.status}
                    </span>
                    {workflow.last_run.duration_ms && (
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {formatDuration(workflow.last_run.duration_ms)}
                      </span>
                    )}
                  </div>
                  <span className="text-muted-foreground">
                    {formatRelativeTime(workflow.last_run.started_at)}
                  </span>
                </div>
              )}

              <div className="mt-4 pt-4 border-t border-border flex items-center justify-between">
                <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                  {workflow.file_path}
                </span>
                <button
                  onClick={(e) => {
                    e.preventDefault()
                    // Run workflow
                  }}
                  className="flex items-center gap-1 px-3 py-1 bg-green-500/20 text-green-500 rounded-md hover:bg-green-500/30 text-sm"
                >
                  <Play className="h-3 w-3" />
                  Run
                </button>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
