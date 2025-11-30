import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { getTaskBoard, createTask, updateTask, decomposeTask } from '@/services/api'
import { cn, getStatusColor, getStatusBgColor } from '@/lib/utils'
import { Plus, Sparkles, GripVertical, User } from 'lucide-react'
import type { Task, TaskColumn } from '@/types'

export function Tasks() {
  const queryClient = useQueryClient()
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showDecomposeModal, setShowDecomposeModal] = useState(false)
  const [newTaskTitle, setNewTaskTitle] = useState('')
  const [decomposeInput, setDecomposeInput] = useState('')

  const { data: board, isLoading } = useQuery({
    queryKey: ['task-board'],
    queryFn: getTaskBoard,
    refetchInterval: 5000,
  })

  const createMutation = useMutation({
    mutationFn: (title: string) => createTask({ title }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task-board'] })
      setNewTaskTitle('')
      setShowCreateModal(false)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) =>
      updateTask(id, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task-board'] })
    },
  })

  const decomposeMutation = useMutation({
    mutationFn: (description: string) => decomposeTask(description),
    onSuccess: (result) => {
      // Show the generated workflow
      console.log('Decomposition result:', result)
    },
  })

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="text-muted-foreground">Loading tasks...</div>
      </div>
    )
  }

  return (
    <div className="p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Task Manager</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowDecomposeModal(true)}
            className="flex items-center gap-2 px-4 py-2 border border-border rounded-md hover:bg-accent"
          >
            <Sparkles className="h-4 w-4" />
            AI Decompose
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
          >
            <Plus className="h-4 w-4" />
            New Task
          </button>
        </div>
      </div>

      {/* Kanban Board */}
      <div className="flex-1 flex gap-4 overflow-x-auto">
        {board?.columns.map((column) => (
          <div
            key={column.status}
            className="flex-shrink-0 w-80 flex flex-col rounded-lg border border-border bg-card"
          >
            {/* Column Header */}
            <div className="p-4 border-b border-border">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">{column.title}</h3>
                <span className="px-2 py-0.5 bg-muted rounded-full text-sm">
                  {column.count}
                </span>
              </div>
            </div>

            {/* Tasks */}
            <div className="flex-1 p-2 space-y-2 overflow-y-auto">
              {column.tasks.map((task) => (
                <TaskCard
                  key={task.id}
                  task={task}
                  onStatusChange={(status) =>
                    updateMutation.mutate({ id: task.id, status })
                  }
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Create Task Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-semibold mb-4">Create Task</h2>
            <input
              type="text"
              value={newTaskTitle}
              onChange={(e) => setNewTaskTitle(e.target.value)}
              placeholder="Task title"
              className="w-full px-3 py-2 bg-background border border-border rounded-md mb-4"
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 border border-border rounded-md hover:bg-accent"
              >
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate(newTaskTitle)}
                disabled={!newTaskTitle || createMutation.isPending}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
              >
                {createMutation.isPending ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Decompose Modal */}
      {showDecomposeModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-card border border-border rounded-lg p-6 w-full max-w-2xl">
            <h2 className="text-xl font-semibold mb-4">AI Task Decomposition</h2>
            <p className="text-muted-foreground mb-4">
              Describe your task and AI will break it down into subtasks with agent assignments.
            </p>
            <textarea
              value={decomposeInput}
              onChange={(e) => setDecomposeInput(e.target.value)}
              placeholder="Describe the task you want to accomplish..."
              className="w-full px-3 py-2 bg-background border border-border rounded-md mb-4 h-32"
            />

            {decomposeMutation.data && (
              <div className="mb-4 p-4 bg-muted/50 rounded-lg">
                <h3 className="font-semibold mb-2">Generated Subtasks:</h3>
                <div className="space-y-2">
                  {decomposeMutation.data.subtasks.map((st, idx) => (
                    <div key={idx} className="flex items-center gap-2 p-2 bg-background rounded">
                      <span className="font-mono text-xs bg-muted px-2 py-1 rounded">
                        {st.assigned_agent}
                      </span>
                      <span>{st.title}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => {
                  setShowDecomposeModal(false)
                  setDecomposeInput('')
                }}
                className="px-4 py-2 border border-border rounded-md hover:bg-accent"
              >
                Cancel
              </button>
              <button
                onClick={() => decomposeMutation.mutate(decomposeInput)}
                disabled={!decomposeInput || decomposeMutation.isPending}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
              >
                {decomposeMutation.isPending ? 'Analyzing...' : 'Decompose'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function TaskCard({
  task,
  onStatusChange,
}: {
  task: Task
  onStatusChange: (status: string) => void
}) {
  return (
    <div className="p-3 bg-background border border-border rounded-lg hover:border-primary transition-colors cursor-pointer">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="font-medium">{task.title}</h4>
          {task.description && (
            <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
              {task.description}
            </p>
          )}
        </div>
        <GripVertical className="h-4 w-4 text-muted-foreground flex-shrink-0" />
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className={cn(
          'px-2 py-0.5 rounded text-xs',
          task.priority === 'critical' && 'bg-red-500/20 text-red-500',
          task.priority === 'high' && 'bg-orange-500/20 text-orange-500',
          task.priority === 'medium' && 'bg-yellow-500/20 text-yellow-500',
          task.priority === 'low' && 'bg-gray-500/20 text-gray-500',
        )}>
          {task.priority}
        </span>

        {task.assigned_agent && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <User className="h-3 w-3" />
            {task.assigned_agent}
          </div>
        )}
      </div>
    </div>
  )
}
