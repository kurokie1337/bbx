import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

export function formatDate(date: string | Date): string {
  const d = new Date(date)
  return d.toLocaleString()
}

export function formatRelativeTime(date: string | Date): string {
  const d = new Date(date)
  const now = new Date()
  const diff = now.getTime() - d.getTime()

  if (diff < 60000) return 'just now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
  return `${Math.floor(diff / 86400000)}d ago`
}

export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    pending: 'text-gray-400',
    waiting: 'text-yellow-400',
    running: 'text-blue-400',
    success: 'text-green-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
    error: 'text-red-400',
    skipped: 'text-gray-500',
    timeout: 'text-orange-400',
    cancelled: 'text-gray-500',
    idle: 'text-gray-400',
    working: 'text-blue-400',
    queued: 'text-yellow-400',
    in_progress: 'text-blue-400',
  }
  return colors[status] || 'text-gray-400'
}

export function getStatusBgColor(status: string): string {
  const colors: Record<string, string> = {
    pending: 'bg-gray-500/20',
    waiting: 'bg-yellow-500/20',
    running: 'bg-blue-500/20',
    success: 'bg-green-500/20',
    completed: 'bg-green-500/20',
    failed: 'bg-red-500/20',
    error: 'bg-red-500/20',
    skipped: 'bg-gray-500/20',
    timeout: 'bg-orange-500/20',
    cancelled: 'bg-gray-500/20',
    idle: 'bg-gray-500/20',
    working: 'bg-blue-500/20',
    queued: 'bg-yellow-500/20',
    in_progress: 'bg-blue-500/20',
  }
  return colors[status] || 'bg-gray-500/20'
}
