import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { useAppStore } from '@/stores/appStore'
import {
  LayoutDashboard,
  Workflow,
  Users,
  Database,
  CircleDot,
  ListTodo,
  Network,
  Wrench,
  ChevronLeft,
  ChevronRight,
  Wifi,
  WifiOff,
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/workflows', icon: Workflow, label: 'Workflows' },
  { path: '/agents', icon: Users, label: 'Agents' },
  { path: '/memory', icon: Database, label: 'Memory' },
  { path: '/ring', icon: CircleDot, label: 'Ring' },
  { path: '/tasks', icon: ListTodo, label: 'Tasks' },
  { path: '/a2a', icon: Network, label: 'A2A' },
  { path: '/mcp', icon: Wrench, label: 'MCP Tools' },
]

export function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const { sidebarCollapsed, setSidebarCollapsed, wsConnected } = useAppStore()

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside
        className={cn(
          'flex flex-col border-r border-border bg-card transition-all duration-300',
          sidebarCollapsed ? 'w-16' : 'w-64'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          {!sidebarCollapsed && (
            <span className="text-xl font-bold text-primary">BBX Console</span>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 rounded-md hover:bg-accent"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 p-2">
          {navItems.map(({ path, icon: Icon, label }) => {
            const isActive = location.pathname === path ||
              (path !== '/' && location.pathname.startsWith(path))

            return (
              <Link
                key={path}
                to={path}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 transition-colors',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'hover:bg-accent'
                )}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!sidebarCollapsed && <span>{label}</span>}
              </Link>
            )
          })}
        </nav>

        {/* Status */}
        <div className="border-t border-border p-4">
          <div className="flex items-center gap-2 text-sm">
            {wsConnected ? (
              <>
                <Wifi className="h-4 w-4 text-green-500" />
                {!sidebarCollapsed && <span className="text-green-500">Connected</span>}
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-500" />
                {!sidebarCollapsed && <span className="text-red-500">Disconnected</span>}
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  )
}
