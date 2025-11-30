import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { Layout } from './components/common/Layout'
import { Dashboard } from './pages/Dashboard'
import { Workflows } from './pages/Workflows'
import { WorkflowDetail } from './pages/WorkflowDetail'
import { Agents } from './pages/Agents'
import { Memory } from './pages/Memory'
import { Ring } from './pages/Ring'
import { Tasks } from './pages/Tasks'
import { A2A } from './pages/A2A'
import { MCP } from './pages/MCP'
import { useAppStore } from './stores/appStore'

function App() {
  const { connectWebSocket, disconnectWebSocket } = useAppStore()

  useEffect(() => {
    connectWebSocket()
    return () => disconnectWebSocket()
  }, [connectWebSocket, disconnectWebSocket])

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/workflows" element={<Workflows />} />
        <Route path="/workflows/:id" element={<WorkflowDetail />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="/memory" element={<Memory />} />
        <Route path="/ring" element={<Ring />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/a2a" element={<A2A />} />
        <Route path="/mcp" element={<MCP />} />
      </Routes>
    </Layout>
  )
}

export default App
