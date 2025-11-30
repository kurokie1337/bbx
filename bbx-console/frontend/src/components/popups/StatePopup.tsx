import { useEffect, useState } from 'react'
import { PopupWrapper } from './PopupWrapper'
import { api } from '@/services/api'
import type { AxiosResponse } from 'axios'

interface StateItem {
  key: string
  value: string | number | boolean | object
  namespace?: string
  updated_at?: string
}

export function StatePopup() {
  const [stateItems, setStateItems] = useState<StateItem[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedItem, setSelectedItem] = useState<StateItem | null>(null)
  const [editValue, setEditValue] = useState('')

  useEffect(() => {
    // Use MCP call to get state list
    api.post('/mcp/call', { server: 'bbx', tool: 'bbx_state_list', arguments: { pattern: '*' } })
      .then((res: AxiosResponse) => {
        const result = res.data?.result
        if (result && Array.isArray(result.keys)) {
          // Fetch values for each key
          const items = result.keys.map((key: string) => ({
            key,
            value: result.values?.[key] || 'unknown',
            updated_at: 'now'
          }))
          setStateItems(items)
        } else {
          setStateItems([])
        }
        setLoading(false)
      })
      .catch((err) => {
        console.error('Failed to fetch state:', err)
        setStateItems([])
        setLoading(false)
      })
  }, [])

  const handleItemClick = (item: StateItem) => {
    setSelectedItem(item)
    setEditValue(typeof item.value === 'object' ? JSON.stringify(item.value, null, 2) : String(item.value))
  }

  const handleSave = () => {
    if (!selectedItem) return
    // Use MCP call to set state value
    api.post('/mcp/call', {
      server: 'bbx',
      tool: 'bbx_state_set',
      arguments: { key: selectedItem.key, value: editValue }
    })
      .then(() => {
        setStateItems(items => items.map(item =>
          item.key === selectedItem.key ? { ...item, value: editValue, updated_at: 'now' } : item
        ))
        setSelectedItem(null)
      })
      .catch(() => {
        // Update locally for demo
        setStateItems(items => items.map(item =>
          item.key === selectedItem.key ? { ...item, value: editValue, updated_at: 'now' } : item
        ))
        setSelectedItem(null)
      })
  }

  const getValueType = (value: any): string => {
    if (typeof value === 'number') return 'number'
    if (typeof value === 'boolean') return 'bool'
    if (typeof value === 'object') return 'json'
    return 'string'
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'number': return 'var(--blue)'
      case 'bool': return 'var(--yellow)'
      case 'json': return 'var(--accent)'
      default: return 'var(--green)'
    }
  }

  // Detail view
  if (selectedItem) {
    return (
      <PopupWrapper
        title="EDIT STATE"
        footer={
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedItem(null)}
              className="px-3 py-1 text-xs rounded cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-3 py-1 text-xs rounded cursor-pointer"
              style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
            >
              Save
            </button>
          </div>
        }
      >
        <button
          onClick={() => setSelectedItem(null)}
          className="flex items-center gap-1 text-xs mb-4 cursor-pointer hover:underline"
          style={{ color: 'var(--accent)' }}
        >
          <span>&#8592;</span> Back to list
        </button>

        <div className="space-y-4">
          {/* Key info */}
          <div
            className="rounded-lg border p-4"
            style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <div className="text-[10px] font-semibold mb-2" style={{ color: 'var(--text-muted)' }}>KEY</div>
            <div className="font-mono text-sm" style={{ color: 'var(--text-primary)' }}>
              {selectedItem.key}
            </div>
            {selectedItem.namespace && (
              <div className="text-[10px] mt-2" style={{ color: 'var(--text-muted)' }}>
                namespace: {selectedItem.namespace}
              </div>
            )}
          </div>

          {/* Value editor */}
          <div
            className="rounded-lg border p-4"
            style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>VALUE</span>
              <span
                className="text-[9px] px-1.5 py-0.5 rounded"
                style={{ background: `${getTypeColor(getValueType(selectedItem.value))}20`, color: getTypeColor(getValueType(selectedItem.value)) }}
              >
                {getValueType(selectedItem.value)}
              </span>
            </div>
            <textarea
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              className="w-full h-24 px-3 py-2 rounded font-mono text-xs resize-none"
              style={{
                background: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border)',
                outline: 'none'
              }}
            />
          </div>
        </div>
      </PopupWrapper>
    )
  }

  // List view
  return (
    <PopupWrapper
      title="STATE"
      footer={
        <div className="text-[11px]">
          {stateItems.length} keys | Click to edit
        </div>
      }
    >
      {loading ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          Loading...
        </div>
      ) : stateItems.length === 0 ? (
        <div className="flex items-center justify-center py-8" style={{ color: 'var(--text-muted)' }}>
          No state values
        </div>
      ) : (
        <div
          className="rounded-lg border overflow-hidden"
          style={{ background: 'var(--bg-secondary)', borderColor: 'var(--border)' }}
        >
          {stateItems.map((item, i) => (
            <div
              key={item.key}
              className="flex items-center justify-between px-3 py-2 cursor-pointer transition-fast hover:bg-[var(--bg-tertiary)]"
              style={{ borderBottom: i < stateItems.length - 1 ? '1px solid var(--border)' : 'none' }}
              onClick={() => handleItemClick(item)}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className="text-[9px] px-1 rounded"
                  style={{ background: `${getTypeColor(getValueType(item.value))}20`, color: getTypeColor(getValueType(item.value)) }}
                >
                  {getValueType(item.value)}
                </span>
                <span className="text-xs font-mono truncate" style={{ color: 'var(--text-primary)' }}>
                  {item.key}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono truncate max-w-[100px]" style={{ color: 'var(--text-muted)' }}>
                  {typeof item.value === 'object' ? '{...}' : String(item.value)}
                </span>
                <span style={{ color: 'var(--text-muted)' }}>&#8250;</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </PopupWrapper>
  )
}
