import { create } from 'zustand'

export type PopupType = 'memory' | 'agents' | 'ring' | 'history' | 'settings' | 'logs' | 'state' | null
export type ViewMode = 'console' | 'desktop' | 'sandbox'

interface UIState {
  // View Mode
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void
  // Command Palette (legacy - keeping for compatibility)
  commandPaletteOpen: boolean
  openCommandPalette: () => void
  closeCommandPalette: () => void
  toggleCommandPalette: () => void

  // Side Panel (Explorer) - right
  sidePanelOpen: boolean
  openSidePanel: () => void
  closeSidePanel: () => void
  toggleSidePanel: () => void

  // Left Panel (Plugins)
  leftPanelOpen: boolean
  openLeftPanel: () => void
  closeLeftPanel: () => void
  toggleLeftPanel: () => void

  // Popups
  activePopup: PopupType
  openPopup: (popup: PopupType) => void
  closePopup: () => void

  // Auto-scroll
  autoScrollOutput: boolean
  toggleAutoScroll: () => void

  // Connection status
  connectionStatus: 'connected' | 'disconnected' | 'connecting'
  setConnectionStatus: (status: 'connected' | 'disconnected' | 'connecting') => void
}

export const useUIStore = create<UIState>((set) => ({
  // View Mode
  viewMode: 'console',
  setViewMode: (mode) => set({ viewMode: mode }),

  // Command Palette (legacy)
  commandPaletteOpen: false,
  openCommandPalette: () => set({ commandPaletteOpen: true }),
  closeCommandPalette: () => set({ commandPaletteOpen: false }),
  toggleCommandPalette: () => set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),

  // Side Panel (Explorer) - open by default
  sidePanelOpen: true,
  openSidePanel: () => set({ sidePanelOpen: true }),
  closeSidePanel: () => set({ sidePanelOpen: false }),
  toggleSidePanel: () => set((state) => ({ sidePanelOpen: !state.sidePanelOpen })),

  // Left Panel (Plugins) - open by default
  leftPanelOpen: true,
  openLeftPanel: () => set({ leftPanelOpen: true }),
  closeLeftPanel: () => set({ leftPanelOpen: false }),
  toggleLeftPanel: () => set((state) => ({ leftPanelOpen: !state.leftPanelOpen })),

  // Popups
  activePopup: null,
  openPopup: (popup) => set({ activePopup: popup, commandPaletteOpen: false }),
  closePopup: () => set({ activePopup: null }),

  // Auto-scroll
  autoScrollOutput: true,
  toggleAutoScroll: () => set((state) => ({ autoScrollOutput: !state.autoScrollOutput })),

  // Connection status
  connectionStatus: 'disconnected',
  setConnectionStatus: (status) => set({ connectionStatus: status }),
}))
