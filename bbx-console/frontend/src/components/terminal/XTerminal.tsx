import { useEffect, useRef, useCallback } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import '@xterm/xterm/css/xterm.css'

interface XTerminalProps {
  onData?: (data: string) => void
  onReady?: (term: Terminal) => void
  className?: string
  welcomeMessage?: string
  wsUrl?: string
}

export function XTerminal({
  onData,
  onReady,
  className = '',
  welcomeMessage,
  wsUrl
}: XTerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const terminalRef = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const commandBufferRef = useRef<string>('')
  const historyRef = useRef<string[]>([])
  const historyIndexRef = useRef<number>(-1)

  // BBX Commands handler (local simulation)
  const handleCommand = useCallback((command: string) => {
    const term = terminalRef.current
    if (!term) return

    const args = command.trim().split(' ')

    // Add to history
    if (command.trim()) {
      historyRef.current.push(command.trim())
      historyIndexRef.current = historyRef.current.length
    }

    switch (args[0]) {
      case 'help':
        term.writeln('')
        term.writeln('\x1b[36m  Available Commands:\x1b[0m')
        term.writeln('')
        term.writeln('  \x1b[33mhelp\x1b[0m          Show this help')
        term.writeln('  \x1b[33mclear\x1b[0m         Clear terminal')
        term.writeln('  \x1b[33mbbx\x1b[0m           BBX workflow commands')
        term.writeln('  \x1b[33mbbx list\x1b[0m      List workflows')
        term.writeln('  \x1b[33mbbx run <wf>\x1b[0m  Run a workflow')
        term.writeln('  \x1b[33mbbx ps\x1b[0m        List running processes')
        term.writeln('  \x1b[33mbbx system\x1b[0m    System health check')
        term.writeln('  \x1b[33mecho <text>\x1b[0m   Echo text')
        term.writeln('  \x1b[33mdate\x1b[0m          Show current date')
        term.writeln('')
        break

      case 'clear':
        term.clear()
        break

      case 'echo':
        term.writeln(args.slice(1).join(' '))
        break

      case 'date':
        term.writeln(new Date().toLocaleString())
        break

      case 'bbx':
        handleBBXCommand(args.slice(1))
        break

      case '':
        // Empty command, just new prompt
        break

      default:
        term.writeln(`\x1b[31mCommand not found: ${args[0]}\x1b[0m`)
        term.writeln('Type \x1b[33mhelp\x1b[0m for available commands')
    }

    // Write new prompt
    term.write('\r\n\x1b[36m$\x1b[0m ')
  }, [])

  const handleBBXCommand = useCallback((args: string[]) => {
    const term = terminalRef.current
    if (!term) return

    const subCmd = args[0] || 'help'

    switch (subCmd) {
      case 'list':
        term.writeln('')
        term.writeln('\x1b[36m  Workflows:\x1b[0m')
        term.writeln('')
        term.writeln('  \x1b[32m*\x1b[0m hello_world.bbx')
        term.writeln('  \x1b[32m*\x1b[0m api_demo.bbx')
        term.writeln('  \x1b[32m*\x1b[0m data_pipeline.bbx')
        term.writeln('')
        break

      case 'run':
        const workflow = args[1]
        if (!workflow) {
          term.writeln('\x1b[31mUsage: bbx run <workflow>\x1b[0m')
        } else {
          term.writeln(`\x1b[33mRunning ${workflow}...\x1b[0m`)
          setTimeout(() => {
            term.writeln('\x1b[32m[OK]\x1b[0m Step 1: Initialize')
            setTimeout(() => {
              term.writeln('\x1b[32m[OK]\x1b[0m Step 2: Execute')
              setTimeout(() => {
                term.writeln('\x1b[32m[OK]\x1b[0m Step 3: Finalize')
                term.writeln('')
                term.writeln(`\x1b[32mWorkflow ${workflow} completed!\x1b[0m`)
                term.write('\r\n\x1b[36m$\x1b[0m ')
              }, 500)
            }, 500)
          }, 500)
          return // Don't show prompt immediately
        }
        break

      case 'ps':
        term.writeln('')
        term.writeln('\x1b[36m  Running Processes:\x1b[0m')
        term.writeln('')
        term.writeln('  \x1b[33mPID     WORKFLOW         STATUS\x1b[0m')
        term.writeln('  1234    hello_world.bbx   running')
        term.writeln('')
        break

      case 'system':
        term.writeln('')
        term.writeln('\x1b[36m  BBX System Status:\x1b[0m')
        term.writeln('')
        term.writeln('  \x1b[32m[OK]\x1b[0m Python 3.11')
        term.writeln('  \x1b[32m[OK]\x1b[0m Docker running')
        term.writeln('  \x1b[32m[OK]\x1b[0m MCP servers: 3 connected')
        term.writeln('  \x1b[32m[OK]\x1b[0m Adapters: 5 loaded')
        term.writeln('')
        break

      case 'help':
      default:
        term.writeln('')
        term.writeln('\x1b[36m  BBX Commands:\x1b[0m')
        term.writeln('')
        term.writeln('  \x1b[33mbbx list\x1b[0m      List all workflows')
        term.writeln('  \x1b[33mbbx run <wf>\x1b[0m  Run a workflow')
        term.writeln('  \x1b[33mbbx ps\x1b[0m        List running processes')
        term.writeln('  \x1b[33mbbx system\x1b[0m    System health check')
        term.writeln('')
    }
  }, [])

  // Initialize terminal
  useEffect(() => {
    if (!containerRef.current) return

    // Create terminal instance
    const terminal = new Terminal({
      cursorBlink: true,
      cursorStyle: 'block',
      fontSize: 13,
      fontFamily: '"SF Mono", "Fira Code", "JetBrains Mono", Monaco, Menlo, monospace',
      theme: {
        background: '#0a0a0a',
        foreground: '#e5e5e5',
        cursor: '#0a84ff',
        cursorAccent: '#0a0a0a',
        selectionBackground: 'rgba(10, 132, 255, 0.3)',
        black: '#1c1c1c',
        red: '#ff453a',
        green: '#30d158',
        yellow: '#ffd60a',
        blue: '#0a84ff',
        magenta: '#bf5af2',
        cyan: '#64d2ff',
        white: '#f5f5f7',
        brightBlack: '#48484a',
        brightRed: '#ff6961',
        brightGreen: '#4cd964',
        brightYellow: '#ffcc00',
        brightBlue: '#5ac8fa',
        brightMagenta: '#ff2d55',
        brightCyan: '#5ac8fa',
        brightWhite: '#ffffff',
      },
      allowTransparency: true,
      scrollback: 1000,
    })

    terminalRef.current = terminal

    // Create and load addons
    const fitAddon = new FitAddon()
    fitAddonRef.current = fitAddon
    terminal.loadAddon(fitAddon)

    const webLinksAddon = new WebLinksAddon()
    terminal.loadAddon(webLinksAddon)

    // Open terminal in container
    terminal.open(containerRef.current)
    fitAddon.fit()

    // Welcome message
    if (welcomeMessage) {
      terminal.writeln(welcomeMessage)
    } else {
      terminal.writeln('\x1b[36m  ____  ______  __\x1b[0m')
      terminal.writeln('\x1b[36m |  _ \\|  _ \\ \\/ /\x1b[0m  \x1b[90mTerminal v1.0.0\x1b[0m')
      terminal.writeln('\x1b[36m | |_) | |_) >  < \x1b[0m  \x1b[90mOperating System for AI Agents\x1b[0m')
      terminal.writeln('\x1b[36m |____/|____/_/\\_\\\x1b[0m')
      terminal.writeln('')
      terminal.writeln('\x1b[90mType \x1b[33mhelp\x1b[90m for available commands\x1b[0m')
      terminal.writeln('')
    }
    terminal.write('\x1b[36m$\x1b[0m ')

    // Handle input
    terminal.onData((data) => {
      // Forward to external handler if provided
      if (onData) {
        onData(data)
      }

      // If WebSocket is connected, send data there
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(data)
        return
      }

      // Local command handling
      const code = data.charCodeAt(0)

      if (code === 13) {
        // Enter
        handleCommand(commandBufferRef.current)
        commandBufferRef.current = ''
      } else if (code === 127 || code === 8) {
        // Backspace
        if (commandBufferRef.current.length > 0) {
          commandBufferRef.current = commandBufferRef.current.slice(0, -1)
          terminal.write('\b \b')
        }
      } else if (code === 27) {
        // Escape sequences (arrow keys, etc.)
        if (data === '\x1b[A') {
          // Up arrow - history
          if (historyIndexRef.current > 0) {
            historyIndexRef.current--
            const cmd = historyRef.current[historyIndexRef.current] || ''
            // Clear current line
            terminal.write('\r\x1b[36m$\x1b[0m ' + ' '.repeat(commandBufferRef.current.length))
            terminal.write('\r\x1b[36m$\x1b[0m ' + cmd)
            commandBufferRef.current = cmd
          }
        } else if (data === '\x1b[B') {
          // Down arrow - history
          if (historyIndexRef.current < historyRef.current.length - 1) {
            historyIndexRef.current++
            const cmd = historyRef.current[historyIndexRef.current] || ''
            terminal.write('\r\x1b[36m$\x1b[0m ' + ' '.repeat(commandBufferRef.current.length))
            terminal.write('\r\x1b[36m$\x1b[0m ' + cmd)
            commandBufferRef.current = cmd
          } else {
            historyIndexRef.current = historyRef.current.length
            terminal.write('\r\x1b[36m$\x1b[0m ' + ' '.repeat(commandBufferRef.current.length))
            terminal.write('\r\x1b[36m$\x1b[0m ')
            commandBufferRef.current = ''
          }
        }
      } else if (code === 3) {
        // Ctrl+C
        terminal.write('^C')
        commandBufferRef.current = ''
        terminal.write('\r\n\x1b[36m$\x1b[0m ')
      } else if (code >= 32) {
        // Printable characters
        commandBufferRef.current += data
        terminal.write(data)
      }
    })

    // Connect to WebSocket if URL provided
    if (wsUrl) {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        terminal.writeln('\x1b[32m[Connected to PTY]\x1b[0m')
      }

      ws.onmessage = (event) => {
        terminal.write(event.data)
      }

      ws.onclose = () => {
        terminal.writeln('\r\n\x1b[31m[Disconnected]\x1b[0m')
      }

      ws.onerror = () => {
        terminal.writeln('\r\n\x1b[31m[Connection error]\x1b[0m')
      }
    }

    // Notify ready
    if (onReady) {
      onReady(terminal)
    }

    // Handle resize
    const handleResize = () => {
      fitAddon.fit()
    }
    window.addEventListener('resize', handleResize)

    // ResizeObserver for container size changes
    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit()
    })
    resizeObserver.observe(containerRef.current)

    return () => {
      window.removeEventListener('resize', handleResize)
      resizeObserver.disconnect()
      terminal.dispose()
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [wsUrl, welcomeMessage, onData, onReady, handleCommand])

  return (
    <div
      ref={containerRef}
      className={`w-full h-full ${className}`}
      style={{ background: '#0a0a0a' }}
    />
  )
}

// Export Terminal type for external use
export type { Terminal }
