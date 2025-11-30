# BBX Console — UI/UX Specification v2.0

## ФИЛОСОФИЯ ДИЗАЙНА

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "Один экран. Один input. Вся мощь."                          │
│                                                                 │
│   BBX Console — это не dashboard с 50 графиками.                │
│   Это терминал нового поколения для управления AI агентами.     │
│   Минимум действий. Максимум результата.                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Принципы

1. **Single View** — всё на одном экране, без навигации
2. **Input First** — главный элемент это поле ввода
3. **Command Palette** — всё остальное через ⌘K
4. **Popups > Pages** — детали показываем поверх, не уходим
5. **Information Density** — много данных компактно
6. **Terminal Aesthetic** — monospace, минимум украшений
7. **Dark by Default** — тёмная тема как основа

### Вдохновение

- **Linear** — скорость, keyboard-first
- **Raycast** — command palette, минимализм
- **Warp** — современный терминал
- **Vercel** — чистота, типографика
- **Superhuman** — keyboard shortcuts везде

---

## СТРУКТУРА ИНТЕРФЕЙСА

### Основной Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER (48px)                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ MAIN CONTENT                                                                │
│                                                                             │
│   ┌─ COMMAND INPUT ───────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌─ AGENTS PANEL ────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌─ LIVE OUTPUT ─────────────────────────────────────────────────────┐    │
│   │                                                                    │    │
│   │                                                                    │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ STATUS BAR (32px)                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Размеры и отступы

```yaml
layout:
  max_width: 1200px  # Центрируется на больших экранах
  padding:
    horizontal: 24px
    vertical: 16px
  gaps:
    between_sections: 16px
    
header:
  height: 48px
  
status_bar:
  height: 32px
  
main_content:
  height: calc(100vh - 48px - 32px)  # Всё оставшееся
```

---

## КОМПОНЕНТЫ

### 1. HEADER

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ◆ BBX                                              ⌘K    ●  connected     │
└─────────────────────────────────────────────────────────────────────────────┘
     │                                                 │         │
     └─ Logo + Name                                    │         └─ Connection status
                                                       └─ Command palette hint
```

#### Спецификация Header

```yaml
header:
  layout: flex, justify-between, align-center
  background: var(--bg-primary)
  border_bottom: 1px solid var(--border)
  padding: 0 24px
  height: 48px

  left_section:
    - logo:
        icon: "◆"  # или SVG
        size: 20px
        color: var(--accent)
    - title:
        text: "BBX"
        font: var(--font-mono)
        size: 14px
        weight: 600
        color: var(--text-primary)
        margin_left: 8px

  right_section:
    layout: flex, align-center, gap-16px
    
    - command_hint:
        text: "⌘K"
        font: var(--font-mono)
        size: 12px
        color: var(--text-muted)
        background: var(--bg-secondary)
        padding: 4px 8px
        border_radius: 4px
        cursor: pointer
        hover: 
          background: var(--bg-tertiary)
    
    - connection_status:
        layout: flex, align-center, gap-6px
        - indicator:
            type: circle
            size: 8px
            color: 
              connected: var(--green)
              disconnected: var(--red)
              connecting: var(--yellow)
            animation:
              connecting: pulse 1s infinite
        - text:
            content: "connected" | "disconnected" | "connecting..."
            font: var(--font-mono)
            size: 12px
            color: var(--text-muted)
```

---

### 2. COMMAND INPUT

**Главный элемент интерфейса. Всё начинается здесь.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  >  Добавить систему уведомлений с email и push                    ⏎ Run   │
└─────────────────────────────────────────────────────────────────────────────┘
   │  │                                                                  │
   │  └─ Input text                                                      └─ Run button
   └─ Prompt symbol
```

#### Состояния Command Input

```
┌─ EMPTY STATE ────────────────────────────────────────────────────────────────┐
│  >  What do you want to build?                                       ⏎ Run  │
└──────────────────────────────────────────────────────────────────────────────┘
     placeholder, muted color

┌─ TYPING STATE ───────────────────────────────────────────────────────────────┐
│  >  Добавить авторизацию с OAu█                                      ⏎ Run  │
└──────────────────────────────────────────────────────────────────────────────┘
     active input, cursor visible

┌─ RUNNING STATE ──────────────────────────────────────────────────────────────┐
│  ◐  Добавить авторизацию с OAuth                                    ■ Stop  │
└──────────────────────────────────────────────────────────────────────────────┘
     spinner animation, Stop button

┌─ COMPLETED STATE ────────────────────────────────────────────────────────────┐
│  ✓  Добавить авторизацию с OAuth                              02:34  ↺ Rerun│
└──────────────────────────────────────────────────────────────────────────────┘
     checkmark, duration, Rerun option

┌─ ERROR STATE ────────────────────────────────────────────────────────────────┐
│  ✗  Добавить авторизацию с OAuth                              Error  ↺ Retry│
└──────────────────────────────────────────────────────────────────────────────┘
     error icon (red), Retry option
```

#### Спецификация Command Input

```yaml
command_input:
  container:
    background: var(--bg-secondary)
    border: 1px solid var(--border)
    border_radius: 8px
    padding: 0
    height: 48px
    margin_bottom: 16px
    
    focus:
      border_color: var(--accent)
      box_shadow: 0 0 0 2px var(--accent-alpha-20)
    
    running:
      border_color: var(--blue)
    
    error:
      border_color: var(--red)
    
    success:
      border_color: var(--green)

  layout: flex, align-center

  prompt_symbol:
    content: ">"
    font: var(--font-mono)
    size: 16px
    color: var(--accent)
    padding: 0 12px 0 16px
    flex_shrink: 0
    
    states:
      running: 
        content: "◐"  # spinner
        animation: spin 1s linear infinite
      completed:
        content: "✓"
        color: var(--green)
      error:
        content: "✗"
        color: var(--red)

  input_field:
    flex: 1
    background: transparent
    border: none
    outline: none
    font: var(--font-mono)
    size: 14px
    color: var(--text-primary)
    
    placeholder:
      content: "What do you want to build?"
      color: var(--text-muted)
    
    disabled:  # when running
      pointer_events: none
      opacity: 0.7

  right_section:
    padding: 0 8px
    flex_shrink: 0
    layout: flex, align-center, gap-8px
    
    duration:  # shown when completed
      font: var(--font-mono)
      size: 12px
      color: var(--text-muted)
    
    action_button:
      background: var(--accent)
      color: var(--bg-primary)
      font: var(--font-mono)
      size: 12px
      weight: 500
      padding: 6px 12px
      border_radius: 4px
      cursor: pointer
      
      hover:
        background: var(--accent-hover)
      
      variants:
        run:
          text: "⏎ Run"
          background: var(--accent)
        stop:
          text: "■ Stop"
          background: var(--red)
        rerun:
          text: "↺ Rerun"
          background: var(--bg-tertiary)
          color: var(--text-primary)
        retry:
          text: "↺ Retry"
          background: var(--orange)

keyboard_shortcuts:
  - Enter: Run task (when input focused)
  - Escape: Clear input / Cancel running
  - Cmd+Enter: Run task (global)
  - Up/Down: Navigate history
```

---

### 3. AGENTS PANEL

**Компактное отображение всех агентов и их статуса.**

```
┌─ AGENTS ─────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ● architect   ████████░░░░░░░░  "проектирую схему базы данных..."          │
│  ◐ coder       ███░░░░░░░░░░░░░  "пишу backend/auth/oauth.py"               │
│  ○ reviewer    waiting                                                       │
│  ○ tester      waiting                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Состояния агента (одна строка)

```
● architect   ████████░░░░░░░░  "проектирую схему..."     # active (green dot)
◐ coder       ███░░░░░░░░░░░░░  "пишу код..."             # working (blue spinner)
○ reviewer    waiting                                      # idle (gray dot)
✓ tester      done in 00:45                               # completed (green check)
✗ analyzer    error: timeout                              # error (red x)
◷ assistant   queued (3rd)                                # queued (clock)
```

#### Спецификация Agents Panel

```yaml
agents_panel:
  container:
    background: var(--bg-secondary)
    border: 1px solid var(--border)
    border_radius: 8px
    padding: 12px 16px
    margin_bottom: 16px

  header:
    text: "AGENTS"
    font: var(--font-mono)
    size: 10px
    weight: 600
    color: var(--text-muted)
    letter_spacing: 0.5px
    margin_bottom: 8px

  agent_list:
    display: flex
    flex_direction: column
    gap: 6px

  agent_row:
    display: flex
    align_items: center
    height: 24px
    gap: 12px
    
    status_indicator:
      width: 8px
      flex_shrink: 0
      
      variants:
        idle:
          type: circle
          color: var(--text-muted)
          filled: false  # outline only
        working:
          type: spinner
          color: var(--blue)
          animation: spin 1s linear infinite
        active:
          type: circle
          color: var(--green)
          filled: true
        completed:
          type: checkmark
          color: var(--green)
        error:
          type: x
          color: var(--red)
        queued:
          type: clock
          color: var(--yellow)
    
    agent_name:
      font: var(--font-mono)
      size: 13px
      color: var(--text-primary)
      width: 80px
      flex_shrink: 0
    
    progress_bar:
      flex: 0 0 120px
      height: 4px
      background: var(--bg-tertiary)
      border_radius: 2px
      overflow: hidden
      
      fill:
        height: 100%
        background: var(--accent)
        border_radius: 2px
        transition: width 0.3s ease
      
      # Hide when not working
      visibility:
        working: visible
        other: hidden
    
    status_text:
      flex: 1
      font: var(--font-mono)
      size: 12px
      color: var(--text-muted)
      white_space: nowrap
      overflow: hidden
      text_overflow: ellipsis
      
      # Different styles for different states
      working:
        content: quoted task description
        color: var(--text-secondary)
      idle:
        content: "waiting"
        color: var(--text-muted)
      completed:
        content: "done in {duration}"
        color: var(--green-muted)
      error:
        content: "error: {message}"
        color: var(--red-muted)
      queued:
        content: "queued ({position})"
        color: var(--yellow-muted)

  interactions:
    agent_row:
      cursor: pointer
      hover:
        background: var(--bg-tertiary)
        border_radius: 4px
      click:
        action: open_agent_popup
```

#### Collapsed State (когда idle)

Если ничего не запущено, показываем компактнее:

```
┌─ AGENTS ─────────────────────────────────────────────────────────────────────┐
│  ○ architect  ○ coder  ○ reviewer  ○ tester                     all idle    │
└──────────────────────────────────────────────────────────────────────────────┘
```

```yaml
agents_panel_collapsed:
  condition: all agents idle AND no task running
  layout: single row
  show: agent names with status dots inline
  right: "all idle" text
```

---

### 4. LIVE OUTPUT

**Поток вывода от агентов. Как терминал.**

```
┌─ OUTPUT ─────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  12:34:56  [architect]  Анализирую требования задачи...                     │
│  12:34:58  [architect]  Определяю архитектуру: OAuth 2.0 + JWT              │
│  12:35:02  [architect]  Создаю файл: docs/auth_design.md                    │
│  12:35:05  [architect]  ✓ Дизайн готов                                      │
│  12:35:06  [coder]      Начинаю реализацию backend...                       │
│  12:35:08  [coder]      Создаю файл: backend/auth/oauth.py                  │
│  12:35:15  [coder]      Создаю файл: backend/auth/jwt.py                    │
│  12:35:20  [coder]      Реализую endpoint: POST /auth/login                 │
│  12:35:28  [coder]      Реализую endpoint: POST /auth/callback              │
│  █                                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Типы сообщений

```
# Standard log
12:34:56  [architect]  Анализирую требования...

# Success
12:35:05  [architect]  ✓ Дизайн готов

# Error
12:35:10  [coder]      ✗ Ошибка: файл не найден

# Warning
12:35:12  [reviewer]   ⚠ Обнаружена потенциальная проблема

# File created
12:35:08  [coder]      + backend/auth/oauth.py

# File modified
12:35:15  [coder]      ~ backend/auth/jwt.py (modified)

# File deleted
12:35:18  [coder]      - temp/cache.py (deleted)

# Agent transition
12:35:20  ─────────────  architect → coder  ─────────────

# System message
12:35:25  [system]     Workflow 50% complete

# Code block
12:35:30  [coder]      Created function:
                       │ def authenticate(token: str) -> User:
                       │     payload = decode_jwt(token)
                       │     return User.get(payload['user_id'])
```

#### Спецификация Live Output

```yaml
live_output:
  container:
    background: var(--bg-primary)
    border: 1px solid var(--border)
    border_radius: 8px
    flex: 1  # Takes remaining space
    min_height: 200px
    overflow: hidden
    display: flex
    flex_direction: column

  header:
    text: "OUTPUT"
    font: var(--font-mono)
    size: 10px
    weight: 600
    color: var(--text-muted)
    letter_spacing: 0.5px
    padding: 12px 16px 8px
    border_bottom: 1px solid var(--border)
    
    right_actions:
      display: flex
      gap: 8px
      
      - clear_button:
          text: "Clear"
          size: 10px
          color: var(--text-muted)
          cursor: pointer
          hover:
            color: var(--text-primary)
      
      - scroll_toggle:
          text: "Auto-scroll"
          size: 10px
          color: var(--text-muted)
          states:
            on: color: var(--accent)
            off: color: var(--text-muted)

  output_area:
    flex: 1
    overflow_y: auto
    padding: 8px 16px
    font: var(--font-mono)
    size: 12px
    line_height: 1.6

  log_line:
    display: flex
    padding: 2px 0
    
    timestamp:
      color: var(--text-muted)
      width: 70px
      flex_shrink: 0
      opacity: 0.6
    
    agent_badge:
      color: var(--text-secondary)
      width: 100px
      flex_shrink: 0
      
      format: "[{agent_name}]"
      
      colors_per_agent:
        architect: var(--purple)
        coder: var(--blue)
        reviewer: var(--orange)
        tester: var(--green)
        system: var(--text-muted)
    
    message:
      color: var(--text-primary)
      flex: 1
      word_break: break-word
      
      # Prefix icons
      prefixes:
        success: "✓ " color: var(--green)
        error: "✗ " color: var(--red)
        warning: "⚠ " color: var(--yellow)
        file_add: "+ " color: var(--green)
        file_mod: "~ " color: var(--blue)
        file_del: "- " color: var(--red)

  code_block:
    background: var(--bg-secondary)
    border_radius: 4px
    padding: 8px 12px
    margin: 4px 0 4px 170px  # Aligned with message column
    font: var(--font-mono)
    size: 11px
    
    line_prefix:
      content: "│ "
      color: var(--border)

  transition_divider:
    display: flex
    align_items: center
    gap: 12px
    margin: 8px 0
    color: var(--text-muted)
    
    line:
      flex: 1
      height: 1px
      background: var(--border)
    
    text:
      font: var(--font-mono)
      size: 10px
      white_space: nowrap

  empty_state:
    display: flex
    align_items: center
    justify_content: center
    height: 100%
    color: var(--text-muted)
    font: var(--font-mono)
    size: 13px
    text: "No output yet. Run a task to see results."

  cursor:
    # Blinking cursor at the end when running
    when: task_running
    content: "█"
    animation: blink 1s step-end infinite

keyboard_shortcuts:
  - Cmd+L: Clear output
  - Cmd+Shift+S: Toggle auto-scroll
  - Cmd+F: Search in output
```

---

### 5. STATUS BAR

**Компактная строка внизу с ключевыми метриками.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Memory: 24 HOT  │  Ring: 3 queued  │  Tasks: 2/5 done  │  ⏱ 00:02:34       │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Спецификация Status Bar

```yaml
status_bar:
  container:
    height: 32px
    background: var(--bg-secondary)
    border_top: 1px solid var(--border)
    padding: 0 24px
    display: flex
    align_items: center
    justify_content: space-between

  left_section:
    display: flex
    align_items: center
    gap: 0
    
  status_item:
    display: flex
    align_items: center
    padding: 0 16px
    height: 100%
    border_right: 1px solid var(--border)
    cursor: pointer
    
    hover:
      background: var(--bg-tertiary)
    
    label:
      font: var(--font-mono)
      size: 11px
      color: var(--text-muted)
      margin_right: 6px
    
    value:
      font: var(--font-mono)
      size: 11px
      color: var(--text-primary)
      weight: 500
    
    click:
      action: open_popup
    
  items:
    - memory:
        label: "Memory:"
        value: "{hot_count} HOT"
        popup: memory_popup
        
    - ring:
        label: "Ring:"
        value: "{queued_count} queued"
        popup: ring_popup
        
    - tasks:
        label: "Tasks:"
        value: "{completed}/{total} done"
        popup: tasks_popup

  right_section:
    display: flex
    align_items: center
    gap: 16px
    
    timer:
      font: var(--font-mono)
      size: 12px
      color: var(--text-muted)
      
      states:
        running:
          prefix: "⏱ "
          color: var(--blue)
        completed:
          prefix: "✓ "
          color: var(--green)
        idle:
          content: "idle"
          color: var(--text-muted)
```

---

### 6. COMMAND PALETTE (⌘K)

**Единая точка доступа ко всему остальному.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                                                                             │
│         ┌─────────────────────────────────────────────────────────┐        │
│         │  >  run feature                                         │        │
│         └─────────────────────────────────────────────────────────┘        │
│                                                                             │
│         ┌─────────────────────────────────────────────────────────┐        │
│         │                                                         │        │
│         │  WORKFLOWS                                              │        │
│         │  ▸ feature_implementation.bbx              ⏎ to run    │        │
│         │  ▸ parallel_review.bbx                     ⏎ to run    │        │
│         │  ▸ bug_fix.bbx                             ⏎ to run    │        │
│         │                                                         │        │
│         │  RECENT                                                 │        │
│         │  ▸ "добавить систему уведомлений"          2 min ago   │        │
│         │  ▸ "рефакторинг модуля auth"               1 hour ago  │        │
│         │                                                         │        │
│         │  COMMANDS                                               │        │
│         │  ▸ memory         View memory tiers             ⌘M     │        │
│         │  ▸ agents         Agent details                 ⌘A     │        │
│         │  ▸ ring           Queue status                  ⌘R     │        │
│         │  ▸ history        Past runs                     ⌘H     │        │
│         │  ▸ settings       Configuration                 ⌘,     │        │
│         │                                                         │        │
│         └─────────────────────────────────────────────────────────┘        │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         backdrop (semi-transparent black)
```

#### Спецификация Command Palette

```yaml
command_palette:
  trigger:
    shortcut: "Cmd+K" | "Ctrl+K"
    click: header hint
  
  overlay:
    position: fixed
    inset: 0
    background: rgba(0, 0, 0, 0.6)
    backdrop_filter: blur(4px)
    display: flex
    justify_content: center
    align_items: flex-start
    padding_top: 15vh
    z_index: 1000
    
    click_outside: close
    escape: close

  container:
    width: 560px
    max_height: 70vh
    background: var(--bg-primary)
    border: 1px solid var(--border)
    border_radius: 12px
    box_shadow: 0 24px 48px rgba(0, 0, 0, 0.4)
    overflow: hidden

  search_input:
    padding: 16px
    border_bottom: 1px solid var(--border)
    
    input:
      width: 100%
      background: transparent
      border: none
      outline: none
      font: var(--font-mono)
      size: 15px
      color: var(--text-primary)
      
      placeholder:
        content: "Type a command or search..."
        color: var(--text-muted)
      
      prefix:
        content: "> "
        color: var(--accent)

  results:
    max_height: calc(70vh - 60px)
    overflow_y: auto
    padding: 8px

  result_group:
    margin_bottom: 16px
    
    header:
      font: var(--font-mono)
      size: 10px
      weight: 600
      color: var(--text-muted)
      letter_spacing: 0.5px
      padding: 8px 12px 4px
    
  result_item:
    display: flex
    align_items: center
    justify_content: space-between
    padding: 8px 12px
    border_radius: 6px
    cursor: pointer
    
    hover:
      background: var(--bg-secondary)
    
    selected:  # keyboard navigation
      background: var(--accent-alpha-10)
      border: 1px solid var(--accent-alpha-30)
    
    left:
      display: flex
      align_items: center
      gap: 8px
      
      icon:
        content: "▸"
        color: var(--text-muted)
      
      text:
        font: var(--font-mono)
        size: 13px
        color: var(--text-primary)
    
    right:
      font: var(--font-mono)
      size: 11px
      color: var(--text-muted)

  keyboard_hints:
    padding: 8px 16px
    border_top: 1px solid var(--border)
    display: flex
    gap: 16px
    
    hint:
      font: var(--font-mono)
      size: 10px
      color: var(--text-muted)
      
      key:
        background: var(--bg-secondary)
        padding: 2px 6px
        border_radius: 3px
        margin_right: 4px

commands:
  - id: memory
    name: "memory"
    description: "View memory tiers"
    shortcut: "⌘M"
    action: open_memory_popup
    
  - id: agents
    name: "agents"
    description: "Agent details"
    shortcut: "⌘A"
    action: open_agents_popup
    
  - id: ring
    name: "ring"
    description: "Queue status"
    shortcut: "⌘R"
    action: open_ring_popup
    
  - id: history
    name: "history"
    description: "Past runs"
    shortcut: "⌘H"
    action: open_history_popup
    
  - id: settings
    name: "settings"
    description: "Configuration"
    shortcut: "⌘,"
    action: open_settings_popup

  - id: clear
    name: "clear"
    description: "Clear output"
    shortcut: "⌘L"
    action: clear_output

  - id: stop
    name: "stop"
    description: "Stop current task"
    shortcut: "⌘."
    action: stop_task
```

---

### 7. POPUP PANELS

**Детальная информация показывается в popup поверх основного интерфейса.**

#### 7.1 Memory Popup

```
┌─ MEMORY ─────────────────────────────────────────────────────────── [×] ────┐
│                                                                             │
│  ┌─ HOT (4 items) ───────────────────────────────────────────────────────┐ │
│  │  current_task          "oauth implementation..."       2 min ago  📌  │ │
│  │  project_context       "bbx console web app..."        5 min ago      │ │
│  │  user_preferences      "dark theme, vim keys..."      12 min ago      │ │
│  │  recent_error          "TypeError in auth.py"          8 min ago      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ WARM (12 items) ─────────────────────────────────────────────────────┐ │
│  │  session_history       [compressed]                   30 min ago      │ │
│  │  code_context          [compressed]                   45 min ago      │ │
│  │  ...                                                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ COOL (8 items) ──────────────────────────────────────────────────────┐ │
│  │  ...                                                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ COLD (3 items) ──────────────────────────────────────────────────────┐ │
│  │  ...                                                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────  │
│  Stats: 27 items │ Hit rate: 94% │ Size: 2.4 MB                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.2 Ring Popup

```
┌─ AGENT RING ─────────────────────────────────────────────────────── [×] ────┐
│                                                                             │
│  SUBMISSION QUEUE (SQ)                    COMPLETION QUEUE (CQ)            │
│  ┌────────────────────────────┐          ┌────────────────────────────┐   │
│  │  ▸ task_003  HIGH    ◐    │          │  ✓ task_001  done   45ms   │   │
│  │  ▸ task_004  NORMAL  ○    │          │  ✓ task_002  done   120ms  │   │
│  │  ▸ task_005  NORMAL  ○    │          │                            │   │
│  │  ▸ task_006  LOW     ○    │          │                            │   │
│  └────────────────────────────┘          └────────────────────────────┘   │
│                                                                             │
│  WORKERS                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [1] ████████████░░░░░░░░  task_003 (architect)                     │   │
│  │  [2] ██████░░░░░░░░░░░░░░  task_004 (coder)                         │   │
│  │  [3] idle                                                            │   │
│  │  [4] idle                                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────  │
│  Throughput: 12 ops/s │ Latency p50: 45ms │ Utilization: 50%               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.3 History Popup

```
┌─ HISTORY ────────────────────────────────────────────────────────── [×] ────┐
│                                                                             │
│  ┌─ TODAY ───────────────────────────────────────────────────────────────┐ │
│  │  ✓  "добавить систему уведомлений"             02:34     2 min ago   │ │
│  │  ✓  "рефакторинг модуля auth"                  05:12     1 hour ago  │ │
│  │  ✗  "интеграция payment API"                   error     2 hours ago │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ YESTERDAY ───────────────────────────────────────────────────────────┐ │
│  │  ✓  "настроить CI/CD pipeline"                 08:45     yesterday   │ │
│  │  ✓  "написать тесты для API"                   12:20     yesterday   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Click on item to view details or rerun                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Спецификация Popup Panel

```yaml
popup_panel:
  overlay:
    position: fixed
    inset: 0
    background: rgba(0, 0, 0, 0.5)
    backdrop_filter: blur(2px)
    display: flex
    justify_content: center
    align_items: center
    z_index: 900

  container:
    width: 700px
    max_width: 90vw
    max_height: 80vh
    background: var(--bg-primary)
    border: 1px solid var(--border)
    border_radius: 12px
    box_shadow: 0 16px 32px rgba(0, 0, 0, 0.3)
    overflow: hidden
    display: flex
    flex_direction: column

  header:
    display: flex
    justify_content: space-between
    align_items: center
    padding: 16px 20px
    border_bottom: 1px solid var(--border)
    
    title:
      font: var(--font-mono)
      size: 13px
      weight: 600
      color: var(--text-primary)
    
    close_button:
      width: 24px
      height: 24px
      display: flex
      align_items: center
      justify_content: center
      border_radius: 4px
      cursor: pointer
      color: var(--text-muted)
      
      hover:
        background: var(--bg-secondary)
        color: var(--text-primary)
      
      content: "×"
      font_size: 18px

  content:
    flex: 1
    overflow_y: auto
    padding: 16px 20px

  footer:
    padding: 12px 20px
    border_top: 1px solid var(--border)
    font: var(--font-mono)
    size: 11px
    color: var(--text-muted)

keyboard:
  Escape: close popup
```

---

## ЦВЕТОВАЯ СХЕМА

### Dark Theme (Primary)

```yaml
colors:
  # Backgrounds
  bg_primary: "#0D0D0D"      # Main background
  bg_secondary: "#161616"    # Cards, panels
  bg_tertiary: "#1F1F1F"     # Hover states
  
  # Text
  text_primary: "#FAFAFA"    # Main text
  text_secondary: "#A3A3A3"  # Secondary text
  text_muted: "#666666"      # Muted text, labels
  
  # Borders
  border: "#262626"          # Default border
  border_focus: "#404040"    # Focus state
  
  # Accent
  accent: "#3B82F6"          # Primary accent (blue)
  accent_hover: "#2563EB"    # Accent hover
  accent_alpha_10: "rgba(59, 130, 246, 0.1)"
  accent_alpha_20: "rgba(59, 130, 246, 0.2)"
  accent_alpha_30: "rgba(59, 130, 246, 0.3)"
  
  # Semantic
  green: "#22C55E"           # Success
  green_muted: "#166534"     # Success muted
  red: "#EF4444"             # Error
  red_muted: "#991B1B"       # Error muted
  yellow: "#EAB308"          # Warning
  yellow_muted: "#854D0E"    # Warning muted
  orange: "#F97316"          # Warning alt
  purple: "#A855F7"          # Architect agent
  blue: "#3B82F6"            # Coder agent / working state
  
  # Special
  code_bg: "#1A1A1A"         # Code blocks
```

### Light Theme (Optional)

```yaml
colors_light:
  bg_primary: "#FFFFFF"
  bg_secondary: "#F5F5F5"
  bg_tertiary: "#EBEBEB"
  
  text_primary: "#171717"
  text_secondary: "#525252"
  text_muted: "#A3A3A3"
  
  border: "#E5E5E5"
  border_focus: "#D4D4D4"
  
  accent: "#2563EB"
  # ... etc
```

---

## ТИПОГРАФИКА

```yaml
typography:
  fonts:
    mono: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace"
    # Только monospace. Без sans-serif.
  
  sizes:
    xs: "10px"    # Labels, hints
    sm: "11px"    # Status bar, badges
    base: "12px"  # Body text, logs
    md: "13px"    # UI elements
    lg: "14px"    # Input fields
    xl: "15px"    # Command palette input
  
  weights:
    normal: 400
    medium: 500
    semibold: 600
  
  line_heights:
    tight: 1.2
    normal: 1.5
    relaxed: 1.6   # For logs/output
```

---

## АНИМАЦИИ

```yaml
animations:
  # Transitions
  transition_fast: "0.1s ease"
  transition_normal: "0.2s ease"
  transition_slow: "0.3s ease"
  
  # Spinner
  spin:
    keyframes:
      from: { transform: "rotate(0deg)" }
      to: { transform: "rotate(360deg)" }
    duration: "1s"
    timing: "linear"
    iteration: "infinite"
  
  # Cursor blink
  blink:
    keyframes:
      "0%, 50%": { opacity: 1 }
      "51%, 100%": { opacity: 0 }
    duration: "1s"
    timing: "step-end"
    iteration: "infinite"
  
  # Pulse (connection status)
  pulse:
    keyframes:
      "0%, 100%": { opacity: 1 }
      "50%": { opacity: 0.5 }
    duration: "1s"
    timing: "ease-in-out"
    iteration: "infinite"
  
  # Slide in (popups)
  slide_in:
    keyframes:
      from: { opacity: 0, transform: "translateY(-10px)" }
      to: { opacity: 1, transform: "translateY(0)" }
    duration: "0.2s"
    timing: "ease-out"
  
  # Fade in (overlay)
  fade_in:
    keyframes:
      from: { opacity: 0 }
      to: { opacity: 1 }
    duration: "0.15s"
    timing: "ease-out"

motion_preferences:
  # Respect prefers-reduced-motion
  reduced_motion:
    disable: [spin, pulse, blink]
    instant: [slide_in, fade_in]
```

---

## KEYBOARD SHORTCUTS

```yaml
global_shortcuts:
  "Cmd+K": "Open command palette"
  "Cmd+Enter": "Run task"
  "Cmd+.": "Stop current task"
  "Escape": "Close popup / Clear input"
  
  "Cmd+M": "Open memory popup"
  "Cmd+A": "Open agents popup"
  "Cmd+R": "Open ring popup"
  "Cmd+H": "Open history popup"
  "Cmd+,": "Open settings"
  
  "Cmd+L": "Clear output"
  "Cmd+Shift+S": "Toggle auto-scroll"
  "Cmd+F": "Search in output"
  
  "Up/Down": "Navigate command history (when input focused)"
  "Tab": "Autocomplete (in command palette)"

context_shortcuts:
  command_palette:
    "Up/Down": "Navigate results"
    "Enter": "Select item"
    "Escape": "Close"
  
  popup:
    "Escape": "Close"
    "Cmd+W": "Close"
```

---

## RESPONSIVE DESIGN

```yaml
breakpoints:
  sm: "640px"
  md: "768px"
  lg: "1024px"
  xl: "1280px"

responsive_rules:
  # Mobile (< 640px)
  mobile:
    - Hide keyboard shortcut hints
    - Full-width command palette
    - Smaller font sizes (-1px)
    - Stack status bar items vertically
    - Popups as full-screen modals
  
  # Tablet (640px - 1024px)
  tablet:
    - Reduce padding
    - Smaller popups
  
  # Desktop (> 1024px)
  desktop:
    - Full experience
    - Max-width 1200px centered
  
  # Wide (> 1280px)
  wide:
    - Consider split view option
    - More horizontal space for logs
```

---

## РЕАЛИЗАЦИЯ

### Структура компонентов (React)

```
frontend/
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # Single view
│   └── globals.css
│
├── components/
│   ├── layout/
│   │   ├── Header.tsx
│   │   └── StatusBar.tsx
│   │
│   ├── core/
│   │   ├── CommandInput.tsx
│   │   ├── AgentsPanel.tsx
│   │   ├── LiveOutput.tsx
│   │   └── CommandPalette.tsx
│   │
│   ├── popups/
│   │   ├── PopupWrapper.tsx
│   │   ├── MemoryPopup.tsx
│   │   ├── AgentsPopup.tsx
│   │   ├── RingPopup.tsx
│   │   ├── HistoryPopup.tsx
│   │   └── SettingsPopup.tsx
│   │
│   └── ui/
│       ├── Badge.tsx
│       ├── Button.tsx
│       ├── ProgressBar.tsx
│       └── Spinner.tsx
│
├── hooks/
│   ├── useWebSocket.ts
│   ├── useCommandPalette.ts
│   ├── useKeyboardShortcuts.ts
│   ├── useAgents.ts
│   ├── useOutput.ts
│   └── useTask.ts
│
├── stores/
│   ├── taskStore.ts
│   ├── agentsStore.ts
│   ├── outputStore.ts
│   └── uiStore.ts
│
├── lib/
│   ├── api.ts
│   ├── ws.ts
│   └── utils.ts
│
├── styles/
│   └── theme.ts             # CSS variables
│
└── types/
    └── index.ts
```

### Ключевые компоненты

#### CommandInput.tsx

```tsx
interface CommandInputProps {
  onRun: (task: string) => void;
  onStop: () => void;
  status: 'idle' | 'running' | 'completed' | 'error';
  duration?: number;
}

export function CommandInput({ onRun, onStop, status, duration }: CommandInputProps) {
  const [value, setValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);
  
  // Handle keyboard shortcuts
  useKeyboardShortcuts({
    'Enter': () => status === 'idle' && value && onRun(value),
    'Escape': () => status === 'running' ? onStop() : setValue(''),
    'Cmd+Enter': () => value && onRun(value),
  });
  
  const renderPrompt = () => {
    switch (status) {
      case 'running': return <Spinner className="text-blue-500" />;
      case 'completed': return <span className="text-green-500">✓</span>;
      case 'error': return <span className="text-red-500">✗</span>;
      default: return <span className="text-accent">&gt;</span>;
    }
  };
  
  const renderAction = () => {
    switch (status) {
      case 'running':
        return <Button variant="danger" onClick={onStop}>■ Stop</Button>;
      case 'completed':
        return (
          <>
            <span className="text-muted text-sm">{formatDuration(duration)}</span>
            <Button variant="ghost" onClick={() => onRun(value)}>↺ Rerun</Button>
          </>
        );
      case 'error':
        return <Button variant="warning" onClick={() => onRun(value)}>↺ Retry</Button>;
      default:
        return <Button onClick={() => onRun(value)} disabled={!value}>⏎ Run</Button>;
    }
  };
  
  return (
    <div className={cn(
      "flex items-center h-12 bg-secondary rounded-lg border",
      status === 'running' && "border-blue-500",
      status === 'error' && "border-red-500",
      status === 'completed' && "border-green-500",
    )}>
      <div className="px-4 flex-shrink-0">
        {renderPrompt()}
      </div>
      
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="What do you want to build?"
        disabled={status === 'running'}
        className="flex-1 bg-transparent outline-none font-mono text-sm"
      />
      
      <div className="px-2 flex items-center gap-2">
        {renderAction()}
      </div>
    </div>
  );
}
```

#### AgentsPanel.tsx

```tsx
interface Agent {
  id: string;
  name: string;
  status: 'idle' | 'working' | 'completed' | 'error' | 'queued';
  progress?: number;
  currentTask?: string;
  duration?: number;
}

export function AgentsPanel({ agents }: { agents: Agent[] }) {
  const allIdle = agents.every(a => a.status === 'idle');
  
  if (allIdle) {
    return (
      <div className="bg-secondary rounded-lg border px-4 py-3">
        <div className="flex items-center gap-3">
          {agents.map(agent => (
            <div key={agent.id} className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full border border-muted" />
              <span className="text-sm font-mono text-muted">{agent.name}</span>
            </div>
          ))}
          <span className="ml-auto text-xs text-muted">all idle</span>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-secondary rounded-lg border p-4">
      <div className="text-xs font-mono text-muted tracking-wide mb-3">AGENTS</div>
      <div className="space-y-2">
        {agents.map(agent => (
          <AgentRow key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
}

function AgentRow({ agent }: { agent: Agent }) {
  return (
    <div className="flex items-center gap-3 h-6 hover:bg-tertiary rounded px-1 -mx-1 cursor-pointer">
      <AgentStatusIcon status={agent.status} />
      <span className="font-mono text-sm w-20">{agent.name}</span>
      
      {agent.status === 'working' && (
        <div className="w-32 h-1 bg-tertiary rounded overflow-hidden">
          <div 
            className="h-full bg-accent rounded transition-all"
            style={{ width: `${agent.progress || 0}%` }}
          />
        </div>
      )}
      
      <span className="flex-1 font-mono text-xs text-muted truncate">
        {getStatusText(agent)}
      </span>
    </div>
  );
}
```

---

## ФИНАЛЬНЫЙ CHECKLIST

### Должно быть

- [ ] Один экран, без навигации
- [ ] Command input как главный элемент
- [ ] Agents panel с real-time статусами
- [ ] Live output как терминал
- [ ] Status bar с метриками
- [ ] Command Palette (⌘K)
- [ ] Popups для деталей (не страницы)
- [ ] Все keyboard shortcuts работают
- [ ] Dark theme
- [ ] Monospace везде
- [ ] WebSocket real-time updates
- [ ] Mobile responsive

### Не должно быть

- [ ] Sidebar с навигацией
- [ ] Множество страниц
- [ ] Карточки с тенями и градиентами
- [ ] Иконки где можно текстом
- [ ] Лишние цвета
- [ ] Анимации ради анимаций
- [ ] Modals для простых действий

---

## ИТОГО

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   БЫЛО: 5+ страниц, кликать везде, dashboard madness                       │
│                                                                             │
│   СТАЛО: 1 экран, пишешь → работает, terminal aesthetic                   │
│                                                                             │
│   Вдохновение: Linear + Raycast + Warp                                     │
│   Эстетика: тёмная, monospace, минимализм                                  │
│   Взаимодействие: keyboard-first, command palette                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Это полный spec. Бери и делай.
