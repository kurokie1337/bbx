# BBX Console — Technical Specification v1.0

## ПРЕАМБУЛА: ИНСТРУКЦИЯ ДЛЯ АГЕНТА

```
┌─────────────────────────────────────────────────────────────────┐
│  ПЕРЕД НАЧАЛОМ РАБОТЫ                                           │
│                                                                 │
│  Ты — команда агентов BBX. Перед реализацией ОБЯЗАТЕЛЬНО:       │
│                                                                 │
│  1. ПРОЧИТАЙ весь этот документ полностью                       │
│  2. ИЗУЧИ существующий код BBX (blackbox/)                      │
│  3. ЗАДАЙ ВОПРОСЫ если что-то неясно                            │
│  4. ПРЕДЛОЖИ улучшения если видишь проблемы                     │
│  5. СОСТАВЬ план работы                                         │
│  6. ТОЛЬКО ПОТОМ начинай кодить                                 │
│                                                                 │
│  Фаза "thinking" обязательна. Не пропускай её.                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 0: THINKING (обязательная)

### 0.1 Анализ существующей системы

Агент ДОЛЖЕН выполнить перед началом:

```yaml
thinking_tasks:
  - name: "Изучение BBX Core"
    actions:
      - Прочитать blackbox/core/runtime.py — понять как работает workflow engine
      - Прочитать blackbox/core/dag.py — понять DAG execution
      - Прочитать blackbox/core/v2/context_tiering.py — понять memory model
      - Прочитать blackbox/core/v2/ring.py — понять AgentRing scheduling
      - Прочитать blackbox/a2a/client.py — понять A2A protocol
      - Прочитать blackbox/core/adapters/agent_sdk.py — понять agent integration
      - Прочитать blackbox/mcp/server.py — понять MCP integration
    output: "docs/analysis/BBX_CORE_ANALYSIS.md"

  - name: "Изучение агентов"
    actions:
      - Прочитать .claude/agents/architect.md
      - Прочитать .claude/agents/coder.md
      - Прочитать .claude/agents/reviewer.md
      - Прочитать .claude/agents/tester.md
    output: "docs/analysis/AGENTS_ANALYSIS.md"

  - name: "Изучение существующих workflows"
    actions:
      - Прочитать все .bbx файлы в examples/
      - Понять паттерны использования
    output: "docs/analysis/WORKFLOWS_ANALYSIS.md"

  - name: "Архитектурные решения"
    questions_to_answer:
      - "Как лучше интегрировать FastAPI с BBX runtime?"
      - "Как организовать real-time updates через WebSocket?"
      - "Как визуализировать DAG в браузере?"
      - "Как показывать ContextTiering в UI?"
      - "Как интегрировать AgentRing метрики?"
      - "Какой формат данных для A2A visualization?"
    output: "docs/analysis/ARCHITECTURE_DECISIONS.md"

  - name: "Risk Assessment"
    identify:
      - "Потенциальные проблемы интеграции"
      - "Узкие места производительности"
      - "Сложные компоненты"
      - "Зависимости между модулями"
    output: "docs/analysis/RISKS.md"

  - name: "План работы"
    create:
      - "Порядок реализации модулей"
      - "Зависимости между задачами"
      - "Оценка времени на каждый модуль"
      - "Критический путь"
    output: "docs/analysis/WORK_PLAN.md"
```

### 0.2 Вопросы для уточнения

Агент МОЖЕТ задать эти вопросы перед началом:

```yaml
clarification_questions:
  architecture:
    - "Нужна ли поддержка нескольких пользователей или single-user?"
    - "Должен ли Task Manager хранить данные персистентно или in-memory достаточно?"
    - "Какой уровень детализации нужен для визуализации DAG?"
    - "Нужна ли история выполнения workflows?"

  integration:
    - "BBX Console будет частью репозитория BBX или отдельным проектом?"
    - "Как запускать: отдельный процесс или встроен в BBX?"
    - "Нужна ли интеграция с существующим MCP server?"

  ui_ux:
    - "Есть ли предпочтения по UI framework (помимо shadcn)?"
    - "Dark mode обязателен?"
    - "Mobile-first или desktop-first?"

  scope:
    - "Все фичи в MVP или можно приоритизировать?"
    - "Какие фичи критичны, какие nice-to-have?"
```

### 0.3 Предложения по улучшению

Агент ДОЛЖЕН предложить улучшения если видит:

```yaml
improvement_areas:
  - "Архитектурные улучшения"
  - "Дополнительные фичи которые имеет смысл добавить"
  - "Упрощения которые не потеряют функциональность"
  - "Альтернативные подходы к реализации"
```

---

## SECTION 1: PROJECT OVERVIEW

### 1.1 Что такое BBX Console

BBX Console — это веб-приложение которое:

1. **Dashboard для BBX** — управление и мониторинг AI Agent OS
2. **Task Manager** — полноценный продукт построенный на BBX агентах
3. **Демонстрация** — showcase всех возможностей BBX

### 1.2 Цели проекта

```yaml
primary_goals:
  - "Создать UI для управления BBX"
  - "Построить реальный продукт на BBX агентах"
  - "Протестировать все компоненты BBX end-to-end"
  - "Получить визуальное демо для презентаций"

secondary_goals:
  - "Улучшить BBX в процессе (найти и пофиксить баги)"
  - "Создать reference implementation"
  - "Документировать best practices"
```

### 1.3 BBX Components Coverage Matrix

**КАЖДЫЙ компонент BBX должен быть задействован:**

| BBX Component | Где используется в Console | Как тестируется |
|---------------|---------------------------|-----------------|
| **runtime.py** (Workflow Engine) | Workflow Manager — запуск .bbx | Запуск workflows через UI |
| **dag.py** (DAG Execution) | DAG Visualizer — граф зависимостей | Визуализация + parallel execution |
| **BBXv6Parser** | Workflow Editor — парсинг .bbx | Загрузка и валидация workflows |
| **context_tiering.py** (MGLRU) | Memory Explorer — 4 tiers | Просмотр HOT/WARM/COOL/COLD |
| **ring.py** (AgentRing) | Agent Monitor — очереди, приоритеты | Метрики SQ/CQ, scheduling |
| **agent_sdk.py** (Agent Adapter) | Task Manager — вызов агентов | AI task decomposition |
| **a2a/client.py** (A2A Protocol) | A2A Playground — межагентная связь | Task lifecycle, Agent Cards |
| **mcp/server.py** (MCP) | MCP Tools — внешние интеграции | Tool invocation |
| **agents/*.md** | Agent Profiles — информация об агентах | Все 4 агента работают |

---

## SECTION 2: ARCHITECTURE

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BBX Console                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FRONTEND (Next.js 14)                     │   │
│  │                                                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ Workflow │ │  Agent   │ │  Memory  │ │   Task   │       │   │
│  │  │ Manager  │ │ Monitor  │ │ Explorer │ │ Manager  │       │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │   │
│  │       │            │            │            │              │   │
│  │  ┌────┴────────────┴────────────┴────────────┴────┐        │   │
│  │  │              WebSocket Client                   │        │   │
│  │  └────────────────────┬───────────────────────────┘        │   │
│  └───────────────────────┼───────────────────────────────────┘   │
│                          │                                        │
│  ════════════════════════╪════════════════════════════════════   │
│                          │ HTTP + WebSocket                       │
│  ════════════════════════╪════════════════════════════════════   │
│                          │                                        │
│  ┌───────────────────────┼───────────────────────────────────┐   │
│  │                 BACKEND (FastAPI)                          │   │
│  │                       │                                    │   │
│  │  ┌────────────────────┴────────────────────┐              │   │
│  │  │           WebSocket Manager              │              │   │
│  │  └────────────────────┬────────────────────┘              │   │
│  │                       │                                    │   │
│  │  ┌──────────┐ ┌───────┴───────┐ ┌──────────┐              │   │
│  │  │   REST   │ │   BBX Bridge  │ │  Auth    │              │   │
│  │  │   API    │ │               │ │  (JWT)   │              │   │
│  │  └────┬─────┘ └───────┬───────┘ └────┬─────┘              │   │
│  └───────┼───────────────┼──────────────┼────────────────────┘   │
│          │               │              │                         │
│  ════════╪═══════════════╪══════════════╪═════════════════════   │
│          │               │              │                         │
│  ┌───────┴───────────────┴──────────────┴────────────────────┐   │
│  │                      BBX CORE                              │   │
│  │                                                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │   │
│  │  │ Runtime  │ │   DAG    │ │ Context  │ │  Agent   │      │   │
│  │  │ Engine   │ │ Executor │ │ Tiering  │ │  Ring    │      │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │   │
│  │                                                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │   │
│  │  │   A2A    │ │   MCP    │ │  Agent   │ │  Config  │      │   │
│  │  │  Client  │ │  Server  │ │   SDK    │ │          │      │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     DATA LAYER                               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │   │
│  │  │PostgreSQL│ │  Redis   │ │  Files   │                     │   │
│  │  │(tasks,   │ │(sessions,│ │(.bbx,    │                     │   │
│  │  │ users)   │ │ realtime)│ │ agents)  │                     │   │
│  │  └──────────┘ └──────────┘ └──────────┘                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
bbx-console/
├── docs/
│   ├── analysis/                    # Phase 0 outputs
│   │   ├── BBX_CORE_ANALYSIS.md
│   │   ├── AGENTS_ANALYSIS.md
│   │   ├── WORKFLOWS_ANALYSIS.md
│   │   ├── ARCHITECTURE_DECISIONS.md
│   │   ├── RISKS.md
│   │   └── WORK_PLAN.md
│   ├── api/
│   │   └── openapi.yaml            # API specification
│   └── guides/
│       ├── DEPLOYMENT.md
│       └── DEVELOPMENT.md
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI entry point
│   │   ├── config.py               # Settings
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py             # Dependencies
│   │   │   ├── router.py           # Main router
│   │   │   │
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py         # Authentication
│   │   │   │   ├── workflows.py    # Workflow management
│   │   │   │   ├── agents.py       # Agent management
│   │   │   │   ├── memory.py       # ContextTiering
│   │   │   │   ├── ring.py         # AgentRing metrics
│   │   │   │   ├── a2a.py          # A2A Protocol
│   │   │   │   ├── tasks.py        # Task Manager
│   │   │   │   └── mcp.py          # MCP integration
│   │   │   │
│   │   │   └── schemas/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py
│   │   │       ├── workflow.py
│   │   │       ├── agent.py
│   │   │       ├── memory.py
│   │   │       ├── ring.py
│   │   │       ├── a2a.py
│   │   │       ├── task.py
│   │   │       └── mcp.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── security.py         # JWT, hashing
│   │   │   └── events.py           # Event system
│   │   │
│   │   ├── bbx/
│   │   │   ├── __init__.py
│   │   │   ├── bridge.py           # BBX integration layer
│   │   │   ├── workflow_runner.py  # Workflow execution
│   │   │   ├── agent_manager.py    # Agent lifecycle
│   │   │   ├── memory_inspector.py # ContextTiering access
│   │   │   ├── ring_monitor.py     # AgentRing metrics
│   │   │   ├── a2a_handler.py      # A2A operations
│   │   │   └── mcp_proxy.py        # MCP tool calls
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # SQLAlchemy base
│   │   │   ├── session.py          # DB session
│   │   │   └── models/
│   │   │       ├── __init__.py
│   │   │       ├── user.py
│   │   │       ├── workspace.py
│   │   │       ├── project.py
│   │   │       ├── task.py
│   │   │       └── workflow_run.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── user_service.py
│   │   │   ├── workspace_service.py
│   │   │   ├── project_service.py
│   │   │   ├── task_service.py
│   │   │   └── ai_service.py       # AI task decomposition
│   │   │
│   │   └── ws/
│   │       ├── __init__.py
│   │       ├── manager.py          # WebSocket manager
│   │       ├── handlers.py         # Message handlers
│   │       └── events.py           # Event types
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_api/
│   │   ├── test_bbx/
│   │   └── test_services/
│   │
│   ├── alembic/                    # Migrations
│   ├── requirements.txt
│   ├── Dockerfile
│   └── pyproject.toml
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                # Dashboard home
│   │   │
│   │   ├── (auth)/
│   │   │   ├── login/
│   │   │   │   └── page.tsx
│   │   │   ├── signup/
│   │   │   │   └── page.tsx
│   │   │   └── layout.tsx
│   │   │
│   │   ├── (dashboard)/
│   │   │   ├── layout.tsx          # Dashboard layout
│   │   │   │
│   │   │   ├── workflows/
│   │   │   │   ├── page.tsx        # Workflow list
│   │   │   │   ├── [id]/
│   │   │   │   │   └── page.tsx    # Workflow detail
│   │   │   │   └── new/
│   │   │   │       └── page.tsx    # Create workflow
│   │   │   │
│   │   │   ├── agents/
│   │   │   │   ├── page.tsx        # Agent list
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx    # Agent detail
│   │   │   │
│   │   │   ├── memory/
│   │   │   │   └── page.tsx        # Memory explorer
│   │   │   │
│   │   │   ├── ring/
│   │   │   │   └── page.tsx        # AgentRing monitor
│   │   │   │
│   │   │   ├── a2a/
│   │   │   │   └── page.tsx        # A2A playground
│   │   │   │
│   │   │   ├── mcp/
│   │   │   │   └── page.tsx        # MCP tools
│   │   │   │
│   │   │   └── tasks/
│   │   │       ├── page.tsx        # Task Manager
│   │   │       ├── [workspaceId]/
│   │   │       │   ├── page.tsx    # Workspace
│   │   │       │   └── [projectId]/
│   │   │       │       └── page.tsx # Project
│   │   │       └── layout.tsx
│   │   │
│   │   └── api/
│   │       └── [...path]/
│   │           └── route.ts        # API proxy
│   │
│   ├── components/
│   │   ├── ui/                     # shadcn components
│   │   │
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Footer.tsx
│   │   │   └── Breadcrumb.tsx
│   │   │
│   │   ├── workflows/
│   │   │   ├── WorkflowList.tsx
│   │   │   ├── WorkflowCard.tsx
│   │   │   ├── WorkflowDAG.tsx     # DAG visualization
│   │   │   ├── WorkflowRunner.tsx
│   │   │   ├── WorkflowEditor.tsx
│   │   │   ├── StepProgress.tsx
│   │   │   └── StepLogs.tsx
│   │   │
│   │   ├── agents/
│   │   │   ├── AgentList.tsx
│   │   │   ├── AgentCard.tsx
│   │   │   ├── AgentStatus.tsx
│   │   │   ├── AgentMetrics.tsx
│   │   │   └── AgentProfile.tsx
│   │   │
│   │   ├── memory/
│   │   │   ├── MemoryExplorer.tsx
│   │   │   ├── TierView.tsx
│   │   │   ├── MemoryItem.tsx
│   │   │   ├── MemoryStats.tsx
│   │   │   └── MemorySearch.tsx
│   │   │
│   │   ├── ring/
│   │   │   ├── RingMonitor.tsx
│   │   │   ├── QueueView.tsx       # SQ/CQ visualization
│   │   │   ├── WorkerPool.tsx
│   │   │   ├── PriorityChart.tsx
│   │   │   └── RingStats.tsx
│   │   │
│   │   ├── a2a/
│   │   │   ├── A2APlayground.tsx
│   │   │   ├── AgentCardView.tsx
│   │   │   ├── TaskSender.tsx
│   │   │   ├── TaskTimeline.tsx
│   │   │   └── MessageFlow.tsx
│   │   │
│   │   ├── mcp/
│   │   │   ├── MCPTools.tsx
│   │   │   ├── ToolCard.tsx
│   │   │   └── ToolInvoker.tsx
│   │   │
│   │   ├── tasks/
│   │   │   ├── TaskManager.tsx
│   │   │   ├── WorkspaceList.tsx
│   │   │   ├── ProjectBoard.tsx
│   │   │   ├── TaskCard.tsx
│   │   │   ├── TaskDetail.tsx
│   │   │   ├── AIDecomposer.tsx    # AI разбиение задач
│   │   │   ├── AgentAssignment.tsx
│   │   │   └── TaskProgress.tsx
│   │   │
│   │   └── common/
│   │       ├── Loading.tsx
│   │       ├── Error.tsx
│   │       ├── Empty.tsx
│   │       ├── RealTimeIndicator.tsx
│   │       └── JsonViewer.tsx
│   │
│   ├── lib/
│   │   ├── api.ts                  # API client
│   │   ├── ws.ts                   # WebSocket client
│   │   ├── auth.ts                 # Auth utilities
│   │   └── utils.ts
│   │
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── useWorkflows.ts
│   │   ├── useAgents.ts
│   │   ├── useMemory.ts
│   │   ├── useRing.ts
│   │   ├── useTasks.ts
│   │   └── useAuth.ts
│   │
│   ├── stores/                     # Zustand stores
│   │   ├── authStore.ts
│   │   ├── workflowStore.ts
│   │   ├── agentStore.ts
│   │   ├── memoryStore.ts
│   │   ├── ringStore.ts
│   │   └── taskStore.ts
│   │
│   ├── types/
│   │   ├── index.ts
│   │   ├── workflow.ts
│   │   ├── agent.ts
│   │   ├── memory.ts
│   │   ├── ring.ts
│   │   ├── a2a.ts
│   │   ├── mcp.ts
│   │   └── task.ts
│   │
│   ├── public/
│   ├── styles/
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml
├── docker-compose.dev.yml
├── Makefile
├── README.md
└── .env.example
```

---

## SECTION 3: DETAILED FEATURE SPECIFICATIONS

### 3.1 WORKFLOW MANAGER

**Цель:** Управление и визуализация BBX workflows

**BBX Components Used:**
- `runtime.py` — запуск workflows
- `dag.py` — DAG execution, levels
- `BBXv6Parser` — парсинг .bbx файлов

#### 3.1.1 Workflow List Page

```yaml
path: /workflows
components:
  - WorkflowList
  - WorkflowCard

features:
  - Список всех .bbx файлов из examples/workflows/
  - Статус каждого workflow (never run / running / completed / failed)
  - Последний запуск (timestamp)
  - Quick actions: Run, Edit, Delete, Clone

api_endpoints:
  - GET /api/workflows — список workflows
  - POST /api/workflows — создать новый
  - DELETE /api/workflows/{id} — удалить

data_model:
  Workflow:
    id: uuid
    name: string
    description: string
    file_path: string
    created_at: datetime
    updated_at: datetime
    last_run_at: datetime | null
    last_run_status: enum(pending, running, completed, failed) | null
```

#### 3.1.2 Workflow Detail Page

```yaml
path: /workflows/{id}
components:
  - WorkflowDAG
  - WorkflowRunner
  - StepProgress
  - StepLogs

features:
  dag_visualization:
    - Интерактивный граф (react-flow или d3)
    - Nodes = steps
    - Edges = dependencies
    - Color coding: pending (gray), running (blue), completed (green), failed (red)
    - Click на node → показать детали step

  execution:
    - Кнопка "Run" с параметрами
    - Real-time progress через WebSocket
    - Streaming logs для каждого step
    - Pause / Resume / Cancel

  step_details:
    - Input параметры
    - Output результат
    - Duration
    - Agent который выполнял
    - Retry count

api_endpoints:
  - GET /api/workflows/{id} — детали workflow
  - GET /api/workflows/{id}/dag — DAG structure
  - POST /api/workflows/{id}/run — запустить
  - POST /api/workflows/{id}/cancel — отменить
  - GET /api/workflows/{id}/runs — история запусков
  - GET /api/workflows/{id}/runs/{runId} — детали запуска
  - GET /api/workflows/{id}/runs/{runId}/logs — логи

websocket_events:
  - workflow:started
  - workflow:step:started
  - workflow:step:progress
  - workflow:step:completed
  - workflow:step:failed
  - workflow:completed
  - workflow:failed

data_model:
  WorkflowRun:
    id: uuid
    workflow_id: uuid
    status: enum(pending, running, completed, failed, cancelled)
    started_at: datetime
    completed_at: datetime | null
    inputs: json
    outputs: json | null
    error: string | null

  StepRun:
    id: uuid
    workflow_run_id: uuid
    step_id: string
    status: enum(pending, running, completed, failed, skipped)
    started_at: datetime | null
    completed_at: datetime | null
    agent_id: string | null
    inputs: json
    outputs: json | null
    logs: text
    retry_count: int
    duration_ms: int | null
```

#### 3.1.3 Workflow Editor

```yaml
path: /workflows/new, /workflows/{id}/edit
components:
  - WorkflowEditor
  - Monaco Editor (YAML)
  - Preview panel

features:
  - YAML editor с syntax highlighting
  - Real-time validation (BBXv6Parser)
  - Preview DAG while editing
  - Templates gallery
  - Save / Save As

api_endpoints:
  - POST /api/workflows/validate — валидация без сохранения
  - PUT /api/workflows/{id} — сохранить изменения
```

---

### 3.2 AGENT MONITOR

**Цель:** Мониторинг статуса и производительности агентов

**BBX Components Used:**
- `agent_sdk.py` — информация об агентах
- `ring.py` — AgentRing scheduling, очереди
- `.claude/agents/*.md` — конфигурация агентов

#### 3.2.1 Agent List Page

```yaml
path: /agents
components:
  - AgentList
  - AgentCard
  - AgentStatus

features:
  - Список всех агентов (architect, coder, reviewer, tester)
  - Real-time статус: Idle / Working / Queued
  - Текущая задача (если working)
  - Метрики: tasks completed, avg duration, success rate
  - Health indicator

api_endpoints:
  - GET /api/agents — список агентов
  - GET /api/agents/stats — общая статистика

data_model:
  Agent:
    id: string (architect, coder, etc)
    name: string
    description: string
    status: enum(idle, working, queued, error)
    current_task: string | null
    tools: string[]
    model: string

  AgentStats:
    agent_id: string
    tasks_completed: int
    tasks_failed: int
    avg_duration_ms: int
    success_rate: float
    last_active_at: datetime | null
```

#### 3.2.2 Agent Detail Page

```yaml
path: /agents/{id}
components:
  - AgentProfile
  - AgentMetrics
  - TaskHistory

features:
  profile:
    - Содержимое .md файла (parsed)
    - Tools list
    - Model configuration
    - System prompt preview

  metrics:
    - Tasks over time (chart)
    - Duration distribution
    - Error rate trend
    - Comparison with other agents

  history:
    - Recent tasks
    - Detailed logs

api_endpoints:
  - GET /api/agents/{id} — детали агента
  - GET /api/agents/{id}/metrics — метрики
  - GET /api/agents/{id}/tasks — история задач
```

---

### 3.3 AGENT RING MONITOR

**Цель:** Визуализация io_uring-style scheduling

**BBX Components Used:**
- `ring.py` — AgentRing, SQ, CQ, workers

#### 3.3.1 Ring Monitor Page

```yaml
path: /ring
components:
  - RingMonitor
  - QueueView
  - WorkerPool
  - PriorityChart
  - RingStats

features:
  queues:
    - Submission Queue (SQ) visualization
    - Completion Queue (CQ) visualization
    - Items in each queue
    - Drain rate

  workers:
    - Worker pool status
    - Active workers count
    - Worker utilization
    - Min/max workers config

  priorities:
    - Priority distribution (LOW, NORMAL, HIGH, REALTIME)
    - Items per priority
    - Wait time per priority

  stats:
    - Throughput (ops/sec)
    - Latency (p50, p95, p99)
    - Utilization %
    - Queue depth over time

api_endpoints:
  - GET /api/ring/status — текущее состояние
  - GET /api/ring/queues — содержимое очередей
  - GET /api/ring/workers — статус workers
  - GET /api/ring/stats — метрики
  - GET /api/ring/history — история метрик

websocket_events:
  - ring:item:submitted
  - ring:item:started
  - ring:item:completed
  - ring:stats:update

data_model:
  RingStatus:
    sq_size: int
    cq_size: int
    active_workers: int
    max_workers: int
    min_workers: int

  RingItem:
    id: uuid
    priority: enum(LOW, NORMAL, HIGH, REALTIME)
    status: enum(queued, processing, completed, failed)
    submitted_at: datetime
    started_at: datetime | null
    completed_at: datetime | null
    agent_id: string | null
    task_type: string

  RingStats:
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    utilization: float
    timestamp: datetime
```

---

### 3.4 MEMORY EXPLORER

**Цель:** Визуализация MGLRU ContextTiering

**BBX Components Used:**
- `context_tiering.py` — MGLRU, 4 tiers, aging

#### 3.4.1 Memory Explorer Page

```yaml
path: /memory
components:
  - MemoryExplorer
  - TierView
  - MemoryItem
  - MemoryStats
  - MemorySearch

features:
  tiers:
    - 4 колонки/секции: HOT, WARM, COOL, COLD
    - Drag & drop между tiers (manual promotion/demotion)
    - Visual indication of age
    - Expand/collapse items

  items:
    - Key, value preview
    - Age (time since last access)
    - Access count
    - Pinned status
    - Size

  stats:
    - Hit rate (overall and per tier)
    - Promotions / Demotions count
    - Memory usage per tier
    - Eviction rate

  search:
    - Search by key
    - Search by content
    - Filter by tier
    - Filter by age

  actions:
    - Pin / Unpin item
    - Manual promote / demote
    - Delete item
    - Clear tier

api_endpoints:
  - GET /api/memory/tiers — все tiers с содержимым
  - GET /api/memory/tiers/{tier} — один tier
  - GET /api/memory/items/{key} — конкретный item
  - PUT /api/memory/items/{key}/pin — pin item
  - PUT /api/memory/items/{key}/tier — переместить в другой tier
  - DELETE /api/memory/items/{key} — удалить
  - GET /api/memory/stats — статистика
  - GET /api/memory/search — поиск

websocket_events:
  - memory:item:accessed
  - memory:item:promoted
  - memory:item:demoted
  - memory:item:evicted
  - memory:stats:update

data_model:
  MemoryTier:
    name: enum(HOT, WARM, COOL, COLD)
    items: MemoryItem[]
    size_bytes: int
    item_count: int

  MemoryItem:
    key: string
    value_preview: string (truncated)
    tier: enum(HOT, WARM, COOL, COLD)
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    size_bytes: int
    is_pinned: bool
    is_compressed: bool

  MemoryStats:
    total_items: int
    total_size_bytes: int
    hit_rate: float
    miss_rate: float
    promotions: int
    demotions: int
    evictions: int
    per_tier_stats: dict
```

---

### 3.5 A2A PLAYGROUND

**Цель:** Тестирование межагентной коммуникации

**BBX Components Used:**
- `a2a/client.py` — A2A Protocol
- Agent Cards, Tasks, Messages

#### 3.5.1 A2A Playground Page

```yaml
path: /a2a
components:
  - A2APlayground
  - AgentCardView
  - TaskSender
  - TaskTimeline
  - MessageFlow

features:
  agent_cards:
    - View Agent Cards (/.well-known/agent-card.json)
    - Agent capabilities
    - Endpoint info

  task_sender:
    - Select target agent
    - Compose message (text, file, data)
    - Send task
    - Watch task lifecycle

  task_timeline:
    - Visual timeline: submitted → working → completed
    - Duration at each stage
    - Artifacts produced

  message_flow:
    - Visualization of message passing
    - Request / Response pairs
    - Parts (TextPart, FilePart, DataPart)

  json_rpc:
    - Raw JSON-RPC interface
    - Custom method calls
    - Response viewer

api_endpoints:
  - GET /api/a2a/agents — registered agents
  - GET /api/a2a/agents/{id}/card — agent card
  - POST /api/a2a/tasks — create task
  - GET /api/a2a/tasks/{id} — get task
  - DELETE /api/a2a/tasks/{id} — cancel task
  - GET /api/a2a/tasks/{id}/stream — SSE stream
  - POST /api/a2a/rpc — raw JSON-RPC

websocket_events:
  - a2a:task:created
  - a2a:task:updated
  - a2a:task:completed
  - a2a:message:received

data_model:
  A2AAgentCard:
    name: string
    description: string
    capabilities: string[]
    endpoint: string
    authentication: object | null

  A2ATask:
    id: uuid
    status: enum(submitted, working, input-required, completed, failed, cancelled)
    from_agent: string
    to_agent: string
    messages: A2AMessage[]
    artifacts: A2AArtifact[]
    created_at: datetime
    updated_at: datetime

  A2AMessage:
    role: enum(user, agent)
    parts: A2APart[]
    timestamp: datetime

  A2APart:
    type: enum(text, file, data)
    content: string | object
```

---

### 3.6 MCP TOOLS

**Цель:** Управление и вызов MCP tools

**BBX Components Used:**
- `mcp/server.py` — MCP server
- Tool handlers

#### 3.6.1 MCP Tools Page

```yaml
path: /mcp
components:
  - MCPTools
  - ToolCard
  - ToolInvoker

features:
  tool_list:
    - All registered tools
    - Tool schema (inputs, outputs)
    - Tool description

  tool_invoker:
    - Select tool
    - Fill input form (generated from schema)
    - Execute
    - View result

  external_mcps:
    - List connected external MCP servers
    - Their tools
    - Connection status

api_endpoints:
  - GET /api/mcp/tools — list tools
  - GET /api/mcp/tools/{name}/schema — tool schema
  - POST /api/mcp/tools/{name}/invoke — invoke tool
  - GET /api/mcp/servers — external servers
  - POST /api/mcp/servers — add server
  - DELETE /api/mcp/servers/{id} — remove server

data_model:
  MCPTool:
    name: string
    description: string
    input_schema: json_schema
    output_schema: json_schema | null

  MCPServer:
    id: uuid
    name: string
    transport: enum(stdio, http)
    endpoint: string
    status: enum(connected, disconnected, error)
    tools: MCPTool[]
```

---

### 3.7 TASK MANAGER (AI-Powered)

**Цель:** Полноценный Task Manager где агенты выполняют задачи

**BBX Components Used:**
- ВСЕ компоненты вместе!
- `agent_sdk.py` — AI decomposition
- `runtime.py` — workflow execution
- `context_tiering.py` — память
- `ring.py` — scheduling

#### 3.7.1 Task Manager Architecture

```
User creates task: "Implement user authentication"
                ↓
        AI Decomposer (architect agent)
                ↓
    ┌───────────┬───────────┬───────────┐
    ↓           ↓           ↓           ↓
Subtask 1   Subtask 2   Subtask 3   Subtask 4
"Design     "Implement  "Write      "Update
 schema"     API"        tests"      docs"
    ↓           ↓           ↓           ↓
architect    coder       tester    architect
    ↓           ↓           ↓           ↓
  Done        Done        Done        Done
                ↓
         Task completed
```

#### 3.7.2 Workspace / Project / Task Hierarchy

```yaml
path: /tasks
components:
  - TaskManager
  - WorkspaceList
  - ProjectBoard
  - TaskCard
  - TaskDetail
  - AIDecomposer
  - AgentAssignment
  - TaskProgress

hierarchy:
  Workspace:
    - Contains multiple Projects
    - Has members
    - Example: "BBX Development"

  Project:
    - Contains Tasks
    - Has settings
    - Example: "Console MVP"

  Task:
    - Can have subtasks (AI-generated)
    - Assigned to agent
    - Has status, priority
    - Example: "Add authentication"

features:
  workspaces:
    - CRUD operations
    - Member management
    - Settings

  projects:
    - Kanban board view
    - List view
    - Filters, sorting

  tasks:
    - Create task (simple text)
    - AI Decompose:
      1. Architect agent analyzes task
      2. Breaks into subtasks
      3. Assigns agents
      4. Creates workflow
    - Manual subtask creation
    - Agent assignment
    - Real-time progress
    - Comments
    - Attachments

  ai_features:
    - Auto-decomposition
    - Smart agent assignment
    - Progress estimation
    - Blocker detection
    - Suggestions

api_endpoints:
  # Workspaces
  - GET /api/workspaces
  - POST /api/workspaces
  - GET /api/workspaces/{id}
  - PUT /api/workspaces/{id}
  - DELETE /api/workspaces/{id}

  # Projects
  - GET /api/workspaces/{wid}/projects
  - POST /api/workspaces/{wid}/projects
  - GET /api/projects/{id}
  - PUT /api/projects/{id}
  - DELETE /api/projects/{id}

  # Tasks
  - GET /api/projects/{pid}/tasks
  - POST /api/projects/{pid}/tasks
  - GET /api/tasks/{id}
  - PUT /api/tasks/{id}
  - DELETE /api/tasks/{id}
  - POST /api/tasks/{id}/decompose — AI decomposition
  - POST /api/tasks/{id}/assign — assign agent
  - POST /api/tasks/{id}/start — start execution
  - POST /api/tasks/{id}/complete — mark complete
  - GET /api/tasks/{id}/subtasks
  - GET /api/tasks/{id}/progress

websocket_events:
  - task:created
  - task:updated
  - task:decomposed
  - task:assigned
  - task:started
  - task:progress
  - task:completed
  - task:failed
  - subtask:* (same events)

data_model:
  Workspace:
    id: uuid
    name: string
    description: string
    created_at: datetime
    updated_at: datetime

  Project:
    id: uuid
    workspace_id: uuid
    name: string
    description: string
    status: enum(active, archived)
    created_at: datetime
    updated_at: datetime

  Task:
    id: uuid
    project_id: uuid
    parent_task_id: uuid | null  # for subtasks
    title: string
    description: string
    status: enum(todo, in_progress, review, done)
    priority: enum(low, medium, high, urgent)
    assigned_agent: string | null
    workflow_run_id: uuid | null  # link to BBX workflow
    estimated_duration: int | null
    actual_duration: int | null
    created_at: datetime
    updated_at: datetime
    started_at: datetime | null
    completed_at: datetime | null

  TaskDecomposition:
    task_id: uuid
    subtasks: Task[]
    workflow: Workflow  # generated BBX workflow
    reasoning: string  # AI explanation
```

#### 3.7.3 AI Decomposition Flow

```yaml
flow:
  1_user_input:
    - User types: "Add user authentication with OAuth"

  2_architect_analysis:
    agent: architect
    prompt: |
      Analyze this task and break it down into subtasks.
      Task: {task.title}
      Description: {task.description}
      Context: {project_context}
      
      Return JSON with:
      - subtasks: list of {title, description, estimated_hours, required_agent}
      - dependencies: which subtasks depend on others
      - workflow: suggested BBX workflow structure
      - reasoning: why you broke it down this way

  3_decomposition_result:
    subtasks:
      - title: "Design auth database schema"
        agent: architect
        depends_on: []
      - title: "Implement OAuth flow"
        agent: coder
        depends_on: [0]
      - title: "Create auth middleware"
        agent: coder
        depends_on: [0]
      - title: "Write auth tests"
        agent: tester
        depends_on: [1, 2]
      - title: "Document auth API"
        agent: architect
        depends_on: [1, 2]

  4_workflow_generation:
    - Generate .bbx workflow from decomposition
    - Save to project directory

  5_user_confirmation:
    - Show decomposition to user
    - Allow adjustments
    - Confirm to start

  6_execution:
    - Run workflow
    - Track progress in real-time
    - Update task/subtask statuses
```

---

## SECTION 4: TECHNICAL IMPLEMENTATION

### 4.1 Backend Implementation Details

#### 4.1.1 BBX Bridge

```python
# backend/app/bbx/bridge.py

from blackbox.core.runtime import run_file, BBXContext
from blackbox.core.v2.context_tiering import ContextTiering
from blackbox.core.v2.ring import AgentRing
from blackbox.a2a.client import A2AClient
from blackbox.core.adapters.agent_sdk import AgentSDKAdapter

class BBXBridge:
    """
    Мост между FastAPI и BBX Core.
    Singleton для всего приложения.
    """
    
    def __init__(self):
        self.context_tiering = ContextTiering()
        self.agent_ring = AgentRing()
        self.a2a_client = A2AClient()
        self.agent_adapter = AgentSDKAdapter()
        self._event_handlers = []
    
    async def run_workflow(
        self, 
        workflow_path: str, 
        inputs: dict,
        on_event: Callable
    ) -> WorkflowResult:
        """
        Запуск workflow с callback'ами для real-time updates.
        """
        ctx = BBXContext(
            inputs=inputs,
            on_step_start=lambda s: on_event("step:started", s),
            on_step_complete=lambda s, r: on_event("step:completed", s, r),
            on_step_fail=lambda s, e: on_event("step:failed", s, e),
        )
        
        result = await run_file(workflow_path, ctx)
        return result
    
    def get_memory_status(self) -> MemoryStatus:
        """Получить состояние ContextTiering."""
        return self.context_tiering.get_status()
    
    def get_ring_status(self) -> RingStatus:
        """Получить состояние AgentRing."""
        return self.agent_ring.get_status()
    
    async def send_a2a_task(self, agent: str, task: dict) -> A2ATask:
        """Отправить task через A2A."""
        return await self.a2a_client.create_task(agent, task)
    
    async def query_agent(self, agent: str, prompt: str) -> str:
        """Прямой запрос к агенту."""
        return await self.agent_adapter.query(agent, prompt)

# Singleton instance
bbx = BBXBridge()
```

#### 4.1.2 WebSocket Manager

```python
# backend/app/ws/manager.py

from fastapi import WebSocket
from typing import Dict, Set
import asyncio
import json

class WebSocketManager:
    """
    Управление WebSocket соединениями.
    Поддержка rooms для разных типов событий.
    """
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {
            "workflows": set(),
            "agents": set(),
            "memory": set(),
            "ring": set(),
            "tasks": set(),
            "a2a": set(),
        }
    
    async def connect(self, websocket: WebSocket, room: str):
        await websocket.accept()
        self.connections[room].add(websocket)
    
    def disconnect(self, websocket: WebSocket, room: str):
        self.connections[room].discard(websocket)
    
    async def broadcast(self, room: str, event: str, data: dict):
        """Отправить событие всем в room."""
        message = json.dumps({"event": event, "data": data})
        dead_connections = set()
        
        for ws in self.connections[room]:
            try:
                await ws.send_text(message)
            except:
                dead_connections.add(ws)
        
        # Cleanup dead connections
        self.connections[room] -= dead_connections
    
    async def send_to_user(self, user_id: str, event: str, data: dict):
        """Отправить событие конкретному пользователю."""
        # Implementation depends on user tracking
        pass

ws_manager = WebSocketManager()
```

#### 4.1.3 AI Service (Task Decomposition)

```python
# backend/app/services/ai_service.py

from app.bbx.bridge import bbx
import json

class AIService:
    """
    AI-powered features для Task Manager.
    """
    
    DECOMPOSE_PROMPT = """
    You are a senior software architect. Analyze the following task and break it down into subtasks.
    
    Task: {title}
    Description: {description}
    Project Context: {context}
    
    Available agents:
    - architect: System design, documentation, planning
    - coder: Implementation, coding
    - reviewer: Code review, quality assurance
    - tester: Writing tests, testing
    
    Return a JSON object with:
    {{
        "subtasks": [
            {{
                "title": "Subtask title",
                "description": "What needs to be done",
                "agent": "which agent should do this",
                "estimated_hours": number,
                "depends_on": [indices of subtasks this depends on]
            }}
        ],
        "workflow": {{
            "name": "workflow name",
            "steps": [...] // BBX workflow format
        }},
        "reasoning": "Explanation of the decomposition"
    }}
    
    Be thorough but practical. Create 3-7 subtasks typically.
    """
    
    async def decompose_task(
        self, 
        task_id: str,
        title: str, 
        description: str,
        project_context: str
    ) -> TaskDecomposition:
        """
        Используем architect агента для разбиения задачи.
        """
        prompt = self.DECOMPOSE_PROMPT.format(
            title=title,
            description=description,
            context=project_context
        )
        
        response = await bbx.query_agent("architect", prompt)
        
        # Parse JSON response
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            result = self._extract_json(response)
        
        return TaskDecomposition(
            task_id=task_id,
            subtasks=result["subtasks"],
            workflow=result["workflow"],
            reasoning=result["reasoning"]
        )
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text that might have other content."""
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("No JSON found in response")

ai_service = AIService()
```

### 4.2 Frontend Implementation Details

#### 4.2.1 WebSocket Hook

```typescript
// frontend/hooks/useWebSocket.ts

import { useEffect, useRef, useCallback, useState } from 'react';

interface WSMessage {
  event: string;
  data: any;
}

export function useWebSocket(room: string) {
  const ws = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const handlers = useRef<Map<string, Set<(data: any) => void>>>(new Map());

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/${room}`;
    
    ws.current = new WebSocket(url);
    
    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);
    
    ws.current.onmessage = (event) => {
      const message: WSMessage = JSON.parse(event.data);
      setLastMessage(message);
      
      // Call registered handlers
      const eventHandlers = handlers.current.get(message.event);
      if (eventHandlers) {
        eventHandlers.forEach(handler => handler(message.data));
      }
    };

    return () => {
      ws.current?.close();
    };
  }, [room]);

  const subscribe = useCallback((event: string, handler: (data: any) => void) => {
    if (!handlers.current.has(event)) {
      handlers.current.set(event, new Set());
    }
    handlers.current.get(event)!.add(handler);
    
    return () => {
      handlers.current.get(event)?.delete(handler);
    };
  }, []);

  const send = useCallback((event: string, data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ event, data }));
    }
  }, []);

  return { isConnected, lastMessage, subscribe, send };
}
```

#### 4.2.2 DAG Visualization Component

```typescript
// frontend/components/workflows/WorkflowDAG.tsx

import React, { useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface Step {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  agent?: string;
  depends_on?: string[];
}

interface WorkflowDAGProps {
  steps: Step[];
  onStepClick?: (step: Step) => void;
}

const statusColors = {
  pending: '#9CA3AF',    // gray
  running: '#3B82F6',    // blue
  completed: '#10B981',  // green
  failed: '#EF4444',     // red
};

export function WorkflowDAG({ steps, onStepClick }: WorkflowDAGProps) {
  const { nodes, edges } = useMemo(() => {
    // Calculate layout (simple left-to-right)
    const levels = calculateLevels(steps);
    
    const nodes: Node[] = steps.map((step, i) => ({
      id: step.id,
      position: { 
        x: levels[step.id] * 250, 
        y: calculateY(step.id, levels, steps) * 100 
      },
      data: { 
        label: (
          <div className="p-2">
            <div className="font-medium">{step.name}</div>
            {step.agent && (
              <div className="text-xs text-gray-500">{step.agent}</div>
            )}
          </div>
        ),
      },
      style: {
        background: statusColors[step.status],
        color: 'white',
        borderRadius: '8px',
        border: 'none',
      },
    }));

    const edges: Edge[] = steps.flatMap(step => 
      (step.depends_on || []).map(dep => ({
        id: `${dep}-${step.id}`,
        source: dep,
        target: step.id,
        animated: step.status === 'running',
      }))
    );

    return { nodes, edges };
  }, [steps]);

  return (
    <div className="h-[500px] w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={(_, node) => {
          const step = steps.find(s => s.id === node.id);
          if (step && onStepClick) onStepClick(step);
        }}
        fitView
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}

function calculateLevels(steps: Step[]): Record<string, number> {
  // Topological sort to determine levels
  const levels: Record<string, number> = {};
  const visited = new Set<string>();
  
  function visit(id: string): number {
    if (levels[id] !== undefined) return levels[id];
    
    const step = steps.find(s => s.id === id);
    if (!step) return 0;
    
    const deps = step.depends_on || [];
    const maxDepLevel = deps.length > 0 
      ? Math.max(...deps.map(visit)) 
      : -1;
    
    levels[id] = maxDepLevel + 1;
    return levels[id];
  }
  
  steps.forEach(s => visit(s.id));
  return levels;
}

function calculateY(
  id: string, 
  levels: Record<string, number>, 
  steps: Step[]
): number {
  const level = levels[id];
  const sameLevel = steps.filter(s => levels[s.id] === level);
  return sameLevel.findIndex(s => s.id === id);
}
```

#### 4.2.3 Memory Tier Visualization

```typescript
// frontend/components/memory/TierView.tsx

import React from 'react';
import { MemoryItem, MemoryTier } from '@/types/memory';
import { MemoryItemCard } from './MemoryItem';

interface TierViewProps {
  tier: MemoryTier;
  onItemClick?: (item: MemoryItem) => void;
  onPin?: (key: string) => void;
  onPromote?: (key: string) => void;
  onDemote?: (key: string) => void;
}

const tierColors = {
  HOT: 'bg-red-500',
  WARM: 'bg-orange-500',
  COOL: 'bg-blue-500',
  COLD: 'bg-gray-500',
};

const tierDescriptions = {
  HOT: 'Active memory, instant access',
  WARM: 'Recent memory, compressed',
  COOL: 'Older memory, on disk',
  COLD: 'Archive, rarely accessed',
};

export function TierView({
  tier,
  onItemClick,
  onPin,
  onPromote,
  onDemote,
}: TierViewProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className={`${tierColors[tier.name]} text-white p-3 rounded-t-lg`}>
        <h3 className="font-bold text-lg">{tier.name}</h3>
        <p className="text-sm opacity-80">{tierDescriptions[tier.name]}</p>
        <div className="mt-2 flex justify-between text-sm">
          <span>{tier.item_count} items</span>
          <span>{formatBytes(tier.size_bytes)}</span>
        </div>
      </div>
      
      {/* Items */}
      <div className="flex-1 overflow-auto p-2 space-y-2 bg-gray-50 rounded-b-lg">
        {tier.items.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No items in this tier
          </div>
        ) : (
          tier.items.map(item => (
            <MemoryItemCard
              key={item.key}
              item={item}
              onClick={() => onItemClick?.(item)}
              onPin={() => onPin?.(item.key)}
              onPromote={tier.name !== 'HOT' ? () => onPromote?.(item.key) : undefined}
              onDemote={tier.name !== 'COLD' ? () => onDemote?.(item.key) : undefined}
            />
          ))
        )}
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
```

---

## SECTION 5: TESTING REQUIREMENTS

### 5.1 Testing Matrix

| Component | Unit Tests | Integration Tests | E2E Tests |
|-----------|------------|-------------------|-----------|
| Workflow Engine | ✓ | ✓ | ✓ |
| Agent Monitor | ✓ | ✓ | |
| Memory Explorer | ✓ | ✓ | |
| AgentRing Monitor | ✓ | ✓ | |
| A2A Playground | ✓ | ✓ | |
| MCP Tools | ✓ | ✓ | |
| Task Manager | ✓ | ✓ | ✓ |
| Authentication | ✓ | ✓ | ✓ |
| WebSocket | | ✓ | ✓ |

### 5.2 Critical Test Scenarios

```yaml
scenarios:
  - name: "Workflow Execution E2E"
    steps:
      - Login
      - Navigate to Workflows
      - Select a workflow
      - Click Run
      - Verify DAG updates in real-time
      - Wait for completion
      - Verify results
    validates:
      - runtime.py
      - dag.py
      - WebSocket
      - UI components

  - name: "Task Decomposition E2E"
    steps:
      - Login
      - Create workspace and project
      - Create new task with description
      - Click "AI Decompose"
      - Verify subtasks created
      - Start execution
      - Monitor progress
      - Verify all subtasks complete
    validates:
      - agent_sdk.py
      - runtime.py
      - AI decomposition
      - Full workflow

  - name: "Memory Tiering"
    steps:
      - Navigate to Memory Explorer
      - Observe current state
      - Trigger activity that creates memory
      - Observe item appears in HOT
      - Wait for aging
      - Observe promotion/demotion
    validates:
      - context_tiering.py
      - Aging loop
      - UI updates

  - name: "AgentRing Scheduling"
    steps:
      - Navigate to Ring Monitor
      - Submit multiple tasks with different priorities
      - Observe queue filling
      - Observe workers processing
      - Verify high priority processed first
    validates:
      - ring.py
      - Priority scheduling
      - Worker pool
```

---

## SECTION 6: DEPLOYMENT

### 6.1 Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/bbx
      - REDIS_URL=redis://redis:6379
      - BBX_PATH=/app/blackbox
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ../blackbox:/app/blackbox:ro  # Mount BBX core
      - ../.claude:/app/.claude:ro     # Mount agents
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=bbx
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 6.2 Success Criteria

```yaml
success_criteria:
  must_have:
    - "docker-compose up запускает всё без ошибок"
    - "Можно залогиниться (OAuth или local)"
    - "Workflow list показывает все .bbx файлы"
    - "Можно запустить workflow и видеть real-time progress"
    - "DAG visualization работает"
    - "Agent Monitor показывает статусы"
    - "Memory Explorer показывает все 4 tier"
    - "Ring Monitor показывает очереди"
    - "Task Manager: создать задачу → AI decompose → выполнить"
    - "WebSocket real-time updates работают везде"

  nice_to_have:
    - "A2A Playground полностью функционален"
    - "MCP Tools можно вызывать"
    - "Mobile responsive"
    - "Dark mode"
    - "Performance < 100ms API response"
```

---

## SECTION 7: IMPLEMENTATION ORDER

```yaml
phases:
  phase_0_thinking:
    duration: "2-4 hours"
    output:
      - docs/analysis/*.md
    validation:
      - All analysis documents complete
      - Questions answered
      - Work plan created

  phase_1_foundation:
    duration: "1 day"
    tasks:
      - Project scaffolding
      - Docker setup
      - Database schema
      - Basic API structure
      - Basic frontend structure
    output:
      - Working docker-compose
      - Empty but running app

  phase_2_bbx_bridge:
    duration: "1 day"
    tasks:
      - BBXBridge implementation
      - All BBX integrations
      - WebSocket manager
    output:
      - BBX accessible from backend
      - WebSocket working

  phase_3_workflow_manager:
    duration: "1-2 days"
    tasks:
      - Workflow API
      - DAG visualization
      - Workflow runner
      - Real-time updates
    output:
      - Can run workflows from UI

  phase_4_monitors:
    duration: "1-2 days"
    tasks:
      - Agent Monitor
      - Memory Explorer
      - Ring Monitor
    output:
      - All monitors showing real data

  phase_5_task_manager:
    duration: "2-3 days"
    tasks:
      - Task CRUD
      - AI Decomposition
      - Workflow generation
      - Progress tracking
    output:
      - Full task management flow

  phase_6_polish:
    duration: "1 day"
    tasks:
      - A2A Playground
      - MCP Tools
      - UI polish
      - Testing
      - Documentation
    output:
      - Complete application

total_estimate: "7-10 days"
```

---

## APPENDIX A: API Endpoints Summary

```yaml
auth:
  - POST /api/auth/login
  - POST /api/auth/signup
  - POST /api/auth/logout
  - GET /api/auth/me

workflows:
  - GET /api/workflows
  - POST /api/workflows
  - GET /api/workflows/{id}
  - PUT /api/workflows/{id}
  - DELETE /api/workflows/{id}
  - GET /api/workflows/{id}/dag
  - POST /api/workflows/{id}/run
  - POST /api/workflows/{id}/cancel
  - GET /api/workflows/{id}/runs
  - GET /api/workflows/{id}/runs/{runId}

agents:
  - GET /api/agents
  - GET /api/agents/{id}
  - GET /api/agents/{id}/metrics
  - GET /api/agents/{id}/tasks
  - GET /api/agents/stats

memory:
  - GET /api/memory/tiers
  - GET /api/memory/tiers/{tier}
  - GET /api/memory/items/{key}
  - PUT /api/memory/items/{key}/pin
  - PUT /api/memory/items/{key}/tier
  - DELETE /api/memory/items/{key}
  - GET /api/memory/stats
  - GET /api/memory/search

ring:
  - GET /api/ring/status
  - GET /api/ring/queues
  - GET /api/ring/workers
  - GET /api/ring/stats
  - GET /api/ring/history

a2a:
  - GET /api/a2a/agents
  - GET /api/a2a/agents/{id}/card
  - POST /api/a2a/tasks
  - GET /api/a2a/tasks/{id}
  - DELETE /api/a2a/tasks/{id}
  - GET /api/a2a/tasks/{id}/stream
  - POST /api/a2a/rpc

mcp:
  - GET /api/mcp/tools
  - GET /api/mcp/tools/{name}/schema
  - POST /api/mcp/tools/{name}/invoke
  - GET /api/mcp/servers
  - POST /api/mcp/servers
  - DELETE /api/mcp/servers/{id}

workspaces:
  - GET /api/workspaces
  - POST /api/workspaces
  - GET /api/workspaces/{id}
  - PUT /api/workspaces/{id}
  - DELETE /api/workspaces/{id}

projects:
  - GET /api/workspaces/{wid}/projects
  - POST /api/workspaces/{wid}/projects
  - GET /api/projects/{id}
  - PUT /api/projects/{id}
  - DELETE /api/projects/{id}

tasks:
  - GET /api/projects/{pid}/tasks
  - POST /api/projects/{pid}/tasks
  - GET /api/tasks/{id}
  - PUT /api/tasks/{id}
  - DELETE /api/tasks/{id}
  - POST /api/tasks/{id}/decompose
  - POST /api/tasks/{id}/assign
  - POST /api/tasks/{id}/start
  - POST /api/tasks/{id}/complete
  - GET /api/tasks/{id}/subtasks
  - GET /api/tasks/{id}/progress

websocket:
  - /ws/workflows
  - /ws/agents
  - /ws/memory
  - /ws/ring
  - /ws/tasks
  - /ws/a2a
```

---

## APPENDIX B: WebSocket Events Summary

```yaml
workflows:
  - workflow:started
  - workflow:step:started
  - workflow:step:progress
  - workflow:step:completed
  - workflow:step:failed
  - workflow:completed
  - workflow:failed

agents:
  - agent:status:changed
  - agent:task:started
  - agent:task:completed

memory:
  - memory:item:accessed
  - memory:item:promoted
  - memory:item:demoted
  - memory:item:evicted
  - memory:stats:update

ring:
  - ring:item:submitted
  - ring:item:started
  - ring:item:completed
  - ring:stats:update

a2a:
  - a2a:task:created
  - a2a:task:updated
  - a2a:task:completed
  - a2a:message:received

tasks:
  - task:created
  - task:updated
  - task:decomposed
  - task:assigned
  - task:started
  - task:progress
  - task:completed
  - task:failed
```

---

## FINAL NOTES

Это ТЗ покрывает **каждый** компонент BBX:

- ✅ runtime.py → Workflow Manager
- ✅ dag.py → DAG Visualization
- ✅ BBXv6Parser → Workflow Editor
- ✅ context_tiering.py → Memory Explorer
- ✅ ring.py → Ring Monitor
- ✅ agent_sdk.py → Agent Monitor + Task Manager
- ✅ a2a/client.py → A2A Playground
- ✅ mcp/server.py → MCP Tools
- ✅ .claude/agents/*.md → Agent Profiles

**Phase 0 (Thinking) обязательна.** Агент должен изучить код перед началом.

**Success = все компоненты BBX работают через UI.**
