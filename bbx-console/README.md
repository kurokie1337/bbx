# BBX Console

Web-based management console for the BBX Workflow Engine. Provides real-time monitoring, workflow visualization, and agent orchestration capabilities.

## Features

- **Dashboard** - Real-time system health, metrics, and status overview
- **Workflow Manager** - Create, run, and monitor workflows with DAG visualization
- **Agent Monitor** - Track agent status, performance metrics, and resource usage
- **Memory Tiering** - Visualize MGLRU-inspired context tiering system
- **Agent Ring** - Monitor io_uring-style batch operation system
- **Task Manager** - Kanban board with AI-powered task decomposition
- **A2A Playground** - Test Agent-to-Agent protocol interactions
- **MCP Tools** - Browse and invoke MCP server tools

## Architecture

```
bbx-console/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── api/     # REST API routes
│   │   ├── bbx/     # BBX Core bridge
│   │   ├── db/      # Database models
│   │   ├── services/ # Business logic
│   │   └── ws/      # WebSocket manager
│   └── tests/
├── frontend/         # React + TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── stores/
│   │   └── hooks/
│   └── public/
├── nginx/           # Nginx reverse proxy config
├── prometheus/      # Prometheus monitoring config
├── grafana/         # Grafana dashboards
└── docker-compose.yml
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for local development)

### Running with Docker

```bash
# Clone and navigate to the project
cd bbx-console

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

Access the console at: http://localhost:3000

### Local Development

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Docker Compose Profiles

```bash
# Development (default)
docker-compose up -d

# Production with Nginx
docker-compose --profile production up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Full production stack
docker-compose --profile production --profile monitoring up -d
```

## API Documentation

Once running, access the API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## WebSocket Events

The console uses WebSocket for real-time updates:

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to channels
ws.send(JSON.stringify({
  action: 'subscribe',
  channels: ['executions', 'agents', 'ring']
}));

// Receive updates
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data.payload);
};
```

### Available Channels

| Channel | Description |
|---------|-------------|
| `executions` | Workflow execution updates |
| `agents` | Agent status changes |
| `memory` | Memory tier changes |
| `ring` | AgentRing operation updates |
| `tasks` | Task board updates |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://bbx:...@postgres:5432/bbx_console` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `SECRET_KEY` | JWT secret key | - |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `BBX_CORE_PATH` | Path to BBX Core module | `/app/bbx` |

## Monitoring

### Prometheus Metrics

Metrics available at `/api/metrics`:
- `bbx_executions_total` - Total workflow executions
- `bbx_execution_duration_seconds` - Execution duration histogram
- `bbx_agents_active` - Number of active agents
- `bbx_ring_operations_total` - AgentRing operations count
- `bbx_memory_tier_size_bytes` - Memory tier sizes

### Grafana Dashboards

Pre-configured dashboards:
- BBX Overview - System health and key metrics
- Workflow Analytics - Execution statistics and trends
- Agent Performance - Per-agent metrics and comparisons
- Ring Monitor - AgentRing operation analysis

Access Grafana at: http://localhost:3001 (admin/admin)

## Testing

```bash
# Backend tests
cd backend
pytest -v

# Frontend tests
cd frontend
npm run test

# E2E tests
npm run test:e2e
```

## License

MIT License - see LICENSE file for details.
