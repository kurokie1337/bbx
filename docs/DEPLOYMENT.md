# Blackbox Deployment Guide

> **Production deployment strategies for the Blackbox workflow engine**

## 🎯 Deployment Modes

Blackbox supports **three deployment modes**:

1. **CLI** - Local workflow execution
2. **Server** - REST API + WebSocket service
3. **Library** - Embedded in Python applications

---

## 📦 Mode 1: CLI Deployment

### Use Cases
- Developer workflows
- CI/CD pipelines
- Scheduled tasks (cron jobs)
- Quick prototyping

### Installation

```bash
pip install blackbox-core
```

### Usage

```bash
# Run local workflow
blackbox run-local workflows/my-flow.bbx

# With environment variables
export API_KEY="YOUR_API_KEY_HERE"
blackbox run-local workflows/api-workflow.bbx

# With input parameters
blackbox run-local workflows/user-flow.bbx --input user_id=123
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Blackbox
RUN pip install blackbox-core

# Copy workflows
COPY workflows/ /app/workflows/

# Run workflow
CMD ["blackbox", "run-local", "workflows/scheduled-job.bbx"]
```

### Cron Setup

```bash
# Edit crontab
crontab -e

# Run workflow every hour
0 * * * * cd /app && blackbox run-local workflows/hourly-check.bbx >> /var/log/blackbox.log 2>&1
```

---

## 🌐 Mode 2: Server Deployment

### Use Cases
- Multi-user platforms
- Workflow marketplace
- SaaS applications
- Real-time monitoring

### Architecture

```
┌─────────────┐       ┌─────────────────┐       ┌──────────────┐
│   Client    │──────▶│  FastAPI Server │──────▶│   Runtime    │
│  (HTTP/WS)  │◀──────│   (Port 8000)   │◀──────│    Engine    │
└─────────────┘       └─────────────────┘       └──────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │   PostgreSQL  │
                      │  (workflows)  │
                      └───────────────┘
```

### Start Server

```bash
# Development
uvicorn blackbox.server.app:app --reload --port 8000

# Production
uvicorn blackbox.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Production Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY blackbox/ /app/blackbox/
COPY .env /app/.env

# Expose port
EXPOSE 8000

# Run with Gunicorn (production)
CMD ["gunicorn", "blackbox.server.app:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### Docker Compose Setup

```yaml
version: '3.8'

services:
  blackbox:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/blackbox
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: blackbox
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - blackbox
    restart: unless-stopped

volumes:
  postgres_data:
```

### Nginx Configuration

```nginx
upstream blackbox_server {
    server blackbox:8000;
}

server {
    listen 80;
    server_name api.blackbox.dev;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.blackbox.dev;
    
    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    
    # API endpoints
    location /api/ {
        proxy_pass http://blackbox_server;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # WebSocket support
    location /ws/ {
        proxy_pass http://blackbox_server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 📚 Mode 3: Library Embedding

### Use Cases
- Integrate into existing Python apps
- Custom workflow platforms
- Microservices

### Installation

```bash
pip install blackbox-core
```

### Basic Usage

```python
from blackbox.core import run_file
import asyncio

async def main():
    result = await run_file("workflow.bbx")
    print(result)

asyncio.run(main())
```

### Advanced Integration

```python
from blackbox.core.runtime import run_file
from blackbox.core.events import EventBus, Event, EventType
from blackbox.core.registry import MCPRegistry
from myapp.adapters import CustomAdapter

async def run_workflow_with_monitoring(workflow_path: str):
    # Create event bus for monitoring
    event_bus = EventBus()
    
    # Register custom event handlers
    @event_bus.on(EventType.STEP_START)
    async def on_step_start(event: Event):
        print(f"Starting: {event.data['step_id']}")
    
    @event_bus.on(EventType.STEP_END)
    async def on_step_end(event: Event):
        print(f"Completed: {event.data['step_id']}")
    
    @event_bus.on(EventType.STEP_ERROR)
    async def on_step_error(event: Event):
        print(f"Error in {event.data['step_id']}: {event.data['error']}")
    
    # Run workflow
    result = await run_file(workflow_path, event_bus=event_bus)
    return result
```

### FastAPI Integration

```python
from fastapi import FastAPI, BackgroundTasks
from blackbox.core import run_file

app = FastAPI()

@app.post("/workflows/execute")
async def execute_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    # Run workflow in background
    background_tasks.add_task(run_workflow_background, workflow_id)
    return {"status": "started", "workflow_id": workflow_id}

async def run_workflow_background(workflow_id: str):
    workflow_path = f"workflows/{workflow_id}.bbx"
    result = await run_file(workflow_path)
    # Save result to database
    await save_execution_result(workflow_id, result)
```

---

## 🔒 Security Best Practices

### 1. Environment Variables

Never commit secrets:

```bash
# .env file (add to .gitignore)
DATABASE_URL=postgresql://user:password@localhost/db
API_KEY=YOUR_API_KEY_HERE
TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("API_KEY")
```

### 2. Input Validation

Validate workflow inputs:

```python
from pydantic import BaseModel, validator

class WorkflowInput(BaseModel):
    user_id: int
    action: str
    
    @validator('action')
    def validate_action(cls, v):
        allowed = ['create', 'update', 'delete']
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v
```

### 3. Rate Limiting

Prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/workflows/execute")
@limiter.limit("10/minute")
async def execute_workflow(request: Request):
    # ...
```

### 4. Authentication

JWT-based auth:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Depends(security)):
    token = credentials.credentials
    # Verify JWT token
    if not is_valid_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

@app.post("/workflows/execute")
async def execute_workflow(token = Depends(verify_token)):
    # Protected endpoint
    pass
```

---

## 📊 Monitoring & Observability

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blackbox.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics (Prometheus)

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
workflow_executions = Counter('workflow_executions_total', 'Total workflow executions')
workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    # Check database connection
    db_healthy = await check_database()
    
    # Check Redis
    redis_healthy = await check_redis()
    
    if db_healthy and redis_healthy:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}, 503
```

---

## 🚀 Scaling

### Horizontal Scaling

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: blackbox
spec:
  replicas: 3  # Scale to 3 instances
  selector:
    matchLabels:
      app: blackbox
  template:
    metadata:
      labels:
        app: blackbox
    spec:
      containers:
      - name: blackbox
        image: blackbox:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: blackbox-secrets
              key: database-url
```

### Background Workers

Use Celery for async execution:

```python
from celery import Celery

celery_app = Celery('blackbox', broker='redis://localhost:6379')

@celery_app.task
def execute_workflow_async(workflow_id: str):
    result = asyncio.run(run_file(f"workflows/{workflow_id}.bbx"))
    return result

# Trigger from API
@app.post("/workflows/execute")
async def execute_workflow(workflow_id: str):
    task = execute_workflow_async.delay(workflow_id)
    return {"task_id": task.id, "status": "queued"}
```

---

## 📦 Production Checklist

- [ ] Environment variables for all secrets
- [ ] HTTPS enabled (SSL certificates)
- [ ] Database backups configured
- [ ] Logging to centralized service (ELK, Datadog)
- [ ] Health checks implemented
- [ ] Rate limiting enabled
- [ ] Authentication/authorization
- [ ] Error monitoring (Sentry)
- [ ] Performance metrics (Prometheus)
- [ ] Auto-scaling configured
- [ ] CI/CD pipeline setup
- [ ] Documentation updated

---

## 📖 See Also

- [BBX Specification](BBX_SPEC.md) - Workflow format
- [Runtime Internals](RUNTIME_INTERNALS.md) - Engine architecture
- [MCP Development](MCP_DEVELOPMENT.md) - Custom adapters
