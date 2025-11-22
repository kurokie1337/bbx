# Redis Cache Backend

Use Redis for distributed caching across BBX instances.

## Configuration

\`\`\`yaml
# config.yaml
cache:
  backend: redis
  ttl: 3600

redis:
  host: localhost
  port: 6379
  password: your-password
\`\`\`

## Environment Variables

\`\`\`bash
export BBX_REDIS_HOST=localhost
export BBX_REDIS_PORT=6379
export BBX_REDIS_PASSWORD=secret
\`\`\`

## Cache Operations

\`\`\`yaml
- id: clear_cache
  mcp: bbx.cache
  method: clear

- id: invalidate_workflow
  mcp: bbx.cache
  method: invalidate
  inputs:
    workflow_id: my_workflow

- id: cache_stats
  mcp: bbx.cache
  method: stats
\`\`\`

## Docker Compose

\`\`\`bash
docker-compose up -d redis
\`\`\`
