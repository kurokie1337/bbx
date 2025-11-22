# Workflow Versioning & Rollback

BBX supports semantic versioning for workflows with rollback capabilities.

## Features

- Semantic versioning (MAJOR.MINOR.PATCH)
- Version history tracking
- Rollback to previous versions
- Diff between versions
- Version tags and descriptions

## CLI Usage

### Create Version
\`\`\`bash
python cli.py version create my_workflow.bbx -v 1.0.0 -m "Initial version"
\`\`\`

### List Versions
\`\`\`bash
python cli.py version list my_workflow
\`\`\`

### Rollback
\`\`\`bash
python cli.py version rollback my_workflow 1.0.0
\`\`\`

### Diff
\`\`\`bash
python cli.py version diff my_workflow 1.0.0 1.1.0
\`\`\`

## API Usage

### Create Version (POST /api/workflows/{id}/versions)
\`\`\`bash
curl -X POST http://localhost:8000/api/workflows/my_workflow/versions \
  -H "Content-Type: application/json" \
  -d '{"version": "1.0.0", "content": {...}, "description": "Initial"}'
\`\`\`

### List Versions (GET /api/workflows/{id}/versions)
\`\`\`bash
curl http://localhost:8000/api/workflows/my_workflow/versions
\`\`\`

### Rollback (POST /api/workflows/{id}/rollback)
\`\`\`bash
curl -X POST http://localhost:8000/api/workflows/my_workflow/rollback \
  -H "Content-Type: application/json" \
  -d '{"target_version": "1.0.0"}'
\`\`\`

## Storage

Versions are stored in `~/.bbx/versions/{workflow_id}/{version}.json`
