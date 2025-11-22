# Workflow Scheduler

Schedule workflows with cron expressions.

## Start Scheduler

\`\`\`bash
docker-compose up -d celery_worker celery_beat
\`\`\`

## Schedule Workflow

\`\`\`bash
python cli.py schedule my_workflow.bbx --cron "0 * * * *"  # Every hour
\`\`\`

## Cron Examples

- \`*/5 * * * *\` - Every 5 minutes
- \`0 0 * * *\` - Daily at midnight
- \`0 9 * * 1\` - Every Monday at 9 AM
