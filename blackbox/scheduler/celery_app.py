"""Celery application for workflow scheduling."""

from celery import Celery
from celery.schedules import crontab

app = Celery("bbx_scheduler", broker="redis://localhost:6379/0")

app.conf.beat_schedule = {
    "example-every-hour": {
        "task": "blackbox.scheduler.tasks.execute_scheduled_workflow",
        "schedule": crontab(minute=0),  # Every hour
        "args": ("hourly_workflow.bbx",),
    },
}
