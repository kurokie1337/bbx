from contextvars import ContextVar

# Tracks the nesting level of workflow executions
workflow_depth = ContextVar("workflow_depth", default=0)
