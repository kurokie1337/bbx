"""
Prometheus metrics for Blackbox Workflow Engine.
Tracks workflow executions, step performance, and errors.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time


# Workflow metrics
workflow_executions_total = Counter(
    'blackbox_workflow_executions_total',
    'Total number of workflow executions',
    ['workflow_id', 'status']
)

workflow_duration_seconds = Histogram(
    'blackbox_workflow_duration_seconds',
    'Workflow execution duration in seconds',
    ['workflow_id'],
    buckets=(.1, .5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
)

workflow_steps_total = Counter(
    'blackbox_workflow_steps_total',
    'Total number of step executions',
    ['workflow_id', 'step_id', 'adapter', 'status']
)

workflow_step_duration_seconds = Histogram(
    'blackbox_workflow_step_duration_seconds',
    'Step execution duration in seconds',
    ['workflow_id', 'step_id', 'adapter'],
    buckets=(.01, .05, .1, .5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

# Active workflows gauge
active_workflows = Gauge(
    'blackbox_active_workflows',
    'Number of currently executing workflows'
)

# Cache metrics
cache_hits_total = Counter(
    'blackbox_cache_hits_total',
    'Total number of cache hits'
)

cache_misses_total = Counter(
    'blackbox_cache_misses_total',
    'Total number of cache misses'
)

cache_size = Gauge(
    'blackbox_cache_size',
    'Current number of items in cache'
)

# Error metrics
errors_total = Counter(
    'blackbox_errors_total',
    'Total number of errors',
    ['error_type', 'workflow_id']
)

# Info metric
blackbox_info = Info(
    'blackbox_build',
    'Blackbox version and build information'
)


class MetricsContext:
    """Context manager for tracking workflow execution metrics."""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        active_workflows.inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        workflow_duration_seconds.labels(workflow_id=self.workflow_id).observe(duration)
        active_workflows.dec()
        
        status = "error" if exc_type else "success"
        workflow_executions_total.labels(
            workflow_id=self.workflow_id,
            status=status
        ).inc()
        
        if exc_type:
            error_type = exc_type.__name__
            errors_total.labels(
                error_type=error_type,
                workflow_id=self.workflow_id
            ).inc()
        
        return False


class StepMetricsContext:
    """Context manager for tracking step execution metrics."""
    
    def __init__(self, workflow_id: str, step_id: str, adapter: str):
        self.workflow_id = workflow_id
        self.step_id = step_id
        self.adapter = adapter
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        workflow_step_duration_seconds.labels(
            workflow_id=self.workflow_id,
            step_id=self.step_id,
            adapter=self.adapter
        ).observe(duration)
        
        status = "error" if exc_type else "success"
        workflow_steps_total.labels(
            workflow_id=self.workflow_id,
            step_id=self.step_id,
            adapter=self.adapter,
            status=status
        ).inc()
        
        return False


def record_cache_hit():
    """Record a cache hit."""
    cache_hits_total.inc()


def record_cache_miss():
    """Record a cache miss."""
    cache_misses_total.inc()


def update_cache_size(size: int):
    """Update cache size gauge."""
    cache_size.set(size)


def init_metrics(version: str = "1.0.0"):
    """Initialize metrics with build information."""
    blackbox_info.info({
        'version': version,
        'python_version': '3.8+'
    })
