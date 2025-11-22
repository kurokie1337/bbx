# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Advanced Monitoring & Observability System for BBX
Provides comprehensive monitoring, tracing, and observability features
"""
import logging


import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import threading

logger = logging.getLogger("bbx.observability")


@dataclass
class Span:
    """Distributed tracing span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # running, success, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Metric:
    """Time-series metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    type: str = "gauge"  # gauge, counter, histogram


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class MetricsCollector:
    """
    Collects and aggregates metrics.

    Supports:
    - Counters
    - Gauges
    - Histograms
    - Summaries
    """

    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Increment counter"""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.counters[key] += value
            self.metrics.append(Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                type="counter"
            ))

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.gauges[key] = value
            self.metrics.append(Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                type="gauge"
            ))

    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram value"""
        with self._lock:
            key = self._make_key(name, tags or {})
            self.histograms[key].append(value)
            self.metrics.append(Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                type="histogram"
            ))

    def timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timing metric"""
        self.histogram(name, duration_ms, tags)

    def _make_key(self, name: str, tags: Dict[str, str]) -> str:
        """Create unique key for metric"""
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }

            for key, values in self.histograms.items():
                if values:
                    summary["histograms"][key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "p50": self._percentile(values, 0.5),
                        "p95": self._percentile(values, 0.95),
                        "p99": self._percentile(values, 0.99),
                    }

            return summary

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c_idx = f + 1 if f + 1 < len(sorted_values) else f
        if f == c_idx:
            return sorted_values[int(k)]
        return sorted_values[f] * (c_idx - k) + sorted_values[c_idx] * (k - f)


class Tracer:
    """
    Distributed tracing implementation.

    Compatible with OpenTelemetry standards.
    """

    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, str] = {}  # thread_id -> span_id
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        import uuid

        trace_id = self._get_or_create_trace_id()
        span_id = str(uuid.uuid4())

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id or self._get_active_span_id(),
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        with self._lock:
            self.spans[span_id] = span
            self._set_active_span(span_id)

        return span

    def end_span(self, span: Span, status: str = "success", error: Optional[str] = None):
        """End a span"""
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status
        span.error = error

        with self._lock:
            if self._get_active_span_id() == span.span_id:
                self._set_active_span(span.parent_span_id)

    def add_event(self, span: Span, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        span.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def _get_or_create_trace_id(self) -> str:
        """Get or create trace ID for current context"""
        import uuid
        thread_id = threading.get_ident()

        with self._lock:
            active_span_id = self.active_spans.get(str(thread_id))
            if active_span_id and active_span_id in self.spans:
                return self.spans[active_span_id].trace_id

            return str(uuid.uuid4())

    def _get_active_span_id(self) -> Optional[str]:
        """Get active span for current thread"""
        thread_id = threading.get_ident()
        return self.active_spans.get(str(thread_id))

    def _set_active_span(self, span_id: Optional[str]):
        """Set active span for current thread"""
        thread_id = threading.get_ident()
        if span_id:
            self.active_spans[str(thread_id)] = span_id
        else:
            self.active_spans.pop(str(thread_id), None)

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        with self._lock:
            return [s for s in self.spans.values() if s.trace_id == trace_id]

    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format"""
        with self._lock:
            traces = [asdict(span) for span in self.spans.values()]

        if format == "json":
            return json.dumps(traces, indent=2)
        elif format == "jaeger":
            # Convert to Jaeger format
            return self._to_jaeger_format(traces)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_jaeger_format(self, traces: List[Dict]) -> str:
        """Convert to Jaeger JSON format"""
        jaeger_traces = {
            "data": [{
                "traceID": trace["trace_id"],
                "spans": [{
                    "traceID": trace["trace_id"],
                    "spanID": trace["span_id"],
                    "operationName": trace["name"],
                    "references": [{
                        "refType": "CHILD_OF",
                        "traceID": trace["trace_id"],
                        "spanID": trace["parent_span_id"]
                    }] if trace.get("parent_span_id") else [],
                    "startTime": int(trace["start_time"] * 1_000_000),
                    "duration": int(trace.get("duration_ms", 0) * 1000),
                    "tags": [
                        {"key": k, "type": "string", "value": str(v)}
                        for k, v in trace.get("attributes", {}).items()
                    ],
                    "logs": [
                        {
                            "timestamp": int(event["timestamp"] * 1_000_000),
                            "fields": [
                                {"key": k, "type": "string", "value": str(v)}
                                for k, v in event.get("attributes", {}).items()
                            ]
                        }
                        for event in trace.get("events", [])
                    ]
                }]
            } for trace in traces]
        }
        return json.dumps(jaeger_traces, indent=2)


class StructuredLogger:
    """
    Structured logging with context propagation.
    """

    def __init__(self):
        self.logs: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self.min_level = "DEBUG"

    LEVELS = {
        "DEBUG": 0,
        "INFO": 1,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4
    }

    def log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None
    ):
        """Log structured message"""
        if self.LEVELS.get(level, 0) < self.LEVELS.get(self.min_level, 0):
            return

        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            context=context or {},
            trace_id=trace_id,
            span_id=span_id
        )

        with self._lock:
            self.logs.append(entry)

        # Print to console
        self._print_log(entry)

    def debug(self, message: str, **kwargs):
        self.log("DEBUG", message, context=kwargs)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, context=kwargs)

    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, context=kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, context=kwargs)

    def critical(self, message: str, **kwargs):
        self.log("CRITICAL", message, context=kwargs)

    def _print_log(self, entry: LogEntry):
        """Print log entry to console"""
        timestamp = datetime.fromtimestamp(entry.timestamp).isoformat()
        context_str = " ".join(f"{k}={v}" for k, v in entry.context.items())
        trace_info = ""
        if entry.trace_id:
            trace_info = f" [trace={entry.trace_id[:8]}]"
        print(f"[{timestamp}] {entry.level:8s}{trace_info} {entry.message} {context_str}")

    def query(
        self,
        level: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        trace_id: Optional[str] = None
    ) -> List[LogEntry]:
        """Query logs"""
        with self._lock:
            results = list(self.logs)

        if level:
            results = [log for log in results if log.level == level]

        if start_time:
            results = [log for log in results if log.timestamp >= start_time]

        if end_time:
            results = [log for log in results if log.timestamp <= end_time]

        if trace_id:
            results = [log for log in results if log.trace_id == trace_id]

        return results


class Observability:
    """
    Unified observability system combining metrics, traces, and logs.
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        self.logger = StructuredLogger()
        self.exporters: List[Callable] = []

    def record_workflow_execution(
        self,
        workflow_name: str,
        node_name: str,
        duration_ms: float,
        status: str,
        error: Optional[str] = None
    ):
        """Record workflow node execution"""
        tags = {
            "workflow": workflow_name,
            "node": node_name,
            "status": status
        }

        # Metrics
        self.metrics.counter("bbx.node.executions", tags=tags)
        self.metrics.timing("bbx.node.duration", duration_ms, tags=tags)

        if status == "error":
            self.metrics.counter("bbx.node.errors", tags=tags)

        # Logging
        self.logger.info(
            f"Node executed: {node_name}",
            workflow=workflow_name,
            duration_ms=duration_ms,
            status=status,
            error=error
        )

    def trace_workflow(self, workflow_name: str):
        """Context manager for tracing workflow execution"""
        return TraceContext(self, f"workflow.{workflow_name}")

    def trace_node(self, node_name: str):
        """Context manager for tracing node execution"""
        return TraceContext(self, f"node.{node_name}")

    def add_exporter(self, exporter: Callable[[Dict[str, Any]], None]):
        """Add telemetry exporter"""
        self.exporters.append(exporter)

    async def export_all(self):
        """Export all telemetry data"""
        data = {
            "metrics": self.metrics.get_summary(),
            "traces": [asdict(span) for span in self.tracer.spans.values()],
            "logs": [asdict(log) for log in self.logger.logs]
        }

        for exporter in self.exporters:
            try:
                if asyncio.iscoroutinefunction(exporter):
                    await exporter(data)
                else:
                    exporter(data)
            except Exception as e:
                logger.error(f"Exporter error: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for observability dashboard"""
        return {
            "metrics": self.metrics.get_summary(),
            "active_traces": len([s for s in self.tracer.spans.values() if s.status == "running"]),
            "total_traces": len(self.tracer.spans),
            "recent_logs": [asdict(log) for log in list(self.logger.logs)[-100:]],
            "error_rate": self._calculate_error_rate(),
            "avg_duration": self._calculate_avg_duration()
        }

    def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent executions"""
        counters = self.metrics.counters
        total = sum(v for k, v in counters.items() if "executions" in k)
        errors = sum(v for k, v in counters.items() if "errors" in k)
        return (errors / total * 100) if total > 0 else 0

    def _calculate_avg_duration(self) -> float:
        """Calculate average execution duration"""
        histograms = self.metrics.histograms
        durations = []
        for key, values in histograms.items():
            if "duration" in key:
                durations.extend(values)
        return sum(durations) / len(durations) if durations else 0


class TraceContext:
    """Context manager for tracing"""

    def __init__(self, observability: Observability, name: str):
        self.observability = observability
        self.name = name
        self.span = None

    def __enter__(self):
        self.span = self.observability.tracer.start_span(self.name)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.observability.tracer.end_span(
                self.span,
                status="error",
                error=f"{exc_type.__name__}: {exc_val}"
            )
        else:
            self.observability.tracer.end_span(self.span, status="success")

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


# Exporters

class PrometheusExporter:
    """Export metrics in Prometheus format"""

    def __init__(self, port: int = 9090):
        self.port = port

    def __call__(self, data: Dict[str, Any]):
        """Export data"""
        metrics = data.get("metrics", {})
        output = []

        # Counters
        for name, value in metrics.get("counters", {}).items():
            output.append(f"# TYPE {name} counter")
            output.append(f"{name} {value}")

        # Gauges
        for name, value in metrics.get("gauges", {}).items():
            output.append(f"# TYPE {name} gauge")
            output.append(f"{name} {value}")

        # Histograms
        for name, stats in metrics.get("histograms", {}).items():
            output.append(f"# TYPE {name} histogram")
            output.append(f"{name}_count {stats['count']}")
            output.append(f"{name}_sum {stats['sum']}")
            output.append(f"{name}_bucket{{le=\"0.5\"}} {stats['p50']}")
            output.append(f"{name}_bucket{{le=\"0.95\"}} {stats['p95']}")
            output.append(f"{name}_bucket{{le=\"0.99\"}} {stats['p99']}")
            output.append(f"{name}_bucket{{le=\"+Inf\"}} {stats['count']}")

        return "\n".join(output)


class JaegerExporter:
    """Export traces to Jaeger"""

    def __init__(self, endpoint: str = "http://localhost:14268/api/traces"):
        self.endpoint = endpoint

    async def __call__(self, data: Dict[str, Any]):
        """Export traces"""
        import aiohttp

        traces = data.get("traces", [])
        if not traces:
            return

        # Convert to Jaeger format
        # (Implementation would format traces for Jaeger API)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"traces": traces}) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to export to Jaeger: {resp.status}")


class FileExporter:
    """Export all telemetry to files"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def __call__(self, data: Dict[str, Any]):
        """Export to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        metrics_file.write_text(json.dumps(data.get("metrics", {}), indent=2))

        # Traces
        traces_file = self.output_dir / f"traces_{timestamp}.json"
        traces_file.write_text(json.dumps(data.get("traces", []), indent=2))

        # Logs
        logs_file = self.output_dir / f"logs_{timestamp}.json"
        logs_file.write_text(json.dumps(data.get("logs", []), indent=2))


# Global instance
_observability = Observability()


def get_observability() -> Observability:
    """Get global observability instance"""
    return _observability
