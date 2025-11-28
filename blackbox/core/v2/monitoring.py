# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Monitoring - Prometheus, OpenTelemetry, and Grafana integration.

Features:
- Prometheus metrics export
- OpenTelemetry tracing
- Structured logging
- Custom dashboards
- Alerting rules
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger("bbx.monitoring")


# =============================================================================
# Metrics
# =============================================================================


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsRegistry:
    """Registry for all metrics"""

    def __init__(self, prefix: str = "bbx"):
        self._prefix = prefix
        self._metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    def counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter"""
        full_name = f"{self._prefix}_{name}"
        label_key = self._label_key(full_name, labels or {})
        self._counters[label_key] += value

        self._metrics[full_name].append(MetricValue(
            name=full_name,
            type=MetricType.COUNTER,
            value=self._counters[label_key],
            labels=labels or {}
        ))

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge value"""
        full_name = f"{self._prefix}_{name}"
        label_key = self._label_key(full_name, labels or {})
        self._gauges[label_key] = value

        self._metrics[full_name].append(MetricValue(
            name=full_name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        ))

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None
    ):
        """Record a histogram value"""
        full_name = f"{self._prefix}_{name}"
        label_key = self._label_key(full_name, labels or {})
        self._histograms[label_key].append(value)

        # Keep only last 10000 observations
        if len(self._histograms[label_key]) > 10000:
            self._histograms[label_key] = self._histograms[label_key][-10000:]

    def _label_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key from name and labels"""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # Export counters
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")

        # Export gauges
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")

        # Export histogram summaries
        for key, values in self._histograms.items():
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")

        return "\n".join(lines)


# =============================================================================
# Tracing (OpenTelemetry-compatible)
# =============================================================================


@dataclass
class Span:
    """A trace span"""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


class Tracer:
    """OpenTelemetry-compatible tracer"""

    def __init__(self, service_name: str = "bbx"):
        self._service_name = service_name
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._exporters: List[Callable[[Span], None]] = []

    def start_span(
        self,
        operation_name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        trace_id = parent_id.split("-")[0] if parent_id else uuid.uuid4().hex[:32]
        span_id = uuid.uuid4().hex[:16]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            operation_name=operation_name,
            start_time=time.time(),
            attributes=attributes or {}
        )

        self._active_spans[span_id] = span
        return span

    def end_span(self, span: Span, status: str = "OK"):
        """End a span"""
        span.end_time = time.time()
        span.status = status
        self._spans.append(span)
        self._active_spans.pop(span.span_id, None)

        # Export
        for exporter in self._exporters:
            try:
                exporter(span)
            except Exception:
                pass

    @contextmanager
    def trace(
        self,
        operation_name: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing"""
        span = self.start_span(operation_name, parent_id, attributes)
        try:
            yield span
            self.end_span(span, "OK")
        except Exception as e:
            span.attributes["error"] = str(e)
            self.end_span(span, "ERROR")
            raise

    def add_exporter(self, exporter: Callable[[Span], None]):
        """Add span exporter"""
        self._exporters.append(exporter)

    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent spans"""
        return self._spans[-limit:]


# =============================================================================
# Logging
# =============================================================================


class StructuredLogger:
    """Structured JSON logger"""

    def __init__(
        self,
        service_name: str = "bbx",
        level: int = logging.INFO
    ):
        self._service_name = service_name
        self._logger = logging.getLogger(service_name)
        self._logger.setLevel(level)
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set persistent context for all logs"""
        self._context.update(kwargs)

    def _log(self, level: str, message: str, **kwargs):
        """Internal log method"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": self._service_name,
            "message": message,
            **self._context,
            **kwargs
        }

        log_line = json.dumps(log_entry)

        if level == "ERROR":
            self._logger.error(log_line)
        elif level == "WARN":
            self._logger.warning(log_line)
        elif level == "DEBUG":
            self._logger.debug(log_line)
        else:
            self._logger.info(log_line)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def warn(self, message: str, **kwargs):
        self._log("WARN", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)


# =============================================================================
# Alerting
# =============================================================================


@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    metric: str
    condition: str  # e.g., "value > 100"
    duration_seconds: int = 60
    severity: str = "warning"  # warning, critical
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Fired alert"""
    rule_id: str
    name: str
    severity: str
    message: str
    value: float
    fired_at: float
    resolved_at: Optional[float] = None


class AlertManager:
    """Manages alerts"""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._metric_values: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self._rules[rule.id] = rule

    def record_metric(self, metric: str, value: float):
        """Record metric for alerting"""
        now = time.time()
        self._metric_values[metric].append((now, value))

        # Keep last 5 minutes
        cutoff = now - 300
        self._metric_values[metric] = [
            (t, v) for t, v in self._metric_values[metric]
            if t >= cutoff
        ]

        # Check rules
        self._evaluate_rules(metric)

    def _evaluate_rules(self, metric: str):
        """Evaluate alert rules for metric"""
        for rule in self._rules.values():
            if rule.metric != metric:
                continue

            values = self._metric_values.get(metric, [])
            if not values:
                continue

            # Get values in duration window
            cutoff = time.time() - rule.duration_seconds
            window_values = [v for t, v in values if t >= cutoff]

            if not window_values:
                continue

            # Evaluate condition
            avg_value = sum(window_values) / len(window_values)

            try:
                triggered = eval(
                    rule.condition,
                    {"__builtins__": {}},
                    {"value": avg_value}
                )
            except Exception:
                continue

            if triggered:
                self._fire_alert(rule, avg_value)
            else:
                self._resolve_alert(rule.id)

    def _fire_alert(self, rule: AlertRule, value: float):
        """Fire an alert"""
        if rule.id in self._active_alerts:
            return  # Already active

        alert = Alert(
            rule_id=rule.id,
            name=rule.name,
            severity=rule.severity,
            message=f"{rule.name}: {rule.condition} (current: {value:.2f})",
            value=value,
            fired_at=time.time()
        )

        self._active_alerts[rule.id] = alert
        self._alert_history.append(alert)

        # Notify handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass

    def _resolve_alert(self, rule_id: str):
        """Resolve an alert"""
        if rule_id in self._active_alerts:
            alert = self._active_alerts.pop(rule_id)
            alert.resolved_at = time.time()

    def on_alert(self, handler: Callable[[Alert], None]):
        """Register alert handler"""
        self._handlers.append(handler)

    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return list(self._active_alerts.values())


# =============================================================================
# Dashboard Configuration
# =============================================================================


@dataclass
class DashboardPanel:
    """Dashboard panel definition"""
    id: str
    title: str
    type: str  # 'graph', 'stat', 'table', 'gauge'
    metrics: List[str]
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard definition"""
    id: str
    title: str
    panels: List[DashboardPanel]
    refresh_seconds: int = 30
    time_range: str = "1h"


class DashboardManager:
    """Manages dashboard configurations"""

    def __init__(self):
        self._dashboards: Dict[str, Dashboard] = {}
        self._default_panels = self._create_default_panels()

    def _create_default_panels(self) -> List[DashboardPanel]:
        """Create default BBX panels"""
        return [
            DashboardPanel(
                id="agent_count",
                title="Active Agents",
                type="stat",
                metrics=["bbx_agents_active"]
            ),
            DashboardPanel(
                id="ring_throughput",
                title="AgentRing Throughput",
                type="graph",
                metrics=["bbx_ring_operations_total", "bbx_ring_completions_total"]
            ),
            DashboardPanel(
                id="memory_tiers",
                title="Memory Tier Distribution",
                type="graph",
                metrics=["bbx_context_hot_size", "bbx_context_warm_size", "bbx_context_cold_size"]
            ),
            DashboardPanel(
                id="quota_usage",
                title="Quota Usage",
                type="gauge",
                metrics=["bbx_quota_cpu_usage", "bbx_quota_memory_usage"]
            ),
            DashboardPanel(
                id="errors",
                title="Error Rate",
                type="graph",
                metrics=["bbx_errors_total"]
            )
        ]

    def create_dashboard(
        self,
        title: str,
        panels: Optional[List[DashboardPanel]] = None
    ) -> Dashboard:
        """Create a dashboard"""
        dashboard = Dashboard(
            id=f"dash_{uuid.uuid4().hex[:8]}",
            title=title,
            panels=panels or self._default_panels
        )
        self._dashboards[dashboard.id] = dashboard
        return dashboard

    def export_grafana_json(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Export dashboard as Grafana JSON"""
        panels = []
        for i, panel in enumerate(dashboard.panels):
            panels.append({
                "id": i,
                "title": panel.title,
                "type": panel.type,
                "gridPos": {"x": (i % 2) * 12, "y": (i // 2) * 8, "w": 12, "h": 8},
                "targets": [
                    {"expr": metric, "refId": chr(65 + j)}
                    for j, metric in enumerate(panel.metrics)
                ]
            })

        return {
            "uid": dashboard.id,
            "title": dashboard.title,
            "panels": panels,
            "refresh": f"{dashboard.refresh_seconds}s",
            "time": {"from": f"now-{dashboard.time_range}", "to": "now"}
        }


# =============================================================================
# Monitoring Manager
# =============================================================================


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    service_name: str = "bbx"
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    enable_alerting: bool = True
    metrics_port: int = 9090
    log_level: int = logging.INFO


class MonitoringManager:
    """
    Unified monitoring manager.

    Combines metrics, tracing, logging, and alerting.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Components
        self._metrics = MetricsRegistry(self.config.service_name)
        self._tracer = Tracer(self.config.service_name)
        self._logger = StructuredLogger(self.config.service_name, self.config.log_level)
        self._alerts = AlertManager()
        self._dashboards = DashboardManager()

        # Standard metrics
        self._setup_standard_metrics()

    def _setup_standard_metrics(self):
        """Setup standard BBX metrics"""
        # These will be updated by the various BBX components
        self._metrics.gauge("info", 1, {"version": "2.0.0"})

    # =========================================================================
    # Metrics API
    # =========================================================================

    def counter(self, name: str, value: float = 1, **labels):
        self._metrics.counter(name, value, labels)

    def gauge(self, name: str, value: float, **labels):
        self._metrics.gauge(name, value, labels)

    def histogram(self, name: str, value: float, **labels):
        self._metrics.histogram(name, value, labels)

    # =========================================================================
    # Tracing API
    # =========================================================================

    def trace(
        self,
        operation_name: str,
        parent_id: Optional[str] = None,
        **attributes
    ):
        """Start a trace span"""
        return self._tracer.trace(operation_name, parent_id, attributes)

    def start_span(self, operation_name: str, **attributes) -> Span:
        return self._tracer.start_span(operation_name, attributes=attributes)

    def end_span(self, span: Span, status: str = "OK"):
        self._tracer.end_span(span, status)

    # =========================================================================
    # Logging API
    # =========================================================================

    def info(self, message: str, **kwargs):
        self._logger.info(message, **kwargs)

    def error(self, message: str, **kwargs):
        self._logger.error(message, **kwargs)

    def warn(self, message: str, **kwargs):
        self._logger.warn(message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._logger.debug(message, **kwargs)

    # =========================================================================
    # Alerting API
    # =========================================================================

    def add_alert_rule(self, rule: AlertRule):
        self._alerts.add_rule(rule)

    def on_alert(self, handler: Callable[[Alert], None]):
        self._alerts.on_alert(handler)

    def get_active_alerts(self) -> List[Alert]:
        return self._alerts.get_active_alerts()

    # =========================================================================
    # Export
    # =========================================================================

    def export_prometheus(self) -> str:
        """Export Prometheus metrics"""
        return self._metrics.export_prometheus()

    def create_dashboard(self, title: str) -> Dashboard:
        """Create a monitoring dashboard"""
        return self._dashboards.create_dashboard(title)


# Factory
_global_monitoring: Optional[MonitoringManager] = None


def get_monitoring() -> MonitoringManager:
    global _global_monitoring
    if _global_monitoring is None:
        _global_monitoring = MonitoringManager()
    return _global_monitoring


def create_monitoring(config: Optional[MonitoringConfig] = None) -> MonitoringManager:
    return MonitoringManager(config)
