# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Network Fabric - Service Mesh for AI Agents

Inspired by Istio/Envoy service mesh, provides:
- Agent-to-agent communication
- Load balancing and routing
- Circuit breaking and retries
- Observability (tracing, metrics)
- Traffic management (canary, A/B testing)
- mTLS between agents

Key concepts:
- Mesh: The network of connected agents
- Sidecar: Proxy for each agent
- VirtualService: Traffic routing rules
- DestinationRule: Load balancing policy
- Gateway: Entry/exit point for the mesh
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.network_fabric")


# =============================================================================
# Service Discovery
# =============================================================================

class ServiceStatus(Enum):
    """Status of a service in the mesh."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    STARTING = "starting"
    STOPPED = "stopped"


@dataclass
class ServiceEndpoint:
    """An endpoint (instance) of a service."""

    id: str
    address: str                    # e.g., "agent://analysis-1"
    port: int = 0
    status: ServiceStatus = ServiceStatus.HEALTHY

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Health
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    weight: int = 100              # Load balancing weight

    # Metrics
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    latency_sum_ms: float = 0

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 1.0
        return self.requests_success / self.requests_total

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.latency_sum_ms / self.requests_total


@dataclass
class Service:
    """A service in the mesh (group of endpoints)."""

    name: str
    namespace: str = "default"
    description: str = ""

    # Endpoints
    endpoints: Dict[str, ServiceEndpoint] = field(default_factory=dict)

    # Configuration
    port: int = 0
    protocol: str = "grpc"

    # Labels
    labels: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add an endpoint to this service."""
        self.endpoints[endpoint.id] = endpoint
        self.updated_at = datetime.now()

    def remove_endpoint(self, endpoint_id: str):
        """Remove an endpoint."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            self.updated_at = datetime.now()

    def healthy_endpoints(self) -> List[ServiceEndpoint]:
        """Get all healthy endpoints."""
        return [e for e in self.endpoints.values()
                if e.status == ServiceStatus.HEALTHY]

    @property
    def fqdn(self) -> str:
        """Fully qualified domain name."""
        return f"{self.name}.{self.namespace}.svc.bbx.local"


class ServiceRegistry:
    """Service discovery registry."""

    def __init__(self):
        self._services: Dict[str, Dict[str, Service]] = defaultdict(dict)
        self._watchers: Dict[str, List[Callable]] = defaultdict(list)

    def register(self, service: Service):
        """Register a service."""
        self._services[service.namespace][service.name] = service
        self._notify_watchers(service.namespace, service.name)

    def deregister(self, name: str, namespace: str = "default"):
        """Deregister a service."""
        if name in self._services[namespace]:
            del self._services[namespace][name]
            self._notify_watchers(namespace, name)

    def get(self, name: str, namespace: str = "default") -> Optional[Service]:
        """Get a service by name."""
        return self._services[namespace].get(name)

    def list(self, namespace: Optional[str] = None) -> List[Service]:
        """List all services."""
        if namespace:
            return list(self._services[namespace].values())
        return [s for ns in self._services.values() for s in ns.values()]

    def watch(self, callback: Callable, namespace: str = "default", name: str = "*"):
        """Watch for service changes."""
        key = f"{namespace}/{name}"
        self._watchers[key].append(callback)

    def _notify_watchers(self, namespace: str, name: str):
        """Notify watchers of changes."""
        for key in [f"{namespace}/{name}", f"{namespace}/*", "*/*"]:
            for callback in self._watchers.get(key, []):
                try:
                    callback(namespace, name)
                except Exception as e:
                    logger.error(f"Watcher error: {e}")


# =============================================================================
# Load Balancing
# =============================================================================

class LoadBalanceAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


class LoadBalancer(ABC):
    """Base class for load balancers."""

    @abstractmethod
    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request: Optional[Dict] = None
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint for the request."""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancer."""

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request: Optional[Dict] = None
    ) -> Optional[ServiceEndpoint]:
        if not endpoints:
            return None

        key = ",".join(e.id for e in endpoints)
        idx = self._counters[key] % len(endpoints)
        self._counters[key] += 1

        return endpoints[idx]


class WeightedBalancer(LoadBalancer):
    """Weighted load balancer."""

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request: Optional[Dict] = None
    ) -> Optional[ServiceEndpoint]:
        if not endpoints:
            return None

        total_weight = sum(e.weight for e in endpoints)
        if total_weight == 0:
            return random.choice(endpoints)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for endpoint in endpoints:
            cumulative += endpoint.weight
            if r <= cumulative:
                return endpoint

        return endpoints[-1]


class ConsistentHashBalancer(LoadBalancer):
    """Consistent hash load balancer."""

    def __init__(self, hash_key: str = "agent_id"):
        self.hash_key = hash_key
        self._ring: Dict[int, str] = {}
        self._ring_size = 360

    def select(
        self,
        endpoints: List[ServiceEndpoint],
        request: Optional[Dict] = None
    ) -> Optional[ServiceEndpoint]:
        if not endpoints:
            return None

        # Build ring
        self._ring = {}
        for endpoint in endpoints:
            for i in range(3):  # Virtual nodes
                key = f"{endpoint.id}:{i}"
                hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16) % self._ring_size
                self._ring[hash_val] = endpoint.id

        # Get request key
        request = request or {}
        hash_input = request.get(self.hash_key, str(uuid.uuid4()))
        request_hash = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % self._ring_size

        # Find next node on ring
        sorted_hashes = sorted(self._ring.keys())
        for h in sorted_hashes:
            if h >= request_hash:
                endpoint_id = self._ring[h]
                for e in endpoints:
                    if e.id == endpoint_id:
                        return e

        # Wrap around
        if sorted_hashes:
            endpoint_id = self._ring[sorted_hashes[0]]
            for e in endpoints:
                if e.id == endpoint_id:
                    return e

        return endpoints[0]


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes before closing
    timeout_seconds: float = 30.0    # Time in open state
    half_open_requests: int = 3      # Requests allowed in half-open


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_requests = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, with automatic transitions."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if request is allowed."""
        state = self.state

        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            if self._half_open_requests < self.config.half_open_requests:
                self._half_open_requests += 1
                return True
            return False

    def record_success(self):
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def record_failure(self):
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN


# =============================================================================
# Traffic Management
# =============================================================================

@dataclass
class TrafficMatch:
    """Traffic matching rules."""

    # Headers to match
    headers: Dict[str, str] = field(default_factory=dict)

    # Source labels
    source_labels: Dict[str, str] = field(default_factory=dict)

    # URI/path matching
    uri_prefix: Optional[str] = None
    uri_exact: Optional[str] = None
    uri_regex: Optional[str] = None

    # Method
    method: Optional[str] = None

    def matches(self, request: Dict[str, Any]) -> bool:
        """Check if request matches this rule."""
        # Check headers
        req_headers = request.get("headers", {})
        for key, value in self.headers.items():
            if req_headers.get(key) != value:
                return False

        # Check URI
        uri = request.get("uri", "")
        if self.uri_prefix and not uri.startswith(self.uri_prefix):
            return False
        if self.uri_exact and uri != self.uri_exact:
            return False

        # Check method
        if self.method and request.get("method") != self.method:
            return False

        return True


@dataclass
class TrafficRoute:
    """A traffic routing destination."""

    service: str
    weight: int = 100
    port: Optional[int] = None
    subset: Optional[str] = None  # Named subset of endpoints


@dataclass
class TrafficRetry:
    """Retry configuration."""

    attempts: int = 3
    per_try_timeout_ms: int = 2000
    retry_on: List[str] = field(default_factory=lambda: ["5xx", "reset", "connect-failure"])


@dataclass
class TrafficTimeout:
    """Timeout configuration."""

    request_timeout_ms: int = 15000
    idle_timeout_ms: int = 300000


@dataclass
class VirtualService:
    """
    Virtual service for traffic routing.

    Similar to Istio VirtualService.
    """

    name: str
    hosts: List[str]                     # Services this applies to
    namespace: str = "default"

    # HTTP routes
    http_routes: List[Dict[str, Any]] = field(default_factory=list)

    # Default route
    default_route: Optional[TrafficRoute] = None

    # Traffic policies
    retry: Optional[TrafficRetry] = None
    timeout: Optional[TrafficTimeout] = None

    # Fault injection
    fault_delay_percent: float = 0
    fault_delay_ms: int = 0
    fault_abort_percent: float = 0
    fault_abort_code: int = 503

    def add_route(
        self,
        match: TrafficMatch,
        routes: List[TrafficRoute],
        name: str = ""
    ):
        """Add a route rule."""
        self.http_routes.append({
            "name": name or f"route-{len(self.http_routes)}",
            "match": match,
            "routes": routes,
        })

    def route(self, request: Dict[str, Any]) -> List[TrafficRoute]:
        """Get routes for a request."""
        for route_config in self.http_routes:
            match = route_config.get("match")
            if match and match.matches(request):
                return route_config.get("routes", [])

        if self.default_route:
            return [self.default_route]

        return []


@dataclass
class DestinationRule:
    """
    Destination rule for load balancing policy.

    Similar to Istio DestinationRule.
    """

    name: str
    host: str                            # Target service
    namespace: str = "default"

    # Load balancing
    load_balancer: LoadBalanceAlgorithm = LoadBalanceAlgorithm.ROUND_ROBIN

    # Connection pool
    max_connections: int = 1000
    max_pending_requests: int = 1000
    max_requests_per_connection: int = 100

    # Outlier detection
    consecutive_errors: int = 5
    interval_seconds: int = 10
    base_ejection_time_seconds: int = 30
    max_ejection_percent: int = 100

    # Subsets
    subsets: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def add_subset(self, name: str, labels: Dict[str, str]):
        """Add a subset definition."""
        self.subsets[name] = labels


# =============================================================================
# Sidecar Proxy
# =============================================================================

@dataclass
class SidecarMetrics:
    """Metrics collected by the sidecar."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    latency_histogram: Dict[str, int] = field(default_factory=lambda: {
        "p50": 0, "p90": 0, "p95": 0, "p99": 0
    })
    bytes_sent: int = 0
    bytes_received: int = 0
    active_connections: int = 0


class Sidecar:
    """
    Sidecar proxy for an agent.

    Handles all network traffic for the agent.
    """

    def __init__(
        self,
        agent_id: str,
        mesh: "NetworkMesh",
    ):
        self.agent_id = agent_id
        self.mesh = mesh

        # Circuit breakers per service
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Connection pools
        self._connections: Dict[str, List[Any]] = defaultdict(list)

        # Metrics
        self.metrics = SidecarMetrics()

        # Request tracking for distributed tracing
        self._traces: Dict[str, Dict] = {}

    async def send_request(
        self,
        service: str,
        method: str,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Any:
        """Send a request to a service."""
        start_time = time.time()
        headers = headers or {}

        # Generate trace ID if not present
        trace_id = headers.get("x-trace-id", str(uuid.uuid4()))
        span_id = str(uuid.uuid4())[:16]
        headers["x-trace-id"] = trace_id
        headers["x-span-id"] = span_id
        headers["x-parent-span-id"] = headers.get("x-span-id", "")

        request = {
            "service": service,
            "method": method,
            "payload": payload,
            "headers": headers,
            "source": self.agent_id,
        }

        self.metrics.requests_total += 1

        try:
            # Check circuit breaker
            cb = self._get_circuit_breaker(service)
            if not cb.allow_request():
                raise RuntimeError(f"Circuit breaker open for {service}")

            # Get virtual service rules
            routes = self.mesh.get_routes(service, request)
            if not routes:
                raise RuntimeError(f"No routes found for {service}")

            # Select endpoint
            endpoint = await self.mesh.select_endpoint(service, routes[0], request)
            if not endpoint:
                raise RuntimeError(f"No healthy endpoints for {service}")

            # Apply timeout
            timeout = (timeout_ms or 15000) / 1000

            # Inject faults if configured
            vs = self.mesh.get_virtual_service(service)
            if vs:
                if random.random() * 100 < vs.fault_delay_percent:
                    await asyncio.sleep(vs.fault_delay_ms / 1000)
                if random.random() * 100 < vs.fault_abort_percent:
                    raise RuntimeError(f"Fault injection: {vs.fault_abort_code}")

            # Send request
            result = await self._do_request(endpoint, request, timeout)

            # Record success
            cb.record_success()
            endpoint.requests_total += 1
            endpoint.requests_success += 1
            latency_ms = (time.time() - start_time) * 1000
            endpoint.latency_sum_ms += latency_ms

            self.metrics.requests_success += 1

            return result

        except Exception as e:
            # Record failure
            cb = self._get_circuit_breaker(service)
            cb.record_failure()

            self.metrics.requests_failed += 1

            raise

    async def _do_request(
        self,
        endpoint: ServiceEndpoint,
        request: Dict,
        timeout: float
    ) -> Any:
        """Actually send the request."""
        # In production, would use actual network transport
        # For now, simulate via mesh's internal routing
        return await self.mesh.deliver_request(endpoint, request, timeout)

    def _get_circuit_breaker(self, service: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service not in self._circuit_breakers:
            self._circuit_breakers[service] = CircuitBreaker(service)
        return self._circuit_breakers[service]

    def get_metrics(self) -> Dict[str, Any]:
        """Get sidecar metrics."""
        return {
            "requests_total": self.metrics.requests_total,
            "requests_success": self.metrics.requests_success,
            "requests_failed": self.metrics.requests_failed,
            "success_rate": (self.metrics.requests_success / self.metrics.requests_total
                           if self.metrics.requests_total > 0 else 1.0),
            "circuit_breakers": {
                name: cb.state.value for name, cb in self._circuit_breakers.items()
            },
        }


# =============================================================================
# Network Mesh
# =============================================================================

class NetworkMesh:
    """
    Central network mesh for agent communication.

    Provides service mesh capabilities for AI agents.
    """

    def __init__(self):
        # Service registry
        self.registry = ServiceRegistry()

        # Traffic management
        self._virtual_services: Dict[str, VirtualService] = {}
        self._destination_rules: Dict[str, DestinationRule] = {}

        # Load balancers
        self._load_balancers: Dict[LoadBalanceAlgorithm, LoadBalancer] = {
            LoadBalanceAlgorithm.ROUND_ROBIN: RoundRobinBalancer(),
            LoadBalanceAlgorithm.WEIGHTED: WeightedBalancer(),
            LoadBalanceAlgorithm.CONSISTENT_HASH: ConsistentHashBalancer(),
        }

        # Sidecars
        self._sidecars: Dict[str, Sidecar] = {}

        # Request handlers
        self._handlers: Dict[str, Callable] = {}

        # Metrics
        self._mesh_metrics = {
            "services_count": 0,
            "endpoints_count": 0,
            "requests_total": 0,
        }

    # -------------------------------------------------------------------------
    # Service Management
    # -------------------------------------------------------------------------

    def register_service(
        self,
        name: str,
        namespace: str = "default",
        **kwargs
    ) -> Service:
        """Register a new service."""
        service = Service(name=name, namespace=namespace, **kwargs)
        self.registry.register(service)
        self._mesh_metrics["services_count"] = len(self.registry.list())
        return service

    def add_endpoint(
        self,
        service_name: str,
        endpoint_id: str,
        address: str,
        namespace: str = "default",
        **kwargs
    ) -> ServiceEndpoint:
        """Add an endpoint to a service."""
        service = self.registry.get(service_name, namespace)
        if not service:
            service = self.register_service(service_name, namespace)

        endpoint = ServiceEndpoint(
            id=endpoint_id,
            address=address,
            **kwargs
        )
        service.add_endpoint(endpoint)

        self._mesh_metrics["endpoints_count"] = sum(
            len(s.endpoints) for s in self.registry.list()
        )

        return endpoint

    def remove_endpoint(
        self,
        service_name: str,
        endpoint_id: str,
        namespace: str = "default"
    ):
        """Remove an endpoint."""
        service = self.registry.get(service_name, namespace)
        if service:
            service.remove_endpoint(endpoint_id)

    # -------------------------------------------------------------------------
    # Traffic Management
    # -------------------------------------------------------------------------

    def add_virtual_service(self, vs: VirtualService):
        """Add a virtual service."""
        key = f"{vs.namespace}/{vs.name}"
        self._virtual_services[key] = vs

    def add_destination_rule(self, dr: DestinationRule):
        """Add a destination rule."""
        key = f"{dr.namespace}/{dr.name}"
        self._destination_rules[key] = dr

    def get_virtual_service(self, service: str) -> Optional[VirtualService]:
        """Get virtual service for a service."""
        for vs in self._virtual_services.values():
            if service in vs.hosts or f"{service}.{vs.namespace}.svc.bbx.local" in vs.hosts:
                return vs
        return None

    def get_routes(
        self,
        service: str,
        request: Dict[str, Any]
    ) -> List[TrafficRoute]:
        """Get routes for a request."""
        vs = self.get_virtual_service(service)
        if vs:
            return vs.route(request)

        # Default route to the service
        return [TrafficRoute(service=service)]

    async def select_endpoint(
        self,
        service: str,
        route: TrafficRoute,
        request: Dict[str, Any]
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint for routing."""
        svc = self.registry.get(route.service)
        if not svc:
            return None

        endpoints = svc.healthy_endpoints()
        if not endpoints:
            return None

        # Apply subset filter
        if route.subset:
            dr = self._get_destination_rule(service)
            if dr and route.subset in dr.subsets:
                subset_labels = dr.subsets[route.subset]
                endpoints = [
                    e for e in endpoints
                    if all(e.labels.get(k) == v for k, v in subset_labels.items())
                ]

        if not endpoints:
            return None

        # Get load balancer
        dr = self._get_destination_rule(service)
        algorithm = dr.load_balancer if dr else LoadBalanceAlgorithm.ROUND_ROBIN
        lb = self._load_balancers.get(algorithm, self._load_balancers[LoadBalanceAlgorithm.ROUND_ROBIN])

        return lb.select(endpoints, request)

    def _get_destination_rule(self, service: str) -> Optional[DestinationRule]:
        """Get destination rule for a service."""
        for dr in self._destination_rules.values():
            if service == dr.host or f"{service}.{dr.namespace}.svc.bbx.local" == dr.host:
                return dr
        return None

    # -------------------------------------------------------------------------
    # Sidecar Management
    # -------------------------------------------------------------------------

    def create_sidecar(self, agent_id: str) -> Sidecar:
        """Create a sidecar for an agent."""
        sidecar = Sidecar(agent_id, self)
        self._sidecars[agent_id] = sidecar
        return sidecar

    def get_sidecar(self, agent_id: str) -> Optional[Sidecar]:
        """Get sidecar for an agent."""
        return self._sidecars.get(agent_id)

    # -------------------------------------------------------------------------
    # Request Handling
    # -------------------------------------------------------------------------

    def register_handler(
        self,
        service: str,
        method: str,
        handler: Callable
    ):
        """Register a request handler."""
        key = f"{service}/{method}"
        self._handlers[key] = handler

    async def deliver_request(
        self,
        endpoint: ServiceEndpoint,
        request: Dict[str, Any],
        timeout: float
    ) -> Any:
        """Deliver a request to an endpoint."""
        self._mesh_metrics["requests_total"] += 1

        service = request.get("service")
        method = request.get("method")
        payload = request.get("payload")

        handler_key = f"{service}/{method}"
        handler = self._handlers.get(handler_key)

        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await asyncio.wait_for(
                        handler(payload, request),
                        timeout=timeout
                    )
                return handler(payload, request)
            except asyncio.TimeoutError:
                raise RuntimeError(f"Request timeout after {timeout}s")

        raise RuntimeError(f"No handler for {handler_key}")

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get mesh-wide metrics."""
        sidecar_metrics = {}
        for agent_id, sidecar in self._sidecars.items():
            sidecar_metrics[agent_id] = sidecar.get_metrics()

        service_metrics = {}
        for service in self.registry.list():
            endpoints_stats = []
            for ep in service.endpoints.values():
                endpoints_stats.append({
                    "id": ep.id,
                    "status": ep.status.value,
                    "requests": ep.requests_total,
                    "success_rate": ep.success_rate,
                    "avg_latency_ms": ep.avg_latency_ms,
                })
            service_metrics[service.name] = {
                "endpoints": endpoints_stats,
                "healthy_count": len(service.healthy_endpoints()),
            }

        return {
            "mesh": self._mesh_metrics,
            "services": service_metrics,
            "sidecars": sidecar_metrics,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_global_mesh: Optional[NetworkMesh] = None


def get_mesh() -> NetworkMesh:
    """Get the global network mesh."""
    global _global_mesh
    if _global_mesh is None:
        _global_mesh = NetworkMesh()
    return _global_mesh


async def send_to_agent(
    from_agent: str,
    to_service: str,
    method: str,
    payload: Any,
    **kwargs
) -> Any:
    """Send a message to another agent via the mesh."""
    mesh = get_mesh()

    sidecar = mesh.get_sidecar(from_agent)
    if not sidecar:
        sidecar = mesh.create_sidecar(from_agent)

    return await sidecar.send_request(
        service=to_service,
        method=method,
        payload=payload,
        **kwargs
    )


def register_agent_service(
    agent_id: str,
    service_name: str,
    handlers: Dict[str, Callable],
) -> Service:
    """Register an agent as a service in the mesh."""
    mesh = get_mesh()

    service = mesh.register_service(service_name)
    mesh.add_endpoint(
        service_name,
        endpoint_id=agent_id,
        address=f"agent://{agent_id}",
    )

    for method, handler in handlers.items():
        mesh.register_handler(service_name, method, handler)

    mesh.create_sidecar(agent_id)

    return service
