# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Agent Abstraction Layer (AAL) - Windows HAL-Inspired

Inspired by Windows Hardware Abstraction Layer (HAL), provides:
- Complete isolation from backend implementations
- Unified interface for LLMs, vector stores, databases
- Backend auto-selection and failover
- Cost optimization via backend routing
- Backend capability negotiation

Key concepts:
- Backend: An implementation (OpenAI, Ollama, Anthropic, etc.)
- Provider: Category of backends (LLM, VectorStore, Database)
- Capability: What a backend can do (streaming, embeddings, etc.)
- AAL: The abstraction layer that hides backend details
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any, AsyncGenerator, Callable, Dict, Generic, List,
    Optional, Protocol, Set, Tuple, TypeVar, Union
)

logger = logging.getLogger("bbx.aal")


# =============================================================================
# Backend Capabilities
# =============================================================================

class Capability(Enum):
    """Capabilities that backends can provide."""

    # LLM Capabilities
    LLM_COMPLETION = "llm.completion"
    LLM_CHAT = "llm.chat"
    LLM_STREAMING = "llm.streaming"
    LLM_FUNCTION_CALLING = "llm.function_calling"
    LLM_VISION = "llm.vision"
    LLM_EMBEDDINGS = "llm.embeddings"
    LLM_JSON_MODE = "llm.json_mode"

    # Vector Store Capabilities
    VECTOR_STORE = "vector.store"
    VECTOR_SEARCH = "vector.search"
    VECTOR_HYBRID_SEARCH = "vector.hybrid_search"
    VECTOR_METADATA_FILTER = "vector.metadata_filter"
    VECTOR_BATCH_INSERT = "vector.batch_insert"

    # Database Capabilities
    DB_SQL = "db.sql"
    DB_NOSQL = "db.nosql"
    DB_TRANSACTIONS = "db.transactions"
    DB_STREAMING = "db.streaming"

    # Storage Capabilities
    STORAGE_FILE = "storage.file"
    STORAGE_OBJECT = "storage.object"
    STORAGE_STREAMING = "storage.streaming"

    # Execution Capabilities
    EXEC_SHELL = "exec.shell"
    EXEC_PYTHON = "exec.python"
    EXEC_SANDBOX = "exec.sandbox"
    EXEC_CONTAINER = "exec.container"


@dataclass
class BackendCapabilities:
    """Capabilities of a specific backend."""

    capabilities: Set[Capability] = field(default_factory=set)
    limits: Dict[str, Any] = field(default_factory=dict)  # e.g., max_tokens, rate_limits
    features: Dict[str, Any] = field(default_factory=dict)  # e.g., models, versions

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def has_all(self, capabilities: List[Capability]) -> bool:
        return all(c in self.capabilities for c in capabilities)

    def has_any(self, capabilities: List[Capability]) -> bool:
        return any(c in self.capabilities for c in capabilities)


# =============================================================================
# Backend Health & Metrics
# =============================================================================

class BackendHealth(Enum):
    """Health status of a backend."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendMetrics:
    """Metrics for a backend."""

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    latency_sum_ms: float = 0
    latency_samples: int = 0

    # Cost tracking
    tokens_used: int = 0
    cost_usd: float = 0

    # Last status
    last_request: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    # Health
    health: BackendHealth = BackendHealth.UNKNOWN
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 1.0
        return self.requests_success / self.requests_total

    @property
    def avg_latency_ms(self) -> float:
        if self.latency_samples == 0:
            return 0.0
        return self.latency_sum_ms / self.latency_samples

    def record_success(self, latency_ms: float, tokens: int = 0, cost: float = 0):
        self.requests_total += 1
        self.requests_success += 1
        self.latency_sum_ms += latency_ms
        self.latency_samples += 1
        self.tokens_used += tokens
        self.cost_usd += cost
        self.last_request = datetime.now()
        self.consecutive_failures = 0
        self.health = BackendHealth.HEALTHY

    def record_failure(self, error: str):
        self.requests_total += 1
        self.requests_failed += 1
        self.last_error = error
        self.last_error_time = datetime.now()
        self.consecutive_failures += 1

        if self.consecutive_failures >= 5:
            self.health = BackendHealth.UNHEALTHY
        elif self.consecutive_failures >= 2:
            self.health = BackendHealth.DEGRADED


# =============================================================================
# Backend Interface
# =============================================================================

T = TypeVar('T')


class Backend(ABC, Generic[T]):
    """
    Abstract base class for AAL backends.

    Each backend implements a specific provider interface
    (LLM, VectorStore, etc.) and hides implementation details.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.capabilities = BackendCapabilities()
        self.metrics = BackendMetrics()
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend connection."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend connection."""
        pass

    @abstractmethod
    async def health_check(self) -> BackendHealth:
        """Check backend health."""
        pass

    @abstractmethod
    async def execute(self, method: str, params: Dict[str, Any]) -> T:
        """Execute a method on this backend."""
        pass

    def get_capabilities(self) -> BackendCapabilities:
        """Get backend capabilities."""
        return self.capabilities

    def get_metrics(self) -> BackendMetrics:
        """Get backend metrics."""
        return self.metrics


# =============================================================================
# LLM Backend Interface
# =============================================================================

@dataclass
class LLMRequest:
    """Request to an LLM backend."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    functions: Optional[List[Dict]] = None
    response_format: Optional[Dict] = None


@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    function_call: Optional[Dict] = None


class LLMBackend(Backend[LLMResponse]):
    """Abstract LLM backend."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a prompt."""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        pass

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        pass

    async def execute(self, method: str, params: Dict[str, Any]) -> LLMResponse:
        if method == "complete":
            request = LLMRequest(**params)
            return await self.complete(request)
        elif method == "embed":
            return await self.embed(params.get("texts", []))
        else:
            raise ValueError(f"Unknown method: {method}")


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("openai", config)
        self.capabilities.capabilities = {
            Capability.LLM_COMPLETION,
            Capability.LLM_CHAT,
            Capability.LLM_STREAMING,
            Capability.LLM_FUNCTION_CALLING,
            Capability.LLM_VISION,
            Capability.LLM_EMBEDDINGS,
            Capability.LLM_JSON_MODE,
        }
        self.capabilities.limits = {
            "max_tokens": 128000,
            "rate_limit_rpm": 10000,
        }
        self._client = None

    async def initialize(self) -> bool:
        try:
            # Would import openai here
            # from openai import AsyncOpenAI
            # self._client = AsyncOpenAI(api_key=self.config.get("api_key"))
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    async def shutdown(self) -> None:
        self._client = None
        self._initialized = False

    async def health_check(self) -> BackendHealth:
        if not self._initialized:
            return BackendHealth.UNKNOWN
        try:
            # Would make a simple API call
            return BackendHealth.HEALTHY
        except Exception:
            return BackendHealth.UNHEALTHY

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.time()
        try:
            # Simulated response - would use actual API
            response = LLMResponse(
                content="Simulated OpenAI response",
                model=request.model or "gpt-4",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )

            latency = (time.time() - start) * 1000
            self.metrics.record_success(
                latency,
                tokens=response.usage.get("total_tokens", 30),
                cost=0.001
            )
            return response

        except Exception as e:
            self.metrics.record_failure(str(e))
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        # Simulated streaming
        for chunk in ["Hello", " ", "world", "!"]:
            yield chunk
            await asyncio.sleep(0.01)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Simulated embeddings
        return [[0.1] * 1536 for _ in texts]


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ollama", config)
        self.capabilities.capabilities = {
            Capability.LLM_COMPLETION,
            Capability.LLM_CHAT,
            Capability.LLM_STREAMING,
            Capability.LLM_EMBEDDINGS,
        }
        self.capabilities.limits = {
            "max_tokens": 32000,
        }
        self._base_url = config.get("base_url", "http://localhost:11434") if config else "http://localhost:11434"

    async def initialize(self) -> bool:
        # Check if Ollama is running
        self._initialized = True
        return True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> BackendHealth:
        if not self._initialized:
            return BackendHealth.UNKNOWN
        return BackendHealth.HEALTHY

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.time()
        try:
            response = LLMResponse(
                content="Simulated Ollama response",
                model=request.model or "llama3",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )

            latency = (time.time() - start) * 1000
            self.metrics.record_success(latency, tokens=30, cost=0)  # Local = free
            return response

        except Exception as e:
            self.metrics.record_failure(str(e))
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        for chunk in ["Local", " ", "LLM", " ", "response"]:
            yield chunk
            await asyncio.sleep(0.01)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 768 for _ in texts]


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("anthropic", config)
        self.capabilities.capabilities = {
            Capability.LLM_COMPLETION,
            Capability.LLM_CHAT,
            Capability.LLM_STREAMING,
            Capability.LLM_FUNCTION_CALLING,
            Capability.LLM_VISION,
        }
        self.capabilities.limits = {
            "max_tokens": 200000,
            "rate_limit_rpm": 4000,
        }

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def shutdown(self) -> None:
        self._initialized = False

    async def health_check(self) -> BackendHealth:
        return BackendHealth.HEALTHY if self._initialized else BackendHealth.UNKNOWN

    async def complete(self, request: LLMRequest) -> LLMResponse:
        start = time.time()
        try:
            response = LLMResponse(
                content="Simulated Claude response",
                model=request.model or "claude-3-opus",
                finish_reason="end_turn",
                usage={"input_tokens": 10, "output_tokens": 20},
            )

            latency = (time.time() - start) * 1000
            self.metrics.record_success(latency, tokens=30, cost=0.002)
            return response

        except Exception as e:
            self.metrics.record_failure(str(e))
            raise

    async def stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        for chunk in ["Claude", " ", "response", " ", "here"]:
            yield chunk
            await asyncio.sleep(0.01)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Anthropic doesn't support embeddings")


# =============================================================================
# Vector Store Backend Interface
# =============================================================================

@dataclass
class VectorQuery:
    """Query for vector search."""
    embedding: List[float]
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True


@dataclass
class VectorResult:
    """Result from vector search."""
    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class VectorStoreBackend(Backend[List[VectorResult]]):
    """Abstract vector store backend."""

    @abstractmethod
    async def insert(self, vectors: List[Dict[str, Any]]) -> List[str]:
        """Insert vectors."""
        pass

    @abstractmethod
    async def search(self, query: VectorQuery) -> List[VectorResult]:
        """Search vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        pass


# =============================================================================
# Backend Router - Smart Selection
# =============================================================================

class RoutingStrategy(Enum):
    """Strategies for backend selection."""
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    HIGHEST_SUCCESS_RATE = "highest_success_rate"
    CAPABILITY_MATCH = "capability_match"
    FAILOVER = "failover"


@dataclass
class RoutingRule:
    """A rule for backend routing."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    backend: str
    priority: int = 0


class BackendRouter:
    """
    Routes requests to appropriate backends.

    Like HAL routing to different hardware, but for AI services.
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.FAILOVER):
        self.strategy = strategy
        self.backends: Dict[str, Backend] = {}
        self.rules: List[RoutingRule] = []
        self._round_robin_idx = 0

    def register_backend(self, backend: Backend):
        """Register a backend."""
        self.backends[backend.name] = backend

    def add_rule(self, rule: RoutingRule):
        """Add a routing rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def select_backend(
        self,
        required_capabilities: Optional[List[Capability]] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Backend]:
        """Select the best backend for a request."""
        context = request_context or {}

        # Apply rules first
        for rule in self.rules:
            if rule.condition(context):
                if rule.backend in self.backends:
                    return self.backends[rule.backend]

        # Filter by capabilities
        candidates = list(self.backends.values())
        if required_capabilities:
            candidates = [
                b for b in candidates
                if b.capabilities.has_all(required_capabilities)
            ]

        if not candidates:
            return None

        # Filter by health
        healthy = [b for b in candidates if b.metrics.health == BackendHealth.HEALTHY]
        if healthy:
            candidates = healthy

        # Apply strategy
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            idx = self._round_robin_idx % len(candidates)
            self._round_robin_idx += 1
            return candidates[idx]

        elif self.strategy == RoutingStrategy.LEAST_LATENCY:
            return min(candidates, key=lambda b: b.metrics.avg_latency_ms or float('inf'))

        elif self.strategy == RoutingStrategy.LEAST_COST:
            return min(candidates, key=lambda b: b.metrics.cost_usd)

        elif self.strategy == RoutingStrategy.HIGHEST_SUCCESS_RATE:
            return max(candidates, key=lambda b: b.metrics.success_rate)

        elif self.strategy == RoutingStrategy.FAILOVER:
            # Return first healthy backend
            for b in candidates:
                if b.metrics.health in (BackendHealth.HEALTHY, BackendHealth.UNKNOWN):
                    return b
            return candidates[0] if candidates else None

        return candidates[0] if candidates else None


# =============================================================================
# Agent Abstraction Layer (Main Interface)
# =============================================================================

class AAL:
    """
    Agent Abstraction Layer - HAL for AI Agents.

    Provides a unified interface to all backend services,
    hiding implementation details from agents.

    Usage:
        aal = AAL()
        await aal.initialize()

        # Agent doesn't know if this uses OpenAI, Ollama, or Anthropic
        response = await aal.llm.complete(prompt="Hello")

        # Automatically routes to best available backend
        embeddings = await aal.llm.embed(["text1", "text2"])
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Routers for different provider types
        self._llm_router = BackendRouter(RoutingStrategy.FAILOVER)
        self._vector_router = BackendRouter(RoutingStrategy.FAILOVER)
        self._db_router = BackendRouter(RoutingStrategy.FAILOVER)

        # Provider interfaces
        self.llm: Optional[LLMProvider] = None
        self.vector: Optional[VectorProvider] = None

        # All backends
        self._all_backends: List[Backend] = []

        # Stats
        self._stats = {
            "requests_total": 0,
            "requests_routed": 0,
            "failovers": 0,
        }

    async def initialize(self) -> bool:
        """Initialize all backends."""
        logger.info("Initializing Agent Abstraction Layer...")

        # Register default backends based on config
        if self.config.get("openai_api_key"):
            backend = OpenAIBackend({"api_key": self.config["openai_api_key"]})
            await backend.initialize()
            self._llm_router.register_backend(backend)
            self._all_backends.append(backend)
            logger.info("  Registered OpenAI backend")

        if self.config.get("anthropic_api_key"):
            backend = AnthropicBackend({"api_key": self.config["anthropic_api_key"]})
            await backend.initialize()
            self._llm_router.register_backend(backend)
            self._all_backends.append(backend)
            logger.info("  Registered Anthropic backend")

        # Always try to register Ollama (local)
        ollama = OllamaBackend()
        if await ollama.initialize():
            self._llm_router.register_backend(ollama)
            self._all_backends.append(ollama)
            logger.info("  Registered Ollama backend")

        # Create provider interfaces
        self.llm = LLMProvider(self._llm_router)

        logger.info("Agent Abstraction Layer initialized")
        return True

    async def shutdown(self):
        """Shutdown all backends."""
        for backend in self._all_backends:
            await backend.shutdown()

    def register_backend(self, provider_type: str, backend: Backend):
        """Register a custom backend."""
        if provider_type == "llm":
            self._llm_router.register_backend(backend)
        elif provider_type == "vector":
            self._vector_router.register_backend(backend)
        elif provider_type == "db":
            self._db_router.register_backend(backend)

        self._all_backends.append(backend)

    def add_routing_rule(self, provider_type: str, rule: RoutingRule):
        """Add a routing rule."""
        if provider_type == "llm":
            self._llm_router.add_rule(rule)
        elif provider_type == "vector":
            self._vector_router.add_rule(rule)

    async def health_check(self) -> Dict[str, BackendHealth]:
        """Check health of all backends."""
        results = {}
        for backend in self._all_backends:
            health = await backend.health_check()
            results[backend.name] = health
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get AAL metrics."""
        backend_metrics = {}
        for backend in self._all_backends:
            backend_metrics[backend.name] = {
                "requests": backend.metrics.requests_total,
                "success_rate": backend.metrics.success_rate,
                "avg_latency_ms": backend.metrics.avg_latency_ms,
                "cost_usd": backend.metrics.cost_usd,
                "health": backend.metrics.health.value,
            }

        return {
            **self._stats,
            "backends": backend_metrics,
        }


# =============================================================================
# Provider Interfaces (what agents see)
# =============================================================================

class LLMProvider:
    """
    LLM Provider interface - what agents use.

    Agents call methods on this class without knowing
    which backend is actually handling the request.
    """

    def __init__(self, router: BackendRouter):
        self._router = router

    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete a prompt or continue a conversation."""
        # Build messages
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Select backend
        backend = self._router.select_backend(
            required_capabilities=[Capability.LLM_CHAT]
        )

        if not backend or not isinstance(backend, LLMBackend):
            raise RuntimeError("No LLM backend available")

        return await backend.complete(request)

    async def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        request = LLMRequest(
            messages=messages,
            stream=True,
            **kwargs
        )

        backend = self._router.select_backend(
            required_capabilities=[Capability.LLM_STREAMING]
        )

        if not backend or not isinstance(backend, LLMBackend):
            raise RuntimeError("No streaming LLM backend available")

        async for chunk in backend.stream(request):
            yield chunk

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for text."""
        if isinstance(texts, str):
            texts = [texts]

        backend = self._router.select_backend(
            required_capabilities=[Capability.LLM_EMBEDDINGS]
        )

        if not backend or not isinstance(backend, LLMBackend):
            raise RuntimeError("No embedding backend available")

        return await backend.embed(texts)


class VectorProvider:
    """Vector Store Provider interface."""

    def __init__(self, router: BackendRouter):
        self._router = router

    async def store(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Store vectors."""
        backend = self._router.select_backend(
            required_capabilities=[Capability.VECTOR_STORE]
        )

        if not backend or not isinstance(backend, VectorStoreBackend):
            raise RuntimeError("No vector store backend available")

        vectors = []
        for i, text in enumerate(texts):
            vectors.append({
                "text": text,
                "embedding": embeddings[i] if embeddings else None,
                "metadata": metadata[i] if metadata else {},
            })

        return await backend.insert(vectors)

    async def search(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[VectorResult]:
        """Search vectors."""
        backend = self._router.select_backend(
            required_capabilities=[Capability.VECTOR_SEARCH]
        )

        if not backend or not isinstance(backend, VectorStoreBackend):
            raise RuntimeError("No vector store backend available")

        # Convert text to embedding if needed
        if isinstance(query, str):
            # Would need LLM backend for embedding
            raise ValueError("Text query requires embedding - pass embedding directly")

        vector_query = VectorQuery(
            embedding=query,
            top_k=top_k,
            filter=filter,
        )

        return await backend.search(vector_query)


# =============================================================================
# Global AAL Instance
# =============================================================================

_global_aal: Optional[AAL] = None


def get_aal() -> AAL:
    """Get the global AAL instance."""
    global _global_aal
    if _global_aal is None:
        _global_aal = AAL()
    return _global_aal


async def initialize_aal(config: Optional[Dict[str, Any]] = None) -> AAL:
    """Initialize and return the global AAL."""
    global _global_aal
    _global_aal = AAL(config)
    await _global_aal.initialize()
    return _global_aal
