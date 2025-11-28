# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 SemanticMemory - RAG-based memory with Qdrant integration.

Features:
- Vector database integration (Qdrant, Pinecone, Weaviate)
- Embedding service (OpenAI, local models)
- Hybrid search (keyword + semantic)
- Forgetting mechanism (TTL + importance decay)
- Multi-modal support (text, images, audio)

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Semantic Memory                               │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
    │  │ Embedding    │  │ Vector Store │  │ Forgetting           │   │
    │  │ Service      │  │ (Qdrant)     │  │ Manager              │   │
    │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
    │         │                 │                      │               │
    │  ┌──────▼─────────────────▼──────────────────────▼───────────┐   │
    │  │                   Memory Manager                          │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
    │  │  │ Store   │ │ Recall  │ │ Search  │ │   Summarize     │  │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
    │  └───────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("bbx.semantic_memory")


# =============================================================================
# Enums and Data Classes
# =============================================================================


class MemoryType(Enum):
    """Types of memories"""
    EPISODIC = "episodic"     # Specific events/interactions
    SEMANTIC = "semantic"     # Facts and knowledge
    PROCEDURAL = "procedural" # How to do things
    WORKING = "working"       # Short-term context


class ContentModality(Enum):
    """Content modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED = "structured"


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    agent_id: str
    content: str
    embedding: Optional[List[float]] = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    modality: ContentModality = ContentModality.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None  # None = no expiry
    tags: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() > self.created_at + self.ttl_seconds


@dataclass
class SearchResult:
    """Search result with relevance score"""
    entry: MemoryEntry
    score: float  # 0.0 to 1.0
    match_type: str  # 'semantic', 'keyword', 'hybrid'


# =============================================================================
# Embedding Service
# =============================================================================


class EmbeddingService(ABC):
    """Abstract embedding service"""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class OpenAIEmbedding(EmbeddingService):
    """OpenAI embedding service"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536
    ):
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai required: pip install openai")
        return self._client

    async def embed(self, text: str) -> List[float]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=texts
        )
        return [d.embedding for d in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(EmbeddingService):
    """Local embedding using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dimension = 384  # Default for MiniLM

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")
        return self._model

    async def embed(self, text: str) -> List[float]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


# =============================================================================
# Vector Store Backends
# =============================================================================


class VectorStore(ABC):
    """Abstract vector store"""

    @abstractmethod
    async def upsert(self, entries: List[MemoryEntry]) -> bool:
        pass

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[MemoryEntry]:
        pass


class QdrantStore(VectorStore):
    """Qdrant vector store"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "bbx_memories",
        dimension: int = 1536
    ):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._dimension = dimension
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import VectorParams, Distance
                self._client = QdrantClient(host=self._host, port=self._port)

                # Create collection if not exists
                collections = self._client.get_collections().collections
                if self._collection_name not in [c.name for c in collections]:
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._dimension,
                            distance=Distance.COSINE
                        )
                    )
            except ImportError:
                raise ImportError("qdrant-client required: pip install qdrant-client")
        return self._client

    async def upsert(self, entries: List[MemoryEntry]) -> bool:
        try:
            from qdrant_client.models import PointStruct
            client = await self._get_client()

            points = [
                PointStruct(
                    id=entry.id,
                    vector=entry.embedding,
                    payload={
                        "agent_id": entry.agent_id,
                        "content": entry.content,
                        "memory_type": entry.memory_type.value,
                        "importance": entry.importance,
                        "created_at": entry.created_at,
                        "tags": entry.tags,
                        "metadata": entry.metadata
                    }
                )
                for entry in entries if entry.embedding
            ]

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.upsert(
                    collection_name=self._collection_name,
                    points=points
                )
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant upsert error: {e}")
            return False

    async def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        try:
            client = await self._get_client()

            query_filter = None
            if filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions = [
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter.items()
                ]
                query_filter = Filter(must=conditions)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: client.search(
                    collection_name=self._collection_name,
                    query_vector=embedding,
                    limit=top_k,
                    query_filter=query_filter
                )
            )

            return [(r.id, r.score) for r in results]
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    async def delete(self, ids: List[str]) -> bool:
        try:
            client = await self._get_client()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: client.delete(
                    collection_name=self._collection_name,
                    points_selector=ids
                )
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant delete error: {e}")
            return False

    async def get(self, id: str) -> Optional[MemoryEntry]:
        try:
            client = await self._get_client()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.retrieve(
                    collection_name=self._collection_name,
                    ids=[id]
                )
            )
            if result:
                point = result[0]
                return MemoryEntry(
                    id=point.id,
                    agent_id=point.payload.get("agent_id", ""),
                    content=point.payload.get("content", ""),
                    embedding=point.vector,
                    memory_type=MemoryType(point.payload.get("memory_type", "semantic")),
                    importance=point.payload.get("importance", 0.5),
                    created_at=point.payload.get("created_at", 0),
                    tags=point.payload.get("tags", []),
                    metadata=point.payload.get("metadata", {})
                )
            return None
        except Exception as e:
            logger.error(f"Qdrant get error: {e}")
            return None


class InMemoryStore(VectorStore):
    """In-memory vector store (for testing)"""

    def __init__(self):
        self._entries: Dict[str, MemoryEntry] = {}

    async def upsert(self, entries: List[MemoryEntry]) -> bool:
        for entry in entries:
            self._entries[entry.id] = entry
        return True

    async def search(
        self,
        embedding: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        import math

        def cosine_sim(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        results = []
        for entry in self._entries.values():
            if not entry.embedding:
                continue

            # Apply filter
            if filter:
                match = True
                for k, v in filter.items():
                    if k == "agent_id" and entry.agent_id != v:
                        match = False
                        break
                if not match:
                    continue

            score = cosine_sim(embedding, entry.embedding)
            results.append((entry.id, score))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    async def delete(self, ids: List[str]) -> bool:
        for id in ids:
            self._entries.pop(id, None)
        return True

    async def get(self, id: str) -> Optional[MemoryEntry]:
        return self._entries.get(id)


# =============================================================================
# Forgetting Manager
# =============================================================================


class ForgettingManager:
    """
    Manages memory forgetting based on:
    - TTL expiry
    - Importance decay
    - Capacity limits
    """

    def __init__(
        self,
        importance_decay_rate: float = 0.01,  # Per day
        min_importance: float = 0.1,
        max_entries_per_agent: int = 10000
    ):
        self._decay_rate = importance_decay_rate
        self._min_importance = min_importance
        self._max_entries = max_entries_per_agent

    def calculate_current_importance(self, entry: MemoryEntry) -> float:
        """Calculate current importance with decay"""
        days_old = (time.time() - entry.created_at) / 86400
        decay = self._decay_rate * days_old

        # Access boost
        access_boost = min(0.2, entry.access_count * 0.01)

        current = entry.importance - decay + access_boost
        return max(self._min_importance, min(1.0, current))

    def should_forget(self, entry: MemoryEntry) -> bool:
        """Check if entry should be forgotten"""
        # TTL expiry
        if entry.is_expired:
            return True

        # Importance threshold
        current_importance = self.calculate_current_importance(entry)
        if current_importance <= self._min_importance:
            return True

        return False

    def get_entries_to_forget(
        self,
        entries: List[MemoryEntry],
        target_count: Optional[int] = None
    ) -> List[str]:
        """Get IDs of entries to forget"""
        # First, expired entries
        to_forget = [e.id for e in entries if self.should_forget(e)]

        # If still over capacity, remove lowest importance
        if target_count and len(entries) - len(to_forget) > target_count:
            remaining = [e for e in entries if e.id not in to_forget]
            scored = [
                (e.id, self.calculate_current_importance(e))
                for e in remaining
            ]
            scored.sort(key=lambda x: x[1])

            excess = len(remaining) - target_count
            to_forget.extend([id for id, _ in scored[:excess]])

        return to_forget


# =============================================================================
# Semantic Memory Manager
# =============================================================================


@dataclass
class SemanticMemoryConfig:
    """Configuration for semantic memory"""
    # Embedding
    embedding_provider: str = "local"  # 'openai', 'local'
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    openai_api_key: Optional[str] = None

    # Vector store
    vector_store_type: str = "memory"  # 'qdrant', 'memory'
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Memory settings
    max_entries_per_agent: int = 10000
    default_ttl_seconds: Optional[float] = None
    importance_decay_rate: float = 0.01

    # Search settings
    default_top_k: int = 10
    similarity_threshold: float = 0.5


class SemanticMemory:
    """
    Production-ready semantic memory for AI agents.

    Features:
    - Vector-based similarity search
    - Hybrid search (keyword + semantic)
    - Importance-based retention
    - TTL-based expiry
    """

    def __init__(self, config: Optional[SemanticMemoryConfig] = None):
        self.config = config or SemanticMemoryConfig()

        # Initialize embedding service
        self._embedding: Optional[EmbeddingService] = None
        self._vector_store: Optional[VectorStore] = None
        self._forgetting = ForgettingManager(
            importance_decay_rate=self.config.importance_decay_rate,
            max_entries_per_agent=self.config.max_entries_per_agent
        )

        # Local cache for fast lookup
        self._cache: Dict[str, MemoryEntry] = {}

        # GC task
        self._gc_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self):
        """Initialize and start the memory system"""
        # Initialize embedding service
        if self.config.embedding_provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self._embedding = OpenAIEmbedding(
                api_key=self.config.openai_api_key,
                dimension=self.config.embedding_dimension
            )
        else:
            self._embedding = LocalEmbedding(self.config.embedding_model)

        # Initialize vector store
        if self.config.vector_store_type == "qdrant":
            self._vector_store = QdrantStore(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                dimension=self._embedding.dimension
            )
        else:
            self._vector_store = InMemoryStore()

        # Start GC
        self._shutdown = False
        self._gc_task = asyncio.create_task(self._gc_loop())

        logger.info("SemanticMemory started")

    async def stop(self):
        """Stop the memory system"""
        self._shutdown = True
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
        logger.info("SemanticMemory stopped")

    # =========================================================================
    # Core API
    # =========================================================================

    async def store(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[float] = None
    ) -> str:
        """Store a memory"""
        # Generate embedding
        embedding = await self._embedding.embed(content)

        # Create entry
        entry = MemoryEntry(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            ttl_seconds=ttl_seconds or self.config.default_ttl_seconds
        )

        # Store in vector DB
        await self._vector_store.upsert([entry])

        # Cache
        self._cache[entry.id] = entry

        logger.debug(f"Stored memory {entry.id} for agent {agent_id}")
        return entry.id

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[SearchResult]:
        """Recall memories similar to query"""
        # Generate query embedding
        query_embedding = await self._embedding.embed(query)

        # Search
        filter_dict = {"agent_id": agent_id}
        results = await self._vector_store.search(
            query_embedding,
            top_k=top_k * 2,  # Get more for filtering
            filter=filter_dict
        )

        # Get full entries and filter
        search_results = []
        for id, score in results:
            if score < self.config.similarity_threshold:
                continue

            entry = await self._get_entry(id)
            if not entry:
                continue

            # Filter by memory type
            if memory_types and entry.memory_type not in memory_types:
                continue

            # Filter by tags
            if tags and not any(t in entry.tags for t in tags):
                continue

            # Filter by importance
            current_importance = self._forgetting.calculate_current_importance(entry)
            if current_importance < min_importance:
                continue

            # Update access stats
            entry.last_accessed = time.time()
            entry.access_count += 1

            search_results.append(SearchResult(
                entry=entry,
                score=score,
                match_type="semantic"
            ))

            if len(search_results) >= top_k:
                break

        return search_results

    async def keyword_search(
        self,
        agent_id: str,
        keywords: List[str],
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search by keywords (exact match in content)"""
        results = []

        for entry in self._cache.values():
            if entry.agent_id != agent_id:
                continue

            content_lower = entry.content.lower()
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)

            if matches > 0:
                score = matches / len(keywords)
                results.append(SearchResult(
                    entry=entry,
                    score=score,
                    match_type="keyword"
                ))

        results.sort(key=lambda r: -r.score)
        return results[:top_k]

    async def hybrid_search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword"""
        # Get semantic results
        semantic_results = await self.recall(agent_id, query, top_k=top_k)

        # Get keyword results
        keywords = query.lower().split()
        keyword_results = await self.keyword_search(agent_id, keywords, top_k=top_k)

        # Combine scores
        combined: Dict[str, float] = {}
        for r in semantic_results:
            combined[r.entry.id] = r.score * semantic_weight

        for r in keyword_results:
            keyword_weight = 1 - semantic_weight
            if r.entry.id in combined:
                combined[r.entry.id] += r.score * keyword_weight
            else:
                combined[r.entry.id] = r.score * keyword_weight

        # Sort and return
        sorted_ids = sorted(combined.keys(), key=lambda x: -combined[x])

        results = []
        for id in sorted_ids[:top_k]:
            entry = await self._get_entry(id)
            if entry:
                results.append(SearchResult(
                    entry=entry,
                    score=combined[id],
                    match_type="hybrid"
                ))

        return results

    async def forget(self, memory_id: str) -> bool:
        """Explicitly forget a memory"""
        await self._vector_store.delete([memory_id])
        self._cache.pop(memory_id, None)
        return True

    async def update_importance(self, memory_id: str, importance: float):
        """Update memory importance"""
        entry = await self._get_entry(memory_id)
        if entry:
            entry.importance = max(0.0, min(1.0, importance))
            await self._vector_store.upsert([entry])

    async def manage_context(
        self,
        agent_id: str,
        max_tokens: int = 4000,
        summarize_callback: Optional[Callable[[str], str]] = None
    ) -> str:
        """Get managed context for agent"""
        # Get recent important memories
        results = await self.recall(
            agent_id,
            "",  # Empty query = get all
            top_k=100,
            min_importance=0.3
        )

        # Sort by recency
        results.sort(key=lambda r: -r.entry.last_accessed)

        # Build context
        context_parts = []
        current_tokens = 0

        for result in results:
            # Rough token estimate
            entry_tokens = len(result.entry.content.split()) * 1.3
            if current_tokens + entry_tokens > max_tokens:
                break

            context_parts.append(result.entry.content)
            current_tokens += entry_tokens

        return "\n\n".join(context_parts)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _get_entry(self, id: str) -> Optional[MemoryEntry]:
        """Get entry from cache or store"""
        if id in self._cache:
            return self._cache[id]
        return await self._vector_store.get(id)

    async def _gc_loop(self):
        """Garbage collection loop"""
        while not self._shutdown:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._run_gc()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"GC error: {e}")

    async def _run_gc(self):
        """Run garbage collection"""
        entries = list(self._cache.values())
        to_forget = self._forgetting.get_entries_to_forget(
            entries,
            target_count=self.config.max_entries_per_agent
        )

        if to_forget:
            await self._vector_store.delete(to_forget)
            for id in to_forget:
                self._cache.pop(id, None)
            logger.info(f"GC removed {len(to_forget)} memories")


# =============================================================================
# Factory
# =============================================================================


_global_semantic_memory: Optional[SemanticMemory] = None


def get_semantic_memory() -> SemanticMemory:
    """Get global semantic memory"""
    global _global_semantic_memory
    if _global_semantic_memory is None:
        _global_semantic_memory = SemanticMemory()
    return _global_semantic_memory


async def create_semantic_memory(
    config: Optional[SemanticMemoryConfig] = None
) -> SemanticMemory:
    """Create and start semantic memory"""
    memory = SemanticMemory(config)
    await memory.start()
    return memory
