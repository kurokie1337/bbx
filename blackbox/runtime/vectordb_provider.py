# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Real VectorDB Provider - Semantic Memory for AI Agents.

Uses ChromaDB for local vector storage with automatic embedding.

Features:
- Persistent storage on disk
- Automatic text embedding
- Semantic search with similarity scores
- Metadata filtering
- Collection management
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.vectordb")


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Document:
    """Document to store in vector DB"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result from vector DB"""
    id: str
    content: str
    score: float  # 0-1, higher is more similar
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ChromaDB Provider
# =============================================================================


class ChromaDBProvider:
    """
    Real ChromaDB provider for vector storage.

    ChromaDB handles:
    - Automatic text embedding (using sentence-transformers)
    - Persistent storage
    - Efficient similarity search
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or Path.home() / ".bbx" / "vectordb"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collections: Dict[str, Any] = {}
        self._initialized = False

        self._stats = {
            "documents_stored": 0,
            "searches": 0,
            "total_search_time_ms": 0,
        }

    async def initialize(self) -> bool:
        """Initialize ChromaDB client"""
        try:
            import chromadb
            from chromadb.config import Settings

            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            self._initialized = True
            logger.info(f"ChromaDB initialized at {self.persist_dir}")
            return True

        except ImportError:
            logger.error("chromadb package not installed")
            logger.error("Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False

    def _get_collection(self, name: str):
        """Get or create a collection"""
        if not self._initialized or not self._client:
            raise RuntimeError("ChromaDB not initialized")

        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

        return self._collections[name]

    async def store(
        self,
        collection: str,
        documents: List[Document],
    ) -> int:
        """
        Store documents in a collection.

        Args:
            collection: Collection name
            documents: Documents to store

        Returns:
            Number of documents stored
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB not initialized")

        col = self._get_collection(collection)

        # Prepare data
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        # ChromaDB requires non-empty metadata, add default if empty
        metadatas = [
            doc.metadata if doc.metadata else {"_created": time.time()}
            for doc in documents
        ]

        # Store (ChromaDB handles embedding automatically)
        col.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas,
        )

        self._stats["documents_stored"] += len(documents)

        logger.debug(f"Stored {len(documents)} documents in {collection}")
        return len(documents)

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            collection: Collection to search
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of search results sorted by similarity
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB not initialized")

        start = time.time()

        col = self._get_collection(collection)

        # Search
        results = col.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )

        search_time = (time.time() - start) * 1000
        self._stats["searches"] += 1
        self._stats["total_search_time_ms"] += search_time

        # Convert to SearchResult
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # For cosine distance

                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=max(0, min(1, similarity)),  # Clamp to 0-1
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        logger.debug(f"Search in {collection}: {len(search_results)} results in {search_time:.1f}ms")
        return search_results

    async def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete documents from a collection.

        Args:
            collection: Collection name
            ids: Document IDs to delete
            filter_metadata: Delete by metadata filter

        Returns:
            Number of documents deleted
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB not initialized")

        col = self._get_collection(collection)

        if ids:
            col.delete(ids=ids)
            return len(ids)
        elif filter_metadata:
            col.delete(where=filter_metadata)
            return -1  # Unknown count
        else:
            return 0

    async def get(
        self,
        collection: str,
        ids: List[str],
    ) -> List[Document]:
        """
        Get documents by ID.

        Args:
            collection: Collection name
            ids: Document IDs

        Returns:
            List of documents
        """
        if not self._initialized:
            raise RuntimeError("ChromaDB not initialized")

        col = self._get_collection(collection)

        results = col.get(ids=ids)

        documents = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                documents.append(Document(
                    id=doc_id,
                    content=results["documents"][i] if results["documents"] else "",
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                ))

        return documents

    async def list_collections(self) -> List[str]:
        """List all collections"""
        if not self._initialized or not self._client:
            raise RuntimeError("ChromaDB not initialized")

        collections = self._client.list_collections()
        return [col.name for col in collections]

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        if not self._initialized or not self._client:
            raise RuntimeError("ChromaDB not initialized")

        try:
            self._client.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
            return True
        except Exception:
            return False

    async def count(self, collection: str) -> int:
        """Get document count in collection"""
        if not self._initialized:
            raise RuntimeError("ChromaDB not initialized")

        col = self._get_collection(collection)
        return col.count()

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            **self._stats,
            "avg_search_time_ms": (
                self._stats["total_search_time_ms"] / self._stats["searches"]
                if self._stats["searches"] > 0 else 0
            ),
        }

    async def shutdown(self):
        """Shutdown the provider"""
        self._collections.clear()
        self._client = None
        self._initialized = False


# =============================================================================
# Memory Store (High-Level API)
# =============================================================================


class MemoryStore:
    """
    High-level memory store for AI agents.

    Provides semantic memory with automatic organization:
    - Store memories with importance scores
    - Recall by semantic similarity
    - Automatic decay of old memories
    - Agent isolation
    """

    def __init__(self, vectordb: ChromaDBProvider):
        self._vectordb = vectordb
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize memory store"""
        if not self._vectordb._initialized:
            if not await self._vectordb.initialize():
                return False
        self._initialized = True
        return True

    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory for an agent.

        Args:
            agent_id: Agent identifier
            content: Memory content
            memory_type: Type of memory (general, fact, experience, skill)
            importance: Importance score 0-1
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        import uuid
        import time

        memory_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"

        doc = Document(
            id=memory_id,
            content=content,
            metadata={
                "agent_id": agent_id,
                "type": memory_type,
                "importance": importance,
                "timestamp": time.time(),
                **(metadata or {}),
            },
        )

        collection = f"memory_{agent_id}"
        await self._vectordb.store(collection, [doc])

        logger.debug(f"Stored memory {memory_id} for agent {agent_id}")
        return memory_id

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[SearchResult]:
        """
        Recall memories by semantic similarity.

        Args:
            agent_id: Agent identifier
            query: Search query
            top_k: Number of results
            memory_type: Filter by memory type
            min_importance: Minimum importance score

        Returns:
            List of matching memories
        """
        collection = f"memory_{agent_id}"

        # Build filter
        filter_metadata = None
        if memory_type:
            filter_metadata = {"type": memory_type}

        results = await self._vectordb.search(
            collection=collection,
            query=query,
            top_k=top_k * 2,  # Get more, filter later
            filter_metadata=filter_metadata,
        )

        # Filter by importance
        filtered = [
            r for r in results
            if r.metadata.get("importance", 0) >= min_importance
        ]

        return filtered[:top_k]

    async def forget(
        self,
        agent_id: str,
        memory_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> int:
        """
        Forget memories.

        Args:
            agent_id: Agent identifier
            memory_id: Specific memory to forget
            memory_type: Forget all of a type

        Returns:
            Number of memories forgotten
        """
        collection = f"memory_{agent_id}"

        if memory_id:
            await self._vectordb.delete(collection, ids=[memory_id])
            return 1
        elif memory_type:
            await self._vectordb.delete(collection, filter_metadata={"type": memory_type})
            return -1

        return 0

    async def get_agent_memory_count(self, agent_id: str) -> int:
        """Get count of memories for an agent"""
        collection = f"memory_{agent_id}"
        try:
            return await self._vectordb.count(collection)
        except Exception:
            return 0

    async def clear_agent_memory(self, agent_id: str) -> bool:
        """Clear all memories for an agent"""
        collection = f"memory_{agent_id}"
        return await self._vectordb.delete_collection(collection)


# =============================================================================
# Global Instance
# =============================================================================


_vectordb: Optional[ChromaDBProvider] = None
_memory_store: Optional[MemoryStore] = None


async def get_vectordb() -> ChromaDBProvider:
    """Get global VectorDB instance"""
    global _vectordb
    if _vectordb is None:
        _vectordb = ChromaDBProvider()
        await _vectordb.initialize()
    return _vectordb


async def get_memory_store() -> MemoryStore:
    """Get global memory store instance"""
    global _memory_store
    if _memory_store is None:
        vectordb = await get_vectordb()
        _memory_store = MemoryStore(vectordb)
        await _memory_store.initialize()
    return _memory_store
