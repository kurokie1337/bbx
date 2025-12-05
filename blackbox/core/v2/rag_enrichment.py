# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX RAG Enrichment - Automatic context enrichment from memory.

Middleware that enriches any AI prompt with relevant context from SemanticMemory.
Your exocortex - never forget, always have context.

Features:
- Automatic semantic search on any prompt
- Smart context injection (prepend, append, system prompt)
- Relevance filtering (threshold-based)
- Token budget management
- Source attribution

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     User Prompt                                  │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   RAG Middleware                                 │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐     │
    │  │ Extract      │  │ Search       │  │ Inject Context     │     │
    │  │ Keywords     │→ │ Memory       │→ │ into Prompt        │     │
    │  └──────────────┘  └──────────────┘  └────────────────────┘     │
    └────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Enriched Prompt → AI                           │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("bbx.rag")


class InjectionMode(Enum):
    """Where to inject context"""
    PREPEND = "prepend"      # Before user prompt
    APPEND = "append"        # After user prompt
    SYSTEM = "system"        # As system prompt addition
    STRUCTURED = "structured"  # XML-like structure


@dataclass
class RAGConfig:
    """Configuration for RAG enrichment"""
    # Search settings
    top_k: int = 5
    min_relevance: float = 0.3
    max_context_chars: int = 4000

    # Injection settings
    mode: InjectionMode = InjectionMode.PREPEND

    # Context formatting
    context_header: str = "Relevant context from memory:"
    context_footer: str = ""
    include_sources: bool = True
    include_scores: bool = False

    # Memory types to search
    memory_types: Optional[List[str]] = None  # None = all types

    # Performance
    timeout_seconds: float = 5.0
    fallback_on_error: bool = True


@dataclass
class RAGResult:
    """Result of RAG enrichment"""
    original_prompt: str
    enriched_prompt: str
    context_added: bool
    memories_found: int
    memories_used: int
    sources: List[Dict[str, Any]] = field(default_factory=list)
    search_time_ms: float = 0


class RAGEnrichment:
    """
    RAG middleware for automatic prompt enrichment.

    Usage:
        rag = RAGEnrichment(memory)

        # Enrich any prompt
        result = await rag.enrich("How do I deploy to AWS?")
        send_to_ai(result.enriched_prompt)

        # Or use as decorator
        @rag.wrap
        async def query_ai(prompt: str) -> str:
            return await claude.query(prompt)

        # With custom config
        config = RAGConfig(top_k=10, min_relevance=0.5)
        result = await rag.enrich("...", config=config)
    """

    def __init__(
        self,
        memory=None,
        default_config: Optional[RAGConfig] = None,
    ):
        """
        Args:
            memory: SemanticMemory instance
            default_config: Default RAG configuration
        """
        self.memory = memory
        self.default_config = default_config or RAGConfig()
        self._stats = {
            "enrichments": 0,
            "memories_retrieved": 0,
            "fallbacks": 0,
        }

    async def enrich(
        self,
        prompt: str,
        config: Optional[RAGConfig] = None,
        agent_id: str = "default",
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResult:
        """
        Enrich a prompt with relevant context from memory.

        Args:
            prompt: Original prompt
            config: Optional override config
            agent_id: Agent ID for memory search
            extra_filters: Additional filters for search

        Returns:
            RAGResult with enriched prompt
        """
        config = config or self.default_config
        result = RAGResult(
            original_prompt=prompt,
            enriched_prompt=prompt,
            context_added=False,
            memories_found=0,
            memories_used=0,
        )

        if not self.memory:
            logger.warning("No memory instance configured for RAG")
            return result

        import time
        start = time.time()

        try:
            # Search memory
            search_results = await asyncio.wait_for(
                self._search_memory(prompt, config, agent_id, extra_filters),
                timeout=config.timeout_seconds
            )

            result.memories_found = len(search_results)

            # Filter by relevance
            relevant = [
                r for r in search_results
                if r.get("score", 0) >= config.min_relevance
            ]

            if not relevant:
                logger.debug(f"No relevant memories found (threshold: {config.min_relevance})")
                return result

            # Build context
            context, sources = self._build_context(relevant, config)

            # Inject into prompt
            result.enriched_prompt = self._inject_context(
                prompt, context, config
            )
            result.context_added = True
            result.memories_used = len(relevant)
            result.sources = sources

            self._stats["enrichments"] += 1
            self._stats["memories_retrieved"] += len(relevant)

        except asyncio.TimeoutError:
            logger.warning(f"RAG search timed out after {config.timeout_seconds}s")
            self._stats["fallbacks"] += 1

        except Exception as e:
            logger.error(f"RAG enrichment error: {e}")
            if not config.fallback_on_error:
                raise
            self._stats["fallbacks"] += 1

        result.search_time_ms = (time.time() - start) * 1000
        return result

    async def _search_memory(
        self,
        query: str,
        config: RAGConfig,
        agent_id: str,
        extra_filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Search memory for relevant context"""
        # Extract keywords for hybrid search
        keywords = self._extract_keywords(query)

        try:
            # Try hybrid search first
            results = await self.memory.hybrid_search(
                agent_id=agent_id,
                query=query,
                keywords=keywords,
                top_k=config.top_k * 2,  # Get more for filtering
            )
        except (AttributeError, TypeError):
            # Fallback to regular recall
            results = await self.memory.recall(
                agent_id=agent_id,
                query=query,
                top_k=config.top_k * 2,
            )

        # Convert to standardized format
        standardized = []
        for r in results:
            if hasattr(r, 'entry'):
                # SearchResult format
                entry = r.entry
                score = r.score
            elif isinstance(r, dict):
                entry = r
                score = r.get('score', 0.5)
            else:
                continue

            content = getattr(entry, 'content', entry.get('content', ''))
            metadata = getattr(entry, 'metadata', entry.get('metadata', {}))
            memory_type = getattr(entry, 'memory_type', entry.get('memory_type', 'unknown'))
            if hasattr(memory_type, 'value'):
                memory_type = memory_type.value

            standardized.append({
                'content': content,
                'score': score,
                'memory_type': memory_type,
                'metadata': metadata,
            })

        # Filter by memory type if specified
        if config.memory_types:
            standardized = [
                r for r in standardized
                if r['memory_type'] in config.memory_types
            ]

        # Sort by score
        standardized.sort(key=lambda x: x['score'], reverse=True)

        return standardized[:config.top_k]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords for hybrid search"""
        # Simple keyword extraction
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'shall', 'can', 'need', 'dare', 'ought', 'used', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why',
            'how', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
            'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        }

        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter
        keywords = [
            w for w in words
            if w not in stopwords and len(w) > 2
        ]

        # Return unique, keeping order
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique[:10]  # Limit to top 10

    def _build_context(
        self,
        memories: List[Dict[str, Any]],
        config: RAGConfig,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build context string from memories"""
        parts = []
        sources = []
        current_chars = 0

        for i, mem in enumerate(memories):
            content = mem['content']
            score = mem['score']
            metadata = mem['metadata']

            # Check character budget
            if current_chars + len(content) > config.max_context_chars:
                # Truncate to fit
                remaining = config.max_context_chars - current_chars
                if remaining < 100:
                    break
                content = content[:remaining] + "..."

            # Format entry
            if config.include_scores:
                entry = f"[{i+1}] (relevance: {score:.2f})\n{content}"
            else:
                entry = f"[{i+1}]\n{content}"

            parts.append(entry)
            current_chars += len(content)

            # Track source
            source_info = {
                'index': i + 1,
                'score': score,
                'type': mem['memory_type'],
            }
            if metadata.get('source_file'):
                source_info['file'] = metadata['source_file']
            if metadata.get('topics'):
                source_info['topics'] = metadata['topics']
            sources.append(source_info)

        context = "\n\n".join(parts)
        return context, sources

    def _inject_context(
        self,
        prompt: str,
        context: str,
        config: RAGConfig,
    ) -> str:
        """Inject context into prompt based on mode"""
        header = config.context_header
        footer = config.context_footer

        if config.mode == InjectionMode.PREPEND:
            return f"{header}\n\n{context}\n\n{footer}\n\n---\n\n{prompt}".strip()

        elif config.mode == InjectionMode.APPEND:
            return f"{prompt}\n\n---\n\n{header}\n\n{context}\n\n{footer}".strip()

        elif config.mode == InjectionMode.SYSTEM:
            # Return as tuple for system prompt handling
            return f"[System Context]\n{header}\n{context}\n{footer}\n[End Context]\n\n{prompt}"

        elif config.mode == InjectionMode.STRUCTURED:
            return f"""<context>
<header>{header}</header>
<memories>
{context}
</memories>
</context>

<user_query>
{prompt}
</user_query>"""

        return prompt

    def wrap(self, fn: Callable) -> Callable:
        """
        Decorator to automatically enrich prompts.

        Usage:
            @rag.wrap
            async def query_ai(prompt: str) -> str:
                return await ai.complete(prompt)

            # Now prompt is automatically enriched
            result = await query_ai("How do I deploy?")
        """
        async def wrapper(prompt: str, *args, **kwargs):
            # Enrich prompt
            result = await self.enrich(prompt)

            # Log enrichment
            if result.context_added:
                logger.debug(
                    f"RAG: enriched with {result.memories_used} memories "
                    f"({result.search_time_ms:.0f}ms)"
                )

            # Call original function with enriched prompt
            return await fn(result.enriched_prompt, *args, **kwargs)

        return wrapper

    @property
    def stats(self) -> Dict[str, int]:
        """Get RAG statistics"""
        return self._stats.copy()


class RAGMiddleware:
    """
    Middleware for integrating RAG into BBX runtime.

    Automatically enriches AI calls in workflows.
    """

    def __init__(
        self,
        memory=None,
        config: Optional[RAGConfig] = None,
    ):
        self.rag = RAGEnrichment(memory=memory, default_config=config)
        self._enabled = True

    def enable(self):
        """Enable RAG enrichment"""
        self._enabled = True

    def disable(self):
        """Disable RAG enrichment"""
        self._enabled = False

    async def process_step(
        self,
        step_id: str,
        adapter: str,
        method: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a workflow step, enriching AI calls.

        Called by runtime before executing steps.
        """
        if not self._enabled:
            return inputs

        # Only enrich AI-related adapters
        ai_adapters = {'agent', 'ai', 'llm', 'claude', 'openai', 'anthropic'}
        if adapter.lower() not in ai_adapters:
            return inputs

        # Only enrich methods that take prompts
        prompt_methods = {'query', 'complete', 'chat', 'generate', 'ask'}
        if method.lower() not in prompt_methods:
            return inputs

        # Find and enrich prompt
        prompt_keys = ['prompt', 'message', 'query', 'input', 'text']
        for key in prompt_keys:
            if key in inputs and isinstance(inputs[key], str):
                result = await self.rag.enrich(inputs[key])
                if result.context_added:
                    inputs[key] = result.enriched_prompt
                    inputs['_rag_sources'] = result.sources
                    logger.info(
                        f"[RAG] Step {step_id}: enriched with "
                        f"{result.memories_used} memories"
                    )
                break

        return inputs


# Convenience function for CLI/direct use

async def enrich_prompt(
    prompt: str,
    qdrant_url: str = "http://localhost:6333",
    collection: str = "bbx_memories",
    top_k: int = 5,
    min_relevance: float = 0.3,
) -> RAGResult:
    """
    Convenience function to enrich a prompt with memory context.

    Args:
        prompt: The prompt to enrich
        qdrant_url: Qdrant server URL
        collection: Collection name
        top_k: Number of memories to retrieve
        min_relevance: Minimum relevance threshold

    Returns:
        RAGResult with enriched prompt
    """
    try:
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            rag_config = RAGConfig(
                top_k=top_k,
                min_relevance=min_relevance,
            )

            rag = RAGEnrichment(memory=mem, default_config=rag_config)
            return await rag.enrich(prompt)
        finally:
            await mem.stop()

    except ImportError as e:
        logger.warning(f"SemanticMemory not available: {e}")
        return RAGResult(
            original_prompt=prompt,
            enriched_prompt=prompt,
            context_added=False,
            memories_found=0,
            memories_used=0,
        )
