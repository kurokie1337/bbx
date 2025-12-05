# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Conversation Ingester - Auto-ingest AI conversations into memory.

Supports:
- Claude Code conversations (.json)
- ChatGPT exports (.json)
- Plain text conversations
- Markdown chat logs

Features:
- Smart chunking by turns or semantic segments
- Speaker extraction (human/assistant)
- Topic/intent detection
- Code block extraction and tagging
- Decision and action tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bbx.conversation_ingest")


class ConversationFormat(Enum):
    """Supported conversation formats"""
    CLAUDE_CODE = "claude_code"    # Claude Code JSON format
    CHATGPT = "chatgpt"            # ChatGPT export
    MARKDOWN = "markdown"          # Markdown chat logs
    PLAINTEXT = "plaintext"        # Plain text
    AUTO = "auto"                  # Auto-detect


@dataclass
class ConversationTurn:
    """A single turn in conversation"""
    role: str  # 'human', 'assistant', 'system'
    content: str
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationChunk:
    """A chunk of conversation for memory storage"""
    id: str
    content: str
    turns: List[ConversationTurn]
    summary: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    code_blocks: List[Dict[str, str]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationParser:
    """Parses various conversation formats"""

    @staticmethod
    def detect_format(file_path: str) -> ConversationFormat:
        """Auto-detect conversation format"""
        path = Path(file_path)

        if path.suffix == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Claude Code format detection
                if isinstance(data, list) and len(data) > 0:
                    if 'role' in data[0] and 'content' in data[0]:
                        return ConversationFormat.CLAUDE_CODE

                # ChatGPT format detection
                if isinstance(data, dict):
                    if 'mapping' in data or 'messages' in data:
                        return ConversationFormat.CHATGPT

            except (json.JSONDecodeError, KeyError):
                pass

        if path.suffix == '.md':
            return ConversationFormat.MARKDOWN

        return ConversationFormat.PLAINTEXT

    @staticmethod
    def parse_claude_code(file_path: str) -> List[ConversationTurn]:
        """Parse Claude Code conversation JSON"""
        turns = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        for msg in data:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Handle content as list (Claude format)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                        elif part.get('type') == 'tool_use':
                            text_parts.append(f"[Tool: {part.get('name', 'unknown')}]")
                        elif part.get('type') == 'tool_result':
                            text_parts.append(f"[Tool Result]")
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = '\n'.join(text_parts)

            if content.strip():
                turns.append(ConversationTurn(
                    role='human' if role == 'user' else role,
                    content=content,
                    timestamp=msg.get('timestamp'),
                    metadata=msg.get('metadata', {})
                ))

        return turns

    @staticmethod
    def parse_chatgpt(file_path: str) -> List[ConversationTurn]:
        """Parse ChatGPT export JSON"""
        turns = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle conversations.json format
        if isinstance(data, list):
            for conv in data:
                if 'mapping' in conv:
                    data = conv
                    break

        # Extract from mapping structure
        if 'mapping' in data:
            for node_id, node in data['mapping'].items():
                msg = node.get('message')
                if msg and msg.get('content', {}).get('parts'):
                    role = msg.get('author', {}).get('role', 'unknown')
                    content = '\n'.join(msg['content']['parts'])

                    if content.strip():
                        turns.append(ConversationTurn(
                            role='human' if role == 'user' else role,
                            content=content,
                            timestamp=msg.get('create_time'),
                        ))

        return turns

    @staticmethod
    def parse_markdown(file_path: str) -> List[ConversationTurn]:
        """Parse markdown chat logs"""
        turns = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Common patterns: **Human:**, **Assistant:**, ## Human, ## Assistant
        patterns = [
            r'\*\*(Human|User|You):\*\*\s*(.*?)(?=\*\*(?:Human|User|You|Assistant|Claude|AI):\*\*|$)',
            r'\*\*(Assistant|Claude|AI):\*\*\s*(.*?)(?=\*\*(?:Human|User|You|Assistant|Claude|AI):\*\*|$)',
            r'^##\s*(Human|User|You)\s*\n(.*?)(?=^##\s*(?:Human|User|You|Assistant|Claude|AI)|$)',
            r'^##\s*(Assistant|Claude|AI)\s*\n(.*?)(?=^##\s*(?:Human|User|You|Assistant|Claude|AI)|$)',
        ]

        # Simple fallback: split by common separators
        parts = re.split(r'\n---\n|\n\*\*\*\n', content)

        for i, part in enumerate(parts):
            if part.strip():
                # Guess role based on position (even=human, odd=assistant)
                role = 'human' if i % 2 == 0 else 'assistant'
                turns.append(ConversationTurn(
                    role=role,
                    content=part.strip()
                ))

        return turns

    @staticmethod
    def parse_plaintext(file_path: str) -> List[ConversationTurn]:
        """Parse plain text (treat as single content block)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return [ConversationTurn(
            role='document',
            content=content
        )]


class ConversationChunker:
    """Chunks conversations into memory-sized pieces"""

    def __init__(
        self,
        max_chunk_turns: int = 6,
        max_chunk_chars: int = 4000,
        overlap_turns: int = 1
    ):
        self.max_chunk_turns = max_chunk_turns
        self.max_chunk_chars = max_chunk_chars
        self.overlap_turns = overlap_turns

    def chunk_conversation(
        self,
        turns: List[ConversationTurn],
        source_file: Optional[str] = None
    ) -> List[ConversationChunk]:
        """Split conversation into chunks"""
        if not turns:
            return []

        chunks = []
        current_turns = []
        current_chars = 0

        for turn in turns:
            turn_chars = len(turn.content)

            # Check if we need to create a new chunk
            if (len(current_turns) >= self.max_chunk_turns or
                current_chars + turn_chars > self.max_chunk_chars):

                if current_turns:
                    chunk = self._create_chunk(current_turns, source_file)
                    chunks.append(chunk)

                    # Keep overlap turns for context continuity
                    current_turns = current_turns[-self.overlap_turns:] if self.overlap_turns > 0 else []
                    current_chars = sum(len(t.content) for t in current_turns)

            current_turns.append(turn)
            current_chars += turn_chars

        # Don't forget the last chunk
        if current_turns:
            chunk = self._create_chunk(current_turns, source_file)
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        turns: List[ConversationTurn],
        source_file: Optional[str]
    ) -> ConversationChunk:
        """Create a chunk from turns"""
        # Build content string
        content_parts = []
        for turn in turns:
            role_label = turn.role.upper()
            content_parts.append(f"[{role_label}]: {turn.content}")

        content = "\n\n".join(content_parts)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        # Extract topics (simple keyword extraction)
        topics = self._extract_topics(content)

        # Extract decisions/actions
        decisions = self._extract_decisions(content)

        # Generate ID from content hash
        chunk_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Get timestamp from first turn with timestamp
        timestamp = time.time()
        for turn in turns:
            if turn.timestamp:
                timestamp = turn.timestamp
                break

        return ConversationChunk(
            id=chunk_id,
            content=content,
            turns=turns,
            topics=topics,
            decisions=decisions,
            code_blocks=code_blocks,
            timestamp=timestamp,
            source_file=source_file
        )

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from content"""
        blocks = []
        pattern = r'```(\w+)?\n(.*?)```'

        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            blocks.append({
                'language': language,
                'code': code[:500]  # Truncate for storage
            })

        return blocks

    def _extract_topics(self, content: str) -> List[str]:
        """Simple topic extraction"""
        topics = []

        # Common programming topics
        topic_patterns = {
            'python': r'\b(python|pip|pytest|django|flask)\b',
            'javascript': r'\b(javascript|nodejs|npm|react|vue|typescript)\b',
            'rust': r'\b(rust|cargo|rustc)\b',
            'database': r'\b(database|sql|postgres|mysql|mongodb|qdrant)\b',
            'api': r'\b(api|rest|graphql|endpoint|http)\b',
            'docker': r'\b(docker|container|kubernetes|k8s)\b',
            'git': r'\b(git|commit|branch|merge|push|pull)\b',
            'testing': r'\b(test|testing|unittest|pytest|jest)\b',
            'ai': r'\b(ai|ml|llm|gpt|claude|embedding|vector)\b',
            'workflow': r'\b(workflow|automation|pipeline|cicd)\b',
        }

        content_lower = content.lower()
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, content_lower):
                topics.append(topic)

        return topics[:5]  # Limit to top 5 topics

    def _extract_decisions(self, content: str) -> List[str]:
        """Extract decisions and action items"""
        decisions = []

        # Look for decision patterns
        patterns = [
            r"(?:let's|we'll|I'll|should|will)\s+(.{20,100}?)(?:\.|$)",
            r"(?:decided to|going to|plan to)\s+(.{20,100}?)(?:\.|$)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                decision = match.group(1).strip()
                if len(decision) > 20:
                    decisions.append(decision[:100])

        return decisions[:3]  # Limit to top 3


class ConversationIngester:
    """Main class for ingesting conversations into BBX memory"""

    def __init__(self, memory_instance=None):
        """
        Args:
            memory_instance: SemanticMemory instance (optional, creates default if None)
        """
        self.memory = memory_instance
        self.parser = ConversationParser()
        self.chunker = ConversationChunker()
        self._stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'turns_processed': 0,
        }

    async def ingest_file(
        self,
        file_path: str,
        format: ConversationFormat = ConversationFormat.AUTO,
        agent_id: str = "default",
        importance: float = 0.6,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        Ingest a conversation file into memory.

        Args:
            file_path: Path to conversation file
            format: Conversation format (AUTO for auto-detect)
            agent_id: Agent ID for memory storage
            importance: Base importance score
            tags: Additional tags

        Returns:
            List of created memory IDs
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format
        if format == ConversationFormat.AUTO:
            format = self.parser.detect_format(file_path)

        logger.info(f"Ingesting {file_path} as {format.value}")

        # Parse conversation
        if format == ConversationFormat.CLAUDE_CODE:
            turns = self.parser.parse_claude_code(file_path)
        elif format == ConversationFormat.CHATGPT:
            turns = self.parser.parse_chatgpt(file_path)
        elif format == ConversationFormat.MARKDOWN:
            turns = self.parser.parse_markdown(file_path)
        else:
            turns = self.parser.parse_plaintext(file_path)

        if not turns:
            logger.warning(f"No turns extracted from {file_path}")
            return []

        # Chunk conversation
        chunks = self.chunker.chunk_conversation(turns, source_file=file_path)

        # Store in memory
        memory_ids = []
        base_tags = tags or []
        base_tags.extend(['conversation', format.value])

        for chunk in chunks:
            chunk_tags = list(set(base_tags + chunk.topics))

            # Build metadata
            metadata = {
                'source_file': file_path,
                'source_format': format.value,
                'chunk_id': chunk.id,
                'turn_count': len(chunk.turns),
                'topics': chunk.topics,
                'decisions': chunk.decisions,
                'has_code': len(chunk.code_blocks) > 0,
                'code_languages': [b['language'] for b in chunk.code_blocks],
            }

            if self.memory:
                # Store in SemanticMemory
                memory_id = await self.memory.store(
                    agent_id=agent_id,
                    content=chunk.content,
                    memory_type="episodic",
                    importance=importance,
                    tags=chunk_tags,
                    metadata=metadata
                )
                memory_ids.append(memory_id)
            else:
                # Just return chunk IDs without storage
                memory_ids.append(chunk.id)

        # Update stats
        self._stats['files_processed'] += 1
        self._stats['chunks_created'] += len(chunks)
        self._stats['turns_processed'] += len(turns)

        logger.info(f"Created {len(memory_ids)} memories from {len(turns)} turns")
        return memory_ids

    async def ingest_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        recursive: bool = True,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Ingest all conversation files from directory.

        Args:
            directory: Directory path
            pattern: File pattern (glob)
            recursive: Include subdirectories
            **kwargs: Passed to ingest_file

        Returns:
            Dict mapping file paths to memory IDs
        """
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        results = {}
        for file in files:
            try:
                memory_ids = await self.ingest_file(str(file), **kwargs)
                results[str(file)] = memory_ids
            except Exception as e:
                logger.error(f"Failed to ingest {file}: {e}")
                results[str(file)] = []

        return results

    async def watch_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        interval: float = 30.0,
        **kwargs
    ):
        """
        Watch directory for new conversation files (async generator).

        Args:
            directory: Directory to watch
            pattern: File pattern
            interval: Check interval in seconds
            **kwargs: Passed to ingest_file

        Yields:
            Tuple of (file_path, memory_ids) for each new file
        """
        path = Path(directory)
        processed_files = set()

        # Initial scan
        for file in path.rglob(pattern):
            processed_files.add(str(file))

        while True:
            await asyncio.sleep(interval)

            # Check for new files
            for file in path.rglob(pattern):
                file_str = str(file)
                if file_str not in processed_files:
                    processed_files.add(file_str)

                    try:
                        memory_ids = await self.ingest_file(file_str, **kwargs)
                        yield (file_str, memory_ids)
                    except Exception as e:
                        logger.error(f"Failed to ingest new file {file}: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        """Get ingestion statistics"""
        return self._stats.copy()


# Convenience functions for CLI

async def ingest_conversation(
    file_path: str,
    qdrant_url: str = "http://localhost:6333",
    collection: str = "bbx_memories",
    **kwargs
) -> List[str]:
    """
    Convenience function to ingest a conversation file.

    Args:
        file_path: Path to conversation file
        qdrant_url: Qdrant server URL
        collection: Collection name
        **kwargs: Passed to ingest_file

    Returns:
        List of created memory IDs
    """
    # Lazy import to avoid dependency issues
    try:
        from blackbox.core.v2.semantic_memory import SemanticMemory, QdrantStore

        store = QdrantStore(url=qdrant_url, collection_name=collection)
        memory = SemanticMemory(store=store)
        ingester = ConversationIngester(memory_instance=memory)

        return await ingester.ingest_file(file_path, **kwargs)

    except ImportError as e:
        logger.warning(f"SemanticMemory not available: {e}")
        # Fallback: just parse and return chunk IDs
        ingester = ConversationIngester(memory_instance=None)
        return await ingester.ingest_file(file_path, **kwargs)


def sync_ingest_conversation(file_path: str, **kwargs) -> List[str]:
    """Synchronous wrapper for ingest_conversation"""
    return asyncio.run(ingest_conversation(file_path, **kwargs))
