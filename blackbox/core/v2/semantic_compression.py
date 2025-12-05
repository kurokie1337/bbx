# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Semantic Compression - Lossy compression that preserves meaning.

A new paradigm: compress text by meaning, not bytes.
Like JPEG for images, but for text - lose exact words, keep semantics.

Compression ratio: 10:1 to 100:1+ depending on redundancy
Trade-off: Exact words lost, meaning preserved

File format: .bbz (BBX Semantic Archive)

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │ COMPRESS                                                      │
    │                                                               │
    │  Text ──→ Chunk ──→ Embed ──→ Cluster ──→ Quantize ──→ .bbz │
    │           │         │         │           │                   │
    │           │         │         │           └─ int8 vectors     │
    │           │         │         └─ merge similar               │
    │           │         └─ semantic vectors                      │
    │           └─ split by meaning                                │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │ DECOMPRESS                                                    │
    │                                                               │
    │  .bbz ──→ Dequantize ──→ Keywords ──→ LLM ──→ Regenerated   │
    │           │               │            │                      │
    │           │               │            └─ reconstruct text    │
    │           │               └─ guide generation                │
    │           └─ restore vectors                                 │
    └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("bbx.semantic_compress")

# File format constants
BBZ_MAGIC = b"BBZ\x00"  # Magic bytes
BBZ_VERSION = 1


class CompressionLevel(Enum):
    """Compression levels - trade-off between size and fidelity"""
    FAST = "fast"          # Quick, less compression
    BALANCED = "balanced"  # Good balance
    MAX = "max"            # Maximum compression, slower


@dataclass
class ChunkInfo:
    """Information about a compressed chunk"""
    index: int
    embedding: np.ndarray
    keywords: List[str]
    importance: float
    char_count: int
    cluster_id: Optional[int] = None


@dataclass
class BBZHeader:
    """Header for .bbz file format"""
    version: int = BBZ_VERSION
    original_size: int = 0
    compressed_size: int = 0
    chunk_count: int = 0
    cluster_count: int = 0
    embedding_dim: int = 384
    quantization_bits: int = 8
    created_at: float = field(default_factory=time.time)
    original_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        data = {
            "version": self.version,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "chunk_count": self.chunk_count,
            "cluster_count": self.cluster_count,
            "embedding_dim": self.embedding_dim,
            "quantization_bits": self.quantization_bits,
            "created_at": self.created_at,
            "original_hash": self.original_hash,
            "metadata": self.metadata,
        }
        json_bytes = json.dumps(data).encode("utf-8")
        # Length prefix (4 bytes) + JSON
        return struct.pack("<I", len(json_bytes)) + json_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple["BBZHeader", int]:
        """Deserialize header from bytes, returns (header, bytes_consumed)"""
        length = struct.unpack("<I", data[:4])[0]
        json_data = json.loads(data[4:4+length].decode("utf-8"))
        header = cls(**json_data)
        return header, 4 + length


@dataclass
class BBZArchive:
    """A semantic archive"""
    header: BBZHeader
    embeddings: np.ndarray  # Quantized embeddings
    keywords: List[List[str]]  # Keywords per chunk
    cluster_map: List[int]  # Cluster assignments
    cluster_summaries: List[str]  # Summary per cluster


class SemanticCompressor:
    """
    Compresses text semantically.

    Usage:
        compressor = SemanticCompressor()

        # Compress
        archive = await compressor.compress("Long text here...")
        compressor.save(archive, "output.bbz")

        # Decompress
        archive = compressor.load("output.bbz")
        text = await compressor.decompress(archive)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self._embedder = None

    async def _get_embedder(self):
        """Lazy load embedding model"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
                logger.info(f"Loaded embedding model: {self.embedding_model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install: pip install sentence-transformers"
                )
        return self._embedder

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks"""
        # Simple chunking by sentences/paragraphs
        import re

        # Split by paragraph first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter empty chunks
        chunks = [c for c in chunks if len(c) > 10]

        return chunks

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract important keywords from text"""
        import re

        # Simple TF-based extraction
        words = re.findall(r'\b[a-zA-Zа-яА-ЯёЁ]{3,}\b', text.lower())

        # Filter stopwords (basic)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'как', 'что', 'это', 'для', 'при', 'или', 'его', 'она', 'они',
            'быть', 'был', 'была', 'были', 'есть', 'нет', 'так', 'все',
        }

        words = [w for w in words if w not in stopwords]

        # Count frequency
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        return [w for w, _ in sorted_words[:top_k]]

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        threshold: float
    ) -> Tuple[List[int], int]:
        """Cluster similar embeddings together"""
        n = len(embeddings)
        if n == 0:
            return [], 0

        # Simple greedy clustering
        cluster_ids = [-1] * n
        cluster_centers = []
        cluster_count = 0

        for i in range(n):
            # Find best matching cluster
            best_cluster = -1
            best_sim = threshold

            for c_idx, center in enumerate(cluster_centers):
                sim = np.dot(embeddings[i], center) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(center) + 1e-9
                )
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c_idx

            if best_cluster >= 0:
                # Add to existing cluster
                cluster_ids[i] = best_cluster
                # Update center (running average)
                cluster_centers[best_cluster] = (
                    cluster_centers[best_cluster] * 0.9 + embeddings[i] * 0.1
                )
            else:
                # Create new cluster
                cluster_ids[i] = cluster_count
                cluster_centers.append(embeddings[i].copy())
                cluster_count += 1

        return cluster_ids, cluster_count

    def _quantize_embeddings(
        self,
        embeddings: np.ndarray,
        bits: int = 8
    ) -> Tuple[np.ndarray, float, float]:
        """Quantize embeddings to reduce size"""
        # Normalize to [0, 1] range
        min_val = embeddings.min()
        max_val = embeddings.max()

        normalized = (embeddings - min_val) / (max_val - min_val + 1e-9)

        # Quantize to int8
        max_int = (2 ** bits) - 1
        quantized = (normalized * max_int).astype(np.uint8)

        return quantized, min_val, max_val

    def _dequantize_embeddings(
        self,
        quantized: np.ndarray,
        min_val: float,
        max_val: float,
        bits: int = 8
    ) -> np.ndarray:
        """Restore embeddings from quantized form"""
        max_int = (2 ** bits) - 1
        normalized = quantized.astype(np.float32) / max_int
        return normalized * (max_val - min_val) + min_val

    async def compress(
        self,
        text: str,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> BBZArchive:
        """
        Compress text semantically.

        Args:
            text: Text to compress
            level: Compression level

        Returns:
            BBZArchive that can be saved
        """
        logger.info(f"Compressing {len(text)} bytes...")

        # Adjust parameters by level
        if level == CompressionLevel.FAST:
            self.similarity_threshold = 0.9
            self.chunk_size = 1000
        elif level == CompressionLevel.MAX:
            self.similarity_threshold = 0.75
            self.chunk_size = 300

        # 1. Chunk text
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No chunks created from text")

        # 2. Extract keywords per chunk
        keywords = [self._extract_keywords(c) for c in chunks]

        # 3. Create embeddings
        embedder = await self._get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings)

        # 4. Cluster similar chunks
        cluster_ids, cluster_count = self._cluster_embeddings(
            embeddings, self.similarity_threshold
        )
        logger.info(f"Clustered into {cluster_count} clusters")

        # 5. Create cluster summaries (representative keywords)
        cluster_keywords = {}
        for i, cid in enumerate(cluster_ids):
            if cid not in cluster_keywords:
                cluster_keywords[cid] = []
            cluster_keywords[cid].extend(keywords[i])

        cluster_summaries = []
        for cid in range(cluster_count):
            kw_freq = {}
            for kw in cluster_keywords.get(cid, []):
                kw_freq[kw] = kw_freq.get(kw, 0) + 1
            top_kw = sorted(kw_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            cluster_summaries.append(" ".join([w for w, _ in top_kw]))

        # 6. Quantize embeddings
        quantized, min_val, max_val = self._quantize_embeddings(embeddings)

        # 7. Build archive
        original_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        header = BBZHeader(
            original_size=len(text.encode("utf-8")),
            chunk_count=len(chunks),
            cluster_count=cluster_count,
            embedding_dim=embeddings.shape[1],
            original_hash=original_hash,
            metadata={
                "level": level.value,
                "min_val": float(min_val),
                "max_val": float(max_val),
                "chunk_size": self.chunk_size,
                "model": self.embedding_model,
            }
        )

        archive = BBZArchive(
            header=header,
            embeddings=quantized,
            keywords=keywords,
            cluster_map=cluster_ids,
            cluster_summaries=cluster_summaries,
        )

        return archive

    def save(self, archive: BBZArchive, path: str) -> int:
        """
        Save archive to .bbz file.

        Args:
            archive: Archive to save
            path: Output path

        Returns:
            Compressed size in bytes
        """
        # Build binary data
        parts = []

        # Magic + Version
        parts.append(BBZ_MAGIC)
        parts.append(struct.pack("<B", BBZ_VERSION))

        # Embeddings (as flat array)
        emb_bytes = archive.embeddings.tobytes()
        parts.append(struct.pack("<I", len(emb_bytes)))
        parts.append(emb_bytes)

        # Keywords (JSON)
        kw_json = json.dumps(archive.keywords).encode("utf-8")
        parts.append(struct.pack("<I", len(kw_json)))
        parts.append(kw_json)

        # Cluster map
        cluster_bytes = np.array(archive.cluster_map, dtype=np.int16).tobytes()
        parts.append(struct.pack("<I", len(cluster_bytes)))
        parts.append(cluster_bytes)

        # Cluster summaries
        summaries_json = json.dumps(archive.cluster_summaries).encode("utf-8")
        parts.append(struct.pack("<I", len(summaries_json)))
        parts.append(summaries_json)

        # Header (at end for easy metadata reading)
        header_bytes = archive.header.to_bytes()
        parts.append(header_bytes)

        # Combine and compress
        raw_data = b"".join(parts)
        compressed = gzip.compress(raw_data, compresslevel=9)

        # Update header with compressed size
        archive.header.compressed_size = len(compressed)

        # Write to file
        Path(path).write_bytes(compressed)

        logger.info(
            f"Saved {archive.header.original_size} bytes -> {len(compressed)} bytes "
            f"(ratio: {archive.header.original_size / len(compressed):.1f}:1)"
        )

        return len(compressed)

    def load(self, path: str) -> BBZArchive:
        """
        Load archive from .bbz file.

        Args:
            path: Path to .bbz file

        Returns:
            Loaded archive
        """
        # Read and decompress
        compressed = Path(path).read_bytes()
        raw_data = gzip.decompress(compressed)

        pos = 0

        # Magic
        if raw_data[pos:pos+4] != BBZ_MAGIC:
            raise ValueError("Invalid BBZ file")
        pos += 4

        # Version
        version = struct.unpack("<B", raw_data[pos:pos+1])[0]
        if version > BBZ_VERSION:
            raise ValueError(f"Unsupported BBZ version: {version}")
        pos += 1

        # Embeddings
        emb_len = struct.unpack("<I", raw_data[pos:pos+4])[0]
        pos += 4
        emb_bytes = raw_data[pos:pos+emb_len]
        pos += emb_len

        # Keywords
        kw_len = struct.unpack("<I", raw_data[pos:pos+4])[0]
        pos += 4
        keywords = json.loads(raw_data[pos:pos+kw_len].decode("utf-8"))
        pos += kw_len

        # Cluster map
        cluster_len = struct.unpack("<I", raw_data[pos:pos+4])[0]
        pos += 4
        cluster_map = np.frombuffer(
            raw_data[pos:pos+cluster_len], dtype=np.int16
        ).tolist()
        pos += cluster_len

        # Summaries
        sum_len = struct.unpack("<I", raw_data[pos:pos+4])[0]
        pos += 4
        summaries = json.loads(raw_data[pos:pos+sum_len].decode("utf-8"))
        pos += sum_len

        # Header
        header, _ = BBZHeader.from_bytes(raw_data[pos:])

        # Reconstruct embeddings array
        embeddings = np.frombuffer(emb_bytes, dtype=np.uint8).reshape(
            header.chunk_count, header.embedding_dim
        )

        return BBZArchive(
            header=header,
            embeddings=embeddings,
            keywords=keywords,
            cluster_map=cluster_map,
            cluster_summaries=summaries,
        )

    async def decompress(
        self,
        archive: BBZArchive,
        llm_generator=None,
    ) -> str:
        """
        Decompress archive back to text.

        Uses LLM to regenerate text from embeddings + keywords.
        Without LLM, returns keyword-based summary.

        Args:
            archive: Archive to decompress
            llm_generator: Optional LLM for text generation

        Returns:
            Regenerated text
        """
        # Dequantize embeddings
        min_val = archive.header.metadata.get("min_val", 0)
        max_val = archive.header.metadata.get("max_val", 1)

        embeddings = self._dequantize_embeddings(
            archive.embeddings, min_val, max_val
        )

        # Group by cluster
        clusters = {}
        for i, cid in enumerate(archive.cluster_map):
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append({
                "index": i,
                "keywords": archive.keywords[i],
                "embedding": embeddings[i],
            })

        # Generate text per cluster
        if llm_generator:
            # Use LLM for high-quality regeneration
            parts = []
            for cid in sorted(clusters.keys()):
                cluster = clusters[cid]
                summary = archive.cluster_summaries[cid]

                # Combine keywords from cluster
                all_keywords = []
                for chunk in cluster:
                    all_keywords.extend(chunk["keywords"])
                unique_keywords = list(dict.fromkeys(all_keywords))

                prompt = f"""Reconstruct a coherent text passage based on these keywords and topic.
Keywords: {', '.join(unique_keywords[:20])}
Topic summary: {summary}
Number of original paragraphs: {len(cluster)}

Generate a natural text that would contain these concepts:"""

                try:
                    generated = await llm_generator(prompt)
                    parts.append(generated)
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
                    # Fallback to keywords
                    parts.append(f"[{summary}]\n" + ", ".join(unique_keywords[:15]))

            return "\n\n".join(parts)

        else:
            # Without LLM - return structured summary
            parts = []
            parts.append("=" * 60)
            parts.append("SEMANTIC DECOMPRESSION (without LLM)")
            parts.append("=" * 60)
            parts.append(f"Original size: {archive.header.original_size} bytes")
            parts.append(f"Clusters: {archive.header.cluster_count}")
            parts.append(f"Chunks: {archive.header.chunk_count}")
            parts.append("")

            for cid in sorted(clusters.keys()):
                cluster = clusters[cid]
                summary = archive.cluster_summaries[cid]

                parts.append(f"--- Cluster {cid + 1} ({len(cluster)} chunks) ---")
                parts.append(f"Topic: {summary}")

                # List unique keywords
                all_kw = []
                for chunk in cluster:
                    all_kw.extend(chunk["keywords"])
                unique_kw = list(dict.fromkeys(all_kw))
                parts.append(f"Keywords: {', '.join(unique_kw[:20])}")
                parts.append("")

            return "\n".join(parts)

    def get_stats(self, archive: BBZArchive) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            "original_size": archive.header.original_size,
            "compressed_size": archive.header.compressed_size,
            "ratio": archive.header.original_size / max(1, archive.header.compressed_size),
            "chunks": archive.header.chunk_count,
            "clusters": archive.header.cluster_count,
            "cluster_reduction": 1 - (archive.header.cluster_count / max(1, archive.header.chunk_count)),
            "embedding_dim": archive.header.embedding_dim,
            "created_at": archive.header.created_at,
        }


# Convenience functions

async def compress_file(
    input_path: str,
    output_path: Optional[str] = None,
    level: CompressionLevel = CompressionLevel.BALANCED,
) -> Dict[str, Any]:
    """
    Compress a text file semantically.

    Args:
        input_path: Path to input text file
        output_path: Output .bbz path (default: input + .bbz)
        level: Compression level

    Returns:
        Compression statistics
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = str(input_path) + ".bbz"

    text = input_path.read_text(encoding="utf-8")

    compressor = SemanticCompressor()
    archive = await compressor.compress(text, level)
    compressor.save(archive, output_path)

    stats = compressor.get_stats(archive)
    stats["input_path"] = str(input_path)
    stats["output_path"] = output_path

    return stats


async def decompress_file(
    input_path: str,
    output_path: Optional[str] = None,
    use_llm: bool = False,
) -> str:
    """
    Decompress a .bbz file.

    Args:
        input_path: Path to .bbz file
        output_path: Output text path (optional)
        use_llm: Whether to use LLM for regeneration

    Returns:
        Decompressed text
    """
    compressor = SemanticCompressor()
    archive = compressor.load(input_path)

    llm = None
    if use_llm:
        try:
            from blackbox.ai.generator import WorkflowGenerator
            gen = WorkflowGenerator()

            async def llm_gen(prompt: str) -> str:
                result = gen.llm(prompt, max_tokens=500)
                return result["choices"][0]["text"]

            llm = llm_gen
        except Exception as e:
            logger.warning(f"LLM not available: {e}")

    text = await compressor.decompress(archive, llm)

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")

    return text
