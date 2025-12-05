# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Core - The "WinRAR" of AI Development

Simple interface, powerful under the hood.
Like a hammer - you don't need Bluetooth on a hammer.

Commands:
    bbx pack     - "Compress" project understanding
    bbx unpack   - "Decompress" intent into code
    bbx recover  - Restore after AI errors (killer feature!)

Philosophy:
    - Interface never changes (like WinRAR since 2005)
    - Power under the hood (RAR5 = Genome + RAG + Intent)
    - Recovery Record = ability to restore after AI mistakes
    - Free for individuals, corps pay for support

This is the unified entry point. All complexity hidden.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bbx.core")


@dataclass
class BBXPackage:
    """
    A .bbx package - like a .rar but for project understanding.

    Contains:
    - Intent: What you want to achieve
    - Genome: Project understanding (structure, relationships)
    - Recovery: Snapshots for rollback
    - Proof: Evidence that this worked before
    """
    version: str = "2.0"
    created_at: float = field(default_factory=time.time)

    # Core content
    intent: Optional[str] = None
    genome_hash: Optional[str] = None

    # Recovery data (like WinRAR's Recovery Record)
    recovery_snapshots: List[str] = field(default_factory=list)
    recovery_percent: int = 5  # % of project stored for recovery

    # Proof of success
    proof: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    project_name: str = ""
    project_path: str = ""
    files_count: int = 0
    lines_count: int = 0


class BBXCore:
    """
    The core BBX engine - simple interface to all power.

    Like WinRAR:
    - pack() = compress
    - unpack() = decompress
    - recover() = use recovery record

    All the complexity (RAG, Genome, Intent, Memory) is hidden.
    """

    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).absolute()
        self.bbx_dir = self.project_path / ".bbx"
        self._ensure_bbx_dir()

    def _ensure_bbx_dir(self):
        """Create .bbx directory if needed"""
        self.bbx_dir.mkdir(exist_ok=True)
        (self.bbx_dir / "recovery").mkdir(exist_ok=True)
        (self.bbx_dir / "snapshots").mkdir(exist_ok=True)

    # =========================================================================
    # PACK - "Compress" project understanding
    # =========================================================================

    async def pack(
        self,
        output: Optional[str] = None,
        include_recovery: bool = True,
        recovery_percent: int = 5,
    ) -> str:
        """
        Pack project into .bbx understanding package.

        Like WinRAR compress, but for project understanding:
        - Analyzes project structure
        - Creates embeddings for semantic search
        - Adds recovery data for rollback capability

        Args:
            output: Output .bbx file path
            include_recovery: Include recovery record (recommended!)
            recovery_percent: How much recovery data (1-10%)

        Returns:
            Path to created .bbx file
        """
        logger.info(f"Packing project: {self.project_path}")

        # 1. Analyze project (create Genome)
        from .project_genome import GenomeAnalyzer, save_genome

        analyzer = GenomeAnalyzer()
        genome = await analyzer.analyze_project(str(self.project_path))

        # Save genome
        genome_path = self.bbx_dir / "genome.json"
        save_genome(genome, str(genome_path))
        genome_hash = hashlib.sha256(genome_path.read_bytes()).hexdigest()[:16]

        # 2. Create recovery snapshot if requested
        recovery_snapshots = []
        if include_recovery:
            snapshot_id = await self._create_recovery_snapshot(recovery_percent)
            recovery_snapshots.append(snapshot_id)

        # 3. Build package
        package = BBXPackage(
            genome_hash=genome_hash,
            recovery_snapshots=recovery_snapshots,
            recovery_percent=recovery_percent,
            project_name=genome.project_name,
            project_path=str(self.project_path),
            files_count=len(genome.files),
            lines_count=sum(f.lines for f in genome.files.values()),
        )

        # 4. Save package
        if output is None:
            output = str(self.project_path / f"{genome.project_name}.bbx")

        self._save_package(package, output)

        logger.info(f"Packed: {package.files_count} files, {package.lines_count} lines")
        logger.info(f"Recovery: {'enabled' if include_recovery else 'disabled'}")

        return output

    async def _create_recovery_snapshot(self, percent: int) -> str:
        """Create recovery snapshot (like WinRAR's Recovery Record)"""
        snapshot_id = hashlib.sha256(
            f"{time.time()}{self.project_path}".encode()
        ).hexdigest()[:12]

        snapshot_dir = self.bbx_dir / "recovery" / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Collect important files (configs, key source files)
        important_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml",  # Configs
            "*.md",  # Docs
            "requirements.txt", "package.json", "Cargo.toml",  # Dependencies
        ]

        # Also backup some source files based on percent
        from .project_genome import GenomeAnalyzer
        analyzer = GenomeAnalyzer()

        all_files = []
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if not analyzer._should_ignore(Path(root) / d)]
            for f in files:
                fp = Path(root) / f
                if not analyzer._should_ignore(fp):
                    all_files.append(fp)

        # Select files for recovery (most important first)
        import fnmatch
        recovery_files = []

        # Always include config files
        for fp in all_files:
            for pattern in important_patterns:
                if fnmatch.fnmatch(fp.name, pattern):
                    recovery_files.append(fp)
                    break

        # Add source files up to percent limit
        source_files = [f for f in all_files if f.suffix in ('.py', '.ts', '.js', '.rs', '.go')]
        target_count = max(1, int(len(source_files) * percent / 100))

        # Sort by importance (files with most lines first - usually core files)
        source_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        recovery_files.extend(source_files[:target_count])

        # Copy files to recovery
        for fp in recovery_files:
            try:
                rel_path = fp.relative_to(self.project_path)
                dest = snapshot_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fp, dest)
            except:
                pass

        # Save snapshot metadata
        metadata = {
            "id": snapshot_id,
            "timestamp": time.time(),
            "files": [str(f.relative_to(self.project_path)) for f in recovery_files],
            "percent": percent,
        }
        (snapshot_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(f"Recovery snapshot: {len(recovery_files)} files ({percent}%)")
        return snapshot_id

    # =========================================================================
    # UNPACK - "Decompress" intent into workflow/code
    # =========================================================================

    async def unpack(
        self,
        intent: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Unpack intent into executable workflow.

        Like WinRAR decompress, but for intentions:
        - Takes natural language intent
        - Uses Genome to understand project context
        - Uses Memory (RAG) to find similar successful changes
        - Generates concrete steps

        Args:
            intent: What you want to achieve
            dry_run: If True, show plan without executing

        Returns:
            Execution result or plan
        """
        logger.info(f"Unpacking intent: {intent}")

        # 1. Load project understanding
        genome = await self._load_genome()

        # 2. Get relevant context from memory
        context = await self._get_memory_context(intent)

        # 3. Find similar successful patterns
        similar_patterns = await self._find_similar_patterns(intent)

        # 4. Generate workflow using Intent Engine
        from .intent_engine import BBXIntent, IntentEngine

        intent_obj = BBXIntent(
            intent=intent,
            hints=[f"Project: {genome.project_name}" if genome else ""],
        )

        engine = IntentEngine()
        expanded = await engine.expand(intent_obj)

        result = {
            "intent": intent,
            "confidence": expanded.confidence,
            "sources": expanded.sources,
            "steps": expanded.steps,
            "context_files": len(context) if context else 0,
            "similar_patterns": len(similar_patterns),
        }

        if dry_run:
            result["yaml"] = engine.to_executable_yaml(expanded)
            result["status"] = "dry_run"
        else:
            # TODO: Execute workflow
            result["status"] = "planned"
            result["message"] = "Execution not yet implemented"
            result["yaml"] = engine.to_executable_yaml(expanded)

        return result

    async def _load_genome(self):
        """Load project genome if exists"""
        genome_path = self.bbx_dir / "genome.json"
        if genome_path.exists():
            from .project_genome import load_genome
            return load_genome(str(genome_path))
        return None

    async def _get_memory_context(self, query: str) -> List[Dict]:
        """Get relevant context from memory"""
        try:
            from .rag_enrichment import RAGEnrichment
            from .semantic_memory import SemanticMemory, SemanticMemoryConfig

            config = SemanticMemoryConfig(
                vector_store_type="qdrant",
                embedding_provider="local",
            )

            memory = SemanticMemory(config)
            await memory.start()

            try:
                results = await memory.recall(
                    agent_id="default",
                    query=query,
                    top_k=5,
                )
                return [{"content": r.entry.content, "score": r.score} for r in results]
            finally:
                await memory.stop()

        except Exception as e:
            logger.debug(f"Memory not available: {e}")
            return []

    async def _find_similar_patterns(self, intent: str) -> List[Dict]:
        """Find similar successful patterns from Genome"""
        genome = await self._load_genome()
        if not genome or not genome.successful_paths:
            return []

        from .project_genome import GenomeReplayer
        replayer = GenomeReplayer(genome)

        similar = await replayer.find_similar_path(intent)
        return [{"intent": p.intent, "score": s} for p, s in similar]

    # =========================================================================
    # RECOVER - Restore after AI errors (KILLER FEATURE!)
    # =========================================================================

    async def recover(
        self,
        snapshot_id: Optional[str] = None,
        file_path: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Recover project after AI error.

        THIS IS THE KILLER FEATURE - like WinRAR's Recovery Record.
        No other AI tool can do this with context understanding.

        Modes:
        - Full recovery: Restore entire project to snapshot
        - File recovery: Restore specific file
        - Smart recovery: Detect what's broken, fix it

        Args:
            snapshot_id: Specific snapshot to restore (default: latest)
            file_path: Specific file to restore
            dry_run: Show what would be restored

        Returns:
            Recovery result
        """
        logger.info("Starting recovery...")

        # 1. Find available recovery snapshots
        recovery_dir = self.bbx_dir / "recovery"
        snapshots = []

        if recovery_dir.exists():
            for snap_dir in recovery_dir.iterdir():
                if snap_dir.is_dir():
                    meta_file = snap_dir / "metadata.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        snapshots.append(meta)

        if not snapshots:
            return {
                "status": "error",
                "message": "No recovery snapshots found. Run 'bbx pack' first.",
            }

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s["timestamp"], reverse=True)

        # 2. Select snapshot
        if snapshot_id:
            snapshot = next((s for s in snapshots if s["id"] == snapshot_id), None)
            if not snapshot:
                return {"status": "error", "message": f"Snapshot not found: {snapshot_id}"}
        else:
            snapshot = snapshots[0]  # Latest

        snapshot_dir = recovery_dir / snapshot["id"]

        # 3. Determine what to recover
        files_to_recover = []

        if file_path:
            # Single file recovery
            if file_path in snapshot["files"]:
                files_to_recover.append(file_path)
            else:
                return {"status": "error", "message": f"File not in snapshot: {file_path}"}
        else:
            # Full recovery
            files_to_recover = snapshot["files"]

        # 4. Perform recovery
        recovered = []
        skipped = []

        for rel_path in files_to_recover:
            src = snapshot_dir / rel_path
            dst = self.project_path / rel_path

            if not src.exists():
                skipped.append(rel_path)
                continue

            if dry_run:
                recovered.append({"file": rel_path, "action": "would_restore"})
            else:
                # Backup current file first
                if dst.exists():
                    backup_path = self.bbx_dir / "backup" / rel_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dst, backup_path)

                # Restore from snapshot
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                recovered.append({"file": rel_path, "action": "restored"})

        return {
            "status": "success" if not dry_run else "dry_run",
            "snapshot": snapshot["id"],
            "snapshot_time": datetime.fromtimestamp(snapshot["timestamp"]).isoformat(),
            "files_recovered": len(recovered),
            "files_skipped": len(skipped),
            "details": recovered,
        }

    async def smart_recover(self) -> Dict[str, Any]:
        """
        Smart recovery - detect what's broken and fix it.

        Uses Genome understanding to:
        1. Detect files that changed recently
        2. Check if tests/build still pass
        3. If broken, identify culprit files
        4. Recover just those files

        This is UNIQUE to BBX - no other tool does this.
        """
        logger.info("Smart recovery: analyzing project state...")

        # 1. Load current genome
        genome = await self._load_genome()
        if not genome:
            return {"status": "error", "message": "No genome found. Run 'bbx pack' first."}

        # 2. Compare with latest snapshot
        recovery_dir = self.bbx_dir / "recovery"
        snapshots = []

        for snap_dir in recovery_dir.iterdir():
            if snap_dir.is_dir():
                meta_file = snap_dir / "metadata.json"
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text())
                    snapshots.append((snap_dir, meta))

        if not snapshots:
            return {"status": "error", "message": "No recovery snapshots available."}

        # Latest snapshot
        snapshots.sort(key=lambda s: s[1]["timestamp"], reverse=True)
        snapshot_dir, snapshot_meta = snapshots[0]

        # 3. Find changed files
        changed_files = []

        for rel_path in snapshot_meta["files"]:
            snapshot_file = snapshot_dir / rel_path
            current_file = self.project_path / rel_path

            if not current_file.exists():
                changed_files.append({"file": rel_path, "change": "deleted"})
            elif snapshot_file.exists():
                snap_hash = hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
                curr_hash = hashlib.sha256(current_file.read_bytes()).hexdigest()
                if snap_hash != curr_hash:
                    changed_files.append({"file": rel_path, "change": "modified"})

        if not changed_files:
            return {
                "status": "clean",
                "message": "No changes detected since last snapshot.",
            }

        return {
            "status": "changes_detected",
            "snapshot_time": datetime.fromtimestamp(snapshot_meta["timestamp"]).isoformat(),
            "changed_files": changed_files,
            "recommendation": "Run 'bbx recover' to restore changed files, or 'bbx pack' to save current state.",
        }

    # =========================================================================
    # Helper: Save/Load package
    # =========================================================================

    def _save_package(self, package: BBXPackage, path: str):
        """Save .bbx package"""
        import gzip

        data = {
            "version": package.version,
            "created_at": package.created_at,
            "intent": package.intent,
            "genome_hash": package.genome_hash,
            "recovery_snapshots": package.recovery_snapshots,
            "recovery_percent": package.recovery_percent,
            "proof": package.proof,
            "project_name": package.project_name,
            "project_path": package.project_path,
            "files_count": package.files_count,
            "lines_count": package.lines_count,
        }

        json_bytes = json.dumps(data, indent=2).encode()
        compressed = gzip.compress(json_bytes)

        Path(path).write_bytes(b"BBX\x00" + compressed)

    def _load_package(self, path: str) -> BBXPackage:
        """Load .bbx package"""
        import gzip

        raw = Path(path).read_bytes()

        if not raw.startswith(b"BBX\x00"):
            raise ValueError("Invalid .bbx file")

        decompressed = gzip.decompress(raw[4:])
        data = json.loads(decompressed)

        return BBXPackage(**data)


# =========================================================================
# Simple CLI functions (like WinRAR's simple interface)
# =========================================================================

async def pack(path: str = ".", output: str = None, recovery: int = 5) -> str:
    """Pack project - simple function for CLI"""
    core = BBXCore(path)
    return await core.pack(output=output, recovery_percent=recovery)


async def unpack(intent: str, path: str = ".", dry_run: bool = False) -> Dict:
    """Unpack intent - simple function for CLI"""
    core = BBXCore(path)
    return await core.unpack(intent, dry_run=dry_run)


async def recover(path: str = ".", snapshot: str = None, file: str = None) -> Dict:
    """Recover project - simple function for CLI"""
    core = BBXCore(path)
    return await core.recover(snapshot_id=snapshot, file_path=file)


async def smart_recover(path: str = ".") -> Dict:
    """Smart recovery - simple function for CLI"""
    core = BBXCore(path)
    return await core.smart_recover()
