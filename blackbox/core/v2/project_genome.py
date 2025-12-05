# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Project Genome - Complete Project Understanding for AI

The problem: AI is imprecise. It reads code but doesn't truly understand.
The solution: Capture the "DNA" of a project - structure, relationships,
              successful patterns - so AI can work with precision.

Like a game replay system:
- State: Current project snapshot
- Actions: Code changes (recorded)
- Determinism: Same actions on same state = same result

This enables:
1. AI that KNOWS the project, not guesses
2. Reproducible changes (replay successful patterns)
3. Verification (did this change produce expected state?)

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                      PROJECT GENOME                            │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ Structure Layer                                          │  │
    │  │   files, directories, dependencies, configs              │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ Understanding Layer                                      │  │
    │  │   embeddings, relationships, patterns                    │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ History Layer                                            │  │
    │  │   successful changes, replays, proofs                    │  │
    │  └──────────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────────┘
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
from typing import Any, Dict, List, Optional, Set, Tuple
import fnmatch

logger = logging.getLogger("bbx.genome")


class FileType(Enum):
    """Types of files in project"""
    SOURCE = "source"       # .py, .ts, .js, .rs, etc.
    CONFIG = "config"       # .json, .yaml, .toml, etc.
    DOCS = "docs"           # .md, .txt, .rst
    TEST = "test"           # test_*, *_test.*, *.spec.*
    BUILD = "build"         # Dockerfile, Makefile, etc.
    DATA = "data"           # .csv, .sql, etc.
    ASSET = "asset"         # images, fonts, etc.
    OTHER = "other"


@dataclass
class FileNode:
    """A file in the project genome"""
    path: str
    relative_path: str
    file_type: FileType
    size: int
    hash: str  # Content hash for change detection

    # Understanding
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None

    # Relationships
    imports: List[str] = field(default_factory=list)      # Files this imports
    imported_by: List[str] = field(default_factory=list)  # Files that import this

    # Metadata
    language: Optional[str] = None
    last_modified: float = 0
    lines: int = 0

    # For AI context
    key_concepts: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)


@dataclass
class ProjectSnapshot:
    """A snapshot of project state at a point in time"""
    id: str
    timestamp: float
    commit_hash: Optional[str] = None

    # State
    file_hashes: Dict[str, str] = field(default_factory=dict)  # path -> hash
    total_files: int = 0
    total_lines: int = 0

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class RecordedAction:
    """A recorded code change action"""
    id: str
    timestamp: float

    # What happened
    action_type: str  # 'create', 'modify', 'delete', 'rename'
    file_path: str

    # Before/After
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    diff: Optional[str] = None  # Unified diff

    # Context
    intent: str = ""  # What was the goal?
    prompt: str = ""  # What was asked?

    # Verification
    tests_before: Optional[bool] = None
    tests_after: Optional[bool] = None
    build_success: Optional[bool] = None


@dataclass
class SuccessfulPath:
    """A recorded successful sequence of changes"""
    id: str
    intent: str  # "Add authentication", "Fix bug #123"

    # The path
    start_snapshot: str
    end_snapshot: str
    actions: List[str]  # Action IDs in order

    # Proof
    verification: Dict[str, bool] = field(default_factory=dict)  # test/build results

    # For replay
    embedding: Optional[List[float]] = None  # For finding similar paths


@dataclass
class ProjectGenome:
    """Complete project understanding"""
    # Identity
    project_path: str
    project_name: str
    created_at: float = field(default_factory=time.time)

    # Structure
    files: Dict[str, FileNode] = field(default_factory=dict)
    directories: Set[str] = field(default_factory=set)

    # Understanding
    project_embedding: Optional[List[float]] = None
    project_summary: Optional[str] = None
    tech_stack: List[str] = field(default_factory=list)

    # History
    snapshots: Dict[str, ProjectSnapshot] = field(default_factory=dict)
    actions: Dict[str, RecordedAction] = field(default_factory=dict)
    successful_paths: Dict[str, SuccessfulPath] = field(default_factory=dict)

    # Current state
    current_snapshot_id: Optional[str] = None


class GenomeAnalyzer:
    """Analyzes project to build genome"""

    # File type patterns
    SOURCE_PATTERNS = {
        '*.py': 'python',
        '*.ts': 'typescript',
        '*.tsx': 'typescript',
        '*.js': 'javascript',
        '*.jsx': 'javascript',
        '*.rs': 'rust',
        '*.go': 'go',
        '*.java': 'java',
        '*.cpp': 'cpp',
        '*.c': 'c',
        '*.rb': 'ruby',
        '*.php': 'php',
    }

    CONFIG_PATTERNS = ['*.json', '*.yaml', '*.yml', '*.toml', '*.ini', '*.env*']
    DOC_PATTERNS = ['*.md', '*.txt', '*.rst', '*.adoc']
    TEST_PATTERNS = ['test_*', '*_test.*', '*.spec.*', '*.test.*']
    BUILD_PATTERNS = ['Dockerfile*', 'Makefile', '*.mk', 'CMakeLists.txt']

    IGNORE_PATTERNS = [
        'node_modules', '__pycache__', '.git', '.venv', 'venv',
        'dist', 'build', '.cache', '*.pyc', '.DS_Store',
        '*.egg-info', '.mypy_cache', '.pytest_cache',
    ]

    def __init__(self, embedder=None):
        self._embedder = embedder

    async def _get_embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not available")
        return self._embedder

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        path_str = str(path)
        for pattern in self.IGNORE_PATTERNS:
            if pattern in path_str or fnmatch.fnmatch(path.name, pattern):
                return True
        return False

    def _detect_file_type(self, path: Path) -> Tuple[FileType, Optional[str]]:
        """Detect file type and language"""
        name = path.name

        # Test files
        for pattern in self.TEST_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return FileType.TEST, None

        # Source files
        for pattern, language in self.SOURCE_PATTERNS.items():
            if fnmatch.fnmatch(name, pattern):
                return FileType.SOURCE, language

        # Config
        for pattern in self.CONFIG_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return FileType.CONFIG, None

        # Docs
        for pattern in self.DOC_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return FileType.DOCS, None

        # Build
        for pattern in self.BUILD_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return FileType.BUILD, None

        return FileType.OTHER, None

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract imports from source file"""
        imports = []

        if language == 'python':
            # import x, from x import y
            for match in re.finditer(r'^(?:from\s+(\S+)|import\s+(\S+))', content, re.MULTILINE):
                module = match.group(1) or match.group(2)
                if module:
                    imports.append(module.split('.')[0])

        elif language in ('typescript', 'javascript'):
            # import ... from 'x', require('x')
            for match in re.finditer(r"(?:from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))", content):
                module = match.group(1) or match.group(2)
                if module and not module.startswith('.'):
                    imports.append(module)

        elif language == 'rust':
            # use x::y
            for match in re.finditer(r'^use\s+(\w+)', content, re.MULTILINE):
                imports.append(match.group(1))

        elif language == 'go':
            # import "x"
            for match in re.finditer(r'import\s+["\']([^"\']+)["\']', content):
                imports.append(match.group(1).split('/')[-1])

        return list(set(imports))

    def _extract_symbols(self, content: str, language: str) -> Tuple[List[str], List[str]]:
        """Extract function and class names"""
        functions = []
        classes = []

        if language == 'python':
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)

        elif language in ('typescript', 'javascript'):
            functions = re.findall(r'(?:function\s+(\w+)|(\w+)\s*[=:]\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))', content)
            functions = [f[0] or f[1] for f in functions if f[0] or f[1]]
            classes = re.findall(r'class\s+(\w+)', content)

        elif language == 'rust':
            functions = re.findall(r'fn\s+(\w+)', content)
            classes = re.findall(r'(?:struct|enum|trait)\s+(\w+)', content)

        return functions[:20], classes[:10]  # Limit

    def _hash_content(self, content: str) -> str:
        """Create content hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def analyze_file(self, path: Path, base_path: Path) -> Optional[FileNode]:
        """Analyze a single file"""
        if self._should_ignore(path):
            return None

        try:
            relative = str(path.relative_to(base_path))
            file_type, language = self._detect_file_type(path)

            # Read content for source/config/docs
            content = ""
            if file_type in (FileType.SOURCE, FileType.CONFIG, FileType.DOCS, FileType.TEST):
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                except:
                    pass

            # Extract info
            imports = []
            functions = []
            classes = []

            if language and content:
                imports = self._extract_imports(content, language)
                functions, classes = self._extract_symbols(content, language)

            # Create embedding for understanding
            embedding = None
            embedder = await self._get_embedder()
            if embedder and content and len(content) > 50:
                # Use first 2000 chars for embedding
                text_for_embedding = f"{relative}\n{content[:2000]}"
                embedding = embedder.encode(text_for_embedding, show_progress_bar=False).tolist()

            return FileNode(
                path=str(path),
                relative_path=relative,
                file_type=file_type,
                size=path.stat().st_size,
                hash=self._hash_content(content) if content else "",
                embedding=embedding,
                imports=imports,
                language=language,
                last_modified=path.stat().st_mtime,
                lines=content.count('\n') + 1 if content else 0,
                functions=functions,
                classes=classes,
            )

        except Exception as e:
            logger.warning(f"Error analyzing {path}: {e}")
            return None

    async def analyze_project(self, project_path: str) -> ProjectGenome:
        """Analyze entire project and build genome"""
        base_path = Path(project_path)

        if not base_path.exists():
            raise ValueError(f"Project path not found: {project_path}")

        logger.info(f"Analyzing project: {base_path}")

        genome = ProjectGenome(
            project_path=str(base_path.absolute()),
            project_name=base_path.name,
        )

        # Find all files
        all_files = []
        for root, dirs, files in os.walk(base_path):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file
                if not self._should_ignore(file_path):
                    all_files.append(file_path)

        logger.info(f"Found {len(all_files)} files to analyze")

        # Analyze files (with concurrency limit)
        semaphore = asyncio.Semaphore(10)

        async def analyze_with_semaphore(path):
            async with semaphore:
                return await self.analyze_file(path, base_path)

        tasks = [analyze_with_semaphore(f) for f in all_files]
        results = await asyncio.gather(*tasks)

        # Build genome
        tech_stack = set()
        total_lines = 0

        for node in results:
            if node:
                genome.files[node.relative_path] = node
                genome.directories.add(str(Path(node.relative_path).parent))

                if node.language:
                    tech_stack.add(node.language)
                total_lines += node.lines

        genome.tech_stack = list(tech_stack)

        # Build import relationships
        self._build_relationships(genome)

        # Create project-level understanding
        await self._create_project_understanding(genome)

        # Create initial snapshot
        snapshot = self._create_snapshot(genome, "Initial analysis")
        genome.snapshots[snapshot.id] = snapshot
        genome.current_snapshot_id = snapshot.id

        logger.info(f"Genome built: {len(genome.files)} files, {total_lines} lines, stack: {genome.tech_stack}")

        return genome

    def _build_relationships(self, genome: ProjectGenome):
        """Build import/dependency relationships between files"""
        # Map module names to files
        module_to_file = {}
        for path, node in genome.files.items():
            # Python: foo/bar.py -> foo.bar
            if node.language == 'python':
                module = path.replace('/', '.').replace('\\', '.').replace('.py', '')
                module_to_file[module] = path
                # Also just the filename
                module_to_file[Path(path).stem] = path

        # Resolve imports
        for path, node in genome.files.items():
            for imp in node.imports:
                if imp in module_to_file:
                    target = module_to_file[imp]
                    if target != path:
                        node.imports = [i for i in node.imports if i != imp]
                        node.imports.append(target)
                        if target in genome.files:
                            genome.files[target].imported_by.append(path)

    async def _create_project_understanding(self, genome: ProjectGenome):
        """Create project-level embedding and summary"""
        embedder = await self._get_embedder()
        if not embedder:
            return

        # Create project description from key files
        key_content = []

        # README
        for name in ['README.md', 'readme.md', 'README.txt']:
            if name in genome.files:
                try:
                    content = Path(genome.files[name].path).read_text()[:1000]
                    key_content.append(content)
                except:
                    pass

        # Package files
        for name in ['package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod']:
            if name in genome.files:
                try:
                    content = Path(genome.files[name].path).read_text()[:500]
                    key_content.append(content)
                except:
                    pass

        if key_content:
            project_text = f"Project: {genome.project_name}\nTech: {', '.join(genome.tech_stack)}\n\n" + "\n".join(key_content)
            genome.project_embedding = embedder.encode(project_text[:3000], show_progress_bar=False).tolist()

    def _create_snapshot(self, genome: ProjectGenome, description: str = "") -> ProjectSnapshot:
        """Create a snapshot of current state"""
        snapshot_id = hashlib.sha256(
            f"{time.time()}{description}".encode()
        ).hexdigest()[:12]

        return ProjectSnapshot(
            id=snapshot_id,
            timestamp=time.time(),
            file_hashes={path: node.hash for path, node in genome.files.items()},
            total_files=len(genome.files),
            total_lines=sum(n.lines for n in genome.files.values()),
            description=description,
        )


class GenomeRecorder:
    """Records changes to project for replay"""

    def __init__(self, genome: ProjectGenome):
        self.genome = genome
        self._recording = False
        self._current_actions: List[RecordedAction] = []

    def start_recording(self, intent: str):
        """Start recording a change sequence"""
        self._recording = True
        self._current_intent = intent
        self._current_actions = []
        self._start_snapshot = self.genome.current_snapshot_id
        logger.info(f"Started recording: {intent}")

    def record_action(
        self,
        action_type: str,
        file_path: str,
        before_content: Optional[str] = None,
        after_content: Optional[str] = None,
        prompt: str = "",
    ):
        """Record a single action"""
        if not self._recording:
            return

        action_id = hashlib.sha256(
            f"{time.time()}{file_path}{action_type}".encode()
        ).hexdigest()[:12]

        # Create diff if both contents provided
        diff = None
        if before_content is not None and after_content is not None:
            import difflib
            diff = '\n'.join(difflib.unified_diff(
                before_content.splitlines(),
                after_content.splitlines(),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                lineterm=""
            ))

        action = RecordedAction(
            id=action_id,
            timestamp=time.time(),
            action_type=action_type,
            file_path=file_path,
            before_hash=hashlib.sha256(before_content.encode()).hexdigest()[:16] if before_content else None,
            after_hash=hashlib.sha256(after_content.encode()).hexdigest()[:16] if after_content else None,
            diff=diff,
            intent=self._current_intent,
            prompt=prompt,
        )

        self._current_actions.append(action)
        self.genome.actions[action_id] = action
        logger.debug(f"Recorded: {action_type} {file_path}")

    def stop_recording(self, success: bool, verification: Dict[str, bool] = None) -> Optional[SuccessfulPath]:
        """Stop recording and optionally save as successful path"""
        if not self._recording:
            return None

        self._recording = False

        if not success or not self._current_actions:
            logger.info("Recording stopped (not saved)")
            return None

        # Create new snapshot
        from .project_genome import GenomeAnalyzer
        analyzer = GenomeAnalyzer()
        end_snapshot = analyzer._create_snapshot(self.genome, f"After: {self._current_intent}")
        self.genome.snapshots[end_snapshot.id] = end_snapshot
        self.genome.current_snapshot_id = end_snapshot.id

        # Create successful path
        path_id = hashlib.sha256(
            f"{self._current_intent}{time.time()}".encode()
        ).hexdigest()[:12]

        path = SuccessfulPath(
            id=path_id,
            intent=self._current_intent,
            start_snapshot=self._start_snapshot,
            end_snapshot=end_snapshot.id,
            actions=[a.id for a in self._current_actions],
            verification=verification or {},
        )

        self.genome.successful_paths[path_id] = path
        logger.info(f"Saved successful path: {self._current_intent} ({len(self._current_actions)} actions)")

        return path


class GenomeReplayer:
    """Replays recorded successful paths"""

    def __init__(self, genome: ProjectGenome):
        self.genome = genome
        self._embedder = None

    async def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                pass
        return self._embedder

    async def find_similar_path(self, intent: str, top_k: int = 3) -> List[Tuple[SuccessfulPath, float]]:
        """Find similar successful paths"""
        if not self.genome.successful_paths:
            return []

        embedder = await self._get_embedder()
        if not embedder:
            # Fallback to keyword matching
            results = []
            intent_words = set(intent.lower().split())
            for path in self.genome.successful_paths.values():
                path_words = set(path.intent.lower().split())
                overlap = len(intent_words & path_words) / max(len(intent_words), 1)
                if overlap > 0:
                    results.append((path, overlap))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        # Semantic search
        import numpy as np
        intent_embedding = embedder.encode(intent, show_progress_bar=False)

        results = []
        for path in self.genome.successful_paths.values():
            if path.embedding:
                similarity = np.dot(intent_embedding, path.embedding) / (
                    np.linalg.norm(intent_embedding) * np.linalg.norm(path.embedding)
                )
                results.append((path, float(similarity)))
            else:
                # Compute embedding
                path_embedding = embedder.encode(path.intent, show_progress_bar=False)
                path.embedding = path_embedding.tolist()
                similarity = np.dot(intent_embedding, path_embedding) / (
                    np.linalg.norm(intent_embedding) * np.linalg.norm(path_embedding)
                )
                results.append((path, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_replay_steps(self, path: SuccessfulPath) -> List[Dict[str, Any]]:
        """Get steps to replay a successful path"""
        steps = []

        for action_id in path.actions:
            action = self.genome.actions.get(action_id)
            if not action:
                continue

            step = {
                "type": action.action_type,
                "file": action.file_path,
                "diff": action.diff,
                "original_prompt": action.prompt,
            }
            steps.append(step)

        return steps

    def verify_replay(self, path: SuccessfulPath, current_state: Dict[str, str]) -> Dict[str, Any]:
        """Verify if replay produced expected state"""
        end_snapshot = self.genome.snapshots.get(path.end_snapshot)
        if not end_snapshot:
            return {"verified": False, "error": "End snapshot not found"}

        # Compare file hashes
        mismatches = []
        for file_path, expected_hash in end_snapshot.file_hashes.items():
            if file_path in current_state:
                actual_hash = hashlib.sha256(current_state[file_path].encode()).hexdigest()[:16]
                if actual_hash != expected_hash:
                    mismatches.append(file_path)

        return {
            "verified": len(mismatches) == 0,
            "mismatches": mismatches,
            "expected_files": len(end_snapshot.file_hashes),
            "checked_files": len(current_state),
        }


# Convenience functions

async def analyze_project(path: str) -> ProjectGenome:
    """Quick function to analyze a project"""
    analyzer = GenomeAnalyzer()
    return await analyzer.analyze_project(path)


def save_genome(genome: ProjectGenome, path: str):
    """Save genome to JSON file"""
    import json

    # Convert to serializable format
    data = {
        "project_path": genome.project_path,
        "project_name": genome.project_name,
        "created_at": genome.created_at,
        "tech_stack": genome.tech_stack,
        "project_summary": genome.project_summary,
        "files": {
            k: {
                "path": v.path,
                "relative_path": v.relative_path,
                "file_type": v.file_type.value,
                "size": v.size,
                "hash": v.hash,
                "language": v.language,
                "lines": v.lines,
                "functions": v.functions,
                "classes": v.classes,
                "imports": v.imports,
            }
            for k, v in genome.files.items()
        },
        "directories": list(genome.directories),
        "snapshots": {
            k: {
                "id": v.id,
                "timestamp": v.timestamp,
                "file_hashes": v.file_hashes,
                "total_files": v.total_files,
                "total_lines": v.total_lines,
                "description": v.description,
            }
            for k, v in genome.snapshots.items()
        },
        "actions": {
            k: {
                "id": v.id,
                "timestamp": v.timestamp,
                "action_type": v.action_type,
                "file_path": v.file_path,
                "intent": v.intent,
                "prompt": v.prompt,
            }
            for k, v in genome.actions.items()
        },
        "successful_paths": {
            k: {
                "id": v.id,
                "intent": v.intent,
                "start_snapshot": v.start_snapshot,
                "end_snapshot": v.end_snapshot,
                "actions": v.actions,
                "verification": v.verification,
            }
            for k, v in genome.successful_paths.items()
        },
        "current_snapshot_id": genome.current_snapshot_id,
    }

    Path(path).write_text(json.dumps(data, indent=2))


def load_genome(path: str) -> ProjectGenome:
    """Load genome from JSON file"""
    import json

    data = json.loads(Path(path).read_text())

    genome = ProjectGenome(
        project_path=data["project_path"],
        project_name=data["project_name"],
        created_at=data["created_at"],
        tech_stack=data["tech_stack"],
        project_summary=data.get("project_summary"),
        current_snapshot_id=data.get("current_snapshot_id"),
    )

    # Restore files
    for k, v in data["files"].items():
        genome.files[k] = FileNode(
            path=v["path"],
            relative_path=v["relative_path"],
            file_type=FileType(v["file_type"]),
            size=v["size"],
            hash=v["hash"],
            language=v.get("language"),
            lines=v.get("lines", 0),
            functions=v.get("functions", []),
            classes=v.get("classes", []),
            imports=v.get("imports", []),
        )

    genome.directories = set(data.get("directories", []))

    # Restore snapshots
    for k, v in data.get("snapshots", {}).items():
        genome.snapshots[k] = ProjectSnapshot(**v)

    # Restore actions
    for k, v in data.get("actions", {}).items():
        genome.actions[k] = RecordedAction(**v)

    # Restore paths
    for k, v in data.get("successful_paths", {}).items():
        genome.successful_paths[k] = SuccessfulPath(**v)

    return genome
