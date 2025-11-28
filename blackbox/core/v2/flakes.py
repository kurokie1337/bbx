# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Flakes - Nix Flakes-Inspired Reproducible Environments

Inspired by Nix Flakes, provides:
- Lockfiles for exact reproducibility
- Flake inputs and outputs
- Pure evaluation mode
- Hermetic builds
- Remote flake references
- Content-addressed storage

Key concepts:
- Flake: A reproducible package with locked dependencies
- FlakeRef: Reference to a flake (local, git, registry)
- FlakeLock: Locked versions of all inputs
- FlakeOutput: What a flake produces (agents, workflows, adapters)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger("bbx.flakes")


# =============================================================================
# Flake Reference Types
# =============================================================================

class FlakeRefType(Enum):
    """Types of flake references."""
    PATH = "path"           # Local path: ./my-flake
    GIT = "git"             # Git repo: git+https://github.com/user/repo
    GITHUB = "github"       # GitHub shorthand: github:user/repo
    REGISTRY = "registry"   # BBX Registry: bbx:name/version
    TARBALL = "tarball"     # HTTP tarball: https://example.com/flake.tar.gz
    INDIRECT = "indirect"   # Indirect reference via registry


@dataclass
class FlakeRef:
    """Reference to a flake."""

    type: FlakeRefType
    url: str

    # Optional specifiers
    ref: Optional[str] = None      # Git ref (branch/tag)
    rev: Optional[str] = None      # Git revision (commit hash)
    dir: Optional[str] = None      # Subdirectory within flake
    narHash: Optional[str] = None  # NAR hash for verification

    # For registry refs
    name: Optional[str] = None
    version: Optional[str] = None

    def __str__(self) -> str:
        if self.type == FlakeRefType.PATH:
            return f"path:{self.url}"
        elif self.type == FlakeRefType.GITHUB:
            base = f"github:{self.url}"
            if self.ref:
                base += f"/{self.ref}"
            if self.rev:
                base += f"?rev={self.rev}"
            return base
        elif self.type == FlakeRefType.GIT:
            base = f"git+{self.url}"
            if self.ref:
                base += f"?ref={self.ref}"
            if self.rev:
                base += f"&rev={self.rev}"
            return base
        elif self.type == FlakeRefType.REGISTRY:
            return f"bbx:{self.name}@{self.version or 'latest'}"
        else:
            return self.url

    @classmethod
    def parse(cls, ref_str: str) -> "FlakeRef":
        """Parse a flake reference string."""
        # Path reference
        if ref_str.startswith("./") or ref_str.startswith("/") or ref_str.startswith("path:"):
            path = ref_str.replace("path:", "")
            return cls(type=FlakeRefType.PATH, url=path)

        # GitHub shorthand
        if ref_str.startswith("github:"):
            parts = ref_str[7:].split("/")
            if len(parts) >= 2:
                url = f"{parts[0]}/{parts[1]}"
                ref = parts[2] if len(parts) > 2 else None
                return cls(type=FlakeRefType.GITHUB, url=url, ref=ref)

        # Git URL
        if ref_str.startswith("git+"):
            url = ref_str[4:]
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            ref = None
            rev = None
            if parsed.query:
                for param in parsed.query.split("&"):
                    key, value = param.split("=")
                    if key == "ref":
                        ref = value
                    elif key == "rev":
                        rev = value
            return cls(type=FlakeRefType.GIT, url=base_url, ref=ref, rev=rev)

        # Registry reference
        if ref_str.startswith("bbx:"):
            parts = ref_str[4:].split("@")
            name = parts[0]
            version = parts[1] if len(parts) > 1 else None
            return cls(type=FlakeRefType.REGISTRY, url=ref_str, name=name, version=version)

        # Tarball
        if ref_str.startswith("http://") or ref_str.startswith("https://"):
            return cls(type=FlakeRefType.TARBALL, url=ref_str)

        # Default to indirect
        return cls(type=FlakeRefType.INDIRECT, url=ref_str, name=ref_str)


# =============================================================================
# Flake Lock
# =============================================================================

@dataclass
class LockedInput:
    """A locked flake input with exact version."""

    original: FlakeRef           # Original reference
    locked: FlakeRef             # Locked reference with exact version
    last_modified: datetime
    nar_hash: str               # Content hash for verification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": str(self.original),
            "locked": {
                "type": self.locked.type.value,
                "url": self.locked.url,
                "ref": self.locked.ref,
                "rev": self.locked.rev,
                "narHash": self.locked.narHash or self.nar_hash,
            },
            "lastModified": self.last_modified.isoformat(),
            "narHash": self.nar_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockedInput":
        original = FlakeRef.parse(data["original"])
        locked_data = data["locked"]
        locked = FlakeRef(
            type=FlakeRefType(locked_data["type"]),
            url=locked_data["url"],
            ref=locked_data.get("ref"),
            rev=locked_data.get("rev"),
            narHash=locked_data.get("narHash"),
        )
        return cls(
            original=original,
            locked=locked,
            last_modified=datetime.fromisoformat(data["lastModified"]),
            nar_hash=data["narHash"],
        )


@dataclass
class FlakeLock:
    """Lock file for a flake - ensures reproducibility."""

    version: int = 1
    root: str = "root"
    nodes: Dict[str, LockedInput] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_input(self, name: str, locked: LockedInput):
        """Add or update a locked input."""
        self.nodes[name] = locked
        self.updated_at = datetime.now()

    def get_input(self, name: str) -> Optional[LockedInput]:
        """Get a locked input by name."""
        return self.nodes.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "root": self.root,
            "nodes": {name: inp.to_dict() for name, inp in self.nodes.items()},
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlakeLock":
        nodes = {
            name: LockedInput.from_dict(inp_data)
            for name, inp_data in data.get("nodes", {}).items()
        }
        return cls(
            version=data.get("version", 1),
            root=data.get("root", "root"),
            nodes=nodes,
            created_at=datetime.fromisoformat(data.get("createdAt", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updatedAt", datetime.now().isoformat())),
        )

    def save(self, path: Path):
        """Save lock file to disk."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FlakeLock":
        """Load lock file from disk."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Flake Outputs
# =============================================================================

class FlakeOutputType(Enum):
    """Types of flake outputs."""
    AGENT = "agent"
    WORKFLOW = "workflow"
    ADAPTER = "adapter"
    HOOK = "hook"
    BUNDLE = "bundle"
    DEV_SHELL = "devShell"
    OVERLAY = "overlay"


@dataclass
class FlakeOutput:
    """An output from a flake."""

    name: str
    type: FlakeOutputType
    description: str = ""

    # Output content
    content: Optional[Any] = None
    path: Optional[str] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Build info
    builder: Optional[str] = None
    build_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "path": self.path,
            "dependsOn": self.depends_on,
            "builder": self.builder,
            "buildInputs": self.build_inputs,
        }


# =============================================================================
# Flake Definition
# =============================================================================

@dataclass
class FlakeInput:
    """An input to a flake."""

    name: str
    ref: FlakeRef
    follows: Optional[str] = None  # Follow another input's version
    flake: bool = True             # Whether this is a flake

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": str(self.ref),
            "follows": self.follows,
            "flake": self.flake,
        }


@dataclass
class Flake:
    """A BBX Flake - reproducible package definition."""

    # Identity
    name: str
    version: str
    description: str = ""

    # Inputs (dependencies)
    inputs: Dict[str, FlakeInput] = field(default_factory=dict)

    # Outputs (what this flake provides)
    outputs: Dict[str, FlakeOutput] = field(default_factory=dict)

    # Lock file
    lock: Optional[FlakeLock] = None

    # Metadata
    path: Optional[Path] = None
    nar_hash: Optional[str] = None

    # Build configuration
    pure_eval: bool = True          # Pure evaluation mode
    allow_unfree: bool = False      # Allow unfree packages
    experimental_features: List[str] = field(default_factory=list)

    def add_input(self, name: str, ref: Union[str, FlakeRef], **kwargs):
        """Add an input to this flake."""
        if isinstance(ref, str):
            ref = FlakeRef.parse(ref)
        self.inputs[name] = FlakeInput(name=name, ref=ref, **kwargs)

    def add_output(
        self,
        name: str,
        output_type: FlakeOutputType,
        content: Optional[Any] = None,
        **kwargs
    ):
        """Add an output to this flake."""
        self.outputs[name] = FlakeOutput(
            name=name,
            type=output_type,
            content=content,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "inputs": {name: inp.to_dict() for name, inp in self.inputs.items()},
            "outputs": {name: out.to_dict() for name, out in self.outputs.items()},
            "pureEval": self.pure_eval,
            "allowUnfree": self.allow_unfree,
            "experimentalFeatures": self.experimental_features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Optional[Path] = None) -> "Flake":
        """Create flake from dictionary."""
        flake = cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            pure_eval=data.get("pureEval", True),
            allow_unfree=data.get("allowUnfree", False),
            experimental_features=data.get("experimentalFeatures", []),
            path=path,
        )

        # Add inputs
        for name, inp_data in data.get("inputs", {}).items():
            ref = FlakeRef.parse(inp_data["url"])
            flake.add_input(
                name,
                ref,
                follows=inp_data.get("follows"),
                flake=inp_data.get("flake", True),
            )

        # Add outputs
        for name, out_data in data.get("outputs", {}).items():
            flake.add_output(
                name,
                FlakeOutputType(out_data["type"]),
                path=out_data.get("path"),
                description=out_data.get("description", ""),
                depends_on=out_data.get("dependsOn", []),
            )

        return flake


# =============================================================================
# Content-Addressed Storage
# =============================================================================

class ContentStore:
    """Content-addressed storage for flake outputs."""

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Index of stored content
        self._index: Dict[str, Path] = {}
        self._load_index()

    def _load_index(self):
        """Load index from disk."""
        index_path = self.store_path / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
                self._index = {k: Path(v) for k, v in data.items()}

    def _save_index(self):
        """Save index to disk."""
        index_path = self.store_path / "index.json"
        with open(index_path, "w") as f:
            json.dump({k: str(v) for k, v in self._index.items()}, f, indent=2)

    def compute_hash(self, content: bytes) -> str:
        """Compute content hash (NAR hash)."""
        return hashlib.sha256(content).hexdigest()

    def store(self, content: bytes, name: str = "") -> str:
        """Store content and return hash."""
        content_hash = self.compute_hash(content)

        if content_hash in self._index:
            return content_hash

        # Store content
        content_path = self.store_path / content_hash[:2] / content_hash
        content_path.parent.mkdir(parents=True, exist_ok=True)

        with open(content_path, "wb") as f:
            f.write(content)

        self._index[content_hash] = content_path
        self._save_index()

        logger.debug(f"Stored content {content_hash[:16]}... ({len(content)} bytes)")
        return content_hash

    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content by hash."""
        if content_hash not in self._index:
            return None

        content_path = self._index[content_hash]
        if not content_path.exists():
            del self._index[content_hash]
            return None

        with open(content_path, "rb") as f:
            return f.read()

    def exists(self, content_hash: str) -> bool:
        """Check if content exists in store."""
        return content_hash in self._index

    def gc(self, keep_hashes: Set[str]) -> int:
        """Garbage collect unused content."""
        removed = 0
        for content_hash in list(self._index.keys()):
            if content_hash not in keep_hashes:
                content_path = self._index[content_hash]
                if content_path.exists():
                    content_path.unlink()
                del self._index[content_hash]
                removed += 1

        self._save_index()
        return removed


# =============================================================================
# Flake Fetcher
# =============================================================================

class FlakeFetcher(ABC):
    """Base class for fetching flakes from different sources."""

    @abstractmethod
    async def fetch(self, ref: FlakeRef, store: ContentStore) -> Tuple[Path, str]:
        """Fetch flake and return (path, nar_hash)."""
        pass

    @abstractmethod
    def supports(self, ref: FlakeRef) -> bool:
        """Check if this fetcher supports the given ref type."""
        pass


class PathFetcher(FlakeFetcher):
    """Fetch flakes from local paths."""

    def supports(self, ref: FlakeRef) -> bool:
        return ref.type == FlakeRefType.PATH

    async def fetch(self, ref: FlakeRef, store: ContentStore) -> Tuple[Path, str]:
        path = Path(ref.url).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Flake path not found: {path}")

        # Compute hash of flake.bbx
        flake_file = path / "flake.bbx"
        if flake_file.exists():
            with open(flake_file, "rb") as f:
                nar_hash = store.compute_hash(f.read())
        else:
            nar_hash = store.compute_hash(str(path).encode())

        return path, nar_hash


class GitFetcher(FlakeFetcher):
    """Fetch flakes from Git repositories."""

    def supports(self, ref: FlakeRef) -> bool:
        return ref.type in (FlakeRefType.GIT, FlakeRefType.GITHUB)

    async def fetch(self, ref: FlakeRef, store: ContentStore) -> Tuple[Path, str]:
        # Convert GitHub shorthand to full URL
        if ref.type == FlakeRefType.GITHUB:
            url = f"https://github.com/{ref.url}.git"
        else:
            url = ref.url

        # Create temp directory for clone
        temp_dir = Path(tempfile.mkdtemp(prefix="bbx-flake-"))

        try:
            # Clone repository
            cmd = ["git", "clone", "--depth", "1"]
            if ref.ref:
                cmd.extend(["--branch", ref.ref])
            cmd.extend([url, str(temp_dir)])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")

            # Get exact commit
            if ref.rev:
                proc = await asyncio.create_subprocess_exec(
                    "git", "-C", str(temp_dir), "checkout", ref.rev,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()

            # Get current commit hash
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(temp_dir), "rev-parse", "HEAD",
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            commit_hash = stdout.decode().strip()

            # Compute NAR hash
            flake_file = temp_dir / "flake.bbx"
            if flake_file.exists():
                with open(flake_file, "rb") as f:
                    nar_hash = store.compute_hash(f.read())
            else:
                nar_hash = store.compute_hash(commit_hash.encode())

            return temp_dir, nar_hash

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise


class RegistryFetcher(FlakeFetcher):
    """Fetch flakes from BBX registry."""

    def __init__(self, registry_url: str = "https://registry.bbx.dev"):
        self.registry_url = registry_url

    def supports(self, ref: FlakeRef) -> bool:
        return ref.type == FlakeRefType.REGISTRY

    async def fetch(self, ref: FlakeRef, store: ContentStore) -> Tuple[Path, str]:
        # In production, this would fetch from the registry API
        # For now, we simulate with a local cache
        cache_dir = Path.home() / ".bbx" / "registry" / ref.name

        if not cache_dir.exists():
            raise FileNotFoundError(f"Package not found in registry: {ref.name}")

        version_dir = cache_dir / (ref.version or "latest")
        if not version_dir.exists():
            raise FileNotFoundError(f"Version not found: {ref.version}")

        # Compute hash
        flake_file = version_dir / "flake.bbx"
        if flake_file.exists():
            with open(flake_file, "rb") as f:
                nar_hash = store.compute_hash(f.read())
        else:
            nar_hash = store.compute_hash(f"{ref.name}@{ref.version}".encode())

        return version_dir, nar_hash


# =============================================================================
# Flake Manager
# =============================================================================

class FlakeManager:
    """
    Central manager for BBX Flakes.

    Handles:
    - Loading and parsing flakes
    - Resolving inputs (dependencies)
    - Locking versions
    - Building outputs
    - Caching and content-addressing
    """

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or (Path.home() / ".bbx" / "store")
        self.store = ContentStore(self.store_path)

        # Fetchers for different ref types
        self.fetchers: List[FlakeFetcher] = [
            PathFetcher(),
            GitFetcher(),
            RegistryFetcher(),
        ]

        # Loaded flakes cache
        self._flakes: Dict[str, Flake] = {}

        # Evaluation context
        self._eval_context: Dict[str, Any] = {}

        # Statistics
        self._stats = {
            "flakes_loaded": 0,
            "inputs_resolved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def register_fetcher(self, fetcher: FlakeFetcher):
        """Register a custom fetcher."""
        self.fetchers.insert(0, fetcher)  # Custom fetchers have priority

    async def load_flake(self, ref: Union[str, FlakeRef, Path]) -> Flake:
        """Load a flake from a reference."""
        # Parse reference
        if isinstance(ref, Path):
            ref = FlakeRef(type=FlakeRefType.PATH, url=str(ref))
        elif isinstance(ref, str):
            ref = FlakeRef.parse(ref)

        # Check cache
        cache_key = str(ref)
        if cache_key in self._flakes:
            self._stats["cache_hits"] += 1
            return self._flakes[cache_key]

        self._stats["cache_misses"] += 1

        # Find appropriate fetcher
        fetcher = None
        for f in self.fetchers:
            if f.supports(ref):
                fetcher = f
                break

        if not fetcher:
            raise ValueError(f"No fetcher found for ref type: {ref.type}")

        # Fetch flake
        path, nar_hash = await fetcher.fetch(ref, self.store)

        # Load flake definition
        flake_file = path / "flake.bbx"
        if not flake_file.exists():
            raise FileNotFoundError(f"No flake.bbx found in {path}")

        with open(flake_file) as f:
            if flake_file.suffix == ".bbx":
                # YAML format
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        flake = Flake.from_dict(data, path)
        flake.nar_hash = nar_hash

        # Load lock file if exists
        lock_file = path / "flake.lock"
        if lock_file.exists():
            flake.lock = FlakeLock.load(lock_file)

        # Cache flake
        self._flakes[cache_key] = flake
        self._stats["flakes_loaded"] += 1

        return flake

    async def resolve_inputs(
        self,
        flake: Flake,
        update: bool = False
    ) -> FlakeLock:
        """Resolve and lock all inputs."""
        lock = flake.lock or FlakeLock()

        for name, input_def in flake.inputs.items():
            # Check if already locked and not updating
            if not update and lock.get_input(name):
                continue

            # Handle follows
            if input_def.follows:
                followed = lock.get_input(input_def.follows)
                if followed:
                    lock.add_input(name, followed)
                    continue

            # Resolve input
            ref = input_def.ref

            # Find fetcher
            fetcher = None
            for f in self.fetchers:
                if f.supports(ref):
                    fetcher = f
                    break

            if not fetcher:
                raise ValueError(f"No fetcher for input {name}: {ref}")

            # Fetch to get exact version
            try:
                path, nar_hash = await fetcher.fetch(ref, self.store)

                # Create locked reference
                locked_ref = FlakeRef(
                    type=ref.type,
                    url=ref.url,
                    ref=ref.ref,
                    rev=ref.rev,
                    narHash=nar_hash,
                )

                locked = LockedInput(
                    original=ref,
                    locked=locked_ref,
                    last_modified=datetime.now(),
                    nar_hash=nar_hash,
                )

                lock.add_input(name, locked)
                self._stats["inputs_resolved"] += 1

            except Exception as e:
                logger.error(f"Failed to resolve input {name}: {e}")
                raise

        flake.lock = lock
        return lock

    async def build_output(
        self,
        flake: Flake,
        output_name: str,
        pure: Optional[bool] = None,
    ) -> Any:
        """Build a specific output from a flake."""
        if output_name not in flake.outputs:
            raise ValueError(f"Output not found: {output_name}")

        output = flake.outputs[output_name]
        pure = pure if pure is not None else flake.pure_eval

        # Set up evaluation context
        context = {
            "inputs": {},
            "self": flake,
            "pure": pure,
        }

        # Load input flakes
        if flake.lock:
            for name, locked in flake.lock.nodes.items():
                try:
                    input_flake = await self.load_flake(locked.locked)
                    context["inputs"][name] = input_flake
                except Exception as e:
                    logger.warning(f"Failed to load input {name}: {e}")

        # Build based on output type
        if output.type == FlakeOutputType.WORKFLOW:
            return await self._build_workflow(output, context)
        elif output.type == FlakeOutputType.AGENT:
            return await self._build_agent(output, context)
        elif output.type == FlakeOutputType.ADAPTER:
            return await self._build_adapter(output, context)
        elif output.type == FlakeOutputType.HOOK:
            return await self._build_hook(output, context)
        elif output.type == FlakeOutputType.DEV_SHELL:
            return await self._build_dev_shell(output, context)
        else:
            raise ValueError(f"Unknown output type: {output.type}")

    async def _build_workflow(
        self,
        output: FlakeOutput,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a workflow output."""
        if output.path:
            flake = context["self"]
            workflow_path = flake.path / output.path
            if workflow_path.exists():
                with open(workflow_path) as f:
                    import yaml
                    return yaml.safe_load(f)

        return output.content or {}

    async def _build_agent(
        self,
        output: FlakeOutput,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build an agent output."""
        return {
            "name": output.name,
            "type": "agent",
            "config": output.content or {},
        }

    async def _build_adapter(
        self,
        output: FlakeOutput,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build an adapter output."""
        return {
            "name": output.name,
            "type": "adapter",
            "config": output.content or {},
        }

    async def _build_hook(
        self,
        output: FlakeOutput,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a hook output."""
        return {
            "name": output.name,
            "type": "hook",
            "config": output.content or {},
        }

    async def _build_dev_shell(
        self,
        output: FlakeOutput,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a development shell output."""
        return {
            "name": output.name,
            "type": "devShell",
            "packages": output.build_inputs,
            "env": output.content or {},
        }

    def create_flake(
        self,
        name: str,
        version: str = "0.1.0",
        path: Optional[Path] = None,
        **kwargs
    ) -> Flake:
        """Create a new flake."""
        flake = Flake(
            name=name,
            version=version,
            path=path,
            **kwargs
        )
        return flake

    def save_flake(self, flake: Flake, path: Optional[Path] = None):
        """Save flake to disk."""
        path = path or flake.path
        if not path:
            raise ValueError("No path specified for flake")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save flake.bbx
        flake_file = path / "flake.bbx"
        import yaml
        with open(flake_file, "w") as f:
            yaml.dump(flake.to_dict(), f, default_flow_style=False)

        # Save lock file if exists
        if flake.lock:
            lock_file = path / "flake.lock"
            flake.lock.save(lock_file)

        flake.path = path

    async def update_flake(
        self,
        flake: Flake,
        inputs: Optional[List[str]] = None
    ) -> FlakeLock:
        """Update flake inputs to latest versions."""
        if inputs:
            # Update specific inputs
            for name in inputs:
                if name in flake.inputs:
                    if flake.lock:
                        # Remove from lock to force re-resolution
                        flake.lock.nodes.pop(name, None)
        else:
            # Update all inputs
            flake.lock = None

        return await self.resolve_inputs(flake, update=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "cached_flakes": len(self._flakes),
            "store_size": len(self.store._index),
        }


# =============================================================================
# Flake Template Generator
# =============================================================================

class FlakeTemplates:
    """Pre-defined flake templates."""

    @staticmethod
    def workflow(name: str, description: str = "") -> Flake:
        """Create a workflow flake template."""
        flake = Flake(
            name=name,
            version="0.1.0",
            description=description or f"BBX workflow: {name}",
        )

        # Add common inputs
        flake.add_input("bbx-lib", "bbx:lib@latest")

        # Add workflow output
        flake.add_output(
            "default",
            FlakeOutputType.WORKFLOW,
            path="workflow.bbx",
            description="Main workflow",
        )

        return flake

    @staticmethod
    def agent(name: str, description: str = "") -> Flake:
        """Create an agent flake template."""
        flake = Flake(
            name=name,
            version="0.1.0",
            description=description or f"BBX agent: {name}",
        )

        flake.add_input("bbx-lib", "bbx:lib@latest")

        flake.add_output(
            "default",
            FlakeOutputType.AGENT,
            description="Main agent",
        )

        flake.add_output(
            "devShell",
            FlakeOutputType.DEV_SHELL,
            description="Development environment",
        )

        return flake

    @staticmethod
    def bundle(name: str, agents: List[str], description: str = "") -> Flake:
        """Create a bundle flake template."""
        flake = Flake(
            name=name,
            version="0.1.0",
            description=description or f"BBX bundle: {name}",
        )

        # Add agents as inputs
        for agent in agents:
            flake.add_input(agent, f"bbx:{agent}@latest")

        flake.add_output(
            "default",
            FlakeOutputType.BUNDLE,
            description="Agent bundle",
            depends_on=agents,
        )

        return flake


# =============================================================================
# Convenience Functions
# =============================================================================

async def init_flake(
    path: Path,
    name: Optional[str] = None,
    template: str = "workflow"
) -> Flake:
    """Initialize a new flake in the given directory."""
    path = Path(path)
    name = name or path.name

    # Create from template
    if template == "workflow":
        flake = FlakeTemplates.workflow(name)
    elif template == "agent":
        flake = FlakeTemplates.agent(name)
    else:
        flake = Flake(name=name, version="0.1.0")

    flake.path = path

    # Save
    manager = FlakeManager()
    manager.save_flake(flake, path)

    return flake


async def build_flake(
    path: Union[str, Path],
    output: str = "default"
) -> Any:
    """Build a flake output."""
    manager = FlakeManager()
    flake = await manager.load_flake(Path(path))

    # Resolve inputs if needed
    if not flake.lock:
        await manager.resolve_inputs(flake)

    return await manager.build_output(flake, output)


async def update_flake_lock(
    path: Union[str, Path],
    inputs: Optional[List[str]] = None
) -> FlakeLock:
    """Update flake lock file."""
    manager = FlakeManager()
    flake = await manager.load_flake(Path(path))
    lock = await manager.update_flake(flake, inputs)
    manager.save_flake(flake)
    return lock
