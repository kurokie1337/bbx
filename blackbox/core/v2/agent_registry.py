# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Agent Registry - AUR-Inspired Community Marketplace

Inspired by Arch User Repository (AUR), provides:
- Community-contributed agents and workflows
- Build-from-source packages (PKGBUILDs)
- Voting and popularity tracking
- Dependency resolution
- Security scanning and trusted user system
- Helper tools (like yay/paru)

Key concepts:
- Package: An agent, workflow, adapter, or bundle
- PKGBUILD: Build script for packages
- Maintainer: Package owner
- TU (Trusted User): Verified maintainer with special privileges
- Orphan: Package without maintainer
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

logger = logging.getLogger("bbx.registry")


# =============================================================================
# Package Types and Status
# =============================================================================

class PackageType(Enum):
    """Types of packages in the registry."""
    AGENT = "agent"
    WORKFLOW = "workflow"
    ADAPTER = "adapter"
    HOOK = "hook"
    BUNDLE = "bundle"
    LIB = "lib"
    THEME = "theme"
    META = "meta"  # Meta-package (depends only)


class PackageStatus(Enum):
    """Package status in registry."""
    ACTIVE = "active"
    ORPHANED = "orphaned"
    FLAGGED = "flagged"      # Flagged for issues
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class PackageSource(Enum):
    """Source of package."""
    COMMUNITY = "community"  # User-contributed (like AUR)
    OFFICIAL = "official"    # Official BBX packages
    CORE = "core"           # Core system packages
    TESTING = "testing"     # Testing repository


# =============================================================================
# Package Definitions
# =============================================================================

@dataclass
class PackageVersion:
    """A specific version of a package."""

    version: str
    epoch: int = 0           # For version comparison
    pkgrel: int = 1          # Package release number

    # Build info
    source_url: Optional[str] = None
    source_hash: Optional[str] = None
    build_date: Optional[datetime] = None

    # Dependencies
    depends: List[str] = field(default_factory=list)
    makedepends: List[str] = field(default_factory=list)
    optdepends: Dict[str, str] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    replaces: List[str] = field(default_factory=list)

    # Checksums
    sha256sum: Optional[str] = None
    b2sum: Optional[str] = None

    # Install info
    install_size: int = 0
    download_size: int = 0

    def __str__(self) -> str:
        if self.epoch:
            return f"{self.epoch}:{self.version}-{self.pkgrel}"
        return f"{self.version}-{self.pkgrel}"

    def __lt__(self, other: "PackageVersion") -> bool:
        """Compare versions."""
        if self.epoch != other.epoch:
            return self.epoch < other.epoch
        # Simple version comparison (could use more sophisticated logic)
        return self.version < other.version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "epoch": self.epoch,
            "pkgrel": self.pkgrel,
            "sourceUrl": self.source_url,
            "sourceHash": self.source_hash,
            "buildDate": self.build_date.isoformat() if self.build_date else None,
            "depends": self.depends,
            "makedepends": self.makedepends,
            "optdepends": self.optdepends,
            "conflicts": self.conflicts,
            "provides": self.provides,
            "replaces": self.replaces,
            "sha256sum": self.sha256sum,
            "installSize": self.install_size,
            "downloadSize": self.download_size,
        }


@dataclass
class Package:
    """A package in the registry."""

    # Identity
    name: str
    package_type: PackageType
    description: str = ""

    # Classification
    source: PackageSource = PackageSource.COMMUNITY
    status: PackageStatus = PackageStatus.ACTIVE

    # Versions
    versions: Dict[str, PackageVersion] = field(default_factory=dict)
    latest_version: Optional[str] = None

    # Maintainer
    maintainer: Optional[str] = None
    co_maintainers: List[str] = field(default_factory=list)
    is_trusted: bool = False

    # Metadata
    url: Optional[str] = None
    license: str = "unknown"
    groups: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Statistics
    votes: int = 0
    popularity: float = 0.0
    downloads: int = 0
    first_submitted: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Flags
    out_of_date: bool = False
    out_of_date_since: Optional[datetime] = None
    flagged_reason: Optional[str] = None

    def add_version(self, version: PackageVersion):
        """Add a new version."""
        self.versions[version.version] = version
        if not self.latest_version or version > self.versions.get(self.latest_version, version):
            self.latest_version = version.version
        self.last_updated = datetime.now()

    def get_latest(self) -> Optional[PackageVersion]:
        """Get latest version."""
        if self.latest_version:
            return self.versions.get(self.latest_version)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.package_type.value,
            "description": self.description,
            "source": self.source.value,
            "status": self.status.value,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "latestVersion": self.latest_version,
            "maintainer": self.maintainer,
            "coMaintainers": self.co_maintainers,
            "isTrusted": self.is_trusted,
            "url": self.url,
            "license": self.license,
            "groups": self.groups,
            "keywords": self.keywords,
            "votes": self.votes,
            "popularity": self.popularity,
            "downloads": self.downloads,
            "firstSubmitted": self.first_submitted.isoformat() if self.first_submitted else None,
            "lastUpdated": self.last_updated.isoformat() if self.last_updated else None,
            "outOfDate": self.out_of_date,
        }


# =============================================================================
# PKGBUILD - Build Script
# =============================================================================

@dataclass
class PKGBUILD:
    """
    Build script for a package (like Arch PKGBUILD).

    Defines how to fetch, build, and package an agent/workflow.
    """

    # Package info
    pkgname: str
    pkgver: str
    pkgrel: int = 1
    epoch: int = 0
    pkgdesc: str = ""
    arch: List[str] = field(default_factory=lambda: ["any"])
    url: str = ""
    license: List[str] = field(default_factory=lambda: ["unknown"])

    # Dependencies
    depends: List[str] = field(default_factory=list)
    makedepends: List[str] = field(default_factory=list)
    optdepends: Dict[str, str] = field(default_factory=dict)
    checkdepends: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    replaces: List[str] = field(default_factory=list)

    # Sources
    source: List[str] = field(default_factory=list)
    sha256sums: List[str] = field(default_factory=list)
    b2sums: List[str] = field(default_factory=list)

    # Build functions
    prepare: Optional[str] = None    # Prepare source
    build: Optional[str] = None      # Build package
    check: Optional[str] = None      # Run tests
    package: Optional[str] = None    # Create package

    # BBX-specific
    bbx_type: PackageType = PackageType.AGENT
    bbx_hooks: List[str] = field(default_factory=list)  # Install hooks

    def to_script(self) -> str:
        """Generate PKGBUILD script."""
        lines = [
            "# BBX PKGBUILD",
            f"pkgname={self.pkgname}",
            f"pkgver={self.pkgver}",
            f"pkgrel={self.pkgrel}",
        ]

        if self.epoch:
            lines.append(f"epoch={self.epoch}")

        lines.extend([
            f'pkgdesc="{self.pkgdesc}"',
            f"arch=({' '.join(repr(a) for a in self.arch)})",
            f'url="{self.url}"',
            f"license=({' '.join(repr(l) for l in self.license)})",
        ])

        if self.depends:
            lines.append(f"depends=({' '.join(repr(d) for d in self.depends)})")
        if self.makedepends:
            lines.append(f"makedepends=({' '.join(repr(d) for d in self.makedepends)})")
        if self.source:
            lines.append(f"source=({' '.join(repr(s) for s in self.source)})")
        if self.sha256sums:
            lines.append(f"sha256sums=({' '.join(repr(s) for s in self.sha256sums)})")

        if self.prepare:
            lines.extend(["", "prepare() {", self.prepare, "}"])
        if self.build:
            lines.extend(["", "build() {", self.build, "}"])
        if self.check:
            lines.extend(["", "check() {", self.check, "}"])
        if self.package:
            lines.extend(["", "package() {", self.package, "}"])

        return "\n".join(lines)

    @classmethod
    def from_script(cls, script: str) -> "PKGBUILD":
        """Parse PKGBUILD script."""
        # Simple parser - in production would use proper shell parsing
        pkg = cls(pkgname="", pkgver="0.0.0")

        lines = script.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("pkgname="):
                pkg.pkgname = line.split("=", 1)[1].strip('"\'')
            elif line.startswith("pkgver="):
                pkg.pkgver = line.split("=", 1)[1].strip('"\'')
            elif line.startswith("pkgrel="):
                pkg.pkgrel = int(line.split("=", 1)[1])
            elif line.startswith("pkgdesc="):
                pkg.pkgdesc = line.split("=", 1)[1].strip('"\'')

        return pkg


# =============================================================================
# User and Authentication
# =============================================================================

class UserRole(Enum):
    """User roles in the registry."""
    USER = "user"
    MAINTAINER = "maintainer"
    TRUSTED_USER = "trusted_user"
    DEVELOPER = "developer"
    ADMIN = "admin"


@dataclass
class User:
    """A registry user."""

    username: str
    email: str
    role: UserRole = UserRole.USER

    # Profile
    real_name: Optional[str] = None
    homepage: Optional[str] = None
    pgp_key: Optional[str] = None

    # Stats
    packages_maintained: int = 0
    votes_cast: int = 0

    # Dates
    registered: datetime = field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None

    # Preferences
    notify_out_of_date: bool = True
    show_email: bool = False


# =============================================================================
# Search and Queries
# =============================================================================

class SearchField(Enum):
    """Fields to search by."""
    NAME = "name"
    NAME_DESC = "name-desc"
    MAINTAINER = "maintainer"
    DEPENDS = "depends"
    MAKEDEPENDS = "makedepends"
    OPTDEPENDS = "optdepends"
    CHECKDEPENDS = "checkdepends"
    KEYWORDS = "keywords"


class SortBy(Enum):
    """Sort options."""
    NAME = "name"
    VOTES = "votes"
    POPULARITY = "popularity"
    LAST_UPDATED = "last_updated"
    FIRST_SUBMITTED = "first_submitted"


@dataclass
class SearchQuery:
    """Search query for packages."""

    keywords: str = ""
    field: SearchField = SearchField.NAME_DESC
    sort_by: SortBy = SortBy.POPULARITY
    sort_asc: bool = False

    # Filters
    package_types: List[PackageType] = field(default_factory=list)
    sources: List[PackageSource] = field(default_factory=list)
    maintainer: Optional[str] = None
    flagged: Optional[bool] = None
    outdated: Optional[bool] = None

    # Pagination
    page: int = 1
    per_page: int = 50


@dataclass
class SearchResult:
    """Search result."""

    packages: List[Package]
    total: int
    page: int
    per_page: int
    query: SearchQuery


# =============================================================================
# Registry Backend
# =============================================================================

class RegistryBackend(ABC):
    """Abstract backend for package storage."""

    @abstractmethod
    async def get_package(self, name: str) -> Optional[Package]:
        pass

    @abstractmethod
    async def save_package(self, package: Package):
        pass

    @abstractmethod
    async def delete_package(self, name: str):
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResult:
        pass

    @abstractmethod
    async def list_all(self) -> List[str]:
        pass


class FileRegistryBackend(RegistryBackend):
    """File-based registry backend."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._packages_path = self.path / "packages"
        self._packages_path.mkdir(exist_ok=True)
        self._index_path = self.path / "index.json"
        self._index: Dict[str, Dict] = {}
        self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            with open(self._index_path) as f:
                self._index = json.load(f)

    def _save_index(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    async def get_package(self, name: str) -> Optional[Package]:
        pkg_path = self._packages_path / f"{name}.json"
        if not pkg_path.exists():
            return None

        with open(pkg_path) as f:
            data = json.load(f)

        pkg = Package(
            name=data["name"],
            package_type=PackageType(data["type"]),
            description=data.get("description", ""),
            source=PackageSource(data.get("source", "community")),
            status=PackageStatus(data.get("status", "active")),
            maintainer=data.get("maintainer"),
            votes=data.get("votes", 0),
            popularity=data.get("popularity", 0.0),
            downloads=data.get("downloads", 0),
            latest_version=data.get("latestVersion"),
        )

        for ver_str, ver_data in data.get("versions", {}).items():
            pkg.versions[ver_str] = PackageVersion(
                version=ver_data["version"],
                epoch=ver_data.get("epoch", 0),
                pkgrel=ver_data.get("pkgrel", 1),
                depends=ver_data.get("depends", []),
                makedepends=ver_data.get("makedepends", []),
            )

        return pkg

    async def save_package(self, package: Package):
        pkg_path = self._packages_path / f"{package.name}.json"
        with open(pkg_path, "w") as f:
            json.dump(package.to_dict(), f, indent=2)

        self._index[package.name] = {
            "type": package.package_type.value,
            "description": package.description,
            "votes": package.votes,
            "popularity": package.popularity,
        }
        self._save_index()

    async def delete_package(self, name: str):
        pkg_path = self._packages_path / f"{name}.json"
        if pkg_path.exists():
            pkg_path.unlink()
        self._index.pop(name, None)
        self._save_index()

    async def search(self, query: SearchQuery) -> SearchResult:
        results = []

        for name in self._index:
            pkg = await self.get_package(name)
            if not pkg:
                continue

            # Apply filters
            if query.package_types and pkg.package_type not in query.package_types:
                continue
            if query.sources and pkg.source not in query.sources:
                continue
            if query.maintainer and pkg.maintainer != query.maintainer:
                continue
            if query.outdated is not None and pkg.out_of_date != query.outdated:
                continue

            # Keyword search
            if query.keywords:
                keywords = query.keywords.lower()
                if query.field == SearchField.NAME:
                    if keywords not in pkg.name.lower():
                        continue
                elif query.field == SearchField.NAME_DESC:
                    if keywords not in pkg.name.lower() and keywords not in pkg.description.lower():
                        continue
                elif query.field == SearchField.MAINTAINER:
                    if not pkg.maintainer or keywords not in pkg.maintainer.lower():
                        continue

            results.append(pkg)

        # Sort
        if query.sort_by == SortBy.NAME:
            results.sort(key=lambda p: p.name, reverse=not query.sort_asc)
        elif query.sort_by == SortBy.VOTES:
            results.sort(key=lambda p: p.votes, reverse=not query.sort_asc)
        elif query.sort_by == SortBy.POPULARITY:
            results.sort(key=lambda p: p.popularity, reverse=not query.sort_asc)

        # Paginate
        total = len(results)
        start = (query.page - 1) * query.per_page
        end = start + query.per_page
        results = results[start:end]

        return SearchResult(
            packages=results,
            total=total,
            page=query.page,
            per_page=query.per_page,
            query=query,
        )

    async def list_all(self) -> List[str]:
        return list(self._index.keys())


# =============================================================================
# Dependency Resolver
# =============================================================================

class DependencyError(Exception):
    """Error during dependency resolution."""
    pass


@dataclass
class ResolvedDependency:
    """A resolved dependency."""

    package: Package
    version: PackageVersion
    is_direct: bool = True
    is_makedep: bool = False
    is_optional: bool = False


class DependencyResolver:
    """Resolves package dependencies."""

    def __init__(self, registry: "AgentRegistry"):
        self.registry = registry
        self._resolved: Dict[str, ResolvedDependency] = {}
        self._visited: Set[str] = set()

    async def resolve(
        self,
        package_name: str,
        include_makedeps: bool = False,
        include_optdeps: bool = False,
    ) -> List[ResolvedDependency]:
        """Resolve all dependencies for a package."""
        self._resolved = {}
        self._visited = set()

        await self._resolve_recursive(
            package_name,
            is_direct=True,
            include_makedeps=include_makedeps,
            include_optdeps=include_optdeps,
        )

        # Return in topological order
        return self._topological_sort()

    async def _resolve_recursive(
        self,
        package_name: str,
        is_direct: bool,
        is_makedep: bool = False,
        is_optional: bool = False,
        include_makedeps: bool = False,
        include_optdeps: bool = False,
    ):
        if package_name in self._visited:
            return
        self._visited.add(package_name)

        # Parse version constraint
        name, version_constraint = self._parse_constraint(package_name)

        # Get package
        pkg = await self.registry.get_package(name)
        if not pkg:
            if is_optional:
                return
            raise DependencyError(f"Package not found: {name}")

        # Get version
        version = self._select_version(pkg, version_constraint)
        if not version:
            raise DependencyError(f"No matching version for {name} {version_constraint}")

        self._resolved[name] = ResolvedDependency(
            package=pkg,
            version=version,
            is_direct=is_direct,
            is_makedep=is_makedep,
            is_optional=is_optional,
        )

        # Resolve dependencies
        for dep in version.depends:
            await self._resolve_recursive(
                dep,
                is_direct=False,
                include_makedeps=include_makedeps,
                include_optdeps=include_optdeps,
            )

        if include_makedeps:
            for dep in version.makedepends:
                await self._resolve_recursive(
                    dep,
                    is_direct=False,
                    is_makedep=True,
                    include_makedeps=include_makedeps,
                    include_optdeps=include_optdeps,
                )

        if include_optdeps:
            for dep in version.optdepends:
                await self._resolve_recursive(
                    dep,
                    is_direct=False,
                    is_optional=True,
                    include_makedeps=include_makedeps,
                    include_optdeps=include_optdeps,
                )

    def _parse_constraint(self, spec: str) -> Tuple[str, Optional[str]]:
        """Parse package name and version constraint."""
        for op in [">=", "<=", ">", "<", "="]:
            if op in spec:
                parts = spec.split(op, 1)
                return parts[0], f"{op}{parts[1]}"
        return spec, None

    def _select_version(
        self,
        package: Package,
        constraint: Optional[str]
    ) -> Optional[PackageVersion]:
        """Select best version matching constraint."""
        if not constraint:
            return package.get_latest()

        # Simple constraint matching
        # In production, would use proper version comparison
        for ver_str, version in sorted(
            package.versions.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if self._matches_constraint(ver_str, constraint):
                return version

        return None

    def _matches_constraint(self, version: str, constraint: str) -> bool:
        """Check if version matches constraint."""
        # Simple matching - would need proper semver in production
        if constraint.startswith(">="):
            return version >= constraint[2:]
        elif constraint.startswith("<="):
            return version <= constraint[2:]
        elif constraint.startswith(">"):
            return version > constraint[1:]
        elif constraint.startswith("<"):
            return version < constraint[1:]
        elif constraint.startswith("="):
            return version == constraint[1:]
        return True

    def _topological_sort(self) -> List[ResolvedDependency]:
        """Sort dependencies topologically."""
        result = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            dep = self._resolved.get(name)
            if dep:
                for subdep in dep.version.depends:
                    subdep_name, _ = self._parse_constraint(subdep)
                    visit(subdep_name)
                result.append(dep)

        for name in self._resolved:
            visit(name)

        return result


# =============================================================================
# Package Builder
# =============================================================================

class BuildError(Exception):
    """Error during package build."""
    pass


@dataclass
class BuildResult:
    """Result of a package build."""

    success: bool
    package_name: str
    version: str
    output_path: Optional[Path] = None
    logs: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


class PackageBuilder:
    """Builds packages from PKGBUILDs."""

    def __init__(self, build_dir: Optional[Path] = None):
        self.build_dir = build_dir or Path(tempfile.gettempdir()) / "bbx-build"
        self.build_dir.mkdir(parents=True, exist_ok=True)

    async def build(
        self,
        pkgbuild: PKGBUILD,
        install: bool = False
    ) -> BuildResult:
        """Build a package from PKGBUILD."""
        start_time = datetime.now()
        logs = []

        # Create build directory
        pkg_build_dir = self.build_dir / pkgbuild.pkgname
        pkg_build_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Fetch sources
            logs.append(f"Fetching sources for {pkgbuild.pkgname}...")
            await self._fetch_sources(pkgbuild, pkg_build_dir)

            # 2. Run prepare
            if pkgbuild.prepare:
                logs.append("Running prepare()...")
                await self._run_function(pkgbuild.prepare, pkg_build_dir)

            # 3. Run build
            if pkgbuild.build:
                logs.append("Running build()...")
                await self._run_function(pkgbuild.build, pkg_build_dir)

            # 4. Run check
            if pkgbuild.check:
                logs.append("Running check()...")
                await self._run_function(pkgbuild.check, pkg_build_dir)

            # 5. Run package
            if pkgbuild.package:
                logs.append("Running package()...")
                await self._run_function(pkgbuild.package, pkg_build_dir)

            # 6. Create package archive
            output_path = await self._create_package(pkgbuild, pkg_build_dir)

            duration = (datetime.now() - start_time).total_seconds()
            logs.append(f"Build completed in {duration:.2f}s")

            return BuildResult(
                success=True,
                package_name=pkgbuild.pkgname,
                version=pkgbuild.pkgver,
                output_path=output_path,
                logs=logs,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logs.append(f"Build failed: {e}")

            return BuildResult(
                success=False,
                package_name=pkgbuild.pkgname,
                version=pkgbuild.pkgver,
                logs=logs,
                duration_seconds=duration,
                error=str(e),
            )

    async def _fetch_sources(self, pkgbuild: PKGBUILD, build_dir: Path):
        """Fetch and verify sources."""
        for i, source in enumerate(pkgbuild.source):
            # Download source
            if source.startswith("http://") or source.startswith("https://"):
                # Would use aiohttp in production
                pass
            elif source.startswith("git+"):
                # Clone git repo
                pass

            # Verify checksum
            if i < len(pkgbuild.sha256sums):
                expected = pkgbuild.sha256sums[i]
                if expected != "SKIP":
                    # Verify hash
                    pass

    async def _run_function(self, script: str, work_dir: Path):
        """Run a build function."""
        # In production, would run in isolated environment
        proc = await asyncio.create_subprocess_shell(
            script,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise BuildError(f"Build function failed: {stderr.decode()}")

    async def _create_package(self, pkgbuild: PKGBUILD, build_dir: Path) -> Path:
        """Create package archive."""
        pkg_dir = build_dir / "pkg"
        output_name = f"{pkgbuild.pkgname}-{pkgbuild.pkgver}-{pkgbuild.pkgrel}.bbx.tar.zst"
        output_path = self.build_dir / output_name

        # Would create actual archive in production
        output_path.touch()

        return output_path


# =============================================================================
# Agent Registry
# =============================================================================

class AgentRegistry:
    """
    Central BBX Agent Registry.

    Provides:
    - Package management (upload, search, install)
    - Dependency resolution
    - Build-from-source support
    - User management
    - Voting and statistics
    """

    def __init__(
        self,
        backend: Optional[RegistryBackend] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.backend = backend or FileRegistryBackend(
            Path.home() / ".bbx" / "registry"
        )
        self.cache_dir = cache_dir or Path.home() / ".bbx" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.resolver = DependencyResolver(self)
        self.builder = PackageBuilder()

        # User session
        self._current_user: Optional[User] = None

        # Statistics
        self._stats = {
            "packages_installed": 0,
            "packages_built": 0,
            "searches_performed": 0,
        }

    async def get_package(self, name: str) -> Optional[Package]:
        """Get a package by name."""
        return await self.backend.get_package(name)

    async def search(
        self,
        keywords: str = "",
        field: SearchField = SearchField.NAME_DESC,
        sort_by: SortBy = SortBy.POPULARITY,
        **filters
    ) -> SearchResult:
        """Search for packages."""
        query = SearchQuery(
            keywords=keywords,
            field=field,
            sort_by=sort_by,
            **filters
        )
        self._stats["searches_performed"] += 1
        return await self.backend.search(query)

    async def install(
        self,
        package_name: str,
        version: Optional[str] = None,
        build_from_source: bool = False,
    ) -> bool:
        """Install a package."""
        # Resolve dependencies
        deps = await self.resolver.resolve(
            package_name,
            include_makedeps=build_from_source,
        )

        # Install each dependency
        for dep in deps:
            if build_from_source:
                # Build from source
                pkgbuild = await self._get_pkgbuild(dep.package.name)
                if pkgbuild:
                    result = await self.builder.build(pkgbuild, install=True)
                    if not result.success:
                        logger.error(f"Failed to build {dep.package.name}: {result.error}")
                        return False
            else:
                # Install binary
                await self._install_binary(dep.package, dep.version)

        self._stats["packages_installed"] += 1
        return True

    async def _get_pkgbuild(self, package_name: str) -> Optional[PKGBUILD]:
        """Get PKGBUILD for a package."""
        pkg = await self.get_package(package_name)
        if not pkg:
            return None

        # Would fetch from package source
        return PKGBUILD(
            pkgname=package_name,
            pkgver=pkg.latest_version or "0.0.0",
            pkgdesc=pkg.description,
        )

    async def _install_binary(self, package: Package, version: PackageVersion):
        """Install pre-built binary package."""
        # Would download and extract package
        logger.info(f"Installing {package.name} {version.version}")

    async def upload(
        self,
        package: Package,
        source_path: Path,
    ) -> bool:
        """Upload a new package or version."""
        if not self._current_user:
            raise PermissionError("Must be logged in to upload")

        # Validate package
        if not package.name or not package.latest_version:
            raise ValueError("Package must have name and version")

        # Set maintainer
        package.maintainer = self._current_user.username
        package.first_submitted = package.first_submitted or datetime.now()

        # Save
        await self.backend.save_package(package)

        return True

    async def vote(self, package_name: str) -> bool:
        """Vote for a package."""
        if not self._current_user:
            raise PermissionError("Must be logged in to vote")

        pkg = await self.get_package(package_name)
        if not pkg:
            return False

        pkg.votes += 1
        await self.backend.save_package(pkg)
        return True

    async def flag_out_of_date(
        self,
        package_name: str,
        reason: str = ""
    ) -> bool:
        """Flag a package as out of date."""
        pkg = await self.get_package(package_name)
        if not pkg:
            return False

        pkg.out_of_date = True
        pkg.out_of_date_since = datetime.now()
        pkg.flagged_reason = reason

        await self.backend.save_package(pkg)
        return True

    async def adopt(self, package_name: str) -> bool:
        """Adopt an orphaned package."""
        if not self._current_user:
            raise PermissionError("Must be logged in to adopt")

        pkg = await self.get_package(package_name)
        if not pkg:
            return False

        if pkg.status != PackageStatus.ORPHANED:
            return False

        pkg.maintainer = self._current_user.username
        pkg.status = PackageStatus.ACTIVE

        await self.backend.save_package(pkg)
        return True

    async def disown(self, package_name: str) -> bool:
        """Disown a maintained package."""
        if not self._current_user:
            raise PermissionError("Must be logged in to disown")

        pkg = await self.get_package(package_name)
        if not pkg:
            return False

        if pkg.maintainer != self._current_user.username:
            return False

        pkg.maintainer = None
        pkg.status = PackageStatus.ORPHANED

        await self.backend.save_package(pkg)
        return True

    def login(self, user: User):
        """Set current user session."""
        self._current_user = user
        user.last_seen = datetime.now()

    def logout(self):
        """Clear current user session."""
        self._current_user = None

    async def sync(self):
        """Sync with remote registry."""
        # Would sync package database
        pass

    async def update(self) -> List[Package]:
        """Check for package updates."""
        updates = []

        # Would compare installed vs available versions
        # and return list of packages with updates

        return updates

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            "current_user": self._current_user.username if self._current_user else None,
        }


# =============================================================================
# Registry Helper (like yay/paru)
# =============================================================================

class RegistryHelper:
    """
    High-level helper for registry operations.

    Similar to yay/paru for AUR.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        self.registry = registry or AgentRegistry()

    async def search(self, query: str) -> List[Package]:
        """Search packages."""
        result = await self.registry.search(query)
        return result.packages

    async def info(self, package_name: str) -> Optional[Package]:
        """Get package info."""
        return await self.registry.get_package(package_name)

    async def install(
        self,
        *packages: str,
        build: bool = False,
        no_confirm: bool = False,
    ) -> bool:
        """Install packages."""
        for pkg in packages:
            success = await self.registry.install(pkg, build_from_source=build)
            if not success:
                return False
        return True

    async def update(self) -> List[Package]:
        """Check for updates."""
        return await self.registry.update()

    async def sync(self):
        """Sync database."""
        await self.registry.sync()


# =============================================================================
# Convenience Functions
# =============================================================================

async def search_registry(query: str) -> List[Package]:
    """Search the registry."""
    registry = AgentRegistry()
    result = await registry.search(query)
    return result.packages


async def install_package(name: str, build: bool = False) -> bool:
    """Install a package from registry."""
    registry = AgentRegistry()
    return await registry.install(name, build_from_source=build)


async def get_package_info(name: str) -> Optional[Package]:
    """Get package information."""
    registry = AgentRegistry()
    return await registry.get_package(name)
