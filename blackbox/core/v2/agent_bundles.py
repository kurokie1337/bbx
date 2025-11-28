# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Agent Bundles - Kali-Inspired Specialized Toolkits

Inspired by Kali Linux's tool categories, provides:
- Pre-configured agent collections for specific domains
- Task-oriented bundling (analysis, security, data, etc.)
- Metapackages with smart dependencies
- Profile-based activation
- Tool documentation and tutorials

Key concepts:
- Bundle: A collection of agents organized by purpose
- Profile: Active bundle configuration
- Category: Domain classification (like Kali menu structure)
- Tool: Individual agent within a bundle
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.bundles")


# =============================================================================
# Bundle Categories (Kali-inspired)
# =============================================================================

class BundleCategory(Enum):
    """Bundle categories inspired by Kali tool categories."""

    # Information Gathering
    INFORMATION_GATHERING = "information-gathering"
    RECONNAISSANCE = "reconnaissance"
    OSINT = "osint"

    # Analysis
    DATA_ANALYSIS = "data-analysis"
    CODE_ANALYSIS = "code-analysis"
    SECURITY_ANALYSIS = "security-analysis"
    LOG_ANALYSIS = "log-analysis"

    # Automation
    WORKFLOW_AUTOMATION = "workflow-automation"
    TESTING = "testing"
    CI_CD = "ci-cd"
    DEVOPS = "devops"

    # Development
    CODE_GENERATION = "code-generation"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"

    # Data
    DATABASE = "database"
    ETL = "etl"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"

    # AI/ML
    ML_OPS = "ml-ops"
    NLP = "nlp"
    VISION = "vision"
    TRAINING = "training"

    # Security
    VULNERABILITY_ASSESSMENT = "vulnerability-assessment"
    PENETRATION_TESTING = "penetration-testing"
    FORENSICS = "forensics"
    COMPLIANCE = "compliance"

    # Infrastructure
    CLOUD = "cloud"
    CONTAINERS = "containers"
    NETWORKING = "networking"
    MONITORING = "monitoring"

    # Productivity
    COMMUNICATION = "communication"
    PROJECT_MANAGEMENT = "project-management"
    KNOWLEDGE_MANAGEMENT = "knowledge-management"

    # Meta
    META = "meta"
    CORE = "core"


# =============================================================================
# Tool Definition
# =============================================================================

class ToolStatus(Enum):
    """Status of a tool in a bundle."""
    AVAILABLE = "available"
    INSTALLED = "installed"
    ACTIVE = "active"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"


@dataclass
class ToolDependency:
    """A tool dependency."""

    name: str
    version: Optional[str] = None
    optional: bool = False
    reason: str = ""


@dataclass
class Tool:
    """
    A tool within a bundle.

    Represents an individual agent/adapter that can be used.
    """

    # Identity
    name: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"

    # Classification
    category: BundleCategory = BundleCategory.CORE
    tags: List[str] = field(default_factory=list)

    # Agent reference
    agent_name: Optional[str] = None
    adapter_name: Optional[str] = None
    workflow_path: Optional[str] = None

    # Dependencies
    depends: List[ToolDependency] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    # Configuration
    default_config: Dict[str, Any] = field(default_factory=dict)
    config_schema: Optional[Dict[str, Any]] = None

    # Status
    status: ToolStatus = ToolStatus.AVAILABLE

    # Metadata
    author: str = ""
    homepage: str = ""
    documentation: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    usage_count: int = 0
    last_used: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "displayName": self.display_name,
            "description": self.description,
            "version": self.version,
            "category": self.category.value,
            "tags": self.tags,
            "agentName": self.agent_name,
            "adapterName": self.adapter_name,
            "workflowPath": self.workflow_path,
            "depends": [{"name": d.name, "version": d.version, "optional": d.optional}
                       for d in self.depends],
            "status": self.status.value,
            "author": self.author,
            "usageCount": self.usage_count,
        }


# =============================================================================
# Bundle Definition
# =============================================================================

class BundleType(Enum):
    """Types of bundles."""
    STANDARD = "standard"       # Regular tool bundle
    META = "meta"               # Metapackage (deps only)
    PROFILE = "profile"         # Activation profile
    CUSTOM = "custom"           # User-created bundle


@dataclass
class Bundle:
    """
    A bundle of related tools.

    Similar to Kali metapackages like kali-linux-large.
    """

    # Identity
    name: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"

    # Type and category
    bundle_type: BundleType = BundleType.STANDARD
    categories: List[BundleCategory] = field(default_factory=list)

    # Tools
    tools: Dict[str, Tool] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)

    # Dependencies on other bundles
    depends: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    # Configuration
    default_profile: Optional[str] = None
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metadata
    author: str = ""
    maintainers: List[str] = field(default_factory=list)
    homepage: str = ""
    documentation: str = ""

    # Statistics
    install_count: int = 0
    last_updated: Optional[datetime] = None

    def add_tool(self, tool: Tool, required: bool = True):
        """Add a tool to this bundle."""
        self.tools[tool.name] = tool
        if required:
            if tool.name not in self.required_tools:
                self.required_tools.append(tool.name)
        else:
            if tool.name not in self.optional_tools:
                self.optional_tools.append(tool.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(
        self,
        category: Optional[BundleCategory] = None,
        status: Optional[ToolStatus] = None,
    ) -> List[Tool]:
        """List tools with optional filtering."""
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.category == category]
        if status:
            tools = [t for t in tools if t.status == status]

        return tools

    def get_all_dependencies(self) -> Set[str]:
        """Get all tool dependencies recursively."""
        deps = set()

        def collect(tool_name: str):
            if tool_name in deps:
                return
            deps.add(tool_name)

            tool = self.tools.get(tool_name)
            if tool:
                for dep in tool.depends:
                    if not dep.optional:
                        collect(dep.name)

        for tool_name in self.required_tools:
            collect(tool_name)

        return deps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "displayName": self.display_name,
            "description": self.description,
            "version": self.version,
            "type": self.bundle_type.value,
            "categories": [c.value for c in self.categories],
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "requiredTools": self.required_tools,
            "optionalTools": self.optional_tools,
            "depends": self.depends,
            "author": self.author,
            "installCount": self.install_count,
        }


# =============================================================================
# Profile Management
# =============================================================================

@dataclass
class Profile:
    """
    An activation profile for bundles.

    Defines which tools and configurations are active.
    """

    name: str
    description: str = ""

    # Active bundles and tools
    bundles: List[str] = field(default_factory=list)
    tools: Dict[str, bool] = field(default_factory=dict)  # tool_name -> enabled

    # Configuration overrides
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def enable_tool(self, tool_name: str):
        """Enable a tool in this profile."""
        self.tools[tool_name] = True
        self.updated_at = datetime.now()

    def disable_tool(self, tool_name: str):
        """Disable a tool in this profile."""
        self.tools[tool_name] = False
        self.updated_at = datetime.now()

    def set_tool_config(self, tool_name: str, config: Dict[str, Any]):
        """Set configuration for a tool."""
        self.tool_configs[tool_name] = config
        self.updated_at = datetime.now()

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return self.tools.get(tool_name, True)  # Enabled by default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "bundles": self.bundles,
            "tools": self.tools,
            "toolConfigs": self.tool_configs,
            "envVars": self.env_vars,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
        }


# =============================================================================
# Bundle Manager
# =============================================================================

class BundleManager:
    """
    Central manager for agent bundles.

    Handles:
    - Bundle registration and discovery
    - Profile management
    - Tool activation/deactivation
    - Dependency resolution
    """

    def __init__(self, bundles_path: Optional[Path] = None):
        self.bundles_path = bundles_path or (Path.home() / ".bbx" / "bundles")
        self.bundles_path.mkdir(parents=True, exist_ok=True)

        # Registered bundles
        self._bundles: Dict[str, Bundle] = {}

        # Profiles
        self._profiles: Dict[str, Profile] = {}
        self._active_profile: Optional[str] = None

        # Tool registry (flat view of all tools)
        self._tools: Dict[str, Tool] = {}

        # Statistics
        self._stats = {
            "bundles_loaded": 0,
            "tools_available": 0,
            "tools_active": 0,
        }

        # Load built-in bundles
        self._load_builtin_bundles()

    def _load_builtin_bundles(self):
        """Load built-in bundles."""
        # Core bundle
        core = Bundle(
            name="bbx-core",
            display_name="BBX Core",
            description="Core BBX tools and agents",
            bundle_type=BundleType.META,
            categories=[BundleCategory.CORE],
        )
        core.add_tool(Tool(
            name="shell",
            display_name="Shell Agent",
            description="Execute shell commands",
            category=BundleCategory.CORE,
            adapter_name="shell",
        ))
        core.add_tool(Tool(
            name="llm",
            display_name="LLM Agent",
            description="Language model interactions",
            category=BundleCategory.CORE,
            adapter_name="llm",
        ))
        core.add_tool(Tool(
            name="http",
            display_name="HTTP Agent",
            description="HTTP requests",
            category=BundleCategory.CORE,
            adapter_name="http",
        ))
        self.register_bundle(core)

        # Data Analysis bundle
        data = Bundle(
            name="bbx-data",
            display_name="BBX Data Analysis",
            description="Tools for data analysis and manipulation",
            categories=[BundleCategory.DATA_ANALYSIS, BundleCategory.VISUALIZATION],
        )
        data.add_tool(Tool(
            name="csv-processor",
            display_name="CSV Processor",
            description="Process and analyze CSV files",
            category=BundleCategory.DATA_ANALYSIS,
        ))
        data.add_tool(Tool(
            name="json-transformer",
            display_name="JSON Transformer",
            description="Transform and query JSON data",
            category=BundleCategory.DATA_ANALYSIS,
        ))
        data.add_tool(Tool(
            name="chart-generator",
            display_name="Chart Generator",
            description="Generate charts and visualizations",
            category=BundleCategory.VISUALIZATION,
        ))
        self.register_bundle(data)

        # Development bundle
        dev = Bundle(
            name="bbx-dev",
            display_name="BBX Development",
            description="Tools for software development",
            categories=[BundleCategory.CODE_GENERATION, BundleCategory.DEBUGGING],
        )
        dev.add_tool(Tool(
            name="code-generator",
            display_name="Code Generator",
            description="Generate code from specifications",
            category=BundleCategory.CODE_GENERATION,
        ))
        dev.add_tool(Tool(
            name="debugger",
            display_name="Debug Assistant",
            description="Help debug code issues",
            category=BundleCategory.DEBUGGING,
        ))
        dev.add_tool(Tool(
            name="refactor",
            display_name="Refactoring Tool",
            description="Automated code refactoring",
            category=BundleCategory.REFACTORING,
        ))
        self.register_bundle(dev)

        # Security bundle
        security = Bundle(
            name="bbx-security",
            display_name="BBX Security",
            description="Security analysis and testing tools",
            categories=[
                BundleCategory.SECURITY_ANALYSIS,
                BundleCategory.VULNERABILITY_ASSESSMENT,
            ],
        )
        security.add_tool(Tool(
            name="vuln-scanner",
            display_name="Vulnerability Scanner",
            description="Scan for security vulnerabilities",
            category=BundleCategory.VULNERABILITY_ASSESSMENT,
        ))
        security.add_tool(Tool(
            name="code-audit",
            display_name="Code Auditor",
            description="Security code review",
            category=BundleCategory.SECURITY_ANALYSIS,
        ))
        security.add_tool(Tool(
            name="compliance-checker",
            display_name="Compliance Checker",
            description="Check compliance requirements",
            category=BundleCategory.COMPLIANCE,
        ))
        self.register_bundle(security)

        # DevOps bundle
        devops = Bundle(
            name="bbx-devops",
            display_name="BBX DevOps",
            description="DevOps and infrastructure tools",
            categories=[BundleCategory.DEVOPS, BundleCategory.CI_CD, BundleCategory.CLOUD],
        )
        devops.add_tool(Tool(
            name="ci-runner",
            display_name="CI Runner",
            description="Run CI/CD pipelines",
            category=BundleCategory.CI_CD,
        ))
        devops.add_tool(Tool(
            name="container-manager",
            display_name="Container Manager",
            description="Manage containers",
            category=BundleCategory.CONTAINERS,
        ))
        devops.add_tool(Tool(
            name="cloud-deployer",
            display_name="Cloud Deployer",
            description="Deploy to cloud providers",
            category=BundleCategory.CLOUD,
        ))
        self.register_bundle(devops)

        # Create default profile
        default_profile = Profile(
            name="default",
            description="Default profile with core tools",
            bundles=["bbx-core"],
        )
        self.register_profile(default_profile)
        self._active_profile = "default"

    def register_bundle(self, bundle: Bundle):
        """Register a bundle."""
        self._bundles[bundle.name] = bundle

        # Register all tools
        for tool in bundle.tools.values():
            self._tools[tool.name] = tool

        self._stats["bundles_loaded"] = len(self._bundles)
        self._stats["tools_available"] = len(self._tools)

        logger.debug(f"Registered bundle: {bundle.name} with {len(bundle.tools)} tools")

    def register_profile(self, profile: Profile):
        """Register a profile."""
        self._profiles[profile.name] = profile

    def get_bundle(self, name: str) -> Optional[Bundle]:
        """Get a bundle by name."""
        return self._bundles.get(name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_bundles(
        self,
        category: Optional[BundleCategory] = None
    ) -> List[Bundle]:
        """List all bundles."""
        bundles = list(self._bundles.values())

        if category:
            bundles = [b for b in bundles if category in b.categories]

        return bundles

    def list_tools(
        self,
        category: Optional[BundleCategory] = None,
        bundle: Optional[str] = None,
    ) -> List[Tool]:
        """List all tools."""
        if bundle:
            b = self.get_bundle(bundle)
            if b:
                return b.list_tools(category=category)
            return []

        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        return tools

    def list_categories(self) -> List[BundleCategory]:
        """List all categories with tools."""
        categories = set()
        for tool in self._tools.values():
            categories.add(tool.category)
        return sorted(categories, key=lambda c: c.value)

    def activate_profile(self, name: str) -> bool:
        """Activate a profile."""
        if name not in self._profiles:
            return False

        self._active_profile = name
        profile = self._profiles[name]

        # Update tool status based on profile
        active_count = 0
        for tool_name, tool in self._tools.items():
            if profile.is_tool_enabled(tool_name):
                tool.status = ToolStatus.ACTIVE
                active_count += 1
            else:
                tool.status = ToolStatus.DISABLED

        self._stats["tools_active"] = active_count
        logger.info(f"Activated profile: {name} ({active_count} tools active)")

        return True

    def get_active_profile(self) -> Optional[Profile]:
        """Get the active profile."""
        if self._active_profile:
            return self._profiles.get(self._active_profile)
        return None

    def create_profile(
        self,
        name: str,
        bundles: Optional[List[str]] = None,
        description: str = "",
    ) -> Profile:
        """Create a new profile."""
        profile = Profile(
            name=name,
            description=description,
            bundles=bundles or [],
        )

        # Enable all tools from specified bundles
        for bundle_name in profile.bundles:
            bundle = self.get_bundle(bundle_name)
            if bundle:
                for tool_name in bundle.required_tools:
                    profile.enable_tool(tool_name)

        self.register_profile(profile)
        return profile

    def install_bundle(self, name: str) -> bool:
        """Install a bundle (download and register)."""
        # Would fetch from registry
        bundle = self.get_bundle(name)
        if bundle:
            bundle.install_count += 1
            return True
        return False

    def create_custom_bundle(
        self,
        name: str,
        tools: List[str],
        display_name: Optional[str] = None,
        description: str = "",
    ) -> Bundle:
        """Create a custom bundle from existing tools."""
        bundle = Bundle(
            name=name,
            display_name=display_name or name,
            description=description,
            bundle_type=BundleType.CUSTOM,
        )

        for tool_name in tools:
            tool = self.get_tool(tool_name)
            if tool:
                bundle.add_tool(tool)

        self.register_bundle(bundle)
        return bundle

    async def execute_tool(
        self,
        tool_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute a tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        if tool.status == ToolStatus.DISABLED:
            raise RuntimeError(f"Tool is disabled: {tool_name}")

        # Get config from profile
        profile = self.get_active_profile()
        if profile and tool_name in profile.tool_configs:
            merged_config = {**profile.tool_configs[tool_name], **(config or {})}
        else:
            merged_config = config or {}

        # Update statistics
        tool.usage_count += 1
        tool.last_used = datetime.now()

        # Execute based on tool type
        if tool.adapter_name:
            # Execute via adapter
            return await self._execute_adapter(tool.adapter_name, inputs, merged_config)
        elif tool.workflow_path:
            # Execute workflow
            return await self._execute_workflow(tool.workflow_path, inputs, merged_config)
        else:
            raise RuntimeError(f"Tool has no executable: {tool_name}")

    async def _execute_adapter(
        self,
        adapter_name: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Any:
        """Execute via adapter."""
        # Would integrate with adapter registry
        return {"adapter": adapter_name, "inputs": inputs, "status": "executed"}

    async def _execute_workflow(
        self,
        workflow_path: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Any:
        """Execute workflow."""
        # Would integrate with workflow runtime
        return {"workflow": workflow_path, "inputs": inputs, "status": "executed"}

    def save_profile(self, profile: Profile, path: Optional[Path] = None):
        """Save profile to disk."""
        path = path or (self.bundles_path / "profiles" / f"{profile.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    def load_profile(self, path: Path) -> Profile:
        """Load profile from disk."""
        with open(path) as f:
            data = json.load(f)

        return Profile(
            name=data["name"],
            description=data.get("description", ""),
            bundles=data.get("bundles", []),
            tools=data.get("tools", {}),
            tool_configs=data.get("toolConfigs", {}),
            env_vars=data.get("envVars", {}),
        )

    def export_bundle(self, name: str, path: Path):
        """Export a bundle to disk."""
        bundle = self.get_bundle(name)
        if not bundle:
            raise ValueError(f"Bundle not found: {name}")

        with open(path, "w") as f:
            json.dump(bundle.to_dict(), f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get bundle manager statistics."""
        return {
            **self._stats,
            "active_profile": self._active_profile,
            "profiles_count": len(self._profiles),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_bundle_manager() -> BundleManager:
    """Get the global bundle manager."""
    if not hasattr(get_bundle_manager, "_instance"):
        get_bundle_manager._instance = BundleManager()
    return get_bundle_manager._instance


def list_available_bundles() -> List[str]:
    """List all available bundle names."""
    manager = get_bundle_manager()
    return [b.name for b in manager.list_bundles()]


def list_available_tools(
    category: Optional[str] = None
) -> List[str]:
    """List all available tool names."""
    manager = get_bundle_manager()
    cat = BundleCategory(category) if category else None
    return [t.name for t in manager.list_tools(category=cat)]


async def execute_tool(
    tool_name: str,
    inputs: Dict[str, Any],
) -> Any:
    """Execute a tool by name."""
    manager = get_bundle_manager()
    return await manager.execute_tool(tool_name, inputs)
