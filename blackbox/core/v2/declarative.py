# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Declarative Configuration - NixOS-inspired infrastructure as code.

This module provides declarative configuration for BBX agent infrastructure,
enabling reproducible, versionable, and rollback-capable agent deployments.

Key concepts from NixOS:
- Declarative: Entire system described in configuration files
- Reproducible: Same config = same system
- Atomic: Updates are atomic, with instant rollback
- Generations: Version history of configurations

Example usage:
    config = BBXConfig.from_file("bbx.config.yaml")
    manager = DeclarativeManager(config)
    await manager.apply()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("bbx.declarative")


# =============================================================================
# Configuration Models
# =============================================================================


@dataclass
class AdapterConfig:
    """Configuration for an adapter"""
    name: str
    version: str = "latest"
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class HookConfig:
    """Configuration for a hook"""
    path: str  # Path to hook file
    enabled: bool = True
    priority: int = 0


@dataclass
class QuotaConfig:
    """Resource quotas for an agent"""
    # Execution limits
    max_concurrent_steps: int = 10
    max_execution_time_seconds: int = 300

    # Memory limits
    max_context_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_state_size_bytes: int = 10 * 1024 * 1024

    # I/O limits
    max_http_requests_per_minute: int = 100
    max_file_operations_per_minute: int = 1000

    # Cost limits (for LLM calls)
    max_tokens_per_hour: int = 100000


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    id: str
    name: str
    type: str = "worker"  # worker | coordinator | specialized

    # Components
    adapters: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    workflows: List[str] = field(default_factory=list)

    # Resources
    quotas: Optional[QuotaConfig] = None

    # Behavior
    enabled: bool = True
    auto_restart: bool = True

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class StateConfig:
    """Configuration for state storage"""
    backend: str = "sqlite"  # sqlite | postgres | redis
    path: str = "./data/state.db"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecretConfig:
    """Configuration for a secret"""
    name: str
    source: str = "env"  # env | vault | file
    path: Optional[str] = None  # For vault/file sources
    env_var: Optional[str] = None  # For env source


@dataclass
class BBXConfig:
    """
    Complete BBX configuration.

    Like NixOS configuration.nix - single source of truth for entire system.
    """
    version: str = "2.0"

    # Agents
    agents: Dict[str, AgentConfig] = field(default_factory=dict)

    # Global settings
    state: StateConfig = field(default_factory=StateConfig)
    secrets: Dict[str, SecretConfig] = field(default_factory=dict)

    # Global adapters and hooks
    global_adapters: List[AdapterConfig] = field(default_factory=list)
    global_hooks: List[HookConfig] = field(default_factory=list)

    # Metadata
    name: str = "bbx-config"
    description: str = ""

    @classmethod
    def from_file(cls, path: str) -> BBXConfig:
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BBXConfig:
        """Create configuration from dictionary"""
        config = cls()
        config.version = data.get("version", "2.0")
        config.name = data.get("name", "bbx-config")
        config.description = data.get("description", "")

        # Parse agents (new format)
        for agent_id, agent_data in data.get("agents", {}).items():
            quotas = None
            if "quotas" in agent_data:
                quotas = QuotaConfig(**agent_data["quotas"])

            config.agents[agent_id] = AgentConfig(
                id=agent_id,
                name=agent_data.get("name", agent_id),
                type=agent_data.get("type", "worker"),
                adapters=agent_data.get("adapters", []),
                hooks=agent_data.get("hooks", []),
                workflows=agent_data.get("workflows", []),
                quotas=quotas,
                enabled=agent_data.get("enabled", True),
                description=agent_data.get("description", ""),
                tags=agent_data.get("tags", []),
            )

        # Backward compatibility: support "agent" (singular) format
        if "agent" in data and not config.agents:
            agent_data = data["agent"]
            quotas = None
            if "quotas" in data:
                quotas = QuotaConfig(**data["quotas"])
            elif "quotas" in agent_data:
                quotas = QuotaConfig(**agent_data["quotas"])

            agent_name = agent_data.get("name", "default")
            config.agents[agent_name] = AgentConfig(
                id=agent_name,
                name=agent_name,
                type=agent_data.get("type", "worker"),
                adapters=list(data.get("adapters", {}).keys()),
                hooks=list(data.get("hooks", {}).keys()),
                workflows=agent_data.get("workflows", []),
                quotas=quotas,
                enabled=agent_data.get("enabled", True),
                description=agent_data.get("description", ""),
                tags=agent_data.get("tags", []),
            )

        # Parse state config
        if "state" in data:
            config.state = StateConfig(**data["state"])

        # Parse secrets
        for secret_name, secret_data in data.get("secrets", {}).items():
            config.secrets[secret_name] = SecretConfig(
                name=secret_name,
                **secret_data
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "type": agent.type,
                    "adapters": agent.adapters,
                    "hooks": agent.hooks,
                    "workflows": agent.workflows,
                    "enabled": agent.enabled,
                    "description": agent.description,
                    "tags": agent.tags,
                }
                for agent_id, agent in self.agents.items()
            },
            "state": {
                "backend": self.state.backend,
                "path": self.state.path,
            },
        }

    def get_hash(self) -> str:
        """Get configuration hash for change detection"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        if not self.agents:
            errors.append("No agents defined in configuration")
        return errors

    # Backward compatibility properties
    @property
    def agent(self) -> Optional[AgentConfig]:
        """Get first agent (backward compatibility)"""
        if self.agents:
            return next(iter(self.agents.values()))
        return None

    @property
    def quotas(self) -> Optional[QuotaConfig]:
        """Get first agent's quotas (backward compatibility)"""
        if self.agent and self.agent.quotas:
            return self.agent.quotas
        return QuotaConfig()

    @property
    def adapters(self) -> Dict[str, AdapterConfig]:
        """Get adapters from first agent (backward compatibility)"""
        return {a: AdapterConfig(enabled=True) for a in (self.agent.adapters if self.agent else [])}

    @property
    def hooks(self) -> Dict[str, HookConfig]:
        """Get hooks from first agent (backward compatibility)"""
        return {h: HookConfig() for h in (self.agent.hooks if self.agent else [])}


# =============================================================================
# Generation Management
# =============================================================================


@dataclass
class Generation:
    """
    A generation represents a point-in-time snapshot of configuration.

    Like NixOS generations - can rollback to any previous generation.
    """
    id: int
    config_hash: str
    created_at: datetime
    config_path: Path
    active: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "config_hash": self.config_hash,
            "created_at": self.created_at.isoformat(),
            "config_path": str(self.config_path),
            "active": self.active,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Generation:
        return cls(
            id=data["id"],
            config_hash=data["config_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            config_path=Path(data["config_path"]),
            active=data.get("active", False),
            description=data.get("description", ""),
        )


class GenerationManager:
    """
    Manages configuration generations.

    Provides:
    - Generation creation on config changes
    - Instant rollback to previous generations
    - Generation listing and cleanup
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.generations_dir = base_path / "generations"
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.generations_file = base_path / "generations.json"
        self._generations: List[Generation] = []
        self._load_generations()

    def _load_generations(self):
        """Load generations from file"""
        if self.generations_file.exists():
            data = json.loads(self.generations_file.read_text())
            self._generations = [Generation.from_dict(g) for g in data]

    def _save_generations(self):
        """Save generations to file"""
        data = [g.to_dict() for g in self._generations]
        self.generations_file.write_text(json.dumps(data, indent=2))

    def create(
        self,
        config: BBXConfig,
        description: str = ""
    ) -> Generation:
        """Create a new generation from config"""
        # Get next generation ID
        next_id = max([g.id for g in self._generations], default=0) + 1

        # Save config to generations directory
        config_path = self.generations_dir / f"gen-{next_id}" / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(config.to_dict(), f)

        # Create generation record
        generation = Generation(
            id=next_id,
            config_hash=config.get_hash(),
            created_at=datetime.now(),
            config_path=config_path,
            active=False,
            description=description,
        )

        self._generations.append(generation)
        self._save_generations()

        logger.info(f"Created generation {next_id} ({config.get_hash()})")
        return generation

    def activate(self, generation_id: int) -> bool:
        """Activate a generation (make it current)"""
        # Deactivate all
        for gen in self._generations:
            gen.active = False

        # Activate specified
        for gen in self._generations:
            if gen.id == generation_id:
                gen.active = True
                self._save_generations()
                logger.info(f"Activated generation {generation_id}")
                return True

        return False

    def get_active(self) -> Optional[Generation]:
        """Get the currently active generation"""
        for gen in self._generations:
            if gen.active:
                return gen
        return None

    def get(self, generation_id: int) -> Optional[Generation]:
        """Get a specific generation"""
        for gen in self._generations:
            if gen.id == generation_id:
                return gen
        return None

    def list(self, limit: int = 10) -> List[Generation]:
        """List recent generations"""
        return sorted(self._generations, key=lambda g: g.id, reverse=True)[:limit]

    def rollback(self, generation_id: int) -> Optional[BBXConfig]:
        """Rollback to a previous generation"""
        gen = self.get(generation_id)
        if not gen:
            return None

        # Load config from generation
        config = BBXConfig.from_file(str(gen.config_path))

        # Activate the generation
        self.activate(generation_id)

        logger.info(f"Rolled back to generation {generation_id}")
        return config

    def cleanup(self, keep_count: int = 10):
        """Remove old generations, keeping the most recent ones"""
        if len(self._generations) <= keep_count:
            return

        # Sort by ID, keep newest
        sorted_gens = sorted(self._generations, key=lambda g: g.id, reverse=True)
        to_keep = sorted_gens[:keep_count]
        to_remove = sorted_gens[keep_count:]

        for gen in to_remove:
            if gen.active:
                continue  # Don't remove active generation

            # Remove generation directory
            gen_dir = gen.config_path.parent
            if gen_dir.exists():
                shutil.rmtree(gen_dir)

            self._generations.remove(gen)
            logger.info(f"Removed generation {gen.id}")

        self._save_generations()

    def diff(self, gen_id_1: int, gen_id_2: int) -> Dict[str, Any]:
        """Show differences between two generations"""
        gen1 = self.get(gen_id_1)
        gen2 = self.get(gen_id_2)

        if not gen1 or not gen2:
            return {"error": "Generation not found"}

        config1 = BBXConfig.from_file(str(gen1.config_path))
        config2 = BBXConfig.from_file(str(gen2.config_path))

        diff = {
            "from_generation": gen_id_1,
            "to_generation": gen_id_2,
            "agents_added": [],
            "agents_removed": [],
            "agents_modified": [],
        }

        # Compare agents
        agents1 = set(config1.agents.keys())
        agents2 = set(config2.agents.keys())

        diff["agents_added"] = list(agents2 - agents1)
        diff["agents_removed"] = list(agents1 - agents2)

        for agent_id in agents1 & agents2:
            a1 = config1.agents[agent_id]
            a2 = config2.agents[agent_id]
            if a1.adapters != a2.adapters or a1.hooks != a2.hooks:
                diff["agents_modified"].append(agent_id)

        return diff


# =============================================================================
# Declarative Manager
# =============================================================================


class DeclarativeManager:
    """
    Manages declarative BBX configuration.

    Like NixOS's nixos-rebuild - applies configuration atomically.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.home() / ".bbx"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.generation_manager = GenerationManager(self.base_path)
        self._current_config: Optional[BBXConfig] = None

    def load(self, config_path: str) -> BBXConfig:
        """Load configuration from file"""
        config = BBXConfig.from_file(config_path)
        self._current_config = config
        return config

    async def apply(
        self,
        config: Optional[BBXConfig] = None,
        description: str = ""
    ) -> Generation:
        """
        Apply configuration.

        Like 'nixos-rebuild switch' - creates new generation and activates it.
        """
        config = config or self._current_config
        if not config:
            raise ValueError("No configuration to apply")

        # Check if config changed
        active_gen = self.generation_manager.get_active()
        if active_gen and active_gen.config_hash == config.get_hash():
            logger.info("Configuration unchanged, nothing to apply")
            return active_gen

        # Create new generation
        generation = self.generation_manager.create(config, description)

        # Apply configuration (activate agents, hooks, etc.)
        await self._apply_config(config)

        # Activate generation
        self.generation_manager.activate(generation.id)

        self._current_config = config
        logger.info(f"Applied configuration, generation {generation.id}")
        return generation

    async def _apply_config(self, config: BBXConfig):
        """Internal: Apply configuration to running system"""
        # This would:
        # 1. Stop agents that were removed
        # 2. Update agents that changed
        # 3. Start new agents
        # 4. Reload hooks
        # 5. Update state backend if changed

        for agent_id, agent_config in config.agents.items():
            if agent_config.enabled:
                logger.info(f"Activating agent: {agent_id}")
                # TODO: Actually start/configure the agent
            else:
                logger.info(f"Agent disabled: {agent_id}")

    async def rollback(self, generation_id: Optional[int] = None) -> Optional[BBXConfig]:
        """
        Rollback to previous generation.

        If generation_id is None, rolls back to previous generation.
        """
        if generation_id is None:
            # Rollback to previous
            active = self.generation_manager.get_active()
            if not active:
                return None

            generations = self.generation_manager.list()
            for gen in generations:
                if gen.id < active.id:
                    generation_id = gen.id
                    break

        if generation_id is None:
            return None

        config = self.generation_manager.rollback(generation_id)
        if config:
            await self._apply_config(config)
            self._current_config = config

        return config

    def list_generations(self, limit: int = 10) -> List[Generation]:
        """List recent generations"""
        return self.generation_manager.list(limit)

    def diff_generations(self, gen1: int, gen2: int) -> Dict[str, Any]:
        """Show differences between generations"""
        return self.generation_manager.diff(gen1, gen2)

    def get_current_config(self) -> Optional[BBXConfig]:
        """Get current active configuration"""
        if self._current_config:
            return self._current_config
        # Try to load from active generation
        active = self.generation_manager.get_active()
        if active:
            return active.config
        return None

    def get_current_generation_id(self) -> Optional[int]:
        """Get current generation ID"""
        active = self.generation_manager.get_active()
        return active.id if active else None

    def diff_configs(self, config1: BBXConfig, config2: BBXConfig) -> List[Dict[str, Any]]:
        """Diff two configurations"""
        diff = []
        # Compare agents
        c1_agents = set(config1.agents.keys())
        c2_agents = set(config2.agents.keys())

        for agent_id in c1_agents - c2_agents:
            diff.append({"type": "removed", "path": f"agents.{agent_id}", "value": config1.agents[agent_id]})
        for agent_id in c2_agents - c1_agents:
            diff.append({"type": "added", "path": f"agents.{agent_id}", "value": config2.agents[agent_id]})
        for agent_id in c1_agents & c2_agents:
            if config1.agents[agent_id] != config2.agents[agent_id]:
                diff.append({"type": "changed", "path": f"agents.{agent_id}",
                            "old": config1.agents[agent_id], "new": config2.agents[agent_id]})

        # Compare top-level agent config
        if hasattr(config1, 'agent') and hasattr(config2, 'agent'):
            if config1.agent and config2.agent:
                if config1.agent.name != config2.agent.name:
                    diff.append({"type": "changed", "path": "agent.name",
                                "old": config1.agent.name, "new": config2.agent.name})

        return diff


# =============================================================================
# Flakes Support
# =============================================================================


@dataclass
class FlakeInput:
    """Input dependency for a flake"""
    name: str
    url: str
    rev: Optional[str] = None  # Git revision


@dataclass
class FlakeLock:
    """Lock file entry for reproducible builds"""
    name: str
    url: str
    rev: str
    hash: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BBXFlake:
    """
    BBX Flake - like Nix Flakes for reproducible agent environments.

    Provides:
    - Pinned dependencies
    - Multiple output configurations
    - Dev shells
    """
    version: str = "1.0"
    description: str = ""

    # Dependencies
    inputs: List[FlakeInput] = field(default_factory=list)

    # Outputs
    outputs: Dict[str, BBXConfig] = field(default_factory=dict)

    # Lock file (generated)
    lock: List[FlakeLock] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> BBXFlake:
        """Load flake from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BBXFlake:
        """Create flake from dictionary"""
        flake = cls()
        flake.version = data.get("version", "1.0")
        flake.description = data.get("description", "")

        # Parse inputs
        for input_data in data.get("inputs", []):
            flake.inputs.append(FlakeInput(
                name=input_data["name"],
                url=input_data["url"],
                rev=input_data.get("rev"),
            ))

        # Parse outputs
        for output_name, output_data in data.get("outputs", {}).items():
            flake.outputs[output_name] = BBXConfig.from_dict(output_data)

        return flake

    def get_output(self, name: str) -> Optional[BBXConfig]:
        """Get a specific output configuration"""
        return self.outputs.get(name)

    def update_lock(self):
        """Update lock file with current dependency versions"""
        self.lock = []
        for input_dep in self.inputs:
            # In real implementation, would resolve actual versions
            self.lock.append(FlakeLock(
                name=input_dep.name,
                url=input_dep.url,
                rev=input_dep.rev or "HEAD",
                hash=hashlib.sha256(input_dep.url.encode()).hexdigest()[:16],
            ))


# =============================================================================
# Example Configuration
# =============================================================================


EXAMPLE_CONFIG = """
# BBX 2.0 Declarative Configuration Example
# Like NixOS configuration.nix

version: "2.0"
name: my-agent-infrastructure
description: Production AI agent infrastructure

# Agent definitions
agents:
  analyst:
    name: Data Analyst Agent
    type: worker
    adapters:
      - http
      - database
      - transform
    hooks:
      - ./hooks/metrics.yaml
      - ./hooks/security.yaml
    workflows:
      - ./workflows/analyze.bbx
    quotas:
      max_concurrent_steps: 10
      max_execution_time_seconds: 300
      max_http_requests_per_minute: 100
    tags:
      - production
      - analytics

  orchestrator:
    name: Orchestrator Agent
    type: coordinator
    adapters:
      - http
      - workflow
    workflows:
      - ./workflows/orchestrate.bbx
    tags:
      - production
      - coordination

# State backend
state:
  backend: sqlite
  path: ./data/state.db

# Secrets (loaded from environment or vault)
secrets:
  API_KEY:
    source: env
    env_var: BBX_API_KEY
  DB_PASSWORD:
    source: vault
    path: secret/database/password
"""


def create_example_config(path: str):
    """Create example configuration file"""
    with open(path, "w") as f:
        f.write(EXAMPLE_CONFIG)
    logger.info(f"Created example config at {path}")
