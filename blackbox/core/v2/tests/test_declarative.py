# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Tests for Declarative BBX - NixOS-inspired infrastructure as code

These tests verify:
- Configuration parsing and validation
- Generation management
- Configuration application and rollback
- Configuration diffing
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from blackbox.core.v2.declarative import (
    BBXConfig,
    AgentConfig,
    QuotaConfig,
    StateConfig,
    SecretConfig,
    AdapterConfig,
    HookConfig,
    Generation,
    GenerationManager,
    DeclarativeManager,
    BBXFlake,
    FlakeInput,
    FlakeLock,
)


@pytest.fixture
def sample_config():
    """Create a sample BBX configuration"""
    config = BBXConfig(version="2.0", name="test-config")
    config.agents["test-agent"] = AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        adapters=["shell", "http"],
        hooks=["metrics"],
        quotas=QuotaConfig(
            max_concurrent_steps=10,
            max_execution_time_seconds=300,
            max_context_size_bytes=512 * 1024 * 1024,  # 512MB
        ),
    )
    return config


@pytest.fixture
def generation_manager():
    """Create a generation manager with temp directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = GenerationManager(base_path=Path(tmpdir))
        yield manager


class TestBBXConfigParsing:
    """Configuration parsing tests"""

    def test_create_config(self, sample_config):
        """Test creating a config object"""
        assert sample_config.version == "2.0"
        assert sample_config.agent.name == "Test Agent"  # From backward compat property
        assert len(sample_config.agents) == 1

    def test_config_to_dict(self, sample_config):
        """Test converting config to dictionary"""
        data = sample_config.to_dict()

        assert data["version"] == "2.0"
        assert "test-agent" in data["agents"]
        assert data["agents"]["test-agent"]["name"] == "Test Agent"

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            "version": "2.0",
            "agent": {
                "name": "from-dict-agent",
                "description": "Created from dict",
            },
            "quotas": {
                "max_concurrent_steps": 20,
            },
            "adapters": {
                "shell": {"enabled": True},
            },
        }

        config = BBXConfig.from_dict(data)
        assert config.agent.name == "from-dict-agent"
        assert config.quotas.max_concurrent_steps == 20

    def test_config_validation_valid(self, sample_config):
        """Test validation of valid config"""
        errors = sample_config.validate()
        assert len(errors) == 0

    def test_config_validation_missing_agent(self):
        """Test validation with missing agent"""
        config = BBXConfig(version="2.0")  # No agents
        errors = config.validate()
        assert len(errors) > 0
        assert any("agent" in e.lower() for e in errors)


class TestGenerationManagement:
    """Generation management tests"""

    def test_create_generation(self, generation_manager, sample_config):
        """Test creating a new generation"""
        gen = generation_manager.create(sample_config, description="Initial config")

        assert gen.id == 1
        assert gen.description == "Initial config"
        assert gen.config_hash is not None

    def test_create_multiple_generations(self, generation_manager, sample_config):
        """Test creating multiple generations"""
        gen1 = generation_manager.create(sample_config)

        # Modify config
        sample_config.agent.description = "Modified"
        gen2 = generation_manager.create(sample_config)

        assert gen1.id == 1
        assert gen2.id == 2

    def test_get_generation(self, generation_manager, sample_config):
        """Test retrieving a generation"""
        created = generation_manager.create(sample_config)
        retrieved = generation_manager.get(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_current(self, generation_manager, sample_config):
        """Test getting current generation"""
        gen1 = generation_manager.create(sample_config)
        generation_manager.activate(gen1.id)

        gen2 = generation_manager.create(sample_config)
        generation_manager.activate(gen2.id)

        current = generation_manager.get_active()
        assert current.id == 2

    def test_list_generations(self, generation_manager, sample_config):
        """Test listing generations"""
        for i in range(5):
            sample_config.agent.description = f"Version {i}"
            generation_manager.create(sample_config)

        generations = generation_manager.list(limit=10)
        assert len(generations) == 5

    def test_switch_generation(self, generation_manager, sample_config):
        """Test switching to a previous generation"""
        gen1 = generation_manager.create(sample_config)
        generation_manager.activate(gen1.id)

        sample_config.agents["test-agent"].description = "Modified"
        gen2 = generation_manager.create(sample_config)
        generation_manager.activate(gen2.id)

        # Switch back to gen1 using activate
        generation_manager.activate(gen1.id)
        current = generation_manager.get_active()
        assert current.id == gen1.id


class TestDeclarativeManager:
    """Declarative manager tests"""

    @pytest.fixture
    def declarative_manager(self):
        """Create a declarative manager with temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeclarativeManager(base_path=Path(tmpdir))
            yield manager

    @pytest.mark.asyncio
    async def test_apply_config(self, declarative_manager, sample_config):
        """Test applying a configuration"""
        generation = await declarative_manager.apply(sample_config)

        assert generation.id == 1
        current = declarative_manager.get_current_config()
        assert current.agent.name == sample_config.agent.name

    @pytest.mark.asyncio
    async def test_rollback(self, declarative_manager, sample_config):
        """Test rolling back configuration"""
        # Create initial config
        await declarative_manager.apply(sample_config)

        # Create modified config
        sample_config.agents["test-agent"].description = "Modified"
        await declarative_manager.apply(sample_config)

        # Rollback to generation 1
        config = await declarative_manager.rollback(1)
        assert config is not None

    def test_diff_configs(self, declarative_manager, sample_config):
        """Test diffing configurations"""
        config1 = sample_config

        # Create modified config
        config2 = BBXConfig(version="2.0")
        config2.agents["different-agent"] = AgentConfig(
            id="different-agent",
            name="different-agent",  # Different agent
            description=sample_config.agent.description if sample_config.agent else "",
        )

        diff = declarative_manager.diff_configs(config1, config2)
        assert len(diff) > 0
        # Should have removed/added for agent changes
        assert any(d["type"] in ["removed", "added", "changed"] for d in diff)


class TestConfigDiff:
    """Configuration diff tests"""

    def test_diff_added(self):
        """Test detecting added fields"""
        config1 = BBXConfig(version="2.0")
        config1.agents["test"] = AgentConfig(id="test", name="test")

        config2 = BBXConfig(version="2.0")
        config2.agents["test"] = AgentConfig(id="test", name="test")
        config2.agents["new_agent"] = AgentConfig(id="new_agent", name="New Agent")

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "added" for d in diff)

    def test_diff_removed(self):
        """Test detecting removed fields"""
        config1 = BBXConfig(version="2.0")
        config1.agents["test"] = AgentConfig(id="test", name="test")
        config1.agents["old_agent"] = AgentConfig(id="old_agent", name="Old Agent")

        config2 = BBXConfig(version="2.0")
        config2.agents["test"] = AgentConfig(id="test", name="test")

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "removed" for d in diff)

    def test_diff_changed(self):
        """Test detecting changed fields"""
        config1 = BBXConfig(version="2.0")
        config1.agents["test"] = AgentConfig(id="test", name="old-name")

        config2 = BBXConfig(version="2.0")
        config2.agents["test"] = AgentConfig(id="test", name="new-name")

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "changed" for d in diff)


class TestBBXFlake:
    """BBX Flake tests"""

    def test_create_flake(self):
        """Test creating a flake"""
        flake = BBXFlake(
            description="Test flake",
            inputs=[
                FlakeInput(
                    name="core",
                    url="github:bbx-project/bbx-core",
                ),
            ],
        )

        assert flake.description == "Test flake"
        assert len(flake.inputs) == 1
        assert flake.inputs[0].name == "core"

    def test_flake_lock_update(self):
        """Test flake lock update"""
        flake = BBXFlake(
            description="Test flake",
            inputs=[
                FlakeInput(
                    name="core",
                    url="github:bbx-project/bbx-core",
                ),
            ],
        )

        flake.update_lock()
        assert len(flake.lock) == 1
        assert flake.lock[0].name == "core"

    def test_flake_from_dict(self):
        """Test creating flake from dictionary"""
        data = {
            "description": "From dict",
            "inputs": [
                {
                    "name": "core",
                    "url": "github:bbx-project/bbx-core",
                },
            ],
        }

        flake = BBXFlake.from_dict(data)
        assert flake.description == "From dict"
        assert len(flake.inputs) == 1
        assert flake.inputs[0].name == "core"


class TestFlakeLock:
    """Flake lock tests"""

    def test_create_lock(self):
        """Test creating a flake lock"""
        lock = FlakeLock(
            name="core",
            url="github:bbx-project/bbx-core",
            rev="abc123",
            hash="somehash",
        )

        assert lock.name == "core"
        assert lock.rev == "abc123"

    def test_lock_fields(self):
        """Test flake lock fields"""
        lock = FlakeLock(
            name="core",
            url="github:bbx-project/bbx-core",
            rev="abc123",
            hash="somehash",
        )

        # Verify all required fields
        assert lock.name == "core"
        assert lock.url == "github:bbx-project/bbx-core"
        assert lock.rev == "abc123"
        assert lock.hash == "somehash"
        assert lock.timestamp is not None


class TestExampleConfig:
    """Example config generation tests"""

    def test_create_config_manually(self):
        """Test creating configuration manually"""
        config = BBXConfig(version="2.0")
        config.agents["test"] = AgentConfig(
            id="test",
            name="Test Agent",
            description="A test agent",
        )

        assert config is not None
        assert config.version == "2.0"
        assert config.agent is not None
        assert config.agent.name == "Test Agent"

    def test_manual_config_is_valid(self):
        """Test that manually created config is valid"""
        config = BBXConfig(version="2.0")
        config.agents["test"] = AgentConfig(
            id="test",
            name="Test Agent",
            description="A test agent",
        )

        errors = config.validate()
        assert len(errors) == 0
