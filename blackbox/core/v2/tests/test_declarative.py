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
    create_example_config,
)


@pytest.fixture
def sample_config():
    """Create a sample BBX configuration"""
    return BBXConfig(
        version="2.0",
        agent=AgentConfig(
            name="test-agent",
            description="Test agent for unit tests",
        ),
        quotas=QuotaConfig(
            max_memory_mb=512,
            max_cpu_percent=50,
            max_concurrent_ops=10,
            max_steps_per_workflow=100,
        ),
        adapters={
            "shell": AdapterConfig(
                enabled=True,
                config={"default_timeout": 30000},
            ),
            "http": AdapterConfig(
                enabled=True,
                config={"timeout": 10000},
            ),
        },
        hooks={
            "metrics": HookConfig(
                enabled=True,
                type="PROBE",
                attach=["step.post_execute"],
            ),
        },
    )


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
        assert sample_config.agent.name == "test-agent"
        assert len(sample_config.adapters) == 2

    def test_config_to_dict(self, sample_config):
        """Test converting config to dictionary"""
        data = sample_config.to_dict()

        assert data["version"] == "2.0"
        assert data["agent"]["name"] == "test-agent"
        assert "shell" in data["adapters"]

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            "version": "2.0",
            "agent": {
                "name": "from-dict-agent",
                "description": "Created from dict",
            },
            "quotas": {
                "max_memory_mb": 256,
            },
            "adapters": {
                "shell": {"enabled": True},
            },
        }

        config = BBXConfig.from_dict(data)
        assert config.agent.name == "from-dict-agent"
        assert config.quotas.max_memory_mb == 256

    def test_config_validation_valid(self, sample_config):
        """Test validation of valid config"""
        errors = sample_config.validate()
        assert len(errors) == 0

    def test_config_validation_missing_agent(self):
        """Test validation with missing agent"""
        config = BBXConfig(
            version="2.0",
            agent=None,  # Missing agent
        )
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
        generation_manager.create(sample_config)
        generation_manager.create(sample_config)

        current = generation_manager.get_current()
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
        sample_config.agent.description = "Modified"
        gen2 = generation_manager.create(sample_config)

        # Switch back to gen1
        generation_manager.switch(gen1.id)
        current = generation_manager.get_current()
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
        sample_config.agent.description = "Modified"
        await declarative_manager.apply(sample_config)

        # Rollback to generation 1
        config = await declarative_manager.rollback(1)
        assert config is not None

    def test_diff_configs(self, declarative_manager, sample_config):
        """Test diffing configurations"""
        config1 = sample_config

        # Create modified config
        config2 = BBXConfig(
            version="2.0",
            agent=AgentConfig(
                name="different-agent",  # Changed
                description=sample_config.agent.description,
            ),
            quotas=sample_config.quotas,
            adapters=sample_config.adapters,
            hooks=sample_config.hooks,
        )

        diff = declarative_manager.diff_configs(config1, config2)
        assert len(diff) > 0
        assert any(d["path"] == "agent.name" for d in diff)


class TestConfigDiff:
    """Configuration diff tests"""

    def test_diff_added(self):
        """Test detecting added fields"""
        config1 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="test"),
            adapters={},
        )
        config2 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="test"),
            adapters={
                "new_adapter": AdapterConfig(enabled=True),
            },
        )

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "added" for d in diff)

    def test_diff_removed(self):
        """Test detecting removed fields"""
        config1 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="test"),
            adapters={
                "old_adapter": AdapterConfig(enabled=True),
            },
        )
        config2 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="test"),
            adapters={},
        )

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "removed" for d in diff)

    def test_diff_changed(self):
        """Test detecting changed fields"""
        config1 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="old-name"),
        )
        config2 = BBXConfig(
            version="2.0",
            agent=AgentConfig(name="new-name"),
        )

        manager = DeclarativeManager()
        diff = manager.diff_configs(config1, config2)

        assert any(d["type"] == "changed" for d in diff)


class TestBBXFlake:
    """BBX Flake tests"""

    def test_create_flake(self):
        """Test creating a flake"""
        flake = BBXFlake(
            description="Test flake",
            inputs={
                "core": FlakeInput(
                    type="github",
                    url="github:bbx-project/bbx-core",
                ),
            },
        )

        assert flake.description == "Test flake"
        assert "core" in flake.inputs

    def test_flake_to_dict(self):
        """Test converting flake to dictionary"""
        flake = BBXFlake(
            description="Test flake",
            inputs={
                "core": FlakeInput(
                    type="github",
                    url="github:bbx-project/bbx-core",
                ),
            },
        )

        data = flake.to_dict()
        assert data["description"] == "Test flake"
        assert "core" in data["inputs"]

    def test_flake_from_dict(self):
        """Test creating flake from dictionary"""
        data = {
            "description": "From dict",
            "inputs": {
                "core": {
                    "type": "github",
                    "url": "github:bbx-project/bbx-core",
                },
            },
        }

        flake = BBXFlake.from_dict(data)
        assert flake.description == "From dict"
        assert flake.inputs["core"].type == "github"


class TestFlakeLock:
    """Flake lock tests"""

    def test_create_lock(self):
        """Test creating a flake lock"""
        lock = FlakeLock()
        lock.lock_input("core", "abc123", datetime.now())

        assert "core" in lock.locked
        assert lock.locked["core"]["hash"] == "abc123"

    def test_verify_lock(self):
        """Test verifying locked inputs"""
        lock = FlakeLock()
        lock.lock_input("core", "abc123", datetime.now())

        # Should pass with same hash
        is_valid = lock.verify("core", "abc123")
        assert is_valid

        # Should fail with different hash
        is_valid = lock.verify("core", "different")
        assert not is_valid


class TestExampleConfig:
    """Example config generation tests"""

    def test_create_example_config(self):
        """Test creating example configuration"""
        config = create_example_config()

        assert config is not None
        assert config.version == "2.0"
        assert config.agent is not None
        assert config.agent.name is not None

    def test_example_config_is_valid(self):
        """Test that example config is valid"""
        config = create_example_config()
        errors = config.validate()
        assert len(errors) == 0
