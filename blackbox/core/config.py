# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Configuration System

Centralized configuration management supporting:
- Environment variables
- Config files (.env, .bbx.yaml)
- Programmatic defaults
- Pydantic validation
- Type-safe access
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.config")

# Check for Pydantic
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

    def Field(*args, **kwargs):  # type: ignore
        return None

    def validator(*args, **kwargs):  # type: ignore
        return lambda f: f


# ============================================================================
# Configuration Models
# ============================================================================

if PYDANTIC_AVAILABLE:

    class PathsConfig(BaseModel):
        """Path configuration"""
        model_config = ConfigDict(arbitrary_types_allowed=True)

        bbx_home: Path = Field(
            default_factory=lambda: Path.home() / ".bbx",
            description="BBX home directory",
        )
        cache_dir: Path = Field(
            default_factory=lambda: Path.home() / ".bbx" / "cache",
            description="Cache directory",
        )
        state_dir: Path = Field(
            default_factory=lambda: Path.home() / ".bbx" / "state",
            description="State directory",
        )
        bundle_dir: Path = Field(
            default_factory=lambda: Path.cwd() / ".bbx_bundle",
            description="Bundle build directory",
        )
        output_dir: Path = Field(
            default_factory=lambda: Path.cwd() / "output",
            description="Default output directory",
        )
        temp_dir: Path = Field(
            default_factory=lambda: Path.home() / ".bbx" / "tmp",
            description="Temporary files directory",
        )
        log_dir: Path = Field(
            default_factory=lambda: Path.home() / ".bbx" / "logs",
            description="Log files directory",
        )

        @field_validator("*", mode="before")
        @classmethod
        def ensure_path(cls, v):
            """Convert strings to Path objects"""
            if isinstance(v, str):
                return Path(v).expanduser()
            return v

    class RuntimeConfig(BaseModel):
        """Runtime execution configuration"""

        default_timeout_ms: int = Field(
            default=30000, description="Default step timeout (ms)", ge=0
        )
        max_parallel_steps: int = Field(
            default=10, description="Maximum parallel step execution", ge=1
        )
        enable_caching: bool = Field(
            default=True, description="Enable workflow parsing cache"
        )
        cache_ttl_seconds: int = Field(
            default=3600, description="Cache TTL in seconds", ge=0
        )
        retry_default: int = Field(default=0, description="Default retry count", ge=0)
        retry_delay_ms: int = Field(
            default=1000, description="Default retry delay (ms)", ge=0
        )
        retry_backoff: float = Field(
            default=2.0, description="Retry backoff multiplier", ge=1.0
        )

    class ObservabilityConfig(BaseModel):
        """Observability configuration"""

        enable_metrics: bool = Field(
            default=True, description="Enable metrics collection"
        )
        enable_tracing: bool = Field(
            default=True, description="Enable distributed tracing"
        )
        enable_logging: bool = Field(
            default=True, description="Enable structured logging"
        )
        log_level: str = Field(default="INFO", description="Logging level")
        metrics_retention: int = Field(
            default=10000, description="Max metrics to retain", ge=100
        )
        trace_retention: int = Field(
            default=1000, description="Max traces to retain", ge=10
        )
        log_retention: int = Field(
            default=10000, description="Max log entries to retain", ge=100
        )
        export_interval_seconds: int = Field(
            default=60, description="Export interval (seconds)", ge=1
        )

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v):
            """Validate log level"""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            v_upper = v.upper()
            if v_upper not in valid_levels:
                raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
            return v_upper

    class AdapterConfig(BaseModel):
        """Adapter configuration"""

        aws_region: Optional[str] = Field(
            default=None, description="Default AWS region"
        )
        gcp_project: Optional[str] = Field(
            default=None, description="Default GCP project"
        )
        azure_subscription: Optional[str] = Field(
            default=None, description="Default Azure subscription"
        )
        http_timeout: int = Field(
            default=30, description="HTTP request timeout (seconds)", ge=1
        )
        http_max_retries: int = Field(default=3, description="HTTP max retries", ge=0)
        enable_ssl_verify: bool = Field(
            default=True, description="Verify SSL certificates"
        )

    class SecurityConfig(BaseModel):
        """Security configuration"""

        enable_sandbox: bool = Field(default=False, description="Enable sandbox mode")
        allowed_adapters: Optional[List[str]] = Field(
            default=None, description="Whitelist of allowed adapters"
        )
        blocked_adapters: Optional[List[str]] = Field(
            default=None, description="Blacklist of blocked adapters"
        )
        max_workflow_size_mb: int = Field(
            default=10, description="Max workflow file size (MB)", ge=1
        )
        allow_network_access: bool = Field(
            default=True, description="Allow network access"
        )
        allow_file_system_access: bool = Field(
            default=True, description="Allow file system access"
        )

    class BBXConfig(BaseModel):
        """Complete BBX configuration"""
        model_config = ConfigDict(arbitrary_types_allowed=True)

        paths: PathsConfig = Field(
            default_factory=PathsConfig, description="Path configuration"
        )
        runtime: RuntimeConfig = Field(
            default_factory=RuntimeConfig, description="Runtime configuration"
        )
        observability: ObservabilityConfig = Field(
            default_factory=ObservabilityConfig,
            description="Observability configuration",
        )
        adapters: AdapterConfig = Field(
            default_factory=AdapterConfig, description="Adapter configuration"
        )
        security: SecurityConfig = Field(
            default_factory=SecurityConfig, description="Security configuration"
        )

else:
    # Fallback dataclass implementation without Pydantic
    @dataclass
    class PathsConfig:  # type: ignore
        """Path configuration (no validation)"""

        bbx_home: Path = field(default_factory=lambda: Path.home() / ".bbx")
        cache_dir: Path = field(default_factory=lambda: Path.home() / ".bbx" / "cache")
        state_dir: Path = field(default_factory=lambda: Path.home() / ".bbx" / "state")
        bundle_dir: Path = field(default_factory=lambda: Path.cwd() / ".bbx_bundle")
        output_dir: Path = field(default_factory=lambda: Path.cwd() / "output")
        temp_dir: Path = field(default_factory=lambda: Path.home() / ".bbx" / "tmp")
        log_dir: Path = field(default_factory=lambda: Path.home() / ".bbx" / "logs")

    @dataclass
    class RuntimeConfig:  # type: ignore
        """Runtime configuration (no validation)"""

        default_timeout_ms: int = 30000
        max_parallel_steps: int = 10
        enable_caching: bool = True
        cache_ttl_seconds: int = 3600
        retry_default: int = 0
        retry_delay_ms: int = 1000
        retry_backoff: float = 2.0

    @dataclass
    class ObservabilityConfig:  # type: ignore
        """Observability configuration (no validation)"""

        enable_metrics: bool = True
        enable_tracing: bool = True
        enable_logging: bool = True
        log_level: str = "INFO"
        metrics_retention: int = 10000
        trace_retention: int = 1000
        log_retention: int = 10000
        export_interval_seconds: int = 60

    @dataclass
    class AdapterConfig:  # type: ignore
        """Adapter configuration (no validation)"""

        aws_region: Optional[str] = None
        gcp_project: Optional[str] = None
        azure_subscription: Optional[str] = None
        http_timeout: int = 30
        http_max_retries: int = 3
        enable_ssl_verify: bool = True

    @dataclass
    class SecurityConfig:  # type: ignore
        """Security configuration (no validation)"""

        enable_sandbox: bool = False
        allowed_adapters: Optional[List[str]] = None
        blocked_adapters: Optional[List[str]] = None
        max_workflow_size_mb: int = 10
        allow_network_access: bool = True
        allow_file_system_access: bool = True

    @dataclass
    class BBXConfig:  # type: ignore
        """Complete BBX configuration (no validation)"""

        paths: PathsConfig = field(default_factory=PathsConfig)
        runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
        observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
        adapters: AdapterConfig = field(default_factory=AdapterConfig)
        security: SecurityConfig = field(default_factory=SecurityConfig)


# ============================================================================
# Configuration Loader
# ============================================================================


class ConfigLoader:
    """Load configuration from multiple sources"""

    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config: Dict[str, Any] = {}

        # Paths
        bbx_home = os.getenv("BBX_HOME")
        if bbx_home:
            config.setdefault("paths", {})["bbx_home"] = bbx_home

        bbx_cache_dir = os.getenv("BBX_CACHE_DIR")
        if bbx_cache_dir:
            config.setdefault("paths", {})["cache_dir"] = bbx_cache_dir

        bbx_state_dir = os.getenv("BBX_STATE_DIR")
        if bbx_state_dir:
            config.setdefault("paths", {})["state_dir"] = bbx_state_dir

        bbx_bundle_dir = os.getenv("BBX_BUNDLE_DIR")
        if bbx_bundle_dir:
            config.setdefault("paths", {})["bundle_dir"] = bbx_bundle_dir

        bbx_output_dir = os.getenv("BBX_OUTPUT_DIR")
        if bbx_output_dir:
            config.setdefault("paths", {})["output_dir"] = bbx_output_dir

        bbx_temp_dir = os.getenv("BBX_TEMP_DIR")
        if bbx_temp_dir:
            config.setdefault("paths", {})["temp_dir"] = bbx_temp_dir

        bbx_log_dir = os.getenv("BBX_LOG_DIR")
        if bbx_log_dir:
            config.setdefault("paths", {})["log_dir"] = bbx_log_dir

        # Runtime
        bbx_timeout = os.getenv("BBX_TIMEOUT")
        if bbx_timeout:
            config.setdefault("runtime", {})["default_timeout_ms"] = int(bbx_timeout)

        bbx_max_parallel = os.getenv("BBX_MAX_PARALLEL")
        if bbx_max_parallel:
            config.setdefault("runtime", {})["max_parallel_steps"] = int(
                bbx_max_parallel
            )

        bbx_enable_cache = os.getenv("BBX_ENABLE_CACHE")
        if bbx_enable_cache:
            config.setdefault("runtime", {})["enable_caching"] = (
                bbx_enable_cache.lower() == "true"
            )

        bbx_cache_ttl = os.getenv("BBX_CACHE_TTL")
        if bbx_cache_ttl:
            config.setdefault("runtime", {})["cache_ttl_seconds"] = int(bbx_cache_ttl)

        # Observability
        bbx_log_level = os.getenv("BBX_LOG_LEVEL")
        if bbx_log_level:
            config.setdefault("observability", {})["log_level"] = bbx_log_level

        bbx_enable_metrics = os.getenv("BBX_ENABLE_METRICS")
        if bbx_enable_metrics:
            config.setdefault("observability", {})["enable_metrics"] = (
                bbx_enable_metrics.lower() == "true"
            )

        bbx_enable_tracing = os.getenv("BBX_ENABLE_TRACING")
        if bbx_enable_tracing:
            config.setdefault("observability", {})["enable_tracing"] = (
                bbx_enable_tracing.lower() == "true"
            )

        # Adapters
        aws_region = os.getenv("AWS_REGION")
        if aws_region:
            config.setdefault("adapters", {})["aws_region"] = aws_region

        gcp_project = os.getenv("GCP_PROJECT")
        if gcp_project:
            config.setdefault("adapters", {})["gcp_project"] = gcp_project

        azure_sub = os.getenv("AZURE_SUBSCRIPTION_ID")
        if azure_sub:
            config.setdefault("adapters", {})["azure_subscription"] = azure_sub

        bbx_http_timeout = os.getenv("BBX_HTTP_TIMEOUT")
        if bbx_http_timeout:
            config.setdefault("adapters", {})["http_timeout"] = int(bbx_http_timeout)

        # Security
        bbx_sandbox = os.getenv("BBX_SANDBOX")
        if bbx_sandbox:
            config.setdefault("security", {})["enable_sandbox"] = (
                bbx_sandbox.lower() == "true"
            )

        bbx_allowed = os.getenv("BBX_ALLOWED_ADAPTERS")
        if bbx_allowed:
            config.setdefault("security", {})["allowed_adapters"] = bbx_allowed.split(
                ","
            )

        bbx_blocked = os.getenv("BBX_BLOCKED_ADAPTERS")
        if bbx_blocked:
            config.setdefault("security", {})["blocked_adapters"] = bbx_blocked.split(
                ","
            )

        return config

    @staticmethod
    def load_from_file(file_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        if not file_path.exists():
            return {}

        try:
            import yaml

            with open(file_path, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed, skipping config file")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple configuration dictionaries"""
        result: Dict[str, Any] = {}
        for config in configs:
            for key, value in config.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = ConfigLoader.merge_configs(result[key], value)
                else:
                    result[key] = value
        return result


# ============================================================================
# Global Configuration Instance
# ============================================================================

_config: Optional[BBXConfig] = None


def get_config() -> BBXConfig:
    """
    Get global BBX configuration

    Configuration is loaded from (in order of precedence):
    1. Environment variables (BBX_*)
    2. .bbx.yaml in current directory
    3. ~/.bbx/config.yaml
    4. Default values

    Returns:
        BBXConfig instance
    """
    global _config

    if _config is None:
        _config = load_config()

    return _config


def load_config(
    config_file: Optional[Path] = None, env_override: bool = True
) -> BBXConfig:
    """
    Load configuration from all sources

    Args:
        config_file: Optional specific config file to load
        env_override: Whether environment variables override file config

    Returns:
        BBXConfig instance
    """
    configs = []

    # 1. Load from default locations
    default_locations = [
        Path.home() / ".bbx" / "config.yaml",
        Path.cwd() / ".bbx.yaml",
    ]

    for location in default_locations:
        if location.exists():
            file_config = ConfigLoader.load_from_file(location)
            if file_config:
                configs.append(file_config)
                logger.debug(f"Loaded config from {location}")

    # 2. Load from specific file if provided
    if config_file:
        file_config = ConfigLoader.load_from_file(config_file)
        if file_config:
            configs.append(file_config)
            logger.debug(f"Loaded config from {config_file}")

    # 3. Load from environment variables
    if env_override:
        env_config = ConfigLoader.load_from_env()
        if env_config:
            configs.append(env_config)
            logger.debug("Loaded config from environment")

    # 4. Merge all configs
    merged = ConfigLoader.merge_configs(*configs) if configs else {}

    # 5. Create BBXConfig instance
    if PYDANTIC_AVAILABLE:
        try:
            return BBXConfig(**merged)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            logger.warning("Using default configuration")
            return BBXConfig()
    else:
        # Manual construction without validation
        config = BBXConfig()

        if "paths" in merged:
            for key, value in merged["paths"].items():
                setattr(
                    config.paths, key, Path(value) if isinstance(value, str) else value
                )

        if "runtime" in merged:
            for key, value in merged["runtime"].items():
                setattr(config.runtime, key, value)

        if "observability" in merged:
            for key, value in merged["observability"].items():
                setattr(config.observability, key, value)

        if "adapters" in merged:
            for key, value in merged["adapters"].items():
                setattr(config.adapters, key, value)

        if "security" in merged:
            for key, value in merged["security"].items():
                setattr(config.security, key, value)

        return config


def reload_config():
    """Reload global configuration"""
    global _config
    _config = load_config()
    logger.info("Configuration reloaded")


def ensure_directories(config: Optional[BBXConfig] = None):
    """Ensure all configured directories exist"""
    if config is None:
        config = get_config()

    directories = [
        config.paths.bbx_home,
        config.paths.cache_dir,
        config.paths.state_dir,
        config.paths.bundle_dir,
        config.paths.output_dir,
        config.paths.temp_dir,
        config.paths.log_dir,
    ]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")


# Initialize directories on import
try:
    ensure_directories()
except Exception as e:
    logger.warning(f"Failed to initialize directories: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================


def get_path(path_name: str) -> Path:
    """Get a configured path by name"""
    config = get_config()
    return getattr(config.paths, path_name)


def get_state_file(filename: str) -> Path:
    """Get path to a state file"""
    return get_config().paths.state_dir / filename


def get_cache_file(filename: str) -> Path:
    """Get path to a cache file"""
    return get_config().paths.cache_dir / filename


def get_log_file(filename: str) -> Path:
    """Get path to a log file"""
    return get_config().paths.log_dir / filename


def get_temp_file(filename: str) -> Path:
    """Get path to a temporary file"""
    return get_config().paths.temp_dir / filename


# Alias for backward compatibility
get_settings = get_config
