# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX Authentication Module

"""
BBX Authentication and Authorization

Provides authentication mechanisms for:
- Docker registries
- MCP servers
- A2A agents
- External APIs

Also provides API key management for BBX services.
"""

import base64
import hashlib
import os
import secrets as secrets_module
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from blackbox.core.secrets import get_secrets_manager


class AuthProvider:
    """Base authentication provider."""

    def __init__(self, name: str):
        self.name = name

    def authenticate(self, **kwargs) -> Dict[str, Any]:
        """Authenticate and return credentials."""
        raise NotImplementedError

    def validate(self, token: str) -> bool:
        """Validate a token."""
        raise NotImplementedError


class DockerAuthProvider(AuthProvider):
    """Docker registry authentication."""

    def __init__(self):
        super().__init__("docker")

    def authenticate(
        self,
        registry: str = "docker.io",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get Docker registry authentication.

        Args:
            registry: Registry URL
            username: Username (or from env DOCKER_USERNAME)
            password: Password (or from env DOCKER_PASSWORD)

        Returns:
            Auth config for Docker
        """
        # Try to get from secrets/env
        secrets = get_secrets_manager()

        if not username:
            result = secrets.get("DOCKER_USERNAME")
            username = result.get("value") if result.get("status") == "success" else None

        if not password:
            result = secrets.get("DOCKER_PASSWORD")
            password = result.get("value") if result.get("status") == "success" else None

        if username and password:
            # Create base64 encoded auth
            auth_string = f"{username}:{password}"
            encoded = base64.b64encode(auth_string.encode()).decode()

            return {
                "status": "success",
                "registry": registry,
                "auth_config": {
                    "username": username,
                    "password": password,
                    "auth": encoded,
                },
            }

        return {
            "status": "error",
            "error": "Docker credentials not found",
        }


class APIKeyProvider(AuthProvider):
    """API Key authentication."""

    def __init__(self):
        super().__init__("api_key")

    def generate(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a new API key.

        Args:
            name: Key name/description
            scopes: Permission scopes
            expires_in_days: Expiration in days

        Returns:
            Generated API key
        """
        # Generate secure random key
        key = f"bbx_{secrets_module.token_urlsafe(32)}"

        expires_at = None
        if expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()

        # Store key hash (not the actual key)
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        return {
            "status": "success",
            "key": key,
            "key_hash": key_hash,
            "name": name,
            "scopes": scopes or ["*"],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at,
        }

    def authenticate(
        self,
        api_key: Optional[str] = None,
        header_name: str = "X-API-Key",
    ) -> Dict[str, Any]:
        """
        Validate API key authentication.

        Args:
            api_key: API key to validate
            header_name: Header name for API key

        Returns:
            Auth result
        """
        if not api_key:
            # Try from environment
            api_key = os.environ.get("BBX_API_KEY")

        if not api_key:
            return {
                "status": "error",
                "error": "API key not provided",
            }

        # Validate format
        if not api_key.startswith("bbx_"):
            return {
                "status": "error",
                "error": "Invalid API key format",
            }

        return {
            "status": "success",
            "authenticated": True,
            "key_prefix": api_key[:8] + "...",
        }


class BearerAuthProvider(AuthProvider):
    """Bearer token authentication."""

    def __init__(self):
        super().__init__("bearer")

    def authenticate(
        self,
        token: Optional[str] = None,
        env_var: str = "BBX_TOKEN",
    ) -> Dict[str, Any]:
        """
        Get bearer token authentication.

        Args:
            token: Bearer token
            env_var: Environment variable name

        Returns:
            Auth headers
        """
        if not token:
            # Try from secrets
            secrets = get_secrets_manager()
            result = secrets.get(env_var)
            if result.get("status") == "success":
                token = result.get("value")

        if not token:
            return {
                "status": "error",
                "error": f"Bearer token not found (set {env_var})",
            }

        return {
            "status": "success",
            "headers": {
                "Authorization": f"Bearer {token}",
            },
        }


class BasicAuthProvider(AuthProvider):
    """HTTP Basic authentication."""

    def __init__(self):
        super().__init__("basic")

    def authenticate(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        username_env: str = "BBX_USERNAME",
        password_env: str = "BBX_PASSWORD",
    ) -> Dict[str, Any]:
        """
        Get HTTP Basic authentication.

        Args:
            username: Username
            password: Password
            username_env: Environment variable for username
            password_env: Environment variable for password

        Returns:
            Auth headers
        """
        secrets = get_secrets_manager()

        if not username:
            result = secrets.get(username_env)
            if result.get("status") == "success":
                username = result.get("value")

        if not password:
            result = secrets.get(password_env)
            if result.get("status") == "success":
                password = result.get("value")

        if not username or not password:
            return {
                "status": "error",
                "error": "Username and password required",
            }

        # Create base64 encoded auth
        auth_string = f"{username}:{password}"
        encoded = base64.b64encode(auth_string.encode()).decode()

        return {
            "status": "success",
            "headers": {
                "Authorization": f"Basic {encoded}",
            },
        }


# === Auth Manager ===

class AuthManager:
    """
    BBX Authentication Manager.

    Centralized authentication for all BBX services.
    """

    def __init__(self):
        self._providers: Dict[str, AuthProvider] = {
            "docker": DockerAuthProvider(),
            "api_key": APIKeyProvider(),
            "bearer": BearerAuthProvider(),
            "basic": BasicAuthProvider(),
        }

    def get_provider(self, name: str) -> Optional[AuthProvider]:
        """Get auth provider by name."""
        return self._providers.get(name)

    def authenticate(
        self,
        provider: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Authenticate using specified provider.

        Args:
            provider: Provider name (docker, api_key, bearer, basic)
            **kwargs: Provider-specific arguments

        Returns:
            Authentication result
        """
        auth_provider = self._providers.get(provider)
        if not auth_provider:
            return {
                "status": "error",
                "error": f"Unknown auth provider: {provider}",
            }

        return auth_provider.authenticate(**kwargs)

    def list_providers(self) -> List[str]:
        """List available auth providers."""
        return list(self._providers.keys())


# Singleton
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get singleton auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


# === Auth Registry for Universal Adapter ===

class AuthInjectionProvider(ABC):
    """Base class for auth injection providers."""

    @abstractmethod
    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Inject authentication into environment and volumes.

        Returns:
            Tuple of (env_vars, volume_mounts)
        """
        pass


class TokenAuthInjector(AuthInjectionProvider):
    """Token-based auth injection."""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        token_key = config.get("token_key", "API_TOKEN")
        token_value = config.get("token")

        if not token_value:
            # Try to get from environment
            secrets = get_secrets_manager()
            result = secrets.get(token_key)
            if result.get("status") == "success":
                token_value = result.get("value")

        if token_value:
            env[token_key] = token_value

        return env, volumes


class BasicAuthInjector(AuthInjectionProvider):
    """HTTP Basic auth injection."""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        username = config.get("username")
        password = config.get("password")

        if username:
            env["AUTH_USERNAME"] = username
        if password:
            env["AUTH_PASSWORD"] = password

        return env, volumes


class FileAuthInjector(AuthInjectionProvider):
    """File-based auth injection (e.g., service account keys)."""

    def inject(self, config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        env = {}
        volumes = {}

        file_path = config.get("file")
        mount_path = config.get("mount_path", "/auth/credentials.json")
        env_var = config.get("env_var", "CREDENTIALS_FILE")

        if file_path and os.path.exists(file_path):
            volumes[file_path] = mount_path
            env[env_var] = mount_path

        return env, volumes


class AuthRegistry:
    """
    Registry for authentication injection providers.
    Used by Universal Adapter for Docker container auth injection.
    """

    _providers: Dict[str, AuthInjectionProvider] = {
        "token": TokenAuthInjector(),
        "basic": BasicAuthInjector(),
        "file": FileAuthInjector(),
    }

    @classmethod
    def get_provider(cls, provider_name: Optional[str]) -> Optional[AuthInjectionProvider]:
        """Get auth provider by name."""
        if not provider_name:
            return None
        return cls._providers.get(provider_name)

    @classmethod
    def register_provider(cls, name: str, provider: AuthInjectionProvider) -> None:
        """Register a custom auth provider."""
        cls._providers[name] = provider

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available auth providers."""
        return list(cls._providers.keys())
