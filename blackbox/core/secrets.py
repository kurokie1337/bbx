# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX Secrets Management

"""
BBX Secrets Management

Secure storage for sensitive data like API keys, passwords, tokens.

Features:
- Encrypted storage (using Fernet)
- Environment variable integration
- Workspace-scoped secrets
- Secret rotation support

Usage:
    bbx secret set API_KEY "sk-xxx"
    bbx secret get API_KEY
    bbx secret list
    bbx secret delete API_KEY
"""

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SecretsManager:
    """
    BBX Secrets Manager.

    Stores secrets securely using encryption.
    Falls back to obfuscation if cryptography is not available.
    """

    SECRETS_FILE = "secrets.enc"
    KEY_FILE = ".secrets.key"

    def __init__(self, workspace_path: Optional[Path] = None):
        """
        Initialize secrets manager.

        Args:
            workspace_path: Path to workspace (for workspace-scoped secrets)
        """
        if workspace_path:
            self._secrets_dir = Path(workspace_path) / ".bbx"
        else:
            self._secrets_dir = Path.home() / ".bbx" / "secrets"

        self._secrets_dir.mkdir(parents=True, exist_ok=True)
        self._secrets_file = self._secrets_dir / self.SECRETS_FILE
        self._key_file = self._secrets_dir / self.KEY_FILE

        # Initialize encryption
        self._fernet: Optional[Fernet] = None
        if CRYPTO_AVAILABLE:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize encryption key."""
        if self._key_file.exists():
            key = self._key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self._key_file.write_bytes(key)
            # Set restrictive permissions
            try:
                os.chmod(self._key_file, 0o600)
            except Exception:
                pass  # Windows doesn't support chmod

        self._fernet = Fernet(key)

    def _encrypt(self, data: str) -> str:
        """Encrypt data."""
        if self._fernet:
            return self._fernet.encrypt(data.encode()).decode()
        else:
            # Fallback: base64 obfuscation (NOT secure, just hiding)
            return base64.b64encode(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """Decrypt data."""
        if self._fernet:
            return self._fernet.decrypt(data.encode()).decode()
        else:
            return base64.b64decode(data.encode()).decode()

    def _load_secrets(self) -> Dict[str, Dict[str, Any]]:
        """Load secrets from file."""
        if not self._secrets_file.exists():
            return {}

        try:
            encrypted = self._secrets_file.read_text()
            decrypted = self._decrypt(encrypted)
            return json.loads(decrypted)
        except Exception:
            return {}

    def _save_secrets(self, secrets: Dict[str, Dict[str, Any]]) -> None:
        """Save secrets to file."""
        data = json.dumps(secrets, indent=2, default=str)
        encrypted = self._encrypt(data)
        self._secrets_file.write_text(encrypted)

        # Set restrictive permissions
        try:
            os.chmod(self._secrets_file, 0o600)
        except Exception:
            pass

    def set(
        self,
        key: str,
        value: str,
        description: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set a secret.

        Args:
            key: Secret key
            value: Secret value
            description: Optional description
            expires_at: Optional expiration date (ISO format)

        Returns:
            Result dict
        """
        secrets = self._load_secrets()

        secrets[key] = {
            "value": value,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at,
        }

        self._save_secrets(secrets)

        return {
            "status": "success",
            "key": key,
            "created": key not in secrets,
            "encrypted": CRYPTO_AVAILABLE,
        }

    def get(
        self,
        key: str,
        default: Optional[str] = None,
        check_env: bool = True,
    ) -> Dict[str, Any]:
        """
        Get a secret.

        Args:
            key: Secret key
            default: Default value if not found
            check_env: Also check environment variables

        Returns:
            Result dict with value
        """
        # First check environment
        if check_env:
            env_value = os.environ.get(key)
            if env_value is not None:
                return {
                    "status": "success",
                    "key": key,
                    "value": env_value,
                    "source": "environment",
                }

        # Then check stored secrets
        secrets = self._load_secrets()

        if key in secrets:
            secret = secrets[key]

            # Check expiration
            if secret.get("expires_at"):
                expires = datetime.fromisoformat(secret["expires_at"])
                if datetime.now(timezone.utc) > expires:
                    return {
                        "status": "error",
                        "key": key,
                        "error": "Secret has expired",
                    }

            return {
                "status": "success",
                "key": key,
                "value": secret["value"],
                "source": "secrets",
                "description": secret.get("description"),
            }

        if default is not None:
            return {
                "status": "success",
                "key": key,
                "value": default,
                "source": "default",
            }

        return {
            "status": "error",
            "key": key,
            "error": "Secret not found",
        }

    def delete(self, key: str) -> Dict[str, Any]:
        """
        Delete a secret.

        Args:
            key: Secret key

        Returns:
            Result dict
        """
        secrets = self._load_secrets()

        if key in secrets:
            del secrets[key]
            self._save_secrets(secrets)
            return {
                "status": "success",
                "key": key,
                "deleted": True,
            }

        return {
            "status": "error",
            "key": key,
            "error": "Secret not found",
        }

    def list(
        self,
        pattern: Optional[str] = None,
        show_values: bool = False,
    ) -> Dict[str, Any]:
        """
        List all secrets.

        Args:
            pattern: Optional filter pattern
            show_values: Include values (masked by default)

        Returns:
            List of secrets
        """
        import fnmatch

        secrets = self._load_secrets()
        results = []

        for key, secret in secrets.items():
            if pattern and not fnmatch.fnmatch(key, pattern):
                continue

            item = {
                "key": key,
                "description": secret.get("description"),
                "created_at": secret.get("created_at"),
                "updated_at": secret.get("updated_at"),
                "expires_at": secret.get("expires_at"),
            }

            if show_values:
                item["value"] = secret["value"]
            else:
                # Mask value
                value = secret["value"]
                if len(value) > 4:
                    item["value"] = value[:2] + "*" * (len(value) - 4) + value[-2:]
                else:
                    item["value"] = "*" * len(value)

            results.append(item)

        return {
            "status": "success",
            "secrets": results,
            "count": len(results),
            "encrypted": CRYPTO_AVAILABLE,
        }

    def rotate(
        self,
        key: str,
        new_value: str,
        keep_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Rotate a secret (update with new value).

        Args:
            key: Secret key
            new_value: New secret value
            keep_history: Keep old value in history

        Returns:
            Result dict
        """
        secrets = self._load_secrets()

        if key not in secrets:
            return {
                "status": "error",
                "key": key,
                "error": "Secret not found",
            }

        old_value = secrets[key]["value"]

        if keep_history:
            history = secrets[key].get("history", [])
            history.append({
                "value": old_value,
                "rotated_at": datetime.now(timezone.utc).isoformat(),
            })
            secrets[key]["history"] = history[-10:]  # Keep last 10

        secrets[key]["value"] = new_value
        secrets[key]["updated_at"] = datetime.now(timezone.utc).isoformat()

        self._save_secrets(secrets)

        return {
            "status": "success",
            "key": key,
            "rotated": True,
        }

    def export(self, format: str = "env") -> Dict[str, Any]:
        """
        Export secrets.

        Args:
            format: Export format ("env", "json")

        Returns:
            Exported data
        """
        secrets = self._load_secrets()

        if format == "env":
            lines = []
            for key, secret in secrets.items():
                value = secret["value"].replace('"', '\\"')
                lines.append(f'{key}="{value}"')
            return {
                "status": "success",
                "format": "env",
                "data": "\n".join(lines),
            }

        elif format == "json":
            data = {key: secret["value"] for key, secret in secrets.items()}
            return {
                "status": "success",
                "format": "json",
                "data": json.dumps(data, indent=2),
            }

        return {
            "status": "error",
            "error": f"Unknown format: {format}",
        }


# Singleton
_manager: Optional[SecretsManager] = None


def get_secrets_manager(workspace_path: Optional[Path] = None) -> SecretsManager:
    """Get singleton secrets manager."""
    global _manager
    if _manager is None:
        _manager = SecretsManager(workspace_path)
    return _manager
