"""
Security utilities for BBX Console

JWT authentication (optional, disabled by default).
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import settings

# JWT (optional import)
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    if not JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed. Install with: pip install pyjwt")

    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.auth_token_expire_minutes))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, settings.auth_secret, algorithm=settings.auth_algorithm)


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    if not JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed")

    try:
        payload = jwt.decode(token, settings.auth_secret, algorithms=[settings.auth_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get current user from JWT token (if auth enabled)"""
    if not settings.auth_enabled:
        # Auth disabled - allow all requests
        return {"id": "anonymous", "role": "admin"}

    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    return verify_token(credentials.credentials)


def require_auth(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires authentication"""
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user
