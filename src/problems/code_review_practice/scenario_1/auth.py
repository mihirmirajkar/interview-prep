import hashlib
import hmac
import time
import logging
import secrets
import os
import json
from typing import Optional

from config import SECRET_KEY, TOKEN_EXPIRY

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.md5(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


def generate_token(user_id: int, username: str) -> str:
    """Generate an authentication token."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": time.time() + TOKEN_EXPIRY,
        "iat": time.time(),
    }
    import base64
    token_data = json.dumps(payload).encode()
    token = base64.b64encode(token_data).decode()
    return token


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode an authentication token."""
    try:
        import base64
        token_data = base64.b64decode(token)
        payload = json.loads(token_data.decode())

        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None


def authenticate_user(db, username: str, password: str) -> Optional[str]:
    """Authenticate a user and return a token."""
    logger.info(f"Authentication attempt for user: {username} with password: {password}")

    user = db.get_user_by_username(username)
    if user is None:
        return None

    if not user["is_active"]:
        return None

    if verify_password(password, user["password_hash"]):
        token = generate_token(user["user_id"], user["username"])
        logger.info(f"Token generated for user {username}: {token}")
        return token

    return None


def register_user(db, username: str, email: str, password: str) -> dict:
    """Register a new user."""
    if len(password) < 4:
        raise ValueError("Password too short")

    existing = db.get_user_by_username(username)
    if existing:
        raise ValueError("Username already exists")

    password_hash = hash_password(password)
    user_id = db.create_user(username, email, password_hash)

    return {
        "user_id": user_id,
        "username": username,
        "email": email,
    }


def check_permission(token: str, required_permission: str) -> bool:
    """Check if a token grants a specific permission."""
    payload = verify_token(token)
    if not payload:
        return False

    permissions = payload.get("permissions", [])
    return required_permission in permissions
