"""Password hashing (PBKDF2, standard library) and JWT tokens."""
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import SECRET_KEY, TOKEN_HOURS
from app.database import get_db
from app.models import User

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(check.hex(), digest)


def create_token(user: User) -> str:
    payload = {"sub": str(user.id), "role": user.role,
               "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer),
                     db: Session = Depends(get_db)) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired. Sign in again.")
    user = db.get(User, int(payload["sub"]))
    if not user or not user.active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account not found or disabled")
    return user


def require_roles(*roles):
    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Your role cannot do this action")
        return user
    return checker


ANALYST = require_roles("admin", "analyst")
ADMIN = require_roles("admin")


def user_dict(u: User) -> dict:
    return {"id": u.id, "name": u.name, "email": u.email, "role": u.role, "active": u.active,
            "created_at": u.created_at.isoformat() if u.created_at else None}
