from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth import ADMIN, create_token, get_current_user, hash_password, user_dict, verify_password
from app.database import get_db
from app.models import User
from app.schemas import LoginIn, UserCreate, UserUpdate
from app.security import check_password, login_allowed, login_failed, login_ok
from app.services.activity import log

router = APIRouter(prefix="/api", tags=["auth & users"])
ROLES = {"admin", "analyst", "viewer"}


@router.get("/auth/setup-status")
def setup_status(db: Session = Depends(get_db)):
    """Public: tells the login page whether the first-run admin password is still in use (to show the hint)."""
    from app import config
    admin = db.query(User).filter(User.email == config.ADMIN_EMAIL.lower()).first()
    default = bool(admin and verify_password(config.ADMIN_PASSWORD, admin.password_hash))
    return {"default_password": default, "email": config.ADMIN_EMAIL if default else None}


@router.post("/auth/login")
def login(body: LoginIn, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else ""
    login_allowed(body.email, ip)
    user = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        login_failed(body.email, ip)
        raise HTTPException(401, "Email or password is incorrect")
    login_ok(body.email, ip)
    if not user.active:
        raise HTTPException(403, "This account is disabled. Contact an administrator.")
    log(db, user, "login")
    return {"token": create_token(user), "user": user_dict(user)}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return user_dict(user)


@router.get("/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(ADMIN)):
    return [user_dict(u) for u in db.query(User).order_by(User.id)]


@router.post("/users")
def create_user(body: UserCreate, db: Session = Depends(get_db), admin: User = Depends(ADMIN)):
    if body.role not in ROLES:
        raise HTTPException(400, f"Role must be one of {sorted(ROLES)}")
    check_password(body.password)
    email = body.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "A user with this email already exists")
    u = User(name=body.name, email=email, role=body.role, password_hash=hash_password(body.password))
    db.add(u)
    db.commit()
    log(db, admin, "user_created", f"{email} ({body.role})")
    return user_dict(u)


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserUpdate, db: Session = Depends(get_db), admin: User = Depends(ADMIN)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    if body.role is not None:
        if body.role not in ROLES:
            raise HTTPException(400, "Unknown role")
        u.role = body.role
    if body.name is not None:
        u.name = body.name
    if body.active is not None:
        if u.id == admin.id and not body.active:
            raise HTTPException(400, "You cannot disable your own account")
        u.active = body.active
    if body.password:
        check_password(body.password)
        u.password_hash = hash_password(body.password)
    db.commit()
    log(db, admin, "user_updated", u.email)
    return user_dict(u)
