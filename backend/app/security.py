"""Security helpers: password policy, login rate limit and browser security headers."""
import re
import time
from collections import defaultdict, deque

from fastapi import HTTPException

MAX_FAILS = 5                 # failed logins allowed ...
WINDOW_S = 10 * 60            # ... within 10 minutes, then the account / address is locked for 10 minutes
_fails = defaultdict(deque)

HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def check_password(pw: str):
    """At least 8 characters with a letter and a digit."""
    if len(pw or "") < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        raise HTTPException(400, "Password must be at least 8 characters and contain a letter and a number")


def _key(email, ip):
    return f"{(email or '').strip().lower()}|{ip or ''}"


def login_allowed(email, ip):
    q = _fails[_key(email, ip)]
    now = time.time()
    while q and now - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= MAX_FAILS:
        wait = int(WINDOW_S - (now - q[0])) // 60 + 1
        raise HTTPException(429, f"Too many failed sign-in attempts. Try again in about {wait} minute(s).")


def login_failed(email, ip):
    _fails[_key(email, ip)].append(time.time())


def login_ok(email, ip):
    _fails.pop(_key(email, ip), None)
