"""Backend settings (override any value with an environment variable or the root .env file)."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# make the ml package importable from the backend
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-before-deployment")
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "12"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'oilspill.db').as_posix()}")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BACKEND_DIR / "uploads"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@oilspill.local")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# optional e-mail alerts
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# optional Telegram alerts (create a bot with @BotFather, then get your chat id)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# automatic satellite feed: check active watch zones (with "auto check" on) every N hours; 0 = off
FEED_INTERVAL_HOURS = float(os.getenv("FEED_INTERVAL_HOURS", "6"))
FEED_LOOKBACK_DAYS = int(os.getenv("FEED_LOOKBACK_DAYS", "3"))
# download + analyse new images automatically (each Sentinel-1 scene is about 1 GB; needs CDSE_USER / CDSE_PASSWORD)
FEED_AUTO_ANALYSE = os.getenv("FEED_AUTO_ANALYSE", "false").lower() == "true"

# dashboard address used in alert links
APP_URL = os.getenv("APP_URL", "http://localhost:5173")

for sub in ("scenes", "ais", "results", "reports"):
    (STORAGE_DIR / sub).mkdir(parents=True, exist_ok=True)
