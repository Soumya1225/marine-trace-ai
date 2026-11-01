"""Create dashboard alerts and (optionally) send them by e-mail and Telegram."""
import json
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

from app import config
from app.models import Alert

SEND_LEVELS = ("Critical", "High")


def send_email(subject, body):
    if not (config.SMTP_HOST and config.ALERT_EMAIL_TO):
        return False
    try:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, config.SMTP_USER, config.ALERT_EMAIL_TO
        msg.set_content(body)
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as s:
            s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as exc:
        print(f"[alerts] e-mail failed: {exc}")
        return False


def send_telegram(text):
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": config.TELEGRAM_CHAT_ID, "text": text,
                                       "disable_web_page_preview": "true"}).encode()
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        with urllib.request.urlopen(url, data=data, timeout=15) as r:
            return bool(json.loads(r.read()).get("ok"))
    except Exception as exc:
        print(f"[alerts] Telegram failed: {exc}")
        return False


def channels():
    return {"email": bool(config.SMTP_HOST and config.ALERT_EMAIL_TO),
            "telegram": bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)}


def alert_text(level, message, incident=None):
    lines = [f"[{level}] Oil spill alert - SpillWatch / Marine Trace AI", message]
    if incident is not None:
        lines.append(f"Area: {incident.area_km2:.2f} km2 at {incident.centroid_lat:.4f}, {incident.centroid_lon:.4f}")
        top = incident.suspects[0] if incident.suspects else None
        if top:
            lines.append(f"Top suspect: {top.vessel_name} (MMSI {top.mmsi}, score {top.score:.2f})")
        lines.append(f"Open: {config.APP_URL}/incidents/{incident.id}")
    return "\n".join(lines)


def notify(level, message, incident=None):
    """Send to every configured channel. Returns {"email": bool, "telegram": bool}."""
    text = alert_text(level, message, incident)
    return {"email": send_email(f"[{level}] Oil spill alert", text), "telegram": send_telegram(text)}


def create_alert(db, incident, level, message):
    alert = Alert(incident_id=incident.id if incident else None, level=level, message=message)
    if level in SEND_LEVELS:
        sent = notify(level, message, incident)
        alert.emailed = sent["email"] or sent["telegram"]
    db.add(alert)
    db.commit()
    return alert
