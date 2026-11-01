# Deployment

## Option 1: laptop (development)
Double-click `start.bat`. The backend runs on http://127.0.0.1:8000, the dashboard on http://localhost:5173.

## Option 2: Docker (any computer or cloud server)
Needs Docker Desktop (Windows / Mac) or Docker Engine (Linux).

```bash
docker compose up -d --build      # first time: about 10-15 minutes (downloads PyTorch, builds the dashboard)
```
Open **http://localhost:8080**. Stop with `docker compose down`.

| Container | What it runs | Port |
|---|---|---|
| `backend` | FastAPI API, AI models, analysis jobs, satellite feed, live AIS | 8000 (internal) |
| `frontend` | React dashboard served by nginx; forwards `/api` to the backend | 8080 |

- Model weights (`models/`) and data (`Datasets/`) stay on the computer and are mounted into the container,
  so the image stays small and nothing large goes into Git.
- The database and uploads live in Docker volumes (`spillwatch-db`, `spillwatch-uploads`), so they survive restarts.
- Settings and keys come from the `.env` file (never copied into the image): `SECRET_KEY`, `ADMIN_PASSWORD`,
  `CDSE_USER` / `CDSE_PASSWORD` (Copernicus downloads), `AISSTREAM_API_KEY` (live ships),
  `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` and `SMTP_*` (alerts), `FEED_INTERVAL_HOURS`, `FEED_AUTO_ANALYSE`.

## Option 3: cloud server
Any VM with 4 GB+ RAM (for example AWS EC2 t3.medium, Azure B2s, Google e2-medium, or a college server):
1. Install Docker, copy the project folder (with `models/` and the needed `Datasets/`) and the `.env` file.
2. `docker compose up -d --build`
3. Open port 8080 (or put nginx / Caddy with HTTPS in front) and set `APP_URL` and `CORS_ORIGINS` to the public address.
4. Change `SECRET_KEY` and the admin password before giving anyone the link.

## Health check
`GET /api/health` returns `{"status": "ok"}`; Docker restarts the backend if it stops answering.
The **System health** page in the dashboard shows models, data freshness, live AIS and alert channels.
