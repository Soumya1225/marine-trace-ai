"""Live AIS ships from AISStream.io (free WebSocket feed, worldwide).

A background thread subscribes to the active watch zones and keeps the latest position of every ship in memory.
Every position is also written to Datasets/raw/ais/live/ais_live_YYYYMMDD.csv (same columns as the other AIS
files), so images from the satellite feed can be analysed together with the real ship tracks.
Needs AISSTREAM_API_KEY in the .env file (free key at aisstream.io).
"""
import asyncio
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from app.database import SessionLocal
from app.models import WatchZone


def _utcnow():
    """Current UTC time without tzinfo (same values as the old datetime.utcnow())."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


URL = "wss://stream.aisstream.io/v0/stream"
TYPES = ["PositionReport", "StandardClassBPositionReport", "ShipStaticData"]
KEEP_HOURS = 6
SILENT_MIN = 30          # a moving ship that stops reporting this long counts as "went silent"
CSV_COLS = ["mmsi", "time", "lat", "lon", "sog", "cog", "heading", "vessel_name", "imo", "vessel_type", "length"]


class LiveAIS:
    def __init__(self):
        self.ships = {}
        self.lock = threading.Lock()
        self.connected = False
        self.started_at = None
        self.messages = 0
        self.last_message_at = None
        self.last_error = ""
        self.zones = []
        self._restart = threading.Event()
        self._rows = []
        self._thread = None

    # ---------- public ----------
    @property
    def key(self):
        return os.getenv("AISSTREAM_API_KEY", "").strip()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="ais-live", daemon=True)
        self._thread.start()

    def restart(self):
        self._restart.set()
        self.start()

    def status(self):
        return {"configured": bool(self.key), "connected": self.connected, "messages": self.messages,
                "ships": len(self.ships), "since": self.started_at, "last_message_at": self.last_message_at,
                "zones": [z["name"] for z in self.zones], "error": self.last_error}

    def snapshot(self, bbox=None, minutes=60):
        now = _utcnow()
        out = []
        with self.lock:
            for s in self.ships.values():
                if s.get("lat") is None:
                    continue
                if bbox and not (bbox[0] <= s["lon"] <= bbox[2] and bbox[1] <= s["lat"] <= bbox[3]):
                    continue
                age = (now - s["seen"]).total_seconds() / 60
                silent = age > SILENT_MIN and (s.get("sog") or 0) > 2
                if age > minutes and not silent:
                    continue
                out.append({k: s.get(k) for k in ("mmsi", "name", "type", "imo", "length", "lat", "lon", "sog", "cog",
                                                     "heading", "destination", "zone")}
                           | {"seen": s["seen"].isoformat() + "Z", "age_min": round(age, 1), "silent": silent,
                              "stopped": (s.get("sog") or 0) < 0.5, "trail": list(s["trail"])})
        return out

    # ---------- internals ----------
    def _load_zones(self):
        db = SessionLocal()
        try:
            return [{"name": z.name, "bbox": [z.min_lon, z.min_lat, z.max_lon, z.max_lat]}
                    for z in db.query(WatchZone).filter(WatchZone.active.is_(True))]
        finally:
            db.close()

    def _zone_of(self, lat, lon):
        for z in self.zones:
            b = z["bbox"]
            if b[0] <= lon <= b[2] and b[1] <= lat <= b[3]:
                return z["name"]
        return None

    def _handle(self, msg):
        kind = msg.get("MessageType")
        meta = msg.get("MetaData", {})
        body = msg.get("Message", {}).get(kind, {})
        mmsi = meta.get("MMSI") or body.get("UserID")
        if not mmsi:
            return
        now = _utcnow()
        with self.lock:
            s = self.ships.setdefault(mmsi, {"mmsi": mmsi, "trail": deque(maxlen=40), "seen": now})
            name = (meta.get("ShipName") or "").strip()
            if name:
                s["name"] = name
            if kind == "ShipStaticData":
                s["type"] = body.get("Type")
                s["imo"] = body.get("ImoNumber")
                s["destination"] = (body.get("Destination") or "").strip()
                d = body.get("Dimension") or {}
                if d.get("A") is not None and d.get("B") is not None:
                    s["length"] = d["A"] + d["B"]
                return
            lat, lon = body.get("Latitude", meta.get("latitude")), body.get("Longitude", meta.get("longitude"))
            if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180:
                return
            s.update(lat=lat, lon=lon, sog=body.get("Sog"), cog=body.get("Cog"), seen=now,
                     heading=None if body.get("TrueHeading") in (None, 511) else body.get("TrueHeading"),
                     zone=self._zone_of(lat, lon))
            s["trail"].append([round(lat, 5), round(lon, 5)])
            self._rows.append([mmsi, now.strftime("%Y-%m-%dT%H:%M:%S"), lat, lon, s.get("sog"), s.get("cog"),
                               s.get("heading"), s.get("name", ""), s.get("imo") or "", s.get("type") or "",
                               s.get("length") or ""])

    def _flush(self):
        rows, self._rows = self._rows, []
        if not rows:
            return
        from ml import config as mlc
        folder = mlc.RAW_DIR / "ais" / "live"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"ais_live_{_utcnow():%Y%m%d}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(CSV_COLS)
            w.writerows(rows)

    def _prune(self):
        cut = _utcnow() - timedelta(hours=KEEP_HOURS)
        with self.lock:
            for k in [k for k, s in self.ships.items() if s["seen"] < cut]:
                del self.ships[k]

    async def _session(self):
        import websockets
        self.zones = self._load_zones()
        if not self.zones:
            self.last_error = "No active watch zones: create one first"
            await asyncio.sleep(60)
            return
        boxes = [[[z["bbox"][1], z["bbox"][0]], [z["bbox"][3], z["bbox"][2]]] for z in self.zones]
        async with websockets.connect(URL, ping_interval=20, close_timeout=5) as ws:
            await ws.send(json.dumps({"APIKey": self.key, "BoundingBoxes": boxes, "FilterMessageTypes": TYPES}))
            self.connected, self.last_error = True, ""
            last_flush = time.time()
            while not self._restart.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                self.messages += 1
                self.last_message_at = _utcnow().isoformat() + "Z"
                self._handle(msg)
                if time.time() - last_flush > 60:
                    self._flush()
                    self._prune()
                    last_flush = time.time()
            self._flush()

    def _run(self):
        self.started_at = _utcnow().isoformat() + "Z"
        wait = 5
        while True:
            if not self.key:
                self.connected, self.last_error = False, "AISSTREAM_API_KEY is not set in the .env file"
                time.sleep(60)
                continue
            self._restart.clear()
            try:
                asyncio.run(self._session())
                wait = 5
            except Exception as exc:
                self.last_error = str(exc)[:300]
                print(f"[ais-live] {self.last_error}; reconnecting in {wait} s")
            self.connected = False
            if not self._restart.is_set():
                time.sleep(wait)
                wait = min(wait * 2, 300)


LIVE = LiveAIS()


def live_files_for(day):
    """Recorded live AIS files for this day and the day before (for analysing a satellite image)."""
    from ml import config as mlc
    folder = mlc.RAW_DIR / "ais" / "live"
    names = [f"ais_live_{(day - timedelta(days=d)):%Y%m%d}.csv" for d in (1, 0)]
    return [folder / n for n in names if (folder / n).exists()]
