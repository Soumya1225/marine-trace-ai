"""LSTM drift model: learns how a slick moves from the sequence of wind and current it experiences.

Training data: drift trajectories simulated with the physics model and the REAL ERA5 wind / Copernicus
Marine current fields of the study period (hourly). Each step the LSTM sees the forcing at the slick's
current position (wind u/v, current u/v) and predicts the next hour's movement (east/north km).
It is used as a second, data-driven forecast next to the physics model and evaluated on held-out
trajectories against a simple "persistence" baseline (assume the first hour's motion continues).

    python -m ml.attribution.lstm_drift          (train; needs Datasets/raw/weather/*.nc for real forcing)
"""
import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from ml.attribution.drift import Environment, simulate
from ml.config import METRICS_DIR, MODELS_DIR
from ml.utils.geo import KM_PER_DEG_LAT, km_per_deg_lon

LSTM_PATH = MODELS_DIR / "lstm_drift.pt"
AREA = (-118.25, 33.45, -117.85, 33.75)
HOURS = 24


def _time_range(env):
    ds = env.wind_ds if env.wind_ds is not None else env.current_ds
    if ds is None:
        return pd.Timestamp("2021-10-01", tz="UTC"), pd.Timestamp("2021-10-05", tz="UTC")
    tname = "valid_time" if "valid_time" in ds.coords else "time"
    t = pd.to_datetime(ds[tname].values)
    return pd.Timestamp(t.min()).tz_localize("UTC"), pd.Timestamp(t.max()).tz_localize("UTC")


def make_sequences(env, n=600, hours=HOURS, seed=0):
    """Return X (n x hours x 4 forcing) and Y (n x hours x 2 hourly displacement km)."""
    rng = np.random.default_rng(seed)
    t0, t1 = _time_range(env)
    span_h = max(1, int((t1 - t0).total_seconds() / 3600) - hours - 1)
    X = np.zeros((n, hours, 4), np.float32)
    Y = np.zeros((n, hours, 2), np.float32)
    for i in range(n):
        lat = rng.uniform(AREA[1], AREA[3])
        lon = rng.uniform(AREA[0], AREA[2])
        start = (t0 + timedelta(hours=int(rng.integers(0, span_h)))).to_pydatetime()
        snaps = simulate([lat], [lon], start, hours, env, step_min=60, diffusion=0.0)
        for k in range(hours):
            la, lo = snaps[k]["lat"][0], snaps[k]["lon"][0]
            X[i, k] = env.at(np.array([la]), np.array([lo]), snaps[k]["time"])
            Y[i, k, 0] = (snaps[k + 1]["lon"][0] - lo) * km_per_deg_lon(la)
            Y[i, k, 1] = (snaps[k + 1]["lat"][0] - la) * KM_PER_DEG_LAT
    return X, Y


def build_model(hidden=32):
    import torch.nn as nn

    class DriftLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(4, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 2)

        def forward(self, x, state=None):
            h, state = self.lstm(x, state)
            return self.out(h), state

    return DriftLSTM()


def train(epochs=300, n=600, seed=0):
    import torch
    torch.manual_seed(seed)
    env = Environment.auto()
    X, Y = make_sequences(env, n=n, seed=seed)
    xm, xs = X.reshape(-1, 4).mean(0), X.reshape(-1, 4).std(0) + 1e-6
    ys = float(np.abs(Y).mean()) + 1e-6
    n_tr = int(0.8 * n)
    Xt = torch.tensor((X - xm) / xs)
    Yt = torch.tensor(Y / ys)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(epochs):
        opt.zero_grad()
        pred, _ = model(Xt[:n_tr])
        loss = ((pred - Yt[:n_tr]) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xt[n_tr:])[0].numpy() * ys
    true = Y[n_tr:]
    errors, persistence = {}, {}
    for h in (6, 12, 24):
        cp, ct = pred[:, :h].sum(1), true[:, :h].sum(1)
        cpers = true[:, :1] * h                     # first hour's motion repeated
        errors[f"{h}h"] = round(float(np.mean(np.hypot(*(cp - ct).T))), 3)
        persistence[f"{h}h"] = round(float(np.mean(np.hypot(*(cpers[:, 0] - ct).T))), 3)
    torch.save({"state_dict": model.state_dict(), "xm": xm, "xs": xs, "ys": ys}, LSTM_PATH)
    metrics = {"model": "LSTM (hidden 32), forcing: wind u/v + current u/v",
               "forcing_source": env.describe()["source"], "train_sequences": n_tr, "test_sequences": n - n_tr,
               "lstm_error_km": errors, "persistence_error_km": persistence}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "lstm_drift.json").write_text(json.dumps(metrics, indent=2))
    return metrics


_cached = {}


def forecast(lat, lon, start, env, hours=24):
    """LSTM forecast of a slick centre: list of (hour, lat, lon); None if the model is not trained."""
    if not LSTM_PATH.exists():
        return None
    try:
        import torch
        if "m" not in _cached:
            ck = torch.load(LSTM_PATH, map_location="cpu", weights_only=False)
            m = build_model()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _cached.update(m=m, ck=ck)
        m, ck = _cached["m"], _cached["ck"]
        out, state, t = [], None, start
        for h in range(1, hours + 1):
            f = np.array(env.at(np.array([lat]), np.array([lon]), t), np.float32)
            x = torch.tensor(((f - ck["xm"]) / ck["xs"]).reshape(1, 1, 4), dtype=torch.float32)
            with torch.no_grad():
                d, state = m(x, state)
            dx, dy = (d[0, 0].numpy() * ck["ys"]).tolist()
            lat, lon = lat + dy / KM_PER_DEG_LAT, lon + dx / km_per_deg_lon(lat)
            t = t + timedelta(hours=1)
            out.append((h, round(float(lat), 6), round(float(lon), 6)))
        return out
    except Exception as exc:
        print(f"[LSTM drift] skipped: {exc}")
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    print(json.dumps(train(epochs=args.epochs, n=args.sequences), indent=2))
