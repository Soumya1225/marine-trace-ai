"""Cross-attention fusion model for vessel attribution.

Each candidate ship is a token (its track/behaviour features). The spill and the sea conditions form a
context token (radar evidence + environment). Ship tokens attend to the context and to each other
(cross-attention), so a ship is judged *relative* to the spill and to the other ships nearby.
The output is one score per ship; a softmax over the candidates gives each ship's probability of being
the source. Trained on the synthetic cases (true ship known), compared with XGBoost and the rule score.
"""
import numpy as np

from ml.config import MODELS_DIR

ATTENTION_PATH = MODELS_DIR / "attention_ranker.pt"
VESSEL_FEATS = ["min_dist_km", "alignment", "overlap", "hours_before", "max_gap_h",
                "speed_ratio", "loiter_h", "type_risk", "length_m"]
CONTEXT_FEATS = ["ctx_area_km2", "ctx_elongation", "ctx_length_km", "ctx_wind_ms", "ctx_current_ms"]
LOG_COLS = {"min_dist_km", "hours_before", "max_gap_h", "loiter_h", "length_m", "ctx_area_km2",
            "ctx_elongation", "ctx_length_km"}


def _tf(names, rows):
    x = np.array([[float(r.get(k) or 0.0) for k in names] for r in rows], dtype=np.float32)
    for j, k in enumerate(names):
        if k in LOG_COLS:
            x[:, j] = np.log1p(np.maximum(x[:, j], 0))
    return x


def build_model(d=32, heads=4):
    import torch
    import torch.nn as nn

    class CrossAttentionRanker(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_emb = nn.Sequential(nn.Linear(len(VESSEL_FEATS), d), nn.ReLU(), nn.Linear(d, d))
            self.c_emb = nn.Sequential(nn.Linear(len(CONTEXT_FEATS), d), nn.ReLU(), nn.Linear(d, d))
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

        def forward(self, v, c, pad):
            """v: B x N x Fv ships, c: B x Fc context, pad: B x N (True = padding)."""
            vt = self.v_emb(v)
            ct = self.c_emb(c).unsqueeze(1)
            kv = torch.cat([ct, vt], 1)
            kpm = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
            att, _ = self.attn(vt, kv, kv, key_padding_mask=kpm)
            h = self.norm(vt + att)
            s = self.head(torch.cat([h, ct.expand(-1, vt.size(1), -1)], -1)).squeeze(-1)
            return s.masked_fill(pad, -1e9)

    return CrossAttentionRanker()


def _batch(groups, mv, sv, mc, sc, max_n=50):
    """groups: list of (vessel_rows, context_row, target_index or -1)."""
    import torch
    n = min(max(len(g[0]) for g in groups), max_n)
    B = len(groups)
    V = np.zeros((B, n, len(VESSEL_FEATS)), np.float32)
    C = np.zeros((B, len(CONTEXT_FEATS)), np.float32)
    P = np.ones((B, n), bool)
    T = np.full(B, -1, np.int64)
    for i, (rows, ctx, target) in enumerate(groups):
        rows = rows[:n]
        V[i, :len(rows)] = (_tf(VESSEL_FEATS, rows) - mv) / sv
        C[i] = (_tf(CONTEXT_FEATS, [ctx])[0] - mc) / sc
        P[i, :len(rows)] = False
        T[i] = target if target < n else -1
    return torch.tensor(V), torch.tensor(C), torch.tensor(P), torch.tensor(T)


def train(table, train_ids, test_ids, epochs=400, seed=0):
    """table: DataFrame from train_ranker.build_table (one row per candidate, with ctx_* columns)."""
    import torch
    torch.manual_seed(seed)

    def groups_for(ids):
        out = []
        for cid in ids:
            g = table[table.case == cid].sort_values("rule_score", ascending=False)
            rows = g.to_dict("records")
            labels = [r["label"] for r in rows]
            out.append((rows, rows[0], labels.index(1) if 1 in labels else -1))
        return out

    tr, te = groups_for(sorted(train_ids)), groups_for(sorted(test_ids))
    all_v = np.concatenate([_tf(VESSEL_FEATS, g[0]) for g in tr])
    all_c = _tf(CONTEXT_FEATS, [g[1] for g in tr])
    mv, sv = all_v.mean(0), all_v.std(0) + 1e-6
    mc, sc = all_c.mean(0), all_c.std(0) + 1e-6

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    V, C, P, T = _batch([g for g in tr if g[2] >= 0], mv, sv, mc, sc)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(V, C, P), T)
        loss.backward()
        opt.step()

    model.eval()
    Vt, Ct, Pt, Tt = _batch(te, mv, sv, mc, sc)
    with torch.no_grad():
        scores = model(Vt, Ct, Pt).numpy()
    order = np.argsort(-scores, axis=1)
    t = Tt.numpy()
    top1 = float(np.mean([(t[i] >= 0) and order[i, 0] == t[i] for i in range(len(t))]))
    top3 = float(np.mean([(t[i] >= 0) and t[i] in order[i, :3] for i in range(len(t))]))
    torch.save({"state_dict": model.state_dict(), "mv": mv, "sv": sv, "mc": mc, "sc": sc}, ATTENTION_PATH)
    return {"attention_top1": round(top1, 4), "attention_top3": round(top3, 4)}


_cached = {}


def score(vessel_rows, context):
    """Probability (softmax) of each candidate being the source; None if the model is not trained."""
    if not ATTENTION_PATH.exists():
        return None
    try:
        import torch
        if "m" not in _cached:
            ck = torch.load(ATTENTION_PATH, map_location="cpu", weights_only=False)
            m = build_model()
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _cached.update(m=m, ck=ck)
        m, ck = _cached["m"], _cached["ck"]
        V, C, P, _ = _batch([(vessel_rows, context, -1)], ck["mv"], ck["sv"], ck["mc"], ck["sc"], max_n=len(vessel_rows))
        with torch.no_grad():
            s = m(V, C, P)[0]
            return torch.softmax(s, 0).numpy()[:len(vessel_rows)]
    except Exception as exc:
        print(f"[attention ranker] skipped: {exc}")
        return None
