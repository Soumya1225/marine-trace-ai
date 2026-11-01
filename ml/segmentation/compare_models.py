"""Compare all trained oil spill detection models on the SOS test set.

    python -m ml.segmentation.compare_models                 (all test images)
    python -m ml.segmentation.compare_models --max-images 200  (quick check)

For every model (FCN, U-Net, DeepLabV3+, SegFormer, Swin) and for the deployed ensemble it measures
pixel accuracy, oil precision / recall / F1, oil IoU, mean IoU, ROC curve + ROC-AUC (oil vs sea),
model size and speed. Output: models/metrics/model_comparison.json (shown on the Model performance page).
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ml.config import ENSEMBLE_MODELS, METRICS_DIR, MODELS_DIR, RAW_DIR
from ml.segmentation.dataset import SOSDataset
from ml.segmentation.model import load_checkpoint
from ml.segmentation.train import confusion, metrics_from_confusion

LABELS = {"fcn": "FCN (baseline)", "unet": "U-Net", "deeplabv3plus": "DeepLabV3+", "segformer": "SegFormer", "swin": "Swin Transformer"}
PATTERNS = ["sos_fcn_*.pth", "sos_unet_*.pth", "sos_deeplabv3plus_*.pth", "sos_segformer_*.pth", "sos_swin_*.pth"]
BINS = 1000


def latest(pattern):
    found = sorted(Path(MODELS_DIR).glob(pattern), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


class Scorer:
    """Collects a confusion matrix and oil-probability histograms (for the ROC curve) for one model."""

    def __init__(self, n):
        self.n = n
        self.cm = np.zeros((n, n), dtype=np.int64)
        self.pos = np.zeros(BINS, dtype=np.int64)
        self.neg = np.zeros(BINS, dtype=np.int64)
        self.seconds = 0.0

    def add(self, probs, gt, oil):
        pred = probs.argmax(0)
        self.cm += confusion(pred.ravel(), gt.ravel(), self.n)
        b = np.clip((probs[oil] * BINS).astype(np.int64), 0, BINS - 1).ravel()
        is_oil = gt.ravel() == oil
        self.pos += np.bincount(b[is_oil], minlength=BINS)
        self.neg += np.bincount(b[~is_oil], minlength=BINS)

    def roc(self):
        tp = np.concatenate([[0], np.cumsum(self.pos[::-1])])     # threshold from 1 down to 0
        fp = np.concatenate([[0], np.cumsum(self.neg[::-1])])
        tpr = tp / max(self.pos.sum(), 1)
        fpr = fp / max(self.neg.sum(), 1)
        auc = float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
        keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
        return auc, [round(float(v), 4) for v in fpr[keep]], [round(float(v), 4) for v in tpr[keep]]


@torch.no_grad()
def softmax(model, x):
    return torch.softmax(model(x), 1)[0].cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW_DIR / "sos"))
    ap.add_argument("--max-images", type=int, default=0, help="0 = whole test set")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for pattern in PATTERNS:
        f = latest(pattern)
        if f is None:
            print(f"  not found, skipped: {pattern}")
            continue
        try:
            m, ck = load_checkpoint(f, device)
        except Exception as exc:
            print(f"  could not load {f.name}: {exc}")
            continue
        models.append({"key": ck["arch"], "label": LABELS.get(ck["arch"], ck["arch"]), "arch": ck["arch"],
                       "encoder": "–" if ck["arch"] == "fcn" else ck.get("encoder", ""), "file": f.name,
                       "params_m": round(sum(p.numel() for p in m.parameters()) / 1e6, 1),
                       "model": m, "classes": ck.get("classes", ["sea", "oil_spill"])})
    if not models:
        raise SystemExit(f"No trained models found in {MODELS_DIR}")
    ens_files = {latest(p).name for p in ENSEMBLE_MODELS if latest(p)}
    ens_members = [m for m in models if m["file"] in ens_files]
    classes = models[0]["classes"]
    oil = classes.index("oil_spill")
    n = len(classes)

    ds = SOSDataset(args.data, "test", augment=False)
    total = len(ds) if args.max_images <= 0 else min(len(ds), args.max_images)
    print(f"Comparing {len(models)} models{' + ensemble' if len(ens_members) >= 2 else ''} on {total} SOS test images ({device})")
    scorers = {m["key"]: Scorer(n) for m in models}
    ens = Scorer(n) if len(ens_members) >= 2 else None

    start = time.time()
    for i in range(total):
        x, y = ds[i]
        x, gt = x.unsqueeze(0).to(device), y.numpy()
        probs = {}
        for m in models:
            t = time.perf_counter()
            probs[m["key"]] = softmax(m["model"], x)
            scorers[m["key"]].seconds += time.perf_counter() - t
            scorers[m["key"]].add(probs[m["key"]], gt, oil)
        if ens:
            ens.add(np.mean([probs[m["key"]] for m in ens_members], axis=0), gt, oil)
        if (i + 1) % 50 == 0 or i + 1 == total:
            el = time.time() - start
            print(f"  {i + 1}/{total} images, {el / 60:.1f} min, about {el / (i + 1) * (total - i - 1) / 60:.1f} min left")

    rows = []
    entries = [(m, scorers[m["key"]]) for m in models]
    if ens:
        ens.seconds = sum(scorers[m["key"]].seconds for m in ens_members)
        entries.append(({"key": "ensemble", "label": "Ensemble (deployed)", "arch": "ensemble",
                         "encoder": " + ".join(LABELS[m["arch"]] for m in ens_members), "file": "",
                         "params_m": round(sum(m["params_m"] for m in ens_members), 1)}, ens))
    for m, s in entries:
        met = metrics_from_confusion(s.cm, classes)
        auc, fpr, tpr = s.roc()
        rows.append({k: m[k] for k in ("key", "label", "arch", "encoder", "file", "params_m")} | {
            "accuracy": met["pixel_accuracy"], "precision": met["oil_precision"], "recall": met["oil_recall"],
            "f1": met["oil_f1"], "iou": met["oil_iou"], "miou": met["mIoU"], "auc": round(auc, 4),
            "ms_per_image": round(s.seconds / max(total, 1) * 1000, 1), "confusion_matrix": met["confusion_matrix"],
            "roc": {"fpr": fpr, "tpr": tpr}})

    best = max(rows, key=lambda r: (r["f1"], r["auc"]))
    best_single = max((r for r in rows if r["key"] != "ensemble"), key=lambda r: (r["f1"], r["auc"]))
    out = {"created": datetime.now().isoformat(timespec="seconds"), "dataset": "SOS test set (Sentinel-1)",
           "images": total, "pixels": int(sum(map(sum, rows[0]["confusion_matrix"]))),
           "device": device, "best": best["key"], "best_single": best_single["key"], "models": rows}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'Model':<22}{'Accuracy':>9}{'Precision':>10}{'Recall':>8}{'F1':>7}{'IoU':>7}{'AUC':>7}{'ms/img':>8}")
    for r in rows:
        print(f"{r['label']:<22}{r['accuracy']:>9.3f}{r['precision']:>10.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}{r['iou']:>7.3f}{r['auc']:>7.3f}{r['ms_per_image']:>8.0f}")
    print(f"\nBest overall: {best['label']}  |  best single model: {best_single['label']}")
    print(f"Saved {METRICS_DIR / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
