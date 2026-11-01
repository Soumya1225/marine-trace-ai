"""Evaluate a trained checkpoint on the test split and save sample prediction images.

    python -m ml.segmentation.evaluate --weights models/segmentation_best.pth --samples 8
    (uses the SOS test set by default; add --dataset krestenitis for the other dataset)
Outputs: models/metrics/test_<name>.json and reports/figures/pred_*.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml.config import METRICS_DIR, RAW_DIR, ROOT, SEGMENTATION_WEIGHTS
from ml.segmentation.dataset import OilSpillDataset, SOSDataset
from ml.segmentation.model import load_checkpoint
from ml.segmentation.train import confusion, metrics_from_confusion

# sea, oil, look-alike, ship, land
PALETTE = np.array([[10, 30, 60], [255, 60, 40], [255, 200, 0], [160, 90, 40], [60, 150, 70]], dtype=np.uint8)


def colorize(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(SEGMENTATION_WEIGHTS))
    ap.add_argument("--dataset", default="sos", choices=["sos", "krestenitis"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.weights, device)
    root = args.data or str(RAW_DIR / args.dataset)
    ds = SOSDataset(root, "test", augment=False) if args.dataset == "sos" else OilSpillDataset(root, "test", augment=False)
    images = [p[0] for p in ds.pairs] if args.dataset == "sos" else ds.images
    dl = DataLoader(ds, batch_size=1)
    classes = ckpt["classes"]
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for i, (x, y) in enumerate(dl):
        pred = model(x.to(device)).argmax(1).cpu().numpy()[0]
        gt = y.numpy()[0]
        cm += confusion(pred.ravel(), gt.ravel(), n)
        if i < args.samples:
            img = np.asarray(Image.open(images[i]).convert("RGB"))
            h, w = img.shape[:2]
            row = np.concatenate([img, colorize(gt[:h, :w]), colorize(pred[:h, :w])], axis=1)
            Image.fromarray(row).save(fig_dir / f"pred_{i:02d}.png")

    metrics = metrics_from_confusion(cm, classes)
    name = Path(args.weights).stem
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"test_{name}.json").write_text(json.dumps({"weights": name, "arch": ckpt["arch"], **metrics}, indent=2))
    print(json.dumps({k: metrics[k] for k in ("mIoU", "oil_iou", "oil_f1", "oil_precision", "oil_recall", "pixel_accuracy")}, indent=2))
    print(f"Sample images (input | ground truth | prediction) saved to {fig_dir}")


if __name__ == "__main__":
    main()
