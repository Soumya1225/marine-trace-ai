"""Train an oil spill segmentation model (run on Google Colab / Kaggle GPU).

Examples (SOS dataset, the default):
    python -m ml.segmentation.train --arch fcn  --epochs 30
    python -m ml.segmentation.train --arch unet --encoder resnet34 --epochs 40
    python -m ml.segmentation.train --arch deeplabv3plus --encoder resnet50 --epochs 40
    python -m ml.segmentation.train --arch segformer --encoder mit_b1 --epochs 40      (transformer)
    python -m ml.segmentation.train --arch swin --encoder swin_tiny --epochs 25        (Swin transformer)
Krestenitis dataset instead:
    python -m ml.segmentation.train --dataset krestenitis --arch unet

Each run saves models/<run_id>.pth, models/metrics/<run_id>.json and appends a row to
docs/experiments.csv. The best run (highest oil IoU) is copied to models/segmentation_best.pth.
"""
import argparse
import csv
import json
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ml.config import CLASSES, DOCS_DIR, METRICS_DIR, MODELS_DIR, RAW_DIR
from ml.segmentation.dataset import SOS_CLASSES, OilSpillDataset, SOSDataset
from ml.segmentation.model import build_model


def dice_loss(logits, target, n_classes, eps=1.0):
    probs = logits.softmax(1)
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def class_weights(dataset, n_classes, max_items=200):
    counts = np.zeros(n_classes)
    for i in range(min(len(dataset), max_items)):
        _, m = dataset[i]
        counts += np.bincount(m.numpy().ravel(), minlength=n_classes)[:n_classes]
    freq = counts / counts.sum()
    w = np.median(freq[freq > 0]) / np.maximum(freq, 1e-6)
    return torch.tensor(np.clip(w, 0.1, 10), dtype=torch.float32)


def confusion(pred, target, n):
    k = (target >= 0) & (target < n)
    return np.bincount(n * target[k] + pred[k], minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm, classes=CLASSES):
    oil = classes.index("oil_spill")
    look = classes.index("look_alike") if "look_alike" in classes else None
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return {
        "per_class_iou": {c: round(float(v), 4) for c, v in zip(classes, iou)},
        "mIoU": round(float(iou.mean()), 4),
        "oil_iou": round(float(iou[oil]), 4),
        "lookalike_iou": round(float(iou[look]), 4) if look is not None else None,
        "oil_precision": round(float(precision[oil]), 4),
        "oil_recall": round(float(recall[oil]), 4),
        "oil_f1": round(float(f1[oil]), 4),
        "pixel_accuracy": round(float(tp.sum() / max(cm.sum(), 1)), 4),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, classes):
    model.eval()
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion(pred.ravel(), y.numpy().ravel(), n)
    return metrics_from_confusion(cm, classes)


def build_datasets(name, root, val_split, sensor="sentinel"):
    """Return (train_ds, val_ds, test_ds or None, classes)."""
    rng = np.random.default_rng(42)
    if name == "sos":
        full = SOSDataset(root, "train", sensor, augment=False)
        idx = rng.permutation(len(full))
        n_val = max(1, int(len(idx) * val_split))
        train_ds = SOSDataset(root, "train", sensor, augment=True, ids=idx[n_val:])
        val_ds = SOSDataset(root, "train", sensor, augment=False, ids=idx[:n_val])
        try:
            test_ds = SOSDataset(root, "test", sensor, augment=False)
        except FileNotFoundError:
            test_ds = None
        return train_ds, val_ds, test_ds, SOS_CLASSES
    full = OilSpillDataset(root, "train", augment=False)
    idx = rng.permutation(len(full))
    n_val = max(1, int(len(idx) * val_split))
    train_ds = OilSpillDataset(root, "train", augment=True, ids=idx[n_val:])
    val_ds = OilSpillDataset(root, "train", augment=False, ids=idx[:n_val])
    try:
        test_ds = OilSpillDataset(root, "test", augment=False)
    except FileNotFoundError:
        test_ds = None
    return train_ds, val_ds, test_ds, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sos", choices=["sos", "krestenitis"])
    ap.add_argument("--data", default=None, help="dataset folder (default Datasets/raw/sos or Datasets/raw/krestenitis)")
    ap.add_argument("--sensor", default="sentinel", help="SOS only: sentinel or palsar")
    ap.add_argument("--arch", default="unet", choices=["fcn", "unet", "deeplabv3plus", "segformer", "swin"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = args.data or str(RAW_DIR / args.dataset)
    train_ds, val_ds, test_ds, classes = build_datasets(args.dataset, data_root, args.val_split, args.sensor)
    n = len(classes)
    run_id = f"{args.dataset}_{args.arch}_{args.encoder if args.arch != 'fcn' else 'base'}_{time.strftime('%Y%m%d_%H%M')}"
    print(f"Run {run_id} on {device}: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds) if test_ds else 0} test images")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = build_model(args.arch, args.encoder, n_classes=n).to(device)
    weights = class_weights(val_ds, n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"{run_id}.pth"
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights) + dice_loss(logits, y, n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate_loader(model, val_dl, device, classes)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "mIoU": val["mIoU"], "oil_iou": val["oil_iou"]})
        print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} val mIoU {val['mIoU']:.4f} oil IoU {val['oil_iou']:.4f}")
        if val["oil_iou"] > best:
            best = val["oil_iou"]
            torch.save({"arch": args.arch, "encoder": args.encoder, "classes": classes, "dataset": args.dataset,
                        "state_dict": model.state_dict(), "val": val, "epoch": epoch}, ckpt_path)

    # final evaluation of the best checkpoint on the official test split
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if test_ds is not None:
        test = evaluate_loader(model, DataLoader(test_ds, batch_size=args.batch), device, classes)
    else:
        test = ckpt["val"]

    record = {"run_id": run_id, "date": str(date.today()), "dataset": args.dataset, "arch": args.arch, "encoder": args.encoder,
              "lr": args.lr, "epochs": args.epochs, "batch": args.batch, "best_epoch": ckpt["epoch"],
              "test": test, "history": history, "notes": args.notes}
    (METRICS_DIR / f"{run_id}.json").write_text(json.dumps(record, indent=2))

    summary_path = METRICS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary.append({k: record[k] for k in ("run_id", "date", "arch", "encoder", "lr", "epochs")} |
                   {k: test[k] for k in ("mIoU", "oil_iou", "lookalike_iou", "oil_precision", "oil_recall", "oil_f1")})
    summary_path.write_text(json.dumps(summary, indent=2))

    best_overall = max(summary, key=lambda r: r["oil_iou"])
    if best_overall["run_id"] == run_id:
        shutil.copy(ckpt_path, MODELS_DIR / "segmentation_best.pth")
        print("New best model -> models/segmentation_best.pth")

    exp = DOCS_DIR / "experiments.csv"
    exp.parent.mkdir(parents=True, exist_ok=True)
    new_file = not exp.exists() or exp.stat().st_size == 0
    with open(exp, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "run_id", "model", "encoder", "lr", "epochs", "batch_size", "mIoU", "oil_IoU", "lookalike_IoU", "notes"])
        w.writerow([date.today(), run_id, args.arch, args.encoder, args.lr, args.epochs, args.batch,
                    test["mIoU"], test["oil_iou"], test["lookalike_iou"], args.notes])
    print(json.dumps({k: test[k] for k in ("mIoU", "oil_iou", "lookalike_iou", "oil_f1")}, indent=2))


if __name__ == "__main__":
    main()
