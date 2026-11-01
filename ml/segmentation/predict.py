"""Oil spill detector used by the pipeline.

If a trained checkpoint exists (models/segmentation_best.pth) it runs the deep model tile by tile.
Otherwise it falls back to a classic adaptive dark-spot detector, so the full system can be
demonstrated before training finishes.
"""
import numpy as np
from scipy import ndimage

from ml.config import (CLASSES, ENSEMBLE_MODELS, LOOKALIKE_CLASS, MIN_SPILL_PIXELS, MODELS_DIR, OIL_CLASS,
                       SEGMENTATION_WEIGHTS, TILE_OVERLAP, TILE_SIZE, USE_ENSEMBLE)


class SpillDetector:
    def __init__(self, weights=SEGMENTATION_WEIGHTS, device=None, ensemble=USE_ENSEMBLE):
        self.mode = "threshold"
        self.model = None
        self.models = []
        self.last_uncertainty = None
        self.arch = "adaptive-threshold baseline"
        try:
            from pathlib import Path
            files = []
            if ensemble:
                for pattern in ENSEMBLE_MODELS:
                    found = sorted(Path(MODELS_DIR).glob(pattern), key=lambda p: p.stat().st_mtime)
                    if found:
                        files.append(found[-1])
            if len(files) >= 2:
                import torch
                from ml.segmentation.model import load_checkpoint
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                names = []
                for f in files:
                    m, ck = load_checkpoint(f, self.device)
                    self.models.append(m)
                    names.append({"unet": "U-Net", "deeplabv3plus": "DeepLabV3+", "segformer": "SegFormer", "swin": "Swin-Tiny"}.get(ck["arch"], ck["arch"]))
                self.n_classes = len(ck.get("classes", CLASSES))
                self.mode = "ensemble"
                self.arch = "Deep ensemble: " + " + ".join(names)
            elif Path(weights).exists():
                import torch
                from ml.segmentation.model import load_checkpoint
                self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                self.model, ckpt = load_checkpoint(weights, self.device)
                self.n_classes = len(ckpt.get("classes", CLASSES))
                self.mode = "deep"
                self.arch = f"{ckpt['arch']} ({ckpt.get('encoder', '')})"
        except Exception as exc:  # torch missing or bad checkpoint -> baseline
            print(f"[SpillDetector] using threshold baseline ({exc})")

    def predict(self, image_u8):
        """Return (class_map uint8, oil_probability float32) for an HxW uint8 image.

        In ensemble mode `self.last_uncertainty` holds the per-pixel spread (std) of the models' oil probability."""
        self.last_uncertainty = None
        if self.mode == "ensemble":
            all_probs = [self._probs(m, image_u8) for m in self.models]
            stack = np.stack(all_probs)                       # models x classes x H x W
            probs = stack.mean(0)
            self.last_uncertainty = stack[:, OIL_CLASS].std(0).astype(np.float32)
            class_map = probs.argmax(0).astype(np.uint8)
            class_map[image_u8 == 0] = 0
            return class_map, probs[OIL_CLASS].astype(np.float32)
        if self.mode == "deep":
            return self._predict_deep(image_u8)
        return self._predict_threshold(image_u8)

    # ---------------- deep model ----------------
    def _predict_deep(self, image_u8):
        probs = self._probs(self.model, image_u8)
        class_map = probs.argmax(0).astype(np.uint8)
        class_map[(image_u8 == 0)] = 0
        return class_map, probs[OIL_CLASS]

    def _probs(self, model, image_u8):
        """Tiled softmax probabilities (classes x H x W) of one model."""
        import torch
        from ml.segmentation.dataset import normalize, pad_to_multiple

        img, _, (h, w) = pad_to_multiple(image_u8, None, multiple=32)
        H, W = img.shape
        n = self.n_classes
        probs = np.zeros((n, H, W), np.float32)
        counts = np.zeros((H, W), np.float32)
        step = TILE_SIZE - TILE_OVERLAP
        ys = list(range(0, max(H - TILE_SIZE, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - TILE_SIZE, 0) + 1, step)) or [0]
        if ys[-1] + TILE_SIZE < H:
            ys.append(H - TILE_SIZE)
        if xs[-1] + TILE_SIZE < W:
            xs.append(W - TILE_SIZE)
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
                    t = normalize(tile).unsqueeze(0).to(self.device)
                    p = model(t).softmax(1)[0].cpu().numpy()
                    th, tw = tile.shape
                    probs[:, y:y + th, x:x + tw] += p[:, :th, :tw]
                    counts[y:y + th, x:x + tw] += 1
        probs /= np.maximum(counts, 1)
        return probs[:, :h, :w]

    # ---------------- baseline ----------------
    def _predict_threshold(self, image_u8, window=301, k=1.5, min_pixels=max(MIN_SPILL_PIXELS, 200)):
        valid = image_u8 > 0
        img = ndimage.gaussian_filter(image_u8.astype(np.float32), 3.5)
        mean = ndimage.uniform_filter(img, window)
        sq = ndimage.uniform_filter(img * img, window)
        std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
        global_low = np.percentile(img[valid], 35) if valid.any() else 0
        dark = (img < mean - k * std) & (img < global_low) & valid
        dark = ndimage.binary_opening(dark, iterations=2)
        dark = ndimage.binary_closing(dark, iterations=3)
        labels, n = ndimage.label(dark)
        class_map = np.zeros(img.shape, np.uint8)
        if n:
            sizes = ndimage.sum(dark, labels, range(1, n + 1))
            for i, size in enumerate(sizes, start=1):
                if size < min_pixels:
                    continue
                region = labels == i
                # real slicks damp backscatter strongly (several dB); weak dips are speckle
                if img[region].mean() > 0.5 * mean[region].mean():
                    continue
                # very round dark blobs are often low-wind look-alikes; elongated ones look like spills
                class_map[region] = OIL_CLASS if _elongation(region) >= 2.0 else LOOKALIKE_CLASS
        prob = np.clip((mean - img) / (3 * std), 0, 1).astype(np.float32)
        prob[class_map != OIL_CLASS] *= 0.3
        return class_map, prob


def _elongation(region):
    rows, cols = np.nonzero(region)
    if len(rows) < 5:
        return 1.0
    cov = np.cov(np.vstack([cols, rows]))
    ev = np.sort(np.linalg.eigvalsh(cov))
    return float(np.sqrt(ev[1] / max(ev[0], 1e-6)))
