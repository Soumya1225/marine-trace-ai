"""PyTorch datasets for oil spill segmentation.

SOSDataset (Deep-SAR Oil Spill, Zhu et al.) - used for training in this project:
    Datasets/raw/sos/train/sentinel/<id>_sat.jpg + <id>_mask.png      (pairs in one folder)
    Datasets/raw/sos/test/sentinel/sat/<id>_sat.jpg and gt/<id>_mask.png
    Masks are white (oil) on black (sea) -> 2 classes: 0 sea, 1 oil.

OilSpillDataset (Krestenitis et al. 2019) - optional, 5 classes:
    train/images/*.jpg   train/labels_1D/*.png   (same for test/)
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(img_u8):
    """HxW or HxWx3 uint8 -> 3xHxW float tensor normalised like ImageNet."""
    if img_u8.ndim == 2:
        img_u8 = np.repeat(img_u8[..., None], 3, axis=2)
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def pad_to_multiple(img, mask=None, multiple=32):
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (img.ndim - 2)
    img = np.pad(img, pad, mode="reflect")
    if mask is not None:
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=0)
    return img, mask, (h, w)


SOS_CLASSES = ["sea", "oil_spill"]


def _augment(img, mask, crop=None):
    """Random crop (if larger than crop), flips, 90-degree rotation (square images) and brightness jitter."""
    if crop is not None:
        ch, cw = crop
        h, w = mask.shape
        if h > ch and w > cw:
            y = np.random.randint(0, h - ch + 1)
            x = np.random.randint(0, w - cw + 1)
            img, mask = img[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]
    if np.random.rand() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if np.random.rand() < 0.5:
        img, mask = img[::-1], mask[::-1]
    if mask.shape[0] == mask.shape[1] and np.random.rand() < 0.5:
        k = np.random.randint(1, 4)
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    if np.random.rand() < 0.5:
        alpha, beta = np.random.uniform(0.8, 1.2), np.random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, mask


class SOSDataset(Dataset):
    classes = SOS_CLASSES

    def __init__(self, root, split="train", sensor="sentinel", augment=True, ids=None, crop=None):
        base = Path(root) / split / sensor
        if (base / "sat").exists():
            sat_dir, gt_dir = base / "sat", base / "gt"
        else:
            sat_dir = gt_dir = base
        masks = {p.name.rsplit("_mask", 1)[0]: p for p in gt_dir.glob("*_mask.*")}
        pairs = []
        for img in sorted(sat_dir.glob("*_sat.*")):
            key = img.name.rsplit("_sat", 1)[0]
            if key in masks:
                pairs.append((img, masks[key]))
        if not pairs:
            raise FileNotFoundError(f"No <id>_sat / <id>_mask pairs found in {base}")
        self.pairs = [pairs[i] for i in ids] if ids is not None else pairs
        self.augment = augment
        self.crop = crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        mask = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.int64)
        if self.augment:
            img, mask = _augment(img, mask, self.crop)
        img, mask, _ = pad_to_multiple(np.ascontiguousarray(img), np.ascontiguousarray(mask))
        return normalize(img), torch.from_numpy(np.ascontiguousarray(mask))


class OilSpillDataset(Dataset):
    def __init__(self, root, split="train", crop=(384, 640), augment=True, ids=None):
        root = Path(root) / split
        self.images = sorted((root / "images").glob("*.jpg")) + sorted((root / "images").glob("*.png"))
        self.label_dir = root / "labels_1D"
        if not self.images:
            raise FileNotFoundError(f"No images found in {root / 'images'}")
        if ids is not None:
            self.images = [self.images[i] for i in ids]
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def _label_path(self, img_path):
        for ext in (".png", ".jpg"):
            p = self.label_dir / (img_path.stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"Label not found for {img_path.name}")

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.images[idx]).convert("RGB"))
        mask = np.asarray(Image.open(self._label_path(self.images[idx])))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.int64)

        if self.augment:
            img, mask = self._random_crop(img, mask)
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if np.random.rand() < 0.5:
                img, mask = img[::-1], mask[::-1]
            if np.random.rand() < 0.5:  # brightness / contrast jitter
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-20, 20)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        else:
            img, mask, _ = pad_to_multiple(img, mask)

        return normalize(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask))

    def _random_crop(self, img, mask):
        ch, cw = self.crop
        h, w = mask.shape
        if h < ch or w < cw:
            img, mask, _ = pad_to_multiple(img, mask, multiple=max(ch, cw))
            h, w = mask.shape
        y = np.random.randint(0, h - ch + 1)
        x = np.random.randint(0, w - cw + 1)
        return img[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]
