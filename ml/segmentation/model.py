"""Model factory: FCN baseline, U-Net, DeepLabV3+ (CNNs), SegFormer and Swin-Tiny (transformers)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.config import CLASSES


class SimpleFCN(nn.Module):
    """Small encoder-decoder CNN used as the baseline in the model comparison."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.enc1, self.enc2, self.enc3 = block(3, 32), block(32, 64), block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


SWIN_NAMES = {"swin_tiny": "swin_tiny_patch4_window7_224", "swin_small": "swin_small_patch4_window7_224"}


class SwinSeg(nn.Module):
    """Swin Transformer encoder (shifted-window self-attention, Liu et al., 2021) with a light FPN-style decoder.

    The Swin backbone from timm expects 224 x 224 input, so each image is resized to 224 inside the model and
    the prediction is resized back to the original size. The four feature maps (1/4 .. 1/32 scale) are projected
    to 128 channels, upsampled to 1/4 scale, concatenated and fused into the class map.
    """

    def __init__(self, encoder="swin_tiny", pretrained=True, n_classes=len(CLASSES), size=224, dim=128):
        super().__init__()
        import timm
        self.size = size
        self.encoder = timm.create_model(SWIN_NAMES.get(encoder, encoder), pretrained=pretrained,
                                         features_only=True, out_indices=(0, 1, 2, 3))
        self.channels = self.encoder.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in self.channels])
        self.fuse = nn.Sequential(nn.Conv2d(4 * dim, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(dim, n_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        z = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        feats = self.encoder(z)
        outs, target = [], None
        for f, lat, c in zip(feats, self.lateral, self.channels):
            if f.shape[1] != c and f.shape[-1] == c:        # timm Swin returns channels-last (NHWC)
                f = f.permute(0, 3, 1, 2).contiguous()
            f = lat(f)
            target = target or f.shape[-2:]
            outs.append(F.interpolate(f, size=target, mode="bilinear", align_corners=False))
        y = self.head(self.fuse(torch.cat(outs, 1)))
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


def build_model(arch="unet", encoder="resnet34", pretrained=True, n_classes=len(CLASSES)):
    arch = arch.lower()
    if arch == "fcn":
        return SimpleFCN(n_classes)
    import segmentation_models_pytorch as smp
    weights = "imagenet" if pretrained else None
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=n_classes)
    if arch in ("deeplabv3plus", "deeplab"):
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=n_classes)
    if arch == "segformer":
        # transformer model: Mix Transformer (MiT) encoder + lightweight all-MLP decoder (Xie et al., 2021)
        enc = encoder if encoder.startswith("mit_") else "mit_b1"
        return smp.Segformer(encoder_name=enc, encoder_weights=weights, in_channels=3, classes=n_classes)
    if arch == "swin":
        return SwinSeg(encoder if encoder.startswith("swin") else "swin_tiny", pretrained=pretrained, n_classes=n_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], ckpt.get("encoder", "resnet34"), pretrained=False,
                        n_classes=len(ckpt.get("classes", CLASSES)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
