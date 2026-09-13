# DINOv3: https://arxiv.org/abs/2508.10104
# CLS-to-patch cosine follows CLIP-DINOiser: https://arxiv.org/abs/2312.12359
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

_DINOV3_REPO = Path(__file__).resolve().parent / "third_party" / "dinov3"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dinov3(model_name: str = "dinov3_vitl16", device: str = "cuda"):
    if str(_DINOV3_REPO) not in sys.path:
        sys.path.insert(0, str(_DINOV3_REPO))
    try:
        return torch.hub.load(
            str(_DINOV3_REPO), model_name,
            source="local", trust_repo=True, pretrained=True,
        ).to(device).eval()
    except Exception:  # noqa: BLE001 - hub.load fails several ways; any of them means fall back
        from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16
        loader = {"dinov3_vitb16": dinov3_vitb16,
                  "dinov3_vitl16": dinov3_vitl16}[model_name]
        return loader(pretrained=True, check_hash=False).to(device).eval()


class DINOv3SaliencyExtractor:
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        device: str | None = None,
        input_size: int = 224,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = _load_dinov3(model_name, device=self.device)
        self.patch_size = self.model.patch_size
        self.n_storage_tokens = getattr(self.model, "n_storage_tokens", 0)
        if self.input_size % self.patch_size != 0:
            self.input_size = (self.input_size // self.patch_size) * self.patch_size
        self.grid = self.input_size // self.patch_size
        # Keep saliency coordinates aligned with the full image.
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract(self, image: Image.Image, target_size: int = 384) -> np.ndarray:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # The backbone excludes register tokens from returned patches.
        outputs = self.model.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True, norm=True,
        )
        feats, cls = outputs[0]
        feats_flat = feats.reshape(1, feats.shape[1], -1)
        feats_n = feats_flat / (feats_flat.norm(dim=1, keepdim=True) + 1e-8)
        cls_n = cls / (cls.norm(dim=1, keepdim=True) + 1e-8)
        sim = (cls_n.unsqueeze(-1) * feats_n).sum(dim=1)
        h, w = feats.shape[-2:]
        sim_map = sim.reshape(h, w).detach().cpu().numpy().astype(np.float32)
        lo, hi = float(sim_map.min()), float(sim_map.max())
        sim_map = (sim_map - lo) / (hi - lo + 1e-8)
        out = cv2.resize(sim_map, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)
