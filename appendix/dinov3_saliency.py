"""
Saliency extractor based on DINOv3 ViT-L/16 with register tokens.

Reference implementations:
  - DINOv3 (Meta, 2024): code in `third_party/dinov3/` (this repo).
    Hub entrypoint `dinov3.hub.backbones.dinov3_vitl16` with default
    `Weights.LVD1689M` pretrained checkpoint.
  - DINO original visualisation: facebookresearch/dino,
    visualize_attention.py — last-layer CLS attention to patches.
  - CLIP-DINOiser (Wysoczanska et al. CVPR 2024, arXiv 2312.12359):
    uses CLS-to-patch cosine similarity on output features rather than
    raw attention; cleaner for downstream pooling.

We follow CLIP-DINOiser's choice: cosine similarity between the CLS
feature and each patch feature at the last layer's output (post-norm).
DINOv3's `SelfAttention` uses
`torch.nn.functional.scaled_dot_product_attention` and does not return
attention weights, so feature-cosine is the practical signal anyway.

The DINOv3 ViT-L/16 LVD-1689M weights are gated via Meta's licence
agreement; we rely on torch.hub finding them in the local cache at
``~/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-*.pth``.

Output: 2D saliency map in [0, 1] of size (target_size, target_size),
ready for ``RegionDetector.detect(...)``. For crop selection we resize the
full image to a square instead of center-cropping, so saliency coordinates
stay aligned with the original image used by the detector.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DINOV3_REPO = _PROJECT_ROOT / "third_party" / "dinov3"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dinov3(model_name: str = "dinov3_vitl16", device: str = "cuda"):
    """Load DINOv3 from the local third_party copy. Tries torch.hub first
    (which uses ``~/.cache/torch/hub/checkpoints``); falls back to
    direct import if the hub-load fails."""
    if str(_DINOV3_REPO) not in sys.path:
        sys.path.insert(0, str(_DINOV3_REPO))
    try:
        return torch.hub.load(
            str(_DINOV3_REPO), model_name,
            source="local", trust_repo=True, pretrained=True,
        ).to(device).eval()
    except Exception:
        # Direct import fallback. Same code path under the hood.
        from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16
        loader = {"dinov3_vitb16": dinov3_vitb16,
                  "dinov3_vitl16": dinov3_vitl16}[model_name]
        return loader(pretrained=True, check_hash=False).to(device).eval()


class DINOv3SaliencyExtractor:
    """Produce a 2D saliency map for an input image using a frozen
    DINOv3 ViT backbone. Saliency = cosine similarity between the CLS
    feature and each patch feature at the last layer's output.

    DINOv3 register / storage tokens are excluded automatically:
    ``get_intermediate_layers`` returns only the patch tokens after
    the CLS and storage-token slices.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        device: Optional[str] = None,
        input_size: int = 224,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = _load_dinov3(model_name, device=self.device)
        self.patch_size = self.model.patch_size
        self.n_storage_tokens = getattr(self.model, "n_storage_tokens", 0)
        # Round input_size to a multiple of patch_size.
        if self.input_size % self.patch_size != 0:
            self.input_size = (self.input_size // self.patch_size) * self.patch_size
        self.grid = self.input_size // self.patch_size
        # Resize the full frame to a square. Center-cropping would make the
        # saliency grid disagree with RegionDetector's full-image coordinates.
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @torch.no_grad()
    def extract(self, image: Image.Image, target_size: int = 384) -> np.ndarray:
        """Return saliency [target_size, target_size] in [0, 1]."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # n=1, reshape=True, return_class_token=True
        # -> tuple of length 1: ((patch_features [1, D, h, w], cls_token [1, D]),)
        # Storage / register tokens are already excluded inside
        # ``get_intermediate_layers`` via the n_storage_tokens patch slice.
        outputs = self.model.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True, norm=True,
        )
        feats, cls = outputs[0]  # [1, D, h, w], [1, D]
        feats_flat = feats.reshape(1, feats.shape[1], -1)             # [1, D, N]
        feats_n = feats_flat / (feats_flat.norm(dim=1, keepdim=True) + 1e-8)
        cls_n = cls / (cls.norm(dim=1, keepdim=True) + 1e-8)          # [1, D]
        sim = (cls_n.unsqueeze(-1) * feats_n).sum(dim=1)              # [1, N]
        h, w = feats.shape[-2:]
        sim_map = sim.reshape(h, w).detach().cpu().numpy().astype(np.float32)
        lo, hi = float(sim_map.min()), float(sim_map.max())
        sim_map = (sim_map - lo) / (hi - lo + 1e-8)
        out = cv2.resize(sim_map, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)
