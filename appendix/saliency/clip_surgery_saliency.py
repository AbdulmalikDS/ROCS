# CLIP Surgery: https://arxiv.org/abs/2304.05653
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLIP_SURGERY_REPO = _PROJECT_ROOT / "third_party" / "CLIP_Surgery"


def _load_clip_surgery(model_name: str = "CS-ViT-L/14",
                       device: str = "cuda"):
    if str(_CLIP_SURGERY_REPO) not in sys.path:
        sys.path.insert(0, str(_CLIP_SURGERY_REPO))
    # CLIP Surgery needs its own fork of the clip package.
    for mod in [m for m in list(sys.modules) if m.startswith("clip")]:
        del sys.modules[mod]
    import clip as cs_clip
    model, preprocess = cs_clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, cs_clip


class CLIPSurgerySaliencyExtractor:
    def __init__(
        self,
        model_name: str = "CS-ViT-L/14",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess, self.cs_clip = _load_clip_surgery(
            model_name=model_name, device=self.device,
        )
        self.model_name = model_name
        self.input_size = self.model.visual.input_resolution
        self.patch_size = self.model.visual.conv1.kernel_size[0]
        self.grid = self.input_size // self.patch_size
        self._img_feat_cache: dict[int, torch.Tensor] = {}
        self._redundant_feat: torch.Tensor | None = None
        self._text_feat_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def extract(self, image: Image.Image, target_size: int = 384) -> np.ndarray:
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(x)
        cls = feats[:, 0, :]
        patch_feats = feats[:, 1:, :]
        cls_n = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
        patch_n = patch_feats / (patch_feats.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (cls_n.unsqueeze(1) * patch_n).sum(dim=-1)
        sim_np = sim[0].detach().cpu().numpy().astype(np.float32)
        sim_map = sim_np.reshape(self.grid, self.grid)
        lo, hi = float(sim_map.min()), float(sim_map.max())
        sim_map = (sim_map - lo) / (hi - lo + 1e-8)
        out = cv2.resize(sim_map, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)

    @torch.no_grad()
    def _img_feat_for_image(self, image: Image.Image) -> torch.Tensor:
        key = id(image)
        cached = self._img_feat_cache.get(key)
        if cached is not None:
            return cached
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(x)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        self._img_feat_cache[key] = img_feat
        return img_feat

    @torch.no_grad()
    def _get_redundant_feat(self) -> torch.Tensor:
        if self._redundant_feat is None:
            self._redundant_feat = self.cs_clip.encode_text_with_prompt_ensemble(
                self.model, [""], self.device)
        return self._redundant_feat

    @torch.no_grad()
    def _get_text_feat(self, text_query: str) -> torch.Tensor:
        cached = self._text_feat_cache.get(text_query)
        if cached is not None:
            return cached
        feat = self.cs_clip.encode_text_with_prompt_ensemble(
            self.model, [text_query], self.device)
        self._text_feat_cache[text_query] = feat
        return feat

    def clear_cache(self) -> None:
        self._img_feat_cache.clear()
        self._text_feat_cache.clear()

    @torch.no_grad()
    def extract_query(self, image: Image.Image, text_query: str,
                      target_size: int = 384) -> np.ndarray:
        img_feat = self._img_feat_for_image(image)
        text_feat = self._get_text_feat(text_query)
        # Empty-prompt features prevent single-class subtraction from erasing the signal.
        redundant_feat = self._get_redundant_feat()
        sim = self.cs_clip.clip_feature_surgery(
            img_feat, text_feat, redundant_feat)
        sim_map = self.cs_clip.get_similarity_map(
            sim[:, 1:, :], (target_size, target_size))
        return sim_map[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
