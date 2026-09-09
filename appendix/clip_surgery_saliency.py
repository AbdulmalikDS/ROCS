"""
Saliency extractor based on CLIP Surgery (Li et al., arXiv 2304.05653).

Reference implementation:
  - Repository: https://github.com/xmed-lab/CLIP_Surgery
    cloned to ``third_party/CLIP_Surgery/``.
  - Their custom ``clip`` package implements an in-architecture
    modification of CLIP's last self-attention layer: a dual-branch
    design that produces both standard CLS features and "surgery"
    per-patch features that are cleaned of attention sinks.
  - The model variant ``CS-ViT-L/14`` matches the OpenAI CLIP-L/14
    weights used elsewhere in this paper (no extra training; weights
    downloaded automatically by their ``clip.load``).

The reference localization path is query-conditioned:
``encode_image`` -> ``clip_feature_surgery(image, text, redundant)`` ->
``get_similarity_map``. We expose that as ``extract_query``. The plain
``extract`` method remains a query-free CLS-to-patch cosine diagnostic,
not the CLIP Surgery paper's text-conditioned explanation recipe.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CLIP_SURGERY_REPO = _PROJECT_ROOT / "third_party" / "CLIP_Surgery"


def _load_clip_surgery(model_name: str = "CS-ViT-L/14",
                       device: str = "cuda"):
    """Load CLIP Surgery from third_party/CLIP_Surgery. Their fork ships
    its own ``clip`` package; we put it on sys.path before import."""
    if str(_CLIP_SURGERY_REPO) not in sys.path:
        sys.path.insert(0, str(_CLIP_SURGERY_REPO))
    # Guard against any pre-imported open_clip or stock clip — CLIP
    # Surgery's ``clip`` is a custom fork.
    for mod in [m for m in list(sys.modules) if m.startswith("clip")]:
        del sys.modules[mod]
    import clip as cs_clip
    model, preprocess = cs_clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, cs_clip


class CLIPSurgerySaliencyExtractor:
    """Produce a 2D saliency map for an input image using CLIP Surgery
    (CS-ViT-L/14). Saliency = cosine similarity between the CLS feature
    and each surgery-cleaned patch feature.
    """

    def __init__(
        self,
        model_name: str = "CS-ViT-L/14",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess, self.cs_clip = _load_clip_surgery(
            model_name=model_name, device=self.device,
        )
        self.model_name = model_name
        # CS-ViT-L/14 takes 224 input by default → 14 grid (14*14=196 patches);
        # CS-ViT-L/14@336px takes 336 → 24 grid. We store grid for reshape.
        # Inferred from the visual.input_resolution attribute.
        self.input_size = self.model.visual.input_resolution
        # Patch size: pull from conv1 kernel.
        self.patch_size = self.model.visual.conv1.kernel_size[0]
        self.grid = self.input_size // self.patch_size
        # Caches for query-conditioned reranking: image features are
        # query-independent, the empty-prompt redundant feature is a
        # constant, and the prompt-ensemble text feature depends only on the
        # query string (so it can be reused across all candidate images for
        # one query). See clear_cache() to reset.
        self._img_feat_cache: dict[int, torch.Tensor] = {}
        self._redundant_feat: Optional[torch.Tensor] = None
        self._text_feat_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def extract(self, image: Image.Image, target_size: int = 384) -> np.ndarray:
        """Return saliency [target_size, target_size] in [0, 1]."""
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        # encode_image returns per-patch features [B, N+1, D] with CLS at 0.
        feats = self.model.encode_image(x)
        # feats shape: [1, N+1, D]
        cls = feats[:, 0, :]                     # [1, D]
        patch_feats = feats[:, 1:, :]            # [1, N, D]
        cls_n = cls / (cls.norm(dim=-1, keepdim=True) + 1e-8)
        patch_n = patch_feats / (patch_feats.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (cls_n.unsqueeze(1) * patch_n).sum(dim=-1)  # [1, N]
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
        """Return CLIP Surgery's reference query-conditioned saliency map.

        For a single text query, the upstream demo uses empty-string
        redundant features. Without that argument, ``clip_feature_surgery``
        computes the redundant feature as the mean over classes; with one
        class that degenerates to nearly zero signal.
        """
        img_feat = self._img_feat_for_image(image)
        text_feat = self._get_text_feat(text_query)
        redundant_feat = self._get_redundant_feat()
        sim = self.cs_clip.clip_feature_surgery(
            img_feat, text_feat, redundant_feat)
        sim_map = self.cs_clip.get_similarity_map(
            sim[:, 1:, :], (target_size, target_size))
        return sim_map[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
