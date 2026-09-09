"""
Saliency extractor based on MaskCLIP (Zhou et al., ECCV 2022 oral,
"Extract Free Dense Labels from CLIP", arXiv 2112.01071).

Paper §3.1 formulation:

    "we propose to remove the query and key embeddings and transform
    both the value embedding and the last linear layer (Linear) into
    two convolutional layers."

That is: in the LAST attention block only, bypass attention entirely
and pass each token through the value projection followed by the
output projection — no QK softmax, no token mixing. Then apply
ln_post and the visual projection to every patch (not just CLS).

This is what we implement below. Earlier blocks run unchanged so the
backbone stays a real CLIP encoder; the trick is local to the final
layer's readout.

Implementation notes (vs. an earlier broken version of this file):

  - open_clip's ``Transformer`` is batch-first (``[B, N+1, D]``); we
    do NOT permute to LND like the OpenAI reference.
  - ``visual.proj`` may be a Linear (open_clip ≥ 2024 conversions) or
    a raw parameter; both paths are handled.
  - Preprocessing uses ``Resize((n, n))`` (no center crop). For
    cropping pipelines the saliency square must align with the
    full image we're cropping from, otherwise boxes shift away
    from the actual object on non-square inputs. CLIP Surgery's
    demo makes the same choice for the same reason.

Reference repository: https://github.com/chongzhou96/MaskCLIP
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


_OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def _v_only_attention(attn: torch.nn.MultiheadAttention,
                      x: torch.Tensor) -> torch.Tensor:
    """Bypass attention: V projection followed by out_proj, per token.

    ``x`` is ``[B, N+1, D]`` (open_clip batch-first). ``attn`` is an
    ``nn.MultiheadAttention`` whose ``in_proj_weight`` is ``[3D, D]``
    stacked as Q | K | V. We slice V, apply, then apply out_proj.
    """
    D = attn.embed_dim
    in_w = attn.in_proj_weight
    in_b = attn.in_proj_bias
    v = F.linear(x, in_w[2 * D:3 * D],
                 in_b[2 * D:3 * D] if in_b is not None else None)
    return F.linear(v, attn.out_proj.weight, attn.out_proj.bias)


def _layerscale(block: torch.nn.Module, name: str,
                x: torch.Tensor) -> torch.Tensor:
    layer = getattr(block, name, None)
    return layer(x) if layer is not None else x


class MaskCLIPSaliencyExtractor:
    """Per-patch saliency from a CLIP backbone with the MaskCLIP
    last-layer readout (no QK in the final block).

    Defaults to ``ViT-L-14`` openai weights so it matches the CLIP-Large
    backbone used elsewhere in this paper.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: Optional[str] = None,
        input_size: int = 224,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        import open_clip
        self.model_name = model_name
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        vis = self.model.visual
        if hasattr(vis, "trunk"):
            raise NotImplementedError(
                "MaskCLIP extractor currently supports only the OpenAI-style "
                "open_clip visual transformer (with conv1, transformer, "
                "ln_post, proj attributes)."
            )
        self.vis = vis
        self.patch_size = vis.conv1.kernel_size[0]
        if self.input_size % self.patch_size != 0:
            self.input_size = (self.input_size // self.patch_size) * self.patch_size
        self.grid = self.input_size // self.patch_size
        # Resize-only preprocessing (no center crop): saliency square must
        # align with the full image we're going to crop from.
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=_OPENAI_MEAN, std=_OPENAI_STD),
        ])
        # Per-image patch-feature cache (keyed by id(image)) for
        # query-conditioned reranking, where the same image is hit by
        # many queries. Per-query text-feature cache reuses the same text
        # encode across all candidate images for one query. Cleared via
        # clear_cache().
        self._patch_n_cache: dict[int, torch.Tensor] = {}
        self._text_feat_cache: dict[str, torch.Tensor] = {}

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        """Apply visual.proj whether it's a Linear or a raw Parameter."""
        proj = self.vis.proj
        if proj is None:
            return h
        if isinstance(proj, torch.nn.Linear):
            return proj(h)
        return h @ proj.to(dtype=h.dtype, device=h.device)  # raw [D, D_emb]

    def _dense_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return MaskCLIP dense image tokens [B, N+1, D_emb]."""
        v = self.vis
        h = v.conv1(x)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(dtype=h.dtype, device=h.device)
        cls = cls.reshape(1, 1, -1).expand(h.shape[0], 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + v.positional_embedding.to(dtype=h.dtype, device=h.device)
        h = v.ln_pre(h)

        resblocks = v.transformer.resblocks
        for blk in resblocks[:-1]:
            h = blk(h)

        last = resblocks[-1]
        attn_out = _v_only_attention(last.attn, last.ln_1(h))
        h = h + _layerscale(last, "ls_1", attn_out)
        h = h + _layerscale(last, "ls_2", last.mlp(last.ln_2(h)))

        h = v.ln_post(h)
        return self._project(h)

    @torch.no_grad()
    def extract(self, image: Image.Image, target_size: int = 384) -> np.ndarray:
        """Return saliency [target_size, target_size] in [0, 1]."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        h = self._dense_features(x)
        # h: [1, N+1, D_emb], CLS at 0, patches at 1:.
        cls_feat = h[:, 0, :]
        patch_feat = h[:, 1:, :]
        cls_n = cls_feat / (cls_feat.norm(dim=-1, keepdim=True) + 1e-8)
        patch_n = patch_feat / (patch_feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (cls_n.unsqueeze(1) * patch_n).sum(dim=-1)  # [1, N]
        sim_np = sim[0].detach().cpu().numpy().astype(np.float32)
        sim_map = sim_np.reshape(self.grid, self.grid)
        lo, hi = float(sim_map.min()), float(sim_map.max())
        sim_map = (sim_map - lo) / (hi - lo + 1e-8)
        out = cv2.resize(sim_map, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)

    @torch.no_grad()
    def _patch_n_for_image(self, image: Image.Image) -> torch.Tensor:
        key = id(image)
        cached = self._patch_n_cache.get(key)
        if cached is not None:
            return cached
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        h = self._dense_features(x)
        patch_feat = h[:, 1:, :]
        patch_n = patch_feat / (patch_feat.norm(dim=-1, keepdim=True) + 1e-8)
        self._patch_n_cache[key] = patch_n
        return patch_n

    def clear_cache(self) -> None:
        self._patch_n_cache.clear()
        self._text_feat_cache.clear()

    @torch.no_grad()
    def _get_text_feat(self, text_query: str) -> torch.Tensor:
        cached = self._text_feat_cache.get(text_query)
        if cached is not None:
            return cached
        toks = self.tokenizer([text_query]).to(self.device)
        feat = self.model.encode_text(toks)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        self._text_feat_cache[text_query] = feat
        return feat

    @torch.no_grad()
    def extract_query(self, image: Image.Image, text_query: str,
                      target_size: int = 384) -> np.ndarray:
        """Query-conditional MaskCLIP saliency: cosine of per-patch
        features against the text embedding. Per MaskCLIP §3.2 this is
        the intended use of the dense per-patch features."""
        text_feat = self._get_text_feat(text_query)

        patch_n = self._patch_n_for_image(image)
        sim = (patch_n * text_feat.unsqueeze(1)).sum(dim=-1)
        sim_np = sim[0].detach().cpu().numpy().astype(np.float32)
        sim_map = sim_np.reshape(self.grid, self.grid)
        lo, hi = float(sim_map.min()), float(sim_map.max())
        sim_map = (sim_map - lo) / (hi - lo + 1e-8)
        out = cv2.resize(sim_map, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)
