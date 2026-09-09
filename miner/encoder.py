from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .crops import fixed5_boxes

MODELS = {
    "clip-large": ("ViT-L-14", "openai"),
    "siglip-so400m": ("ViT-SO400M-14-SigLIP-384", "webli"),
    "siglip2": ("ViT-SO400M-16-SigLIP2-384", "webli"),
}


def _open(source):
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as handle:
        return handle.convert("RGB")


class Encoder:
    def __init__(self, name="siglip2", device=None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, weights = MODELS[name]
        # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=weights, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        size = self.model.visual.image_size
        self.image_size = (size, size) if isinstance(size, int) else tuple(size)

    @torch.inference_mode()
    def images(self, sources, batch_size=4, cropper=None):
        if not len(sources) or batch_size < 1:
            raise ValueError("Provide images and a positive batch size.")
        cropper = cropper or (lambda image: fixed5_boxes(*image.size))
        batches = []
        for start in range(0, len(sources), batch_size):
            views = []
            for index in range(start, min(start + batch_size, len(sources))):
                image = _open(sources[index])
                views.append(self.preprocess(image))
                boxes = cropper(image)
                for box in boxes:
                    crop = image.crop(box).resize(self.image_size, Image.Resampling.BICUBIC)
                    views.append(self.preprocess(crop))
            pixels = torch.stack(views).to(self.device)
            features = self.model.encode_image(pixels, normalize=True)
            features = features.float().cpu().numpy()
            batches.append(features.reshape(-1, len(boxes) + 1, features.shape[-1]))
        features = np.concatenate(batches)
        return features[:, 0], features[:, 1:]

    @torch.inference_mode()
    def texts(self, queries, batch_size=128):
        if not queries or batch_size < 1:
            raise ValueError("Provide queries and a positive batch size.")
        batches = []
        for start in range(0, len(queries), batch_size):
            tokens = self.tokenizer(queries[start:start + batch_size]).to(self.device)
            features = self.model.encode_text(tokens, normalize=True)
            batches.append(features.float().cpu().numpy())
        return np.concatenate(batches)
