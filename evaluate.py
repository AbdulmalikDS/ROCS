import argparse
import json
from functools import partial
from pathlib import Path
from random import Random

import numpy as np

from miner.crops import fixed5_boxes, grid_boxes, random_boxes
from miner.encoder import MODELS, Encoder
from miner.retrieval import recall_at_k, score


class _Rows:
    """Lazy image access so a whole split is not decoded into memory."""

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        return self.split[index]["image"]


def load_split(dataset, config, limit=None):
    from datasets import load_dataset

    split = load_dataset(dataset, config, split="test")
    if limit:
        split = split.select(range(min(limit, len(split))))
    queries, targets = [], []
    for index, captions in enumerate(split["captions"]):
        queries.extend(captions)
        targets.extend([index] * len(captions))
    if not len(split) or not queries:
        raise ValueError("The selected split has no images or queries.")
    return _Rows(split), queries, np.asarray(targets)


def load_queries(annotations, images_dir, limit=None):
    data = json.loads(Path(annotations).read_text())
    images = data["images"][:limit]
    index = {image["id"]: i for i, image in enumerate(images)}
    paths = [Path(images_dir) / image["file_name"] for image in images]
    queries, targets = [], []
    for entry in data["annotations"]:
        image_id = entry.get("image_id", entry.get("id"))
        if image_id in index:
            queries.append(entry["caption"])
            targets.append(index[image_id])
    if not paths or not queries:
        raise ValueError("The selected split has no images or queries.")
    return paths, queries, np.asarray(targets)


def main():
    parser = argparse.ArgumentParser(description="Evaluate global retrieval and MINER on ROCS.")
    parser.add_argument("--split", choices=["coco", "flickr30k"], default="coco")
    parser.add_argument("--dataset", default="AbdulmalekDS/ROCS")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--model", choices=MODELS, default="siglip2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--crops", choices=["fixed", "grid", "random"], default="fixed")
    parser.add_argument("--crop-ratio", type=float, default=0.6)
    parser.add_argument("--n-regions", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if bool(args.annotations) != bool(args.images_dir):
        parser.error("pass --annotations and --images-dir together, or neither")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if not 0 <= args.alpha <= 1 or args.k < 0:
        parser.error("alpha must be in [0, 1] and k must be nonnegative")
    if not 0 < args.crop_ratio <= 1 or args.n_regions < 1 or args.grid_size < 1:
        parser.error("crop-ratio must be in (0, 1]; region count and grid size must be positive")
    if args.crops == "fixed" and args.n_regions > 5:
        parser.error("fixed crops support at most five regions")
    cropper = {
        "fixed": partial(fixed5_boxes, r=args.crop_ratio, n=args.n_regions),
        "grid": partial(grid_boxes, n=args.grid_size),
        "random": partial(random_boxes, n=args.n_regions, rng=Random(args.seed)),
    }[args.crops]
    if args.annotations:
        sources, queries, targets = load_queries(args.annotations, args.images_dir, args.limit)
    else:
        sources, queries, targets = load_split(args.dataset, args.split, args.limit)
    print(f"Encoding {len(sources)} images and {len(queries)} queries...", flush=True)
    encoder = Encoder(args.model, args.device)
    global_features, crops = encoder.images(sources, args.batch_size, cropper=cropper)
    texts = encoder.texts(queries)
    runs = [("Global", None, 0)]
    if args.k:
        runs.append(("Global + CSLS", None, args.k))
    label = "MINER" if args.crops == "fixed" else f"{args.crops.capitalize()} crops"
    runs.append((label, crops, args.k))
    for label, regions, k in runs:
        sim = score(texts, global_features, regions, alpha=args.alpha, k=k)
        metrics = recall_at_k(sim, targets)
        print(label + ": " + "  ".join(f"{key}={value:.2f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
