import argparse
import json
from functools import partial
from pathlib import Path
from random import Random

import numpy as np

from miner.crops import fixed5_boxes, grid_boxes, random_boxes
from miner.encoder import MODELS, Encoder
from miner.retrieval import recall_at_k, score


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
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
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
    paths, queries, targets = load_queries(args.annotations, args.images_dir, args.limit)
    print(f"Encoding {len(paths)} images and {len(queries)} queries...", flush=True)
    encoder = Encoder(args.model, args.device)
    global_features, crops = encoder.images(paths, args.batch_size, cropper=cropper)
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
