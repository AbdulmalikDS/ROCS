"""Evaluate saliency-guided crop placement."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from appendix.saliency.crops import saliency_boxes
from evaluate import load_queries, load_split
from miner.encoder import MODELS, Encoder
from miner.retrieval import recall_at_k, score

SOURCES = ("maskclip", "dinov3", "clip-surgery")


def build_extractor(source, device, dinov3_weights=None):
    if source == "maskclip":
        from appendix.saliency.maskclip_saliency import MaskCLIPSaliencyExtractor
        return MaskCLIPSaliencyExtractor(device=device)
    if source == "dinov3":
        from appendix.saliency.dinov3_saliency import DINOv3SaliencyExtractor
        return DINOv3SaliencyExtractor(device=device, weights=dinov3_weights)
    from appendix.saliency.clip_surgery_saliency import CLIPSurgerySaliencyExtractor
    return CLIPSurgerySaliencyExtractor(device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--dinov3-weights", type=Path,
                        help="local DINOv3 ViT-L/16 LVD-1689M checkpoint (required for dinov3)")
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
    parser.add_argument("--crop-ratio", type=float, default=0.6)
    args = parser.parse_args()
    if bool(args.annotations) != bool(args.images_dir):
        parser.error("pass --annotations and --images-dir together, or neither")
    if args.source == "dinov3" and args.dinov3_weights is None:
        parser.error("--source dinov3 requires --dinov3-weights; see appendix/saliency/README.md")
    if args.source != "dinov3" and args.dinov3_weights is not None:
        parser.error("--dinov3-weights is only used with --source dinov3")
    if args.dinov3_weights is not None and not args.dinov3_weights.expanduser().is_file():
        parser.error("DINOv3 checkpoint does not exist")

    if args.annotations:
        sources, queries, targets = load_queries(args.annotations, args.images_dir, args.limit)
    else:
        sources, queries, targets = load_split(args.dataset, args.split, args.limit)

    encoder = Encoder(args.model, args.device)
    extractor = build_extractor(args.source, encoder.device, args.dinov3_weights)
    print(f"Encoding {len(sources)} images and {len(queries)} queries with {args.source}...",
          flush=True)

    def cropper(image):
        return saliency_boxes(image, extractor.extract(image), crop_ratio=args.crop_ratio)

    global_features, crops = encoder.images(sources, args.batch_size, cropper=cropper)
    texts = encoder.texts(queries)
    metrics = recall_at_k(score(texts, global_features, crops, alpha=args.alpha, k=args.k), targets)
    print(args.source + ": " + "  ".join(f"{key}={value:.2f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
