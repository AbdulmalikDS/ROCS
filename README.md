# MINER / ROCS

MINER improves text-to-image retrieval with a frozen encoder. It encodes the full image and five square crops, blends their scores, and applies CSLS to reduce hubness.

This repo contains the small evaluation release for the ROCS benchmark. The main method and lightweight crop comparisons share one encoder and scorer.

## Run

Use Python 3.10 or newer and install PyTorch for your hardware, then:

```bash
pip install -r requirements.txt
python evaluate.py --model siglip2 \
  --annotations /path/to/rocs_coco.json \
  --images-dir /path/to/coco/val2014
```

Supported models: `clip-large`, `siglip-so400m`, and `siglip2`. Add `--limit 20 --batch-size 2` for a small check. Model weights download on first use.

Provide a ROCS caption JSON and its source images. The JSON contains `images` entries with `id` and `file_name`, and `annotations` entries with `caption` and `id` (or COCO-style `image_id`). Images and annotations are not bundled.

The command reports R@1, R@5, and R@10 for global retrieval, global retrieval with CSLS, and MINER. `--alpha` controls crop weight (default `0.4`); `--k` controls CSLS neighbors (default `10`, or `0` to disable). CSLS uses the evaluation query batch; this is the batch benchmark protocol.

Compare crop placement with `--crops grid --grid-size 3` or `--crops random --seed 0`. Random crops use the original 30–70% scale range. `--crop-ratio` and `--n-regions` control the fixed-crop ablations.

Attention-based crops, saliency models, curation, audits, and paper files stay in the local `research/` archive. They are not part of this release.

## Code

- [encoder.py](miner/encoder.py): frozen models through [OpenCLIP](https://github.com/mlfoundations/open_clip).
- [crops.py](miner/crops.py): fixed squares, grid cells, and seeded random crops.
- [retrieval.py](miner/retrieval.py): best-crop blending, [CSLS](https://arxiv.org/abs/1710.04087), and recall.
- [evaluate.py](evaluate.py): data loading and evaluation.

The fixed crops follow the MINER cache-generation path in the original LARE research workspace. No attention hooks or saliency models are needed.

## Test

```bash
python -m unittest discover -s tests -p test_core.py -v
```
