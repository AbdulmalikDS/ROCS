# MINER / ROCS

MINER improves text-to-image retrieval with a frozen encoder. It encodes the full image and five square crops, blends their scores, and applies CSLS to reduce hubness.

Data: [AbdulmalekDS/ROCS](https://huggingface.co/datasets/AbdulmalekDS/ROCS) on the Hugging Face Hub.

This repo contains the small evaluation release for the ROCS benchmark. The main method and lightweight crop comparisons share one encoder and scorer.

## Run

Use Python 3.10 or newer and install PyTorch for your hardware, then:

```bash
pip install -r requirements.txt
python evaluate.py --model siglip2 --split coco
```

Images and captions are pulled from [the ROCS dataset](https://huggingface.co/datasets/AbdulmalekDS/ROCS) on first run. `--split` takes `coco` or `flickr30k`. To score your own COCO-format file instead, pass `--annotations` and `--images-dir` together.

`./reproduce.sh` runs every backbone on both splits, which is the ROCS half of Table 2 in the paper; it forwards extra flags, so `./reproduce.sh --limit 20` is a quick check.

Supported models: `clip-large`, `siglip-so400m`, and `siglip2`. Add `--limit 20 --batch-size 2` for a small check. Model weights download on first use.

A local file needs `images` entries with `id` and `file_name`, and `annotations` entries with `caption` and `id` (or COCO-style `image_id`).

The command reports R@1, R@5, and R@10 for global retrieval, global retrieval with CSLS, and MINER. `--alpha` controls crop weight (default `0.4`); `--k` controls CSLS neighbors (default `10`, or `0` to disable). CSLS uses the evaluation query batch; this is the batch benchmark protocol.

Compare crop placement with `--crops grid --grid-size 3` or `--crops random --seed 0`. Random crops use the original 30–70% scale range. `--crop-ratio` and `--n-regions` control the fixed-crop ablations.

## Results

R@1, reported as ROCS-COCO / ROCS-Flickr30K. `evaluate.py` prints these three rows, so they double as the targets to check a run against.

| Backbone | Global | Global + CSLS | MINER |
|---|---|---|---|
| CLIP L/14 | 29.10 / 31.98 | 34.93 / 37.14 | 37.12 / 39.47 |
| SigLIP So/14 | 45.41 / 46.18 | 48.75 / 50.05 | 50.59 / 52.15 |
| SigLIP 2 So/16 | 47.08 / 48.00 | 50.29 / 52.28 | 52.36 / 53.84 |
