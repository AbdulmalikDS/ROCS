# Appendix: saliency-placed crops

The paper reports that the saliency source used to place a crop barely matters:
swapping it changes R@1 by at most 0.42 in any cell. This directory holds the code
to check that, kept apart from `miner/` because it needs extra models the method
itself never uses.

## Install

```bash
pip install -r appendix/saliency/requirements.txt
```

DINOv3 and CLIP-Surgery also need their upstream repositories, which are not
vendored here:

```bash
git clone https://github.com/facebookresearch/dinov3 appendix/saliency/third_party/dinov3
git clone https://github.com/xmed-lab/CLIP_Surgery appendix/saliency/third_party/CLIP_Surgery
```

MaskCLIP needs no clone; it reads dense features from the CLIP checkpoint directly.

## Run

```bash
python appendix/saliency/evaluate_saliency.py --source maskclip --split coco
```

`--source` takes `maskclip`, `dinov3`, or `clip-surgery`. Every other flag matches
`evaluate.py`. The centre crop moves to the saliency peak and the four corners stay
fixed, so only crop placement changes.

Reference numbers on ROCS-COCO with SigLIP 2, R@1: MaskCLIP 52.02, DINOv3 52.44,
CLIP-Surgery 52.23, against 52.36 for the parameter-free fixed crops. The paper's
fourth row, the encoder's own attention, uses a rollout hook that lives in the
research workspace and is not included here.
