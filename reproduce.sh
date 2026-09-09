#!/usr/bin/env bash
# Reproduces the ROCS rows of Table 2: every backbone on both splits.
# Extra flags are forwarded, so `./reproduce.sh --limit 20` runs a quick check.
set -euo pipefail

for model in clip-large siglip-so400m siglip2; do
    for split in coco flickr30k; do
        echo "== ${model} / rocs-${split}"
        python evaluate.py --model "${model}" --split "${split}" "$@"
    done
done
