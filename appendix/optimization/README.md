# CSLS at gallery scale

Six implementations of one small function, written for fun. The paper needs none
of them; `miner/retrieval.py` stays in numpy.

`miner/retrieval.py` computes CSLS in numpy. On ROCS-COCO that is an
8,231 x 3,248 matrix and about half a second, against 25 minutes to encode the
images, so the default stays as it is. This folder measures what happens when
the gallery is larger.

## Results

8,231 queries against a growing gallery, k=10, 32 threads, best of three.

| gallery | numpy | C++ | Rust | Julia | torch | CUDA |
|---|---|---|---|---|---|---|
| 2,000 | 0.364 s | 0.035 s | 0.040 s | 0.025 s | 0.0043 s | 0.0081 s |
| 8,000 | 1.998 s | 0.122 s | 0.163 s | 0.095 s | 0.0157 s | 0.0109 s |
| 32,000 | 7.929 s | 0.425 s | 0.488 s | 0.379 s | 0.0571 s | 0.0194 s |

All within 5e-7 of the numpy result. numpy is CPU-only and serves as the
baseline. GPU times have the matrix already resident, which is the real case
since it is produced there; from host memory the totals are 0.014 s, 0.050 s and
0.196 s.

Single-threaded at 32,000: numpy 7.68 s, C++ 0.96 s, Julia 2.86 s. The bounded
heap accounts for ~8x, threads for ~2x more.

## The CUDA kernel

torch needs three passes over the matrix, `topk` along each axis then the
rank-one update, about 4 GB of traffic at 32,000. `csls.cu` does the same work
in three kernels but keeps k in registers throughout, and it beats torch above
8,000 columns. What each step bought, measured at 32,000:

| version | time |
|---|---|
| thread per column, block per row, thread 0 merges the block's 2,560 candidates | 0.0217 s |
| warp-shuffle merge (`__shfl_down_sync`) and `float4` loads on the row pass | 0.0194 s |

Small k is why registers win: a sorted insertion list of 10 floats beats a heap,
and merging two lanes' lists costs 10 shuffles. The shuffle has to read the
neighbour's whole list before touching its own, otherwise lanes merge
half-updated state; that bug passed the timing and failed the correctness check.

Still 216 GB/s of the card's 672. The next step would be fusing the two
reductions into one read of the matrix, taking traffic from ~4.2 GB to ~3.2 GB,
which is where the remaining headroom is.

## What this is not

The reference CSLS implementation ([MUSE](https://github.com/facebookresearch/MUSE),
`get_nn_avg_dist`) uses faiss, `IndexFlatIP` or `GpuIndexFlatIP`, and searches
the embeddings directly. That is the right shape at scale, because it never
materialises the similarity matrix. These kernels assume the matrix exists,
which holds for the transductive protocol in the paper and stops holding once
the gallery is large enough that N x M does not fit. Past that point the top-M
shortlist of Section 4.4, backed by an ANN index, is the path, not a faster
reduction.

## Run

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
rustc -O --crate-type=cdylib csls.rs -o libcsls_rs.so
python bench.py
```

`bench.py` calls `bench.jl` when julia is on the path and adds the GPU columns
when torch reports a device. Build the CUDA one with
`nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC csls.cu -o libcsls_cuda.so`. `--matrix file.npy` times a real similarity matrix
instead of random values:

```python
from datasets import load_dataset
from miner.encoder import Encoder
import numpy as np
split = load_dataset("AbdulmalekDS/ROCS", "coco", split="test")
encoder = Encoder("siglip2")
images, _ = encoder.images([row["image"] for row in split])
texts = encoder.texts([c for row in split for c in row["captions"]])
np.save("rocs_coco.npy", texts @ images.T)
```
