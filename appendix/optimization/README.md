# CSLS at gallery scale

`miner/retrieval.py` computes CSLS in numpy. On ROCS-COCO that is an
8,231 x 3,248 matrix and about half a second, against 25 minutes to encode the
images, so the default stays as it is. This folder measures what happens when
the gallery is larger.

## Results

8,231 queries against a growing gallery, k=10, 32 threads, best of three.

| gallery | numpy | C++ | Rust | Julia | GPU |
|---|---|---|---|---|---|
| 2,000 | 0.424 s | 0.033 s | 0.045 s | 0.026 s | 0.0043 s |
| 8,000 | 1.766 s | 0.103 s | 0.140 s | 0.092 s | 0.0157 s |
| 32,000 | 7.370 s | 0.396 s | 0.496 s | 0.381 s | 0.0571 s |

All within 5e-7 of the numpy result. numpy is CPU-only and serves as the
baseline. GPU times have the matrix already resident, which is the real case
since it is produced there; from host memory the totals are 0.014 s, 0.050 s and
0.196 s.

Single-threaded at 32,000: numpy 7.68 s, C++ 0.96 s, Julia 2.86 s. The bounded
heap accounts for ~8x, threads for ~2x more.

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

`bench.py` calls `bench.jl` when julia is on the path and adds the GPU column
when torch reports a device. `--matrix file.npy` times a real similarity matrix
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
