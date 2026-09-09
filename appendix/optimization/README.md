# CSLS at gallery scale

`miner/retrieval.py` computes CSLS in numpy, which is fine for the benchmark: on
ROCS-COCO the matrix is 8,231 x 3,248 and rescoring takes about half a second,
against 25 minutes to encode the images. Nothing here changes that default.

The question this folder answers is what happens when the gallery is not 3,248
images. CSLS costs O(rows x cols) per side, and numpy pays it by materialising a
partitioned copy of the whole matrix once per axis. The two kernels here keep the
k best in a bounded heap instead, so the matrix is read twice and never copied.

## Results

Same formula, same inputs, 8,231 queries against a growing gallery, k=10,
32 threads. Times are the best of three.

| gallery | numpy | C++ | Rust | Julia | GPU |
|---|---|---|---|---|---|
| 2,000 | 0.424 s | 0.033 s | 0.045 s | 0.026 s | 0.0043 s |
| 8,000 | 1.766 s | 0.103 s | 0.140 s | 0.092 s | 0.0157 s |
| 32,000 | 7.370 s | 0.396 s | 0.496 s | 0.381 s | 0.0571 s |

numpy is CPU-only, so it is the baseline rather than a rival to the GPU column.
GPU times are measured with the matrix already resident on the device, which is
the real case: the similarity matrix is produced there by the query-image
product and never leaves. Starting from host memory instead adds a 1 GB copy
over PCIe and the totals become 0.014 s, 0.050 s and 0.196 s, so at this size the
transfer costs more than the work.

All three stay within 5e-7 of the numpy result. Rust and C++ land on each other,
which is what the same algorithm at the same optimisation level should do. Julia
is ahead because its arrays are column-major, so the gallery pass, the one that
dominates, walks contiguous memory.

On the CPU side, two things account for the gap, and they split unevenly. Single-threaded at
32,000 the numbers are numpy 7.68 s, C++ 0.96 s, Julia 2.86 s, so the bounded
heap alone is worth about 8x against numpy's partitioned copy per axis. Thirty-two
threads then add only about 2x more: a single pass moves a gigabyte, so past a few
threads this is memory-bound and the bus, not the CPU, sets the limit. That is why
the total is 20x and not 200x, and why the deployment setting in Section 4.4 is
where it matters.

## Run

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
rustc -O --crate-type=cdylib csls.rs -o libcsls_rs.so
python bench.py
```

The GPU column appears when torch reports a device; every other column runs
without one.

`bench.py` runs numpy, C++ and Rust, and calls `bench.jl` as well if `julia` is on
the path. Build the Rust library with
`rustc -O --crate-type=cdylib csls.rs -o libcsls_rs.so`.

Random values are a stand-in. To time a real similarity matrix instead, save one
from an evaluation run and pass `--matrix`:

```python
from datasets import load_dataset
from miner.encoder import Encoder
import numpy as np
split = load_dataset("AbdulmalekDS/ROCS", "coco", split="test")
encoder = Encoder("siglip2")
images, _ = encoder.images([row["image"] for row in split])
texts = encoder.texts([c for row in split for c in row["captions"]])
np.save("rocs_coco.npy", texts @ images.T)
``` Julia sets its own thread count with `-t auto`; C++ follows `OMP_NUM_THREADS`.
