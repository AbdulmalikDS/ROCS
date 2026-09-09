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

| gallery | numpy | C++ | Rust | Julia |
|---|---|---|---|---|
| 2,000 | 0.419 s | 0.047 s | 0.044 s | 0.014 s |
| 8,000 | 1.775 s | 0.130 s | 0.139 s | 0.093 s |
| 32,000 | 7.313 s | 0.415 s | 0.462 s | 0.367 s |

All three stay within 5e-7 of the numpy result. Rust and C++ land on each other,
which is what the same algorithm at the same optimisation level should do. Julia
is ahead because its arrays are column-major, so the gallery pass, the one that
dominates, walks contiguous memory.

Two things account for the gap, and they split unevenly. Single-threaded at
32,000 the numbers are numpy 7.68 s, C++ 0.96 s, Julia 2.86 s, so the bounded
heap alone is worth about 8x against numpy's partitioned copy per axis. Thirty-two
threads then add only about 2x more: a single pass moves a gigabyte, so past a few
threads this is memory-bound and the bus, not the CPU, sets the limit. That is why
the total is 20x and not 200x, and why the deployment setting in Section 4.4 is
where it matters.

## Run

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
python bench.py
```

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
