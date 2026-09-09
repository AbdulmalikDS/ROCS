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

| gallery | numpy | C++ | Julia |
|---|---|---|---|
| 2,000 | 0.361 s | 0.048 s | 0.027 s |
| 8,000 | 1.798 s | 0.138 s | 0.095 s |
| 32,000 | 10.422 s | 0.427 s | 0.356 s |

Both compiled versions stay within 5e-7 of the numpy result. Julia is ahead
throughout, most clearly on small matrices where the C++ version spends
proportionally more time in its strided column pass; the two converge as the
gallery grows and both become memory-bound.

The gap widens with gallery size, which is the point: at 32,000 images numpy is
24x behind, and the deployment setting in Section 4.4 of the paper is larger
still.

## Run

```bash
g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
python bench.py
```

`bench.py` runs numpy and C++, and calls `bench.jl` as well if `julia` is on the
path. Julia sets its own thread count with `-t auto`; C++ follows `OMP_NUM_THREADS`.
