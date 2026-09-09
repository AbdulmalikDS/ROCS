"""CSLS in C++ against the numpy version, over growing galleries.

Same formula either way; this only asks how each scales. Build first:

    g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
"""
import argparse
import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miner.retrieval import csls as csls_numpy

HERE = Path(__file__).resolve().parent
SIGNATURE = [np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
             ctypes.c_int, ctypes.c_int, ctypes.c_int]


def load(name):
    """Bind a compiled csls(), or None if that library was never built."""
    path = HERE / name
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    lib.csls.argtypes = SIGNATURE

    def call(sim, k=10):
        out = np.ascontiguousarray(sim, dtype=np.float32)
        lib.csls(out, out.shape[0], out.shape[1], k)
        return out

    return call


csls_cpp = load("libcsls.so")
csls_rust = load("libcsls_rs.so")

try:                                    # optional: only if torch sees a GPU
    import torch
    from csls_gpu import csls_gpu
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False


def timed_gpu(sim, repeats=3):
    """Time with the matrix already on the device, as it is in a real pipeline."""
    resident = torch.from_numpy(sim).cuda()
    csls_gpu(resident)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = csls_gpu(resident)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best, result.cpu().numpy()


def timed(fn, sim, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(sim.copy())
        best = min(best, time.perf_counter() - start)
    return best, result


def julia_times():
    """Run bench.jl if julia is installed; returns {gallery: seconds}."""
    if not shutil.which("julia"):
        return {}
    script = Path(__file__).resolve().parent
    done = subprocess.run(["julia", "-t", "auto", "bench.jl"], cwd=script,
                          capture_output=True, text=True)
    return {int(size): float(seconds)
            for size, seconds in (line.split() for line in done.stdout.split("\n") if line)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        help="a saved .npy similarity matrix; random values otherwise")
    args = parser.parse_args()
    if args.matrix:
        sim = np.load(args.matrix).astype(np.float32)
        print(f"{args.matrix.name}: {sim.shape[0]:,} queries x {sim.shape[1]:,} images")
        numpy_time, expected = timed(csls_numpy, sim)
        row = [f"{numpy_time:8.3f}s"]
        for fn in (csls_cpp, csls_rust):
            seconds, actual = timed(fn, sim) if fn else (float("nan"), expected)
            row.append(f"{seconds:8.3f}s")
            assert np.abs(expected - actual).max() < 1e-4
        print(f"{'numpy':>9} {'c++':>9} {'rust':>9}")
        print(" ".join(row))
        return
    rng = np.random.default_rng(0)
    julia = julia_times()
    print(f"{'queries x gallery':>22} {'numpy':>9} {'c++':>9} {'rust':>9} {'julia':>9} {'gpu':>9}  max diff")
    for gallery in (2_000, 8_000, 32_000):
        queries = 8_231                       # the ROCS-COCO query count
        sim = rng.random((queries, gallery), dtype=np.float32)
        numpy_time, expected = timed(csls_numpy, sim)
        diff = 0.0
        columns = []
        for fn in (csls_cpp, csls_rust):
            if fn is None:
                columns.append("        -")
                continue
            seconds, actual = timed(fn, sim)
            diff = max(diff, float(np.abs(expected - actual).max()))
            columns.append(f"{seconds:8.3f}s")
        assert diff < 1e-4, f"results diverged by {diff}"
        columns.append(f"{julia[gallery]:8.3f}s" if gallery in julia else "        -")
        if HAS_GPU:
            gpu_time, actual = timed_gpu(sim)
            diff = max(diff, float(np.abs(expected - actual).max()))
            columns.append(f"{gpu_time:8.4f}s")
        else:
            columns.append("        -")
        print(f"{queries:>9,} x {gallery:<9,}{numpy_time:8.3f}s " + " ".join(columns) + f"  {diff:.2e}")


if __name__ == "__main__":
    main()
