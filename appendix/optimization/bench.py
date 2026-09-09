"""CSLS in C++ against the numpy version, over growing galleries.

Same formula either way; this only asks how each scales. Build first:

    g++ -O3 -march=native -fopenmp -shared -fPIC csls.cpp -o libcsls.so
"""
import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miner.retrieval import csls as csls_numpy

LIB = ctypes.CDLL(str(Path(__file__).resolve().parent / "libcsls.so"))
LIB.csls.argtypes = [np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
                     ctypes.c_int, ctypes.c_int, ctypes.c_int]


def csls_cpp(sim, k=10):
    out = np.ascontiguousarray(sim, dtype=np.float32)
    LIB.csls(out, out.shape[0], out.shape[1], k)
    return out


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
    rng = np.random.default_rng(0)
    julia = julia_times()
    print(f"{'queries x gallery':>22} {'numpy':>9} {'c++':>9} {'julia':>9}  max diff")
    for gallery in (2_000, 8_000, 32_000):
        queries = 8_231                       # the ROCS-COCO query count
        sim = rng.random((queries, gallery), dtype=np.float32)
        numpy_time, expected = timed(csls_numpy, sim)
        cpp_time, actual = timed(csls_cpp, sim)
        diff = float(np.abs(expected - actual).max())
        assert diff < 1e-4, f"results diverged by {diff}"
        shown = f"{julia[gallery]:8.3f}s" if gallery in julia else "        -"
        print(f"{queries:>9,} x {gallery:<9,}{numpy_time:8.3f}s {cpp_time:8.3f}s {shown}  {diff:.2e}")


if __name__ == "__main__":
    main()
