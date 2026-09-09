"""CSLS on the GPU, same formula as miner/retrieval.py:csls.

The operation is two top-k reductions and a rank-one update, no matmul, so it is
purely memory-bound: a pass over the similarity matrix at whatever bandwidth the
device gives. In a real pipeline the matrix is already on the GPU, having just
been produced by the query-image product, so `resident=True` is the honest
setting and the transfer is reported separately.

CSLS: https://arxiv.org/abs/1710.04087
"""


def csls_gpu(sim, k=10, device="cuda"):
    matrix = sim if sim.is_cuda else sim.to(device, non_blocking=True)
    query_mean = matrix.topk(min(k, matrix.shape[1]), dim=1).values.mean(1, keepdim=True)
    image_mean = matrix.topk(min(k, matrix.shape[0]), dim=0).values.mean(0, keepdim=True)
    return 2.0 * matrix - query_mean - image_mean
