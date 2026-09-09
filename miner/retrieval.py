import numpy as np


def csls(sim, k=10):
    if sim.ndim != 2 or not sim.size:
        raise ValueError("Provide a nonempty similarity matrix.")
    if k < 0:
        raise ValueError("k must be nonnegative.")
    if k == 0:
        return sim
    # CSLS: https://arxiv.org/abs/1710.04087
    rows, cols = sim.shape
    qk, ik = min(k, cols), min(k, rows)
    query_mean = np.partition(sim, -qk, axis=1)[:, -qk:].mean(1, keepdims=True)
    image_mean = np.partition(sim, -ik, axis=0)[-ik:, :].mean(0, keepdims=True)
    return 2 * sim - query_mean - image_mean


def score(texts, global_features, crops=None, alpha=0.4, k=10):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    sim = texts @ global_features.T
    if not sim.size:
        raise ValueError("Provide at least one query and one image.")
    if crops is not None:
        if (crops.ndim != 3 or crops.shape[0] != len(global_features)
                or crops.shape[1] == 0 or crops.shape[2] != global_features.shape[1]):
            raise ValueError("Crops must have shape [images, regions, embedding size].")
        for start in range(0, len(crops), 128):
            regions = crops[start:start + 128]
            local = texts @ regions.reshape(-1, regions.shape[-1]).T
            best = local.reshape(len(texts), len(regions), -1).max(axis=2)
            sim[:, start:start + len(regions)] *= 1 - alpha
            sim[:, start:start + len(regions)] += alpha * best
    return csls(sim, k)


def recall_at_k(sim, targets, ks=(1, 5, 10)):
    ranked = np.argsort(-sim, axis=1)
    return {f"R@{k}": float((ranked[:, :k] == targets[:, None]).any(1).mean() * 100)
            for k in ks}
