"""Label-skewed non-IID partition with a Dirichlet prior (Hsu et al. 2019; NIID-Bench §IV).

For every class c we draw p_c ~ Dir(alpha * 1_n) and give client k the share p_c[k] of the
samples of class c. Small alpha -> each class lands on a few clients (strong skew);
alpha -> infinity -> every client gets the same class mix (IID).
"""
from __future__ import annotations

import numpy as np


def dirichlet_partition(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int,
    min_size: int = 50,
    max_tries: int = 1000,
) -> list[np.ndarray]:
    """Return one index array per client; indices are disjoint and cover all of y.

    Redraws until every client holds at least `min_size` samples (NIID-Bench does the same),
    otherwise a client with a handful of samples makes its update pure noise.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        shards: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(y == c))
            p = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                shards[k].append(part)
        parts = [np.sort(np.concatenate(s)) for s in shards]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"no partition with min_size={min_size} after {max_tries} draws (alpha={alpha})")


def iid_partition(n: int, n_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(p) for p in np.array_split(rng.permutation(n), n_clients)]


def class_counts(y: np.ndarray, parts: list[np.ndarray], n_classes: int) -> np.ndarray:
    """Matrix [client, class] of sample counts, for heatmaps and sanity checks."""
    return np.stack([np.bincount(y[p], minlength=n_classes) for p in parts])
