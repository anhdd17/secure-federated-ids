"""Single-process FL simulator (honest server, full participation).

One round:  broadcast w -> every client runs local SGD -> returns Δ_k
            -> aggregate(Δ[n, d], ctx) -> w <- w + server_lr * Δ_agg
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.defenses import AGGREGATORS
from src.fl.client import local_update
from src.fl.data import load_npz
from src.fl.metrics import evaluate
from src.fl.model import get_flat, make_mlp, set_flat
from src.fl.partition import dirichlet_partition, iid_partition


@dataclass
class FLConfig:
    data: str = "data/processed/ciciot_8cls.npz"
    n_clients: int = 10
    alpha: float | None = 0.5  # None = IID
    rounds: int = 50
    local_epochs: int = 1
    lr: float = 0.05
    batch_size: int = 64
    server_lr: float = 1.0
    aggregator: str = "fedavg"
    seed: int = 0
    eval_every: int = 1
    min_client_size: int = 50
    device: str = "cpu"

    def key(self) -> str:
        """Stable hash of the config: one results line per key, so reruns can skip finished runs."""
        payload = {k: v for k, v in asdict(self).items() if k != "device"}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_partition(y: np.ndarray, cfg: FLConfig) -> list[np.ndarray]:
    if cfg.alpha is None:
        return iid_partition(len(y), cfg.n_clients, cfg.seed)
    return dirichlet_partition(y, cfg.n_clients, cfg.alpha, cfg.seed, cfg.min_client_size)


def run_fl(cfg: FLConfig, root: Path = Path(".")) -> dict:
    """Run one FL experiment; returns config, per-round history and timing."""
    t0 = time.perf_counter()
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    d = load_npz(root / cfg.data)
    n_classes = len(d["classes"])

    X_tr = torch.from_numpy(d["X_train"]).to(dev)
    y_tr = torch.from_numpy(d["y_train"]).to(dev)
    X_te = torch.from_numpy(d["X_test"]).to(dev)
    y_te = torch.from_numpy(d["y_test"]).to(dev)

    parts = make_partition(d["y_train"], cfg)
    client_data = [(X_tr[p], y_tr[p]) for p in parts]
    n_samples = [len(p) for p in parts]

    model = make_mlp(X_tr.shape[1], n_classes).to(dev)
    w = get_flat(model)
    gen = torch.Generator().manual_seed(cfg.seed)
    aggregate = AGGREGATORS[cfg.aggregator]

    history = []
    for r in range(1, cfg.rounds + 1):
        updates = torch.stack([
            local_update(model, w, X, y, cfg.local_epochs, cfg.lr, cfg.batch_size, gen)
            for X, y in client_data
        ])
        ctx = {"n_samples": n_samples, "round": r}
        delta, info = aggregate(updates, ctx)
        w = w + cfg.server_lr * delta

        if r % cfg.eval_every == 0 or r == cfg.rounds:
            set_flat(model, w)
            m = evaluate(model, X_te, y_te, n_classes)
            history.append({
                "round": r,
                **m,
                "update_norms": [round(v, 5) for v in updates.norm(dim=1).tolist()],
                "weights": [round(v, 5) for v in info.get("weights", [])],
            })

    return {
        "key": cfg.key(),
        "config": asdict(cfg),
        "client_sizes": n_samples,
        "final": {k: history[-1][k] for k in ("acc", "macro_f1", "loss", "f1_per_class")},
        "history": history,
        "seconds": round(time.perf_counter() - t0, 2),
    }
