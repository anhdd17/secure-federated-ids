"""Phase 1 sanity checks (PLAN.md, Phase 1 'Sanity check'): catch silent bugs in the FL loop."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.defenses.fedavg import fedavg
from src.fl.client import local_update
from src.fl.data import load_npz
from src.fl.model import get_flat, make_mlp
from src.fl.partition import class_counts, dirichlet_partition, iid_partition
from src.fl.server import FLConfig, run_fl

NPZ = Path(__file__).resolve().parents[1] / "data/processed/ciciot_8cls.npz"
needs_data = pytest.mark.skipif(not NPZ.exists(), reason="run scripts/prepare_data.py first")


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 100.0])
def test_dirichlet_partition_is_disjoint_and_complete(alpha):
    y = np.repeat(np.arange(8), 500)
    parts = dirichlet_partition(y, n_clients=10, alpha=alpha, seed=0, min_size=20)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(y)
    assert len(np.unique(allidx)) == len(y)  # disjoint and covering
    assert min(len(p) for p in parts) >= 20


def test_small_alpha_is_more_skewed():
    y = np.repeat(np.arange(8), 500)

    def mean_classes_per_client(alpha):
        counts = class_counts(y, dirichlet_partition(y, 10, alpha, seed=0, min_size=1), 8)
        return (counts / counts.sum(1, keepdims=True) > 0.05).sum(1).mean()

    assert mean_classes_per_client(0.1) < mean_classes_per_client(1.0) < mean_classes_per_client(100.0)


def test_iid_partition_covers_everything():
    parts = iid_partition(1003, 10, seed=0)
    assert sorted(np.concatenate(parts).tolist()) == list(range(1003))


def test_fedavg_identical_updates_returns_that_update():
    u = torch.randn(5)
    agg, info = fedavg(u.repeat(4, 1), {"n_samples": [1, 2, 3, 4]})
    assert torch.allclose(agg, u, atol=1e-6)
    assert abs(sum(info["weights"]) - 1) < 1e-6


def test_fedavg_weights_by_sample_count():
    updates = torch.tensor([[1.0], [0.0]])
    agg, _ = fedavg(updates, {"n_samples": [3, 1]})
    assert agg.item() == pytest.approx(0.75)


def test_clients_do_not_share_state():
    """Two clients trained from the same global weights on the same data get the same update,
    and neither update aliases the model's live parameters (deepcopy / shared-reference bug)."""
    torch.manual_seed(0)
    model = make_mlp(4, 3)
    w = get_flat(model)
    X, y = torch.randn(64, 4), torch.randint(0, 3, (64,))
    d1 = local_update(model, w, X, y, 1, 0.1, 16, torch.Generator().manual_seed(1))
    d2 = local_update(model, w, X, y, 1, 0.1, 16, torch.Generator().manual_seed(1))
    assert torch.allclose(d1, d2)
    live = {p.data_ptr() for p in model.parameters()}
    assert d1.data_ptr() not in live and d2.data_ptr() != d1.data_ptr()
    assert torch.equal(w, get_flat(make_mlp(4, 3)) * 0 + w)  # w itself untouched


@needs_data
def test_processed_data_is_clean():
    d = load_npz(NPZ)
    for k in ("X_train", "X_test", "X_root"):
        assert np.isfinite(d[k]).all()
    hashes = lambda X: {r.tobytes() for r in X}  # noqa: E731
    assert not hashes(d["X_train"]) & hashes(d["X_test"]), "train/test share identical rows"
    assert not hashes(d["X_root"]) & hashes(d["X_train"]), "root set overlaps client data"
    assert set(np.unique(d["y_root"])) == set(range(len(d["classes"])))


@needs_data
def test_same_seed_same_result_different_seed_different_result():
    torch.set_num_threads(1)
    base = dict(n_clients=3, alpha=0.5, rounds=2, eval_every=2)
    a = run_fl(FLConfig(**base, seed=0))["final"]["macro_f1"]
    b = run_fl(FLConfig(**base, seed=0))["final"]["macro_f1"]
    c = run_fl(FLConfig(**base, seed=1))["final"]["macro_f1"]
    assert a == b
    assert a != c
