"""Floor and ceiling for macro-F1: majority class, centralized MLP (same optimizer), random forest."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fl.client import local_update  # noqa: E402
from src.fl.data import load_npz  # noqa: E402
from src.fl.metrics import evaluate  # noqa: E402
from src.fl.model import get_flat, make_mlp, set_flat  # noqa: E402


def centralized(d, epochs, lr, batch_size, seed):
    """Plain SGD on all training data; equals FedAvg with one client and E=1 per round."""
    torch.manual_seed(seed)
    X, y = torch.from_numpy(d["X_train"]), torch.from_numpy(d["y_train"])
    Xt, yt = torch.from_numpy(d["X_test"]), torch.from_numpy(d["y_test"])
    model = make_mlp(X.shape[1], len(d["classes"]))
    w, gen, curve = get_flat(model), torch.Generator().manual_seed(seed), []
    for _ in range(epochs):
        w = w + local_update(model, w, X, y, 1, lr, batch_size, gen)
        set_flat(model, w)
        curve.append(round(evaluate(model, Xt, yt, len(d["classes"]))["macro_f1"], 4))
    return evaluate(model, Xt, yt, len(d["classes"])), curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/ciciot_8cls.npz")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "results/baselines.json")
    a = p.parse_args()
    torch.set_num_threads(1)
    d = load_npz(a.data)

    majority = np.bincount(d["y_train"]).argmax()
    pred = np.full_like(d["y_test"], majority)
    res = {"majority": {"acc": accuracy_score(d["y_test"], pred),
                        "macro_f1": f1_score(d["y_test"], pred, average="macro")}}

    t = time.perf_counter()
    m, curve = centralized(d, a.epochs, a.lr, a.batch_size, seed=0)
    res["centralized_mlp"] = {**m, "curve": curve, "epochs": a.epochs, "lr": a.lr,
                              "batch_size": a.batch_size, "seconds": round(time.perf_counter() - t, 1)}

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(d["X_train"], d["y_train"])
    pred = rf.predict(d["X_test"])
    res["random_forest"] = {"acc": accuracy_score(d["y_test"], pred),
                            "macro_f1": f1_score(d["y_test"], pred, average="macro"),
                            "f1_per_class": [round(v, 4) for v in f1_score(d["y_test"], pred, average=None)]}

    for k, v in res.items():
        print(f"{k:<16} acc={v['acc']:.4f}  macro_f1={v['macro_f1']:.4f}")
    print("centralized per-class F1:", res["centralized_mlp"]["f1_per_class"])
    print("random forest per-class F1:", res["random_forest"]["f1_per_class"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
