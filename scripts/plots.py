"""Phase 1 figures: Dirichlet partition heatmaps and FedAvg convergence vs alpha.

Palette: categorical slots 1-5 of the reference palette (validated: CVD and normal-vision
checks pass on #fcfcfb; three slots sit below 3:1 contrast, so lines carry direct labels
and a CSV table is written next to each figure).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fl.data import load_npz  # noqa: E402
from src.fl.partition import class_counts, dirichlet_partition  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def partition_heatmaps(data_path: Path, out: Path, alphas=(0.1, 0.5, 1.0), n_clients=10, seed=0):
    d = load_npz(data_path)
    y, classes = d["y_train"], list(d["classes"])
    fig, axes = plt.subplots(1, len(alphas), figsize=(4.2 * len(alphas), 4.2), sharey=True,
                             constrained_layout=True)
    rows = []
    for ax, alpha in zip(axes, alphas):
        counts = class_counts(y, dirichlet_partition(y, n_clients, alpha, seed), len(classes))
        share = counts / counts.sum(0, keepdims=True)  # where each class's samples went
        im = ax.imshow(share.T, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"α = {alpha}", color=INK, fontsize=11, loc="left")
        ax.set_xticks(range(n_clients), [str(k) for k in range(n_clients)])
        ax.set_xlabel("client")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for k in range(n_clients):
            for c, cls in enumerate(classes):
                rows.append({"alpha": alpha, "client": k, "class": cls, "samples": int(counts[k, c]),
                             "share_of_class": round(float(share[k, c]), 4)})
    axes[0].set_yticks(range(len(classes)), classes)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("share of the class held by the client", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Dirichlet label partition, {n_clients} clients (seed {seed}): "
                 "smaller α concentrates each class on fewer clients", x=0.01, ha="left", color=INK)
    fig.savefig(out, dpi=160)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    print(f"saved {out}")


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge label positions apart so end-of-line labels never overlap."""
    order = np.argsort(ys)
    placed = list(np.array(ys)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = [0.0] * len(ys)
    for rank, idx in enumerate(order):
        out[idx] = placed[rank]
    return out


def convergence(results: Path, baselines: Path, out: Path):
    runs = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    by_alpha = defaultdict(list)
    for r in runs:
        by_alpha[r["config"]["alpha"]].append(r)
    order = [0.1, 0.5, 1.0, 100.0, None]
    names = {0.1: "α = 0.1", 0.5: "α = 0.5", 1.0: "α = 1.0", 100.0: "α = 100", None: "IID"}

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ends, table = [], []
    for color, alpha in zip(SERIES, order):
        rs = by_alpha.get(alpha, [])
        if not rs:
            continue
        rounds = [h["round"] for h in rs[0]["history"]]
        f1 = np.array([[h["macro_f1"] for h in r["history"]] for r in rs])
        mean, lo, hi = f1.mean(0), f1.min(0), f1.max(0)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(rounds, mean, color=color, linewidth=2, label=f"{names[alpha]} (n={len(rs)} seeds)")
        ends.append((mean[-1], names[alpha]))
        table += [{"series": names[alpha], "round": rd, "macro_f1_mean": round(m, 4),
                   "macro_f1_min": round(a, 4), "macro_f1_max": round(b, 4)}
                  for rd, m, a, b in zip(rounds, mean, lo, hi)]

    base = json.loads(baselines.read_text())
    cen = base["centralized_mlp"]["macro_f1"]
    ax.axhline(cen, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ends.append((cen, "centralized"))
    table.append({"series": "centralized", "round": base["centralized_mlp"]["epochs"],
                  "macro_f1_mean": round(cen, 4)})

    x_end = max(h["round"] for h in runs[0]["history"])
    label_y = _spread([y for y, _ in ends], gap=0.025)
    for (y, name), ly in zip(ends, label_y):
        ax.annotate(f"{name}  {y:.3f}", xy=(x_end, y), xytext=(x_end + 1.5, ly),
                    color=INK, fontsize=9, va="center", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8})
    ax.set_xlim(0, x_end)
    ax.set_ylim(top=max(label_y) + 0.03)
    ax.set_xlabel("communication round")
    ax.set_ylabel("macro-F1 on held-out test set")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("FedAvg on CICIoT2023 (8 classes, 10 clients): stronger label skew, lower macro-F1\n"
                 "line = mean over seeds, band = min–max; dashed = centralized SGD, same epochs",
                 loc="left", color=INK, fontsize=10.5)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    pd.DataFrame(table).to_csv(out.with_suffix(".csv"), index=False)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/ciciot_8cls.npz")
    p.add_argument("--results", type=Path, default=ROOT / "results/fedavg_alpha.jsonl")
    p.add_argument("--baselines", type=Path, default=ROOT / "results/baselines.json")
    p.add_argument("--outdir", type=Path, default=ROOT / "report/figures")
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    partition_heatmaps(a.data, a.outdir / "partition_heatmap.png")
    convergence(a.results, a.baselines, a.outdir / "fedavg_convergence.png")


if __name__ == "__main__":
    main()
