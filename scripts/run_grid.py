"""Run every config of a YAML grid in parallel; one JSONL line per finished run.

YAML format:
    name: fedavg_alpha
    base: {rounds: 50, lr: 0.05}          # FLConfig fields shared by all runs
    grid: {alpha: [0.1, 0.5, 1.0, null], seed: [0, 1, 2]}   # cartesian product

Runs whose config key is already in results/<name>.jsonl are skipped, so an interrupted
grid resumes where it stopped. Each worker uses one CPU thread (fastest for this small MLP).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fl.server import FLConfig, run_fl  # noqa: E402


def expand(spec: dict) -> list[FLConfig]:
    valid = {f.name for f in fields(FLConfig)}
    base, grid = spec.get("base", {}), spec.get("grid", {})
    unknown = (set(base) | set(grid)) - valid
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    keys = list(grid)
    return [FLConfig(**{**base, **dict(zip(keys, combo))}) for combo in itertools.product(*grid.values())]


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line.strip()}


def _worker(cfg: FLConfig) -> dict:
    import torch

    torch.set_num_threads(1)
    return run_fl(cfg, root=ROOT)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, help="default: results/<name>.jsonl")
    a = p.parse_args()

    spec = yaml.safe_load(a.config.read_text())
    out = a.out or ROOT / "results" / f"{spec['name']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    configs = expand(spec)
    done = done_keys(out)
    todo = [c for c in configs if c.key() not in done]
    print(f"{spec['name']}: {len(configs)} runs, {len(configs) - len(todo)} already done, {len(todo)} to run")

    start = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers) as pool, out.open("a") as f:
        for i, res in enumerate(pool.imap_unordered(_worker, todo), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            c = res["config"]
            print(f"[{i}/{len(todo)}] alpha={c['alpha']} seed={c['seed']} lr={c['lr']} "
                  f"bs={c['batch_size']} agg={c['aggregator']} -> F1={res['final']['macro_f1']:.4f} "
                  f"({res['seconds']}s)", flush=True)
    print(f"done in {time.perf_counter() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
