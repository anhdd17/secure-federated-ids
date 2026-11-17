# Phase 1 — data, FL simulator, FedAvg: results and sanity checks

Run on 2026-11-17, Mac M4, CPU, 1 thread per worker, 8 workers. Reproduce:

```bash
.venv/bin/python scripts/prepare_data.py                       # data/processed/ciciot_8cls.npz
.venv/bin/python scripts/baselines.py --epochs 50              # results/baselines.json
.venv/bin/python scripts/run_grid.py configs/lr_sweep.yaml
.venv/bin/python scripts/run_grid.py configs/fedavg_alpha.yaml
.venv/bin/python scripts/run_grid.py configs/fedavg_one_client.yaml
.venv/bin/python scripts/plots.py                              # report/figures/*.png (+ .csv tables)
.venv/bin/python -m pytest -q
```

## Data (`src/fl/data.py`)

| Step | Result |
|---|---|
| Source | UNB `CSV.zip`: 309 captures, 46,776,697 rows, 34 labels (one folder per label, no label column) |
| Sampling | per label, ≥ 3 captures read in random order; cap 25,000 rows per group, rare groups kept whole |
| Cleaning | NaN / ±inf rows dropped (`Rate` overflows), exact duplicates dropped per capture (~37% of rows in a sample) |
| Transform | `log1p` on all 39 features (non-negative, heavy-tailed: `Variance` up to 1e8), then StandardScaler fit on client train data only |
| Leakage guard | every float32 feature vector that repeats is removed (119 rows), before the split |
| Split | stratified 80/20; root set of 296 rows (37 per class) carved out of train for FLTrust, never given to clients |

| Class | train | test | root |
|---|---:|---:|---:|
| Benign | 19,961 | 5,000 | 37 |
| DDoS | 19,943 | 4,995 | 37 |
| DoS | 19,938 | 4,994 | 37 |
| Mirai | 19,963 | 5,000 | 37 |
| Recon | 17,761 | 4,449 | 37 |
| Spoofing | 19,953 | 4,997 | 37 |
| Web-based | 16,595 | 4,158 | 37 |
| BruteForce | 10,405 | 2,611 | 37 |
| **total** | **144,519** | **36,204** | **296** |

## Floor and ceiling (`scripts/baselines.py`)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 0.138 | 0.030 |
| Centralized MLP, SGD, 50 epochs (same lr/batch as FL) | 0.757 | **0.743** |
| Random forest, 100 trees | 0.782 | 0.766 |

The ceiling is low because per-window features of Benign, Web-based, BruteForce and Recon overlap
(per-class F1 0.58–0.68), while Mirai reaches 0.998. The MLP is within 0.02 of the random forest,
so the model is not the bottleneck.

## Hyperparameters (`configs/lr_sweep.yaml` → locked in `configs/base.yaml`)

FedAvg IID, 10 clients, 50 rounds, E = 1. Final macro-F1:

| lr \ batch | 64 | 256 |
|---|---:|---:|
| 0.01 | 0.738 | 0.711 |
| 0.05 | **0.747** | 0.737 |
| 0.1 | 0.749 | 0.742 |
| 0.2 | 0.748 | 0.745 |

Locked: **lr 0.05, batch 64, R = 50, E = 1, server lr 1.0** (within noise of the best, smoothest curve).

## FedAvg vs non-IID strength (`configs/fedavg_alpha.yaml`, 3 seeds)

| Partition | Macro-F1 at round 50 (mean ± sd) |
|---|---:|
| Dirichlet α = 0.1 | 0.520 ± 0.030 |
| Dirichlet α = 0.5 | 0.591 ± 0.017 |
| Dirichlet α = 1.0 | 0.657 ± 0.073 |
| Dirichlet α = 100 | 0.743 ± 0.004 |
| IID | 0.748 ± 0.002 |

Figures: `report/figures/fedavg_convergence.png`, `report/figures/partition_heatmap.png`.

## Compute budget (PLAN §0.3 update)

FedAvg run, 10 clients, 50 rounds, evaluation every round: **27.4 s per run with 8 runs in parallel**
(28.8 s worst case), i.e. ~8 runs per 28 s. The ~1,420-run grid costs ~1.4 h of FedAvg-equivalent work;
with ×3–4 for DeepSight, Min-Max and Opacus, ~4–6 h: one overnight run on the Mac.

## Sanity checks

| Check (PLAN Phase 1) | Status | Evidence |
|---|---|---|
| Majority-class and centralized baselines | ✅ | 0.030 / 0.743 |
| FedAvg with 1 client ≈ centralized | ✅ | 0.7429 vs 0.7429 (identical: same SGD path) |
| FedAvg α = 100 ≈ centralized; α = 0.1 clearly lower; monotone in α | ✅ | 0.743 ≈ 0.743; 0.520 < 0.591 < 0.657 < 0.743 |
| Partition: sizes sum to train, indices disjoint | ✅ | `test_dirichlet_partition_is_disjoint_and_complete` |
| Partition heatmap client × class | ✅ | `partition_heatmap.png` |
| No train/test duplicates (row hashes) | ✅ | `test_processed_data_is_clean` — caught 2 leaking rows from float32 rounding, fixed by deduplicating after the transform |
| Clients do not share parameters | ✅ | `test_clients_do_not_share_state` (fresh optimizer, update is a new tensor) |
| Aggregator: identical updates → same update; weights sum to 1 | ✅ | `test_fedavg_identical_updates_returns_that_update` |
| Determinism: same seed same metric, other seed different | ✅ | `test_same_seed_same_result_different_seed_different_result` |
| No NaN/inf after scaling | ✅ | assert in `prepare()` and in tests |
