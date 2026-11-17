# btl-fl-security

Robust aggregation for federated IDS on CICIoT2023 (FedAvg, FLTrust, EMA reputation). Plan: `PLAN.md`.

## Layout

```
data/raw/        CICIoT2023 CSVs (gitignored)
data/processed/  .npz, label_counts.csv (gitignored)
src/fl/          data, partition, model, client, server; labels.py (34 -> 8 classes)
src/attacks/ src/defenses/ src/privacy/
configs/         one YAML per experiment grid
scripts/         download, sanity checks, run_grid, aggregate, plots
results/         jsonl, one line per run
notes/           threat_model.md, hypotheses.md
report/
```

## Local (macOS, CPU/MPS)

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch==2.15.0 -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/gpu_sanity.py
```

## GPU server gpu2 (venv, verified 2026-11-16)

Quadro RTX 5000 (Turing, sm_75), driver 595.45.04 / CUDA 13.2. User is not in the `docker` group,
so the venv path is used. torch 2.15.0+cu132 includes sm_75; FP16 autocast works, no BF16 on Turing.
The GPU is shared: check `nvidia-smi` before long runs. Exact versions: `requirements.lock.gpu2.txt`.

```bash
cd ~/btl-fl-security
python3.11 -m venv .venv
.venv/bin/pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/gpu_sanity.py --require-cuda
```

## GPU server (Docker)

torch 2.15.0 on Linux ships CUDA builds `cu132` and `cu134` only, so check `nvidia-smi` first.
If the driver's CUDA version is lower, pass an older `TORCH_VERSION` with a matching `TORCH_CUDA`.

```bash
docker build -t btl-fl --build-arg TORCH_CUDA=cu132 .
docker run --rm --gpus all btl-fl python scripts/gpu_sanity.py --require-cuda
docker run --rm --gpus all -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" btl-fl <command>
```

## Data

UNB asks for a registration form, and the file browser only works in that browser session
(scripted requests get 403), so download by hand:
https://www.unb.ca/cic/datasets/iotdataset-2023.html -> `CSV/CSV` -> save as `data/raw/CSV.zip` (1.4 GB zipped, 8.9 GB unzipped).

The zip has no label column: each of the 34 classes is a folder, `CSV/<label>/<capture>.pcap.csv`.
Scripts read the zip directly, no extraction needed.

```bash
.venv/bin/python scripts/count_labels.py   # -> data/processed/label_counts.csv; exit 2 if a group is missing
```
