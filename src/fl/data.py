"""CICIoT2023 preprocessing: CSV.zip -> data/processed/ciciot_8cls.npz.

Pipeline (order matters, see notes on leakage):
  1. per label: read CSVs (captures) in random order until enough rows and >= 3 captures,
     drop NaN/inf, deduplicate
  2. sample each label so that every group gets at most `cap_per_group` rows
     (rare groups such as Web-based and BruteForce are kept whole)
  3. drop constant columns, log1p (all features are non-negative and heavy-tailed)
  4. drop every row whose (transformed, float32) feature vector repeats: removes train/test
     leakage and vectors labelled with two different classes
  5. stratified 80/20 train/test split; carve a balanced root set out of train (FLTrust)
  6. fit StandardScaler on the client part of train only, apply to all splits
"""
from __future__ import annotations

import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.fl.labels import GROUPS, to_group


@dataclass
class PrepConfig:
    zip_path: Path
    out_path: Path
    cap_per_group: int = 25_000
    read_factor: int = 4  # read about read_factor x the needed rows per label before sampling
    min_files: int = 3  # and at least this many captures per label, for diversity
    test_size: float = 0.2
    root_size: int = 300
    seed: int = 0


def _files_by_label(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    files = defaultdict(list)
    for name in zf.namelist():
        if name.endswith(".csv"):
            files[PurePosixPath(name).parent.name].append(name)
    return files


def _label_quota(files: dict[str, list[str]], cap_per_group: int) -> dict[str, int]:
    """Split each group's cap evenly over its labels (labels are sampled independently)."""
    labels_per_group = defaultdict(list)
    for label in files:
        labels_per_group[to_group(label)].append(label)
    return {
        label: -(-cap_per_group // len(labels))  # ceil
        for labels in labels_per_group.values()
        for label in labels
    }


def _read_label(zf, names, need, min_files, rng) -> pd.DataFrame:
    parts, rows = [], 0
    for i in rng.permutation(len(names)):
        df = pd.read_csv(zf.open(names[i]))
        df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
        parts.append(df)
        rows += len(df)
        if rows >= need and len(parts) >= min_files:
            break
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def load_sampled(cfg: PrepConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    with zipfile.ZipFile(cfg.zip_path) as zf:
        files = _files_by_label(zf)
        quota = _label_quota(files, cfg.cap_per_group)
        frames = []
        for label in sorted(files):
            df = _read_label(zf, sorted(files[label]), cfg.read_factor * quota[label], cfg.min_files, rng)
            if len(df) > quota[label]:
                df = df.sample(n=quota[label], random_state=cfg.seed)
            df["label"] = label
            frames.append(df)
            print(f"  {label:<26} {len(df):>7,} rows")
    data = pd.concat(frames, ignore_index=True)
    data["group"] = data["label"].map(to_group)
    # cap each group again (ceil rounding can overshoot by a few rows)
    capped = [
        g.sample(n=min(len(g), cfg.cap_per_group), random_state=cfg.seed)
        for _, g in data.groupby("group", sort=True)
    ]
    return pd.concat(capped, ignore_index=True)


def unique_rows_mask(X: np.ndarray) -> np.ndarray:
    """True for rows whose feature vector occurs exactly once.

    Every copy of a repeated vector is dropped, not all but one: copies can carry different
    labels, and keeping one would still let the same vector land in both train and test.
    """
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse.ravel()] == 1


def prepare(cfg: PrepConfig) -> dict:
    print(f"Reading {cfg.zip_path} (cap {cfg.cap_per_group:,} rows per group)")
    data = load_sampled(cfg)
    feature_cols = [c for c in data.columns if c not in ("label", "group")]

    constant = [c for c in feature_cols if data[c].nunique() <= 1]
    feature_cols = [c for c in feature_cols if c not in constant]
    print(f"Dropped constant columns: {constant}")

    # Deduplicate on the values the model actually sees: rows that differ only in the
    # 7th decimal can become identical after log1p + float32 and would leak across splits.
    X = np.log1p(data[feature_cols].to_numpy(dtype=np.float64)).astype(np.float32)
    keep = unique_rows_mask(X)
    print(f"Dropped {int((~keep).sum()):,} rows with duplicated features")
    X = X[keep].astype(np.float64)
    y = data.loc[keep, "group"].map({g: i for i, g in enumerate(GROUPS)}).to_numpy(dtype=np.int64)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
    )
    root_idx = _balanced_indices(y_tr, cfg.root_size, cfg.seed)
    mask = np.ones(len(y_tr), dtype=bool)
    mask[root_idx] = False
    X_root, y_root = X_tr[root_idx], y_tr[root_idx]
    X_tr, y_tr = X_tr[mask], y_tr[mask]

    scaler = StandardScaler().fit(X_tr)  # client data only: no test or root statistics leak in
    out = {
        "X_train": scaler.transform(X_tr).astype(np.float32),
        "y_train": y_tr,
        "X_test": scaler.transform(X_te).astype(np.float32),
        "y_test": y_te,
        "X_root": scaler.transform(X_root).astype(np.float32),
        "y_root": y_root,
        "feature_names": np.array(feature_cols),
        "classes": np.array(GROUPS),
    }
    for k in ("X_train", "X_test", "X_root"):
        assert np.isfinite(out[k]).all(), f"non-finite values in {k}"

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out_path, **out)
    return out


def _balanced_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = size // len(classes)
    picked = [rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in classes]
    return np.concatenate(picked)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def class_table(d: dict) -> pd.DataFrame:
    classes = list(d["classes"])
    table = pd.DataFrame(
        {
            split: np.bincount(d[f"y_{split}"], minlength=len(classes))
            for split in ("train", "test", "root")
        },
        index=classes,
    )
    table.loc["total"] = table.sum()
    return table
