"""Build data/processed/ciciot_8cls.npz from data/raw/CSV.zip and print the class table."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fl.data import PrepConfig, class_table, prepare  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=ROOT / "data" / "raw" / "CSV.zip")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "ciciot_8cls.npz")
    p.add_argument("--cap", type=int, default=25_000, help="max rows per group before dedup/split")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    start = time.perf_counter()
    d = prepare(PrepConfig(zip_path=a.zip, out_path=a.out, cap_per_group=a.cap, seed=a.seed))
    table = class_table(d)
    print(f"\n{len(d['feature_names'])} features, saved {a.out} in {time.perf_counter() - start:.0f}s\n")
    print(table.to_string())
    table.to_csv(a.out.with_name("class_table.csv"))


if __name__ == "__main__":
    main()
