"""Count rows per label (34 classes) and per group (8 classes) in CICIoT2023.

The UNB "CSV" release has no label column: each class is a folder,
CSV/<label>/<capture>.pcap.csv. The label is therefore the parent folder name.
Reads straight from data/raw/CSV.zip (no extraction) or from an extracted folder.
Writes data/processed/label_counts.csv and flags missing groups.
"""
import argparse
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fl.labels import GROUPS, to_group  # noqa: E402


def count_rows(stream: io.BufferedIOBase) -> int:
    lines = sum(chunk.count(b"\n") for chunk in iter(lambda: stream.read(1 << 20), b""))
    return max(lines - 1, 0)  # minus header


def iter_csvs(source: Path):
    """Yield (label, n_rows) for every CSV, label = parent folder name."""
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if name.endswith(".csv"):
                    with zf.open(name) as f:
                        yield PurePosixPath(name).parent.name, count_rows(f)
    else:
        for csv in sorted(source.rglob("*.csv")):
            with csv.open("rb") as f:
                yield csv.parent.name, count_rows(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "CSV.zip",
                        help="CSV.zip or the extracted CSV/ folder")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "label_counts.csv")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"Not found: {args.source}")
        return 1

    rows, files = Counter(), Counter()
    for label, n in iter_csvs(args.source):
        rows[label] += n
        files[label] += 1

    table = pd.DataFrame({"rows": pd.Series(rows), "files": pd.Series(files)})
    table.index.name = "label"
    table = table.sort_values("rows", ascending=False)
    table["group"] = [to_group(label) for label in table.index]
    by_group = table.groupby("group")[["rows", "files"]].sum().reindex(GROUPS, fill_value=0)
    by_group["share_%"] = (100 * by_group["rows"] / by_group["rows"].sum()).round(3)

    print(f"{int(table['files'].sum())} files, {int(table['rows'].sum()):,} rows, {len(table)} labels\n")
    print(table.to_string())
    print("\nPer group:")
    print(by_group.to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    print(f"\nSaved {args.out}")

    missing = [g for g, n in by_group["rows"].items() if n == 0]
    if missing:
        print(f"\nMISSING groups: {missing}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
