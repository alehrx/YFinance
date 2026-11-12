"""Read-only overview of a folder of battery datasets, so an adapter can be written for them.

    python -m knee_analogs.inspect_datasets "C:\\Users\\alexa\\Datasets Batteries" > datasets_overview.txt

Lists every sub-folder with its file types, and for a few sample files per folder prints the
columns / variable names, their shape and the first rows. It never modifies anything and never
loads pickle files (they can execute code).
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

SAMPLES_PER_FOLDER = 2
TABULAR = {".csv", ".txt", ".tsv", ".xlsx", ".xls", ".parquet", ".feather"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def peek_table(path: str, ext: str) -> str:
    import pandas as pd

    if ext in (".csv", ".txt", ".tsv"):
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(path, sep=sep, engine="python", nrows=5)
    elif ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, nrows=5)
        return "\n".join(f"      sheet '{s}': columns {list(d.columns)[:30]}\n{d.head(3).to_string()[:800]}"
                         for s, d in list(sheets.items())[:4])
    elif ext == ".parquet":
        df = pd.read_parquet(path).head(5)
    else:
        df = pd.read_feather(path).head(5)
    return f"      columns: {list(df.columns)[:40]}\n{df.head(3).to_string()[:800]}"


def peek_mat(path: str) -> str:
    try:
        from scipy.io import loadmat, whosmat

        return "      variables: " + "; ".join(f"{n} {s} {t}" for n, s, t in whosmat(path)[:30])
    except NotImplementedError:  # MATLAB v7.3 = HDF5
        return peek_h5(path)


def peek_h5(path: str) -> str:
    try:
        import h5py
    except ImportError:
        return "      (HDF5 file: pip install h5py to inspect)"
    lines = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if len(lines) < 40 and hasattr(obj, "shape"):
                lines.append(f"      {name} {obj.shape} {obj.dtype}")
        f.visititems(visit)
    return "\n".join(lines)


def main(root: str):
    by_dir = defaultdict(list)
    for d, _, files in os.walk(root):
        for f in files:
            by_dir[d].append(f)
    total = sum(len(v) for v in by_dir.values())
    print(f"ROOT {root}: {total} files in {len(by_dir)} folders\n")
    for d in sorted(by_dir):
        files = sorted(by_dir[d])
        exts = Counter(os.path.splitext(f)[1].lower() for f in files)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        print(f"== {os.path.relpath(d, root)}  ({len(files)} files, {human(size)}): "
              + ", ".join(f"{e or '(none)'} x{n}" for e, n in exts.most_common()))
        print("   e.g. " + ", ".join(files[:6]) + (" ..." if len(files) > 6 else ""))
        shown = 0
        for f in files:
            if shown >= SAMPLES_PER_FOLDER:
                break
            p, ext = os.path.join(d, f), os.path.splitext(f)[1].lower()
            try:
                if ext in TABULAR:
                    info = peek_table(p, ext)
                elif ext == ".mat":
                    info = peek_mat(p)
                elif ext in (".h5", ".hdf5"):
                    info = peek_h5(p)
                elif ext in (".pkl", ".pickle"):
                    info = "      (pickle: not opened for safety; tell me which library produced it)"
                else:
                    continue
            except Exception as e:  # keep going on unreadable files
                info = f"      could not read: {type(e).__name__}: {str(e)[:150]}"
            print(f"   -- {f} ({human(os.path.getsize(p))})\n{info}")
            shown += 1
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
