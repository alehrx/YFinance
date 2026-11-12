"""Download the public Severson et al. (2019) batch 2017-06-30 per-cell summary files.

    python scripts/fetch_severson_b2.py          ->  data/severson_b2/2017_06_30_cell*_data.csv

Source: a GitHub mirror (github.com/Txiaoxiao/LSTM-Neural-Network-for-Time-Series-Prediction,
folder data/). The original data is at data.matr.io (Severson et al., Nature Energy 4, 383, 2019).
"""

import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/Txiaoxiao/LSTM-Neural-Network-for-Time-Series-Prediction/"
        "master/data/2017_06_30_cell{}_data.csv")
OUT = Path(__file__).resolve().parent.parent / "data" / "severson_b2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(48):
        dest = OUT / f"2017_06_30_cell{i}_data.csv"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        with urllib.request.urlopen(BASE.format(i), timeout=60) as r:
            dest.write_bytes(r.read())
        print("downloaded", dest.name)
    print(f"{len(list(OUT.glob('*.csv')))} files in {OUT}")


if __name__ == "__main__":
    main()
