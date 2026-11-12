"""Battery ageing analogs: predict the capacity 'knee' of a cell from the early cycles of
other cells that looked similar (numeric RAG / nearest-neighbour), compared with regression.

Data adapter: ``load_cells(folder)`` expects one CSV per cell with per-cycle summary columns
``cycle, QD`` (discharge capacity, Ah) and optionally ``IR, Tavg, chargetime, charge_policy``.
The Severson et al. (2019) batch-2 files (2017_06_30_cell*_data.csv) have exactly this layout;
other datasets only need a small adapter producing the same columns.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

EOL_FRACTION = 0.8          # end of life at 80% of nominal capacity
NOMINAL_AH = 1.1            # A123 APR18650M1A


def parse_policy(p: str) -> dict:
    """'4C(13%)-5C' -> C1=4, Q1=13, C2=5 (charge at C1 up to Q1% SOC, then C2 to 80%)."""
    m = re.match(r"^\s*([\d.]+)C\((\d+)%\)-([\d.]+)C?", str(p))
    if not m:
        return {"C1": np.nan, "Q1": np.nan, "C2": np.nan, "C_avg": np.nan}
    c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    q1c = min(q1, 80.0)
    # average C-rate over 0-80% SOC (time-weighted): 80 / (q1/c1 + (80-q1)/c2)
    c_avg = 80.0 / (q1c / c1 + (80.0 - q1c) / c2)
    return {"C1": c1, "Q1": q1, "C2": c2, "C_avg": c_avg}


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("cycle").drop_duplicates("cycle").reset_index(drop=True)
    d = d[(d["QD"] > 0.5) & (d["QD"] < 1.3)].copy()
    med = d["QD"].rolling(9, center=True, min_periods=3).median()
    d = d[(d["QD"] - med).abs() < 0.015].copy()          # drop measurement glitches
    d["QDs"] = d["QD"].rolling(7, center=True, min_periods=3).median().rolling(5, center=True, min_periods=2).mean()
    return d.reset_index(drop=True)


def load_cells(folder: str, pattern: str = "*.csv", min_start_capacity: float = 1.04) -> dict:
    """Returns {cell_name: DataFrame}. Cells that start below ``min_start_capacity`` Ah are
    skipped (in Severson batch 2 these continued from batch 1, so their history is incomplete)."""
    import hashlib

    cells, seen = {}, set()
    for f in sorted(glob.glob(os.path.join(folder, pattern)), key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)]):
        digest = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if digest in seen:  # identical file under another name: would match itself
            continue
        seen.add(digest)
        d = pd.read_csv(f)
        if not {"cycle", "QD"} <= set(d.columns):
            continue
        d = _clean(d)
        if len(d) < 150 or d["QDs"].iloc[:10].median() < min_start_capacity:
            continue
        name = re.sub(r"_data\.csv$|\.csv$", "", os.path.basename(f))
        d.attrs["policy"] = str(d["charge_policy"].iloc[0]) if "charge_policy" in d else ""
        cells[name] = d
    return cells


def eol_cycle(d: pd.DataFrame) -> float:
    below = d.loc[d["QDs"] < EOL_FRACTION * NOMINAL_AH, "cycle"]
    return float(below.iloc[0]) if len(below) else float(d["cycle"].iloc[-1])


def knee_point(d: pd.DataFrame, until_eol: bool = True) -> tuple[float, float, float]:
    """Bacon-Watts-style knee: the breakpoint of a continuous two-segment linear fit to the
    smoothed capacity curve (up to end of life). Returns (knee cycle, slope before, slope after),
    slopes in mAh per cycle."""
    x = d["cycle"].to_numpy(float)
    y = d["QDs"].to_numpy(float)
    if until_eol:
        keep = x <= eol_cycle(d)
        x, y = x[keep], y[keep]
    best = (np.inf, np.nan, np.nan, np.nan)
    lo, hi = np.quantile(x, [0.15, 0.97])
    for b in np.linspace(lo, hi, 200):
        A = np.column_stack([np.ones_like(x), x, np.maximum(0, x - b)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = ((A @ coef - y) ** 2).sum()
        if sse < best[0]:
            best = (sse, b, coef[1], coef[1] + coef[2])
    return best[1], best[2] * 1000, best[3] * 1000


def early_features(d: pd.DataFrame, n: int) -> dict:
    """Summary features from the first ``n`` cycles only (Severson-style 'summary' features)."""
    e = d[d["cycle"] <= n]
    if len(e) < max(20, n // 3):
        return {}
    x, y = e["cycle"].to_numpy(float), e["QDs"].to_numpy(float)
    q_start = np.median(y[(x >= 2) & (x <= 10)])
    tail = (x > n - 0.25 * n)
    out = {
        "q_start": q_start,
        "q_n": np.median(y[tail]),
        "fade_n": np.median(y[tail]) - q_start,                         # capacity lost so far
        "slope_all": np.polyfit(x, y, 1)[0] * 1000,                      # mAh / cycle
        "slope_late": np.polyfit(x[tail], y[tail], 1)[0] * 1000,         # recent fade rate
        "curv": np.polyfit(x, y, 2)[0] * 1e6,                            # acceleration of fade
        "q_max_minus_start": y.max() - q_start,
    }
    for col in ("IR", "Tavg", "chargetime"):
        if col in e:
            v = e[col].replace(0, np.nan)
            out[f"{col}_start"] = v.iloc[:10].median()
            out[f"{col}_delta"] = v[e["cycle"] > n - 0.25 * n].median() - v.iloc[:10].median()
    out.update(parse_policy(d.attrs.get("policy", "")))
    return out


def curve_vector(d: pd.DataFrame, n: int, points: int = 20) -> np.ndarray:
    """The capacity-fade curve over the first ``n`` cycles, resampled to ``points`` values,
    relative to its own start (shape of the curve, not the absolute capacity)."""
    e = d[d["cycle"] <= n]
    grid = np.linspace(max(5, e["cycle"].min()), n, points)
    q = np.interp(grid, e["cycle"], e["QDs"])
    return (q - q[0]) * 1000  # mAh lost since start
