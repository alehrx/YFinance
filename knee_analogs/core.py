"""Battery ageing analogs: predict the capacity 'knee' of a cell from the early cycles of
other cells that looked similar (nearest-neighbour analogs, a "numeric RAG"), compared with regression.

Data layout: ``load_cells(folder)`` expects one CSV per cell with per-cycle summary columns
``cycle, QD`` (discharge capacity, Ah) and optionally ``IR, Tavg, chargetime, charge_policy,
condition``. An optional ``dataset.json`` in the folder sets the per-dataset settings in
``DatasetSpec`` (nominal capacity, end-of-life fraction, sparse check-up data, test cycles).
Without it the Severson et al. (2019) batch-2 settings are used, so those results reproduce.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Severson 2019 batch 2 (A123 APR18650M1A); defaults of DatasetSpec, kept for old imports.
EOL_FRACTION = 0.8
NOMINAL_AH = 1.1


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset settings, read from ``<data folder>/dataset.json`` (all keys optional).

    Fix these from the dataset description before looking at any test result.
    """

    name: str = "severson_b2"
    nominal_ah: float = NOMINAL_AH
    eol_fraction: float = EOL_FRACTION           # end of life at this fraction of nominal
    # "nominal": end of life = eol_fraction x nominal_ah (Severson); "initial": eol_fraction x the
    # cell's own initial capacity (median of its first 10 smoothed points), for datasets whose
    # protocols use different voltage windows
    eol_basis: str = "nominal"
    # cells whose first cycles are below this fraction of nominal are skipped (Severson batch 2:
    # cells continued from batch 1). None keeps every cell.
    min_start_fraction: float | None = 1.04 / 1.1
    # True when capacity is only measured at check-ups (tens of points per cell, not hundreds)
    sparse: bool = False
    min_points: int | None = None                # default 150 dense, 8 sparse
    early_cycles: tuple = (50, 100, 150, 200)    # Test A: predict from the first N cycles
    main_n: int = 100                            # the N used for figures and the headline
    # Test B: checkpoints, snapshot spacing, and the "knee within" window, in cycles
    checkpoints: tuple = (100, 150, 200, 250, 300, 350)
    snapshot_step: int = 10
    snapshot_start: int = 60
    state_window: int = 50                       # fade rate is measured over this many cycles
    warning_window: int = 50
    # Where the knee and end of life are measured. "QD": the per-cycle capacity (Severson).
    # "checkup": the slow check-up capacities in <folder>/checkups/<cell>.csv (columns cycle, QD),
    # treated as sparse points; the per-cycle table still provides every input feature.
    knee_from: str = "QD"
    # only breakpoints where the fade accelerates count as a knee
    knee_accelerating_only: bool = False
    # cells left out of the study, {cell: reason}; decided from the capacity curves only
    exclude_cells: dict = field(default_factory=dict)
    # cells without a clear knee that stay in the end-of-life test, {cell: reason}
    no_knee_cells: dict = field(default_factory=dict)
    # fit the knee only up to end of life (Severson) or over the whole measured trajectory
    knee_until_eol: bool = True
    extra: dict = field(default_factory=dict)    # free-form notes (chemistry, conditions...)

    @property
    def eol_ah(self) -> float:
        return self.eol_fraction * self.nominal_ah

    @property
    def min_pts(self) -> int:
        return self.min_points if self.min_points is not None else (8 if self.sparse else 150)

    @classmethod
    def from_folder(cls, folder: str) -> "DatasetSpec":
        p = os.path.join(folder, "dataset.json")
        if not os.path.exists(p):
            return cls()
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{p}: unknown keys {sorted(unknown)}; allowed: {sorted(known)}")
        for k in ("early_cycles", "checkpoints"):
            if k in raw:
                raw[k] = tuple(raw[k])
        return cls(**raw)

    def to_json(self, folder: str) -> None:
        with open(os.path.join(folder, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)


def spec_of(d: pd.DataFrame) -> DatasetSpec:
    return d.attrs.get("spec") or DatasetSpec()


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


def _clean(d: pd.DataFrame, spec: DatasetSpec | None = None) -> pd.DataFrame:
    """Drop impossible values and glitches, add the smoothed capacity ``QDs``.

    Dense data (one point per cycle): rolling-median glitch filter and smoothing, as tuned on
    Severson batch 2 (all thresholds scale with nominal capacity). Sparse check-up data: only
    remove points far from both neighbours, and do not smooth (``QDs`` = ``QD``).
    """
    spec = spec or DatasetSpec()
    nom = spec.nominal_ah
    d = d.sort_values("cycle").drop_duplicates("cycle").reset_index(drop=True)
    d = d[(d["QD"] > 0.5 / 1.1 * nom) & (d["QD"] < 1.3 / 1.1 * nom)].copy()
    if not spec.sparse:
        med = d["QD"].rolling(9, center=True, min_periods=3).median()
        d = d[(d["QD"] - med).abs() < 0.015 / 1.1 * nom].copy()
        d["QDs"] = d["QD"].rolling(7, center=True, min_periods=3).median().rolling(5, center=True, min_periods=2).mean()
    else:
        med = d["QD"].rolling(3, center=True, min_periods=3).median()
        d = d[~((d["QD"] - med).abs() > 0.03 * nom)].copy()  # NaN at the ends -> kept
        d["QDs"] = d["QD"]
    return d.reset_index(drop=True)


def _natural_key(f: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)]


def load_cells(folder: str, pattern: str = "*.csv", min_start_capacity: float | None = None,
               spec: DatasetSpec | None = None) -> dict:
    """Returns {cell_name: DataFrame}; each frame carries ``attrs['spec']``, ``attrs['policy']``
    and ``attrs['condition']``.

    Files with identical content are loaded once (a duplicate would match itself). Cells with
    fewer than ``spec.min_pts`` points, or starting below ``min_start_capacity`` Ah (default:
    ``spec.min_start_fraction`` x nominal), are skipped.
    """
    spec = spec or DatasetSpec.from_folder(folder)
    if min_start_capacity is None and spec.min_start_fraction is not None:
        min_start_capacity = spec.min_start_fraction * spec.nominal_ah
    cells, seen = {}, set()
    for f in sorted(glob.glob(os.path.join(folder, pattern)), key=_natural_key):
        with open(f, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        d = pd.read_csv(f)
        if not {"cycle", "QD"} <= set(d.columns):
            continue
        d = _clean(d, spec)
        if len(d) < spec.min_pts:
            continue
        if min_start_capacity is not None and d["QDs"].iloc[:10].median() < min_start_capacity:
            continue
        name = re.sub(r"_data\.csv$|\.csv$", "", os.path.basename(f))
        if name in spec.exclude_cells:
            continue
        d.attrs["spec"] = spec
        d.attrs["q_initial"] = float(d["QDs"].iloc[:10].median())
        d.attrs["policy"] = str(d["charge_policy"].iloc[0]) if "charge_policy" in d else ""
        d.attrs["condition"] = str(d["condition"].iloc[0]) if "condition" in d else d.attrs["policy"]
        if spec.knee_from == "checkup":
            p = os.path.join(os.path.dirname(f), "checkups", os.path.basename(f))
            if not os.path.exists(p):
                continue
            c = _clean(pd.read_csv(p)[["cycle", "QD"]], dataclasses.replace(spec, sparse=True))
            if len(c) < 6:
                continue
            # tuples, not a DataFrame: pandas compares attrs when combining frames
            d.attrs["ref"] = (tuple(c["cycle"].astype(float)), tuple(c["QDs"].astype(float)))
            d.attrs["ref_q_initial"] = float(c["QDs"].iloc[:3].median())
        cells[name] = d
    return cells


def target_frame(d: pd.DataFrame) -> pd.DataFrame:
    """The capacity series on which knee and end of life are measured: the cell's own table, or
    its check-up capacities when ``spec.knee_from == "checkup"`` (sparse points, no smoothing)."""
    spec = spec_of(d)
    if spec.knee_from == "QD":
        return d
    if spec.knee_from != "checkup":
        raise ValueError(f"knee_from must be 'QD' or 'checkup', not {spec.knee_from!r}")
    if "ref" not in d.attrs:  # already a target frame, or a cell without check-ups
        raise ValueError("knee_from='checkup' but this cell has no check-up data")
    t = pd.DataFrame({"cycle": d.attrs["ref"][0], "QDs": d.attrs["ref"][1]})
    t["QD"] = t["QDs"]
    t.attrs["spec"] = dataclasses.replace(spec, sparse=True, knee_from="QD")
    t.attrs["q_initial"] = d.attrs["ref_q_initial"]
    return t


def near_duplicates(cells: dict, tol_frac: float = 0.002) -> list[tuple[str, str, float]]:
    """Pairs of cells whose capacity curves are nearly identical on their common cycles
    (RMS difference below ``tol_frac`` x nominal): likely the same cell exported twice.
    Check these before testing; exact duplicates are already dropped by ``load_cells``."""
    names = list(cells)
    out = []
    for i, a in enumerate(names):
        da = cells[a]
        for b in names[i + 1:]:
            db = cells[b]
            m = da[["cycle", "QD"]].merge(db[["cycle", "QD"]], on="cycle")
            if len(m) < 0.8 * min(len(da), len(db)):
                continue
            rms = float(np.sqrt(((m["QD_x"] - m["QD_y"]) ** 2).mean()))
            if rms < tol_frac * spec_of(da).nominal_ah:
                out.append((a, b, rms))
    return out


def eol_ah(d: pd.DataFrame) -> float:
    """End-of-life capacity of this cell (Ah), from ``spec.eol_basis``."""
    d = target_frame(d)
    spec = spec_of(d)
    if spec.eol_basis == "initial":
        q0 = d.attrs.get("q_initial", float(d["QDs"].iloc[:10].median()))
        return spec.eol_fraction * q0
    if spec.eol_basis != "nominal":
        raise ValueError(f"eol_basis must be 'nominal' or 'initial', not {spec.eol_basis!r}")
    return spec.eol_ah


def eol_cycle(d: pd.DataFrame) -> float:
    """First cycle below end of life (interpolated between check-ups for sparse data); the last
    cycle if the cell never got there (see ``reached_eol``)."""
    d = target_frame(d)
    spec = spec_of(d)
    eol = eol_ah(d)
    below = np.flatnonzero(d["QDs"].to_numpy() < eol)
    if not len(below):
        return float(d["cycle"].iloc[-1])
    j = below[0]
    if not spec.sparse or j == 0:
        return float(d["cycle"].iloc[j])
    x0, x1 = d["cycle"].iloc[j - 1], d["cycle"].iloc[j]
    y0, y1 = d["QDs"].iloc[j - 1], d["QDs"].iloc[j]
    return float(x0 + (y0 - eol) / (y0 - y1) * (x1 - x0))


def reached_eol(d: pd.DataFrame) -> bool:
    d = target_frame(d)
    return bool((d["QDs"] < eol_ah(d)).any())


def knee_point(d: pd.DataFrame, until_eol: bool | None = None) -> tuple[float, float, float]:
    """Bacon-Watts-style knee: the breakpoint of a continuous two-segment linear fit to the
    smoothed capacity curve (up to end of life). Returns (knee cycle, slope before, slope after),
    slopes in mAh per cycle. For sparse data the breakpoint is kept at least 3 check-ups from
    either end, so each segment rests on real points; with fewer than 6 points it is NaN.
    With ``spec.knee_accelerating_only`` only breakpoints where the fade gets faster are allowed.
    The series is ``target_frame(d)`` (the check-ups if ``spec.knee_from == "checkup"``)."""
    d = target_frame(d)
    accel_only = spec_of(d).knee_accelerating_only
    x = d["cycle"].to_numpy(float)
    y = d["QDs"].to_numpy(float)
    if until_eol is None:
        until_eol = spec_of(d).knee_until_eol
    if until_eol:
        keep = x <= eol_cycle(d)
        x, y = x[keep], y[keep]
    if spec_of(d).sparse:
        if len(x) < 6:
            return np.nan, np.nan, np.nan
        lo, hi = x[2], x[-3]
    else:
        lo, hi = np.quantile(x, [0.15, 0.97])
    best = (np.inf, np.nan, np.nan, np.nan)
    for b in np.linspace(lo, hi, 200):
        A = np.column_stack([np.ones_like(x), x, np.maximum(0, x - b)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if accel_only and coef[2] >= 0:
            continue
        sse = ((A @ coef - y) ** 2).sum()
        if sse < best[0]:
            best = (sse, b, coef[1], coef[1] + coef[2])
    return best[1], best[2] * 1000, best[3] * 1000


def early_features(d: pd.DataFrame, n: int) -> dict:
    """Summary features from the first ``n`` cycles only (Severson-style 'summary' features).
    For sparse data, the start is the first check-up and the recent window at least the last
    two check-ups."""
    spec = spec_of(d)
    e = d[d["cycle"] <= n]
    if len(e) < (4 if spec.sparse else max(20, n // 3)):
        return {}
    x, y = e["cycle"].to_numpy(float), e["QDs"].to_numpy(float)
    first = (x >= 2) & (x <= 10)
    q_start = np.median(y[first]) if first.any() else y[0]
    tail = (x > n - 0.25 * n)
    if tail.sum() < 2:
        tail = np.zeros(len(x), bool)
        tail[-2:] = True
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
    out.update(protocol_features(d))
    return out


def protocol_features(d: pd.DataFrame) -> dict:
    """Descriptors of the test protocol, fixed before the test starts (so known at prediction
    time): the Severson charge policy if the cell has one, plus any ``proto_*`` column of the
    per-cell table (constant per cell, written by the dataset adapter)."""
    out = {}
    if d.attrs.get("policy", ""):
        out.update(parse_policy(d.attrs["policy"]))
    for col in d.columns:
        if col.startswith("proto_"):
            out[col] = float(d[col].iloc[0])
    return out


def curve_vector(d: pd.DataFrame, n: int, points: int = 20) -> np.ndarray:
    """The capacity-fade curve over the first ``n`` cycles, resampled to ``points`` values,
    relative to its own start (shape of the curve, not the absolute capacity)."""
    e = d[d["cycle"] <= n]
    grid = np.linspace(max(5, e["cycle"].min()), n, points)
    q = np.interp(grid, e["cycle"], e["QDs"])
    return (q - q[0]) * 1000  # mAh lost since start
