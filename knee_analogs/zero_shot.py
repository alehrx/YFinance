"""Zero-shot trajectory forecasts across cell types: observe a cell only until its state of health
(SOH) first reaches 90 %, and forecast the rest of its curve without any ageing data of its own cell type.
Everything below was fixed before the first run.

Library: data/library (scripts/build_library.py), SOH vs equivalent full cycles (EFC).

Query: a cell with SOH points up to t90 (the first EFC at 90 % SOH, interpolated between the last point
above and the first point at or below 90 %) and at least one point after that. It is seen up to and
including the first point at or below 90 % (the measurement at which the test would stop); its future
after that point, up to 3 t90, is scored.

Shape: SOH at 0.1, 0.2, ..., 0.9 of t90 (a point (0, 100 %) is assumed at the start). Conditions:
temperature, log2 charge C-rate, log2 discharge C-rate (NaN filled with the memory median). Features are
z-scored with the memory's statistics.

Normalised future of a memory cell j: g_j(u) = SOH_j(u t90_j) for u > 1, NaN beyond its data. A forecast
for the query at EFC t uses u = t / t90_q.

Methods:
* ``lib_shape`` / ``lib_shape_cond``: memory = all cells of the *other* datasets. The 10 nearest by shape
  (or shape + conditions); weighted mean of g_j(u) (inverse distance) where >= 3 analogs have data;
  band = 10-90 % of those analogs.
* ``lib_median``: the median g_j(u) of every memory cell (the "one universal curve" hypothesis).
* ``own_linear``: the query's straight-line trend over (t90 / 2, t90].
* ``own_powerlaw``: 1 - SOH = a t^b fitted to the query's points up to t90 (b in [0.3, 3]), through 90 % at t90.
* ``chronos2`` / ``chronos_bolt``: zero-shot time-series foundation models on the query's curve up to t90
  (64 equally spaced points), 128 steps ahead (to 3 t90); median and 10-90 % band
  (scripts/chronos_forecast.py, run in the environment that has Chronos).
* ``within_shape`` (upper bound, uses ageing data of the same cell type): as ``lib_shape`` but the memory is
  the other cells of the query's own dataset.

Score: absolute SOH error (percentage points) per future point; per cell the mean, then the mean over cells
of each dataset, and the mean over datasets (each dataset counts once). Methods are compared on the points
all of them forecast; coverage is reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LIB = Path(__file__).resolve().parents[1] / "data" / "library"
FRACS = np.round(np.arange(0.1, 0.91, 0.1), 2)
K = 10
MIN_ANALOGS = 3
U_MAX = 3.0
CTX_POINTS = 64
PRED_STEPS = 128  # 64 points span [0, t90]; 128 more reach 3 t90


def load_library():
    meta = pd.read_csv(LIB / "cells.csv").set_index("id")
    curves = {}
    for cid in meta.index:
        d = pd.read_csv(LIB / "curves" / f"{cid}.csv")
        t, s = d["efc"].to_numpy(float), d["soh"].to_numpy(float)
        if t[0] > 0:
            t, s = np.r_[0.0, t], np.r_[1.0, s]
        curves[cid] = (t, s)
    return meta, curves


def t_at(t, s, level=0.9, with_stop=False):
    """EFC at which SOH first reaches ``level`` (interpolated); with ``with_stop`` also the EFC of the
    first measurement at or below it, where a test run to ``level`` would stop."""
    m = np.minimum.accumulate(s)
    j = np.flatnonzero(m <= level)
    if not len(j) or j[0] == 0:
        return (np.nan, np.nan) if with_stop else np.nan
    j = j[0]
    tx = float(t[j - 1] + (m[j - 1] - level) / (m[j - 1] - m[j]) * (t[j] - t[j - 1]))
    return (tx, float(t[j])) if with_stop else tx


def cell_table(meta, curves) -> pd.DataFrame:
    rows = []
    for cid, (t, s) in curves.items():
        t90, t_stop = t_at(t, s, with_stop=True)
        if not np.isfinite(t90) or t90 <= 0:
            continue
        seen = t <= t_stop
        r = {"id": cid, "dataset": meta.loc[cid, "dataset"], "t90": t90, "t_stop": t_stop, "n_seen": int(seen.sum()),
             "n_future": int(((t > t_stop) & (t <= U_MAX * t90)).sum())}
        for f in FRACS:
            r[f"s{f:.1f}"] = float(np.interp(f * t90, t, s))
        r["T"] = meta.loc[cid, "T"]
        r["lchg"] = np.log2(meta.loc[cid, "chg_C"]) if meta.loc[cid, "chg_C"] > 0 else np.nan
        r["ldis"] = np.log2(meta.loc[cid, "dis_C"]) if meta.loc[cid, "dis_C"] > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("id")


SHAPE = [f"s{f:.1f}" for f in FRACS]
COND = ["T", "lchg", "ldis"]


def g(curves, cid, t90, u):
    t, s = curves[cid]
    return np.interp(u * t90, t, s, left=np.nan, right=np.nan)


def _analog_forecast(Q, M, curves, cols, q, u):
    X = M[cols].astype(float)
    xq = Q.loc[q, cols].astype(float)
    med = X.median()
    X, xq = X.fillna(med), xq.fillna(med)
    sd = X.std().replace(0, 1).fillna(1)
    Z, zq = ((X - X.mean()) / sd).to_numpy(), ((xq - X.mean()) / sd).to_numpy()
    dist = np.sqrt(((Z - zq) ** 2).mean(1))
    idx = np.argsort(dist)[:K]
    w = 1 / (dist[idx] + 1e-6)
    F = np.array([g(curves, c, M.loc[c, "t90"], u) for c in M.index[idx]])
    ok = np.isfinite(F)
    cnt = ok.sum(0)
    ww = np.where(ok, w[:, None], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(ok, F, 0) * ww).sum(0) / ww.sum(0)
    lo = np.nanpercentile(np.where(ok, F, np.nan), 10, axis=0) if ok.any() else np.full(len(u), np.nan)
    hi = np.nanpercentile(np.where(ok, F, np.nan), 90, axis=0) if ok.any() else np.full(len(u), np.nan)
    bad = cnt < MIN_ANALOGS
    mean[bad] = lo[bad] = hi[bad] = np.nan
    return mean, lo, hi, ",".join(M.index[idx])


def own_powerlaw(t, s, t90, tt, t_stop):
    m = (t <= t_stop) & (t > 0) & (s < 1)
    if m.sum() < 3:
        return np.full(len(tt), np.nan)
    b, _ = np.polyfit(np.log(t[m]), np.log(1 - s[m]), 1)
    b = float(np.clip(b, 0.3, 3.0))
    a = np.exp(np.mean(np.log(1 - s[m]) - b * np.log(t[m])))
    return 0.9 - a * (tt ** b - t90 ** b)


def chronos_context(Q, curves) -> pd.DataFrame:
    """The query curves up to t90 on CTX_POINTS equally spaced EFC, for the Chronos runner."""
    rows = []
    for q in Q.index:
        t, s = curves[q]
        grid = np.linspace(0, Q.loc[q, "t90"], CTX_POINTS)  # interpolated from points up to t_stop only
        rows.append(pd.DataFrame({"id": q, "step": range(CTX_POINTS), "efc": grid, "soh": np.interp(grid, t, s)}))
    return pd.concat(rows)


def forecast(meta, curves, chronos: dict | None = None) -> pd.DataFrame:
    """One row per (query cell, future point within 3 t90)."""
    Q = cell_table(meta, curves)
    Q = Q[(Q["n_seen"] >= 4) & (Q["n_future"] >= 1)]
    T = cell_table(meta, curves)  # memory candidates: every cell with a t90 and data after it
    T = T[T["n_future"] >= 1]
    rows = []
    for q in Q.index:
        ds, t90, t_stop = Q.loc[q, "dataset"], Q.loc[q, "t90"], Q.loc[q, "t_stop"]
        t, s = curves[q]
        fut = (t > t_stop) & (t <= U_MAX * t90)
        tt, actual = t[fut], s[fut]
        u = tt / t90
        other = T[T["dataset"] != ds]
        same = T[(T["dataset"] == ds) & (T.index != q)]
        out = {}
        out["lib_shape"], out["lib_shape_lo"], out["lib_shape_hi"], nb1 = _analog_forecast(Q, other, curves, SHAPE, q, u)
        out["lib_shape_cond"], out["lib_shape_cond_lo"], out["lib_shape_cond_hi"], _ = _analog_forecast(
            Q, other, curves, SHAPE + COND, q, u)
        G = np.array([g(curves, c, other.loc[c, "t90"], u) for c in other.index])
        cnt = np.isfinite(G).sum(0)
        out["lib_median"] = np.where(cnt >= MIN_ANALOGS, np.nanmedian(np.where(np.isfinite(G), G, np.nan), 0), np.nan)
        m = (t > t90 / 2) & (t <= t_stop)
        slope = np.polyfit(t[m], s[m], 1)[0] if m.sum() >= 2 else np.nan
        out["own_linear"] = 0.9 + slope * (tt - t90)
        out["own_powerlaw"] = own_powerlaw(t, s, t90, tt, t_stop)
        if len(same) >= K:
            out["within_shape"], out["within_shape_lo"], out["within_shape_hi"], _ = _analog_forecast(
                Q, same, curves, SHAPE, q, u)
        for name, fc in (chronos or {}).items():
            if q in fc.index.get_level_values(0):
                f = fc.loc[q]
                for col in ("q50", "q10", "q90"):
                    key = name if col == "q50" else f"{name}_{'lo' if col == 'q10' else 'hi'}"
                    out[key] = np.interp(tt, f["efc"].to_numpy(), f[col].to_numpy(), left=np.nan, right=np.nan)
        for j in range(len(tt)):
            rows.append({"id": q, "dataset": ds, "t90": t90, "efc": tt[j], "u": u[j], "actual": actual[j],
                         "lib_analogs": nb1, **{k: float(v[j]) for k, v in out.items()}})
    return pd.DataFrame(rows)


def score(F: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset and overall (mean over datasets): MAE in SOH points on common points, by horizon."""
    methods = [m for m in methods if m in F and F[m].notna().any()]
    common = F[methods].notna().all(1)
    G = F[common].copy()
    G["h"] = pd.cut(G["u"], [1, 1.5, 2, 3], include_lowest=False)
    per = []
    for ds, D in G.groupby("dataset"):
        for m in methods:
            e = (D[m] - D["actual"]).abs() * 100
            r = {"dataset": ds, "method": m, "cells": D["id"].nunique(), "MAE": e.groupby(D["id"]).mean().mean()}
            for h, eh in e.groupby(D["h"], observed=True):
                r[f"u {h.left}-{h.right}"] = eh.groupby(D.loc[eh.index, "id"]).mean().mean()
            if f"{m}_lo" in D:
                r["band_covers_%"] = ((D["actual"] >= D[f"{m}_lo"]) & (D["actual"] <= D[f"{m}_hi"])).mean() * 100
            per.append(r)
    P = pd.DataFrame(per)
    cov = pd.Series({m: F[m].notna().mean() * 100 for m in methods}, name="coverage_%")
    overall = P.groupby("method").mean(numeric_only=True).drop(columns="cells").join(cov)
    overall["datasets"] = P.groupby("method")["dataset"].nunique()
    return P, overall.loc[methods]
