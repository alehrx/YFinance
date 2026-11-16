"""Forecast a cell's whole future ageing trajectory from its first N cycles, leave-one-cell-out.

Not one number (a knee or an end-of-life cycle) but the capacity curve itself: the state of health
(SOH) at every later check-up, with a band. Everything below was fixed before the first run.

Series. SOH = capacity of the knee series (``target_frame``: the slow check-ups when
``knee_from == "checkup"``) divided by the nominal capacity (``eol_basis == "nominal"``) or by the
cell's initial capacity (``"initial"``).

Inputs at N. The query's early-cycle features (``early_features``, as in Test A) and its SOH points
with cycle <= N. Its current SOH (the *anchor*) is a straight-line fit through its SOH points in
(N/2, N], evaluated at N (the last point if there are fewer than 2).

Methods (``k`` = 5 analogs; analog candidates are the other cells with measurements after N):
* ``analogs_cycle``: the 5 nearest cells in scaled early-feature space (inverse-distance weights,
  scaling fit on the other cells only). Each analog's change after N, y_j(t) - anchor_j(N), is
  added to the query's anchor. Weighted mean where >= 3 analogs have data; band = min-max of them.
* ``analogs_state``: the same analogs, each aligned at the cycle tau_j where its SOH first equals the
  query's anchor (its first point if it starts below), then followed: anchor + y_j(tau_j + D) -
  y_j(tau_j), D = t - N. Same weights, coverage rule and band.
* ``own_linear``: the query's own straight-line trend over (N/2, N], extrapolated.
* ``own_powerlaw``: 1 - SOH = a t^b fitted to the query's points <= N (b clipped to [0.3, 3]),
  shifted to pass through the anchor.
* ``population``: unweighted mean of all candidate cells' anchored same-cycle futures (>= 3 cells).
* ``same_condition``: the same over the candidates of the query's condition (>= 1 cell).
* ``regression``: ridge regression of y(t) - anchor(N) on the scaled features at 30 cycles
  geometrically spaced between N and the last measured cycle (training cells with data at t, >= 5),
  interpolated to the query's check-ups.

Scoring. Absolute SOH error in percentage points at every measured check-up after N; per cell the
mean, then the mean over cells. Methods are compared on the points every method predicts
(``common``), with each method's own coverage reported; errors by horizon quartile; and for the
analogs, how often the measured SOH lies inside the band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from . import evaluate as ev
from .core import spec_of, target_frame

K = 5
MIN_ANALOGS = 3
METHODS = ["analogs_cycle", "analogs_state", "own_linear", "own_powerlaw", "population", "same_condition",
           "regression"]


def soh_series(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = target_frame(d)
    spec = spec_of(d)
    denom = spec.nominal_ah if spec.eol_basis == "nominal" else t.attrs.get("q_initial", d.attrs["q_initial"])
    return t["cycle"].to_numpy(float), t["QDs"].to_numpy(float) / denom


def anchor(x: np.ndarray, y: np.ndarray, n: float) -> float:
    m = (x > n / 2) & (x <= n)
    if m.sum() >= 2:
        a, b = np.polyfit(x[m], y[m], 1)
        return float(a * n + b)
    m = x <= n
    return float(y[m][-1]) if m.any() else np.nan


def _interp(x, y, t):
    return np.interp(t, x, y, left=np.nan, right=np.nan)


def future_same_cycle(xj, yj, n, a_q, t):
    return a_q + _interp(xj, yj, t) - anchor(xj, yj, n)


def future_same_state(xj, yj, n, a_q, t):
    ymono = np.minimum.accumulate(yj)
    below = np.flatnonzero(ymono <= a_q)
    if not len(below):
        return np.full(len(t), np.nan)  # the analog never got as low as the query is now
    j = below[0]
    if j == 0:
        tau = xj[0]
    else:  # interpolate the crossing
        y0, y1 = ymono[j - 1], ymono[j]
        tau = xj[j - 1] + (y0 - a_q) / (y0 - y1) * (xj[j] - xj[j - 1]) if y0 != y1 else xj[j]
    y_tau = _interp(xj, yj, np.array([tau]))[0]
    return a_q + _interp(xj, yj, tau + (t - n)) - y_tau


def _combine(futs: np.ndarray, w: np.ndarray):
    ok = np.isfinite(futs)
    cnt = ok.sum(0)
    ww = np.where(ok, w[:, None], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(ok, futs, 0) * ww).sum(0) / ww.sum(0)
    lo, hi = np.nanmin(np.where(ok, futs, np.inf), 0), np.nanmax(np.where(ok, futs, -np.inf), 0)
    bad = cnt < MIN_ANALOGS
    mean[bad] = lo[bad] = hi[bad] = np.nan
    return mean, lo, hi


def own_powerlaw(x, y, n, a_q, t):
    m = (x <= n) & (x > 0) & (y < 1)
    if m.sum() < 3:
        return np.full(len(t), np.nan)
    b, la = np.polyfit(np.log(x[m]), np.log(1 - y[m]), 1)
    b = float(np.clip(b, 0.3, 3.0))
    a = np.exp(np.mean(np.log(1 - y[m]) - b * np.log(x[m])))
    return a_q - a * (t ** b - n ** b)


def forecast(cells: dict, n: float) -> pd.DataFrame:
    """One row per (query cell, measured check-up after n): actual SOH and every method's forecast."""
    T, _ = ev.cell_table(cells, n)
    feats = [c for c in T.columns if c not in ev.NON_FEATURES]
    series = {k: soh_series(cells[k]) for k in T.index}
    has_future = np.array([series[k][0][-1] > n for k in T.index])
    rows = []
    for i, q in enumerate(T.index):
        xq, yq = series[q]
        t = xq[xq > n]
        if not len(t):
            continue
        a_q = anchor(xq, yq, n)
        cand = (T.index != q) & has_future  # the query never sees itself
        Ttr = T[cand]
        Xtr = ev._scale(Ttr[feats], Ttr[feats])
        xte = ev._scale(Ttr[feats], T.loc[[q], feats])[0]
        dist = np.sqrt(((Xtr - xte) ** 2).mean(1))
        idx = np.argsort(dist)[:K]
        w = 1 / (dist[idx] + 1e-6)
        names = list(Ttr.index[idx])
        out = {}
        for meth, fn in (("analogs_cycle", future_same_cycle), ("analogs_state", future_same_state)):
            futs = np.array([fn(*series[c], n, a_q, t) for c in names])
            out[meth], out[f"{meth}_lo"], out[f"{meth}_hi"] = _combine(futs, w)
        m = (xq > n / 2) & (xq <= n)
        slope = np.polyfit(xq[m], yq[m], 1)[0] if m.sum() >= 2 else np.nan
        out["own_linear"] = a_q + slope * (t - n)
        out["own_powerlaw"] = own_powerlaw(xq, yq, n, a_q, t)
        allf = np.array([future_same_cycle(*series[c], n, a_q, t) for c in Ttr.index])
        out["population"] = _combine(allf, np.ones(len(allf)))[0]
        same = (Ttr["condition"] == T.loc[q, "condition"]).to_numpy()
        out["same_condition"] = np.full(len(t), np.nan)
        if same.any():
            f = allf[same]
            cnt = np.isfinite(f).sum(0)
            with np.errstate(invalid="ignore"):
                out["same_condition"] = np.where(cnt >= 1, np.nansum(np.nan_to_num(f), 0) / np.maximum(cnt, 1), np.nan)
        out["regression"] = _regression(Ttr, Xtr, xte, series, n, a_q, t)
        for j, tj in enumerate(t):
            rows.append({"cell": q, "condition": T.loc[q, "condition"], "n": n, "cycle": tj, "horizon": tj - n,
                         "actual": float(yq[xq > n][j]), "anchor": a_q, "analogs": ",".join(names),
                         **{k: float(v[j]) for k, v in out.items()}})
    return pd.DataFrame(rows)


def _regression(Ttr, Xtr, xte, series, n, a_q, t):
    t_max = max(series[c][0][-1] for c in Ttr.index)
    grid = np.geomspace(n + 1, t_max, 30)
    pred = np.full(len(grid), np.nan)
    for g, tg in enumerate(grid):
        ys, rows = [], []
        for r, c in enumerate(Ttr.index):
            xc, yc = series[c]
            v = _interp(xc, yc, np.array([tg]))[0]
            if np.isfinite(v):
                ys.append(v - anchor(xc, yc, n))
                rows.append(r)
        if len(ys) >= 5:
            m = RidgeCV(alphas=np.logspace(-2, 3, 30)).fit(Xtr[rows], np.array(ys))
            pred[g] = m.predict(xte[None])[0]
    ok = np.isfinite(pred)
    if ok.sum() < 2:
        return np.full(len(t), np.nan)
    return a_q + np.interp(t, grid[ok], pred[ok], left=np.nan, right=np.nan)


def score(F: pd.DataFrame, methods=None) -> pd.DataFrame:
    """Per method: mean over cells of the per-cell mean absolute SOH error (percentage points) on the
    points every method predicts, coverage, and errors by horizon quartile."""
    methods = [m for m in (methods or METHODS) if m in F and F[m].notna().any()]
    common = F[methods].notna().all(1)
    G = F[common].copy()
    edges = np.unique(np.quantile(G["horizon"], [0, 0.25, 0.5, 0.75, 1]))
    G["h_bin"] = pd.cut(G["horizon"], edges, include_lowest=True)
    rows = []
    for m in methods:
        e = (G[m] - G["actual"]).abs() * 100
        r = {"method": m, "MAE_pct_points": e.groupby(G["cell"]).mean().mean(),
             "median_cell_MAE": e.groupby(G["cell"]).mean().median(),
             "coverage_%": F[m].notna().mean() * 100, "n_cells": G["cell"].nunique(), "n_points": len(G)}
        for b, eb in e.groupby(G["h_bin"], observed=True):
            r[f"MAE_horizon_{b.left:.0f}-{b.right:.0f}"] = eb.mean()
        if f"{m}_lo" in F:
            inside = (G["actual"] >= G[f"{m}_lo"]) & (G["actual"] <= G[f"{m}_hi"])
            r["band_covers_%"] = inside.mean() * 100
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


def bootstrap_vs(F: pd.DataFrame, a: str, b: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Per-cell MAE(a) - MAE(b) on their common points, 95% bootstrap CI over cells (negative: a better)."""
    G = F.dropna(subset=[a, b])
    ea = (G[a] - G["actual"]).abs().groupby(G["cell"]).mean().to_numpy() * 100
    eb = (G[b] - G["actual"]).abs().groupby(G["cell"]).mean().to_numpy() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), (n_boot, len(ea)))
    boots = ea[idx].mean(1) - eb[idx].mean(1)
    return {"diff": float(ea.mean() - eb.mean()), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "n_cells": len(ea)}
